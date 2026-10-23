// Records the WebGPU work of one onnxruntime-web run and replays it without
// ORT: no shape inference, uniform uploads, bind group creation or wasm<->JS
// calls per kernel, just the recorded dispatches re-encoded in one command
// buffer.
//
// Valid while the session's inputs/outputs are fixed GPU buffers with fixed
// shapes, and while no other ORT run reuses the buffers ORT pooled for this
// one (any run made outside a recording invalidates every replay: see
// `stale`). Everything the run touches stays referenced by the recording.
//
//   const recorder = createOrtRecorder(device);
//   const replay = await recorder.record(() => session.run(feeds, fetches));
//   replay.run();          // same GPU work as session.run(feeds, fetches)

// a bind group entry's buffer (resource is a GPUBuffer or a GPUBufferBinding)
const bufferOf = (e) => (e.resource instanceof GPUBuffer ? e.resource : e.resource?.buffer);
const COPY = (x) => (ArrayBuffer.isView(x) ? x.slice() : x);

function snapshotBytes(data, dataOffset = 0, size) {
  if (ArrayBuffer.isView(data)) {
    const e = data.BYTES_PER_ELEMENT ?? 1;
    const count = size ?? ((data.length ?? data.byteLength) - dataOffset);
    return new Uint8Array(data.buffer, data.byteOffset + dataOffset * e, count * e).slice();
  }
  return new Uint8Array(data, dataOffset, size ?? data.byteLength - dataOffset).slice();
}

let active = null;      // the recording in progress
let epoch = 0;          // counts ORT runs made outside recordings
let installed = false;

function install() {
  if (installed) return;
  installed = true;
  const wrap = (proto, name, hook) => {
    const f = proto[name];
    if (typeof f !== 'function') return;
    proto[name] = function (...a) {
      const r = f.apply(this, a);
      if (active) hook(this, a, r);
      return r;
    };
  };
  // command encoders / compute passes: record every call generically
  for (const name of Object.getOwnPropertyNames(GPUCommandEncoder.prototype)) {
    if (name === 'constructor' || typeof Object.getOwnPropertyDescriptor(GPUCommandEncoder.prototype, name).value !== 'function') continue;
    wrap(GPUCommandEncoder.prototype, name, (enc, a, r) => {
      const cmds = active.encoders.get(enc);
      if (!cmds) return;
      if (name === 'finish') { active.buffers.set(r, cmds); return; }
      if (name === 'beginComputePass') {
        const pass = [];
        active.passes.set(r, pass);
        cmds.push({ pass, desc: a[0] });
        return;
      }
      if (name === 'beginRenderPass') active.unsupported.add(name);
      cmds.push({ m: name, a: a.map(COPY) });
    });
  }
  for (const name of Object.getOwnPropertyNames(GPUComputePassEncoder.prototype)) {
    if (name === 'constructor' || name === 'end' ||
        typeof Object.getOwnPropertyDescriptor(GPUComputePassEncoder.prototype, name).value !== 'function') continue;
    wrap(GPUComputePassEncoder.prototype, name, (pass, a) => {
      const cmds = active.passes.get(pass);
      if (cmds) cmds.push({ m: name, a: a.map(COPY) });
    });
  }
  wrap(GPUDevice.prototype, 'createBindGroup', (dev, a, bg) => {
    if (dev === active.device) active.bindGroups.set(bg, { ...a[0], entries: a[0].entries.map((e) => ({ ...e, resource: e.resource })) });
  });
  wrap(GPUDevice.prototype, 'createCommandEncoder', (dev, a, enc) => { if (dev === active.device) active.encoders.set(enc, []); });
  wrap(GPUQueue.prototype, 'writeBuffer', (q, a) => {
    if (q !== active.device.queue) return;
    const bytes = snapshotBytes(a[2], a[3], a[4]);
    active.timeline.push({ write: a[0], offset: a[1], bytes });
  });
  wrap(GPUQueue.prototype, 'submit', (q, a) => {
    if (q !== active.device.queue) return;
    for (const cb of a[0]) {
      const cmds = active.buffers.get(cb);
      if (cmds) active.timeline.push({ submit: cmds });
      else active.unsupported.add('foreign command buffer');
    }
  });
  for (const name of ['writeTexture', 'copyExternalImageToTexture']) wrap(GPUQueue.prototype, name, () => active.unsupported.add(name));
  wrap(GPUBuffer.prototype, 'mapAsync', () => active.unsupported.add('mapAsync'));
  // buffers ORT frees during a recording may still be referenced by it: leak them
  const destroy = GPUBuffer.prototype.destroy;
  GPUBuffer.prototype.destroy = function () { if (!active) destroy.call(this); };
}

function replayCommands(encoder, cmds, mergePasses) {
  let pass = null;
  for (const c of cmds) {
    if (c.pass) {
      if (!pass || !mergePasses || c.desc?.timestampWrites) {
        pass?.end();
        pass = encoder.beginComputePass(c.desc?.timestampWrites ? c.desc : undefined);
      }
      for (const p of c.pass) pass[p.m](...p.a);
      continue;
    }
    pass?.end();
    pass = null;
    encoder[c.m](...c.a);
  }
  pass?.end();
}

export function createOrtRecorder(device) {
  install();
  return {
    // marks an ORT run made without recording: it may reuse pooled buffers
    // that existing replays depend on
    plainRun() { epoch++; },
    async record(run) {
      if (active) throw new Error('a recording is already in progress');
      epoch++;
      active = { device, encoders: new Map(), passes: new Map(), buffers: new Map(), timeline: [], unsupported: new Set(),
                 bindGroups: new Map() };
      let rec;
      try {
        await run();
        await device.queue.onSubmittedWorkDone();
      } finally {
        rec = active;
        active = null;
      }
      if (rec.unsupported.size) throw new Error(`ORT replay: unsupported calls during the run: ${[...rec.unsupported].join(', ')}`);
      // Single command buffer: every uniform-only buffer ORT uploads to gets
      // one private copy per distinct content (its state when a submit used
      // it), and the bind groups are re-created on those copies, so the
      // uploads happen once, here, instead of every replay.
      const shadow = new Map();     // ORT buffer -> current bytes
      const dirty = new Set();
      const current = new Map();    // ORT buffer -> private copy for its current content
      const rebound = new Map();    // `${bindGroup}` per content version -> new bind group
      const commands = [];
      let single = true, uploads = 0, submits = 0;
      const privateCopy = (b) => {
        if (dirty.has(b) || !current.has(b)) {
          const copy = device.createBuffer({ size: b.size, usage: b.usage, mappedAtCreation: true });
          new Uint8Array(copy.getMappedRange()).set(shadow.get(b));
          copy.unmap();
          current.set(b, copy);
          dirty.delete(b);
        }
        return current.get(b);
      };
      const rebind = (bg) => {
        const desc = rec.bindGroups.get(bg);
        if (!desc || !desc.entries.some((e) => shadow.has(bufferOf(e)))) return bg;
        const copies = desc.entries.map((e) => (shadow.has(bufferOf(e)) ? privateCopy(bufferOf(e)) : null));
        const known = rebound.get(bg)?.find((r) => r.copies.every((c, i) => c === copies[i]));
        if (known) return known.group;
        const group = device.createBindGroup({ ...desc, entries: desc.entries.map((e, i) =>
          (!copies[i] ? e : { ...e, resource: e.resource instanceof GPUBuffer ? copies[i] : { ...e.resource, buffer: copies[i] } })) });
        rebound.set(bg, [...(rebound.get(bg) ?? []), { copies, group }]);
        return group;
      };
      for (const e of rec.timeline) {
        if (e.write) {
          uploads++;
          if (e.write.usage & GPUBufferUsage.STORAGE) { single = false; break; }
          if (!shadow.has(e.write)) shadow.set(e.write, new Uint8Array(e.write.size));
          shadow.get(e.write).set(e.bytes, e.offset);
          dirty.add(e.write);
          continue;
        }
        submits++;
        for (const c of e.submit) {
          if (!c.pass) {
            if (c.a.some((x) => shadow.has(x))) single = false;   // a copy/clear touching an uploaded buffer
            commands.push(c);
            continue;
          }
          commands.push({ desc: c.desc, pass: c.pass.map((p) => (p.m === 'setBindGroup' ? { m: p.m, a: [p.a[0], rebind(p.a[1]), ...p.a.slice(2)] } : p)) });
        }
      }
      const myEpoch = epoch;
      const dispatches = commands.reduce((n, c) => n + (c.pass ? c.pass.filter((p) => p.m.startsWith('dispatch')).length : 0), 0);
      const stats = { submits, uploads, single, dispatches, uniformCopies: [...rebound.values()].reduce((n, r) => n + r.length, 0) };
      return {
        stats,
        get stale() { return epoch !== myEpoch; },
        run({ mergePasses = false } = {}) {
          if (epoch !== myEpoch) throw new Error('ORT replay is stale (another ORT run reused its buffers)');
          if (single) {
            const encoder = device.createCommandEncoder();
            replayCommands(encoder, commands, mergePasses);
            device.queue.submit([encoder.finish()]);
            return;
          }
          for (const e of rec.timeline) {
            if (e.write) { device.queue.writeBuffer(e.write, e.offset, e.bytes); continue; }
            const encoder = device.createCommandEncoder();
            replayCommands(encoder, e.submit, mergePasses);
            device.queue.submit([encoder.finish()]);
          }
        },
      };
    },
  };
}

// max |a - b| over two float32 GPU buffers of equal size
export async function maxAbsDiff(device, a, b) {
  const read = async (src) => {
    const staging = device.createBuffer({ size: src.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(src, 0, staging, 0, src.size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    return v;
  };
  const [x, y] = [await read(a), await read(b)];
  let m = 0;
  for (let i = 0; i < x.length; i++) {
    const d = Math.abs(x[i] - y[i]);
    if (!(d <= m)) m = d;  // NaN propagates
  }
  return m;
}
