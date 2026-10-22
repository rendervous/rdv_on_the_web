// Draws an image straight out of an rdv evaluator's heap region onto a WebGPU
// canvas - no readback, no CPU pass over the pixels. The canvas' own pixel
// size is set to the image size (CSS scales it), so one fragment = one pixel.
// Rows are flipped (image row 0 at the bottom), rendervous' presentation
// convention.
//
//   const display = createDisplay(evaluator.device, canvas);
//   display.draw(evaluator.region('image'), { width, height, channels, scale, mode, gamma, planar });
//
// mode 0: clamp(v * scale), mode 1: 1 - exp(-v * scale) (filmic); gamma applies
// a 1/2.2 encode after either. A 1-channel image is shown as gray. planar: the
// image is channels-first ([c][y][x], e.g. a network's NCHW output) instead of
// interleaved ([y][x][c]).

const SHADER = /* wgsl */ `
struct Params {
  offset : u32,   // in floats
  width : u32,
  height : u32,
  channels : u32,
  scale : f32,
  mode : u32,
  gamma : u32,
  planar : u32,
};
@group(0) @binding(0) var<storage, read> heap : array<f32>;
@group(0) @binding(1) var<uniform> p : Params;

@vertex fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) pos : vec4f) -> @location(0) vec4f {
  let x = min(u32(pos.x), p.width - 1u);
  let y = p.height - 1u - min(u32(pos.y), p.height - 1u); // row 0 at the bottom
  let pixel = y * p.width + x;
  var at = p.offset + pixel * p.channels;
  var stride = 1u;
  if (p.planar == 1u) {
    at = p.offset + pixel;
    stride = p.width * p.height;
  }
  var c = vec3f(heap[at]);
  if (p.channels >= 3u) {
    c = vec3f(heap[at], heap[at + stride], heap[at + 2u * stride]);
  }
  c = c * p.scale;
  if (p.mode == 1u) {
    c = 1.0 - exp(-max(c, vec3f(0.0)));
  }
  c = clamp(c, vec3f(0.0), vec3f(1.0));
  if (p.gamma == 1u) {
    c = pow(c, vec3f(1.0 / 2.2));
  }
  return vec4f(c, 1.0);
}
`;

export function createDisplay(device, canvas) {
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  const module = device.createShaderModule({ code: SHADER });
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let bindGroup = null, boundHeap = null;

  function draw(region, { width, height, channels = 3, scale = 1, mode = 0, gamma = false, planar = false }) {
    if (!region) return;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    if (region.buffer !== boundHeap) {
      boundHeap = region.buffer;
      bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: region.buffer } }, { binding: 1, resource: { buffer: params } }],
      });
    }
    const data = new ArrayBuffer(32);
    new Uint32Array(data, 0, 4).set([region.offset / 4, width, height, channels]);
    new Float32Array(data, 16, 1)[0] = scale;
    new Uint32Array(data, 20, 3).set([mode, gamma ? 1 : 0, planar ? 1 : 0]);
    device.queue.writeBuffer(params, 0, data);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store',
                           clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  return { draw };
}
