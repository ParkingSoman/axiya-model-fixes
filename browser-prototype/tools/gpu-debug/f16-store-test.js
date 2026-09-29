// Script of f16-store-test.html (kept apart: the app's CSP allows no inline script).
const N = 4096;
const out = document.getElementById('out');
const post = new URLSearchParams(location.search).get('post') === '1';
const result = {userAgent: navigator.userAgent};
try {
  const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
  result.adapter = {vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description};
  const device = await adapter.requestDevice({requiredFeatures: ['shader-f16']});
  const bytes = N * 2;
  const buffer = size => device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST});
  const read = async (src, size) => {
    const staging = device.createBuffer({size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
    const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(src, 0, staging, 0, size); device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ); const copy = staging.getMappedRange().slice(0); staging.unmap(); return copy;
  };
  const run = async (code, bindings, groups) => {
    const module = device.createShaderModule({code: 'enable f16;\n' + code});
    const pipeline = device.createComputePipeline({layout: 'auto', compute: {module, entryPoint: 'main'}});
    const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: bindings.map((b, i) => ({binding: i, resource: {buffer: b}}))});
    const enc = device.createCommandEncoder(); const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(groups); pass.end();
    device.queue.submit([enc.finish()]);
  };
  // Expected value of element i: f16(i % 2048) — exact in f16.
  const expect = i => i % 2048;
  const f16 = h => { const s = h >> 15 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e ? s * 2 ** (e - 15) * (1 + m / 1024) : s * 2 ** -14 * (m / 1024); };
  const check = (values, name) => {
    let odd = 0, even = 0; const first = [];
    values.forEach((v, i) => { if (v !== expect(i)) { if (i % 2) odd++; else even++; if (first.length < 6) first.push([i, v]); } });
    result[name] = {wrongOdd: odd, wrongEven: even, first};
  };
  const halves = async b => Array.from(new Uint16Array(await read(b, bytes)), f16);

  const k1 = buffer(bytes);
  await run(`@group(0) @binding(0) var<storage, read_write> o : array<f16>;
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g : vec3<u32>) { o[g.x] = f16(g.x % 2048u); }`, [k1], N / 64);
  check(await halves(k1), 'K1_scalarStore');

  const k2 = buffer(bytes);
  await run(`@group(0) @binding(0) var<storage, read_write> o : array<vec2<f16>>;
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g : vec3<u32>) { let i = 2u * g.x; o[g.x] = vec2<f16>(f16(i % 2048u), f16((i + 1u) % 2048u)); }`, [k2], N / 128);
  check(await halves(k2), 'K2_vec2Store');

  const k3 = buffer(bytes);
  await run(`@group(0) @binding(0) var<storage, read_write> o : array<f16>;
    @compute @workgroup_size(1) fn main() { for (var i = 0u; i < ${N}u; i++) { o[i] = f16(i % 2048u); } }`, [k3], 1);
  check(await halves(k3), 'K3_singleWriter');

  const input = buffer(bytes), k4 = buffer(N * 4);
  const init = new Uint16Array(N);
  const toHalf = v => { if (v === 0) return 0; const e = Math.floor(Math.log2(v)); return ((e + 15) << 10) | Math.round((v / 2 ** e - 1) * 1024); };
  for (let i = 0; i < N; i++) init[i] = toHalf(expect(i));
  device.queue.writeBuffer(input, 0, init);
  await run(`@group(0) @binding(0) var<storage, read> x : array<f16>;
    @group(0) @binding(1) var<storage, read_write> o : array<f32>;
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g : vec3<u32>) { o[g.x] = f32(x[g.x]); }`, [input, k4], N / 64);
  check(Array.from(new Float32Array(await read(k4, N * 4))), 'K4_scalarLoad');

  // Load variants: which way of reading one 16-bit value works?
  const loadTest = async (name, decl, expr) => {
    const o = buffer(N * 4);
    await run(`@group(0) @binding(0) ${decl};
      @group(0) @binding(1) var<storage, read_write> o : array<f32>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g : vec3<u32>) { let i = g.x; o[i] = ${expr}; }`, [input, o], N / 64);
    check(Array.from(new Float32Array(await read(o, N * 4))), name);
  };
  await loadTest('K5_readWriteScalarLoad', 'var<storage, read_write> x : array<f16>', 'f32(x[i])');
  await loadTest('K6_vec2Load', 'var<storage, read> x : array<vec2<f16>>', 'f32(x[i / 2u][i % 2u])');
  await loadTest('K7_u32Unpack', 'var<storage, read> x : array<u32>', 'unpack2x16float(x[i / 2u])[i % 2u]');
  await loadTest('K8_u32Bitcast', 'var<storage, read> x : array<u32>', 'f32(bitcast<vec2<f16>>(x[i / 2u])[i % 2u])');
  await loadTest('K9_vec4Load', 'var<storage, read> x : array<vec4<f16>>', 'f32(x[i / 4u][i % 4u])');
  await loadTest('K10_vec2Select', 'var<storage, read> x : array<vec2<f16>>', 'f32(select(x[i / 2u].x, x[i / 2u].y, (i % 2u) == 1u))');
  await loadTest('K11_vec2SumTrick', 'var<storage, read> x : array<vec2<f16>>', 'f32(dot(x[i / 2u], select(vec2<f16>(1.0h, 0.0h), vec2<f16>(0.0h, 1.0h), (i % 2u) == 1u)))');
  await loadTest('K12_vec2ToF32', 'var<storage, read> x : array<vec2<f16>>', 'vec2<f32>(x[i / 2u])[i % 2u]');
  await loadTest('K13_scalarPlusZero', 'var<storage, read> x : array<f16>', 'f32(x[i] + 0.0h)');
  // The code the fix would emit: static picks after a bitcast.
  await loadTest('K14_u32BitcastStaticY', 'var<storage, read> x : array<u32>', 'f32(select(bitcast<vec2<f16>>(x[i / 2u]).x, bitcast<vec2<f16>>(x[i / 2u]).y, (i % 2u) == 1u))');
  await loadTest('K15_vec2u32BitcastVec4', 'var<storage, read> x : array<vec2<u32>>', 'f32(bitcast<vec4<f16>>(x[i / 4u])[i % 4u])');
  await loadTest('K16_vec2u32BitcastStaticW', 'var<storage, read> x : array<vec2<u32>>', 'f32(select(select(bitcast<vec4<f16>>(x[i / 4u]).x, bitcast<vec4<f16>>(x[i / 4u]).y, (i % 4u) == 1u), select(bitcast<vec4<f16>>(x[i / 4u]).z, bitcast<vec4<f16>>(x[i / 4u]).w, (i % 4u) == 3u), (i % 4u) >= 2u))');
  await loadTest('K17_roundTripBitcast', 'var<storage, read> x : array<vec2<f16>>', 'f32(bitcast<vec2<f16>>(bitcast<u32>(x[i / 2u]))[i % 2u])');
  // Workgroup memory: stage 64 values, then pick one element.
  const wgTest = async (name, decl, stage) => {
    const o = buffer(N * 4);
    await run(`@group(0) @binding(0) ${decl};
      @group(0) @binding(1) var<storage, read_write> o : array<f32>;
      var<workgroup> w : array<vec2<f16>, 32>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(local_invocation_index) l : u32) {
        let i = g.x; if (l < 32u) { w[l] = ${stage}; } workgroupBarrier(); o[i] = f32(w[l / 2u][l % 2u]); }`, [input, o], N / 64);
    check(Array.from(new Float32Array(await read(o, N * 4))), name);
  };
  await wgTest('K18_workgroupFromVec2', 'var<storage, read> x : array<vec2<f16>>', 'x[g.x / 2u - l / 2u + l]');
  await wgTest('K19_workgroupFromU32', 'var<storage, read> x : array<u32>', 'bitcast<vec2<f16>>(x[g.x / 2u - l / 2u + l])');
  // Work-done latency on an idle queue: submit nothing, time onSubmittedWorkDone.
  const latency = [];
  for (let i = 0; i < 20; i++) {
    const t = performance.now();
    device.queue.submit([device.createCommandEncoder().finish()]);
    await device.queue.onSubmittedWorkDone();
    latency.push(+(performance.now() - t).toFixed(2));
  }
  result.workDoneLatencyMs = latency;
} catch (error) { result.error = String(error?.stack ?? error); }
out.textContent = JSON.stringify(result, null, 1);
if (post) fetch('/__gpu-trace', {method: 'POST', headers: {'content-type': 'text/plain'}, body: 'F16TEST ' + JSON.stringify(result)}).catch(() => {});
