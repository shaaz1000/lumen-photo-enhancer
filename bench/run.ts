// Benchmark harness: runs the real app pipelines on every case and writes PNGs
// for scoring. Models run through onnxruntime-node using graphs that match the
// browser networks (same fp16 weights; parity is tested), so the pixels here
// are the pixels the browser produces, only faster to obtain.
//
//   node bench/.build/run.mjs [--server http://127.0.0.1:8000] [--v1 path/to/v1/src]
import * as ort from 'onnxruntime-node';
import * as tf from '@tensorflow/tfjs';
import { readFileSync, writeFileSync, mkdirSync, statSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { encodePng, decodePng } from './png';
import { analyze } from '../src/engine/analysis';
import { paramsFor, planFor, type Fidelity, type ModeId } from '../src/engine/presets';
import { enhance, type Box, type Runtime, type ServerSr } from '../src/engine/pipeline';
import { resample, type Img } from '../src/engine/ops';
import type { SrModel } from '../src/engine/srvgg';
import type { Scale } from '../src/engine/tiling';
import { setOutputCap, DESKTOP_OUTPUT_PIXELS } from '../src/engine/tiling';
import * as faceapi from '@vladmandic/face-api/dist/face-api.esm-nobundle.js';
// v1.0 pipeline, bundled from a checkout of the 1.0.0 commit
import * as v1p from 'v1/engine/pipeline';
import * as v1a from 'v1/engine/analysis';
import * as v1s from 'v1/engine/presets';

const ROOT = new URL('../', import.meta.url).pathname.replace(/\/\.build\/$/, '/').replace(/\/bench\/$/, '/bench/');
const CASES = join(ROOT, 'out', 'cases');
const MODELS = join(ROOT, '.models');
const arg = (k: string) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : undefined; };
const SERVER = arg('--server');
setOutputCap(DESKTOP_OUTPUT_PIXELS);

/* ---------- ONNX runtime adapter (same SrModel contract as the browser) ---------- */
const sessions = new Map<string, Promise<ort.InferenceSession>>();
function onnxModel(file: string, name: string, pad: number, feather: number): SrModel {
  return {
    name, scale: 4, pad, feather, dispose() {},
    async infer(data, h, w) {
      if (!sessions.has(file)) sessions.set(file, ort.InferenceSession.create(join(MODELS, file), { intraOpNumThreads: 0 }));
      const s = await sessions.get(file)!;
      const chw = new Float32Array(3 * h * w);
      for (let i = 0; i < h * w; i++) for (let c = 0; c < 3; c++) chw[c * h * w + i] = data[i * 3 + c];
      const t0 = Date.now();
      const out = (await s.run({ input: new ort.Tensor('float32', chw, [1, 3, h, w]) }))[s.outputNames[0]];
      if (process.env.DEBUG) console.log(`     infer ${name} ${w}x${h} ${Date.now() - t0} ms`);
      const o = out.data as Float32Array, H = h * 4, W = w * 4, hwc = new Float32Array(H * W * 3);
      for (let i = 0; i < H * W; i++) for (let c = 0; c < 3; c++) hwc[i * 3 + c] = o[c * H * W + i];
      return hwc;
    },
  };
}
const rt: Runtime = {
  backend: 'ONNX Runtime (benchmark)',
  tile: (k) => (k === 'deep' ? 128 : 192),
  async getModel(kind, dni) {
    if (kind === 'deep') return onnxModel('deep.onnx', 'Real-ESRGAN x4plus', 32, 12);
    const i = Math.round(dni * 20);
    return onnxModel(`general_${String(i).padStart(2, '0')}.onnx`, 'Real-ESRGAN general v3', 16, 8);
  },
};

/* ---------- server adapter for GFPGAN runs ---------- */
const serverSr: ServerSr = async (src, req) => {
  const fd = new FormData();
  fd.append('image', new Blob([encodePng(src)], { type: 'image/png' }), 'input.png');
  fd.append('scale', String(req.scale)); fd.append('model', req.model); fd.append('denoise', String(req.dni));
  fd.append('face_restore', String(req.faceRestore)); fd.append('weak', req.weak ? '1' : '0');
  fd.append('landmarks', JSON.stringify(req.faces.filter((f) => f.lm).map((f) => f.lm)));
  const r = await fetch(`${SERVER}/api/enhance`, { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`server ${r.status}: ${await r.text()}`);
  const img = decodePng(new Uint8Array(await r.arrayBuffer()));
  const info = decodeURIComponent(r.headers.get('x-lumen-info') || '').split('|').filter(Boolean);
  if (r.headers.get('x-lumen-stack') !== '2') return { primary: img, weak: null, info };
  const h = img.h / 2, n = img.w * h * 4;
  return { primary: { w: img.w, h, data: img.data.slice(0, n) }, weak: { w: img.w, h, data: img.data.slice(n) }, info };
};

/* ---------- face detection and identity (face-api, MIT) ---------- */
async function loadFaceApi() {
  const dir = join(ROOT, '..', 'node_modules', '@vladmandic', 'face-api', 'model');
  const nets: [any, string][] = [
    [faceapi.nets.tinyFaceDetector, 'tiny_face_detector_model'],
    [faceapi.nets.faceLandmark68Net, 'face_landmark_68_model'],
    [faceapi.nets.faceRecognitionNet, 'face_recognition_model'],
  ];
  for (const [net, f] of nets) {
    const manifest = JSON.parse(readFileSync(join(dir, `${f}-weights_manifest.json`), 'utf8'));
    const buf = readFileSync(join(dir, `${f}.bin`));
    const weightMap = tf.io.decodeWeights(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), manifest[0].weights);
    await net.loadFromWeightMap(weightMap);
  }
}
const toTensor = (img: Img) => tf.tidy(() => tf.tensor3d(Uint8Array.from(img.data), [img.h, img.w, 4]).slice([0, 0, 0], [img.h, img.w, 3]).toInt());

async function detect(img: Img): Promise<Box[]> {
  const long = Math.max(img.w, img.h);
  const k = long < 640 ? Math.min(4, 640 / long) : 1;
  const big = k === 1 ? img : resample(img, Math.round(img.w * k), Math.round(img.h * k));
  const t = toTensor(big);
  const dets = await faceapi.detectAllFaces(t as any, new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.45 })).withFaceLandmarks();
  t.dispose();
  return dets.map((d: any) => {
    const b = d.detection.box, pts = d.landmarks.positions;
    const mean = (a: number, z: number): [number, number] => { let x = 0, y = 0; for (let i = a; i <= z; i++) { x += pts[i].x; y += pts[i].y; } return [x / (z - a + 1) / k, y / (z - a + 1) / k]; };
    return { x: b.x / k, y: b.y / k, w: b.width / k, h: b.height / k, lm: [mean(36, 41), mean(42, 47), [pts[30].x / k, pts[30].y / k], [pts[48].x / k, pts[48].y / k], [pts[54].x / k, pts[54].y / k]] as [number, number][] };
  });
}
async function descriptor(img: Img): Promise<Float32Array | null> {
  const t = toTensor(img);
  const d = await faceapi.detectSingleFace(t as any, new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.3 })).withFaceLandmarks().withFaceDescriptor();
  t.dispose();
  return d ? d.descriptor : null;
}
const dist = (a: Float32Array, b: Float32Array) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));

/* ---------- v1.0 baseline ---------- */
async function runV1(src: Img, faces: Box[], scale: Scale, bytes: number, format: string) {
  // v1.0 called model.run(tensor) and awaited .data(); give it the same ONNX
  // networks so the comparison isolates the pipeline, not the runtime.
  const rtv1 = {
    tf, tile: 192,
    getModel: async (dni: number) => {
      const m = await rt.getModel('general', dni);
      return { scale: 4, dispose() {}, run: (x: any) => {
        const [, h, w] = x.shape;
        const data = x.dataSync() as Float32Array;
        return { data: () => m.infer(data, h, w), dispose() {} };
      } };
    },
  };
  const a = v1a.analyze(src, bytes, format, faces.length);
  const p = v1s.paramsFor('auto', a, scale);
  const r = await v1p.enhance(src, faces.map(({ x, y, w, h }) => ({ x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) })), p, 'ai', rtv1 as any, () => {});
  return r.img as Img;
}

/* ---------- main ---------- */
interface Method { id: string; label: string; mode?: ModeId; fidelity?: Fidelity; faceRestore?: number; server?: boolean; v1?: boolean; bicubic?: boolean }
const METHODS: Method[] = [
  { id: 'bicubic', label: 'Bicubic', bicubic: true },
  { id: 'v1_auto', label: 'v1.0 Auto (before)', v1: true },
  { id: 'faithful', label: 'v1.1 Faithful', fidelity: 'faithful' },
  { id: 'balanced', label: 'v1.1 Balanced (recommended mode)', fidelity: 'balanced' },
  { id: 'deep', label: 'v1.1 Deep Restoration', mode: 'deep', fidelity: 'balanced' },
  { id: 'max', label: 'v1.1 Deep Restoration, Maximum Detail', mode: 'deep', fidelity: 'max' },
  { id: 'deep_gfpgan', label: 'v1.1 Deep Restoration + GFPGAN 50% (server)', mode: 'deep', fidelity: 'balanced', faceRestore: 0.5, server: true },
];

async function main() {
  await tf.setBackend('cpu');
  await loadFaceApi();
  const only = arg('--only')?.split(',');
  const pick = arg('--cases')?.split(',');
  const cases = JSON.parse(readFileSync(join(CASES, 'cases.json'), 'utf8')).filter((c: any) => !pick || pick.includes(c.name));
  const results: any[] = [];
  for (const c of cases) {
    const dir = join(CASES, c.name);
    const src: Img = { w: c.input_size[0], h: c.input_size[1], data: new Uint8ClampedArray(readFileSync(join(dir, 'input.rgba'))) };
    const gt: Img = { w: c.gt_size[0], h: c.gt_size[1], data: new Uint8ClampedArray(readFileSync(join(dir, 'gt.rgba'))) };
    const bytes = statSync(join(dir, c.input)).size, format = c.input.endsWith('.png') ? 'PNG' : 'JPEG';
    const faces = await detect(src);
    const analysis = analyze(src, bytes, format, faces);
    const gtDesc = faces.length ? await descriptor(gt) : null;
    const entry: any = { case: c.name, note: c.note, scale: c.scale, input: [src.w, src.h], faces: faces.length,
      analysis: { summary: analysis.summary, issue: analysis.issue, blur: analysis.blur, effectiveScale: analysis.effectiveScale, recommendedMode: analysis.recommendedMode }, methods: [] };
    console.log(`\n== ${c.name}: ${analysis.summary} (faces ${faces.length})`);
    for (const m of METHODS) {
      if (only && !only.includes(m.id)) continue;
      if (m.server && (!SERVER || !faces.length)) continue;
      if (m.v1 && !('enhance' in v1p)) continue;
      const t0 = performance.now(), rss0 = process.memoryUsage().rss;
      let img: Img, models: string[] = [], stages: string[] = [];
      try {
        if (m.bicubic) img = resample(src, src.w * c.scale, src.h * c.scale);
        else if (m.v1) img = await runV1(src, faces, c.scale, bytes, format);
        else {
          const mode = m.mode ?? analysis.recommendedMode;
          const p = { ...paramsFor(mode, analysis, c.scale, m.fidelity), faceRestore: m.faceRestore ?? 0 };
          const plan = planFor(p, mode, analysis, { deepOk: true, server: !!m.server, faceRestoreOk: !!m.server });
          const r = await enhance(src, faces, c.scale, plan, m.server ? 'server' : 'ai', rt, () => {}, undefined, serverSr);
          img = r.img; models = r.models; stages = r.stages;
        }
      } catch (e) { console.log(`  ${m.label}: FAILED ${(e as Error).message}`); entry.methods.push({ id: m.id, label: m.label, error: (e as Error).message }); continue; }
      const ms = performance.now() - t0, rss = (process.memoryUsage().rss - rss0) / 1048576;
      writeFileSync(join(dir, `${m.id}.png`), encodePng(img));
      let identity: number | null = null;
      if (gtDesc) {
        const d = await descriptor(img.w === gt.w ? img : resample(img, gt.w, gt.h));
        identity = d ? dist(d, gtDesc) : null;
      }
      entry.methods.push({ id: m.id, label: m.label, ms: Math.round(ms), rssDeltaMB: Math.round(rss), identityDistance: identity, models, stages, size: [img.w, img.h] });
      console.log(`  ${m.label}: ${(ms / 1000).toFixed(1)} s${identity != null ? `, identity distance ${identity.toFixed(3)}` : ''}`);
      if (process.env.DEBUG) console.log('    ' + stages.join('\n    '));
    }
    results.push(entry);
  }
  writeFileSync(join(ROOT, 'out', arg('--out') ?? 'run.json'), JSON.stringify(results, null, 1));
}
main().catch((e) => { console.error(e); process.exit(1); });
