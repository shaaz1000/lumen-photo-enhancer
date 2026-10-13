// Browser runtime: TensorFlow.js backend, model weights, face detection.
import type { Tf, SrModel } from './srvgg';
import { buildSrvgg } from './srvgg';
import { blendWeights, parseWeights, type ModelWeights } from './weights';
import { chooseTileSize } from './tiling';
import type { Box, Runtime } from './pipeline';
import * as faceapi from '@vladmandic/face-api/dist/face-api.esm-nobundle.js';

declare global { interface Window { tf: Tf } }

export interface RuntimeInfo { backend: string; gpu: boolean; tile: number; error?: string }

const MODEL_BASE = './models/';
// Artifact hosting serves only web file types, so the raw weight files carry a .wasm extension.
// They are plain bytes read with fetch().arrayBuffer(); nothing is instantiated as WebAssembly.
const WEIGHT_FILES = ['general-x4v3.wasm', 'general-wdn-x4v3.wasm'];
let weights: Promise<[ModelWeights, ModelWeights]> | null = null;
const models = new Map<number, SrModel>();

async function fetchBin(name: string): Promise<ArrayBuffer> {
  const r = await fetch(MODEL_BASE + name);
  if (!r.ok) throw new Error(`Could not load ${name} (HTTP ${r.status}).`);
  return r.arrayBuffer();
}

export async function initRuntime(): Promise<{ rt: Runtime | null; info: RuntimeInfo }> {
  const tf = window.tf;
  if (!tf) return { rt: null, info: { backend: 'none', gpu: false, tile: 0, error: 'TensorFlow.js did not load.' } };
  let backend = 'cpu';
  try { if (await tf.setBackend('webgl')) backend = 'webgl'; } catch { /* fall through */ }
  if (backend !== 'webgl') { await tf.setBackend('cpu'); }
  await tf.ready();
  const mobile = /iPhone|iPad|Android|Mobile/i.test(navigator.userAgent) || Math.min(screen.width, screen.height) < 600;
  let maxTex = 4096;
  try { maxTex = (tf.env().getNumber('WEBGL_MAX_TEXTURE_SIZE') as number) || 4096; } catch { /* cpu */ }
  const tile = backend === 'webgl' ? chooseTileSize(mobile, maxTex) : 96;

  weights ??= Promise.all([fetchBin(WEIGHT_FILES[0]), fetchBin(WEIGHT_FILES[1])]).then(([a, b]) => [parseWeights(a), parseWeights(b)]);
  try { await weights; } catch (err) {
    weights = null;
    return { rt: null, info: { backend, gpu: backend === 'webgl', tile, error: (err as Error).message } };
  }

  const rt: Runtime = {
    tf, tile,
    async getModel(dni) {
      const key = Math.round(dni * 20) / 20;
      const hit = models.get(key);
      if (hit) return hit;
      const [strong, weak] = await weights!;
      if (models.size >= 2) { const [k, m] = models.entries().next().value!; m.dispose(); models.delete(k); }
      const m = buildSrvgg(tf, blendWeights(strong, weak, key));
      models.set(key, m);
      return m;
    },
  };
  return { rt, info: { backend, gpu: backend === 'webgl', tile } };
}

let faceReady: Promise<boolean> | null = null;
export async function detectFaces(canvas: HTMLCanvasElement): Promise<Box[] | null> {
  faceReady ??= faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_BASE).then(() => true, () => false);
  if (!(await faceReady)) return null;
  try {
    // Detect on a copy whose long side is 640 to 1024 px: big photos stay fast and
    // small screenshots are enlarged so their faces reach the detector's minimum size.
    const long = Math.max(canvas.width, canvas.height);
    const k = long > 1024 ? 1024 / long : long < 640 ? Math.min(4, 640 / long) : 1;
    let input = canvas;
    if (k !== 1) {
      input = document.createElement('canvas');
      input.width = Math.round(canvas.width * k); input.height = Math.round(canvas.height * k);
      input.getContext('2d')!.drawImage(canvas, 0, 0, input.width, input.height);
    }
    const dets = await faceapi.detectAllFaces(input, new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.45 }));
    return dets.map((d) => {
      const b = d.box;
      const x = Math.max(0, Math.floor(b.x / k)), y = Math.max(0, Math.floor(b.y / k));
      return { x, y, w: Math.min(canvas.width - x, Math.ceil(b.width / k)), h: Math.min(canvas.height - y, Math.ceil(b.height / k)) };
    }).filter((b) => b.w > 4 && b.h > 4);
  } catch { return null; }
}
