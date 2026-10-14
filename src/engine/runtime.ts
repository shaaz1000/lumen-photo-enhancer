// Browser runtime: TensorFlow.js backend, model weights, face detection.
import type { Tf, SrModel } from './srvgg';
import { buildRrdb, buildSrvgg } from './srvgg';
import { blendWeights, concatParts, parseWeights, type ModelWeights } from './weights';
import { chooseTileSize, DESKTOP_OUTPUT_PIXELS, MOBILE_OUTPUT_PIXELS, setOutputCap } from './tiling';
import type { Box, Runtime } from './pipeline';
import type { ModelKind } from './presets';
import * as faceapi from '@vladmandic/face-api/dist/face-api.esm-nobundle.js';

declare global { interface Window { tf: Tf } }

export interface RuntimeInfo { backend: string; gpu: boolean; mobile: boolean; tile: number; deepOk: boolean; error?: string }

const MODEL_BASE = './models/';
// Artifact hosting serves only web file types, so the raw weight files carry a .wasm extension.
// They are plain bytes read with fetch().arrayBuffer(); nothing is instantiated as WebAssembly.
const GENERAL_FILES = ['general-x4v3.wasm', 'general-wdn-x4v3.wasm'];
const DEEP_PARTS = ['deep-x4plus.part0.wasm', 'deep-x4plus.part1.wasm', 'deep-x4plus.part2.wasm'];

let general: Promise<[ModelWeights, ModelWeights]> | null = null;
let deepWeights: Promise<ModelWeights> | null = null;
const cache = new Map<string, SrModel>();

async function fetchBin(name: string): Promise<ArrayBuffer> {
  const r = await fetch(MODEL_BASE + name);
  if (!r.ok) throw new Error(`Could not load ${name} (HTTP ${r.status}).`);
  return r.arrayBuffer();
}

export async function initRuntime(): Promise<{ rt: Runtime | null; info: RuntimeInfo }> {
  const tf = window.tf;
  const mobile = /iPhone|iPad|Android|Mobile/i.test(navigator.userAgent) || Math.min(screen.width, screen.height) < 600;
  setOutputCap(mobile ? MOBILE_OUTPUT_PIXELS : DESKTOP_OUTPUT_PIXELS);
  if (!tf) return { rt: null, info: { backend: 'none', gpu: false, mobile, tile: 0, deepOk: false, error: 'TensorFlow.js did not load.' } };
  let backend = 'cpu';
  try { if (await tf.setBackend('webgl')) backend = 'webgl'; } catch { /* fall through */ }
  if (backend !== 'webgl') { await tf.setBackend('cpu'); }
  await tf.ready();
  let maxTex = 4096;
  try { maxTex = (tf.env().getNumber('WEBGL_MAX_TEXTURE_SIZE') as number) || 4096; } catch { /* cpu */ }
  const gpu = backend === 'webgl';
  const tiles: Record<ModelKind, number> = {
    general: gpu ? chooseTileSize(mobile, maxTex) : 96,
    deep: gpu ? chooseTileSize(mobile, maxTex, true) : 48,
  };

  general ??= Promise.all(GENERAL_FILES.map(fetchBin)).then(([a, b]) => [parseWeights(a), parseWeights(b)]);
  try { await general; } catch (err) {
    general = null;
    return { rt: null, info: { backend, gpu, mobile, tile: tiles.general, deepOk: false, error: (err as Error).message } };
  }

  const keep = (key: string, m: SrModel) => {
    // keep at most three models on the GPU: deep, and two general variants
    if (cache.size >= 3) {
      const evict = [...cache.keys()].find((k) => k !== 'deep' && k !== key);
      if (evict) { cache.get(evict)!.dispose(); cache.delete(evict); }
    }
    cache.set(key, m);
    return m;
  };

  const rt: Runtime = {
    backend: gpu ? 'GPU (WebGL)' : 'CPU',
    tile: (kind) => tiles[kind],
    async getModel(kind, dni) {
      if (kind === 'deep') {
        const hit = cache.get('deep');
        if (hit) return hit;
        deepWeights ??= Promise.all(DEEP_PARTS.map(fetchBin)).then((parts) => parseWeights(concatParts(parts)));
        let w: ModelWeights;
        try { w = await deepWeights; } catch (e) { deepWeights = null; throw new Error(`The deep model could not be downloaded: ${(e as Error).message}`); }
        return keep('deep', buildRrdb(tf, w));
      }
      const key = `g${Math.round(dni * 20) / 20}`;
      const hit = cache.get(key);
      if (hit) return hit;
      const [strong, weak] = await general!;
      return keep(key, buildSrvgg(tf, blendWeights(strong, weak, Math.round(dni * 20) / 20)));
    },
  };
  return { rt, info: { backend, gpu, mobile, tile: tiles.general, deepOk: gpu } };
}

let faceReady: Promise<boolean> | null = null;

/** Detects faces and five alignment points (eye centres, nose tip, mouth corners). */
export async function detectFaces(canvas: HTMLCanvasElement): Promise<Box[] | null> {
  faceReady ??= Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_BASE),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_BASE),
  ]).then(() => true, () => false);
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
    const dets = await faceapi.detectAllFaces(input, new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.45 })).withFaceLandmarks();
    return dets.map((d) => {
      const b = d.detection.box;
      const pts = d.landmarks.positions;
      const mean = (a: number, z: number): [number, number] => {
        let x = 0, y = 0; for (let i = a; i <= z; i++) { x += pts[i].x; y += pts[i].y; }
        return [x / (z - a + 1) / k, y / (z - a + 1) / k];
      };
      const p = (i: number): [number, number] => [pts[i].x / k, pts[i].y / k];
      const x = Math.max(0, Math.floor(b.x / k)), y = Math.max(0, Math.floor(b.y / k));
      return {
        x, y, w: Math.min(canvas.width - x, Math.ceil(b.width / k)), h: Math.min(canvas.height - y, Math.ceil(b.height / k)),
        lm: [mean(36, 41), mean(42, 47), p(30), p(48), p(54)],
      };
    }).filter((b) => b.w > 4 && b.h > 4);
  } catch { return null; }
}
