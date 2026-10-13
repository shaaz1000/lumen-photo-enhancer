// End-to-end enhancement: AI super resolution (or the labeled conventional
// path), face-aware finishing, colour, and halo-free sharpening.
import type { Tf, SrModel } from './srvgg';
import { runSr, countTiles, Cancelled } from './upscale';
import { applyColor, bilateral, blend, faceMask, makeImg, planColor, resample, sharpenNoHalo, type Img } from './ops';
import { effective, type Params } from './presets';

export type EngineKind = 'ai' | 'classic' | 'server';
export type ServerSr = (src: Img, p: Params, signal?: AbortSignal) => Promise<Img>;
export interface Box { x: number; y: number; w: number; h: number }

export interface Runtime {
  tf: Tf;
  getModel(dni: number): Promise<SrModel>;
  tile: number;
}

export interface Progress { stage: string; done?: number; total?: number }

export interface Result { img: Img; engine: EngineKind; notes: string[]; ms: number }

const tick = () => new Promise<void>((r) => setTimeout(r, 0));

/** Cache key for the expensive reconstruction stage; finishing controls are not part of it. */
export function srKey(p: Params, engine: EngineKind, faces: Box[]): string {
  const e = effective(p);
  const r = (v: number) => Math.round(v * 20) / 20;
  return [engine, p.scale, r(e.dni), faces.length && p.faceProtect > 0 ? r(e.faceDni) : 'nf', faces.length].join('|');
}

export async function enhance(
  src: Img, faces: Box[], p: Params, engine: EngineKind, rt: Runtime | null,
  onProgress: (p: Progress) => void, signal?: AbortSignal, server?: ServerSr,
): Promise<Result> {
  const t0 = performance.now();
  const sr = await superResolve(src, faces, p, engine, rt, onProgress, signal, server);
  const img = await finish(src, sr.img, faces, p, engine, onProgress, signal);
  return { img, engine, notes: sr.notes, ms: performance.now() - t0 };
}

export async function superResolve(
  src: Img, faces: Box[], p: Params, engine: EngineKind, rt: Runtime | null,
  onProgress: (p: Progress) => void, signal?: AbortSignal, server?: ServerSr,
): Promise<{ img: Img; notes: string[] }> {
  const e = effective(p);
  const notes: string[] = [];
  const W = src.w * p.scale, H = src.h * p.scale;
  const check = () => { if (signal?.aborted) throw new Cancelled(); };

  let out: Img;
  if (engine === 'ai') {
    if (!rt) throw new Error('The AI engine is not loaded.');
    const outFactor: 2 | 4 = p.scale === 2 ? 2 : 4;
    const pad = 24; // wide enough that tile seams are invisible (verified in tests)
    const useFace = faces.length > 0 && p.faceProtect > 0 && Math.abs(e.faceDni - e.dni) > 0.04;
    const regions = useFace ? faces.map((b) => faceRegion(b, src.w, src.h)) : [];
    const total = countTiles(src.w, src.h, rt.tile) + regions.reduce((n, r) => n + countTiles(r.w, r.h, rt.tile), 0);
    let done = 0;
    const onTile = () => onProgress({ stage: 'Reconstructing detail', done: ++done, total });
    onProgress({ stage: 'Loading model' });
    const model = await rt.getModel(e.dni);
    onProgress({ stage: 'Reconstructing detail', done: 0, total });
    out = await runSr(rt.tf, model, src, { tile: rt.tile, pad, outFactor, signal, onTile, yieldFn: tick });

    if (regions.length) {
      const fm = await rt.getModel(e.faceDni);
      for (const r of regions) {
        const crop = cropImg(src, r);
        const up = await runSr(rt.tf, fm, crop, { tile: rt.tile, pad, outFactor, signal, onTile, yieldFn: tick });
        const box = faces[regions.indexOf(r)];
        const local = faceMask(up.w, up.h, [{ x: (box.x - r.x) * outFactor, y: (box.y - r.y) * outFactor, w: box.w * outFactor, h: box.h * outFactor }])!;
        pasteMasked(out, up, r.x * outFactor, r.y * outFactor, local);
      }
      notes.push(`Gentler denoising on ${faces.length === 1 ? 'one face' : `${faces.length} faces`} to keep real skin texture`);
    }
    if (p.scale === 8) {
      onProgress({ stage: 'Resampling 4× result to 8×' }); await tick(); check();
      out = resample(out, W, H);
      notes.push('8× is 4× AI reconstruction followed by 2× bicubic resampling');
    }
  } else if (engine === 'server') {
    if (!server) throw new Error('No enhancement server is connected.');
    onProgress({ stage: 'Enhancing on server' });
    out = await server(src, p, signal);
    if (out.w !== W || out.h !== H) out = resample(out, W, H);
    notes.push('Real-ESRGAN on the connected server');
  } else {
    onProgress({ stage: 'Reducing noise' }); await tick(); check();
    const clean = bilateral(src, e.dni);
    onProgress({ stage: 'Resampling' }); await tick(); check();
    out = resample(clean, W, H);
    notes.push('Conventional processing: bicubic resampling, bilateral denoise and sharpening. No AI reconstruction.');
  }
  return { img: out, notes };
}

/** Fast finishing stage: texture balance, colour and halo-free sharpening. */
export async function finish(
  src: Img, sr: Img, faces: Box[], p: Params, engine: EngineKind,
  onProgress: (p: Progress) => void, signal?: AbortSignal,
): Promise<Img> {
  const e = effective(p);
  const W = sr.w, H = sr.h;
  const check = () => { if (signal?.aborted) throw new Cancelled(); };
  let out = sr;
  const mask = faces.length && p.faceProtect > 0 ? faceMask(W, H, faces.map((b) => scaleBox(b, W / src.w))) : undefined;

  onProgress({ stage: 'Balancing texture' }); await tick(); check();
  const plain = resample(src, W, H);
  if (engine !== 'classic' && (e.texture > 0 || mask)) out = blend(out, plain, e.texture, mask, e.faceTextureBoost);

  onProgress({ stage: 'Correcting colour' }); await tick(); check();
  out = applyColor(out, planColor(src), e.color);

  onProgress({ stage: 'Refining edges' }); await tick(); check();
  const amount = engine === 'classic' ? e.sharpen + 0.6 : e.sharpen;
  out = sharpenNoHalo(out, amount, mask, e.faceSharpenCut);

  if (out === sr) out = { w: sr.w, h: sr.h, data: new Uint8ClampedArray(sr.data) };
  copyAlpha(out, plain);
  return out;
}

export function faceRegion(b: Box, w: number, h: number): Box {
  const m = 0.45;
  const x = Math.max(0, Math.floor(b.x - b.w * m)), y = Math.max(0, Math.floor(b.y - b.h * m));
  const x1 = Math.min(w, Math.ceil(b.x + b.w * (1 + m))), y1 = Math.min(h, Math.ceil(b.y + b.h * (1 + m)));
  return { x, y, w: x1 - x, h: y1 - y };
}

const scaleBox = (b: Box, s: number): Box => ({ x: b.x * s, y: b.y * s, w: b.w * s, h: b.h * s });

export function cropImg(img: Img, r: Box): Img {
  const out = makeImg(r.w, r.h);
  for (let y = 0; y < r.h; y++) out.data.set(img.data.subarray(((r.y + y) * img.w + r.x) * 4, ((r.y + y) * img.w + r.x + r.w) * 4), y * r.w * 4);
  return out;
}

function pasteMasked(dst: Img, src: Img, ox: number, oy: number, m: Uint8Array) {
  for (let y = 0; y < src.h; y++) for (let x = 0; x < src.w; x++) {
    const k = m[y * src.w + x] / 255; if (!k) continue;
    const dx = ox + x, dy = oy + y; if (dx >= dst.w || dy >= dst.h) continue;
    const p = (dy * dst.w + dx) * 4, q = (y * src.w + x) * 4;
    for (let c = 0; c < 3; c++) dst.data[p + c] += k * (src.data[q + c] - dst.data[p + c]);
  }
}

function copyAlpha(dst: Img, src: Img) {
  for (let p = 3; p < dst.data.length; p += 4) dst.data[p] = src.data[p];
}
