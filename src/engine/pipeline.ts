// Degradation-aware enhancement pipeline.
//
//   1. match detail scale  shrink soft images so detail sits at pixel scale (only when the plan says so)
//   2. reconstruct         Real-ESRGAN x4plus (deep) or general v3, tiled with feathered overlap
//   3. texture             weak-denoise model output mixed in to keep natural grain (and more on faces)
//   4. faces               generative restoration, server only and opt in
//   5. finish              restrained colour correction, halo-limited sharpening
//   6. verify              output must agree with the input at input size, or the run fails
//
// Stages 1 to 4 are cached; changing only finishing controls re-runs stage 5 in about a second.
import type { SrModel } from './srvgg';
import { runSr, countTiles, Cancelled } from './upscale';
import { applyColor, bilateral, blend, faceMask, hasAlpha, makeImg, meanDiffAtSize, planColor, resample, sharpenNoHalo, type Img } from './ops';
import type { ModelKind, Plan } from './presets';
import type { Scale } from './tiling';

export type EngineKind = 'ai' | 'classic' | 'server';
export interface Box { x: number; y: number; w: number; h: number; /** 5 points: eyes, nose, mouth corners */ lm?: [number, number][] }

export interface Runtime {
  getModel(kind: ModelKind, dni: number): Promise<SrModel>;
  tile(kind: ModelKind): number;
  backend: string;
}

export interface ServerRequest { scale: number; model: ModelKind; dni: number; preShrink: number; faceRestore: number; faces: Box[]; weak: boolean }
export interface ServerResult { primary: Img; weak: Img | null; info: string[] }
export type ServerSr = (src: Img, req: ServerRequest, signal?: AbortSignal) => Promise<ServerResult>;

export interface Progress { stage: string; done?: number; total?: number }

export interface Reconstruction { primary: Img; weak: Img | null; stages: string[]; models: string[]; ms: number }
export interface Result { img: Img; engine: EngineKind; stages: string[]; models: string[]; ms: number }

const tick = () => new Promise<void>((r) => setTimeout(r, 0));

/** Cache key for the expensive stages; finishing controls are not part of it. */
export function srKey(scale: Scale, plan: Plan, engine: EngineKind, faces: Box[]): string {
  const r = (v: number) => Math.round(v * 20) / 20;
  return [engine, scale, plan.model, r(plan.dni), plan.preShrink, plan.textureMix > 0.02 ? 'w' : '-', r(plan.faceRestore), faces.length].join('|');
}

export function targetSize(src: Img, scale: number) { return { W: src.w * scale, H: src.h * scale }; }

export async function reconstruct(
  src: Img, faces: Box[], scale: Scale, plan: Plan, engine: EngineKind, rt: Runtime | null,
  onProgress: (p: Progress) => void, signal?: AbortSignal, server?: ServerSr,
): Promise<Reconstruction> {
  const t0 = performance.now();
  const check = () => { if (signal?.aborted) throw new Cancelled(); };
  const { W, H } = targetSize(src, scale);
  const stages: string[] = [], models: string[] = [];
  const needWeak = plan.textureMix > 0.02 || (faces.length > 0 && plan.faceTextureBoost > 0);

  if (engine === 'classic') {
    onProgress({ stage: 'Reducing noise' }); await tick(); check();
    const clean = bilateral(src, plan.dni);
    onProgress({ stage: 'Resampling' }); await tick(); check();
    stages.push('Bilateral denoise', `Bicubic resampling to ${W} × ${H}`);
    models.push('Conventional processing, no AI');
    return { primary: resample(clean, W, H), weak: null, stages, models, ms: performance.now() - t0 };
  }

  // Stage 1: match detail scale
  let input = src;
  if (plan.preShrink < 1) {
    input = resample(src, Math.max(8, Math.round(src.w * plan.preShrink)), Math.max(8, Math.round(src.h * plan.preShrink)));
    stages.push(`Matched detail scale: ${src.w} × ${src.h} → ${input.w} × ${input.h}`);
  }
  // the model always enlarges 4x; box-average to 2x when that already covers the target
  const outFactor = input.w * 2 >= W * 0.98 ? 2 : 4;
  const toTarget = (img: Img) => (img.w === W && img.h === H ? img : resample(img, W, H));

  if (engine === 'server') {
    if (!server) throw new Error('No enhancement server is connected.');
    onProgress({ stage: 'Enhancing on server' });
    const r = await server(input, { scale: outFactor, model: plan.model, dni: plan.dni, preShrink: plan.preShrink, faceRestore: plan.faceRestore,
      faces: faces.map((b) => scaleBox(b, input.w / src.w)), weak: needWeak }, signal);
    stages.push(`AI reconstruction ${outFactor}× on server`, ...r.info);
    models.push(plan.model === 'deep' ? 'Real-ESRGAN x4plus (server)' : 'Real-ESRGAN general v3 (server)');
    if (plan.faceRestore > 0 && r.info.some((s) => s.startsWith('GFPGAN'))) models.push('GFPGAN v1.4 (server)');
    const primary = toTarget(r.primary);
    if (primary !== r.primary) stages.push(`Resampled ${r.primary.w} × ${r.primary.h} → ${W} × ${H}`);
    return { primary, weak: r.weak ? toTarget(r.weak) : null, stages, models, ms: performance.now() - t0 };
  }

  if (!rt) throw new Error('The AI engine is not loaded.');
  const primaryTile = rt.tile(plan.model), weakTile = rt.tile('general');
  const total = countTiles(input.w, input.h, primaryTile) + (needWeak ? countTiles(input.w, input.h, weakTile) : 0);
  let done = 0;
  const onTile = () => onProgress({ stage: 'Reconstructing detail', done: ++done, total });

  // Stage 2: reconstruct
  onProgress({ stage: plan.model === 'deep' ? 'Loading deep model (33 MB, first time only)' : 'Loading model' });
  const model = await rt.getModel(plan.model, plan.dni);
  check();
  onProgress({ stage: 'Reconstructing detail', done: 0, total });
  let primary = await runSr(model, input, { tile: primaryTile, outFactor, signal, onTile, yieldFn: tick });
  models.push(model.name + (plan.model === 'general' ? ` (denoise ${Math.round(plan.dni * 100)}%)` : ''));
  stages.push(`AI reconstruction ${outFactor}× with ${model.name} on ${rt.backend}`);

  // Stage 3: texture source
  let weak: Img | null = null;
  if (needWeak) {
    const wm = await rt.getModel('general', 0);
    weak = await runSr(wm, input, { tile: weakTile, outFactor, signal, onTile, yieldFn: tick });
    models.push(`${wm.name} (weak denoise, texture source)`);
  }
  check();
  if (primary.w !== W || primary.h !== H) {
    stages.push(`Resampled ${primary.w} × ${primary.h} → ${W} × ${H}${scale === 8 ? ' (8× is 4× AI plus 2× resampling)' : ''}`);
    primary = resample(primary, W, H);
    if (weak) weak = resample(weak, W, H);
  }
  return { primary, weak, stages, models, ms: performance.now() - t0 };
}

/** Fast finishing stage: texture mix, colour and halo-limited sharpening, then verification. */
export async function finish(
  src: Img, rec: Reconstruction, faces: Box[], plan: Plan, engine: EngineKind,
  onProgress: (p: Progress) => void, signal?: AbortSignal,
): Promise<{ img: Img; stages: string[] }> {
  const check = () => { if (signal?.aborted) throw new Cancelled(); };
  const W = rec.primary.w, H = rec.primary.h, k = W / src.w;
  const stages: string[] = [];
  const mask = faces.length ? faceMask(W, H, faces.map((b) => scaleBox(b, k))) : undefined;

  let out = rec.primary;
  if (rec.weak && (plan.textureMix > 0.02 || mask)) {
    onProgress({ stage: 'Balancing texture' }); await tick(); check();
    out = blend(out, rec.weak, plan.textureMix, mask, plan.faceTextureBoost);
    stages.push(`Texture: ${Math.round(plan.textureMix * 100)}% weak-denoise detail mixed in${mask && plan.faceTextureBoost > 0 ? `, ${Math.round((plan.textureMix + plan.faceTextureBoost) * 100)}% on faces` : ''}`);
  }

  if (plan.color > 0.01) {
    onProgress({ stage: 'Correcting colour' }); await tick(); check();
    out = applyColor(out, planColor(src), plan.color);
    stages.push(`Colour correction ${Math.round(plan.color * 100)}%`);
  }

  const amount = engine === 'classic' ? plan.sharpen + 0.6 : plan.sharpen;
  if (amount > 0.01) {
    onProgress({ stage: 'Refining edges' }); await tick(); check();
    out = sharpenNoHalo(out, amount, mask, plan.faceSharpenCut, 0.12, 2);
    stages.push(`Halo-limited sharpening ${Math.round(amount * 100)}%${mask && plan.faceSharpenCut > 0 ? ', reduced on faces' : ''}`);
  }

  if (out === rec.primary) out = { w: W, h: H, data: new Uint8ClampedArray(rec.primary.data) };
  if (hasAlpha(src)) {
    const a = resample(src, W, H);
    for (let p = 3; p < out.data.length; p += 4) out.data[p] = a.data[p];
  }

  // Stage 6: the result must still be this photo. A broken GPU path can return
  // black, noise or a shifted image; refuse to show or export that.
  const diff = meanDiffAtSize(out, src);
  if (!(diff < 28)) throw new Error(`The enhanced image does not match the original (mean difference ${diff.toFixed(1)} levels), so it was discarded. This usually means the GPU ran out of memory or lacks precision; try Faithful or a smaller image.`);
  stages.push(`Verified against the original: mean difference ${diff.toFixed(1)} levels at original size`);
  return { img: out, stages };
}

export async function enhance(
  src: Img, faces: Box[], scale: Scale, plan: Plan, engine: EngineKind, rt: Runtime | null,
  onProgress: (p: Progress) => void, signal?: AbortSignal, server?: ServerSr,
): Promise<Result> {
  const t0 = performance.now();
  const rec = await reconstruct(src, faces, scale, plan, engine, rt, onProgress, signal, server);
  const fin = await finish(src, rec, faces, plan, engine, onProgress, signal);
  return { img: fin.img, engine, stages: [...rec.stages, ...fin.stages], models: rec.models, ms: performance.now() - t0 };
}

export function faceRegion(b: Box, w: number, h: number): Box {
  const m = 0.45;
  const x = Math.max(0, Math.floor(b.x - b.w * m)), y = Math.max(0, Math.floor(b.y - b.h * m));
  const x1 = Math.min(w, Math.ceil(b.x + b.w * (1 + m))), y1 = Math.min(h, Math.ceil(b.y + b.h * (1 + m)));
  return { x, y, w: x1 - x, h: y1 - y };
}

export const scaleBox = (b: Box, s: number): Box => ({ x: b.x * s, y: b.y * s, w: b.w * s, h: b.h * s, lm: b.lm?.map(([x, y]) => [x * s, y * s] as [number, number]) });

export function cropImg(img: Img, r: Box): Img {
  const out = makeImg(r.w, r.h);
  for (let y = 0; y < r.h; y++) out.data.set(img.data.subarray(((r.y + y) * img.w + r.x) * 4, ((r.y + y) * img.w + r.x + r.w) * 4), y * r.w * 4);
  return out;
}
