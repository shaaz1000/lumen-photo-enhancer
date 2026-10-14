// Runs a super-resolution model tile by tile. Neighbouring tiles overlap and
// are blended with linear ramps, and each tile gets extra context pixels, so
// tiled output matches whole-image inference (verified in tests).
import type { SrModel } from './srvgg';
import { makeImg, type Img } from './ops';

export interface RunOpts {
  tile: number;
  /** output size relative to the input: the model scale, or scale / 2 (box-averaged) */
  outFactor: number;
  signal?: AbortSignal;
  onTile?: () => void;
  yieldFn?: () => Promise<void>;
}

export class Cancelled extends Error { constructor() { super('Cancelled'); this.name = 'Cancelled'; } }

export interface FTile {
  x0: number; y0: number; x1: number; y1: number;      // core
  ex0: number; ey0: number; ex1: number; ey1: number;  // core + feather (kept in output)
  px0: number; py0: number; px1: number; py1: number;  // + context (fed to the model)
}

export function planFeatherTiles(w: number, h: number, tile: number, feather: number, pad: number): FTile[] {
  const t = Math.max(16, Math.floor(tile));
  const out: FTile[] = [];
  for (let y0 = 0; y0 < h; y0 += t) for (let x0 = 0; x0 < w; x0 += t) {
    const x1 = Math.min(w, x0 + t), y1 = Math.min(h, y0 + t);
    const ex0 = x0 > 0 ? Math.max(0, x0 - feather) : 0, ey0 = y0 > 0 ? Math.max(0, y0 - feather) : 0;
    const ex1 = x1 < w ? Math.min(w, x1 + feather) : w, ey1 = y1 < h ? Math.min(h, y1 + feather) : h;
    out.push({ x0, y0, x1, y1, ex0, ey0, ex1, ey1,
      px0: Math.max(0, ex0 - pad), py0: Math.max(0, ey0 - pad), px1: Math.min(w, ex1 + pad), py1: Math.min(h, ey1 + pad) });
  }
  return out;
}

export function countTiles(w: number, h: number, tile: number) {
  return Math.ceil(w / tile) * Math.ceil(h / tile);
}

/** Linear ramp weights: 0..1 over the leading band, 1..0 over the trailing band. */
export function ramp(n: number, lead: number, trail: number): Float32Array {
  const r = new Float32Array(n).fill(1);
  for (let i = 0; i < lead && i < n; i++) r[i] = (i + 1) / (lead + 1);
  for (let i = 0; i < trail && i < n; i++) r[n - 1 - i] = Math.min(r[n - 1 - i], (i + 1) / (trail + 1));
  return r;
}

function cropRGB(img: Img, x0: number, y0: number, x1: number, y1: number): Float32Array {
  const pw = x1 - x0, ph = y1 - y0;
  const out = new Float32Array(pw * ph * 3);
  for (let y = 0; y < ph; y++) {
    let p = ((y0 + y) * img.w + x0) * 4, q = y * pw * 3;
    for (let x = 0; x < pw; x++, p += 4, q += 3) {
      out[q] = img.data[p] / 255; out[q + 1] = img.data[p + 1] / 255; out[q + 2] = img.data[p + 2] / 255;
    }
  }
  return out;
}

export async function runSr(model: SrModel, img: Img, o: RunOpts): Promise<Img> {
  const s = model.scale;
  const f = s / o.outFactor;
  if (!Number.isInteger(f) || f < 1) throw new Error('Unsupported output factor.');
  const W = Math.round(img.w * o.outFactor), H = Math.round(img.h * o.outFactor);
  // Ramps of neighbouring tiles sum to exactly 1, so weighted values can be added
  // straight into the output bytes: no float buffers, which matters on phones.
  const out = makeImg(W, H);
  const o8 = out.data;
  for (const t of planFeatherTiles(img.w, img.h, o.tile, model.feather, model.pad)) {
    if (o.signal?.aborted) throw new Cancelled();
    const pw = t.px1 - t.px0, ph = t.py1 - t.py0;
    const y = await model.infer(cropRGB(img, t.px0, t.py0, t.px1, t.py1), ph, pw);
    const srcW = pw * s;
    // region kept, in output pixels
    const ow = (t.ex1 - t.ex0) * o.outFactor, oh = (t.ey1 - t.ey0) * o.outFactor;
    const leadX = t.x0 > 0 ? (t.x0 - t.ex0 + Math.min(model.feather, t.x1 - t.x0)) * o.outFactor : 0;
    const trailX = t.x1 < img.w ? (t.ex1 - t.x1 + Math.min(model.feather, t.x1 - t.x0)) * o.outFactor : 0;
    const leadY = t.y0 > 0 ? (t.y0 - t.ey0 + Math.min(model.feather, t.y1 - t.y0)) * o.outFactor : 0;
    const trailY = t.y1 < img.h ? (t.ey1 - t.y1 + Math.min(model.feather, t.y1 - t.y0)) * o.outFactor : 0;
    const wx = ramp(ow, leadX, trailX), wy = ramp(oh, leadY, trailY);
    const sx0 = (t.ex0 - t.px0) * s, sy0 = (t.ey0 - t.py0) * s;
    const dx0 = t.ex0 * o.outFactor, dy0 = t.ey0 * o.outFactor;
    const inv = 1 / (f * f);
    for (let i = 0; i < y.length; i += 97) if (y[i] !== y[i]) throw new Error(`${model.name} produced invalid pixels. Your GPU may not support this model; try the Faithful setting.`);
    for (let yy = 0; yy < oh; yy++) {
      const dRow = (dy0 + yy) * W;
      for (let xx = 0; xx < ow; xx++) {
        let r = 0, g = 0, b = 0;
        for (let j = 0; j < f; j++) for (let i = 0; i < f; i++) {
          const q = ((sy0 + yy * f + j) * srcW + sx0 + xx * f + i) * 3;
          r += y[q]; g += y[q + 1]; b += y[q + 2];
        }
        const k = inv * wx[xx] * wy[yy] * 255, di = (dRow + dx0 + xx) * 4;
        o8[di] += r * k; o8[di + 1] += g * k; o8[di + 2] += b * k; o8[di + 3] = 255;
      }
    }
    o.onTile?.();
    if (o.yieldFn) await o.yieldFn();
  }
  return out;
}
