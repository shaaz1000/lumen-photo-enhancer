// Runs the super-resolution network tile by tile and writes into an RGBA buffer.
import type { Tf, SrModel } from './srvgg';
import { makeImg, type Img } from './ops';
import { planTiles, type Tile } from './tiling';

export interface RunOpts {
  tile: number;
  pad: number;
  /** 4 writes the network output, 2 box-averages each tile to half size */
  outFactor: 2 | 4;
  signal?: AbortSignal;
  onTile?: () => void;
  yieldFn?: () => Promise<void>;
}

export class Cancelled extends Error { constructor() { super('Cancelled'); this.name = 'Cancelled'; } }

export function countTiles(w: number, h: number, tile: number) {
  return Math.ceil(w / tile) * Math.ceil(h / tile);
}

function cropRGB(img: Img, t: Tile): Float32Array {
  const out = new Float32Array(t.pw * t.ph * 3);
  for (let y = 0; y < t.ph; y++) {
    let p = ((t.py + y) * img.w + t.px) * 4, q = y * t.pw * 3;
    for (let x = 0; x < t.pw; x++, p += 4, q += 3) {
      out[q] = img.data[p] / 255; out[q + 1] = img.data[p + 1] / 255; out[q + 2] = img.data[p + 2] / 255;
    }
  }
  return out;
}

/** Copies the core of a tile's network output (scale s) into dst at outFactor. */
export function writeTile(dst: Img, src: Float32Array, t: Tile, s: number, outFactor: number) {
  const srcW = t.pw * s;
  const ox = (t.x - t.px) * s, oy = (t.y - t.py) * s;
  const cw = t.w * s, ch = t.h * s;
  const f = s / outFactor; // 1 or 2
  const dx0 = t.x * outFactor, dy0 = t.y * outFactor;
  const outW = Math.floor(cw / f), outH = Math.floor(ch / f);
  const norm = 255 / (f * f);
  for (let y = 0; y < outH; y++) {
    const dRow = (dy0 + y) * dst.w;
    for (let x = 0; x < outW; x++) {
      let r = 0, g = 0, b = 0;
      for (let j = 0; j < f; j++) for (let i = 0; i < f; i++) {
        const q = ((oy + y * f + j) * srcW + ox + x * f + i) * 3;
        r += src[q]; g += src[q + 1]; b += src[q + 2];
      }
      const p = (dRow + dx0 + x) * 4;
      dst.data[p] = r * norm; dst.data[p + 1] = g * norm; dst.data[p + 2] = b * norm; dst.data[p + 3] = 255;
    }
  }
}

export async function runSr(tf: Tf, model: SrModel, img: Img, o: RunOpts): Promise<Img> {
  const s = model.scale;
  const out = makeImg(img.w * o.outFactor, img.h * o.outFactor);
  for (const t of planTiles(img.w, img.h, o.tile, o.pad)) {
    if (o.signal?.aborted) throw new Cancelled();
    const x = tf.tensor4d(cropRGB(img, t), [1, t.ph, t.pw, 3]);
    const y = model.run(x);
    const data = (await y.data()) as Float32Array;
    x.dispose(); y.dispose();
    writeTile(out, data, t, s, o.outFactor);
    o.onTile?.();
    if (o.yieldFn) await o.yieldFn();
  }
  return out;
}
