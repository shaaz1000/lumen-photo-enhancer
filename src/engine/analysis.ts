// Fast, approximate quality analysis. Results are reported as coarse words
// (soft / moderate / crisp) because no-reference metrics are rough estimates.
import type { Img } from './ops';
import { luma } from './ops';
import { allowedScales, type Scale } from './tiling';

export type Level3<A extends string, B extends string, C extends string> = A | B | C;
export interface Analysis {
  w: number; h: number; bytes: number; format: string;
  sharpness: Level3<'soft', 'moderate', 'crisp'>;
  compression: Level3<'low', 'moderate', 'heavy'>;
  noise: Level3<'low', 'moderate', 'high'>;
  faces: number | null;
  likelyScreenshot: boolean;
  recommendedScale: Scale;
  opportunities: string[];
  summary: string;
  raw: { blur: number; blockiness: number; sigma: number; bpp: number };
}

/**
 * Crete-Roffet no-reference blur estimate in 0..1 (higher is blurrier). It re-blurs
 * the image and measures how much neighbour variation is lost, so it does not
 * depend on image size or contrast the way raw Laplacian variance does.
 */
export function blurEffect(Y: Float32Array, w: number, h: number): number {
  if (w < 12 || h < 12) return 0.5;
  const est = (horizontal: boolean) => {
    let sumD = 0, sumV = 0;
    const n = horizontal ? w : h, m = horizontal ? h : w;
    const at = (i: number, j: number) => (horizontal ? Y[j * w + i] : Y[i * w + j]);
    const line = new Float32Array(n), blur = new Float32Array(n);
    for (let j = 0; j < m; j++) {
      for (let i = 0; i < n; i++) line[i] = at(i, j);
      let acc = 0;
      for (let i = 0; i < n; i++) {
        acc += line[i];
        if (i >= 9) acc -= line[i - 9];
        blur[i] = acc / Math.min(9, i + 1);
      }
      for (let i = 9; i < n; i++) {
        const d = Math.abs(line[i] - line[i - 1]);
        const db = Math.abs(blur[i] - blur[i - 1]);
        sumD += d; sumV += Math.max(0, d - db);
      }
    }
    return sumD > 0 ? (sumD - sumV) / sumD : 1;
  };
  return Math.max(est(true), est(false));
}

/**
 * Blocking on the 8 px JPEG grid, measured only in flat areas where block edges
 * are visible. ~1 means no blocking. Resized screenshots lose the grid, so this
 * is combined with bits-per-pixel for JPEG and WebP files.
 */
export function blockiness(Y: Float32Array, w: number, h: number): number {
  const phase = new Float64Array(8), count = new Float64Array(8);
  const acc = (a: number, b: number, prev: number, next: number, k: number) => {
    const d = Math.abs(b - a);
    if (Math.abs(a - prev) < 6 && Math.abs(next - b) < 6 && d < 12) { phase[k] += d; count[k]++; }
  };
  for (let y = 0; y < h; y++) for (let x = 1; x < w - 2; x++) {
    const i = y * w + x;
    acc(Y[i], Y[i + 1], Y[i - 1], Y[i + 2], x % 8);
  }
  for (let y = 1; y < h - 2; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    acc(Y[i], Y[i + w], Y[i - w], Y[i + 2 * w], y % 8);
  }
  let other = 0, nOther = 0;
  for (let k = 0; k < 7; k++) if (count[k]) { other += phase[k] / count[k]; nOther++; }
  if (!count[7] || !nOther || other === 0) return 1;
  return phase[7] / count[7] / (other / nOther);
}

/** Immerkaer's noise estimate (8-bit units), restricted to flat areas so edges and text do not count as noise. */
export function noiseSigma(Y: Float32Array, w: number, h: number): number {
  if (w < 3 || h < 3) return 0;
  let sum = 0, n = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    const gx = Y[i - w + 1] + 2 * Y[i + 1] + Y[i + w + 1] - Y[i - w - 1] - 2 * Y[i - 1] - Y[i + w - 1];
    const gy = Y[i + w - 1] + 2 * Y[i + w] + Y[i + w + 1] - Y[i - w - 1] - 2 * Y[i - w] - Y[i - w + 1];
    if (Math.abs(gx) + Math.abs(gy) > 120) continue;
    const v = Y[i - w - 1] - 2 * Y[i - w] + Y[i - w + 1] - 2 * Y[i - 1] + 4 * Y[i] - 2 * Y[i + 1] + Y[i + w - 1] - 2 * Y[i + w] + Y[i + w + 1];
    sum += Math.abs(v); n++;
  }
  if (n < 100) return 0;
  return (Math.sqrt(Math.PI / 2) * sum) / (6 * n);
}

/** Centre crop aligned to the 8 px grid so blockiness stays measurable on big images. */
export function analysisCrop(img: Img, max = 1024): Img {
  if (img.w <= max && img.h <= max) return img;
  const cw = Math.min(max, img.w), ch = Math.min(max, img.h);
  const x0 = Math.floor((img.w - cw) / 16) * 8, y0 = Math.floor((img.h - ch) / 16) * 8;
  const out = new Uint8ClampedArray(cw * ch * 4);
  for (let y = 0; y < ch; y++) out.set(img.data.subarray(((y0 + y) * img.w + x0) * 4, ((y0 + y) * img.w + x0 + cw) * 4), y * cw * 4);
  return { w: cw, h: ch, data: out };
}

export function recommendScale(w: number, h: number): Scale {
  const allowed = allowedScales(w, h);
  const long = Math.max(w, h);
  const want: Scale = long <= 400 ? 8 : long <= 1400 ? 4 : 2;
  const fits = allowed.filter((s) => s <= want);
  return fits.length ? fits[fits.length - 1] : 2;
}

export function analyze(img: Img, bytes: number, format: string, faces: number | null): Analysis {
  const crop = analysisCrop(img);
  const Y = luma(crop);
  const blur = blurEffect(Y, crop.w, crop.h);
  const blk = blockiness(Y, crop.w, crop.h);
  const sigma = noiseSigma(Y, crop.w, crop.h);
  const bpp = bytes / (img.w * img.h);

  const sharpness = blur > 0.45 ? 'soft' : blur > 0.3 ? 'moderate' : 'crisp';
  const rank = { low: 0, moderate: 1, heavy: 2 } as const;
  const byGrid = blk > 1.6 ? 'heavy' : blk > 1.2 ? 'moderate' : 'low';
  // very few bytes per pixel means aggressive lossy compression
  const byBpp = format === 'JPEG' ? (bpp < 0.15 ? 'heavy' : bpp < 0.35 ? 'moderate' : 'low')
    : format === 'WebP' ? (bpp < 0.08 ? 'heavy' : bpp < 0.2 ? 'moderate' : 'low') : 'low';
  const compression = rank[byGrid] >= rank[byBpp] ? byGrid : byBpp;
  const noise = sigma > 6 ? 'high' : sigma > 2.5 ? 'moderate' : 'low';
  // Screenshots are usually PNG captures of something already soft or compressed.
  const likelyScreenshot = format === 'PNG' && (sharpness !== 'crisp' || compression !== 'low');
  const recommendedScale = recommendScale(img.w, img.h);

  const opportunities: string[] = [];
  if (compression !== 'low') opportunities.push('Remove JPEG blocking and ringing');
  if (noise !== 'low') opportunities.push('Reduce grain and digital noise');
  if (sharpness === 'soft') opportunities.push('Recover edge definition');
  if (faces) opportunities.push(`Protect natural detail on ${faces === 1 ? 'one face' : `${faces} faces`}`);
  opportunities.push(`Upscale ${recommendedScale}× to ${img.w * recommendedScale} × ${img.h * recommendedScale}`);

  const quality =
    compression === 'heavy' ? 'appears heavily compressed' :
    compression === 'moderate' ? 'appears moderately compressed' :
    sharpness === 'soft' ? 'looks soft' : noise === 'high' ? 'looks noisy' : 'looks fairly clean';
  const extra = compression !== 'low' || noise !== 'low' ? ' with noise and artifact reduction' : sharpness === 'soft' ? ' with gentle sharpening' : '';
  const summary = `This image is ${img.w} × ${img.h} pixels and ${quality}. We recommend ${recommendedScale}× AI upscaling${extra}.`;

  return { w: img.w, h: img.h, bytes, format, sharpness, compression, noise, faces, likelyScreenshot, recommendedScale, opportunities, summary, raw: { blur, blockiness: blk, sigma, bpp } };
}
