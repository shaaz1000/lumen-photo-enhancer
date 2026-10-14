// Fast, approximate quality analysis. Results are reported as coarse words
// (soft / moderate / crisp) because no-reference metrics are rough estimates.
import type { Img } from './ops';
import { luma, resample } from './ops';
import { allowedScales, MAX_OUTPUT_PIXELS, MOBILE_OUTPUT_PIXELS, type Scale } from './tiling';

export interface FaceBox { x: number; y: number; w: number; h: number }
export type Issue = 'heavy-blur' | 'compression' | 'noise' | 'soft' | 'low-res' | 'none';
export type Content = 'portrait' | 'screenshot' | 'photo';

export type Level3<A extends string, B extends string, C extends string> = A | B | C;
export interface Analysis {
  w: number; h: number; bytes: number; format: string;
  sharpness: Level3<'soft', 'moderate', 'crisp'>;
  compression: Level3<'low', 'moderate', 'heavy'>;
  noise: Level3<'low', 'moderate', 'high'>;
  faces: number | null;
  likelyScreenshot: boolean;
  /** fraction of the pixel grid that holds real detail (1 = sharp, 0.5 = soft as a 2x enlargement) */
  effectiveScale: number;
  blur: 'none' | 'mild' | 'strong' | 'motion';
  content: Content;
  issue: Issue;
  recommendedScale: Scale;
  recommendedMode: 'auto' | 'deep' | 'portrait' | 'screenshot' | 'restore';
  opportunities: string[];
  summary: string;
  raw: { blur: number; blockiness: number; sigma: number; bpp: number; anisotropy: number };
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

/**
 * Effective detail scale by round trip: shrink by f and enlarge back. If almost
 * nothing is lost (PSNR at or above 37 dB) the image holds no detail finer than
 * that scale. Returns the smallest f that passes, between 0.3 and 1.
 */
export function effectiveScale(img: Img): number {
  const c = analysisCrop(img, 384);
  if (c.w < 48 || c.h < 48) return 1;
  const Y = luma(c);
  let best = 1;
  for (const f of [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]) {
    const small = resample(c, Math.max(8, Math.round(c.w * f)), Math.max(8, Math.round(c.h * f)));
    const back = luma(resample(small, c.w, c.h));
    let se = 0, n = 0;
    for (let y = 4; y < c.h - 4; y++) for (let x = 4; x < c.w - 4; x++) { const d = Y[y * c.w + x] - back[y * c.w + x]; se += d * d; n++; }
    const psnr = 10 * Math.log10((255 * 255) / Math.max(1e-6, se / n));
    if (psnr >= 37) best = f; else break;
  }
  return best;
}

/** Ratio of horizontal to vertical gradient energy; far from 1 on soft images hints at motion blur. */
export function gradientAnisotropy(Y: Float32Array, w: number, h: number): number {
  let gx = 0, gy = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    const a = Y[i + 1] - Y[i - 1], b = Y[i + w] - Y[i - w];
    gx += a * a; gy += b * b;
  }
  if (gx === 0 || gy === 0) return 1;
  return gx / gy;
}

export function recommendScale(w: number, h: number, eff = 1): Scale {
  const allowed = allowedScales(w, h);
  const long = Math.max(w, h);
  // 8x is 4x AI plus resampling, so it is never recommended; it stays available by hand
  let want: Scale = long <= 1400 ? 4 : 2;
  // a soft image has less real detail; enlarging it a lot mostly enlarges blur
  if (eff <= 0.7 && long > 400) want = 2;
  // phones: keep the recommended result under 8 MP so it fits in browser memory
  const budget = MAX_OUTPUT_PIXELS <= MOBILE_OUTPUT_PIXELS ? 8_000_000 : Infinity;
  const fits = allowed.filter((s) => s <= want && (s === 2 || w * h * s * s <= budget));
  return fits.length ? fits[fits.length - 1] : 2;
}

export function analyze(img: Img, bytes: number, format: string, faces: FaceBox[] | null): Analysis {
  const crop = analysisCrop(img);
  const Y = luma(crop);
  const blur = blurEffect(Y, crop.w, crop.h);
  const blk = blockiness(Y, crop.w, crop.h);
  const sigma = noiseSigma(Y, crop.w, crop.h);
  const bpp = bytes / (img.w * img.h);
  const anisotropy = gradientAnisotropy(Y, crop.w, crop.h);
  const eff = effectiveScale(img);

  const sharpness = blur > 0.45 || eff <= 0.6 ? 'soft' : blur > 0.3 || eff <= 0.8 ? 'moderate' : 'crisp';
  const rank = { low: 0, moderate: 1, heavy: 2 } as const;
  const byGrid = blk > 2.2 ? 'heavy' : blk > 1.35 ? 'moderate' : 'low';
  // very few bytes per pixel means aggressive lossy compression
  // (smooth or blurred images compress well, so bytes per pixel only flags extreme cases)
  const byBpp = format === 'JPEG' ? (bpp < 0.07 ? 'heavy' : bpp < 0.11 ? 'moderate' : 'low')
    : format === 'WebP' ? (bpp < 0.04 ? 'heavy' : bpp < 0.07 ? 'moderate' : 'low') : 'low';
  const compression = rank[byGrid] >= rank[byBpp] ? byGrid : byBpp;
  const noise = sigma > 6 ? 'high' : sigma > 2.5 ? 'moderate' : 'low';
  const likelyScreenshot = format === 'PNG' && (sharpness !== 'crisp' || compression !== 'low');

  const aniso = Math.max(anisotropy, 1 / anisotropy);
  const blurKind: Analysis['blur'] =
    eff <= 0.3 && blur > 0.5 ? (aniso > 2.5 ? 'motion' : 'strong') :
    (eff <= 0.6 && aniso > 3) || (blur > 0.4 && aniso > 3.5) ? 'motion' :
    eff <= 0.8 || blur > 0.35 ? 'mild' : 'none';

  const area = img.w * img.h;
  const bigFace = (faces ?? []).some((f) => (f.w * f.h) / area > 0.02);
  const content: Content = bigFace ? 'portrait' : likelyScreenshot ? 'screenshot' : 'photo';
  const long = Math.max(img.w, img.h);
  const issue: Issue =
    blurKind === 'strong' || blurKind === 'motion' ? 'heavy-blur' :
    compression === 'heavy' ? 'compression' :
    noise === 'high' ? 'noise' :
    blurKind === 'mild' ? 'soft' :
    compression === 'moderate' ? 'compression' :
    long < 900 ? 'low-res' : 'none';

  const recommendedScale = recommendScale(img.w, img.h, eff);
  const recommendedMode: Analysis['recommendedMode'] =
    content === 'portrait' ? 'portrait' : content === 'screenshot' ? 'screenshot' :
    issue === 'compression' || issue === 'soft' || issue === 'low-res' || issue === 'noise' ? 'deep' : 'auto';

  const opportunities: string[] = [];
  if (compression !== 'low') opportunities.push('Remove JPEG blocking and ringing');
  if (noise !== 'low') opportunities.push('Reduce grain and digital noise');
  if (blurKind === 'mild') opportunities.push('Reconstruct soft edges and fine texture');
  if (blurKind === 'strong' || blurKind === 'motion') opportunities.push('Clean up, but strong blur cannot be reversed');
  if (faces?.length) opportunities.push(`Natural detail on ${faces.length === 1 ? 'one face' : `${faces.length} faces`}`);
  opportunities.push(`Upscale ${recommendedScale}× to ${img.w * recommendedScale} × ${img.h * recommendedScale}`);

  const pct = Math.round(eff * 100);
  const quality =
    issue === 'heavy-blur' ? `appears strongly ${blurKind === 'motion' ? 'motion blurred' : 'out of focus'}` :
    compression === 'heavy' ? 'appears heavily compressed' :
    noise === 'high' ? 'looks noisy' :
    blurKind === 'mild' ? (eff < 1 ? `looks soft, holding about as much detail as an image ${pct}% this size` : 'looks slightly soft') :
    compression === 'moderate' ? 'appears moderately compressed' : 'looks fairly clean';
  const modeName = { auto: 'Auto', deep: 'Deep Restoration', portrait: 'Portrait', screenshot: 'Screenshot to HD', restore: 'Restoration' }[recommendedMode];
  let summary = `This image is ${img.w} × ${img.h} pixels and ${quality}. We recommend ${modeName} at ${recommendedScale}×.`;
  if (issue === 'heavy-blur') summary += ' AI upscaling will reduce noise and blockiness but cannot undo this much blur, so expect limited improvement.';

  return { w: img.w, h: img.h, bytes, format, sharpness, compression, noise, faces: faces ? faces.length : null, likelyScreenshot,
    effectiveScale: eff, blur: blurKind, content, issue, recommendedScale, recommendedMode, opportunities, summary,
    raw: { blur, blockiness: blk, sigma, bpp, anisotropy } };
}
