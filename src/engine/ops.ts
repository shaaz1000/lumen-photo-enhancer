// Conventional image operations on RGBA buffers. These are the classical
// building blocks used for post-processing and for the non-AI fallback.

export interface Img { w: number; h: number; data: Uint8ClampedArray }

export const makeImg = (w: number, h: number): Img => ({ w, h, data: new Uint8ClampedArray(w * h * 4) });
const clamp255 = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : v);

/* ---------- resampling ---------- */

function cubic(x: number) {
  // Catmull-Rom (a = -0.5): sharp but low ringing
  x = Math.abs(x);
  if (x < 1) return 1.5 * x * x * x - 2.5 * x * x + 1;
  if (x < 2) return -0.5 * x * x * x + 2.5 * x * x - 4 * x + 2;
  return 0;
}

interface Weights { start: Int32Array; count: Int32Array; w: Float32Array; taps: number }

function weights(src: number, dst: number): Weights {
  const scale = dst / src;
  const support = scale < 1 ? 2 / scale : 2; // widen kernel when shrinking (antialias)
  const taps = Math.ceil(support) * 2 + 1;
  const start = new Int32Array(dst), count = new Int32Array(dst), w = new Float32Array(dst * taps);
  for (let i = 0; i < dst; i++) {
    const center = (i + 0.5) / scale - 0.5;
    const lo = Math.max(0, Math.floor(center - support) + 1);
    const hi = Math.min(src - 1, Math.floor(center + support));
    let sum = 0, n = 0;
    for (let j = lo; j <= hi && n < taps; j++, n++) {
      const v = cubic(scale < 1 ? (j - center) * scale : j - center);
      w[i * taps + n] = v; sum += v;
    }
    if (sum !== 0) for (let k = 0; k < n; k++) w[i * taps + k] /= sum;
    start[i] = lo; count[i] = n;
  }
  return { start, count, w, taps };
}

/** Separable bicubic resample with antialiasing on downscale. Conventional, not AI. */
export function resample(img: Img, W: number, H: number): Img {
  const { w, h, data } = img;
  const wx = weights(w, W), wy = weights(h, H);
  const tmp = new Float32Array(W * h * 4);
  for (let y = 0; y < h; y++) {
    const row = y * w * 4;
    for (let x = 0; x < W; x++) {
      let r = 0, g = 0, b = 0, a = 0;
      const s = wx.start[x], n = wx.count[x], o = x * wx.taps;
      for (let k = 0; k < n; k++) {
        const c = wx.w[o + k], p = row + (s + k) * 4;
        r += data[p] * c; g += data[p + 1] * c; b += data[p + 2] * c; a += data[p + 3] * c;
      }
      const q = (y * W + x) * 4;
      tmp[q] = r; tmp[q + 1] = g; tmp[q + 2] = b; tmp[q + 3] = a;
    }
  }
  const out = makeImg(W, H);
  for (let y = 0; y < H; y++) {
    const s = wy.start[y], n = wy.count[y], o = y * wy.taps;
    for (let x = 0; x < W; x++) {
      let r = 0, g = 0, b = 0, a = 0;
      for (let k = 0; k < n; k++) {
        const c = wy.w[o + k], p = ((s + k) * W + x) * 4;
        r += tmp[p] * c; g += tmp[p + 1] * c; b += tmp[p + 2] * c; a += tmp[p + 3] * c;
      }
      const q = (y * W + x) * 4;
      out.data[q] = r; out.data[q + 1] = g; out.data[q + 2] = b; out.data[q + 3] = a;
    }
  }
  return out;
}

/* ---------- luminance helpers ---------- */

export function luma(img: Img): Float32Array {
  const n = img.w * img.h, out = new Float32Array(n), d = img.data;
  for (let i = 0, p = 0; i < n; i++, p += 4) out[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
  return out;
}

function blur3(src: Float32Array, w: number, h: number): Float32Array {
  // 1-2-1 separable gaussian
  const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    const r = y * w;
    for (let x = 0; x < w; x++) {
      const l = src[r + (x > 0 ? x - 1 : x)], c = src[r + x], rr = src[r + (x < w - 1 ? x + 1 : x)];
      tmp[r + x] = (l + 2 * c + rr) * 0.25;
    }
  }
  for (let y = 0; y < h; y++) {
    const up = (y > 0 ? y - 1 : y) * w, r = y * w, dn = (y < h - 1 ? y + 1 : y) * w;
    for (let x = 0; x < w; x++) out[r + x] = (tmp[up + x] + 2 * tmp[r + x] + tmp[dn + x]) * 0.25;
  }
  return out;
}

/**
 * Unsharp mask on luminance with an overshoot limiter. The sharpened value may
 * pass the local 3x3 min/max by only `overshoot` of the local range, which keeps
 * edges crisp without the bright and dark halos of classic sharpening.
 * `passes` widens the blur (1: fine detail, 2: medium detail at upscaled sizes).
 * mask (0..255) lowers the amount, e.g. on skin.
 */
export function sharpenNoHalo(img: Img, amount: number, mask?: Uint8Array, maskScale = 1, overshoot = 0.15, passes = 2): Img {
  if (amount <= 0) return img;
  const { w, h, data } = img;
  const Y = luma(img);
  let B = blur3(Y, w, h);
  for (let i = 1; i < passes; i++) B = blur3(B, w, h);
  const out = makeImg(w, h);
  for (let y = 0; y < h; y++) {
    const y0 = y > 0 ? y - 1 : y, y1 = y < h - 1 ? y + 1 : y;
    for (let x = 0; x < w; x++) {
      const i = y * w + x, x0 = x > 0 ? x - 1 : x, x1 = x < w - 1 ? x + 1 : x;
      let lo = 255, hi = 0;
      for (let r = 0; r < 3; r++) {
        const base = (r === 0 ? y0 : r === 1 ? y : y1) * w;
        const a = Y[base + x0], b = Y[base + x], c = Y[base + x1];
        if (a < lo) lo = a; if (a > hi) hi = a;
        if (b < lo) lo = b; if (b > hi) hi = b;
        if (c < lo) lo = c; if (c > hi) hi = c;
      }
      const slack = (hi - lo) * overshoot;
      let k = amount;
      if (mask) k *= 1 - (maskScale * mask[i]) / 255;
      let s = Y[i] + k * (Y[i] - B[i]);
      s = s < lo - slack ? lo - slack : s > hi + slack ? hi + slack : s;
      const d = s - Y[i], p = i * 4;
      out.data[p] = clamp255(data[p] + d); out.data[p + 1] = clamp255(data[p + 1] + d);
      out.data[p + 2] = clamp255(data[p + 2] + d); out.data[p + 3] = data[p + 3];
    }
  }
  return out;
}

/** Mean absolute difference per channel after box-shrinking `big` to `small`'s size. */
export function meanDiffAtSize(big: Img, small: Img): number {
  const fx = big.w / small.w, fy = big.h / small.h;
  let sum = 0, n = 0;
  const step = Math.max(1, Math.floor(Math.sqrt((small.w * small.h) / 40000)));
  for (let y = 0; y < small.h; y += step) for (let x = 0; x < small.w; x += step) {
    const bx0 = Math.floor(x * fx), by0 = Math.floor(y * fy), bx1 = Math.max(bx0 + 1, Math.floor((x + 1) * fx)), by1 = Math.max(by0 + 1, Math.floor((y + 1) * fy));
    for (let c = 0; c < 3; c++) {
      let acc = 0, cnt = 0;
      for (let yy = by0; yy < by1; yy++) for (let xx = bx0; xx < bx1; xx++) { acc += big.data[(yy * big.w + xx) * 4 + c]; cnt++; }
      sum += Math.abs(acc / cnt - small.data[(y * small.w + x) * 4 + c]); n++;
    }
  }
  return n ? sum / n : 0;
}

export function hasAlpha(img: Img): boolean {
  for (let p = 3; p < img.data.length; p += 4) if (img.data[p] < 255) return true;
  return false;
}

/* ---------- colour ---------- */

export interface ColorPlan { gains: [number, number, number]; black: number; white: number; gamma: number }

/** Gentle auto levels + white balance, measured on midtones so skin and sky do not drive it. */
export function planColor(img: Img): ColorPlan {
  const { data } = img, n = img.w * img.h;
  const step = Math.max(1, Math.floor(n / 250_000));
  const hist = new Uint32Array(256);
  let sr = 0, sg = 0, sb = 0, cnt = 0, total = 0;
  for (let i = 0; i < n; i += step) {
    const p = i * 4, r = data[p], g = data[p + 1], b = data[p + 2];
    const L = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    hist[L]++; total++;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    if (L > 40 && L < 220 && mx - mn < 60) { sr += r; sg += g; sb += b; cnt++; }
  }
  const pct = (q: number) => { let acc = 0; for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= q * total) return v; } return 255; };
  const black = Math.min(pct(0.005), 40), white = Math.max(pct(0.995), 200);
  let gains: [number, number, number] = [1, 1, 1];
  if (cnt > 200) {
    const m = (sr + sg + sb) / 3 / cnt;
    const lim = (v: number) => Math.min(1.08, Math.max(0.92, v));
    gains = [lim(m / (sr / cnt)), lim(m / (sg / cnt)), lim(m / (sb / cnt))];
  }
  // midtone gamma: nudge median toward 118 without crushing
  const med = pct(0.5);
  const norm = Math.min(0.95, Math.max(0.05, (med - black) / Math.max(1, white - black)));
  const gamma = Math.min(1.25, Math.max(0.8, Math.log(0.46) / Math.log(norm)));
  return { gains, black, white, gamma };
}

export function applyColor(img: Img, plan: ColorPlan, strength: number, inPlace = false): Img {
  if (strength <= 0) return img;
  const luts = [0, 1, 2].map((c) => {
    const lut = new Uint8ClampedArray(256);
    for (let v = 0; v < 256; v++) {
      let x = v * plan.gains[c];
      x = (x - plan.black) / Math.max(1, plan.white - plan.black);
      x = x <= 0 ? 0 : Math.pow(Math.min(1, x), plan.gamma);
      lut[v] = Math.round(v + strength * (x * 255 - v));
    }
    return lut;
  });
  const out = inPlace ? img : makeImg(img.w, img.h), d = img.data, o = out.data;
  for (let p = 0; p < d.length; p += 4) {
    o[p] = luts[0][d[p]]; o[p + 1] = luts[1][d[p + 1]]; o[p + 2] = luts[2][d[p + 2]]; o[p + 3] = d[p + 3];
  }
  return out;
}

/* ---------- blending and denoise ---------- */

/** out = a*(1-t) + b*t, with t raised inside the mask by maskBoost. */
export function blend(a: Img, b: Img, t: number, mask?: Uint8Array, maskBoost = 0): Img {
  const out = makeImg(a.w, a.h), A = a.data, Bd = b.data, O = out.data;
  for (let i = 0, p = 0; p < A.length; i++, p += 4) {
    const k = Math.min(1, t + (mask ? (maskBoost * mask[i]) / 255 : 0));
    O[p] = A[p] + k * (Bd[p] - A[p]); O[p + 1] = A[p + 1] + k * (Bd[p + 1] - A[p + 1]);
    O[p + 2] = A[p + 2] + k * (Bd[p + 2] - A[p + 2]); O[p + 3] = A[p + 3];
  }
  return out;
}

/** 5x5 bilateral filter. Edge-preserving noise and block reduction for the conventional path. */
export function bilateral(img: Img, strength: number): Img {
  if (strength <= 0) return img;
  const { w, h, data } = img, out = makeImg(w, h);
  const sr = 8 + 30 * strength, inv2r = 1 / (2 * sr * sr), inv2s = 1 / (2 * 1.6 * 1.6);
  const sw = new Float32Array(25);
  for (let j = -2, k = 0; j <= 2; j++) for (let i = -2; i <= 2; i++, k++) sw[k] = Math.exp(-(i * i + j * j) * inv2s);
  const rw = new Float32Array(766);
  for (let d = 0; d < 766; d++) rw[d] = Math.exp(-((d / 3) * (d / 3)) * inv2r);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = (y * w + x) * 4, r0 = data[p], g0 = data[p + 1], b0 = data[p + 2];
    let r = 0, g = 0, b = 0, ws = 0, k = 0;
    for (let j = -2; j <= 2; j++) {
      const yy = Math.min(h - 1, Math.max(0, y + j));
      for (let i = -2; i <= 2; i++, k++) {
        const xx = Math.min(w - 1, Math.max(0, x + i)), q = (yy * w + xx) * 4;
        const dd = Math.abs(data[q] - r0) + Math.abs(data[q + 1] - g0) + Math.abs(data[q + 2] - b0);
        const wt = sw[k] * rw[dd];
        r += data[q] * wt; g += data[q + 1] * wt; b += data[q + 2] * wt; ws += wt;
      }
    }
    out.data[p] = r / ws; out.data[p + 1] = g / ws; out.data[p + 2] = b / ws; out.data[p + 3] = data[p + 3];
  }
  return out;
}

/** Feathered elliptical mask (0..255) covering face boxes, in output pixel space. */
export function faceMask(w: number, h: number, boxes: { x: number; y: number; w: number; h: number }[]): Uint8Array | undefined {
  if (!boxes.length) return undefined;
  const m = new Uint8Array(w * h);
  for (const b of boxes) {
    const cx = b.x + b.w / 2, cy = b.y + b.h / 2, rx = b.w * 0.75, ry = b.h * 0.85;
    const x0 = Math.max(0, Math.floor(cx - rx)), x1 = Math.min(w - 1, Math.ceil(cx + rx));
    const y0 = Math.max(0, Math.floor(cy - ry)), y1 = Math.min(h - 1, Math.ceil(cy + ry));
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const d = Math.hypot((x - cx) / rx, (y - cy) / ry);
      if (d >= 1) continue;
      const v = d < 0.7 ? 255 : Math.round(255 * (1 - (d - 0.7) / 0.3));
      const i = y * w + x; if (v > m[i]) m[i] = v;
    }
  }
  return m;
}
