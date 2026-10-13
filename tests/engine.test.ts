import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { readFileSync } from 'node:fs';
import { analyze, blurEffect, blockiness, noiseSigma, recommendScale, analysisCrop } from '../src/engine/analysis';
import { planTiles, allowedScales, fitPixels, MAX_OUTPUT_PIXELS } from '../src/engine/tiling';
import { validateUpload, validateDimensions, sniffFormat, MAX_FILE_BYTES } from '../src/engine/validate';
import { resample, sharpenNoHalo, luma, faceMask, planColor, applyColor, makeImg, type Img } from '../src/engine/ops';
import { writeTile } from '../src/engine/upscale';
import { enhance, srKey, faceRegion, type Runtime } from '../src/engine/pipeline';
import { paramsFor, effective, MODES } from '../src/engine/presets';
import { parseWeights } from '../src/engine/weights';
import { buildSrvgg } from '../src/engine/srvgg';

const meta = JSON.parse(readFileSync('tests/fixtures/analysis_meta.json', 'utf8'));
const fixture = (k: string): Img => ({ w: meta[k].w, h: meta[k].h, data: new Uint8ClampedArray(readFileSync(`tests/fixtures/${k}.rgba`)) });
const solid = (w: number, h: number, v: number): Img => { const i = makeImg(w, h); i.data.fill(v); return i; };

describe('analysis on synthetic degradations', () => {
  it('rates a sharp image crisp, clean and uncompressed', () => {
    const a = analyze(fixture('sharp'), meta.sharp.bytes, 'PNG', 0);
    expect([a.sharpness, a.compression, a.noise]).toEqual(['crisp', 'low', 'low']);
  });
  it('detects blur', () => {
    expect(analyze(fixture('blurry'), meta.blurry.bytes, 'PNG', 0).sharpness).toBe('soft');
  });
  it('detects heavy JPEG compression', () => {
    const a = analyze(fixture('jpeg'), meta.jpeg.bytes, 'JPEG', 0);
    expect(a.compression).toBe('heavy');
    expect(a.summary).toContain('heavily compressed');
    expect(a.summary).toMatch(/We recommend \dx|We recommend \d×/);
  });
  it('detects noise', () => {
    expect(analyze(fixture('noisy'), meta.noisy.bytes, 'PNG', 0).noise).toBe('high');
  });
  it('blocking grid ratio is ~1 for an uncompressed image and high for a JPEG', () => {
    const s = fixture('sharp'), j = fixture('jpeg');
    expect(blockiness(luma(s), s.w, s.h)).toBeLessThan(1.15);
    expect(blockiness(luma(j), j.w, j.h)).toBeGreaterThan(1.6);
  });
  it('flat images are not called noisy or blocky', () => {
    const f = solid(64, 64, 128);
    expect(noiseSigma(luma(f), 64, 64)).toBe(0);
    expect(blockiness(luma(f), 64, 64)).toBe(1);
    expect(blurEffect(luma(f), 64, 64)).toBeGreaterThanOrEqual(0);
  });
  it('reports faces in the opportunities', () => {
    expect(analyze(fixture('sharp'), 1000, 'PNG', 2).opportunities.join(' ')).toContain('2 faces');
  });
  it('keeps the 8 px grid when cropping big images', () => {
    const big = makeImg(2000, 1500);
    const c = analysisCrop(big);
    expect([c.w, c.h]).toEqual([1024, 1024]);
  });
  it('recommends a bigger factor for smaller images and never exceeds the output cap', () => {
    expect(recommendScale(300, 200)).toBe(8);
    expect(recommendScale(720, 480)).toBe(4);
    expect(recommendScale(2400, 1600)).toBe(2);
    expect(recommendScale(3000, 2900)).toBe(2);
  });
});

describe('tiling', () => {
  it('covers every pixel exactly once with cores', () => {
    const cover = new Uint8Array(301 * 157);
    for (const t of planTiles(301, 157, 64, 16)) {
      expect(t.px).toBeLessThanOrEqual(t.x); expect(t.px + t.pw).toBeGreaterThanOrEqual(t.x + t.w);
      for (let y = t.y; y < t.y + t.h; y++) for (let x = t.x; x < t.x + t.w; x++) cover[y * 301 + x]++;
    }
    expect(cover.every((v) => v === 1)).toBe(true);
  });
  it('handles extreme aspect ratios', () => {
    const tiles = planTiles(4000, 10, 128, 16);
    expect(tiles.length).toBe(32);
    expect(tiles.every((t) => t.ph === 10)).toBe(true);
  });
  it('limits scales to the output pixel budget', () => {
    expect(allowedScales(512, 512)).toEqual([2, 4, 8]);
    expect(allowedScales(1200, 900)).toEqual([2]);
    expect(allowedScales(5000, 4000)).toEqual([]);
    const f = fitPixels(8000, 6000, MAX_OUTPUT_PIXELS);
    expect(f.w * f.h).toBeLessThanOrEqual(MAX_OUTPUT_PIXELS);
    expect(f.w / f.h).toBeCloseTo(8000 / 6000, 2);
  });
});

describe('upload validation', () => {
  const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0]);
  const jpg = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 0, 0, 0, 0, 0, 0, 0]);
  const webp = new TextEncoder().encode('RIFF\0\0\0\0WEBPVP8 ');
  it('recognises formats by content', () => {
    expect(sniffFormat(png)).toBe('PNG'); expect(sniffFormat(jpg)).toBe('JPEG'); expect(sniffFormat(webp)).toBe('WebP');
    expect(sniffFormat(new TextEncoder().encode('GIF89a......'))).toBeNull();
  });
  it('rejects empty, oversized and disguised files with actionable messages', () => {
    expect(validateUpload(0, png)).toMatchObject({ ok: false });
    const big = validateUpload(MAX_FILE_BYTES + 1, png);
    expect(big.ok).toBe(false); if (!big.ok) expect(big.error).toContain('25 MB');
    const fake = validateUpload(100, new TextEncoder().encode('<html>hi</html>'));
    expect(fake.ok).toBe(false);
    expect(validateUpload(100, jpg)).toEqual({ ok: true, format: 'JPEG' });
  });
  it('rejects tiny, extreme and huge dimensions', () => {
    expect(validateDimensions(4, 400, 12e6)).toMatch(/too small/);
    expect(validateDimensions(4000, 100, 12e6)).toMatch(/20 times/);
    expect(validateDimensions(5000, 4000, 12e6)).toMatch(/megapixels/);
    expect(validateDimensions(1080, 1920, 12e6)).toBeNull();
  });
});

describe('conventional image operations', () => {
  it('resampling a flat image keeps it flat and sized correctly', () => {
    const r = resample(solid(10, 7, 90), 37, 26);
    expect([r.w, r.h]).toEqual([37, 26]);
    expect(r.data.every((v) => v === 90)).toBe(true);
  });
  it('downscaling keeps aspect and averages', () => {
    const img = makeImg(4, 4);
    for (let i = 0; i < 16; i++) { const v = i % 2 ? 255 : 0; img.data.set([v, v, v, 255], i * 4); }
    const r = resample(img, 2, 2);
    expect(Math.abs(r.data[0] - 128)).toBeLessThan(40);
  });
  it('sharpening never overshoots the local range (no halos)', () => {
    const img = makeImg(20, 1);
    for (let x = 0; x < 20; x++) { const v = x < 10 ? 50 : 200; img.data.set([v, v, v, 255], x * 4); }
    const s = sharpenNoHalo(img, 3);
    for (let p = 0; p < s.data.length; p += 4) { expect(s.data[p]).toBeGreaterThanOrEqual(50); expect(s.data[p]).toBeLessThanOrEqual(200); }
  });
  it('colour correction is bounded and leaves a neutral grey neutral', () => {
    const img = fixture('sharp');
    const plan = planColor(img);
    plan.gains.forEach((g) => { expect(g).toBeGreaterThanOrEqual(0.92); expect(g).toBeLessThanOrEqual(1.08); });
    const grey = applyColor(solid(8, 8, 128), { gains: [1, 1, 1], black: 0, white: 255, gamma: 1 }, 1);
    expect(grey.data[0]).toBe(128);
  });
  it('face mask is solid in the middle and feathered at the edge', () => {
    const m = faceMask(100, 100, [{ x: 40, y: 40, w: 20, h: 20 }])!;
    expect(m[50 * 100 + 50]).toBe(255);
    expect(m[0]).toBe(0);
    expect(faceMask(10, 10, [])).toBeUndefined();
  });
  it('face regions stay inside the image', () => {
    const r = faceRegion({ x: 2, y: 3, w: 30, h: 30 }, 40, 40);
    expect(r.x).toBe(0); expect(r.x + r.w).toBeLessThanOrEqual(40);
  });
});

describe('tile writer', () => {
  it('writes the core of a tile and box-averages for 2x output', () => {
    const t = { x: 1, y: 0, w: 1, h: 1, px: 0, py: 0, pw: 2, ph: 1 };
    const src = new Float32Array(2 * 4 * 1 * 4 * 3); // 8x4 output of a 2x1 tile at scale 4
    for (let i = 0; i < src.length; i += 3) { const x = (i / 3) % 8; src[i] = x >= 4 ? 1 : 0; }
    const dst = makeImg(4, 2);
    writeTile(dst, src, t, 4, 2);
    expect(dst.data[(0 * 4 + 2) * 4]).toBe(255); // core starts at x=2 in 2x space
    expect(dst.data[0]).toBe(0);
  });
});

describe('presets', () => {
  it('every mode yields controls in range', () => {
    for (const m of MODES) {
      const p = paramsFor(m.id, null, 4);
      for (const v of [p.denoise, p.sharpness, p.texture, p.color, p.faceProtect, p.strength]) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(1); }
    }
  });
  it('auto mode raises denoise for compressed images', () => {
    const jpeg = analyze(fixture('jpeg'), meta.jpeg.bytes, 'JPEG', 0);
    const clean = analyze(fixture('sharp'), meta.sharp.bytes, 'PNG', 0);
    expect(paramsFor('auto', jpeg, 4).denoise).toBeGreaterThan(paramsFor('auto', clean, 4).denoise);
  });
  it('face protection lowers denoise on faces', () => {
    const e = effective({ ...paramsFor('portrait', null, 4) });
    expect(e.faceDni).toBeLessThan(e.dni);
  });
  it('finishing-only changes keep the reconstruction cache key', () => {
    const p = paramsFor('auto', null, 4);
    expect(srKey({ ...p, sharpness: 0.9, color: 0.1 }, 'ai', [])).toBe(srKey(p, 'ai', []));
    expect(srKey({ ...p, denoise: 0.1 }, 'ai', [])).not.toBe(srKey(p, 'ai', []));
  });
});

describe('end-to-end pipeline (CPU)', () => {
  const load = (p: string) => { const b = readFileSync(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer; };
  const rt = async (): Promise<Runtime> => {
    await tf.setBackend('cpu');
    const w = parseWeights(load('public/models/general-x4v3.bin'));
    return { tf, tile: 16, getModel: async () => buildSrvgg(tf, w) };
  };
  const crop = (img: Img, w: number, h: number): Img => {
    const o = makeImg(w, h);
    for (let y = 0; y < h; y++) o.data.set(img.data.subarray(y * img.w * 4, (y * img.w + w) * 4), y * w * 4);
    return o;
  };

  it('AI 4x on a compressed crop returns the right size, opaque, and differs from bicubic', async () => {
    const clean = crop(fixture('sharp'), 48, 40), jpeg = crop(fixture('jpeg'), 48, 40);
    const small = resample(jpeg, 12, 10);
    const steps: string[] = [];
    const r = await enhance(small, [], { ...paramsFor('natural', null, 4), texture: 0, sharpness: 0, color: 0 }, 'ai', await rt(), (p) => steps.push(p.stage));
    expect([r.img.w, r.img.h]).toEqual([48, 40]);
    expect(steps).toContain('Reconstructing detail');
    expect(r.img.data[3]).toBe(255);
    const mse = (a: Img, b: Img) => { let s = 0; for (let i = 0; i < a.data.length; i += 4) for (let c = 0; c < 3; c++) s += (a.data[i + c] - b.data[i + c]) ** 2; return s / (a.w * a.h * 3); };
    const bic = resample(small, 48, 40);
    expect(Number.isFinite(mse(r.img, clean))).toBe(true);
    expect(mse(r.img, bic)).toBeGreaterThan(0); // the network changed something
  }, 120000);

  it('tiled and single-tile inference agree away from borders', async () => {
    const R = await rt();
    const img = crop(fixture('sharp'), 40, 24);
    const m = await R.getModel(1);
    const { runSr } = await import('../src/engine/upscale');
    const a = await runSr(tf, m, img, { tile: 12, pad: 24, outFactor: 4 });
    const b = await runSr(tf, m, img, { tile: 64, pad: 24, outFactor: 4 });
    let max = 0;
    for (let i = 0; i < a.data.length; i++) max = Math.max(max, Math.abs(a.data[i] - b.data[i]));
    expect(max).toBeLessThanOrEqual(2);
  }, 120000);

  it('conventional path is labelled as such', async () => {
    const r = await enhance(crop(fixture('sharp'), 20, 20), [], paramsFor('auto', null, 2), 'classic', null, () => {});
    expect([r.img.w, r.img.h]).toEqual([40, 40]);
    expect(r.notes.join(' ')).toMatch(/No AI reconstruction/);
  });

  it('can be cancelled', async () => {
    const ctl = new AbortController(); ctl.abort();
    await expect(enhance(crop(fixture('sharp'), 20, 20), [], paramsFor('auto', null, 2), 'ai', await rt(), () => {}, ctl.signal)).rejects.toThrow('Cancelled');
  });

  it('reports a failed server instead of faking a result', async () => {
    const failing = async () => { throw new Error('The server returned 503.'); };
    await expect(enhance(crop(fixture('sharp'), 20, 20), [], paramsFor('auto', null, 2), 'server', null, () => {}, undefined, failing)).rejects.toThrow('503');
  });
});
