import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { readFileSync } from 'node:fs';
import { analyze, blurEffect, blockiness, noiseSigma, recommendScale, analysisCrop } from '../src/engine/analysis';
import { planTiles, allowedScales, fitPixels, MAX_OUTPUT_PIXELS } from '../src/engine/tiling';
import { validateUpload, validateDimensions, sniffFormat, MAX_FILE_BYTES } from '../src/engine/validate';
import { resample, sharpenNoHalo, luma, faceMask, planColor, applyColor, makeImg, type Img } from '../src/engine/ops';
import { planFeatherTiles, ramp, runSr } from '../src/engine/upscale';
import { enhance, srKey, faceRegion, type Runtime } from '../src/engine/pipeline';
import { paramsFor, planFor, MODES, type Caps } from '../src/engine/presets';
import { parseWeights } from '../src/engine/weights';
import { buildSrvgg, type SrModel } from '../src/engine/srvgg';

const meta = JSON.parse(readFileSync('tests/fixtures/analysis_meta.json', 'utf8'));
const fixture = (k: string): Img => ({ w: meta[k].w, h: meta[k].h, data: new Uint8ClampedArray(readFileSync(`tests/fixtures/${k}.rgba`)) });
const solid = (w: number, h: number, v: number): Img => { const i = makeImg(w, h); i.data.fill(v); return i; };

describe('analysis on synthetic degradations', () => {
  it('rates a sharp image crisp, clean and uncompressed', () => {
    const a = analyze(fixture('sharp'), meta.sharp.bytes, 'PNG', []);
    expect([a.sharpness, a.compression, a.noise]).toEqual(['crisp', 'low', 'low']);
  });
  it('detects blur', () => {
    expect(analyze(fixture('blurry'), meta.blurry.bytes, 'PNG', []).sharpness).toBe('soft');
  });
  it('classifies content', () => {
    expect(analyze(fixture('sharp'), 1000, 'PNG', [{ x: 60, y: 60, w: 90, h: 90 }]).content).toBe('portrait');
    expect(analyze(fixture('sharp'), meta.sharp.bytes, 'JPEG', []).content).toBe('photo');
  });
  it('detects heavy JPEG compression', () => {
    const a = analyze(fixture('jpeg'), meta.jpeg.bytes, 'JPEG', []);
    expect(a.compression).toBe('heavy');
    expect(a.summary).toContain('heavily compressed');
    expect(a.summary).toMatch(/We recommend .+ at \d×/);
  });
  it('separates mild softness from blur it cannot repair', () => {
    const soft = analyze(fixture('soft'), meta.soft.bytes, 'PNG', []);
    expect(soft.blur).toBe('mild');
    expect(soft.issue).toBe('soft');
    expect(soft.effectiveScale).toBeLessThan(0.9);
    const heavy = analyze(fixture('blurry'), meta.blurry.bytes, 'PNG', []);
    expect(heavy.issue).toBe('heavy-blur');
    expect(heavy.summary).toMatch(/cannot undo this much blur/);
    const motion = analyze(fixture('motion'), meta.motion.bytes, 'PNG', []);
    expect(motion.blur).toBe('motion');
  });
  it('sharp images keep full effective resolution', () => {
    expect(analyze(fixture('sharp'), meta.sharp.bytes, 'PNG', []).effectiveScale).toBe(1);
  });
  it('detects noise', () => {
    expect(analyze(fixture('noisy'), meta.noisy.bytes, 'PNG', []).noise).toBe('high');
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
    expect(analyze(fixture('sharp'), 1000, 'PNG', [{ x: 10, y: 10, w: 40, h: 40 }, { x: 100, y: 10, w: 40, h: 40 }]).opportunities.join(' ')).toContain('2 faces');
  });
  it('keeps the 8 px grid when cropping big images', () => {
    const big = makeImg(2000, 1500);
    const c = analysisCrop(big);
    expect([c.w, c.h]).toEqual([1024, 1024]);
  });
  it('recommends a bigger factor for smaller images and never exceeds the output cap', () => {
    expect(recommendScale(300, 200)).toBe(4);
    expect(recommendScale(720, 480, 0.6)).toBe(2);
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
    const strict = sharpenNoHalo(img, 3, undefined, 1, 0);
    for (let p = 0; p < strict.data.length; p += 4) { expect(strict.data[p]).toBeGreaterThanOrEqual(50); expect(strict.data[p]).toBeLessThanOrEqual(200); }
    // default allows a small, bounded overshoot (15% of the local range) for visible crispness
    const soft = sharpenNoHalo(img, 3);
    for (let p = 0; p < soft.data.length; p += 4) { expect(soft.data[p]).toBeGreaterThanOrEqual(50 - 23); expect(soft.data[p]).toBeLessThanOrEqual(200 + 23); }
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

describe('feathered tiling', () => {
  it('ramps sum to one across every seam', () => {
    const a = ramp(20, 0, 6), b = ramp(20, 6, 0);
    for (let i = 0; i < 6; i++) expect(a[14 + i] + b[i]).toBeCloseTo(1, 6);
  });
  it('extended regions cover the image and context stays in bounds', () => {
    const t = planFeatherTiles(301, 157, 64, 12, 32);
    for (const x of t) {
      expect(x.px0).toBeGreaterThanOrEqual(0); expect(x.px1).toBeLessThanOrEqual(301);
      expect(x.ex0).toBeLessThanOrEqual(x.x0); expect(x.ex1).toBeGreaterThanOrEqual(x.x1);
    }
    const cover = new Uint8Array(301 * 157);
    for (const x of t) for (let y = x.y0; y < x.y1; y++) for (let i = x.x0; i < x.x1; i++) cover[y * 301 + i]++;
    expect(cover.every((v) => v === 1)).toBe(true);
  });
});

const caps: Caps = { deepOk: true, server: false, faceRestoreOk: false };

describe('presets and planning', () => {
  it('every mode yields controls in range', () => {
    for (const m of MODES) {
      const p = paramsFor(m.id, null, 4);
      for (const v of [p.denoise, p.sharpness, p.texture, p.color, p.faceProtect, p.faceRestore]) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(1); }
    }
  });
  it('auto mode raises denoise for compressed images', () => {
    const jpeg = analyze(fixture('jpeg'), meta.jpeg.bytes, 'JPEG', []);
    const clean = analyze(fixture('sharp'), meta.sharp.bytes, 'PNG', []);
    expect(paramsFor('auto', jpeg, 4).denoise).toBeGreaterThan(paramsFor('auto', clean, 4).denoise);
  });
  it('faithfulness picks the model and reconstruction strength', () => {
    const base = paramsFor('auto', null, 2);
    expect(planFor({ ...base, fidelity: 'faithful' }, 'auto', null, caps).model).toBe('general');
    expect(planFor({ ...base, fidelity: 'balanced' }, 'auto', null, caps).model).toBe('deep');
    expect(planFor({ ...base, fidelity: 'max' }, 'natural', null, caps).model).toBe('deep');
    expect(planFor({ ...base, fidelity: 'balanced' }, 'natural', null, caps).model).toBe('general');
    const f = planFor({ ...base, fidelity: 'faithful' }, 'auto', null, caps).textureMix;
    const m = planFor({ ...base, fidelity: 'max' }, 'auto', null, caps).textureMix;
    expect(f).toBeGreaterThan(m);
  });
  it('Deep Restoration uses the deep model and matches detail scale on soft images', () => {
    const soft = analyze(fixture('soft'), meta.soft.bytes, 'PNG', []);
    const p = planFor(paramsFor('deep', soft, 2), 'deep', soft, caps);
    expect(p.model).toBe('deep');
    expect(p.preShrink).toBeLessThan(1);
    expect(p.preShrink).toBeGreaterThanOrEqual(0.5);
    expect(p.notes.join(' ')).toMatch(/Processed at/);
  });
  it('never shrinks so far that the target needs plain resampling', () => {
    const soft = analyze(fixture('blurry'), meta.blurry.bytes, 'PNG', []);
    expect(planFor(paramsFor('deep', soft, 4, 'max'), 'deep', soft, caps).preShrink).toBe(1);
  });
  it('falls back to the fast model without a GPU and says so', () => {
    const p = planFor(paramsFor('deep', null, 2), 'deep', null, { ...caps, deepOk: false });
    expect(p.model).toBe('general');
    expect(p.notes[0]).toMatch(/GPU/);
  });
  it('phones use the fast model for large photos unless Maximum Detail is chosen', () => {
    const big = { ...analyze(fixture('soft'), meta.soft.bytes, 'PNG', []), w: 1170, h: 2532 };
    const phone = { ...caps, mobile: true };
    const bal = planFor(paramsFor('deep', big, 2), 'deep', big, phone);
    expect(bal.model).toBe('general');
    expect(bal.notes.join(' ')).toMatch(/On phones/);
    expect(bal.textureMix).toBe(0);
    expect(planFor(paramsFor('deep', big, 2, 'max'), 'deep', big, phone).model).toBe('deep');
    const small = { ...big, w: 500, h: 500 };
    expect(planFor(paramsFor('deep', small, 2), 'deep', small, phone).model).toBe('deep');
    expect(planFor(paramsFor('deep', big, 2), 'deep', big, caps).model).toBe('deep');
  });
  it('face restoration only when the server offers it', () => {
    const p = paramsFor('deep', null, 2);
    expect(planFor(p, 'deep', null, caps).faceRestore).toBe(0);
    expect(planFor(p, 'deep', null, { ...caps, server: true, faceRestoreOk: true }).faceRestore).toBeGreaterThan(0);
  });
  it('finishing-only changes keep the reconstruction cache key', () => {
    const p = paramsFor('auto', null, 4);
    const k = (q: typeof p) => srKey(q.scale, planFor(q, 'auto', null, caps), 'ai', []);
    expect(k({ ...p, sharpness: 0.9, color: 0.1 })).toBe(k(p));
    expect(k({ ...p, fidelity: 'faithful' })).not.toBe(k(p));
  });
});

describe('end-to-end pipeline (CPU)', () => {
  const load = (p: string) => { const b = readFileSync(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer; };
  let general: SrModel;
  const rt = async (tile = 16): Promise<Runtime> => {
    await tf.setBackend('cpu');
    general ??= buildSrvgg(tf, parseWeights(load('public/models/general-x4v3.bin')));
    return { backend: 'CPU', tile: () => tile, getModel: async () => general };
  };
  const crop = (img: Img, w: number, h: number): Img => {
    const o = makeImg(w, h);
    for (let y = 0; y < h; y++) o.data.set(img.data.subarray(y * img.w * 4, (y * img.w + w) * 4), y * w * 4);
    return o;
  };
  const plan = (scale: 2 | 4, over: object = {}) => planFor({ ...paramsFor('natural', null, scale, 'faithful'), ...over }, 'natural', null, { ...caps, deepOk: false });

  it('AI 4x on a compressed crop returns the right size, opaque, verified, and differs from bicubic', async () => {
    const small = resample(crop(fixture('jpeg'), 48, 40), 12, 10);
    const steps: string[] = [];
    const r = await enhance(small, [], 4, plan(4, { texture: 0, sharpness: 0, color: 0 }), 'ai', await rt(), (p) => steps.push(p.stage));
    expect([r.img.w, r.img.h]).toEqual([48, 40]);
    expect(steps).toContain('Reconstructing detail');
    expect(r.img.data[3]).toBe(255);
    expect(r.models[0]).toMatch(/general v3/);
    expect(r.stages.join(' ')).toMatch(/Verified against the original/);
    const bic = resample(small, 48, 40);
    let d = 0; for (let i = 0; i < bic.data.length; i++) d += Math.abs(bic.data[i] - r.img.data[i]);
    expect(d).toBeGreaterThan(0);
  }, 120000);

  it('tiled and whole-image inference agree (feathered seams)', async () => {
    const R = await rt();
    const img = crop(fixture('sharp'), 40, 24);
    const m = await R.getModel('general', 1);
    const a = await runSr(m, img, { tile: 12, outFactor: 4 });
    const b = await runSr(m, img, { tile: 64, outFactor: 4 });
    let max = 0;
    for (let i = 0; i < a.data.length; i++) max = Math.max(max, Math.abs(a.data[i] - b.data[i]));
    expect(max).toBeLessThanOrEqual(2);
  }, 120000);

  it('2x output is the box average of the 4x output', async () => {
    const R = await rt(64);
    const img = crop(fixture('sharp'), 16, 12);
    const m = await R.getModel('general', 1);
    const a4 = await runSr(m, img, { tile: 64, outFactor: 4 }), a2 = await runSr(m, img, { tile: 64, outFactor: 2 });
    const avg = (a4.data[0] + a4.data[4] + a4.data[a4.w * 4] + a4.data[a4.w * 4 + 4]) / 4;
    expect(Math.abs(a2.data[0] - avg)).toBeLessThanOrEqual(1);
  }, 60000);

  it('conventional path is labelled as such', async () => {
    const r = await enhance(crop(fixture('sharp'), 20, 20), [], 2, plan(2), 'classic', null, () => {});
    expect([r.img.w, r.img.h]).toEqual([40, 40]);
    expect(r.models.join(' ')).toMatch(/no AI/);
  });

  it('can be cancelled', async () => {
    const ctl = new AbortController(); ctl.abort();
    await expect(enhance(crop(fixture('sharp'), 20, 20), [], 2, plan(2), 'ai', await rt(), () => {}, ctl.signal)).rejects.toThrow('Cancelled');
  });

  it('reports a failed server instead of faking a result', async () => {
    const failing = async () => { throw new Error('The server returned 503.'); };
    await expect(enhance(crop(fixture('sharp'), 20, 20), [], 2, plan(2), 'server', null, () => {}, undefined, failing)).rejects.toThrow('503');
  });

  it('refuses a broken model output instead of exporting it', async () => {
    const black: SrModel = { name: 'broken', scale: 4, pad: 0, feather: 0, dispose() {}, infer: async (_d, h, w) => new Float32Array(h * 4 * w * 4 * 3) };
    const bad: Runtime = { backend: 'test', tile: () => 64, getModel: async () => black };
    await expect(enhance(crop(fixture('sharp'), 20, 20), [], 2, plan(2, { texture: 0 }), 'ai', bad, () => {})).rejects.toThrow(/does not match the original/);
  });

  it('refuses NaN output', async () => {
    const nan: SrModel = { name: 'nan model', scale: 4, pad: 0, feather: 0, dispose() {}, infer: async (_d, h, w) => new Float32Array(h * 4 * w * 4 * 3).fill(NaN) };
    const bad: Runtime = { backend: 'test', tile: () => 64, getModel: async () => nan };
    await expect(enhance(crop(fixture('sharp'), 20, 20), [], 2, plan(2, { texture: 0 }), 'ai', bad, () => {})).rejects.toThrow(/invalid pixels/);
  });
});
