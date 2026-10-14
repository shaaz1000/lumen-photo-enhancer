import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { readFileSync, readdirSync } from 'node:fs';
import { parseWeights, blendWeights, halfToFloat, concatParts } from '../src/engine/weights';
import { buildRrdb, buildSrvgg } from '../src/engine/srvgg';

const load = (p: string) => {
  const b = readFileSync(p);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};
const deepParts = () => readdirSync('public/models').filter((f) => f.startsWith('deep-x4plus.part')).sort().map((f) => load(`public/models/${f}`));

describe('fp16 decoding', () => {
  it('decodes known half floats', () => {
    expect(halfToFloat(0x3c00)).toBe(1);
    expect(halfToFloat(0xc000)).toBe(-2);
    expect(halfToFloat(0x0000)).toBe(0);
    expect(halfToFloat(0x3555)).toBeCloseTo(0.33325, 4);
  });
});

describe('weight files', () => {
  it('rejects files that are not model weights', () => {
    expect(() => parseWeights(new TextEncoder().encode('hello world, not a model').buffer)).toThrow(/damaged/);
  });
  it('blends checkpoints linearly', () => {
    const a = parseWeights(load('public/models/general-x4v3.bin'));
    const b = parseWeights(load('public/models/general-wdn-x4v3.bin'));
    const m = blendWeights(a, b, 0.25);
    expect(m.tensors[0].data[5]).toBeCloseTo(0.25 * a.tensors[0].data[5] + 0.75 * b.tensors[0].data[5], 6);
    expect(blendWeights(a, b, 1).tensors[3].data).toEqual(a.tensors[3].data);
  });
  it('reassembles the split deep model', () => {
    const w = parseWeights(concatParts(deepParts()));
    expect(w.arch).toBe('RRDBNet');
    expect(w.tensors.reduce((n, t) => n + t.data.length, 0)).toBe(16_697_987);
    expect(w.tensors.every((t) => t.name)).toBe(true);
  });
  it('detects a missing part', () => {
    const parts = deepParts();
    expect(() => parseWeights(concatParts(parts.slice(0, 2)))).toThrow(/truncated/);
  });
});

async function parity(model: ReturnType<typeof buildSrvgg>, fixture: string) {
  const ref = JSON.parse(readFileSync(fixture, 'utf8'));
  const out = await model.infer(new Float32Array(ref.input_hwc), ref.h, ref.w);
  expect(out.length).toBe(ref.h * 4 * ref.w * 4 * 3);
  let maxErr = 0;
  for (let i = 0; i < out.length; i++) maxErr = Math.max(maxErr, Math.abs(out[i] - ref.output_hwc[i]));
  model.dispose();
  return maxErr;
}

describe('networks on TF.js match the PyTorch-layout references', () => {
  it('general v3 (SRVGGNetCompact)', async () => {
    await tf.setBackend('cpu');
    const err = await parity(buildSrvgg(tf, parseWeights(load('public/models/general-x4v3.bin'))), 'tests/fixtures/srvgg_ref.json');
    expect(err).toBeLessThan(1e-3);
  }, 60000);
  it('x4plus (RRDBNet)', async () => {
    await tf.setBackend('cpu');
    const err = await parity(buildRrdb(tf, parseWeights(concatParts(deepParts()))), 'tests/fixtures/rrdb_ref.json');
    expect(err).toBeLessThan(2e-3);
  }, 300000);
});
