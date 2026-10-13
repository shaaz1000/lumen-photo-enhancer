import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { readFileSync } from 'node:fs';
import { parseWeights, blendWeights, halfToFloat } from '../src/engine/weights';
import { buildSrvgg } from '../src/engine/srvgg';

const load = (p: string) => {
  const b = readFileSync(p);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};

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
    const i = 5;
    expect(m.tensors[0].data[i]).toBeCloseTo(0.25 * a.tensors[0].data[i] + 0.75 * b.tensors[0].data[i], 6);
    expect(blendWeights(a, b, 1).tensors[3].data).toEqual(a.tensors[3].data);
  });
});

describe('SRVGGNetCompact on TF.js matches the PyTorch-layout reference', () => {
  it('produces the same 4x output as the NumPy reference', async () => {
    await tf.setBackend('cpu');
    const ref = JSON.parse(readFileSync('tests/fixtures/srvgg_ref.json', 'utf8'));
    const model = buildSrvgg(tf, parseWeights(load('public/models/general-x4v3.bin')));
    const x = tf.tensor4d(ref.input_hwc, [1, ref.h, ref.w, 3]);
    const y = model.run(x);
    expect(y.shape).toEqual([1, ref.h * 4, ref.w * 4, 3]);
    const out = await y.data();
    let maxErr = 0;
    for (let i = 0; i < out.length; i++) maxErr = Math.max(maxErr, Math.abs(out[i] - ref.output_hwc[i]));
    expect(maxErr).toBeLessThan(1e-3);
    model.dispose();
  }, 60000);
});
