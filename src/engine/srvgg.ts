// Super-resolution networks implemented directly on TensorFlow.js ops, so they
// run on the GPU through WebGL with no model server.
//   SRVGGNetCompact: Real-ESRGAN "realesr-general-x4v3" (1.2 M parameters, fast)
//   RRDBNet:         Real-ESRGAN "RealESRGAN_x4plus" (16.7 M parameters, deep)
import type * as TF from '@tensorflow/tfjs';
import type { ModelWeights } from './weights';

export type Tf = typeof TF;

/** Backend independent model handle. Data is HWC float RGB in 0..1. */
export interface SrModel {
  name: string;
  scale: number;
  /** context pixels each side a tile needs, and the feather band for blending tiles */
  pad: number;
  feather: number;
  infer(data: Float32Array, h: number, w: number): Promise<Float32Array>;
  dispose(): void;
}

type T4 = TF.Tensor4D;

function wrap(tf: Tf, name: string, scale: number, pad: number, feather: number, fwd: (x: T4) => T4, held: TF.Tensor[]): SrModel {
  return {
    name, scale, pad, feather,
    async infer(data, h, w) {
      const x = tf.tensor4d(data, [1, h, w, 3]);
      const y = tf.tidy(() => fwd(x));
      x.dispose();
      try {
        return (await y.data()) as Float32Array;
      } finally { y.dispose(); }
    },
    dispose() { held.forEach((t) => t.dispose()); },
  };
}

export function buildSrvgg(tf: Tf, w: ModelWeights, name = 'Real-ESRGAN general v3'): SrModel {
  const held: TF.Tensor[] = [];
  type Layer = { k: TF.Tensor4D; b: TF.Tensor1D; a?: TF.Tensor1D };
  const layers: Layer[] = [];
  let cur: Partial<Layer> = {};
  for (const t of w.tensors) {
    if (t.kind === 'conv') {
      if (cur.k) layers.push(cur as Layer);
      cur = { k: tf.tensor4d(t.data, t.shape as [number, number, number, number]) };
      held.push(cur.k!);
    } else if (t.kind === 'bias') {
      cur.b = tf.tensor1d(t.data); held.push(cur.b);
    } else {
      cur.a = tf.tensor1d(t.data); held.push(cur.a);
    }
  }
  if (cur.k) layers.push(cur as Layer);
  const scale = w.scale;
  const fwd = (x: T4): T4 => {
    let y: T4 = x;
    for (const L of layers) {
      y = tf.add(tf.conv2d(y, L.k, 1, 'same'), L.b) as T4;
      if (L.a) y = tf.prelu(y, L.a as unknown as T4) as T4;
    }
    y = tf.depthToSpace(y, scale, 'NHWC');
    const [, h, wd] = x.shape;
    const base = tf.image.resizeNearestNeighbor(x, [h * scale, wd * scale], false, false);
    return tf.add(y, base) as T4;
  };
  return wrap(tf, name, scale, 16, 8, fwd, held);
}

export function buildRrdb(tf: Tf, w: ModelWeights, name = 'Real-ESRGAN x4plus'): SrModel {
  const held: TF.Tensor[] = [];
  const P = new Map<string, TF.Tensor>();
  for (const t of w.tensors) {
    if (!t.name) throw new Error('Deep model file is missing tensor names.');
    const tensor = t.kind === 'conv' ? tf.tensor4d(t.data, t.shape as [number, number, number, number]) : tf.tensor1d(t.data);
    P.set(t.name, tensor); held.push(tensor);
  }
  const get = (k: string) => { const v = P.get(k); if (!v) throw new Error(`Deep model is missing ${k}.`); return v; };
  const conv = (x: T4, k: string) => tf.add(tf.conv2d(x, get(k + '.weight') as TF.Tensor4D, 1, 'same'), get(k + '.bias')) as T4;
  const lrelu = (x: T4) => tf.leakyRelu(x, 0.2) as T4;
  const nBlocks = new Set([...P.keys()].filter((k) => k.startsWith('body.')).map((k) => k.split('.')[1])).size;

  const rdb = (x: T4, k: string): T4 => {
    const feats: T4[] = [x];
    for (let j = 1; j <= 4; j++) {
      const inp = feats.length === 1 ? feats[0] : (tf.concat(feats, 3) as T4);
      feats.push(lrelu(conv(inp, `${k}.conv${j}`)));
    }
    const x5 = conv(tf.concat(feats, 3) as T4, `${k}.conv5`);
    return tf.add(tf.mul(x5, 0.2), x) as T4;
  };
  const fwd = (x: T4): T4 => {
    const feat = conv(x, 'conv_first');
    let t = feat;
    for (let i = 0; i < nBlocks; i++) {
      // free intermediate tensors block by block to keep GPU memory bounded
      const next = tf.tidy(() => {
        const o = rdb(rdb(rdb(t, `body.${i}.rdb1`), `body.${i}.rdb2`), `body.${i}.rdb3`);
        return tf.add(tf.mul(o, 0.2), t) as T4;
      });
      if (t !== feat) t.dispose();
      t = next;
    }
    let f = tf.add(feat, conv(t, 'conv_body')) as T4;
    for (const k of ['conv_up1', 'conv_up2']) {
      const [, h, wd] = f.shape;
      f = lrelu(conv(tf.image.resizeNearestNeighbor(f, [h * 2, wd * 2], false, false), k));
    }
    return conv(lrelu(conv(f, 'conv_hr')), 'conv_last');
  };
  return wrap(tf, name, 4, 32, 12, fwd, held);
}
