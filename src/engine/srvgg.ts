// SRVGGNetCompact (Real-ESRGAN "realesr-general-x4v3") implemented directly on
// TensorFlow.js ops, so it runs on the GPU through WebGL with no model server.
import type * as TF from '@tensorflow/tfjs';
import type { ModelWeights } from './weights';

export type Tf = typeof TF;

export interface SrModel {
  scale: number;
  /** x: [1, h, w, 3] float in 0..1. Returns [1, h*scale, w*scale, 3]. */
  run(x: TF.Tensor4D): TF.Tensor4D;
  dispose(): void;
}

export function buildSrvgg(tf: Tf, w: ModelWeights): SrModel {
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
      cur.b = tf.tensor1d(t.data);
      held.push(cur.b);
    } else {
      cur.a = tf.tensor1d(t.data);
      held.push(cur.a);
    }
  }
  if (cur.k) layers.push(cur as Layer);
  const scale = w.scale;

  return {
    scale,
    run(x) {
      return tf.tidy(() => {
        let y: TF.Tensor4D = x;
        for (const L of layers) {
          y = tf.add(tf.conv2d(y, L.k, 1, 'same'), L.b) as TF.Tensor4D;
          if (L.a) y = tf.prelu(y, L.a as unknown as TF.Tensor4D) as TF.Tensor4D;
        }
        y = tf.depthToSpace(y, scale, 'NHWC');
        const [, h, wd] = x.shape;
        const base = tf.image.resizeNearestNeighbor(x, [h * scale, wd * scale], false, false);
        return tf.add(y, base) as TF.Tensor4D;
      });
    },
    dispose() { held.forEach((t) => t.dispose()); },
  };
}
