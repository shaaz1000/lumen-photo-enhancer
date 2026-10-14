// Parses the compact Real-ESRGAN weight files produced by tools/convert_weights.py
// and blends the two "general" checkpoints the same way Real-ESRGAN's
// --denoise_strength option does (deep network interpolation).

export type TensorKind = 'conv' | 'bias' | 'prelu';
export interface RawTensor { kind: TensorKind; shape: number[]; data: Float32Array; name?: string }
export interface ModelWeights { arch: string; scale: number; tensors: RawTensor[] }

const HALF_TABLE = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    if (e === 0) t[h] = s * Math.pow(2, -14) * (f / 1024);
    else if (e === 31) t[h] = f ? NaN : s * Infinity;
    else t[h] = s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return t;
})();

export function halfToFloat(h: number): number { return HALF_TABLE[h & 0xffff]; }

export function parseWeights(buf: ArrayBuffer): ModelWeights {
  const bytes = new Uint8Array(buf);
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== 'RESR') throw new Error('Model file is damaged or not a Real-ESRGAN weight file.');
  const dv = new DataView(buf);
  const version = dv.getUint32(4, true);
  if (version !== 1 && version !== 2) throw new Error(`Unsupported weight file version ${version}.`);
  const hlen = dv.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + hlen)));
  let off = 12 + hlen;
  off += off % 2;
  const tensors: RawTensor[] = [];
  for (const t of header.tensors as { kind: TensorKind; shape: number[]; name?: string }[]) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    if (off + n * 2 > buf.byteLength) throw new Error('Model file is truncated.');
    const halves = new Uint16Array(buf.slice(off, off + n * 2));
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = HALF_TABLE[halves[i]];
    tensors.push({ kind: t.kind, shape: t.shape, data, name: t.name });
    off += n * 2;
  }
  return { arch: header.arch, scale: header.scale, tensors };
}

/** Joins split weight files back into one buffer. */
export function concatParts(parts: ArrayBuffer[]): ArrayBuffer {
  const n = parts.reduce((a, p) => a + p.byteLength, 0);
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(new Uint8Array(p), o); o += p.byteLength; }
  return out.buffer;
}

/** strength 1 = full denoise checkpoint, 0 = weak-denoise checkpoint. */
export function blendWeights(strong: ModelWeights, weak: ModelWeights, strength: number): ModelWeights {
  const a = Math.min(1, Math.max(0, strength));
  if (strong.tensors.length !== weak.tensors.length) throw new Error('Model checkpoints do not match.');
  return {
    arch: strong.arch,
    scale: strong.scale,
    tensors: strong.tensors.map((t, i) => {
      const u = weak.tensors[i];
      const out = new Float32Array(t.data.length);
      for (let j = 0; j < out.length; j++) out[j] = a * t.data[j] + (1 - a) * u.data[j];
      return { kind: t.kind, shape: t.shape, data: out };
    }),
  };
}
