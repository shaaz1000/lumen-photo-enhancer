// Modes, the faithfulness setting, and the planner that turns user controls
// plus image analysis into concrete processing decisions. Every control here
// changes what the pipeline actually computes.
import type { Analysis } from './analysis';
import type { Scale } from './tiling';

export type Fidelity = 'faithful' | 'balanced' | 'max';

export interface Params {
  scale: Scale;
  /** general model denoise strength (checkpoint interpolation); for the deep model it sets how much clean-up wins over texture */
  denoise: number;
  /** halo-limited sharpening */
  sharpness: number;
  /** keeps skin texture on faces: less sharpening and more natural grain there */
  faceProtect: number;
  /** mixes in the weak-denoise model's output to keep natural grain and fine texture */
  texture: number;
  /** auto levels and white balance */
  color: number;
  fidelity: Fidelity;
  /** generative face restoration strength (server only, 0 = off) */
  faceRestore: number;
}

export type ModeId = 'auto' | 'deep' | 'restore' | 'portrait' | 'screenshot' | 'ultra' | 'natural' | 'social';

export interface Mode { id: ModeId; name: string; blurb: string; params: Omit<Params, 'scale' | 'fidelity'> }

export const MODES: Mode[] = [
  { id: 'auto', name: 'Auto', blurb: 'Tuned from the analysis of this image', params: { denoise: 0.6, sharpness: 0.35, faceProtect: 0.5, texture: 0.25, color: 0.35, faceRestore: 0 } },
  { id: 'deep', name: 'Deep Restoration', blurb: 'Maximum clarity for soft, low resolution internet photos', params: { denoise: 0.75, sharpness: 0.45, faceProtect: 0.35, texture: 0.15, color: 0.35, faceRestore: 0.5 } },
  { id: 'restore', name: 'Restoration', blurb: 'Old, faded or heavily compressed photos', params: { denoise: 0.9, sharpness: 0.35, faceProtect: 0.5, texture: 0.1, color: 0.5, faceRestore: 0.4 } },
  { id: 'portrait', name: 'Portrait', blurb: 'Natural faces and skin texture', params: { denoise: 0.55, sharpness: 0.3, faceProtect: 0.8, texture: 0.35, color: 0.3, faceRestore: 0.35 } },
  { id: 'screenshot', name: 'Screenshot to HD', blurb: 'Screenshots of existing photos', params: { denoise: 0.8, sharpness: 0.4, faceProtect: 0.5, texture: 0.15, color: 0.2, faceRestore: 0 } },
  { id: 'ultra', name: 'Ultra Sharp', blurb: 'Crisper edges, no halos', params: { denoise: 0.7, sharpness: 0.85, faceProtect: 0.4, texture: 0.1, color: 0.3, faceRestore: 0 } },
  { id: 'natural', name: 'Natural', blurb: 'Minimal change, maximum authenticity', params: { denoise: 0.35, sharpness: 0.15, faceProtect: 0.8, texture: 0.5, color: 0.1, faceRestore: 0 } },
  { id: 'social', name: 'Social Ready', blurb: 'Posts, reels, stories, thumbnails', params: { denoise: 0.7, sharpness: 0.5, faceProtect: 0.5, texture: 0.15, color: 0.5, faceRestore: 0 } },
];

export const FIDELITY: { id: Fidelity; name: string; note: string }[] = [
  { id: 'faithful', name: 'Faithful', note: 'Fast model, minimal reconstruction, keeps original detail.' },
  { id: 'balanced', name: 'Balanced', note: 'Deep model with moderate reconstruction. Visible gain in clarity.' },
  { id: 'max', name: 'Maximum Detail', note: 'Strongest reconstruction. Adds inferred texture that may not match the real scene; it cannot recover detail that was never captured.' },
];

export function paramsFor(mode: ModeId, a: Analysis | null, scale: Scale, fidelity: Fidelity = 'balanced'): Params {
  const base: Params = { ...MODES.find((m) => m.id === mode)!.params, scale, fidelity };
  if (mode !== 'auto' || !a) return base;
  const p = { ...base };
  p.denoise = a.compression === 'heavy' || a.noise === 'high' ? 0.9 : a.compression === 'moderate' || a.noise === 'moderate' ? 0.7 : 0.5;
  p.sharpness = a.sharpness === 'soft' ? 0.45 : a.sharpness === 'moderate' ? 0.35 : 0.15;
  p.texture = a.noise === 'high' ? 0.1 : a.sharpness === 'crisp' ? 0.35 : 0.25;
  p.faceProtect = a.faces ? 0.6 : 0.5;
  return p;
}

export type ModelKind = 'general' | 'deep';

export interface Plan {
  model: ModelKind;
  dni: number;
  /** process at this fraction of the input size so the model sees detail at its native scale */
  preShrink: number;
  /** mix toward the weak-denoise model output */
  textureMix: number;
  faceTextureBoost: number;
  faceSharpenCut: number;
  sharpen: number;
  color: number;
  faceRestore: number;
  notes: string[];
}

export interface Caps { deepOk: boolean; server: boolean; faceRestoreOk: boolean; /** phone or tablet running in the browser */ mobile?: boolean }

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function planFor(p: Params, mode: ModeId, a: Analysis | null, caps: Caps): Plan {
  const notes: string[] = [];
  const wantDeep = mode === 'deep' || p.fidelity === 'max' || (p.fidelity === 'balanced' && mode !== 'natural');
  let model: ModelKind = wantDeep && caps.deepOk ? 'deep' : 'general';
  if (wantDeep && !caps.deepOk) notes.push('Deep model needs GPU acceleration, which is not available here, so the fast model was used.');
  if (p.fidelity === 'faithful') model = 'general';
  // Phones: the deep model on a large photo needs minutes and more memory than
  // many phones give a browser tab, so it only runs there when asked for explicitly.
  const px = a ? a.w * a.h : 0;
  const phone = !!caps.mobile && !caps.server;
  if (phone && model === 'deep' && px > 300_000) {
    if (p.fidelity === 'max') notes.push('Deep model on a phone: this can take several minutes and may run out of memory on large photos.');
    else { model = 'general'; notes.push('On phones, photos over 0.3 megapixels use the fast model to stay within memory. Choose Maximum Detail to force the deep model (slow).'); }
  }

  // Effective detail scale: a soft 1000 px image may only hold 600 px of detail.
  const eff = a?.effectiveScale ?? 1;
  const minF = Math.min(p.scale, 4) / 4; // keep the result mostly AI reconstructed, never upscaled by resampling
  let f = 1;
  if (p.fidelity === 'max') f = clamp(eff + 0.1, 0.6, 1);
  else if (mode === 'deep') f = clamp(eff + 0.15, 0.7, 1);
  else if (p.fidelity === 'balanced' && eff <= 0.75) f = 0.85;
  f = Math.max(f, minF);
  f = f > 0.97 ? 1 : Math.round(f * 100) / 100;
  if (f < 1) notes.push(`Processed at ${Math.round(f * 100)}% size first, because the image holds less detail than its pixel count. This lets the model reconstruct edges instead of enlarging blur.`);

  const fid = p.fidelity === 'faithful' ? 0 : p.fidelity === 'balanced' ? 1 : 2;
  let textureMix = p.texture * [0.8, 0.5, 0.25][fid];
  if (model === 'deep') textureMix += (1 - p.denoise) * 0.3;
  textureMix = clamp(textureMix, 0, 0.8);
  // the texture source is a second full model run; skip it for big phone outputs
  const skipWeak = phone && px * p.scale * p.scale > 4_000_000;
  if (skipWeak) { textureMix = 0; notes.push('Texture mixing is skipped on phones for large outputs to save memory.'); }

  const sharpen = p.sharpness * [0.6, 1, 1.4][fid] * (model === 'deep' ? 0.7 : 1);
  const color = p.color * [0.6, 1, 1.1][fid];
  let faceRestore = caps.faceRestoreOk ? p.faceRestore : 0;
  if (p.fidelity === 'faithful') faceRestore = Math.min(faceRestore, 0.3);
  if (faceRestore > 0) notes.push(`Generative face restoration at ${Math.round(faceRestore * 100)}%. Facial detail is partly inferred; likeness is kept as close as possible but is not guaranteed.`);

  return {
    model, dni: p.denoise, preShrink: f, textureMix,
    // with generative restoration on, faces keep the restored detail instead of extra grain
    faceTextureBoost: faceRestore > 0 || skipWeak ? 0 : 0.25 * p.faceProtect, faceSharpenCut: 0.6 * p.faceProtect,
    sharpen, color, faceRestore, notes,
  };
}
