import type { Analysis } from './analysis';
import type { Scale } from './tiling';

export interface Params {
  scale: Scale;
  /** Real-ESRGAN denoise strength (blend between the two general checkpoints) */
  denoise: number;
  /** halo-free unsharp amount */
  sharpness: number;
  /** 0 = off. Lowers denoise and sharpening on detected faces so skin keeps its real texture. */
  faceProtect: number;
  /** blends the plain resampled original back in to avoid a plastic look */
  texture: number;
  /** auto levels + white balance strength */
  color: number;
  /** natural (0) to stronger (1) */
  strength: number;
}

export type ModeId = 'auto' | 'restore' | 'portrait' | 'screenshot' | 'ultra' | 'natural' | 'social';

export interface Mode { id: ModeId; name: string; blurb: string; params: Omit<Params, 'scale'> }

export const MODES: Mode[] = [
  { id: 'auto', name: 'Auto', blurb: 'Tuned from the analysis of this image', params: { denoise: 0.6, sharpness: 0.3, faceProtect: 0.7, texture: 0.15, color: 0.35, strength: 0.5 } },
  { id: 'restore', name: 'Restoration', blurb: 'Old or heavily compressed photos', params: { denoise: 0.9, sharpness: 0.35, faceProtect: 0.6, texture: 0.08, color: 0.6, strength: 0.6 } },
  { id: 'portrait', name: 'Portrait', blurb: 'Natural faces and skin texture', params: { denoise: 0.45, sharpness: 0.2, faceProtect: 0.9, texture: 0.3, color: 0.3, strength: 0.4 } },
  { id: 'screenshot', name: 'Screenshot to HD', blurb: 'Screenshots of existing photos', params: { denoise: 0.75, sharpness: 0.35, faceProtect: 0.7, texture: 0.12, color: 0.2, strength: 0.55 } },
  { id: 'ultra', name: 'Ultra Sharp', blurb: 'Crisper edges, no halos', params: { denoise: 0.6, sharpness: 0.85, faceProtect: 0.6, texture: 0.08, color: 0.3, strength: 0.8 } },
  { id: 'natural', name: 'Natural', blurb: 'Minimal change, maximum authenticity', params: { denoise: 0.3, sharpness: 0.1, faceProtect: 0.8, texture: 0.35, color: 0.1, strength: 0.2 } },
  { id: 'social', name: 'Social Ready', blurb: 'Posts, reels, stories, thumbnails', params: { denoise: 0.65, sharpness: 0.45, faceProtect: 0.7, texture: 0.12, color: 0.5, strength: 0.6 } },
];

export function paramsFor(mode: ModeId, a: Analysis | null, scale: Scale): Params {
  const base = { ...MODES.find((m) => m.id === mode)!.params, scale };
  if (mode !== 'auto' || !a) return base;
  const p = { ...base };
  p.denoise = a.compression === 'heavy' || a.noise === 'high' ? 0.9 : a.compression === 'moderate' || a.noise === 'moderate' ? 0.7 : 0.45;
  p.sharpness = a.sharpness === 'soft' ? 0.45 : a.sharpness === 'moderate' ? 0.3 : 0.12;
  p.texture = a.noise === 'high' ? 0.06 : a.sharpness === 'crisp' ? 0.25 : 0.15;
  p.faceProtect = a.faces ? 0.85 : 0.7;
  return p;
}

/** Turns user-facing 0..1 controls into the numbers the pipeline uses. */
export function effective(p: Params) {
  const s = p.strength;
  return {
    dni: p.denoise,
    sharpen: p.sharpness * (0.5 + 0.9 * s) * 1.4,
    texture: Math.min(0.5, p.texture * (1.3 - 0.6 * s) * 0.6),
    color: p.color * (0.6 + 0.8 * s),
    faceDni: p.denoise * (1 - 0.6 * p.faceProtect),
    faceSharpenCut: 0.6 * p.faceProtect,
    faceTextureBoost: 0.25 * p.faceProtect,
  };
}
