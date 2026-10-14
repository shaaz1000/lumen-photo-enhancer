// Tile planning for memory-safe super resolution. Each tile is run with a
// context margin (pad) so seams fall where the network had full context.

export interface Tile {
  /** core region in input pixels */
  x: number; y: number; w: number; h: number;
  /** padded region actually fed to the network */
  px: number; py: number; pw: number; ph: number;
}

export function planTiles(width: number, height: number, tile: number, pad: number): Tile[] {
  if (width <= 0 || height <= 0) return [];
  const t = Math.max(16, Math.floor(tile));
  const tiles: Tile[] = [];
  for (let y = 0; y < height; y += t) {
    for (let x = 0; x < width; x += t) {
      const w = Math.min(t, width - x);
      const h = Math.min(t, height - y);
      const px = Math.max(0, x - pad);
      const py = Math.max(0, y - pad);
      const pw = Math.min(width, x + w + pad) - px;
      const ph = Math.min(height, y + h + pad) - py;
      tiles.push({ x, y, w, h, px, py, pw, ph });
    }
  }
  return tiles;
}

/** Picks a tile size that keeps activation memory bounded on phones and laptops. */
export function chooseTileSize(isMobile: boolean, maxTextureSize: number, deep = false): number {
  const base = deep ? (isMobile ? 96 : 128) : isMobile ? 128 : 192;
  return Math.max(64, Math.min(base, Math.floor(maxTextureSize / 16)));
}

// Phones: a memory budget, well under the 16.7 MP iOS canvas ceiling, because the browser
// page is killed (and reloads) when finishing and encoding a larger result.
export const MOBILE_OUTPUT_PIXELS = 6_000_000;
export const DESKTOP_OUTPUT_PIXELS = 34_000_000; // fits 1080p at 4x (33.2 MP)
export let MAX_OUTPUT_PIXELS = MOBILE_OUTPUT_PIXELS;
/** Desktop browsers handle larger canvases; phones stay at the iOS limit. */
export function setOutputCap(pixels: number) { MAX_OUTPUT_PIXELS = pixels; }
export const MAX_INPUT_PIXELS = 12_000_000;

export type Scale = 2 | 4 | 8;

export function outputSize(w: number, h: number, s: number) {
  return { w: Math.round(w * s), h: Math.round(h * s) };
}

/** 2x is always offered (shrunk to the device limit if needed); 4x and 8x only when they fit. */
export function allowedScales(w: number, h: number): Scale[] {
  return ([2, 4, 8] as Scale[]).filter((s) => s === 2 || w * h * s * s <= MAX_OUTPUT_PIXELS);
}

/** Output size for a scale, limited to the device's pixel budget, keeping aspect ratio. */
export function outputDims(w: number, h: number, scale: number) {
  if (w * h * scale * scale <= MAX_OUTPUT_PIXELS) return { W: w * scale, H: h * scale, capped: false };
  const k = Math.sqrt(MAX_OUTPUT_PIXELS / (w * h));
  return { W: Math.max(1, Math.floor(w * k)), H: Math.max(1, Math.floor(h * k)), capped: true };
}

/** Fits a size inside a pixel budget, keeping aspect ratio. */
export function fitPixels(w: number, h: number, maxPixels: number) {
  if (w * h <= maxPixels) return { w, h };
  const k = Math.sqrt(maxPixels / (w * h));
  return { w: Math.max(1, Math.floor(w * k)), h: Math.max(1, Math.floor(h * k)) };
}
