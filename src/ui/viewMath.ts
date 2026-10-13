// Pure geometry for the zoomable viewer and comparison slider.

export interface View { x: number; y: number; z: number }

export function fitView(cw: number, ch: number, iw: number, ih: number, margin = 24): View {
  const z = Math.min((cw - margin * 2) / iw, (ch - margin * 2) / ih, 8);
  const zz = z > 0 && Number.isFinite(z) ? z : 1;
  return { z: zz, x: (cw - iw * zz) / 2, y: (ch - ih * zz) / 2 };
}

/** Zoom about a point in container coordinates, keeping that image point fixed. */
export function zoomAt(v: View, factor: number, px: number, py: number, min = 0.02, max = 16): View {
  const z = Math.min(max, Math.max(min, v.z * factor));
  const k = z / v.z;
  return { z, x: px - (px - v.x) * k, y: py - (py - v.y) * k };
}

/** Image-space fraction (0..1) under a container x coordinate. */
export function sliderFraction(containerX: number, v: View, imageW: number): number {
  const f = (containerX - v.x) / (imageW * v.z);
  return f < 0 ? 0 : f > 1 ? 1 : f;
}

/** Container x of the slider line for a given fraction. */
export function sliderX(frac: number, v: View, imageW: number): number {
  return v.x + frac * imageW * v.z;
}

/** Zoom level that shows one image pixel per device pixel. */
export const pixelZoom = (dpr: number) => 1 / (dpr || 1);

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1048576).toFixed(n < 10 * 1048576 ? 1 : 0)} MB`;
}

/** Export sizes that keep aspect ratio and never exceed the enhanced result. */
export function exportSizes(w: number, h: number, srcW: number, srcH: number) {
  const long = Math.max(w, h);
  const sizes = [{ id: 'full', label: 'Full', w, h }];
  for (const L of [3840, 2560, 2160, 1440, 1080]) {
    if (L < long - 32) {
      const k = L / long;
      sizes.push({ id: String(L), label: `${L} px`, w: Math.round(w * k), h: Math.round(h * k) });
    }
  }
  if (srcW < w) sizes.push({ id: 'orig', label: 'Original size', w: srcW, h: srcH });
  return sizes;
}
