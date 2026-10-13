// Upload validation by content (magic bytes), not by file name or MIME claims.

export const MAX_FILE_BYTES = 25 * 1024 * 1024;
export type Format = 'PNG' | 'JPEG' | 'WebP';

export function sniffFormat(head: Uint8Array): Format | null {
  if (head.length >= 8 && head[0] === 0x89 && head[1] === 0x50 && head[2] === 0x4e && head[3] === 0x47 && head[4] === 0x0d && head[5] === 0x0a && head[6] === 0x1a && head[7] === 0x0a) return 'PNG';
  if (head.length >= 3 && head[0] === 0xff && head[1] === 0xd8 && head[2] === 0xff) return 'JPEG';
  if (head.length >= 12 && head[0] === 0x52 && head[1] === 0x49 && head[2] === 0x46 && head[3] === 0x46 && head[8] === 0x57 && head[9] === 0x45 && head[10] === 0x42 && head[11] === 0x50) return 'WebP';
  return null;
}

export type Validation = { ok: true; format: Format } | { ok: false; error: string };

export function validateUpload(size: number, head: Uint8Array): Validation {
  if (size === 0) return { ok: false, error: 'This file is empty. Choose another image.' };
  if (size > MAX_FILE_BYTES) return { ok: false, error: `This file is ${(size / 1048576).toFixed(1)} MB. The limit is 25 MB; export a smaller copy and try again.` };
  const format = sniffFormat(head);
  if (!format) return { ok: false, error: 'This file is not a PNG, JPEG or WebP image. HEIC photos need converting to JPEG first.' };
  return { ok: true, format };
}

export function validateDimensions(w: number, h: number, maxPixels: number): string | null {
  if (w < 8 || h < 8) return 'This image is smaller than 8 pixels on a side, too small to enhance.';
  if (Math.max(w, h) / Math.min(w, h) > 20) return 'This image is more than 20 times longer than it is wide. Crop it closer and try again.';
  if (w * h > maxPixels) return `This image has ${(w * h / 1e6).toFixed(1)} megapixels. The limit is ${maxPixels / 1e6} megapixels; it is already large enough for most uses.`;
  return null;
}
