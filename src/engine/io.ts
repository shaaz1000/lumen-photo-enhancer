// Decoding, encoding and server calls. Browser only.
import type { Img } from './ops';
import type { ServerRequest, ServerResult } from './pipeline';

export async function decodeFile(blob: Blob): Promise<{ img: Img; canvas: HTMLCanvasElement }> {
  let bmp: ImageBitmap;
  try {
    bmp = await createImageBitmap(blob, { imageOrientation: 'from-image', colorSpaceConversion: 'default' } as ImageBitmapOptions);
  } catch {
    throw new Error('This image could not be decoded. The file may be damaged or only partly downloaded.');
  }
  const canvas = document.createElement('canvas');
  canvas.width = bmp.width; canvas.height = bmp.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  const d = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return { img: { w: d.width, h: d.height, data: d.data }, canvas };
}

export function imgToCanvas(img: Img): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = img.w; c.height = img.h;
  c.getContext('2d')!.putImageData(new ImageData(img.data, img.w, img.h), 0, 0);
  return c;
}

export type ExportFormat = 'png' | 'jpeg' | 'webp';
export const MIME: Record<ExportFormat, string> = { png: 'image/png', jpeg: 'image/jpeg', webp: 'image/webp' };

export function supportsWebpEncode(): boolean {
  const c = document.createElement('canvas'); c.width = c.height = 1;
  return c.toDataURL('image/webp').startsWith('data:image/webp');
}

export function encode(canvas: HTMLCanvasElement, fmt: ExportFormat, quality: number): Promise<Blob> {
  return new Promise((res, rej) => {
    let src = canvas;
    if (fmt === 'jpeg') {
      // JPEG has no alpha: flatten onto white so transparent areas do not turn black
      src = document.createElement('canvas'); src.width = canvas.width; src.height = canvas.height;
      const ctx = src.getContext('2d')!; ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, src.width, src.height); ctx.drawImage(canvas, 0, 0);
    }
    src.toBlob((b) => (b ? res(b) : rej(new Error('The browser could not encode this format. Try PNG.'))), MIME[fmt], fmt === 'png' ? undefined : quality);
  });
}

/** Detects the companion FastAPI server when the app is served by it. */
export async function probeServer(): Promise<{ faceModel: boolean; model: string } | null> {
  try {
    const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), 2500);
    const r = await fetch('./api/health', { signal: ctl.signal, cache: 'no-store' });
    clearTimeout(t);
    if (!r.ok) return null;
    const j = await r.json();
    return j && j.status === 'ok' ? { faceModel: !!j.face_restoration, model: String(j.model) } : null;
  } catch { return null; }
}

export async function serverEnhance(src: Img, req: ServerRequest, signal?: AbortSignal): Promise<ServerResult> {
  const blob = await encode(imgToCanvas(src), 'png', 1);
  const fd = new FormData();
  fd.append('image', blob, 'input.png');
  fd.append('scale', String(req.scale));
  fd.append('model', req.model);
  fd.append('denoise', String(req.dni));
  fd.append('face_restore', String(req.faceRestore));
  fd.append('weak', req.weak ? '1' : '0');
  fd.append('landmarks', JSON.stringify(req.faces.filter((f) => f.lm).map((f) => f.lm)));
  const r = await fetch('./api/enhance', { method: 'POST', body: fd, signal });
  if (!r.ok) {
    let msg = `The server returned ${r.status}.`;
    try { const j = await r.json(); if (j.detail) msg = String(j.detail); } catch { /* keep */ }
    throw new Error(msg);
  }
  // One PNG, nothing stored on the server: when the texture source is requested
  // the server stacks it under the main result and says so in X-Lumen-Stack.
  const info = decodeURIComponent(r.headers.get('X-Lumen-Info') || '').split('|').filter(Boolean);
  const stacked = r.headers.get('X-Lumen-Stack') === '2';
  const { img } = await decodeFile(await r.blob());
  if (!stacked) return { primary: img, weak: null, info };
  const h = img.h / 2, n = img.w * h * 4;
  return {
    primary: { w: img.w, h, data: img.data.slice(0, n) },
    weak: { w: img.w, h, data: img.data.slice(n) },
    info,
  };
}
