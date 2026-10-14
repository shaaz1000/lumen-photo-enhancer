// End-to-end test in headless Chromium (software WebGL, so it is slow but real).
// Usage: CHROME=/path/to/chrome node tests/e2e/smoke.mjs
// Serves dist/ locally, maps the pinned CDN scripts to node_modules copies, then:
//   runs the fast model (Faithful) and the deep model (Deep Restoration) on a
//   small image through WebGL, checks sizes and the processing report, drags the
//   slider, checks zoom levels, downloads the PNG and proves it holds exactly the
//   previewed pixels, rejects a corrupt file, and checks the phone layout.
import { chromium } from 'playwright-core';
import { createServer } from 'node:http';
import { readFileSync, mkdirSync, existsSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { extname, join } from 'node:path';

const root = new URL('../../', import.meta.url).pathname;
const dist = join(root, 'dist');
const out = join(root, 'tests/e2e/out');
mkdirSync(out, { recursive: true });

const types = { '.html': 'text/html', '.wasm': 'application/wasm', '.json': 'application/json', '.js': 'text/javascript' };
const server = createServer((req, res) => {
  let p = decodeURIComponent(req.url.split('?')[0]);
  if (p === '/') p = '/index.html';
  const f = join(dist, p);
  if (!f.startsWith(dist) || !existsSync(f)) { res.writeHead(404); res.end(); return; }
  let body = readFileSync(f);
  if (p === '/index.html') body = Buffer.from('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>' + body + '</body></html>');
  res.writeHead(200, { 'content-type': types[extname(f)] || 'application/octet-stream' });
  res.end(body);
}).listen(0);
const base = `http://127.0.0.1:${server.address().port}/`;

const vendor = {
  'react.production.min.js': join(root, 'node_modules/react/umd/react.production.min.js'),
  'react-dom.production.min.js': join(root, 'node_modules/react-dom/umd/react-dom.production.min.js'),
  'tf.es2017.min.js': join(root, 'node_modules/@tensorflow/tfjs/dist/tf.es2017.min.js'),
};

const browser = await chromium.launch({ executablePath: process.env.CHROME, args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
let failures = 0;
const check = (ok, msg) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${msg}`); if (!ok) failures++; };

async function page(opts) {
  const ctx = await browser.newContext(opts);
  const pg = await ctx.newPage();
  pg.on('pageerror', (e) => console.log('pageerror:', e.message));
  pg.on('console', (m) => { if (m.type() === 'error') console.log('console:', m.text()); });
  await pg.route(/^https:\/\/(cdnjs|cdn\.jsdelivr)/, (r) => {
    const name = Object.keys(vendor).find((k) => r.request().url().endsWith(k));
    return name ? r.fulfill({ body: readFileSync(vendor[name]), contentType: 'text/javascript' }) : r.abort();
  });
  await pg.goto(base);
  return pg;
}

// RGB pixels of a PNG (8-bit RGB/RGBA, non-interlaced), for the export check
function pngRgb(buf) {
  let p = 8, w = 0, h = 0, ct = 0; const idat = [];
  while (p < buf.length) {
    const len = buf.readUInt32BE(p), type = buf.toString('latin1', p + 4, p + 8);
    if (type === 'IHDR') { w = buf.readUInt32BE(p + 8); h = buf.readUInt32BE(p + 12); ct = buf[p + 17]; }
    if (type === 'IDAT') idat.push(buf.subarray(p + 8, p + 8 + len));
    p += 12 + len;
  }
  const bpp = ct === 6 ? 4 : 3, stride = w * bpp, z = inflateSync(Buffer.concat(idat));
  const out = Buffer.alloc(w * h * 3); let prev = Buffer.alloc(stride);
  for (let y = 0; y < h; y++) {
    const f = z[y * (stride + 1)], row = z.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1)), cur = Buffer.alloc(stride);
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? cur[i - bpp] : 0, b = prev[i], c = i >= bpp ? prev[i - bpp] : 0;
      let v = row[i];
      if (f === 1) v += a; else if (f === 2) v += b; else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) { const q = a + b - c, pa = Math.abs(q - a), pb = Math.abs(q - b), pc = Math.abs(q - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      cur[i] = v & 255;
    }
    for (let x = 0; x < w; x++) for (let k = 0; k < 3; k++) out[(y * w + x) * 3 + k] = cur[x * bpp + k];
    prev = cur;
  }
  return { w, h, rgb: out };
}

const pg = await page({ viewport: { width: 1360, height: 860 }, colorScheme: 'light', acceptDownloads: true });
await pg.waitForSelector('.summary', { timeout: 30000 });
await pg.waitForFunction(() => document.querySelector('.status')?.textContent?.includes('AI ready'), null, { timeout: 60000 });
check(true, 'AI engine started: ' + (await pg.textContent('.status')));
await pg.screenshot({ path: join(out, 'desktop-loaded.png') });

async function load(file) {
  const prev = await pg.textContent('.summary').catch(() => '');
  await pg.setInputFiles('#file', file);
  await pg.waitForFunction((p) => { const s = document.querySelector('.summary')?.textContent; return s && s !== p; }, prev, { timeout: 60000 });
  await pg.waitForFunction(() => ![...document.querySelectorAll('.facts dd')].some((d) => d.textContent?.includes('Checking')), null, { timeout: 120000 });
}
async function enhance(label) {
  const t0 = Date.now();
  await pg.click('.inspector button.btn.primary.wide');
  await pg.waitForSelector('.progress', { timeout: 10000 }).catch(() => {});
  await pg.waitForSelector('.progress', { state: 'detached', timeout: 1800000 });
  const err = await pg.locator('.alert[role=alert]').textContent().catch(() => null);
  if (err) { check(false, `${label}: ${err}`); return null; }
  const dims = await pg.evaluate(async () => { const ims = [...document.querySelectorAll('img.layer')]; await Promise.all(ims.map((i) => i.decode().catch(() => {}))); return ims.map((x) => [x.naturalWidth, x.naturalHeight]); });
  const report = await pg.locator('.card.quiet').textContent();
  console.log(`  ${label}: ${((Date.now() - t0) / 1000).toFixed(1)} s, canvases ${JSON.stringify(dims)}`);
  return { dims, report };
}

await load(join(root, 'tests/e2e/small.png'));
await pg.click('[aria-label="Upscale factor"] >> text=2×');

// 1. fast model through WebGL
await pg.click('[aria-label="Reconstruction strength"] >> text=Faithful');
const f = await enhance('Faithful (general v3)');
check(f && f.dims[1][0] === 240 && f.dims[1][1] === 160, 'Faithful produced 240 × 160');
check(f && /general v3/.test(f.report) && /Verified against the original/.test(f.report), 'report names the fast model and the verification');

// 2. deep model through WebGL
await pg.click('.modes >> text=Deep Restoration');
await pg.click('[aria-label="Reconstruction strength"] >> text=Balanced');
const est = await pg.locator('.inspector').textContent();
check(/Estimated time on this device/.test(est), 'shows a time estimate before running');
// software GL here is very slow, so Balanced should fall back to the fast model with a note
check(/would take about \d+ min on this device/.test(est), 'slow device: Balanced explains why it uses the fast model');
await pg.click('[aria-label="Reconstruction strength"] >> text=Maximum Detail');
const d = await enhance('Deep Restoration, Maximum Detail (x4plus)');
check(d && d.dims[1][0] === 240, 'Deep Restoration produced 240 × 160');
check(d && /x4plus/.test(d.report) && /GPU \(WebGL\)/.test(d.report), 'report names x4plus running on WebGL');
await pg.waitForTimeout(1200);
await pg.screenshot({ path: join(out, 'desktop-deep-result.png') });

// slider follows the pointer
const stage = await pg.locator('.stage').boundingBox();
const handle = await pg.locator('.handle').boundingBox();
await pg.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2);
await pg.mouse.down();
await pg.mouse.move(stage.x + stage.width * 0.25, handle.y + handle.height / 2, { steps: 8 });
await pg.mouse.up();
const val = +(await pg.getAttribute('.handle', 'aria-valuenow'));
check(Math.abs(val - 25) <= 1, `slider tracks pointer (${val}%)`);

// zoom presets
for (const z of [100, 200, 400]) {
  await pg.click(`[aria-label="Zoom to ${z}%"]`);
  const t = (await pg.textContent('.zoom-val')).trim();
  check(t === `${z}%`, `zoom ${z}% (${t})`);
}
await pg.screenshot({ path: join(out, 'desktop-deep-400.png') });
await pg.click('.zoom >> text=Fit');

// export: the downloaded PNG holds exactly the previewed pixels
await pg.waitForFunction(() => /\d+(\.\d)? (KB|MB|B)/.test(document.querySelectorAll('.facts.two dd')[1]?.textContent || ''), null, { timeout: 30000 });
const [dl] = await Promise.all([pg.waitForEvent('download', { timeout: 30000 }), pg.click('text=Download PNG')]);
const file = await dl.path();
const saved = readFileSync(file);
const png = pngRgb(saved);
const shown = Buffer.from(await pg.evaluate(async () => {
  const r = await fetch(document.querySelector('img.layer.enh').src);
  return [...new Uint8Array(await r.arrayBuffer())];
}));
check(png.w === 240 && createHash('sha256').update(saved).digest('hex') === createHash('sha256').update(shown).digest('hex'),
  `downloaded PNG is byte for byte the previewed image (${png.w} × ${png.h})`);

// crop: drag a box on the original and enhance only that part
await pg.click('.viewbar >> text=Crop');
const cs = await pg.locator('.stage').boundingBox();
await pg.mouse.move(cs.x + cs.width * 0.25, cs.y + cs.height * 0.25);
await pg.mouse.down();
await pg.mouse.move(cs.x + cs.width * 0.75, cs.y + cs.height * 0.75, { steps: 6 });
await pg.mouse.up();
await pg.click('text=Apply crop');
await pg.waitForFunction(() => /This image is 6\d × 4\d pixels/.test(document.querySelector('.summary')?.textContent || ''), null, { timeout: 30000 });
check(true, 'crop applied: ' + (await pg.textContent('.summary')).slice(0, 40));

// invalid upload is rejected with a clear message
await pg.setInputFiles('#file', { name: 'notes.png', mimeType: 'image/png', buffer: Buffer.from('this is not an image at all') });
await pg.waitForSelector('.alert');
check((await pg.textContent('.alert')).includes('not a PNG'), 'corrupt file rejected');

// phone layout in dark mode
const ph = await page({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, colorScheme: 'dark', isMobile: true, hasTouch: true });
await ph.waitForSelector('.summary', { timeout: 30000 });
await ph.waitForTimeout(500);
const overflow = await ph.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check(overflow <= 0, `no horizontal scroll on phone (${overflow}px)`);
await ph.screenshot({ path: join(out, 'phone-dark.png'), fullPage: true });

await browser.close();
server.close();
console.log(failures ? `${failures} failure(s)` : 'all checks passed');
process.exit(failures ? 1 : 0);
