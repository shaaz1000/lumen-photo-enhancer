// End-to-end smoke test in headless Chromium.
// Usage: CHROME=/path/to/chrome node tests/e2e/smoke.mjs [image ...]
// Serves dist/ locally, maps the pinned CDN scripts to node_modules copies,
// loads the app, enhances the sample (and any given images), checks output
// dimensions and the comparison slider, and writes screenshots to tests/e2e/out.
import { chromium } from 'playwright-core';
import { createServer } from 'node:http';
import { readFileSync, mkdirSync, existsSync } from 'node:fs';
import { extname, join } from 'node:path';

const root = new URL('../../', import.meta.url).pathname;
const dist = join(root, 'dist');
const out = join(root, 'tests/e2e/out');
mkdirSync(out, { recursive: true });

const types = { '.html': 'text/html', '.bin': 'application/octet-stream', '.json': 'application/json', '.js': 'text/javascript' };
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
  'tf.min.js': join(root, 'node_modules/@tensorflow/tfjs/dist/tf.min.js'),
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

const pg = await page({ viewport: { width: 1360, height: 860 }, colorScheme: 'light' });
await pg.waitForSelector('.summary', { timeout: 30000 });
await pg.waitForFunction(() => document.querySelector('.status')?.textContent?.includes('AI ready'), null, { timeout: 60000 });
check(true, 'AI engine started: ' + (await pg.textContent('.status')));
await pg.screenshot({ path: join(out, 'desktop-loaded.png') });

async function enhanceAndCheck(label) {
  const t0 = Date.now();
  await pg.click('button.btn.primary.wide');
  await pg.waitForSelector('.progress', { timeout: 10000 }).catch(() => {});
  await pg.waitForSelector('.handle', { timeout: 600000 });
  await pg.waitForSelector('.progress', { state: 'detached', timeout: 600000 });
  const dims = await pg.evaluate(() => {
    const c = document.querySelectorAll('canvas.layer');
    return [...c].map((x) => [x.width, x.height]);
  });
  const facts = await pg.textContent('.facts');
  console.log(`  ${label}: ${((Date.now() - t0) / 1000).toFixed(1)}s, canvases ${JSON.stringify(dims)}`);
  return { dims, facts };
}

const s = await enhanceAndCheck('sample');
const want = +(await pg.locator('[aria-label="Upscale factor"] button.on').textContent()).replace('×', '');
check(s.dims.length === 2 && s.dims[1][0] === s.dims[0][0] * want, `sample enhanced to ${want}x`);
await pg.waitForTimeout(1200);
await pg.screenshot({ path: join(out, 'desktop-sample-result.png') });

// slider: drag handle to 25% of the image and read aria-valuenow
const stage = await pg.locator('.stage').boundingBox();
const handle = await pg.locator('.handle').boundingBox();
await pg.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2);
await pg.mouse.down();
await pg.mouse.move(stage.x + stage.width * 0.25, handle.y + handle.height / 2, { steps: 8 });
await pg.mouse.up();
const val = +(await pg.getAttribute('.handle', 'aria-valuenow'));
check(Math.abs(val - 25) <= 1, `slider tracks pointer (${val}%)`);

// pixel zoom and fit
await pg.click('text=1:1');
const zoomTxt = await pg.textContent('.zoom-val');
check(zoomTxt.trim() === '100%', `1:1 shows 100% (${zoomTxt})`);
await pg.screenshot({ path: join(out, 'desktop-sample-1to1.png') });
await pg.click('text=Fit');

// export: size estimate appears
await pg.waitForFunction(() => /\d+(\.\d)? (KB|MB)/.test(document.querySelectorAll('.facts.two dd')[1]?.textContent || ''), null, { timeout: 30000 });
check(true, 'export size measured: ' + (await pg.locator('.facts.two dd').nth(1).textContent()));

// invalid upload is rejected with a clear message
await pg.setInputFiles('#file', { name: 'notes.png', mimeType: 'image/png', buffer: Buffer.from('this is not an image at all') });
await pg.waitForSelector('.alert');
check((await pg.textContent('.alert')).includes('not a PNG'), 'corrupt file rejected');

for (const img of process.argv.slice(2)) {
  const prev = await pg.textContent('.summary').catch(() => '');
  await pg.setInputFiles('#file', img);
  await pg.waitForFunction((p) => { const s = document.querySelector('.summary')?.textContent; return s && s !== p; }, prev, { timeout: 60000 });
  await pg.waitForFunction(() => ![...document.querySelectorAll('.facts dd')].some((d) => d.textContent?.includes('Checking')), null, { timeout: 120000 });
  const summary = await pg.textContent('.summary');
  const facesTxt = await pg.locator('.facts > div').nth(5).textContent();
  console.log(`  ${img}: ${summary} | ${facesTxt}`);
  const r = await enhanceAndCheck(img);
  check(r.dims.length === 2, `${img} enhanced`);
  await pg.waitForTimeout(1200);
  const name = img.split('/').pop().replace(/\W+/g, '_');
  await pg.screenshot({ path: join(out, `desktop-${name}.png`) });
  await pg.click('text=Side by side');
  await pg.click('text=1:1');
  await pg.screenshot({ path: join(out, `desktop-${name}-side.png`) });
  await pg.click('text=Slider');
  await pg.click('text=Fit');
}

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
