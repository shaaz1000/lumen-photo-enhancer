// Loads the built page inside a sandboxed iframe with a strict CSP (no eval,
// same-origin fetch only), as close as we can get to the artifact viewer, and
// runs a fast enhancement. Prints every console error and CSP violation.
import { chromium } from 'playwright-core';
import { createServer } from 'node:http';
import { readFileSync, existsSync } from 'node:fs';
import { extname, join } from 'node:path';
const root = new URL('../../', import.meta.url).pathname, dist = join(root, 'dist');
const CSP = "default-src 'none'; script-src 'unsafe-inline' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; style-src 'unsafe-inline' https://fonts.googleapis.com; img-src 'self' data: blob:; connect-src 'self'; font-src https://fonts.gstatic.com";
const srv = createServer((q, r) => {
  let p = q.url.split('?')[0];
  if (p === '/') { r.writeHead(200, { 'content-type': 'text/html' }); return r.end('<!doctype html><body style="margin:0"><iframe id=f src="/app/" sandbox="allow-scripts allow-same-origin allow-downloads allow-forms" style="border:0;width:100vw;height:100vh"></iframe>'); }
  p = p.replace(/^\/app/, '') || '/';
  if (p === '/') p = '/index.html';
  const f = join(dist, p);
  if (!existsSync(f)) { r.writeHead(404); return r.end(); }
  let b = readFileSync(f);
  if (p === '/index.html') b = Buffer.from('<!doctype html><html><head><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>' + b + '</body></html>');
  r.writeHead(200, { 'content-type': extname(f) === '.html' ? 'text/html' : extname(f) === '.json' ? 'application/json' : 'application/wasm', 'content-security-policy': CSP });
  r.end(b);
}).listen(0);
const v = { 'react.production.min.js': root + 'node_modules/react/umd/react.production.min.js', 'react-dom.production.min.js': root + 'node_modules/react-dom/umd/react-dom.production.min.js', 'tf.es2017.min.js': root + 'node_modules/@tensorflow/tfjs/dist/tf.es2017.min.js' };
const b = await chromium.launch({ executablePath: process.env.CHROME, args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const pg = await b.newPage({ viewport: { width: 1360, height: 860 } });
pg.on('console', (m) => { if (['error', 'warning'].includes(m.type()) && !/GL Driver|GPU stall/.test(m.text())) console.log('console:', m.text().slice(0, 300)); });
pg.on('pageerror', (e) => console.log('pageerror:', e.message));
await pg.route(/^https:\/\/(cdnjs|cdn\.jsdelivr)/, (r) => { const k = Object.keys(v).find((k) => r.request().url().endsWith(k)); return r.fulfill({ body: readFileSync(v[k]), contentType: 'text/javascript' }); });
await pg.goto(`http://127.0.0.1:${srv.address().port}/`);
const fr = pg.frameLocator('#f');
await fr.locator('.summary').waitFor({ timeout: 60000 });
await pg.waitForTimeout(3000);
console.log('status:', await fr.locator('.status').textContent());
await fr.locator('#file').setInputFiles(join(root, 'tests/e2e/small.png'));
await fr.locator('text=This image is 120').waitFor({ timeout: 60000 });
await fr.locator('[aria-label="Upscale factor"] >> text=2×').click();
await fr.locator('[aria-label="Reconstruction strength"] >> text=Faithful').click();
await fr.locator('.inspector button.btn.primary.wide').click();
await fr.locator('.progress').waitFor({ timeout: 10000 }).catch(() => console.log('no progress overlay'));
await fr.locator('.progress').waitFor({ state: 'detached', timeout: 600000 });
const err = await fr.locator('.alert[role=alert]').textContent().catch(() => null);
const ok = await fr.locator('img.layer.enh').evaluate((i) => i.decode().then(() => [i.naturalWidth, i.naturalHeight])).catch((e) => 'no enhanced image: ' + e.message);
console.log('result:', err ? 'ERROR ' + err : JSON.stringify(ok));
await b.close(); srv.close();
