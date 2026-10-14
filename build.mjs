// Bundles the TypeScript app into one self-contained HTML page.
// React and TensorFlow.js load from pinned CDN builds as globals.
import * as esbuild from 'esbuild';
import { readFileSync, writeFileSync, mkdirSync, copyFileSync, readdirSync } from 'node:fs';

const globals = {
  'react': 'window.React',
  'react-dom/client': 'window.ReactDOM',
  '@tensorflow/tfjs': 'window.tf',
  '@tensorflow/tfjs/dist/index.js': 'window.tf',
  '@tensorflow/tfjs-backend-webgl/dist/index.js': 'window.tf',
  '@tensorflow/tfjs-backend-wasm/dist/index.js': '({ setWasmPaths() {}, version_wasm: "none" })',
};
const globalsPlugin = {
  name: 'globals',
  setup(b) {
    const filter = new RegExp('^(' + Object.keys(globals).map((k) => k.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&')).join('|') + ')$');
    b.onResolve({ filter }, (a) => ({ path: a.path, namespace: 'global' }));
    b.onLoad({ filter: /.*/, namespace: 'global' }, (a) => ({ contents: `module.exports = ${globals[a.path]};`, loader: 'js' }));
  },
};

const r = await esbuild.build({
  entryPoints: ['src/main.tsx'], bundle: true, minify: true, format: 'iife', target: 'es2020',
  write: false, plugins: [globalsPlugin], jsx: 'transform', legalComments: 'none',
  define: { 'process.env.NODE_ENV': '"production"' },
});
const js = r.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
const css = readFileSync('src/ui/styles.css', 'utf8');
const html = readFileSync('src/index.html', 'utf8').replace('/*__CSS__*/', () => css).replace('/*__JS__*/', () => js);
mkdirSync('dist/models', { recursive: true });
writeFileSync('dist/index.html', html);
for (const f of ['general-x4v3', 'general-wdn-x4v3']) copyFileSync(`public/models/${f}.bin`, `dist/models/${f}.wasm`);
const fa = 'node_modules/@vladmandic/face-api/model/';
for (const f of readdirSync('public/models').filter((x) => x.endsWith('.wasm'))) copyFileSync(`public/models/${f}`, `dist/models/${f}`);
for (const m of ['tiny_face_detector_model', 'face_landmark_68_model']) {
  copyFileSync(fa + m + '.bin', `dist/models/${m}.wasm`);
  const manifest = JSON.parse(readFileSync(fa + m + '-weights_manifest.json', 'utf8'));
  for (const g of manifest) g.paths = g.paths.map((x) => x.replace(/\.bin$/, '.wasm'));
  writeFileSync(`dist/models/${m}-weights_manifest.json`, JSON.stringify(manifest));
}
console.log(`dist/index.html ${(html.length / 1024).toFixed(0)} KB`);
