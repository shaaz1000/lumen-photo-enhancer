// Bundles the benchmark harness. Pass the v1.0 source dir to compare against it.
import * as esbuild from 'esbuild';
import { resolve } from 'node:path';
const v1 = resolve(process.argv[2] || 'bench/v1-stub');
const tfjs = {
  name: 'tfjs',
  setup(b) {
    b.onResolve({ filter: /^@tensorflow\/tfjs(-backend-webgl)?\/dist\/index\.js$/ }, () => ({ path: '@tensorflow/tfjs', external: true }));
    b.onResolve({ filter: /^@tensorflow\/tfjs-backend-wasm/ }, () => ({ path: 'wasm-stub', namespace: 'stub' }));
    b.onLoad({ filter: /.*/, namespace: 'stub' }, () => ({ contents: 'export const setWasmPaths = () => {}; export const version_wasm = "none";', loader: 'js' }));
  },
};
await esbuild.build({
  entryPoints: ['bench/run.ts'], bundle: true, platform: 'node', format: 'esm', target: 'node20',
  outfile: 'bench/.build/run.mjs', external: ['onnxruntime-node', '@tensorflow/tfjs'],
  alias: { v1 }, plugins: [tfjs], logLevel: 'error',
  banner: { js: "import { createRequire } from 'module'; const require = createRequire(import.meta.url);" },
});
console.log('bench/.build/run.mjs');
