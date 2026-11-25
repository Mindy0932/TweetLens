const path = require('node:path');
const fs = require('node:fs');
const esbuild = require('esbuild');
const out = path.resolve(process.argv[2] || path.join(__dirname, '../assets'));
(async () => {
  fs.mkdirSync(path.join(out, 'runtime'), { recursive: true });
  await esbuild.build({entryPoints:[path.join(__dirname,'worker.js')],bundle:true,platform:'browser',format:'esm',outfile:path.join(out,'roberta-worker.js'),minify:true});
  const runtime = path.dirname(require.resolve('onnxruntime-web/wasm'));
  for (const name of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
    fs.copyFileSync(path.join(runtime, name), path.join(out, 'runtime', name));
  }
  console.log('Browser worker and runtime built');
})().catch(error => { console.error(error); process.exit(1); });
