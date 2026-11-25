import { AutoTokenizer, env } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web/wasm';

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL('./', self.location.href).href;
ort.env.wasm.wasmPaths = new URL('./runtime/', self.location.href).href;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
let resources;
let queue = Promise.resolve();

function progress(id, stage, fraction) {
    self.postMessage({ id, type: 'progress', stage, fraction });
}
async function readChunk(url, expected, cache, onBytes) {
    const cached = cache && await cache.match(url);
    const response = cached || await fetch(url);
    if (!response.ok) throw Error('Model download failed (' + response.status + ').');
    const bytes = new Uint8Array(expected.bytes);
    let written = 0;
    const reader = response.body.getReader();
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (written + value.length > bytes.length) throw Error('Model size mismatch.');
        bytes.set(value, written);
        written += value.length;
        onBytes(value.length);
    }
    if (written !== bytes.length) throw Error('Model download was incomplete.');
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    const hash = Array.from(new Uint8Array(digest), x => x.toString(16).padStart(2, '0')).join('');
    if (hash !== expected.sha256) {
        if (cache) await cache.delete(url);
        throw Error('Model integrity check failed. Please retry.');
    }
    if (!cached && cache) {
        try { await cache.put(url, new Response(bytes)); } catch {}
    }
    return bytes;
}
async function load(id) {
    progress(id, 'download', 0);
    const response = await fetch(new URL('./roberta/manifest.json', self.location.href));
    if (!response.ok) throw Error('Model manifest is unavailable.');
    const manifest = await response.json();
    let cache = null;
    try { cache = await caches.open('tweetlens-roberta-' + manifest.sha256); } catch {}
    const bytes = new Uint8Array(manifest.bytes);
    let offset = 0, loaded = 0, lastReport = 0;
    for (const chunk of manifest.chunks) {
        const url = new URL('./roberta/' + chunk.file, self.location.href).href;
        const data = await readChunk(url, chunk, cache, count => {
            loaded += count;
            if (performance.now() - lastReport > 100) {
                progress(id, 'download', loaded / manifest.bytes);
                lastReport = performance.now();
            }
        });
        bytes.set(data, offset);
        offset += data.length;
    }
    progress(id, 'initialize', 1);
    const tokenizer = await AutoTokenizer.from_pretrained('roberta', { local_files_only: true });
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'disabled' });
    return { tokenizer, session, manifest };
}
async function predict(message) {
    try {
        resources ||= load(message.id).catch(error => { resources = null; throw error; });
        const { tokenizer, session, manifest } = await resources;
        progress(message.id, 'infer', 1);
        const tokens = await tokenizer(message.text, { truncation: true, max_length: manifest.max_length, padding: false });
        const feeds = {};
        for (const name of session.inputNames) {
            const tensor = tokens[name];
            feeds[name] = new ort.Tensor('int64', tensor.data, tensor.dims);
        }
        const outputs = await session.run(feeds);
        const probs = Array.from(outputs.probabilities.data);
        const label = manifest.classes[probs.indexOf(Math.max(...probs))];
        self.postMessage({ id: message.id, type: 'result', result: { label, classes: manifest.classes, probs } });
    } catch (error) {
        self.postMessage({ id: message.id, type: 'error', error: error.message || String(error) });
    }
}
self.onmessage = event => {
    queue = queue.then(() => predict(event.data));
};
