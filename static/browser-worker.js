/* In-browser backend (static site): the same inputs the server builds (src/spoiler/data.py), run with
   onnxruntime-web.  in:  {type:"config", config} | {type:"load", key} | {type:"predict", id, sentences, genres, models}
                     out: {type:"progress", key, loaded, total} | {type:"ready", key} | {type:"result", id, p, ms}
                          | {type:"error", id?, key?, message}                                                    */
"use strict";
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
if (typeof importScripts === "function") {
  importScripts(ORT + "ort.wasm.min.js", "tokenizer.js" + self.location.search);   // ?v=<build>: no stale copy
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
}

let CFG = null;           // {models: {key: {model, spec}}, vocab, iif, cache}
const S = { wp: null, iif: null, spec: {}, sess: {}, loading: {} };

self.onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === "config") {
      CFG = m.config;
      try {   // models cached by an older build of the site are dead weight (~240 MB): drop them
        for (const k of await caches.keys()) if (k.startsWith("spoiler-models-") && k !== CFG.cache) await caches.delete(k);
      } catch (_) { /* no Cache API */ }
      return;
    }
    if (m.type === "load") { await load(m.key); return; }
    if (m.type === "predict") {
      for (const k of m.models) await load(k);
      const t0 = performance.now();
      const p = {};
      for (const k of m.models) p[k] = await predict(k, m.sentences, m.genres);
      self.postMessage({ type: "result", id: m.id, p, ms: Math.round(performance.now() - t0) });
    }
  } catch (e) {
    self.postMessage({ type: "error", id: m.id, key: m.key, message: String((e && e.message) || e), stack: e && e.stack });
  }
};

async function cached(url, onProgress) {
  // the Cache API keeps the ~120 MB model between visits; plain download where it is unavailable
  let cache = null;
  try { cache = await caches.open(CFG.cache); } catch (_) { /* private mode etc. */ }
  if (cache) {
    const hit = await cache.match(url);
    if (hit) { const b = await hit.arrayBuffer(); onProgress(b.byteLength, b.byteLength); return b; }
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`모델 다운로드 실패 (${r.status})`);
  const total = +r.headers.get("content-length") || 0;
  const reader = r.body.getReader();
  const parts = [];
  let loaded = 0, last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); loaded += value.length;
    if (loaded - last > 1 << 20) { last = loaded; onProgress(loaded, total); }
  }
  onProgress(loaded, loaded);
  const buf = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  if (cache) { try { await cache.put(url, new Response(buf.slice())); } catch (_) { /* quota */ } }
  return buf;
}

function load(key) {
  if (!S.loading[key]) {
    S.loading[key] = (async () => {
      const mc = CFG.models[key];
      S.spec[key] = await (await fetch(mc.spec)).json();
      if (!S.wp) S.wp = new WordPiece(await (await fetch(CFG.vocab)).text());
      if (S.spec[key].dfiif && !S.iif) S.iif = await (await fetch(CFG.iif)).json();
      const buf = await cached(mc.model, (loaded, total) => self.postMessage({ type: "progress", key, loaded, total }));
      self.postMessage({ type: "progress", key, loaded: buf.byteLength, total: buf.byteLength, phase: "init" });
      S.sess[key] = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      self.postMessage({ type: "ready", key });
    })();
    S.loading[key].catch(() => { delete S.loading[key]; });
  }
  return S.loading[key];
}

/* ---------------------------------------------------------------- inputs (src/spoiler/data.py) */
function waterCap(lengths, budget) {
  const sum = (cap) => lengths.reduce((a, l) => a + Math.min(l, cap), 0);
  const mx = lengths.length ? Math.max(...lengths) : 0;
  if (sum(Infinity) <= budget) return mx;
  let lo = 1, hi = mx;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (sum(mid) <= budget) lo = mid; else hi = mid - 1; }
  return lo;
}

function arraySplit(n, k) {             // numpy.array_split(arange(n), k)
  const out = [], base = Math.floor(n / k), extra = n % k;
  let s = 0;
  for (let i = 0; i < k; i++) {
    const len = base + (i < extra ? 1 : 0);
    out.push(Array.from({ length: len }, (_, j) => s + j));
    s += len;
  }
  return out;
}

function buildChunks(sentIds, genreIds, spec) {
  const { context, max_len: maxLen, cls_id: cls, sep_id: sep, pool } = spec;
  const groups = arraySplit(sentIds.length, Math.max(1, Math.ceil(sentIds.length / context)));
  const chunks = [], positions = [];
  for (const g of groups) {
    const gIds = genreIds.slice(0, Math.max(0, Math.floor(maxLen / 4)));
    const tail = gIds.length ? [...gIds, sep] : [];
    const budget = maxLen - 1 - g.length - tail.length;
    const cap = waterCap(g.map((i) => sentIds[i].length), budget);
    const ids = [cls];
    for (const i of g) {
      if (cap) ids.push(...sentIds[i].slice(0, cap));
      ids.push(sep);
      positions.push([chunks.length, pool === "cls" ? 0 : ids.length - 1]);
    }
    ids.push(...tail);
    chunks.push(ids);
  }
  return { chunks, positions };
}

function genreText(genres, spec) {
  const names = [];
  for (const g of genres) { const k = spec.genre_ko[g] || g; if (!names.includes(k)) names.push(k); }
  return names.length ? "장르: " + names.join(", ") : "";
}

/* DF-IIF for one review (DF = 1 for every word, as in the server's live mode). The server finds nouns with Kiwi;
   here each word is cut to its longest prefix found in the IIF vocabulary (webdemo/README.md has the measured gap). */
const WORD = /[가-힣]+|[A-Za-z]+|\d+/g;
function words(s, iif) {
  const out = new Set();
  for (let w of s.match(WORD) || []) {
    w = w.toLowerCase();
    if (/^[a-z]/.test(w)) { out.add(w); continue; }
    if (/^\d/.test(w)) continue;
    const cs = [...w];
    for (let L = cs.length; L > 0; L--) {
      const pre = cs.slice(0, L).join("");
      if (Object.prototype.hasOwnProperty.call(iif, pre) && (L > 1 || cs.length === 1)) { out.add(pre); break; }
    }
  }
  return out;
}
function dfiif(sentences, table) {
  const { iif, iif_unseen: unseen } = table;
  return sentences.map((s) => {
    const ws = [...words(s, iif)];
    if (!ws.length) return [0, 0, 0, 0];
    const v = ws.map((w) => (Object.prototype.hasOwnProperty.call(iif, w) ? iif[w] : unseen)).sort((a, b) => b - a);
    const top = v.slice(0, 3);
    return [v[0], v.reduce((a, b) => a + b, 0) / v.length, top.reduce((a, b) => a + b, 0) / top.length, Math.log1p(v.length)];
  });
}

function inputs(spec, wp, iifTable, sentences, genres) {
  const n = sentences.length;
  const sentIds = sentences.map((s) => wp.encode(s));
  const g = spec.genre ? genreText(genres, spec) : "";
  const { chunks, positions } = buildChunks(sentIds, g ? wp.encode(g) : [], spec);
  const f = spec.dfiif ? dfiif(sentences, iifTable) : null;
  const extra = sentences.map((_, j) => {
    const e = f ? f[j].map((x, k) => (x - spec.feat_mean[k]) / spec.feat_std[k]) : [];
    if (spec.position) e.push(n > 1 ? j / (n - 1) : 0);
    return e;
  });
  return { chunks, positions, extra };
}

async function predict(key, sentences, genres) {
  const spec = S.spec[key], sess = S.sess[key];
  const n = sentences.length;
  const { chunks, positions, extra } = inputs(spec, S.wp, S.iif, sentences, genres);
  const L = Math.max(...chunks.map((c) => c.length));
  const ids = new BigInt64Array(chunks.length * L), mask = new BigInt64Array(chunks.length * L);
  chunks.forEach((c, i) => c.forEach((t, j) => { ids[i * L + j] = BigInt(t); mask[i * L + j] = 1n; }));
  const feed = {
    input_ids: new ort.Tensor("int64", ids, [chunks.length, L]),
    attention_mask: new ort.Tensor("int64", mask, [chunks.length, L]),
    sent_rows: new ort.Tensor("int64", BigInt64Array.from(positions, (p) => BigInt(p[0])), [n]),
    sent_cols: new ort.Tensor("int64", BigInt64Array.from(positions, (p) => BigInt(p[1])), [n]),
  };
  if (sess.inputNames.includes("extra")) {
    const d = extra[0].length || 1;
    feed.extra = new ort.Tensor("float32", Float32Array.from(extra.flatMap((r) => (r.length ? r : [0]))), [n, d]);
  }
  const out = await sess.run(feed);
  return Array.from(out[sess.outputNames[0]].data, (x) => Math.round(x * 1e4) / 1e4);
}

if (typeof module !== "undefined") module.exports = { buildChunks, waterCap, arraySplit, genreText, dfiif, words, inputs };
