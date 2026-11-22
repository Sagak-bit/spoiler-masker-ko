/* 스포일러 가리개 — front end (no build step, no external requests).
   The server returns P(spoiler) per sentence; every masking decision is made here so the strength
   controls react instantly. Thresholds come from /api/meta (chosen on validation, as in the report). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) el.append(c);
  return el;
};
const pct = (x, d = 0) => `${(100 * x).toFixed(d)}%`;
const logit = (p) => Math.log(p / (1 - p));
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

/* consecutive runs of equal decisions: [true,true,false,true] -> [[0,1],[2],[3]] */
function runs(flags) {
  const out = [];
  flags.forEach((f, i) => { if (i && f === flags[i - 1]) out[out.length - 1].push(i); else out.push([i]); });
  return out;
}

/* one masked block for a run of consecutive spoiler sentences; parts: [{text, i, cls}] */
function maskBlock(parts, open, onToggle) {
  const n = parts.length;
  const b = h("button", { type: "button", class: `mask${open ? " revealed" : ""}`,
    "aria-expanded": String(open),
    "aria-label": open ? `펼친 스포일러 ${n}문장. 눌러서 다시 가리기` : `가려진 스포일러 ${n}문장. 눌러서 보기`, onclick: onToggle },
    h("span", { class: "mask-label", text: open ? "다시 가리기" : (n > 1 ? `스포일러 · ${n}문장` : "스포일러") }));
  const txt = h("span", { class: "mask-text" });
  parts.forEach((p, k) => { if (k) txt.append(" "); txt.append(h("span", { class: p.cls || null, "data-i": String(p.i), text: p.text })); });
  b.append(txt);
  return b;
}

const S = {
  meta: null, feed: null, ready: false,
  run: null,              // current result in "직접 가려 보기": {sentences, p:{m5,m1}, tags, en, source, ms}
  strength: 100,          // 0 = 넉넉(val recall 0.8), 100 = 엄격(val best F1); in between: logit interpolation
  revealed: new Set(), exampleKey: null,
  // method: "none" (show everything) | "old" (fold any author-tagged review) | "model" (mask sentences)
  site: { book: 0, method: "model", mode: "strict", tags: false, revealed: new Set(), openOld: new Set(), openEn: new Set() },
};

/* threshold of a model at a strength in [0, 100] */
function threshold(model, strength) {
  const m = S.meta.models[model];
  if (strength >= 100) return m.strict;
  if (strength <= 0) return m.lenient;
  return sigmoid(logit(m.lenient) + (logit(m.strict) - logit(m.lenient)) * strength / 100);
}

/* ================================================================== backend
   Server mode (python -m webdemo.server): /api/*.  Browser mode (static site, <html data-backend="browser">):
   data/*.json + onnxruntime-web in a Web Worker (browser-worker.js); the model is downloaded on first use. */
const BROWSER = document.documentElement.dataset.backend === "browser";
const SPLIT = /(?<=[.!?…~])\s+|\n+/;          // src/spoiler/mask.py split_sentences
const API = BROWSER ? browserApi() : {
  meta: () => fetch("api/meta").then(okJson),
  feed: () => fetch("api/feed").then(okJson),
  health: () => fetch("api/health").then(okJson),
  predict: (body) => fetch("api/predict", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) }).then(okJson),
  where: "이 컴퓨터에서",
};

function browserApi() {
  const cfg = window.SPOILER_CONFIG;
  let worker = null, seq = 0;
  const pending = new Map(), loaded = new Set(), progress = {};
  const start = () => {
    if (worker) return worker;
    worker = new Worker("static/browser-worker.js?v=" + cfg.version);
    // worker-relative paths would point into static/: hand it absolute URLs
    const abs = (u) => { const x = new URL(u, location.href); if (x.origin === location.origin) x.searchParams.set("v", cfg.version); return x.href; };
    const models = Object.fromEntries(Object.entries(cfg.models).map(([k, v]) => [k, { ...v, model: abs(v.model), spec: abs(v.spec) }]));
    worker.postMessage({ type: "config", config: { ...cfg, vocab: abs(cfg.vocab), iif: abs(cfg.iif), models } });
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === "progress") { progress[m.key] = m; onProgress(); }
      else if (m.type === "ready") { loaded.add(m.key); delete progress[m.key]; onProgress(); }
      else if (m.type === "result" || m.type === "error") {
        if (m.type === "error") console.error("[worker]", m.message, m.stack || "");
        if (m.type === "error" && m.key) { delete progress[m.key]; onProgress(m.message); }
        const cb = pending.get(m.id);
        if (cb) { pending.delete(m.id); m.type === "result" ? cb.ok(m) : cb.fail(new Error(m.message)); }
      }
    };
    worker.onerror = (e) => { for (const cb of pending.values()) cb.fail(new Error(e.message || "worker error")); pending.clear(); };
    return worker;
  };
  // status line: download progress of whichever model is coming in
  const onProgress = (err) => {
    const st = $("#status"), txt = $(".status-text", st);
    if (err) { st.className = "status error"; txt.textContent = "모델 오류"; st.title = err; return; }
    const cur = Object.values(progress)[0];
    if (cur) {
      st.className = "status loading";
      txt.textContent = cur.phase === "init" ? `${cur.key.toUpperCase()} 준비 중…`
        : `${cur.key.toUpperCase()} 받는 중 ${cur.total ? Math.round(100 * cur.loaded / cur.total) + "%" : (cur.loaded / 1e6).toFixed(0) + " MB"}`;
      const btn = $("#run");
      if (btn.disabled && S.busy) btn.lastChild.textContent = txt.textContent;
      return;
    }
    st.className = loaded.size ? "status ready" : "status idle";
    txt.textContent = loaded.size ? `${[...loaded].map((k) => k.toUpperCase()).join("·")} 준비됨 (브라우저)` : "브라우저에서 실행";
    st.title = loaded.size ? "모델이 이 브라우저 안에서 돌아갑니다. 입력은 어디로도 보내지 않습니다."
      : `처음 [스포일러 가리기]를 누를 때 모델(약 ${cfg.models.m5.mb} MB)을 한 번 받습니다. 다음부터는 브라우저에 저장된 것을 씁니다.`;
  };
  return {
    meta: () => fetch("data/meta.json").then(okJson),
    feed: () => fetch("data/feed.json").then(okJson),
    health: async () => { onProgress(); return { ready: true, browser: true }; },
    predict: async (body) => {
      const lim = S.meta.limits;
      const sents = body.sentences ? body.sentences.map((x) => x.trim()).filter(Boolean)
        : body.text.split(SPLIT).map((x) => (x || "").trim()).filter(Boolean);
      if (!sents.length) throw new Error("리뷰를 입력해 주세요.");
      if (sents.length > lim.sentences || sents.join("").length > lim.chars) throw new Error(`리뷰가 너무 깁니다(${lim.chars.toLocaleString()}자, ${lim.sentences}문장까지).`);
      const genres = (body.genres || []).filter((g) => S.meta.genres.some((x) => x.key === g)).slice(0, 5);
      const id = ++seq;
      const r = await new Promise((ok, fail) => {
        pending.set(id, { ok, fail });
        start().postMessage({ type: "predict", id, sentences: sents, genres, models: body.models || ["m5"] });
      });
      return { sentences: sents, genres, p: r.p, ms: r.ms };
    },
    where: "이 브라우저에서",
    mb: (k) => cfg.models[k].mb,
    isLoaded: (k) => loaded.has(k),
  };
}

/* ================================================================== boot */
async function boot() {
  setupTabs();
  pollHealth();
  try {
    const [meta, feed] = await Promise.all([API.meta(), API.feed()]);
    S.meta = meta; S.feed = feed;
  } catch (e) {
    showError(`초기 데이터를 불러오지 못했습니다: ${e.message}`);
    return;
  }
  $("#where").textContent = API.where;
  if (BROWSER) for (const k of Object.keys(S.meta.models)) if (!window.SPOILER_CONFIG.models[k]) delete S.meta.models[k];
  buildInputPanel();
  buildSite();
  pickExample("demo");
}
async function okJson(r) {
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* not json */ }
    throw new Error(msg);
  }
  return r.json();
}

async function pollHealth() {
  const st = $("#status"), txt = $(".status-text", st);
  try {
    const r = await API.health();
    if (r.browser) { S.ready = true; $("#run").disabled = false; return; }
    if (r.error) { st.className = "status error"; txt.textContent = "모델 오류"; st.title = r.error; return; }
    if (r.ready) {
      S.ready = true; st.className = "status ready";
      txt.textContent = `${r.models.map((m) => m.toUpperCase()).join("·")} 준비됨`;
      st.title = "이 컴퓨터의 CPU에서 추론합니다";
      $("#run").disabled = false;
      return;
    }
    st.className = "status loading"; txt.textContent = "모델 불러오는 중…";
  } catch (_) {
    st.className = "status error"; txt.textContent = "서버 연결 안 됨";
  }
  setTimeout(pollHealth, 1500);
}

function setupTabs() {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const show = (id) => {
    for (const t of tabs) {
      const on = t.id === id;
      t.setAttribute("aria-selected", on);
      t.tabIndex = on ? 0 : -1;
      $("#" + t.getAttribute("aria-controls")).hidden = !on;
    }
    history.replaceState(null, "", id === "tab-site" ? "#site" : "#try");
  };
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => show(t.id));
    t.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const n = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      show(n.id); n.focus();
    });
  });
  show(location.hash === "#site" ? "tab-site" : "tab-try");
}

/* ================================================================== 직접 가려 보기 */
function examples() {
  const out = [{ key: "demo", label: "데모 리뷰", sentences: S.feed.demo.sentences, genres: S.feed.demo.genres, p: S.feed.demo.p,
    note: "직접 쓴 데모 리뷰(데이터셋 문장 아님)" }];
  // the first tagged review of the first three books on the shelf (books were picked without looking at the model)
  for (const b of S.feed.books.slice(0, 3)) {
    const r = b.reviews.find((x) => x.tags.some(Boolean));
    out.push({ key: r.id, label: `${b.name} 리뷰`, sentences: r.sentences, genres: r.genres, p: r.p, tags: r.tags, en: r.en,
      note: "Goodreads test 리뷰 번역본 · 미리 계산한 결과" });
  }
  return out;
}

function buildInputPanel() {
  const ex = $("#examples");
  for (const e of examples()) {
    ex.append(h("button", { type: "button", class: "example", "data-key": e.key, "aria-pressed": "false",
      onclick: () => pickExample(e.key) }, e.tags ? h("span", { class: "tagdot", title: "작성자 태그가 있는 실제 리뷰" }) : null, e.label));
  }
  const gl = $("#genre-list");
  for (const g of S.meta.genres) {
    gl.append(h("button", { type: "button", class: "chip", "data-key": g.key, "aria-pressed": "false",
      onclick: (ev) => { const b = ev.currentTarget; b.setAttribute("aria-pressed", b.getAttribute("aria-pressed") !== "true"); markDirty(); } }, g.label));
  }
  const ta = $("#review");
  ta.addEventListener("input", () => { updateCounter(); markDirty(); });
  ta.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); runLive(); } });
  $("#run").addEventListener("click", runLive);
  $("#run").disabled = !S.ready;
  $("#cmp-m1").addEventListener("change", async () => {
    // a live result computed without M1: fetch M1 for the same sentences before drawing the comparison
    if ($("#cmp-m1").checked && S.run && S.run.live && !S.run.p.m1 && S.ready) {
      $("#result").classList.add("busy");
      try {
        const r = await API.predict({ sentences: S.run.sentences, genres: S.run.genres, models: ["m1"] });
        S.run.p.m1 = r.p.m1;
      } catch (e) { showError(e.message); }
      $("#result").classList.remove("busy");
    }
    renderResult();
  });
  $("#show-tags").addEventListener("change", renderResult);
  $("#strength-try").append(strengthControl());
  if (!S.meta.models.m1) $("#cmp-m1").closest("label").hidden = true;
}

function updateCounter() {
  const n = $("#review").value.length, lim = S.meta.limits.chars;
  const c = $("#counter");
  c.textContent = `${n.toLocaleString()} / ${lim.toLocaleString()}자`;
  c.classList.toggle("over", n > lim);
}

function selectedGenres() {
  return [...document.querySelectorAll("#genre-list .chip[aria-pressed=true]")].map((b) => b.dataset.key);
}

function setGenres(keys) {
  for (const b of document.querySelectorAll("#genre-list .chip")) b.setAttribute("aria-pressed", keys.includes(b.dataset.key));
}

function markDirty() {
  // the shown result no longer matches the input; it stays visible (dimmed, with a notice) until the next run
  if (S.run && !S.stale) { S.stale = true; renderResult(); }
  if (S.exampleKey) {
    S.exampleKey = null;
    for (const b of document.querySelectorAll(".example")) b.setAttribute("aria-pressed", "false");
  }
}

function pickExample(key) {
  const e = examples().find((x) => x.key === key);
  if (!e) return;
  $("#review").value = e.sentences.join(" ");
  setGenres(e.genres);
  updateCounter();
  S.exampleKey = key;
  for (const b of document.querySelectorAll(".example")) b.setAttribute("aria-pressed", b.dataset.key === key);
  hideError();
  S.revealed = new Set();
  S.stale = false;
  S.run = { sentences: e.sentences, p: e.p, tags: e.tags || null, en: e.en || null, source: e.note, ms: null };
  renderResult();
}

async function runLive() {
  const text = $("#review").value.trim();
  hideError();
  if (!text) { showError("리뷰를 입력해 주세요."); $("#review").focus(); return; }
  if (text.length > S.meta.limits.chars) { showError(`리뷰가 너무 깁니다(${S.meta.limits.chars.toLocaleString()}자까지).`); return; }
  if (!S.ready) { showError("모델을 아직 불러오는 중입니다. 오른쪽 위 표시가 초록색이 되면 다시 눌러 주세요."); return; }
  // an untouched example keeps its precomputed result (same input as the test evaluation)
  const btn = $("#run");
  btn.disabled = true;
  const first = BROWSER && !API.isLoaded("m5");
  btn.replaceChildren(h("span", { class: "spin", "aria-hidden": "true" }), first ? "모델 받는 중…" : "가리는 중…");
  S.busy = true;
  $("#result").classList.add("busy");
  try {
    const r = await API.predict({ text, genres: selectedGenres(), models: $("#cmp-m1").checked ? ["m5", "m1"] : ["m5"] });
    S.revealed = new Set();
    S.stale = false;
    S.run = { sentences: r.sentences, genres: r.genres, p: r.p, tags: null, en: null, source: `${API.where} 방금 추론`, ms: r.ms, live: true };
    renderResult();
  } catch (e) {
    showError(e.message);
  } finally {
    $("#result").classList.remove("busy");
    S.busy = false;
    btn.disabled = false;
    btn.textContent = "스포일러 가리기";
  }
}

function showError(msg) { const e = $("#error"); e.textContent = msg; e.hidden = false; }
function hideError() { $("#error").hidden = true; }

function strengthControl() {
  const wrap = h("div", { class: "strength-inner", style: "display:contents" });
  const seg = h("div", { class: "seg", role: "radiogroup", "aria-label": "가림 강도" });
  // the slider runs left = 엄격 (fewer masks) to right = 넉넉 (more masks), the same order as the buttons
  const range = h("input", { type: "range", min: "0", max: "100", step: "1", value: String(100 - S.strength), "aria-label": "가림 강도 세부 조절" });
  const note = h("div", { class: "strength-note" });
  const setS = (v) => { S.strength = v; range.value = String(100 - v); renderResult(); };
  for (const [label, v] of [["엄격", 100], ["넉넉", 0]]) {
    seg.append(h("button", { type: "button", role: "radio", "data-v": String(v), "aria-checked": String(S.strength === v), onclick: () => setS(v) }, label));
  }
  range.addEventListener("input", () => setS(100 - +range.value));
  wrap.append(seg, h("div", { class: "range-wrap" }, range,
    h("div", { class: "range-scale" }, h("span", { text: "엄격: 확실한 것만 가림" }), h("span", { text: "넉넉: 스포일러를 더 많이 가림" }))), note);
  wrap.update = () => {
    for (const b of seg.children) b.setAttribute("aria-checked", String(+b.dataset.v === S.strength));
    const m5 = threshold("m5", S.strength);
    const parts = [h("span", {}, "M5 임계값 ", h("b", { text: m5.toFixed(3) }))];
    if (S.strength === 100) parts.push(" · val에서 F1이 가장 높은 값 (test에서 스포일러 문장의 ", h("b", { text: pct(S.meta.models.m5.test_recall_strict) }), "를 가림)");
    else if (S.strength === 0) parts.push(" · val 재현율 0.8이 되는 값 (test에서 비스포일러 문장의 ", h("b", { text: pct(S.meta.models.m5.test_overmask_lenient, 1) }), "도 함께 가림)");
    else parts.push(" · 두 값 사이");
    note.replaceChildren(...parts);
  };
  S.updateStrength = wrap.update;
  return wrap;
}

function decisions(model) {
  const t = threshold(model, S.strength);
  return S.run.p[model].map((x) => x >= t);
}

function renderResult() {
  if (!S.run || !S.meta) return;
  $("#empty").hidden = true;
  $("#result-body").hidden = false;
  S.updateStrength && S.updateStrength();
  const { sentences, tags } = S.run;
  const m5 = decisions("m5");
  const hasM1 = !!S.run.p.m1;
  const cmp = $("#cmp-m1").checked && hasM1;
  const m1 = hasM1 ? decisions("m1") : null;
  const showTags = !!tags && $("#show-tags").checked;
  $("#tags-wrap").hidden = !tags;

  // summary
  const k = m5.filter(Boolean).length;
  const sum = $("#summary");
  if (k) sum.replaceChildren(`${sentences.length}문장 중 `, h("b", { text: `${k}문장` }), "을 가렸어요");
  else sum.replaceChildren(`${sentences.length}문장 모두 그대로 보여요 `, h("span", { class: "hint", text: "(가린 문장 없음)" }));
  const meta = [S.run.source];
  if (S.run.ms !== null) meta.push(S.run.ms < 100 ? "0.1초 미만" : `${(S.run.ms / 1000).toFixed(1)}초`);
  if (cmp) meta.push(`M1이었다면 ${m1.filter(Boolean).length}문장`);
  $("#meta-line").textContent = meta.join(" · ");

  // notices: the input changed since this result / the text is mostly not Korean
  const notice = $("#notice");
  const letters = sentences.join("").match(/[A-Za-zÀ-ɏ가-힣ㄱ-ㆎ]/g) || [];
  const hangul = letters.filter((c) => /[가-힣ㄱ-ㆎ]/.test(c)).length;
  if (S.stale) notice.textContent = "입력을 바꿨어요. 아래 결과는 바꾸기 전 글에 대한 것이라 [스포일러 가리기]를 다시 눌러야 해요.";
  else if (letters.length >= 10 && hangul / letters.length < 0.3) notice.textContent = "한국어가 아닌 글로 보여요. 이 모델은 한국어(번역) 리뷰로만 학습해서 다른 언어에서는 거의 가리지 못해요.";
  else notice.textContent = "";
  notice.hidden = !notice.textContent;
  $("#result-body").classList.toggle("stale", !!S.stale);

  // reader view
  const rv = $("#reader-view");
  rv.replaceChildren();
  for (const g of runs(m5)) {
    if (g[0]) rv.append(" ");
    if (m5[g[0]]) {
      const open = S.revealed.has(g[0]);
      rv.append(maskBlock(g.map((i) => ({ text: sentences[i], i })), open, () => {
        open ? S.revealed.delete(g[0]) : S.revealed.add(g[0]); renderResult();
      }));
    } else {
      g.forEach((i, k) => { if (k) rv.append(" "); rv.append(h("span", { class: "sent", "data-i": String(i), text: sentences[i] })); });
    }
  }

  // per-sentence table
  const tb = $("#sent-table");
  tb.replaceChildren();
  let hit = 0, miss = 0, over = 0;
  sentences.forEach((s, i) => {
    const flags = [];
    if (showTags) {
      if (tags[i] && m5[i]) { hit++; flags.push(h("span", { class: "flag hit", text: "맞게 가림" })); }
      else if (tags[i]) { miss++; flags.push(h("span", { class: "flag miss", text: "놓침" })); }
      else if (m5[i]) { over++; flags.push(h("span", { class: "flag over", text: "과잉" })); }
      if (tags[i]) flags.unshift(h("span", { class: "flag tag", text: "작성자 태그" }));
    }
    const bars = [bar("m5", "M5", S.run.p.m5[i], m5[i])];
    if (cmp) bars.push(bar("m1", "M1", S.run.p.m1[i], m1[i]));
    const row = h("li", { class: "row" },
      h("span", { class: "idx", text: String(i + 1) }),
      h("div", { class: "txt" }, s, flags.length ? h("span", { class: "flags" }, flags) : null),
      h("div", { class: "bars" }, bars));
    row.addEventListener("mouseenter", () => flash(i, true));
    row.addEventListener("mouseleave", () => flash(i, false));
    tb.append(row);
  });

  // legend
  const lg = $("#legend");
  const items = [h("span", {}, h("i", { style: "background:var(--blue)" }), "M5 확률(임계값 이상 = 가림)"),
    h("span", {}, h("i", { class: "thr-i" }), "임계값")];
  if (cmp) items.splice(1, 0, h("span", {}, h("i", { style: "background:var(--m1)" }), "M1 확률(문장 하나만 봄)"));
  if (showTags) items.push(h("span", {}, `작성자 태그 ${tags.filter(Boolean).length}문장: 맞게 가림 ${hit} · 놓침 ${miss} · 과잉 ${over}`));
  lg.replaceChildren(...items);
}

function bar(model, who, p, on) {
  const t = threshold(model, S.strength);
  return h("div", { class: `bar ${model}${on ? " on" : ""}`, title: `${who} P(스포일러) = ${p.toFixed(3)}, 임계값 ${t.toFixed(3)}` },
    h("span", { class: "who", text: who }),
    h("span", { class: "track" }, h("span", { class: "fill", style: `width:${(100 * p).toFixed(1)}%` }),
      h("span", { class: "thr", style: `left:calc(${(100 * t).toFixed(2)}% - 1px)` })),
    h("span", { class: "num", text: p.toFixed(3) }));
}

function flash(i, on) {
  const el = document.querySelector(`#reader-view [data-i="${i}"]`);
  if (el) el.classList.toggle("flash", on);
}

/* ================================================================== 리뷰 사이트 목업 */
const COVER = {
  "fantasy, paranormal": ["#3b2a6b", "#6b4aa8"], "mystery, thriller, crime": ["#14213d", "#34507a"],
  "romance": ["#8c2f4d", "#c45a7a"], "young-adult": ["#0f5f63", "#2f8f8c"], "fiction": ["#4c5a2a", "#7d8b4a"],
  "history, historical fiction, biography": ["#6b3d1f", "#a26a3c"], "non-fiction": ["#2f3d4c", "#5b6e82"],
  "children": ["#a0522d", "#d68a4c"], "comics, graphic": ["#7a1f1f", "#b54a3a"], "poetry": ["#4a3b52", "#7a6386"],
};
const AVATAR = ["#b5452f", "#3d6b8a", "#6a7d3a", "#8a5a9e", "#b07a1e", "#2f7a6b"];
const coverBg = (g) => { const [a, b] = COVER[g[0]] || ["#444", "#777"]; return `linear-gradient(160deg, ${b}, ${a})`; };

function buildSite() {
  const shelf = $("#shelf");
  S.feed.books.forEach((b, i) => {
    shelf.append(h("button", { type: "button", class: "shelf-item", "data-i": String(i), onclick: () => openBook(i) },
      h("span", { class: "mini cover", style: `background:${coverBg(b.genres)}`, "aria-hidden": "true" }),
      h("span", {}, h("div", { class: "nm", text: b.name }), h("div", { class: "gn", text: b.genres_ko.slice(0, 2).join(" · ") }))));
  });
  const tg = $("#site-tags");
  tg.addEventListener("change", () => { S.site.tags = tg.checked; renderBook(); });
  for (const b of document.querySelectorAll("#site-method button")) {
    b.addEventListener("click", () => {
      S.site.method = b.dataset.method;
      for (const x of document.querySelectorAll("#site-method button")) x.setAttribute("aria-checked", String(x === b));
      // the strength only matters when the model masks sentences
      const wrap = $("#site-mode-wrap");
      wrap.classList.toggle("disabled", S.site.method !== "model");
      for (const x of wrap.querySelectorAll("button")) x.disabled = S.site.method !== "model";
      renderBook();
    });
  }
  for (const b of document.querySelectorAll("#site-mode button")) {
    b.addEventListener("click", () => {
      S.site.mode = b.dataset.mode;
      for (const x of document.querySelectorAll("#site-mode button")) x.setAttribute("aria-checked", String(x === b));
      renderBook();
    });
  }
  const m = S.meta.models.m5;
  $("#site-foot").textContent = `책 고르는 규칙: ${S.feed.rule} 원 데이터에는 책 제목이 없어(Goodreads 책 ID만 있음) 제목 대신 번호로 표시했습니다. `
    + `가림 강도: 엄격 = val F1 최대 임계값(${m.strict.toFixed(3)}), 넉넉 = val 재현율 0.8 임계값(${m.lenient.toFixed(3)}). `
    + `M5의 test 문장 PR-AUC는 ${m.test_pr_auc.toFixed(3)}(seed 42)이고, 엄격 설정에서 test 스포일러 문장의 ${pct(m.test_recall_strict)}를 가립니다. `
    + `"리뷰 통째로 접기"는 작성자가 스포일러 태그를 하나라도 붙인 리뷰를 전부 접는 기존 사이트의 방식입니다.`;
  openBook(0);
}

function openBook(i) {
  S.site.book = i;
  S.site.revealed = new Set(); S.site.openOld = new Set(); S.site.openEn = new Set();
  for (const b of document.querySelectorAll(".shelf-item")) b.setAttribute("aria-current", String(+b.dataset.i === i));
  $("#site-url").textContent = `chaekgalpi.example/book/${i + 1}`;
  renderBook();
}

/* what the model would mask at the chosen strength (also used for the page meter in every method) */
function modelMasked(r) {
  const t = S.site.mode === "strict" ? S.meta.models.m5.strict : S.meta.models.m5.lenient;
  return r.p.m5.map((x) => x >= t);
}

function renderBook() {
  const b = S.feed.books[S.site.book];
  const page = $("#book-page");
  const avg = b.reviews.reduce((a, r) => a + (r.rating || 0), 0) / b.reviews.length;
  const nTagged = b.reviews.filter((r) => r.tags.some(Boolean)).length;

  // how much of the page each approach hides
  let total = 0, modelHidden = 0, oldHidden = 0, tagSents = 0, caught = 0;
  for (const r of b.reviews) {
    const m = modelMasked(r);
    total += r.sentences.length;
    modelHidden += m.filter(Boolean).length;
    if (r.tags.some(Boolean)) oldHidden += r.sentences.length;
    r.tags.forEach((t, j) => { if (t) { tagSents++; if (m[j]) caught++; } });
  }

  const head = h("div", { class: "book-head" },
    h("div", { class: "cover", style: `background:${coverBg(b.genres)}`, "aria-hidden": "true" },
      h("div", { class: "c-genre", text: b.genres_ko.join(" · ") }), h("div", { class: "c-line" }), h("div", { class: "c-name", text: b.name })),
    h("div", {},
      h("div", { class: "gen" }, b.genres_ko.map((g) => h("span", { text: g }))),
      h("h2", { text: b.name }),
      h("div", { class: "book-stats" },
        h("span", {}, h("b", { text: avg.toFixed(1) }), " 평균 별점"),
        h("span", {}, h("b", { text: String(b.reviews.length) }), " 리뷰"),
        h("span", {}, h("b", { text: String(nTagged) }), " 스포일러 태그가 있는 리뷰"))));

  const meter = h("div", { class: "page-meter" },
    h("span", {}, "이 페이지 ", h("b", { text: `${total}문장` }), " 중"),
    h("span", { class: S.site.method === "model" ? "cur" : null }, "문장만 가리기 ", h("b", { style: "color:#b5452f", text: `${modelHidden}문장` })),
    h("span", { class: S.site.method === "old" ? "cur" : null }, "리뷰 통째로 접기 ", h("b", { text: `${oldHidden}문장` })),
    h("span", { class: "meter-bar", title: "왼쪽: 모델이 가린 비율, 회색 전체: 기존 방식이 숨긴 비율", "aria-hidden": "true" },
      h("i", { style: `width:${pct(modelHidden / total)};background:#b5452f` }),
      h("i", { style: `width:${pct(Math.max(0, oldHidden - modelHidden) / total)};background:#d8cbb7` })),
    S.site.tags ? h("span", {}, "작성자 태그 ", h("b", { text: `${tagSents}문장` }), ` 중 ${caught}문장 가림`) : null);

  const list = h("div", { class: "reviews" });
  b.reviews.forEach((r, ri) => list.append(reviewCard(r, ri)));
  page.replaceChildren(head, meter, list);
}

function stars(n) {
  const out = h("span", { class: "stars", "aria-label": `별점 ${n}점` });
  for (let i = 1; i <= 5; i++) out.append(h("span", { class: i <= n ? "" : "off", text: "★" }));
  return out;
}

function reviewCard(r, ri) {
  const key = (j) => `${r.id}:${j}`;
  const tagged = r.tags.some(Boolean);
  const card = h("article", { class: "review" },
    h("div", { class: "rv-head" },
      h("span", { class: "avatar", style: `background:${AVATAR[ri % AVATAR.length]}`, text: r.reader.slice(-1) }),
      h("span", { class: "rv-name", text: r.reader }), r.rating ? stars(r.rating) : null,
      tagged && S.site.tags ? h("span", { class: "rv-badge", text: "작성자가 스포일러 태그를 붙인 리뷰" }) : null));

  // 기존 방식: the whole review is folded if the author tagged anything
  const method = S.site.method;
  if (method === "old" && tagged && !S.site.openOld.has(r.id)) {
    card.append(h("div", { class: "collapsed" },
      h("span", {}, h("b", { text: "스포일러가 포함된 리뷰입니다." }), ` 리뷰 전체(${r.sentences.length}문장)가 숨겨져 있어요.`),
      h("button", { type: "button", class: "linkish", onclick: () => { S.site.openOld.add(r.id); renderBook(); } }, "리뷰 보기")));
    return card;
  }

  const masked = method === "model" ? modelMasked(r) : r.sentences.map(() => false);
  const text = h("div", { class: "rv-text" });
  let hiddenNow = 0;
  const groups = runs(masked);
  for (const g of groups) {
    if (g[0]) text.append(" ");
    if (masked[g[0]]) {
      const open = S.site.revealed.has(key(g[0]));
      if (!open) hiddenNow++;
      text.append(maskBlock(g.map((j) => ({ text: r.sentences[j], i: j, cls: S.site.tags && r.tags[j] ? "tagged" : null })), open,
        () => { open ? S.site.revealed.delete(key(g[0])) : S.site.revealed.add(key(g[0])); renderBook(); }));
    } else {
      g.forEach((j, k) => {
        if (k) text.append(" ");
        const isTag = S.site.tags && r.tags[j];
        // red only when the model was masking and left a tagged sentence visible
        const miss = isTag && method === "model";
        text.append(h("span", { class: isTag ? (miss ? "tagged-miss" : "tagged") : null,
          title: miss ? "작성자 태그가 있지만 모델이 가리지 않은 문장" : (isTag ? "작성자가 스포일러 태그를 붙인 문장" : null), text: r.sentences[j] }));
      });
    }
  }
  card.append(text);

  const nMasked = masked.filter(Boolean).length;
  const foot = h("div", { class: "rv-foot" });
  if (method === "model" && nMasked) {
    foot.append(h("span", { text: `스포일러 ${nMasked}문장 가림` }),
      h("button", { type: "button", class: "linkish", onclick: () => {
        const all = hiddenNow > 0;
        for (const g of groups) if (masked[g[0]]) all ? S.site.revealed.add(key(g[0])) : S.site.revealed.delete(key(g[0]));
        renderBook();
      } }, hiddenNow ? "모두 보기" : "다시 가리기"));
  } else if (method === "model") {
    foot.append(h("span", { text: "가린 문장 없음" }));
  }
  if (method === "old" && tagged) {
    foot.append(h("button", { type: "button", class: "linkish", onclick: () => { S.site.openOld.delete(r.id); renderBook(); } }, "다시 접기"));
  }
  if (r.en) {
    const open = S.site.openEn.has(r.id);
    foot.append(h("button", { type: "button", class: "linkish", "aria-expanded": String(open), onclick: () => {
      open ? S.site.openEn.delete(r.id) : S.site.openEn.add(r.id); renderBook();
    } }, open ? "영어 원문 닫기" : "영어 원문 보기"));
  }
  card.append(foot);
  if (r.en && S.site.openEn.has(r.id)) card.append(h("div", { class: "en", lang: "en", text: r.en.join(" ") }));
  return card;
}

document.addEventListener("DOMContentLoaded", boot);
