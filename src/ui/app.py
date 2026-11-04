#!/usr/bin/env python3
# The local web page: the question page and the answer test bench.


import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))
from search import HybridSearcher, source_name, INDEX_DIR
from ask import answer_question, generate_stream, preload_model, sources_for_question, DEFAULT_MODEL, CONTEXT_WINDOW

HOST = "127.0.0.1"
PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else DEFAULT_MODEL

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dekati Knowledge Assistant</title>
<style>
 /* Dekati look: white page, brand red #C1272D, charcoal text #32373C (dekati.com).
    Always light, also when the computer is in dark mode. */
 :root { color-scheme: light; --bg:#ffffff; --soft:#f6f6f7; --card:#fff; --ink:#32373c; --muted:#6b7075;
         --line:#e4e4e6; --accent:#c1272d; --accent-dark:#9e1f24; --accent-soft:#fbeaea; --accent-ink:#ffffff;
         --warn:#b3261e; --shadow:0 1px 2px rgba(50,55,60,.06), 0 4px 14px rgba(50,55,60,.06); }
 body { margin:0; background:var(--soft); color:var(--ink); font:16px/1.5 "Barlow", "Helvetica Neue", Arial, sans-serif }
 .top { background:#fff; border-bottom:4px solid var(--accent); box-shadow:var(--shadow) }
 .top .in { max-width:860px; margin:0 auto; padding:10px 20px; display:flex; align-items:center; gap:16px; flex-wrap:wrap }
 .top img { height:46px; width:auto; display:block }
 .top .title { border-left:1px solid var(--line); padding-left:16px }
 .top .title b { display:block; font-size:19px; letter-spacing:.01em }
 .top .title span { color:var(--muted); font-size:13px }
 .top a.nav { margin-left:auto; color:var(--accent); font-weight:600; text-decoration:none; font-size:14px;
              border:1px solid var(--accent); border-radius:99px; padding:6px 14px }
 .top a.nav:hover { background:var(--accent); color:#fff }
 main { max-width:860px; margin:0 auto; padding:28px 16px 60px }
 h1 { font-size:22px; margin:0 0 4px } .sub { color:var(--muted); margin:0 0 20px; font-size:14px }
 form { display:flex; gap:8px; flex-wrap:wrap }
 textarea { flex:1 1 100%; min-height:70px; padding:12px; font:inherit; color:inherit;
            background:var(--card); border:1px solid var(--line); border-radius:10px; resize:vertical }
 button { padding:10px 18px; font:inherit; border-radius:10px; border:1px solid var(--line);
          background:var(--card); color:var(--ink); cursor:pointer }
 button.primary { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); font-weight:600 }
 button.primary:hover { background:var(--accent-dark) }
 button:disabled { opacity:.5; cursor:default }
 .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin-top:18px;
         box-shadow:var(--shadow) }
 .src b { color:var(--accent) }
 .answer { white-space:pre-wrap }
 .src { border-top:1px solid var(--line); padding:10px 0 }
 .src:first-of-type { border-top:0 }
 .src b { margin-right:6px } .meta { color:var(--muted); font-size:13px }
 details { margin-top:6px } summary { cursor:pointer; color:var(--muted); font-size:13px }
 .passage { white-space:pre-wrap; font-size:14px; margin-top:6px; padding:10px;
            background:var(--bg); border-radius:8px; max-height:260px; overflow:auto }
 .badge { display:inline-block; font-size:12px; padding:1px 8px; border-radius:99px;
          border:1px solid var(--line); color:var(--muted); margin-left:6px }
 .warn { color:var(--warn) } .status { color:var(--muted); margin-top:14px; min-height:1.5em }
 .examples button { font-size:13px; padding:5px 12px; margin:6px 6px 0 0; border-color:var(--accent);
                    color:var(--accent); border-radius:99px; background:#fff }
 .examples button:hover { background:var(--accent-soft) }
 .adv { display:inline-flex; align-items:center; gap:6px; margin-left:10px; font-size:14px; color:var(--muted, #555); cursor:pointer }
 h1.page { font-size:24px; margin:22px 0 2px }
</style></head>
<body>
<div class="top"><div class="in"><img src="/logo.png" alt="Dekati">
 <div class="title"><b>Knowledge Assistant</b><span>Answers from Dekati's internal documents &middot; prototype</span></div>
 <a class="nav" href="/test">Answer test bench</a></div></div>
<main>
 <h1 class="page">Ask a question</h1>
 <p class="sub">Answers come only from Dekati's internal documents and cite their sources.
 Runs locally on this computer.</p>
 <form id="f">
  <textarea id="q" placeholder="Ask a question, e.g. What are the technical specifications of the DOFR?"></textarea>
  <button class="primary" id="ask" type="submit">Ask</button>
  <button id="searchonly" type="button" title="Show matching passages without generating an answer">Search only</button>
  <label class="adv" title="Reads 8 sources and adds suggestions based on them. Takes about 3 minutes on this laptop.">
   <input type="checkbox" id="advisor"> Advisor mode: facts + suggestions (slower)</label>
 </form>
 <div class="examples" id="ex"></div>
 <div class="status" id="status"></div>
 <div id="out"></div>
</main>
<script>
const EXAMPLES = ["What kind of connectors does Dekati Cyclone have?",
 "What is the maximum temperature for the heated probes and heated sampling lines?",
 "What are the technical specifications of the DOFR?"];
const $ = id => document.getElementById(id);
for (const e of EXAMPLES) { const b = document.createElement("button"); b.type="button";
  b.textContent = e; b.onclick = () => { $("q").value = e; }; $("ex").appendChild(b); }
function esc(s) { return String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
async function streamAsk(q, on) {
  const r = await fetch("/api/ask_stream", { method:"POST", headers:{"Content-Type":"application/json"},
                                             body: JSON.stringify({ question:q, mode: ($("advisor") && $("advisor").checked) ? "advisor" : "facts" }) });
  if (!r.ok) { let m; try { m = (await r.json()).error; } catch {} throw new Error(m || r.status); }
  const reader = r.body.getReader(), dec = new TextDecoder(); let buf = "";
  for (;;) {
    const { value, done } = await reader.read(); if (done) break;
    buf += dec.decode(value, { stream:true }); let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1); if (!line) continue;
      const ev = JSON.parse(line); if (ev.type === "error") throw new Error(ev.error);
      if (on[ev.type]) on[ev.type](ev);
    }
  }
}
function sourcesHtml(sources) {
  let html = `<div class="card"><b>Sources</b>`;
  for (const s of sources) {
    const also = s.also_in.slice(0,5).map(a => `${esc(a.document)} | ${esc(a.location || "-")}`).join("<br>");
    html += `<div class="src"><b>[${s.n}]</b>${esc(s.document)} <span class="meta">| ${esc(s.location || "-")}</span>
      <span class="badge">${esc(s.type)}</span>
      ${s.copies_total > 1 ? `<div class="meta">Same text also in ${s.copies_total-1} other place(s):<br>${also}${s.copies_total>6?"<br>...":""}</div>` : ""}
      <details><summary>Show passage</summary><div class="passage">${esc(s.text)}</div></details></div>`;
  }
  return html + `</div>`;
}
async function run(useLlm) {
  const q = $("q").value.trim(); if (!q) return;
  $("ask").disabled = $("searchonly").disabled = true; $("out").innerHTML = "";
  const t0 = Date.now(); let phase = useLlm ? "Searching..." : "Searching...";
  const timer = setInterval(() => { $("status").textContent = phase + " " + Math.round((Date.now()-t0)/1000) + " s"; }, 1000);
  try {
    if (!useLlm) {
      const r = await fetch("/api/ask", { method:"POST", headers:{"Content-Type":"application/json"},
                                          body: JSON.stringify({ question:q, llm:false, mode: $("advisor").checked ? "advisor" : "facts" }) });
      const d = await r.json(); if (!r.ok) throw new Error(d.error || r.status);
      $("out").innerHTML = sourcesHtml(d.sources); $("status").textContent = ""; return;
    }
    let srcHtml = "";
    await streamAsk(q, {
      sources: ev => { phase = "Reading the sources and writing the answer...";
        srcHtml = sourcesHtml(ev.sources);
        $("out").innerHTML = `<div class="card"><div class="answer" id="ans"></div><p class="meta" id="ameta">writing...</p></div>` + srcHtml; },
      token: ev => { phase = "Writing the answer..."; $("ans").textContent += ev.text; },
      done: ev => { const g = ev.grounding;
        $("ans").textContent = ev.answer;
        $("ameta").innerHTML = (g.ok ? "Citations checked" : '<span class="warn">Check this answer: no valid citation found</span>')
          + ` &middot; answer ${ev.generation_seconds} s &middot; ${esc(ev.model)}`; }
    });
    $("status").textContent = "";
  } catch (e) { $("status").innerHTML = `<span class="warn">Error: ${esc(e.message)}</span>`; }
  finally { clearInterval(timer); $("ask").disabled = $("searchonly").disabled = false; }
}
$("f").onsubmit = e => { e.preventDefault(); run(true); };
$("searchonly").onclick = () => run(false);
$("q").addEventListener("keydown", e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(true); });
</script></body></html>"""

TEST_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Answer Test Bench</title>
<style>
 /* Dekati look: white page, brand red #C1272D, charcoal text #32373C (dekati.com).
    Always light, also when the computer is in dark mode. */
 :root { color-scheme: light; --bg:#ffffff; --soft:#f6f6f7; --card:#fff; --ink:#32373c; --muted:#6b7075;
         --line:#e4e4e6; --accent:#c1272d; --accent-dark:#9e1f24; --accent-soft:#fbeaea; --accent-ink:#ffffff;
         --warn:#b3261e; --good:#1e7a3c; --mid:#9a6a00; --bad:#8e1b5a; --good-bg:#e6f4ea; --mid-bg:#fdf3d8; --bad-bg:#f8e6ef; --shadow:0 1px 2px rgba(50,55,60,.06), 0 4px 14px rgba(50,55,60,.06); }
 body { margin:0; background:var(--soft); color:var(--ink); font:15px/1.5 "Barlow", "Helvetica Neue", Arial, sans-serif }
 .top { background:#fff; border-bottom:4px solid var(--accent); box-shadow:var(--shadow) }
 .top .in { max-width:1400px; margin:0 auto; padding:10px 20px; display:flex; align-items:center; gap:16px; flex-wrap:wrap }
 .top img { height:46px; width:auto; display:block }
 .top .title { border-left:1px solid var(--line); padding-left:16px }
 .top .title b { display:block; font-size:19px; letter-spacing:.01em }
 .top .title span { color:var(--muted); font-size:13px }
 .top a.nav { margin-left:auto; color:var(--accent); font-weight:600; text-decoration:none; font-size:14px;
              border:1px solid var(--accent); border-radius:99px; padding:6px 14px }
 .top a.nav:hover { background:var(--accent); color:#fff }
 header { padding:18px 20px 0; max-width:1400px; margin:0 auto }
 h1 { font-size:20px; margin:0 } .sub { color:var(--muted); font-size:13px; margin:2px 0 0 }
 .score { display:flex; gap:10px; flex-wrap:wrap; margin:14px 0 0 }
 .pill { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:8px 14px; min-width:96px }
 .pill b { display:block; font-size:20px } .pill span { color:var(--muted); font-size:12px }
 .pill.good b { color:var(--good) } .pill.mid b { color:var(--mid) } .pill.bad b { color:var(--bad) }
 main { max-width:1400px; margin:0 auto; padding:14px 20px 50px;
        display:grid; grid-template-columns: minmax(0,1.1fr) minmax(0,1fr); gap:16px }
 @media (max-width:900px) { main { grid-template-columns:1fr } }
 .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; box-shadow:var(--shadow) }
 .card h2 { font-size:13px; margin:0 0 10px; color:var(--accent); text-transform:uppercase; letter-spacing:.06em }
 textarea { width:100%; min-height:64px; padding:10px; font:inherit; color:inherit; background:var(--bg);
            border:1px solid var(--line); border-radius:8px; resize:vertical }
 .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:8px }
 button { padding:8px 14px; font:inherit; border-radius:8px; border:1px solid var(--line);
          background:var(--card); color:var(--ink); cursor:pointer }
 button.primary { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); font-weight:600 }
 button.primary:hover { background:var(--accent-dark) }
 button:disabled { opacity:.5; cursor:default }
 .chips button { font-size:12px; padding:4px 11px; border-color:var(--accent); color:var(--accent); border-radius:99px; background:#fff }
 .chips button:hover { background:var(--accent-soft) }
 .pill { box-shadow:var(--shadow); border-top:3px solid var(--line) }
 .pill.good { border-top-color:var(--good) } .pill.mid { border-top-color:var(--mid) } .pill.bad { border-top-color:var(--bad) }
 .answer { white-space:pre-wrap; padding:12px; background:var(--bg); border-radius:8px; min-height:80px }
 .meta { color:var(--muted); font-size:12px; margin-top:6px }
 .flag { display:inline-block; font-size:12px; padding:2px 8px; border-radius:99px; margin-right:6px }
 .flag.ok { background:var(--good-bg); color:var(--good) } .flag.no { background:var(--bad-bg); color:var(--bad) }
 .flag.na { background:var(--mid-bg); color:var(--mid) }
 .rate { margin-top:14px; border-top:1px solid var(--line); padding-top:12px }
 .rate p { margin:8px 0 4px; font-weight:600; font-size:13px }
 .opt button.sel.good { background:var(--good-bg); border-color:var(--good); color:var(--good) }
 .opt button.sel.mid  { background:var(--mid-bg); border-color:var(--mid); color:var(--mid) }
 .opt button.sel.bad  { background:var(--bad-bg); border-color:var(--bad); color:var(--bad) }
 .opt button.sel.neutral { border-color:var(--ink) }
 input[type=text] { width:100%; padding:8px; font:inherit; color:inherit; background:var(--bg);
                    border:1px solid var(--line); border-radius:8px }
 .src { border:1px solid var(--line); border-radius:8px; padding:10px; margin-bottom:10px }
 .src .hd { display:flex; justify-content:space-between; gap:8px; font-size:13px }
 .src .num { font-weight:700; margin-right:6px; color:var(--accent) }
 .cited { border-left:4px solid var(--accent); background:var(--accent-soft) }
 .passage { white-space:pre-wrap; font-size:13px; margin-top:6px; max-height:180px; overflow:auto;
            background:var(--bg); border-radius:6px; padding:8px }
 table { width:100%; border-collapse:collapse; font-size:13px }
 th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top }
 th { color:var(--muted); font-weight:600 }
 .hist { grid-column:1 / -1 } .status { color:var(--muted); font-size:13px; min-height:1.4em; margin-top:6px }
 .tag { font-weight:600 } .tag.good { color:var(--good) } .tag.mid { color:var(--mid) } .tag.bad { color:var(--bad) }
</style></head><body>
<div class="top"><div class="in"><img src="/logo.png" alt="Dekati">
 <div class="title"><b>Answer Test Bench</b><span>Check and rate the assistant's answers &middot; prototype</span></div>
 <a class="nav" href="/">Back to the assistant</a></div></div>
<header>
 <p class="sub">Ask Dekati questions, read each answer next to its sources, and rate it.
 Ratings stay in this browser only.</p>
 <div class="score">
  <div class="pill"><b id="sTotal">0</b><span>rated</span></div>
  <div class="pill good"><b id="sGood">0</b><span>correct</span></div>
  <div class="pill mid"><b id="sMid">0</b><span>partly correct</span></div>
  <div class="pill bad"><b id="sBad">0</b><span>wrong</span></div>
  <div class="pill good"><b id="sRef">0</b><span>correct refusals</span></div>
  <div class="pill bad"><b id="sHal">0</b><span>made-up facts</span></div>
  <div class="pill"><b id="sTime">-</b><span>median answer time</span></div>
 </div>
</header>
<main>
 <section class="card">
  <h2>Question</h2>
  <textarea id="q" placeholder="Type a Dekati question"></textarea>
  <div class="row chips" id="chips"></div>
  <div class="row"><button class="primary" id="go">Ask</button>
   <span class="status" id="status"></span></div>

  <h2 style="margin-top:16px">Answer</h2>
  <div class="answer" id="answer">-</div>
  <div class="meta" id="ameta"></div>

  <div class="rate" id="rate" hidden>
   <p>1. Is the answer correct?</p>
   <div class="row opt" data-k="correct">
    <button data-v="1" class="good">Correct</button><button data-v="0.5" class="mid">Partly</button>
    <button data-v="0" class="bad">Wrong</button><button data-v="refusal" class="good">Correctly said "not found"</button>
    <button data-v="missed" class="bad">Said "not found" but answer exists</button></div>
   <p>2. Is every statement supported by its cited source?</p>
   <div class="row opt" data-k="grounded">
    <button data-v="1" class="good">Yes</button><button data-v="0" class="bad">No, it added facts</button>
    <button data-v="na" class="neutral">No answer given</button></div>
   <p>3. Did the right document appear in the sources?</p>
   <div class="row opt" data-k="retrieval">
    <button data-v="1" class="good">Yes</button><button data-v="0" class="bad">No</button>
    <button data-v="na" class="neutral">Don't know</button></div>
   <p>Notes (optional)</p>
   <input type="text" id="notes" placeholder="e.g. right document, wrong value">
   <p>Rated by (name or role)</p>
   <input type="text" id="rater" placeholder="e.g. Dekati product specialist">
   <div class="row"><button class="primary" id="save">Save rating</button></div>
  </div>
 </section>

 <section class="card">
  <h2>Sources the answer used</h2>
  <div id="sources"><p class="meta">Sources appear here. Cited ones are highlighted.</p></div>
 </section>

 <section class="card hist">
  <h2>Rated questions</h2>
  <table><thead><tr><th>#</th><th>Question</th><th>Correct</th><th>Grounded</th><th>Right document</th>
   <th>Time</th><th>Rated by</th><th>Notes</th></tr></thead><tbody id="hist"></tbody></table>
  <div class="row"><button id="csv">Download results (CSV)</button><button id="clear">Clear all ratings</button></div>
 </section>
</main>
<script>
const CHIPS = ["What kind of connectors does Dekati Cyclone have?","Give me a report about the Company HORIBA?",
 "Give me a report about carbon capture application?",
 "What is the maximum temperatures for the heated probes and heated sampling lines?",
 "What are the Technical specifications of the DOFR?","What does Dekati do and what products does Dekati make?",
 "Who won the football World Cup in 2018?"];
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let current = null, choice = {};
function load() { try { return JSON.parse(localStorage.getItem("dekati_ratings") || "[]"); } catch { return []; } }
function store(v) { try { localStorage.setItem("dekati_ratings", JSON.stringify(v)); } catch {} }
for (const c of CHIPS) { const b = document.createElement("button"); b.textContent = c.length > 48 ? c.slice(0,46)+"..." : c;
  b.title = c; b.onclick = () => { $("q").value = c; }; $("chips").appendChild(b); }
document.querySelectorAll(".opt").forEach(g => g.addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  g.querySelectorAll("button").forEach(x => x.classList.remove("sel")); b.classList.add("sel");
  choice[g.dataset.k] = b.dataset.v; }));
function label(v) { return ({ "1":"Correct","0.5":"Partly","0":"Wrong","refusal":"Correct refusal","missed":"Missed answer",
  "na":"-" })[v] ?? v; }
function cls(v) { return v==="1"||v==="refusal" ? "good" : v==="0.5" ? "mid" : v==="0"||v==="missed" ? "bad" : ""; }
function render() {
  const r = load();
  $("sTotal").textContent = r.length;
  $("sGood").textContent = r.filter(x => x.correct === "1").length;
  $("sMid").textContent = r.filter(x => x.correct === "0.5").length;
  $("sBad").textContent = r.filter(x => x.correct === "0" || x.correct === "missed").length;
  $("sRef").textContent = r.filter(x => x.correct === "refusal").length;
  $("sHal").textContent = r.filter(x => x.grounded === "0").length;
  const t = r.map(x => x.seconds).filter(x => x != null).sort((a,b) => a-b);
  $("sTime").textContent = t.length ? t[Math.floor(t.length/2)].toFixed(0) + " s" : "-";
  $("hist").innerHTML = r.map((x,i) => `<tr><td>${i+1}</td><td>${esc(x.question)}</td>
   <td class="tag ${cls(x.correct)}">${label(x.correct)}</td><td class="tag ${x.grounded==="1"?"good":x.grounded==="0"?"bad":""}">${x.grounded==="1"?"Yes":x.grounded==="0"?"No":"-"}</td>
   <td class="tag ${x.retrieval==="1"?"good":x.retrieval==="0"?"bad":""}">${x.retrieval==="1"?"Yes":x.retrieval==="0"?"No":"-"}</td>
   <td>${x.seconds != null ? x.seconds.toFixed(0)+" s" : "-"}</td><td>${esc(x.rater)}</td><td>${esc(x.notes)}</td></tr>`).join("");
}
async function streamAsk(q, on) {
  const r = await fetch("/api/ask_stream", { method:"POST", headers:{"Content-Type":"application/json"},
                                             body: JSON.stringify({ question:q, mode: ($("advisor") && $("advisor").checked) ? "advisor" : "facts" }) });
  if (!r.ok) { let m; try { m = (await r.json()).error; } catch {} throw new Error(m || r.status); }
  const reader = r.body.getReader(), dec = new TextDecoder(); let buf = "";
  for (;;) {
    const { value, done } = await reader.read(); if (done) break;
    buf += dec.decode(value, { stream:true }); let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1); if (!line) continue;
      const ev = JSON.parse(line); if (ev.type === "error") throw new Error(ev.error);
      if (on[ev.type]) on[ev.type](ev);
    }
  }
}
async function ask() {
  const q = $("q").value.trim(); if (!q) return;
  $("go").disabled = true; $("rate").hidden = true; $("answer").textContent = ""; $("ameta").textContent = "";
  $("sources").innerHTML = ""; choice = {}; document.querySelectorAll(".opt button").forEach(b => b.classList.remove("sel"));
  const t0 = Date.now(); let phase = "Searching...", retrieval = 0, sources = [];
  const tick = setInterval(() => $("status").textContent = phase + " " + Math.round((Date.now()-t0)/1000) + " s", 1000);
  const draw = cited => { $("sources").innerHTML = sources.map(s => `<div class="src ${cited.has(s.n)?"cited":""}">
      <div class="hd"><span><span class="num">[${s.n}]</span>${esc(s.document)}</span><span class="meta">${esc(s.location||"-")} &middot; ${esc(s.type)}</span></div>
      ${s.copies_total>1?`<div class="meta">Same text also in ${s.copies_total-1} other place(s)</div>`:""}
      <details ${cited.has(s.n)?"open":""}><summary class="meta">Passage</summary><div class="passage">${esc(s.text)}</div></details></div>`).join(""); };
  try {
    await streamAsk(q, {
      sources: ev => { sources = ev.sources; retrieval = ev.retrieval_seconds;
        phase = "Reading the sources and writing the answer..."; draw(new Set()); },
      token: ev => { phase = "Writing the answer..."; $("answer").textContent += ev.text; },
      done: ev => { const g = ev.grounding;
        current = { question:q, seconds: retrieval + ev.generation_seconds };
        $("answer").textContent = ev.answer;
        $("ameta").innerHTML = `<span class="flag ${g.ok?"ok":"no"}">${g.ok?"citations valid":"no valid citation"}</span>`
          + (g.said_not_found ? `<span class="flag na">said not found</span>` : "")
          + `search ${retrieval} s &middot; answer ${ev.generation_seconds} s &middot; ${esc(ev.model)}`;
        draw(new Set(g.cited_sources)); }
    });
    $("rate").hidden = false; $("notes").value = ""; $("status").textContent = "";
  } catch (e) { $("status").textContent = "Error: " + e.message; }
  finally { clearInterval(tick); $("go").disabled = false; }
}
$("go").onclick = ask;
$("q").addEventListener("keydown", e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask(); });
$("save").onclick = () => {
  if (!current) return;
  if (!choice.correct) { $("status").textContent = "Choose at least question 1 before saving."; return; }
  try { localStorage.setItem("dekati_rater", $("rater").value); } catch {}
  const r = load(); r.push({ ...current, ...choice, rater: $("rater").value, notes: $("notes").value, at: new Date().toISOString() });
  store(r); render(); $("rate").hidden = true; $("status").textContent = "Saved in this browser.";
};
$("csv").onclick = () => {
  const r = load(); if (!r.length) return;
  const cols = ["question","correct","grounded","retrieval","seconds","rater","notes","at"];
  const csv = [cols.join(",")].concat(r.map(x => cols.map(c => '"' + String(x[c] ?? "").replace(/"/g,'""') + '"').join(","))).join("\n");
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], {type:"text/csv"}));
  a.download = "dekati_answer_ratings.csv"; a.click();
};
$("clear").onclick = () => { if (confirm("Delete all ratings stored in this browser?")) { store([]); render(); } };
try { $("rater").value = localStorage.getItem("dekati_rater") || ""; } catch {}
render();
</script></body></html>"""

LOGO = (Path(__file__).resolve().parent / "dekati-logo.png").read_bytes()

searcher = None
search_lock = threading.Lock()
answer_lock = threading.Lock()


# Prepare the sources for showing on the web page.
def sources_for(results):
    return [{
        "n": n, "document": source_name(r), "type": r["source_type"],
        "location": r.get("location"), "chunk_id": r["chunk_id"],
        "text": r.get("context_text") or r.get("text"),
        "also_in": [{"document": source_name(c), "location": c.get("location")}
                    for c in r.get("copies", []) if c["chunk_id"] != r["chunk_id"]],
        "copies_total": r.get("copies_total", 1),
    } for n, r in enumerate(results, start=1)]


# Answers the browser's requests.
class Handler(BaseHTTPRequestHandler):
    # Do not log requests, so questions stay private.
    def log_message(self, fmt, *args):
        pass

    # Send a reply to the browser.
    def send(self, code, body, content_type="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # Serve the pages, the logo and the status check.
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send(200, PAGE, "text/html")
        elif self.path == "/test":
            self.send(200, TEST_PAGE, "text/html")
        elif self.path == "/logo.png":
            self.send(200, LOGO, "image/png")
        elif self.path == "/api/health":
            self.send(200, json.dumps({"ok": True, "model": MODEL}))
        else:
            self.send(404, json.dumps({"error": "not found"}))

    # Send the sources first, then the answer piece by piece.
    def stream(self, question, mode="facts"):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        # Send one piece to the browser.
        def write(obj):
            self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()

        try:
            t0 = time.time()
            with search_lock:
                results = searcher.search(question, top_k=sources_for_question(question, mode), context_window=CONTEXT_WINDOW)
            write({"type": "sources", "sources": sources_for(results),
                   "retrieval_seconds": round(time.time() - t0, 2)})
            with answer_lock:
                for event in generate_stream(question, results, model=MODEL, mode=mode):
                    write(event)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                write({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass

    # Handle a question sent from the page.
    def do_POST(self):
        if self.path == "/api/ask_stream":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            question = (body.get("question") or "").strip()
            mode = "advisor" if body.get("mode") == "advisor" else "facts"
            if not question:
                return self.send(400, json.dumps({"error": "empty question"}))
            return self.stream(question, mode)
        if self.path != "/api/ask":
            return self.send(404, json.dumps({"error": "not found"}))
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            question = (body.get("question") or "").strip()
            if not question:
                return self.send(400, json.dumps({"error": "empty question"}))
            if body.get("llm", True):
                with search_lock, answer_lock:
                    result = answer_question(question, searcher, model=MODEL)
                out = {k: result[k] for k in ("answer", "grounding", "retrieval_seconds",
                                              "generation_seconds")}
                out["model"] = MODEL
                out["sources"] = sources_for(result["results"])
            else:
                t0 = time.time()
                with search_lock:
                    mode = "advisor" if body.get("mode") == "advisor" else "facts"
                    results = searcher.search(question, top_k=sources_for_question(question, mode), context_window=CONTEXT_WINDOW)
                out = {"sources": sources_for(results),
                       "retrieval_seconds": round(time.time() - t0, 2)}
            self.send(200, json.dumps(out, ensure_ascii=False))
        except Exception as exc:
            self.send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


# Load everything and start the local web page.
def main():
    global searcher
    index_dir = INDEX_DIR
    if "--root" in sys.argv:
        index_dir = Path(sys.argv[sys.argv.index("--root") + 1]) / "data" / "processed" / "index"
    print("Loading search index (takes a few seconds)...")
    searcher = HybridSearcher(index_dir)
    print("Loading the answer model into memory (kept for 30 minutes between questions)...")
    try:
        preload_model(MODEL)
    except Exception as exc:
        print(f"  Ollama not reachable ({type(exc).__name__}); answers will fail until it runs.")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Ready: open http://localhost:{PORT} in your browser.  Model: {MODEL}")
    print("Stop with Ctrl+C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
