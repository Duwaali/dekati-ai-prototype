#!/usr/bin/env python3
# Builds a local web page that shows the answers of two models side by side, with their
# sources, so that a person can rate them. The page works offline and stays on this laptop.
#
# Usage:
#   python src/evaluation/make_comparison_page.py --pair RUN_A RUN_B [--pair RUN_C RUN_D ...]
#   python src/evaluation/make_comparison_page.py --bundle bundle.json   (build from a saved bundle)
#   Add --label "Dekati" to give the page its own, empty rating store (and a label in the title).
#
# RUN_A, RUN_B are folders written by run_evaluation.py with the same questions.
# Screen output: numbers only.

import html
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "index"
OUT_DIR = PROJECT_ROOT / "data" / "evaluation"


# Read the answers of two runs and the sources each answer used.
def collect(pairs):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))
    from search import HybridSearcher, source_name
    from ask import sources_for_question, CONTEXT_WINDOW, DEFAULT_MODEL

    searcher = HybridSearcher(INDEX_DIR)
    checks = {"sources": 0, "same_as_run": 0, "fetched_by_id": 0, "missing": 0}

    def by_chunk_id(chunk_id):
        row = searcher.db.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return dict(row) if row else None

    def sources(question, chunk_ids, mode="facts"):
        # Search again to get the exact passages the model read; search is deterministic.
        results = searcher.search(question, top_k=sources_for_question(question, mode),
                                  context_window=CONTEXT_WINDOW)
        by_id = {r["chunk_id"]: r for r in results}
        out = []
        for n, cid in enumerate(chunk_ids, start=1):
            checks["sources"] += 1
            r = by_id.get(cid)
            if r is not None:
                checks["same_as_run"] += 1
                text = r.get("context_text") or r.get("text") or ""
            else:
                r = by_chunk_id(cid)
                if r is None:
                    checks["missing"] += 1
                    out.append({"n": n, "document": "(source not found)", "location": "", "type": "", "text": ""})
                    continue
                checks["fetched_by_id"] += 1
                text = r.get("text") or ""
            out.append({"n": n, "document": source_name(r), "location": r.get("location") or "",
                        "type": r.get("source_type") or "", "text": text})
        return out

    groups = []
    for run_a, run_b in pairs:
        a = json.loads((Path(run_a) / "run.json").read_text(encoding="utf-8"))
        b = json.loads((Path(run_b) / "run.json").read_text(encoding="utf-8"))
        qa = {q["id"]: q for q in a["questions"]}
        qb = {q["id"]: q for q in b["questions"]}
        if list(qa) != list(qb):
            raise SystemExit("STOP: the two runs do not have the same question ids.")
        items = []
        for qid in qa:
            x, y = qa[qid], qb[qid]
            if x["question"] != y["question"]:
                raise SystemExit(f"STOP: question {qid} differs between the two runs.")
            items.append({"id": qid, "question": x["question"], "answers": [
                {"model": (a.get("model") or DEFAULT_MODEL) + ("" if a.get("mode", "facts") == "facts" else " " + a["mode"]), "answer": x.get("answer", ""),
                 "seconds": x.get("generation_seconds"), "sources": sources(x["question"], x.get("answer_sources", []), a.get("mode", "facts"))},
                {"model": (b.get("model") or DEFAULT_MODEL) + ("" if b.get("mode", "facts") == "facts" else " " + b["mode"]), "answer": y.get("answer", ""),
                 "seconds": y.get("generation_seconds"), "sources": sources(y["question"], y.get("answer_sources", []), b.get("mode", "facts"))},
            ]})
        groups.append({"runs": [Path(run_a).name, Path(run_b).name], "items": items})
    return {"groups": groups}, checks


# Write the page. Answers are shown as "Answer A" and "Answer B" in a random order per question,
# so that the rater does not know which model wrote which answer.
def render(bundle, seed=None, label=""):
    rng = random.Random(seed)
    for g in bundle["groups"]:
        for it in g["items"]:
            it["swap"] = rng.random() < 0.5
    data = json.dumps(bundle, ensure_ascii=False).replace("</", "<\\/")
    return (PAGE.replace("__DATA__", data)
            .replace("__LABEL__", json.dumps(label))
            .replace("__LABELTEXT__", html.escape(label))
            .replace("__CREATED__", html.escape(datetime.now().strftime("%Y-%m-%d %H:%M"))))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Answer Comparison</title>
<style>
:root { --bg:#f5f6f7; --card:#fff; --ink:#1d2329; --muted:#5f6b77; --line:#d9dee3; --accent:#00579b;
  --good:#1e7a3c; --goodbg:#e4f3e8; --mid:#8a5a00; --midbg:#fbefd8; --bad:#a32020; --badbg:#fbe4e4; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
header { position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--line); padding:12px 20px;
  display:flex; flex-wrap:wrap; gap:10px 20px; align-items:center; }
header h1 { font-size:18px; margin:0; }
header .grow { flex:1; }
header input { padding:6px 8px; border:1px solid var(--line); border-radius:6px; font:inherit; width:230px; }
.progress { color:var(--muted); font-variant-numeric:tabular-nums; }
button { font:inherit; padding:6px 11px; border:1px solid var(--line); border-radius:6px; background:var(--card); cursor:pointer; }
button:hover { border-color:var(--accent); }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
main { max-width:1300px; margin:0 auto; padding:20px; display:grid; gap:22px; }
.intro { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 18px; }
.intro p { margin:4px 0; }
h2.group { margin:10px 0 0; font-size:17px; }
.q { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; display:grid; gap:14px; }
.q.done { border-color:var(--good); }
.qhead { display:flex; gap:12px; align-items:baseline; }
.qid { font-weight:700; color:var(--accent); }
.qtext { font-size:16px; font-weight:600; }
.pair { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media (max-width:900px) { .pair { grid-template-columns:1fr; } }
.ans { border:1px solid var(--line); border-radius:8px; padding:12px 14px; display:grid; gap:10px; min-width:0; }
.ans h3 { margin:0; font-size:15px; display:flex; justify-content:space-between; }
.ans h3 span { color:var(--muted); font-weight:400; font-size:13px; }
.text { white-space:pre-wrap; overflow-wrap:anywhere; background:#fafbfc; border-radius:6px; padding:10px; max-height:420px; overflow:auto; }
.cite { background:#dceaf7; border-radius:3px; padding:0 2px; font-weight:600; }
details.src { border-top:1px dashed var(--line); padding-top:6px; }
details.src summary { cursor:pointer; color:var(--accent); }
.s { margin:8px 0; }
.s b { display:block; overflow-wrap:anywhere; }
.s .loc { color:var(--muted); font-size:13px; }
.s .stext { white-space:pre-wrap; overflow-wrap:anywhere; font-size:13px; background:#fafbfc; padding:8px; border-radius:6px; max-height:260px; overflow:auto; margin-top:4px; }
.rate { display:grid; gap:6px; }
.rate .lab { font-size:13px; color:var(--muted); }
.opts { display:flex; flex-wrap:wrap; gap:6px; }
.opts button.sel.g { background:var(--goodbg); border-color:var(--good); color:var(--good); font-weight:600; }
.opts button.sel.m { background:var(--midbg); border-color:var(--mid); color:var(--mid); font-weight:600; }
.opts button.sel.b { background:var(--badbg); border-color:var(--bad); color:var(--bad); font-weight:600; }
.opts button.sel.n { background:#e8ecf0; border-color:var(--muted); font-weight:600; }
.pref { display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; border-top:1px solid var(--line); padding-top:10px; }
.pref input[type=text] { flex:1; min-width:200px; padding:6px 8px; border:1px solid var(--line); border-radius:6px; font:inherit; }
button:focus-visible, input:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
</style></head><body>
<header>
  <h1>Answer Comparison <span class="progress">__LABELTEXT__</span></h1>
  <span class="progress" id="progress"></span>
  <span class="grow"></span>
  <label>Rated by <input id="rater" placeholder="name or role, e.g. Dekati product specialist"></label>
  <button class="primary" id="csv">Download ratings (CSV)</button>
</header>
<main>
  <div class="intro">
    <p><b>How to rate.</b> For each question, two answers (A and B) were written by two different AI models from the same company documents. Which model wrote which answer is hidden.</p>
    <p>Read each answer, open its sources if needed, and rate it. Then choose which answer is better. Ratings are saved in this browser automatically; at the end press <b>Download ratings (CSV)</b>.</p>
    <p class="progress">Page created __CREATED__ · confidential, contains Dekati information · do not share outside Dekati and the thesis author.</p>
  </div>
  <div id="list"></div>
</main>
<script>
const DATA = __DATA__;
const KEY = "dekati_compare_" + __LABEL__ + "_" + DATA.groups.map(g => g.runs.join("+")).join("|");
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let R = {};
try { R = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { R = {}; }
try { $("rater").value = localStorage.getItem("dekati_rater") || ""; } catch (e) {}
function save() { try { localStorage.setItem(KEY, JSON.stringify(R)); } catch (e) {} progress(); }
$("rater").addEventListener("input", () => { try { localStorage.setItem("dekati_rater", $("rater").value); } catch (e) {} });

const CORRECT = [["1","Correct","g"],["0.5","Partly","m"],["0","Wrong","b"],["R","Correctly said not found","g"],["M","Said not found, but answer exists","b"]];
const GROUNDED = [["Y","Yes, all from sources","g"],["N","No, it added facts","b"],["-","No answer given","n"]];
const USEFUL = [["2","Useful","g"],["1","A little","m"],["0","Not useful","b"]];
const PREF = [["A","A is better","g"],["B","B is better","g"],["same","About the same","n"]];

function opts(key, list) {
  return `<div class="opts" data-key="${key}">` + list.map(([v,l,c]) =>
    `<button type="button" data-v="${v}" data-c="${c}" class="${R[key]===v?"sel "+c:""}">${esc(l)}</button>`).join("") + `</div>`;
}
function answerHtml(text) { return esc(text).replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>'); }

function card(g, gi, it) {
  const order = it.swap ? [1,0] : [0,1];
  const side = (slot, a) => {
    const k = `${gi}|${it.id}|${slot}`;
    const srcs = a.sources.map(s => `<div class="s"><b>[${s.n}] ${esc(s.document)}</b>
      <span class="loc">${esc(s.location || "-")} · ${esc(s.type)}</span><div class="stext">${esc(s.text)}</div></div>`).join("");
    return `<div class="ans"><h3>Answer ${slot}<span>${a.seconds != null ? Math.round(a.seconds) + " s" : ""}</span></h3>
      <div class="text">${answerHtml(a.answer)}</div>
      <details class="src"><summary>Sources (${a.sources.length})</summary>${srcs || "<p>No sources.</p>"}</details>
      <div class="rate"><span class="lab">Is the answer correct?</span>${opts(k+"|correct", CORRECT)}
      <span class="lab">Is everything in it supported by the sources?</span>${opts(k+"|grounded", GROUNDED)}
      <span class="lab">Is it useful for the task?</span>${opts(k+"|useful", USEFUL)}</div></div>`;
  };
  const k = `${gi}|${it.id}`;
  return `<section class="q" id="q-${gi}-${esc(it.id)}"><div class="qhead"><span class="qid">${esc(it.id)}</span><span class="qtext">${esc(it.question)}</span></div>
    <div class="pair">${side("A", it.answers[order[0]])}${side("B", it.answers[order[1]])}</div>
    <div class="pref"><span class="lab">Which answer is better?</span>${opts(k+"|pref", PREF)}
      <input type="text" data-note="${k}" placeholder="Note (optional)" value="${esc(R[k+"|note"] || "")}"></div></section>`;
}

function draw() {
  $("list").innerHTML = DATA.groups.map((g, gi) =>
    `<h2 class="group">Question set ${gi+1}</h2>` + g.items.map(it => card(g, gi, it)).join("")).join("");
  progress();
}
function isDone(gi, id) { return ["A","B"].every(s => R[`${gi}|${id}|${s}|correct`]) && R[`${gi}|${id}|pref`]; }
function progress() {
  let n = 0, d = 0;
  DATA.groups.forEach((g, gi) => g.items.forEach(it => { n++; const ok = isDone(gi, it.id); if (ok) d++;
    const el = document.getElementById(`q-${gi}-${it.id}`); if (el) el.classList.toggle("done", !!ok); }));
  $("progress").textContent = `${d} of ${n} questions rated`;
}
document.addEventListener("click", e => {
  const b = e.target.closest(".opts button"); if (!b) return;
  const box = b.parentElement, key = box.dataset.key;
  box.querySelectorAll("button").forEach(x => x.className = "");
  if (R[key] === b.dataset.v) { delete R[key]; } else { R[key] = b.dataset.v; b.className = "sel " + b.dataset.c; }
  save();
});
document.addEventListener("input", e => { const k = e.target.dataset.note; if (k) { R[k+"|note"] = e.target.value; save(); } });

$("csv").onclick = () => {
  const cols = ["set","question_id","question","slot","model","correct","grounded","useful","better","note","rated_by","answer_seconds"];
  const rows = [cols];
  DATA.groups.forEach((g, gi) => g.items.forEach(it => {
    const order = it.swap ? [1,0] : [0,1];
    ["A","B"].forEach((slot, i) => { const a = it.answers[order[i]], k = `${gi}|${it.id}`;
      const p = R[k+"|pref"]; const better = p === "same" ? "same" : p ? (p === slot ? "yes" : "no") : "";
      rows.push([gi+1, it.id, it.question, slot, a.model, R[`${k}|${slot}|correct`]||"", R[`${k}|${slot}|grounded`]||"",
        R[`${k}|${slot}|useful`]||"", better, R[k+"|note"]||"", $("rater").value, a.seconds ?? ""]); });
  }));
  const csv = rows.map(r => r.map(v => '"' + String(v).replace(/"/g,'""') + '"').join(",")).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["﻿" + csv], {type:"text/csv"}));
  a.download = "answer_comparison_ratings.csv"; document.body.appendChild(a); a.click(); a.remove();
};
draw();
</script></body></html>"""


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "--bundle" in sys.argv:
        bundle = json.loads(Path(sys.argv[sys.argv.index("--bundle") + 1]).read_text(encoding="utf-8"))
        out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else OUT_DIR / f"compare_{stamp}.html"
        checks = None
    else:
        pairs = [(sys.argv[i + 1], sys.argv[i + 2]) for i, a in enumerate(sys.argv) if a == "--pair"]
        if not pairs:
            print(__doc__ or "usage: see the top of this file")
            sys.exit(1)
        print("loading index...")
        bundle, checks = collect(pairs)
        out = OUT_DIR / f"compare_{stamp}.html"

    with open(out, "x", encoding="utf-8") as f:   # "x": never overwrite an existing file
        f.write(render(bundle, label=sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else ""))

    n_q = sum(len(g["items"]) for g in bundle["groups"])
    print(f"question sets          : {len(bundle['groups'])}")
    print(f"questions              : {n_q}")
    print(f"answers                : {2 * n_q}")
    if checks:
        print(f"sources                : {checks['sources']}")
        print(f"  same passage as run  : {checks['same_as_run']}")
        print(f"  fetched by chunk id  : {checks['fetched_by_id']}")
        print(f"  not found            : {checks['missing']}")
        print("CHECK sources found     : " + ("PASS" if checks["missing"] == 0 else "FAIL"))
    print(f"written to             : {out.parent.name}/{out.name}")


if __name__ == "__main__":
    main()
