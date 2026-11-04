#!/usr/bin/env python3
# Builds a local web page for judging the retrieved passages of one evaluation run:
# for each question, every pooled passage is shown with a "Has the answer" / "No" choice.
# The page works offline, stays on this laptop and exports the judgements as a CSV file
# with the same columns as judge_passages.csv, which score_evaluation.py reads.
#
# Usage:
#   python src/evaluation/make_judging_page.py data/evaluation/run_<timestamp> [--db chunks.sqlite]
#          [--top 3] [--skip Q8]
#   --top K   show only passages ranked in the top K by at least one search method
#   --skip Q  leave out a question (e.g. a control question whose answer is not in the corpus)
#
# The passages are shown in a random order per question and without their ranks,
# so the judge cannot tell which search method found them.
# Screen output: numbers only. Never overwrites an existing file.

import csv
import html
import json
import random
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "index" / "chunks.sqlite"


# The document name shown for a chunk (same rule as the search module).
def source_name(chunk):
    return (chunk.get("parent_source_file") or chunk.get("title")
            or chunk.get("source_file") or chunk.get("doc_id"))


# Read the questions, the passages to judge and their text.
def collect(run_dir, db_path, top=None, skip=()):
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run["questions"] = [q for q in run["questions"] if q["id"] not in skip]
    keep = {q["id"]: {r for m in ("keyword", "semantic", "hybrid") for r in q["ranks"][m][:top]}
            for q in run["questions"]} if top else None
    with open(run_dir / "judge_passages.csv", "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["question_id"] not in skip
                and (keep is None or int(r["row_id"]) in keep.get(r["question_id"], set()))]
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row

    by_question = {}
    missing = 0
    for r in rows:
        chunk = db.execute("SELECT * FROM chunks WHERE row_id = ?", (int(r["row_id"]),)).fetchone()
        if chunk is None:
            missing += 1
            continue
        chunk = dict(chunk)
        by_question.setdefault(r["question_id"], []).append({
            "pid": r["passage_id"], "qid": r["question_id"], "row": r["row_id"],
            "document": r.get("document") or source_name(chunk) or "",
            "location": r.get("location") or chunk.get("location") or "",
            "text": chunk.get("text") or "",
        })
    db.close()

    rng = random.Random(20261029)
    questions = []
    for q in run["questions"]:
        items = by_question.get(q["id"], [])
        rng.shuffle(items)
        questions.append({"id": q["id"], "question": q["question"], "passages": items})
    return {"run": run_dir.name, "questions": questions}, len(rows), missing


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Passage judging</title>
<style>
:root { --bg:#f7f7f5; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#dcdcd6;
        --yes:#1f7a3f; --no:#9a2b2b; --accent:#00507d; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; }
header { position:sticky; top:0; background:var(--accent); color:#fff; padding:12px 16px; z-index:2;
         display:flex; flex-wrap:wrap; gap:12px; align-items:center; justify-content:space-between; }
header button { background:#fff; color:var(--accent); border:0; border-radius:6px; padding:8px 14px;
                font-weight:600; cursor:pointer; }
main { max-width:900px; margin:0 auto; padding:16px; }
.help { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 16px; }
h2 { font-size:18px; margin:28px 0 8px; }
.p { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 16px; margin:10px 0; }
.p.yes { border-left:6px solid var(--yes); } .p.no { border-left:6px solid var(--no); }
.meta { color:var(--muted); font-size:13px; margin-bottom:6px; overflow-wrap:anywhere; }
.text { white-space:pre-wrap; overflow-wrap:anywhere; max-height:260px; overflow:auto;
        background:#fafaf8; border:1px solid var(--line); border-radius:6px; padding:8px; }
.choice { margin-top:8px; display:flex; gap:8px; }
.choice button { border:1px solid var(--line); background:#fff; border-radius:6px; padding:6px 12px; cursor:pointer; }
.choice button.on.y { background:var(--yes); color:#fff; border-color:var(--yes); }
.choice button.on.n { background:var(--no); color:#fff; border-color:var(--no); }
</style></head><body>
<header><div><b>Passage judging</b> &middot; <span id="count"></span></div>
<button id="export">Download judgements (CSV)</button></header>
<main>
<div class="help"><b>For each passage:</b> click <b>Has the answer</b> if the passage contains the
information that answers the question (fully or partly), otherwise <b>No</b>. Judge only the passage
itself. The order is random and does not show which search found it. Your choices are kept in this
browser while you work; click <b>Download judgements</b> at the end.</div>
<div id="list"></div>
</main>
<script>
const DATA = __DATA__;
const KEY = "judging|" + DATA.run;
let R = {};
try { R = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { R = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(R)); } catch (e) {} };
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const all = DATA.questions.flatMap(q => q.passages);
function count() {
  const done = all.filter(p => R[p.pid] !== undefined).length;
  document.getElementById("count").textContent = done + " of " + all.length + " judged";
}
function draw() {
  document.getElementById("list").innerHTML = DATA.questions.map(q =>
    `<h2>${esc(q.id)}: ${esc(q.question)}</h2>` + (q.passages.length ? q.passages.map((p, i) => `
    <div class="p ${R[p.pid] === 1 ? "yes" : R[p.pid] === 0 ? "no" : ""}" id="${esc(p.pid)}">
      <div class="meta">Passage ${i + 1} of ${q.passages.length} &middot; ${esc(p.document)}${p.location ? " &middot; " + esc(p.location) : ""}</div>
      <div class="text">${esc(p.text)}</div>
      <div class="choice">
        <button class="y ${R[p.pid] === 1 ? "on" : ""}" data-pid="${esc(p.pid)}" data-v="1">Has the answer</button>
        <button class="n ${R[p.pid] === 0 ? "on" : ""}" data-pid="${esc(p.pid)}" data-v="0">No</button>
      </div></div>`).join("") : "<p>No passages.</p>")).join("");
  count();
}
document.getElementById("list").addEventListener("click", e => {
  const b = e.target.closest("button[data-pid]"); if (!b) return;
  R[b.dataset.pid] = Number(b.dataset.v); save();
  const card = document.getElementById(b.dataset.pid);
  card.className = "p " + (R[b.dataset.pid] === 1 ? "yes" : "no");
  card.querySelectorAll("button").forEach(x => x.classList.toggle("on", Number(x.dataset.v) === R[b.dataset.pid]));
  count();
});
document.getElementById("export").onclick = () => {
  const left = all.filter(p => R[p.pid] === undefined).length;
  if (left) { document.getElementById("count").textContent = left + " passages still to judge"; return; }
  const rows = [["passage_id","question_id","row_id","document","location","relevant"]]
    .concat(all.map(p => [p.pid, p.qid, p.row, p.document, p.location, R[p.pid]]));
  const csv = rows.map(r => r.map(v => '"' + String(v).replace(/"/g,'""') + '"').join(",")).join("\\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {type:"text/csv"}));
  a.download = "judge_passages_rated.csv"; document.body.appendChild(a); a.click(); a.remove();
};
draw();
</script></body></html>"""


def main():
    args = [a for a in sys.argv[1:]]
    db_path, top, skip = DB_PATH, None, []
    if "--db" in args:
        i = args.index("--db"); db_path = Path(args[i + 1]); del args[i:i + 2]
    if "--top" in args:
        i = args.index("--top"); top = int(args[i + 1]); del args[i:i + 2]
    while "--skip" in args:
        i = args.index("--skip"); skip.append(args[i + 1]); del args[i:i + 2]
    if not args:
        print("usage: python src/evaluation/make_judging_page.py data/evaluation/run_<timestamp> [--db chunks.sqlite]")
        sys.exit(1)
    run_dir = Path(args[0])
    data, n_rows, missing = collect(run_dir, db_path, top, skip)
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out = run_dir / f"judge_passages_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(out, "x", encoding="utf-8") as f:   # "x": never overwrite an existing file
        f.write(PAGE.replace("__DATA__", blob))

    shown = sum(len(q["passages"]) for q in data["questions"])
    print(f"questions              : {len(data['questions'])}")
    print(f"passages in sheet      : {n_rows}")
    print(f"passages on page       : {shown}")
    print(f"passages not found     : {missing}")
    print("CHECK all passages shown: " + ("PASS" if shown == n_rows and missing == 0 else "FAIL"))
    print(f"written to             : {run_dir.name}/{out.name}")


if __name__ == "__main__":
    main()
