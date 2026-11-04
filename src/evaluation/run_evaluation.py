#!/usr/bin/env python3
# Runs the test questions and writes review sheets for judging the answers.
# Use --model qwen2.5:3b to answer with another Ollama model (default: llama3.2:3b).
# Use --mode advisor for answers with facts plus suggestions, from 8 sources.


import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if "--root" in sys.argv:
    PROJECT_ROOT = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()

QUESTIONS_FILE = Path(__file__).resolve().parents[2] / "evaluation" / "questions.json"
if "--questions" in sys.argv:
    QUESTIONS_FILE = Path(sys.argv[sys.argv.index("--questions") + 1]).resolve()

INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "index"
OUT_ROOT = PROJECT_ROOT / "data" / "evaluation"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))
from search import HybridSearcher, source_name, RRF_K
from ask import answer_question, DEFAULT_MODEL

TOP = 10
EXCERPT_CHARS = 700


# Get the top results from keyword, meaning and combined search.
def rank_lists(searcher, question):
    t0 = time.time()
    keyword = [r for r, _ in searcher.keyword(question)]
    t1 = time.time()
    semantic = [r for r, _ in searcher.semantic(question)]
    multilingual = [r for r, _ in searcher.semantic_multilingual(question)]
    t2 = time.time()
    fused = {}
    for hits in (keyword, semantic, multilingual):
        for rank, row in enumerate(hits, start=1):
            fused[row] = fused.get(row, 0.0) + 1.0 / (RRF_K + rank)
    hybrid = [r for r, _ in sorted(fused.items(), key=lambda x: -x[1])]

    # Remove repeated copies of the same text from a result list.
    def dedupe(rows):
        out, seen = [], set()
        for row in rows:
            chunk = searcher.fetch(row)
            key = chunk["duplicate_of"] or chunk["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
            if len(out) == TOP:
                break
        return out

    return ({"keyword": dedupe(keyword), "semantic": dedupe(semantic),
             "hybrid": dedupe(hybrid)},
            {"keyword_seconds": round(t1 - t0, 3), "semantic_seconds": round(t2 - t1, 3)})


# Ask the model, and try once more after a pause if the local model server fails.
def answer_with_retry(question, searcher, model, mode, pause=30):
    for attempt in (1, 2):
        try:
            return answer_question(question, searcher, model=model, mode=mode)
        except Exception as e:
            print(f"        answer failed (attempt {attempt}): {type(e).__name__}")
            if attempt == 1:
                time.sleep(pause)
    return None


# Run all test questions and save the review sheets.
def main():
    use_llm = "--no-llm" not in sys.argv
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else DEFAULT_MODEL
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "facts"
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "" if model == DEFAULT_MODEL else "_" + model.replace(":", "-").replace("/", "-")
    tag += "" if mode == "facts" else "_" + mode
    out_dir = OUT_ROOT / f"run_{stamp}{tag}"
    out_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 60)
    print("EVALUATION RUN  (screen output: numbers only)")
    print("=" * 60)
    print(f"questions : {len(questions)}")
    print(f"LLM       : {model if use_llm else 'no (retrieval only)'}  mode: {mode}")
    print("loading index...")
    searcher = HybridSearcher(INDEX_DIR)

    run = {"created": stamp, "top": TOP, "model": model if use_llm else None, "mode": mode, "questions": []}
    passage_rows, answer_rows = [], []
    md = [f"# Evaluation review {stamp}  (model: {model if use_llm else '-'})\n",
          "CONFIDENTIAL - contains Dekati text. Do not share outside the project.\n"]

    for qi, q in enumerate(questions, start=1):
        lists, timing = rank_lists(searcher, q["question"])
        pool = []
        for method in ("hybrid", "semantic", "keyword"):
            for row in lists[method]:
                if row not in pool:
                    pool.append(row)

        record = {"id": q["id"], "question": q["question"],
                  "answer_in_corpus": q.get("answer_in_corpus"),
                  "ranks": {m: lists[m] for m in lists}, "pool": pool, **timing}

        md.append(f"\n---\n\n## {q['id']}: {q['question']}\n")

        if use_llm:
            result = answer_with_retry(q["question"], searcher, model, mode)
            if result is None:
                result = {"answer": "(no answer: the model failed twice)", "sources": [],
                          "grounding": {"ok": False, "said_not_found": False, "cited_sources": [],
                                        "invalid_citations": [], "has_citation": False},
                          "retrieval_seconds": 0.0, "generation_seconds": 0.0, "failed": True}
                record["failed"] = True
            record.update({
                "answer": result["answer"],
                "answer_sources": [s["chunk_id"] for s in result["sources"]],
                "grounding_check": result["grounding"],
                "retrieval_seconds": result["retrieval_seconds"],
                "generation_seconds": result["generation_seconds"],
            })
            md.append("### System answer\n")
            md.append(result["answer"] + "\n")
            md.append("Sources:\n")
            for s in result["sources"]:
                md.append(f"- [{s['n']}] {s['document']} | {s['location'] or '-'} ({s['type']})")
            md.append("")
            answer_rows.append({
                "question_id": q["id"], "question": q["question"],
                "answer_in_corpus": "" if record["answer_in_corpus"] is None
                                    else int(record["answer_in_corpus"]),
                "auto_citations_ok": int(result["grounding"]["ok"]),
                "said_not_found": int(result["grounding"]["said_not_found"]),
                "correct": "", "grounded": "", "notes": "",
            })

        md.append("### Retrieved passages (judge each once)\n")
        for pi, row in enumerate(pool, start=1):
            c = searcher.fetch(row)
            pid = f"{q['id']}-P{pi:02d}"
            where = {m: (lists[m].index(row) + 1 if row in lists[m] else "-") for m in lists}
            md.append(f"**{pid}**  {source_name(c)} | {c.get('location') or '-'} "
                      f"({c['source_type']})  ranks: hybrid {where['hybrid']}, "
                      f"semantic {where['semantic']}, keyword {where['keyword']}\n")
            text = (c["text"] or "").replace("\n", " ")
            md.append("> " + text[:EXCERPT_CHARS] + ("..." if len(text) > EXCERPT_CHARS else "") + "\n")
            passage_rows.append({"passage_id": pid, "question_id": q["id"], "row_id": row,
                                 "document": source_name(c), "location": c.get("location") or "",
                                 "relevant": ""})
        run["questions"].append(record)
        print(f"  {qi:>3}/{len(questions)}  pooled passages {len(pool):>2}  "
              f"search {timing['keyword_seconds'] + timing['semantic_seconds']:.2f}s"
              + (f"  answer {record['generation_seconds']:.0f}s" if use_llm else ""))

    (out_dir / "review.md").write_text("\n".join(md), encoding="utf-8")
    with open(out_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
    with open(out_dir / "judge_passages.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(passage_rows[0].keys()) if passage_rows else ["passage_id"])
        w.writeheader()
        w.writerows(passage_rows)
    if answer_rows:
        with open(out_dir / "judge_answers.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(answer_rows[0].keys()))
            w.writeheader()
            w.writerows(answer_rows)

    failed = sum(1 for q in run["questions"] if q.get("failed"))
    print()
    print(f"answers failed           : {failed}")
    print(f"pooled passages to judge : {len(passage_rows)}")
    print(f"answers to judge         : {len(answer_rows)}")
    print(f"written to               : data/evaluation/{out_dir.name}/")
    print("Next: open review.md, fill the CSV files, then run score_evaluation.py")


if __name__ == "__main__":
    main()
