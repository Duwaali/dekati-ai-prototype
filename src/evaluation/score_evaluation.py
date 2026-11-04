#!/usr/bin/env python3
# Turns the filled-in review sheets into evaluation scores.


import csv
import json
import statistics
import sys
import time
from pathlib import Path


# Read a filled-in review sheet.
def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# Read a rating as a number, and stop if it is empty.
def as_number(value, field, row_id):
    value = (value or "").strip().replace(",", ".")
    if value == "":
        raise SystemExit(f"STOP: {field} is empty for {row_id}. Fill every row first.")
    return float(value)


# Calculate the scores from the review sheets.
def main():
    if len(sys.argv) < 2:
        print("usage: python src/evaluation/score_evaluation.py data/evaluation/run_<timestamp> [--answers judgements.json]")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    judged = None
    if "--answers" in sys.argv:
        judged = json.loads(Path(sys.argv[sys.argv.index("--answers") + 1]).read_text(encoding="utf-8"))["answers"]

    passages_path = run_dir / "judge_passages.csv"
    if "--passages" in sys.argv:   # e.g. judge_passages_rated.csv from make_judging_page.py
        passages_path = Path(sys.argv[sys.argv.index("--passages") + 1])
    passages = read_csv(passages_path)
    passages_judged = any((r["relevant"] or "").strip() for r in passages)
    relevant_rows = {}
    for row in passages if passages_judged else []:
        if as_number(row["relevant"], "relevant", row["passage_id"]) >= 1:
            relevant_rows.setdefault(row["question_id"], set()).add(int(row["row_id"]))

    scored = [q for q in run["questions"] if q["id"] in relevant_rows]
    metrics = {"questions": len(run["questions"]),
               "questions_with_relevant_passage": len(scored),
               "retrieval": {}}

    if not passages_judged:
        metrics["retrieval"] = "passages not judged yet"
    # --top K: only the top K of each method were judged, so score hit@1, hit@K and MRR@K.
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 10
    cutoffs = sorted({1, top} if top < 10 else {1, 5, 10})
    metrics["judged_depth"] = top
    for method in ("keyword", "semantic", "hybrid") if passages_judged else ():
        hits = {k: 0 for k in cutoffs}
        rr = []
        for q in scored:
            ranks = q["ranks"][method]
            good = relevant_rows.get(q["id"], set())
            first = next((i for i, r in enumerate(ranks[:top], start=1) if r in good), None)
            for k in hits:
                if first is not None and first <= k:
                    hits[k] += 1
            rr.append(1.0 / first if first else 0.0)
        n = max(len(scored), 1)
        metrics["retrieval"][method] = {f"hit@{k}": round(hits[k] / n, 3) for k in cutoffs}
        metrics["retrieval"][method][f"mrr@{top}"] = round(sum(rr) / n, 3)

    answers_path = run_dir / "judge_answers.csv"
    if answers_path.exists() and "--retrieval-only" not in sys.argv:
        rows = read_csv(answers_path)
        for r in rows if judged else []:
            r.update({k: str(judged[r["question_id"]][k]) for k in ("answer_in_corpus", "correct", "grounded")})
        for r in rows:
            r["answer_in_corpus"] = str(int(as_number(r["answer_in_corpus"], "answer_in_corpus", r["question_id"])))
        in_corpus = [r for r in rows if r["answer_in_corpus"] == "1"]
        not_in = [r for r in rows if r["answer_in_corpus"] == "0"]
        correct = [as_number(r["correct"], "correct", r["question_id"]) for r in in_corpus]
        grounded = [as_number(r["grounded"], "grounded", r["question_id"]) for r in rows]
        metrics["answers"] = {
            "judged": len(rows),
            "correctness_mean": round(sum(correct) / max(len(correct), 1), 3),
            "fully_correct": sum(1 for c in correct if c >= 1),
            "partly_correct": sum(1 for c in correct if 0 < c < 1),
            "grounded_share": round(sum(grounded) / max(len(grounded), 1), 3),
            "citation_check_pass": sum(int(r["auto_citations_ok"]) for r in rows),
            "control_questions": len(not_in),
            "correct_refusals": sum(int(r["said_not_found"]) for r in not_in),
        }
        gen = [q["generation_seconds"] for q in run["questions"] if "generation_seconds" in q]
        ret = [q["retrieval_seconds"] for q in run["questions"] if "retrieval_seconds" in q]
        if gen:
            metrics["answers"]["median_answer_seconds"] = round(statistics.median(gen), 1)
            metrics["answers"]["median_retrieval_seconds"] = round(statistics.median(ret), 2)

    out = run_dir / f"metrics_{time.strftime('%Y%m%d_%H%M%S')}.json"
    if out.exists():
        raise SystemExit(f"STOP: {out.name} already exists. Nothing was overwritten.")
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
