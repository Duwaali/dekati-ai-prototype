#!/usr/bin/env python3
# Calculates the scores from the ratings CSV downloaded from the comparison page.
# Screen output: numbers only. Notes and rater names are never printed.
#
# Usage: python src/evaluation/score_comparison.py answer_comparison_ratings.csv [more.csv ...]

import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

CORRECT_SCORE = {"1": 1.0, "0.5": 0.5, "0": 0.0, "M": 0.0}


# Read one or more ratings files.
def read(paths):
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8-sig") as f:
            rows += list(csv.DictReader(f))
    return rows


# Count the ratings of one model in one question set.
def summarise(rows):
    rated = [r for r in rows if r["correct"]]
    scores = [CORRECT_SCORE[r["correct"]] for r in rated if r["correct"] in CORRECT_SCORE]
    grounded = [r for r in rated if r["grounded"] in ("Y", "N")]
    useful = [r for r in rated if r["useful"] in ("0", "1", "2")]
    secs = [float(r["answer_seconds"]) for r in rows if r["answer_seconds"]]
    return {
        "answers": len(rows),
        "rated": len(rated),
        "correct": sum(r["correct"] == "1" for r in rated),
        "partly": sum(r["correct"] == "0.5" for r in rated),
        "wrong": sum(r["correct"] == "0" for r in rated),
        "correct_refusal": sum(r["correct"] == "R" for r in rated),
        "missed_answer": sum(r["correct"] == "M" for r in rated),
        "correctness_mean": round(sum(scores) / len(scores), 2) if scores else None,
        "grounded_yes": sum(r["grounded"] == "Y" for r in grounded),
        "grounded_no": sum(r["grounded"] == "N" for r in grounded),
        "useful_2": sum(r["useful"] == "2" for r in useful),
        "useful_1": sum(r["useful"] == "1" for r in useful),
        "useful_0": sum(r["useful"] == "0" for r in useful),
        "usefulness_mean": round(statistics.mean(int(r["useful"]) for r in useful), 2) if useful else None,
        "chosen_better": sum(r["better"] == "yes" for r in rows),
        "median_answer_seconds": round(statistics.median(secs), 1) if secs else None,
    }


def main():
    if len(sys.argv) < 2:
        print("usage: python src/evaluation/score_comparison.py ratings.csv [more.csv ...]")
        sys.exit(1)
    rows = read(sys.argv[1:])
    by = defaultdict(list)
    for r in rows:
        by[(r["set"], r["model"])].append(r)

    # Self-checks: every question has exactly two answers, one per slot, and the
    # preference of the two answers agrees (one "yes" and one "no", or both "same").
    per_q = defaultdict(list)
    for r in rows:
        per_q[(r.get("rated_by", ""), r["set"], r["question_id"])].append(r)
    pairs_ok = all(len(v) == 2 and {x["slot"] for x in v} == {"A", "B"} for v in per_q.values())
    pref_ok = all(sorted(x["better"] for x in v) in (["", ""], ["same", "same"], ["no", "yes"]) for v in per_q.values())

    result = {"files": len(sys.argv) - 1, "raters": len({r.get("rated_by", "") for r in rows}),
              "rows": len(rows), "sets": {}}
    for (s, m), rs in sorted(by.items()):
        result["sets"].setdefault(f"set {s}", {})[m] = summarise(rs)
    for s in result["sets"]:
        same = sum(1 for k, v in per_q.items() if k[1] == s.split()[1] and v[0]["better"] == "same")
        result["sets"][s]["_about_the_same"] = same
    result["checks"] = {"two_answers_per_question": "PASS" if pairs_ok else "FAIL",
                        "preferences_consistent": "PASS" if pref_ok else "FAIL"}

    out = Path(sys.argv[1]).with_name(f"comparison_scores_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "x", encoding="utf-8") as f:   # never overwrite
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"written to: {out.name}")


if __name__ == "__main__":
    main()
