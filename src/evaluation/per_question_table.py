#!/usr/bin/env python3
# Writes the rating of every answer, per question, rater and model, as LaTeX table rows.
# Screen output: counts only. Notes and rater names are never read into the table.
#
# Usage: python src/evaluation/per_question_table.py author.csv staff.csv new_questions.csv out.tex

import csv
import sys
from collections import Counter
from pathlib import Path

LETTER = {"1": "C", "0.5": "P", "0": "W", "R": "R", "M": "M"}
COLUMNS = ("AL", "AQ", "SL", "SQ")


# Read one ratings file into {question_id: {column: letter}}.
def read(path, rater, grid):
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            model = "L" if row["model"].startswith("llama") else "Q"
            grid.setdefault(row["question_id"], {})[rater + model] = LETTER[row["correct"]]


def order(qid):
    return "QUN".index(qid[0]), int(qid[1:])


def main():
    if len(sys.argv) != 5:
        print(__doc__ or "usage: author.csv staff.csv new_questions.csv out.tex")
        return 1
    out = Path(sys.argv[4])
    if out.exists():
        print("STOP: output exists, nothing overwritten")
        return 1

    grid = {}
    read(sys.argv[1], "A", grid)
    read(sys.argv[2], "S", grid)
    read(sys.argv[3], "A", grid)

    lines, counts = [], Counter()
    agree = pairs = 0
    for qid in sorted(grid, key=order):
        cells = [grid[qid].get(c, "--") for c in COLUMNS]
        for c, v in zip(COLUMNS, cells):
            counts[qid[0], c, v] += 1
        if qid[0] == "Q":
            # A correct refusal is recorded as C by the author and R by the staff.
            for a, s in (("AL", "SL"), ("AQ", "SQ")):
                pairs += 1
                agree += grid[qid][a] == grid[qid][s] or {grid[qid][a], grid[qid][s]} == {"C", "R"}
        lines.append(f"{qid} & " + " & ".join(cells) + r" \\")
        if qid in ("Q8", "U12"):
            lines.append(r"\midrule")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"questions {len(grid)}  rows {len(lines)}")
    for key in sorted(counts):
        print(*key, counts[key])
    print(f"author-staff agreement on Q1-Q8: {agree} of {pairs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
