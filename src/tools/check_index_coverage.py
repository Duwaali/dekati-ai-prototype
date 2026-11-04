#!/usr/bin/env python3
# Checks that every chunk can be found by at least one kind of search.


import os
import random
import re
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if "--root" in sys.argv:
    PROJECT_ROOT = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()
INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "index"
SAMPLE = 300
if "--sample" in sys.argv:
    SAMPLE = int(sys.argv[sys.argv.index("--sample") + 1])

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))


# Check that every chunk can be found, and test how findable they are.
def main():
    import faiss
    import numpy as np
    from search import HybridSearcher

    db = sqlite3.connect(f"file:{INDEX_DIR / 'chunks.sqlite'}?mode=ro", uri=True)
    total = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    embedded_rows = np.array([r[0] for r in db.execute(
        "SELECT row_id FROM chunks WHERE embedded = 1 ORDER BY row_id")], dtype="int64")

    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    faiss_ids = np.sort(faiss.vector_to_array(index.id_map).astype("int64"))

    print("=" * 60)
    print("SEARCH COVERAGE CHECK  (numbers only)")
    print("=" * 60)
    print(f"chunks in database       : {total:,}")
    print(f"chunks with a vector     : {len(embedded_rows):,}")
    ids_match = np.array_equal(faiss_ids, embedded_rows)
    print(f"FAISS ids = embedded rows: {'PASS' if ids_match else 'FAIL'}")

    print("\n1. Indexed (all chunks)")
    word = re.compile(r"[^\W_]")
    own_text = np.zeros(total, dtype=bool)
    header_only = np.zeros(total, dtype=bool)
    for row, text, header in db.execute("SELECT row_id, text, embed_header FROM chunks"):
        if word.search(text or ""):
            own_text[row] = True
        elif word.search(header or ""):
            header_only[row] = True
    fts_rows = db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    has_vector = np.zeros(total, dtype=bool)
    has_vector[embedded_rows] = True
    in_fts = own_text | header_only
    reachable = in_fts | has_vector
    print(f"   rows in keyword index   : {fts_rows:,}  {'PASS' if fts_rows == total else 'FAIL'}")
    print(f"   searchable by own words : {int(own_text.sum()):,}")
    print(f"   searchable only by header (document name, location): {int(header_only.sum()):,}")
    print(f"   in FAISS                : {int(has_vector.sum()):,}")
    print(f"   reachable somehow       : {int(reachable.sum()):,} of {total:,}")
    unreachable = total - int(reachable.sum())
    print(f"   in neither (unreachable): {unreachable:,}  {'PASS' if unreachable == 0 else 'FAIL'}")

    print(f"\n2. Findable (random sample of {SAMPLE} per group, top 50)")
    searcher = HybridSearcher(INDEX_DIR)
    rng = random.Random(42)

    # Find all chunks with the same text.
    def same_text_rows(row):
        c = searcher.fetch(row)
        canonical = c["duplicate_of"] or c["chunk_id"]
        return {r[0] for r in db.execute(
            "SELECT row_id FROM chunks WHERE chunk_id = ? OR duplicate_of = ?",
            (canonical, canonical))}

    emb_sample = rng.sample(embedded_rows.tolist(), min(SAMPLE, len(embedded_rows)))
    found = 0
    for row in emb_sample:
        hits = {r for r, _ in searcher.semantic(searcher.fetch(row)["text"])}
        found += bool(hits & same_text_rows(row))
    print(f"   vector chunks, semantic search : {found}/{len(emb_sample)} "
          f"({found * 100 / max(len(emb_sample), 1):.1f}%)")

    db.execute("CREATE VIRTUAL TABLE temp.vrow USING fts5vocab(main, 'chunks_fts', 'row')")
    tokenize = re.compile(r"[^\W_]+")
    kw_pool = np.flatnonzero(~has_vector & own_text)
    kw_sample = rng.sample(kw_pool.tolist(), min(SAMPLE, len(kw_pool)))
    found = 0
    for row in kw_sample:
        terms = sorted(set(m.group().lower() for m in tokenize.finditer(searcher.fetch(row)["text"])))
        rarity = []
        for term in terms:
            r = db.execute("SELECT doc FROM temp.vrow WHERE term = ?", (term,)).fetchone()
            if r:
                rarity.append((r[0], term))
        query = " ".join(term for _, term in sorted(rarity)[:8])
        hits = {r for r, _ in searcher.keyword(query)}
        found += bool(hits & same_text_rows(row))
    print(f"   keyword-only chunks, BM25      : {found}/{len(kw_sample)} "
          f"({found * 100 / max(len(kw_sample), 1):.1f}%)")
    print("\n(1) must be 0 unreachable. (2) is an estimate of how findable chunks are.")


if __name__ == "__main__":
    main()
