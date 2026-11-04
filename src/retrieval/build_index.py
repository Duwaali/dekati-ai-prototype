#!/usr/bin/env python3
# Builds the search indexes: SQLite with keyword search for all chunks, FAISS for the vectors.


import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if "--root" in sys.argv:
    PROJECT_ROOT = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()

PROCESSED = PROJECT_ROOT / "data" / "processed"
CHUNKS_JSONL = PROCESSED / "chunked" / "chunks.jsonl"
EMBEDDINGS_NPY = PROCESSED / "vectors" / "embeddings.npy"
CHUNK_IDS_JSON = PROCESSED / "vectors" / "chunk_ids.json"
ROW_IDS_NPY = PROCESSED / "vectors" / "row_ids.npy"
CHUNKING_META = PROCESSED / "chunked" / "chunking_meta.json"

INDEX_DIR = PROCESSED / "index"
SQLITE_FINAL = INDEX_DIR / "chunks.sqlite"
FAISS_FINAL = INDEX_DIR / "faiss.index"
META_FINAL = INDEX_DIR / "index_meta.json"

SQLITE_BUILDING = INDEX_DIR / "chunks.building.sqlite"
FAISS_BUILDING = INDEX_DIR / "faiss.building.index"
META_BUILDING = INDEX_DIR / "index_meta.building.json"

SELF_CHECK_SAMPLE = 200
INSERT_BATCH = 2000

COLUMNS = [
    "row_id", "chunk_id", "doc_id", "section_id", "chunk_index",
    "source_type", "source_file", "parent_source_file", "title", "origin",
    "author", "date", "heading", "location", "embed_header", "text",
    "is_duplicate", "duplicate_of", "is_boilerplate", "text_tokens",
    "content_kind", "embedded",
]


# Print a message and stop without changing anything.
def stop(message):
    print()
    print(message)
    print("Nothing was overwritten or deleted.")
    sys.exit(1)


# Turn a value into text for the database.
def as_text(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


# Create the database tables and the keyword index.
def create_sqlite(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute(f"""
        CREATE TABLE chunks (
            row_id INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE NOT NULL,
            doc_id TEXT, section_id TEXT, chunk_index INTEGER,
            source_type TEXT, source_file TEXT, parent_source_file TEXT,
            title TEXT, origin TEXT, author TEXT, date TEXT,
            heading TEXT, location TEXT, embed_header TEXT, text TEXT,
            is_duplicate INTEGER, duplicate_of TEXT, is_boilerplate INTEGER,
            text_tokens INTEGER,
            content_kind TEXT, embedded INTEGER
        )
    """)
    db.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text, embed_header,
            content='chunks', content_rowid='row_id',
            tokenize='unicode61'
        )
    """)
    return db


# Build the database, the keyword index and the vector index, then check them.
def main():
    import numpy as np
    import faiss

    print("=" * 70)
    print("BUILD SEARCH INDEXES  (SQLite + BM25 + FAISS)")
    print("=" * 70)

    for path in (CHUNKS_JSONL, EMBEDDINGS_NPY, CHUNK_IDS_JSON, ROW_IDS_NPY, CHUNKING_META):
        if not path.exists():
            stop(f"Missing {path.name}. Run token_chunking.py and vectorization.py first.")
    for path in (SQLITE_FINAL, FAISS_FINAL, META_FINAL):
        if path.exists():
            stop(f"{path.name} already exists. Move it out of data/processed/index/ to rebuild.")
    for path in (SQLITE_BUILDING, FAISS_BUILDING, META_BUILDING):
        if path.exists():
            stop(f"Leftover {path.name} from an earlier run. Move it away first.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    vectors = np.load(EMBEDDINGS_NPY, mmap_mode="r")
    with open(CHUNK_IDS_JSON, "r", encoding="utf-8") as f:
        chunk_ids = json.load(f)
    row_ids = np.load(ROW_IDS_NPY)
    with open(CHUNKING_META, "r", encoding="utf-8") as f:
        total_chunks = json.load(f)["total_chunks"]
    total = len(chunk_ids)
    print(f"chunks         : {total_chunks:,}")
    print(f"vectors        : {vectors.shape[0]:,} x {vectors.shape[1]}")
    print(f"chunk ids      : {total:,} | row ids: {len(row_ids):,}")
    if not (vectors.shape[0] == total == len(row_ids)):
        stop("Vector, chunk id and row id counts differ.")
    if len(row_ids) and (np.any(np.diff(row_ids) <= 0) or row_ids[-1] >= total_chunks):
        stop("row_ids are not strictly increasing line numbers inside chunks.jsonl.")

    print("\nSTEP 1: SQLite + BM25 keyword index")
    db = create_sqlite(SQLITE_BUILDING)
    order_mismatch = 0
    rows = 0
    v = 0
    batch = []
    placeholders = ",".join("?" * len(COLUMNS))

    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            embedded = v < total and row_ids[v] == rows
            if embedded:
                if chunk["chunk_id"] != chunk_ids[v]:
                    order_mismatch += 1
                v += 1
            record = dict(chunk)
            record["row_id"] = rows
            record["embedded"] = int(embedded)
            record["is_duplicate"] = int(bool(chunk.get("is_duplicate")))
            record["is_boilerplate"] = int(bool(chunk.get("is_boilerplate")))
            batch.append(tuple(as_text(record.get(c)) if c not in
                               ("row_id", "chunk_index", "is_duplicate",
                                "is_boilerplate", "text_tokens", "embedded")
                               else record.get(c) for c in COLUMNS))
            rows += 1
            if len(batch) >= INSERT_BATCH:
                db.executemany(f"INSERT INTO chunks VALUES ({placeholders})", batch)
                batch.clear()
    if batch:
        db.executemany(f"INSERT INTO chunks VALUES ({placeholders})", batch)
    db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
    db.execute("CREATE INDEX idx_doc ON chunks(doc_id)")
    db.execute("CREATE INDEX idx_dup ON chunks(duplicate_of)")
    db.commit()

    sqlite_rows = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    sqlite_embedded = db.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 1").fetchone()[0]
    fts_rows = db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    db.close()
    print(f"  rows written : {sqlite_rows:,} | keyword index rows: {fts_rows:,}")

    print("\nSTEP 2: FAISS vector index")
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
    block = 20000
    for start in range(0, total, block):
        index.add_with_ids(np.ascontiguousarray(vectors[start:start + block], dtype="float32"),
                           np.ascontiguousarray(row_ids[start:start + block], dtype="int64"))
    faiss.write_index(index, str(FAISS_BUILDING))
    print(f"  vectors in FAISS: {index.ntotal:,}")

    print("\nSTEP 3: Self-retrieval check")
    rng = random.Random(42)
    sample = rng.sample(range(total), min(SELF_CHECK_SAMPLE, total))
    self_found = 0
    for i in sample:
        query = np.ascontiguousarray(vectors[i:i + 1], dtype="float32")
        scores, ids = index.search(query, 1)
        top = int(ids[0][0])
        j = int(np.searchsorted(row_ids, top))
        if top == row_ids[i] or (j < total and row_ids[j] == top
                                 and np.allclose(vectors[j], vectors[i], atol=1e-6)):
            self_found += 1
    print(f"  found itself : {self_found}/{len(sample)}")

    checks = {
        "chunk ids match row ids": order_mismatch == 0 and v == total,
        "SQLite rows = all chunks": sqlite_rows == total_chunks == rows,
        "keyword index rows = all chunks": fts_rows == total_chunks,
        "SQLite embedded rows = vectors": sqlite_embedded == total,
        "FAISS entries = vectors": index.ntotal == total,
        "self-retrieval": self_found == len(sample),
    }
    print()
    for name, ok in checks.items():
        print(f"  {name:<36} {'PASS' if ok else 'FAIL'}")

    with open(META_BUILDING, "w", encoding="utf-8") as f:
        json.dump({
            "total_chunks": total_chunks,
            "total_vectors": total,
            "dimensions": int(vectors.shape[1]),
            "faiss_index": "IndexIDMap2(IndexFlatIP): exact cosine on unit vectors, ids = row_id",
            "keyword_index": "SQLite FTS5, BM25 ranking, over text + embed_header, all chunks",
            "self_retrieval": f"{self_found}/{len(sample)}",
            "checks": {k: "PASS" if ok else "FAIL" for k, ok in checks.items()},
            "elapsed_seconds": round(time.time() - started, 1),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    if not all(checks.values()):
        print("\nA CHECK FAILED. Final files were NOT created. Building files are kept.")
        sys.exit(1)

    for building, final in ((SQLITE_BUILDING, SQLITE_FINAL),
                            (FAISS_BUILDING, FAISS_FINAL),
                            (META_BUILDING, META_FINAL)):
        if final.exists():
            stop(f"{final.name} appeared during the run, refusing to overwrite it.")
        os.rename(building, final)

    print(f"\nAll checks passed in {time.time() - started:,.0f} s. "
          f"Files in data/processed/index/: chunks.sqlite, faiss.index, index_meta.json")


if __name__ == "__main__":
    main()
