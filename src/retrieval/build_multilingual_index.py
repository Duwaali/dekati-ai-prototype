#!/usr/bin/env python3
# Builds an extra index so English questions can find Finnish text.


import json
import os
import random
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if "--root" in sys.argv:
    PROJECT_ROOT = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()

INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "index"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "index_multilingual"
FINAL = {"faiss": OUT_DIR / "faiss.index", "meta": OUT_DIR / "meta.json"}
BUILDING = {"faiss": OUT_DIR / "faiss.building.index", "meta": OUT_DIR / "meta.building.json"}

MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_WINDOW = 512
DIMENSIONS = 384
LANGUAGES = {"fi", "mixed"}
BATCH = 32

sys.path.insert(0, str(Path(__file__).resolve().parent))
from language import detect_language


# Prepare a passage the way the multilingual model expects.
def passage_input(header, text):
    body = f"{header}\n\n{text}" if header else text
    return "passage: " + body


# Print a message and stop without changing anything.
def stop(msg):
    print(msg)
    print("Nothing was overwritten or deleted.")
    sys.exit(1)


# Embed the Finnish chunks again with the multilingual model and save the index.
def main():
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    for p in FINAL.values():
        if p.exists():
            stop(f"{p.name} already exists in {OUT_DIR.name}/. Move it away to rebuild.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in BUILDING.values():
        if p.exists():
            stop(f"Leftover {p.name}. Move it away first.")

    print("=" * 60)
    print("MULTILINGUAL INDEX FOR FINNISH CONTENT  (numbers only)")
    print("=" * 60)
    started = time.time()

    db = sqlite3.connect(f"file:{INDEX_DIR / 'chunks.sqlite'}?mode=ro", uri=True)
    rows, inputs, langs = [], [], Counter()
    for row_id, header, text in db.execute(
            "SELECT row_id, embed_header, text FROM chunks WHERE embedded = 1 ORDER BY row_id"):
        lang = detect_language(text)
        langs[lang] += 1
        if lang in LANGUAGES:
            rows.append(row_id)
            inputs.append(passage_input(header, text))
    print(f"embedded chunks scanned : {sum(langs.values()):,}")
    for k in ("fi", "mixed", "en", "little text"):
        print(f"   {k:<12}: {langs[k]:,}")
    print(f"selected (fi + mixed)   : {len(rows):,}")
    if not rows:
        stop("No Finnish chunks found.")

    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MODEL_WINDOW
    tok = model.tokenizer
    lengths = [len(ids) for ids in tok(inputs, add_special_tokens=True, truncation=False,
                                         padding=False)["input_ids"]]
    over = sum(n > MODEL_WINDOW for n in lengths)
    print(f"longest input           : {max(lengths)} tokens (window {MODEL_WINDOW}); over window: {over:,}")
    if over:
        stop("Some inputs exceed the window; they would be truncated. Stopping.")

    vectors = model.encode(inputs, batch_size=BATCH, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False).astype("float32")
    norms = np.linalg.norm(vectors, axis=1)
    finite = bool(np.isfinite(vectors).all())
    unit = bool((np.abs(norms - 1) < 1e-3).all())

    index = faiss.IndexIDMap2(faiss.IndexFlatIP(DIMENSIONS))
    index.add_with_ids(vectors, np.asarray(rows, dtype="int64"))

    rng = random.Random(42)
    sample = rng.sample(range(len(rows)), min(200, len(rows)))
    found = 0
    for i in sample:
        _, ids = index.search(vectors[i:i + 1], 1)
        top = int(ids[0][0])
        j = rows.index(top) if top in rows else None
        found += top == rows[i] or (j is not None and np.allclose(vectors[j], vectors[i], atol=1e-6))

    checks = {
        "no input over window": over == 0,
        "vectors finite": finite,
        "vectors unit length": unit,
        "FAISS entries = selected chunks": index.ntotal == len(rows),
        "self-retrieval": found == len(sample),
    }
    for k, ok in checks.items():
        print(f"  {k:<34} {'PASS' if ok else 'FAIL'}")

    faiss.write_index(index, str(BUILDING["faiss"]))
    with open(BUILDING["meta"], "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME, "dimensions": DIMENSIONS, "model_window": MODEL_WINDOW,
            "languages": sorted(LANGUAGES), "total_vectors": len(rows),
            "language_counts_in_embedded_chunks": dict(langs),
            "longest_input_tokens": max(lengths),
            "self_retrieval": f"{found}/{len(sample)}",
            "checks": {k: "PASS" if ok else "FAIL" for k, ok in checks.items()},
            "elapsed_seconds": round(time.time() - started, 1),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    if not all(checks.values()):
        stop("A CHECK FAILED. Final files were NOT created. Building files are kept.")
    for key in ("faiss", "meta"):
        os.rename(BUILDING[key], FINAL[key])
    print(f"\nAll checks passed in {time.time() - started:,.0f} s. "
          f"Files in data/processed/index_multilingual/: faiss.index, meta.json")


if __name__ == "__main__":
    main()
