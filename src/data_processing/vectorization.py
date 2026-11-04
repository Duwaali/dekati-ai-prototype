#!/usr/bin/env python3
# Turns the chosen chunks into 384-number vectors (embeddings) that capture their meaning.


import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if "--root" in sys.argv:
    PROJECT_ROOT = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()

CHUNKED_DIR = PROJECT_ROOT / "data" / "processed" / "chunked"
VECTOR_DIR = PROJECT_ROOT / "data" / "processed" / "vectors"

CHUNKS_JSONL = CHUNKED_DIR / "chunks.jsonl"
CHUNKING_META = CHUNKED_DIR / "chunking_meta.json"

EMBEDDINGS_NPY = VECTOR_DIR / "embeddings.npy"
CHUNK_IDS_JSON = VECTOR_DIR / "chunk_ids.json"
META_JSON = VECTOR_DIR / "vectorization_meta.json"

ROW_IDS_NPY = VECTOR_DIR / "row_ids.npy"

BUILDING_NPY = VECTOR_DIR / "embeddings.building.npy"
BUILDING_IDS = VECTOR_DIR / "chunk_ids.building.json"
BUILDING_ROWS = VECTOR_DIR / "row_ids.building.npy"
BUILDING_META = VECTOR_DIR / "vectorization_meta.building.json"
PROGRESS_JSON = VECTOR_DIR / "vectorization_progress.json"

EXCLUDE_FROM_EMBEDDING = {("text", "measurement"), ("text", "header")}
SKIP_DUPLICATES = True


# Decide whether a chunk gets a vector (numeric data and duplicates do not).
def should_embed(chunk):
    kind = (chunk.get("source_type"), chunk.get("content_kind"))
    if kind in EXCLUDE_FROM_EMBEDDING:
        return False, f"{kind[0]}/{kind[1]} numeric data (keyword index only)"
    if SKIP_DUPLICATES and chunk.get("is_duplicate"):
        return False, "exact duplicate of an earlier chunk"
    return True, None

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_WINDOW = 256
DIMENSIONS = 384
HEADER_SEPARATOR = "\n\n"

BATCH_SIZE = 64
SAVE_EVERY = 40
NORMALIZE = True
VERIFY_BLOCK = 8192


# Count the lines in a file quickly.
def count_lines(path):
    total = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            total += block.count(b"\n")
    return total


# Read how many chunks the chunker reported.
def read_expected_total():
    if not CHUNKING_META.exists():
        return None
    with open(CHUNKING_META, "r", encoding="utf-8") as f:
        return json.load(f).get("total_chunks")


# Join the header and the chunk text into the text that is embedded.
def embedding_input(chunk):
    header = chunk.get("embed_header") or ""
    text = chunk.get("text") or ""
    return f"{header}{HEADER_SEPARATOR}{text}" if header else text


# Save a JSON file without leaving a half-written file if it crashes.
def write_json_safely(path, data):
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temp, path)


# Read how far an earlier run got, so it can continue.
def load_progress():
    if not PROGRESS_JSON.exists():
        return 0, 0
    try:
        with open(PROGRESS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("rows_done", 0)), int(data.get("inputs_over_window_so_far", 0))
    except Exception:
        return 0, 0


# Save how far the run has got.
def save_progress(rows_done, total, started, over_window):
    write_json_safely(PROGRESS_JSON, {
        "rows_done": rows_done,
        "total": total,
        "model": MODEL_NAME,
        "dimensions": DIMENSIONS,
        "normalized": NORMALIZE,
        "inputs_over_window_so_far": over_window,
        "elapsed_seconds_this_session": round(time.time() - started, 1),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


# Show seconds as an easy-to-read time.
def human_time(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


# Print a message and stop without changing anything.
def stop(message):
    print()
    print(message)
    print("Nothing was overwritten or deleted.")
    sys.exit(1)


# Embed all chosen chunks, check every vector and save the results.
def main():
    print("=" * 70)
    print("VECTORIZATION  -  chunks.jsonl -> embeddings.npy")
    print("=" * 70)

    import numpy as np
    from sentence_transformers import SentenceTransformer

    if not CHUNKS_JSONL.exists():
        stop("chunks.jsonl not found. Run token_chunking.py first.")

    for path in (EMBEDDINGS_NPY, CHUNK_IDS_JSON, ROW_IDS_NPY, META_JSON):
        if path.exists():
            stop(f"{path.name} already exists. Vectorization looks finished.\n"
                 f"To build again, move the old files out of data/processed/vectors/ yourself.")

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("STEP 1: Checking the chunk file")
    print("-" * 70)
    total_lines = count_lines(CHUNKS_JSONL)
    expected = read_expected_total()
    print(f"  chunks in chunks.jsonl     : {total_lines:,}")
    if expected is None:
        stop("chunking_meta.json missing or unreadable, cannot cross-check the count.")
    if expected != total_lines:
        stop(f"MISMATCH: chunking_meta.json says {expected:,}, file holds {total_lines:,}.")
    print(f"  chunking_meta.json agrees  : {expected:,}  PASS")

    print()
    print("STEP 1b: Choosing which chunks get a vector")
    print("-" * 70)
    row_ids, chunk_ids, excluded = [], [], {}
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            chunk = json.loads(line)
            if not chunk.get("chunk_id"):
                stop(f"line {line_index + 1} has no chunk_id")
            ok, reason = should_embed(chunk)
            if ok:
                row_ids.append(line_index)
                chunk_ids.append(chunk["chunk_id"])
            else:
                excluded[reason] = excluded.get(reason, 0) + 1
    total = len(row_ids)
    print(f"  chunks to embed            : {total:,}")
    for reason, n in sorted(excluded.items(), key=lambda x: -x[1]):
        print(f"  not embedded               : {n:,}  ({reason})")
    if total + sum(excluded.values()) != total_lines:
        stop("Selection does not add up to the chunk count.")
    selected = set(row_ids)

    rows_done, over_window_before = load_progress()
    if rows_done > 0:
        if not BUILDING_NPY.exists():
            stop("Progress file says work was done, but the partial vector file is missing.")
        partial = np.load(BUILDING_NPY, mmap_mode="r")
        if partial.shape != (total, DIMENSIONS):
            stop(f"Partial file has shape {partial.shape}, this corpus needs ({total}, {DIMENSIONS}).")
        del partial
        print(f"  resuming from row          : {rows_done:,} ({rows_done * 100 // total}% done)")
    elif BUILDING_NPY.exists():
        stop(f"{BUILDING_NPY.name} exists but there is no progress record.\n"
             f"Move it out of data/processed/vectors/ yourself, then run again.")

    print()
    print("STEP 2: Loading the model")
    print("-" * 70)
    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer
    if model.max_seq_length != MODEL_WINDOW:
        stop(f"Model window is {model.max_seq_length}, expected {MODEL_WINDOW}.")
    get_dimension = getattr(model, "get_embedding_dimension", None) or \
        model.get_sentence_embedding_dimension
    if get_dimension() != DIMENSIONS:
        stop("Model dimension is not 384.")
    print(f"  {MODEL_NAME}")
    print(f"  window {MODEL_WINDOW} tokens | {DIMENSIONS} dimensions  PASS")

    if rows_done == 0:
        vectors = np.lib.format.open_memmap(
            BUILDING_NPY, mode="w+", dtype="float32", shape=(total, DIMENSIONS)
        )
    else:
        vectors = np.lib.format.open_memmap(BUILDING_NPY, mode="r+")

    print()
    print("STEP 3: Encoding  (Ctrl+C is safe, run again to continue)")
    print("-" * 70)

    started = time.time()
    batch_texts, batch_rows = [], []
    batches_since_save = 0
    last_written = rows_done
    over_window = over_window_before
    stored_count_mismatch = 0

    # Embed the current batch of chunks and store the vectors.
    def flush():
        nonlocal batch_texts, batch_rows, batches_since_save, last_written
        if not batch_texts:
            return
        encoded = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            convert_to_numpy=True,
            normalize_embeddings=NORMALIZE,
            show_progress_bar=False,
        ).astype("float32")
        if encoded.shape != (len(batch_texts), DIMENSIONS):
            raise RuntimeError(f"model returned shape {encoded.shape}")
        vectors[batch_rows[0]:batch_rows[-1] + 1] = encoded
        last_written = batch_rows[-1] + 1
        batch_texts, batch_rows = [], []
        batches_since_save += 1

    try:
        with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
            row = 0
            for line_index, line in enumerate(f):
                if line_index not in selected:
                    continue
                current = row
                row += 1
                if current < rows_done:
                    continue

                chunk = json.loads(line)
                if chunk["chunk_id"] != chunk_ids[current]:
                    raise RuntimeError("chunk order changed since selection, stopping")
                text = embedding_input(chunk)

                n = len(tokenizer(text, add_special_tokens=True, truncation=False,
                                  padding=False)["input_ids"])
                if n > MODEL_WINDOW:
                    over_window += 1
                if "input_tokens" in chunk and chunk["input_tokens"] != n:
                    stored_count_mismatch += 1

                batch_texts.append(text)
                batch_rows.append(current)

                if len(batch_texts) >= BATCH_SIZE:
                    flush()
                    if batches_since_save >= SAVE_EVERY:
                        vectors.flush()
                        save_progress(last_written, total, started, over_window)
                        batches_since_save = 0
                        done_now = last_written - rows_done
                        rate = done_now / max(time.time() - started, 0.001)
                        left = (total - last_written) / max(rate, 0.001)
                        print(f"  {last_written:>9,}/{total:,}  {last_written * 100 / total:5.1f}%  "
                              f"{rate:6.1f} chunks/s  about {human_time(left)} left", flush=True)
        flush()
        vectors.flush()
        save_progress(last_written, total, started, over_window)

    except KeyboardInterrupt:
        flush()
        vectors.flush()
        save_progress(max(last_written, rows_done), total, started, over_window)
        print()
        print(f"Interrupted at row {max(last_written, rows_done):,} of {total:,}. Progress saved.")
        print("Run this again to continue. Nothing was lost.")
        sys.exit(130)

    elapsed = time.time() - started

    print()
    print("STEP 4: Verifying every vector")
    print("-" * 70)
    del vectors
    check = np.load(BUILDING_NPY, mmap_mode="r")

    checks = {
        "shape": check.shape == (total, DIMENSIONS),
        "dtype float32": str(check.dtype) == "float32",
        "chunk id count": len(chunk_ids) == total,
        "chunk ids unique": len(set(chunk_ids)) == len(chunk_ids),
        "no input over model window": over_window == 0,
        "token counts match chunker": stored_count_mismatch == 0,
    }

    bad_finite = bad_norm = 0
    for start in range(0, total, VERIFY_BLOCK):
        block = np.asarray(check[start:start + VERIFY_BLOCK])
        bad_finite += int((~np.isfinite(block)).any(axis=1).sum())
        norms = np.linalg.norm(block, axis=1)
        bad_norm += int((np.abs(norms - 1.0) > 1e-3).sum())
    checks["all vectors finite"] = bad_finite == 0
    checks["all vectors unit length"] = bad_norm == 0
    del check

    for name, ok in checks.items():
        print(f"  {name:<30} {'PASS' if ok else 'FAIL'}")

    if not all(checks.values()):
        print()
        print("VERIFICATION FAILED. Final files were NOT created.")
        print(f"  vectors not finite     : {bad_finite:,}")
        print(f"  vectors not unit length: {bad_norm:,}")
        print(f"  inputs over window     : {over_window:,}")
        print("The partial work is kept, nothing was deleted.")
        sys.exit(1)

    print()
    print("STEP 5: Writing the final files")
    print("-" * 70)
    with open(BUILDING_IDS, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False, separators=(",", ":"))
    with open(BUILDING_IDS, "r", encoding="utf-8") as f:
        if json.load(f) != chunk_ids:
            stop("chunk_ids file did not read back correctly.")
    with open(BUILDING_ROWS, "wb") as f:
        np.save(f, np.asarray(row_ids, dtype="int64"))
    if np.load(BUILDING_ROWS).tolist() != row_ids:
        stop("row_ids file did not read back correctly.")

    with open(BUILDING_META, "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME,
            "model_window": MODEL_WINDOW,
            "dimensions": DIMENSIONS,
            "total_chunks": total_lines,
            "total_vectors": total,
            "not_embedded": excluded,
            "selection_policy": {
                "excluded_kinds": sorted("/".join(k) for k in EXCLUDE_FROM_EMBEDDING),
                "skip_duplicates": SKIP_DUPLICATES,
                "note": "not-embedded chunks remain in SQLite and the BM25 keyword index",
            },
            "normalized": NORMALIZE,
            "embedded_input": "embed_header + '\\n\\n' + text",
            "inputs_over_window": over_window,
            "checks": {k: "PASS" for k in checks},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds_last_session": round(elapsed, 1),
        }, f, indent=2)

    for building, final in ((BUILDING_NPY, EMBEDDINGS_NPY),
                            (BUILDING_IDS, CHUNK_IDS_JSON),
                            (BUILDING_ROWS, ROW_IDS_NPY),
                            (BUILDING_META, META_JSON)):
        if final.exists():
            stop(f"{final.name} appeared during the run, refusing to overwrite it.")
        os.rename(building, final)

    print(f"  {EMBEDDINGS_NPY.name:<26} {EMBEDDINGS_NPY.stat().st_size / (1024 * 1024):>8.0f} MB")
    print(f"  {CHUNK_IDS_JSON.name:<26} {CHUNK_IDS_JSON.stat().st_size / (1024 * 1024):>8.1f} MB")
    print()
    print("=" * 70)
    print("VECTORIZATION COMPLETE")
    print("=" * 70)
    print(f"vectors           : {total:,} x {DIMENSIONS}")
    print(f"inputs over window: {over_window:,}  (measured with the model's tokenizer)")
    print(f"time this session : {human_time(elapsed)}")
    print("Next: build the FAISS index from embeddings.npy and chunk_ids.json")


if __name__ == "__main__":
    main()
