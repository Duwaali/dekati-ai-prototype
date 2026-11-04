#!/usr/bin/env python3
# Prints safe numbers (counts and sizes only) about the processed data.


import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FOLDERS = ["normalized", "chunked", "vectors"]

KNOWN_FILES = {
    "normalized": [
        "pdf_normalized.json",
        "word_normalized.json",
        "excel_normalized.json",
        "ppt_normalized.json",
        "email_normalized.json",
        "text_normalized.json",
        "ocr_normalized.json",
    ],
    "chunked": [
        "chunks.json",
        "chunks.jsonl",
    ],
    "vectors": [
        "embeddings.npy",
        "chunk_ids.json",
        "vectorization_meta.json",
    ],
}


# Turn bytes into megabytes.
def mb(size_bytes):
    return size_bytes / (1024 * 1024)


# Count the lines in a file quickly.
def count_newlines(path):
    total = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            total += block.count(b"\n")
    return total


# Count the valid records in a JSON-lines file.
def count_jsonl_records(path):
    records = 0
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                json.loads(line)
                records += 1
            except json.JSONDecodeError:
                bad += 1
    return records, bad


# Print how many files a folder has and how big they are.
def report_folder(processed_dir, name):
    folder = processed_dir / name
    print(f"\n[{name}]")

    if not folder.exists():
        print("  folder exists : NO")
        return

    files = [p for p in folder.rglob("*") if p.is_file()]
    total_size = sum(p.stat().st_size for p in files)
    print(f"  folder exists : yes")
    print(f"  files         : {len(files):,}")
    print(f"  total size    : {mb(total_size):,.1f} MB")

    known = KNOWN_FILES.get(name, [])
    for filename in known:
        path = folder / filename
        if path.exists():
            print(f"  {filename:<26} present  {mb(path.stat().st_size):>10,.1f} MB")
        else:
            print(f"  {filename:<26} missing")

    other = len([p for p in files if p.parent == folder and p.name not in known])
    nested = len([p for p in files if p.parent != folder])
    print(f"  other files   : {other:,} (names not shown)")
    if nested:
        print(f"  files in sub-folders: {nested:,} (names not shown)")


# Count the chunks in two ways and compare.
def report_chunks(processed_dir):
    path = processed_dir / "chunked" / "chunks.jsonl"
    print("\n[chunk count]")
    if not path.exists():
        print("  chunks.jsonl missing, nothing to count")
        return None

    lines = count_newlines(path)
    records, bad = count_jsonl_records(path)

    print(f"  newline count      : {lines:,}")
    print(f"  valid JSON records : {records:,}")
    print(f"  invalid lines      : {bad:,}")

    ok = bad == 0 and abs(lines - records) <= 1
    print(f"  two counts agree   : {'PASS' if ok else 'FAIL'}")
    return records


KNOWN_DOCUMENTS = 8_672
KNOWN_SECTIONS = 166_711
KNOWN_WORDS = 23_343_579

CROSS_CHECK_MAX_MB = 100


# Read documents one at a time to save memory.
def iter_documents(path):
    import ijson

    with open(path, "rb") as f:
        first = b""
        while True:
            byte = f.read(1)
            if not byte:
                return
            if not byte.isspace():
                first = byte
                break
        f.seek(0)

        prefix = "item" if first == b"[" else "documents.item"
        for document in ijson.items(f, prefix, use_float=True):
            yield document


# Count the words in one section.
def section_words(section):
    text = section.get("text") if isinstance(section, dict) else None
    if text is None:
        return 0
    return len(str(text).split())


# Count documents, sections and words in one file.
def count_normalized_file(path):
    documents = 0
    sections = 0
    empty_sections = 0
    words = 0
    max_sections_in_doc = 0

    for document in iter_documents(path):
        documents += 1
        doc_sections = document.get("sections") or []
        sections += len(doc_sections)
        max_sections_in_doc = max(max_sections_in_doc, len(doc_sections))
        for section in doc_sections:
            n = section_words(section)
            words += n
            if n == 0:
                empty_sections += 1

    return {
        "documents": documents,
        "sections": sections,
        "empty_sections": empty_sections,
        "words": words,
        "max_sections_in_doc": max_sections_in_doc,
    }


# Count again in a second way, to double-check.
def cross_check_full_load(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = data if isinstance(data, list) else data.get("documents", [])
    sections = sum(len(d.get("sections") or []) for d in docs)
    words = sum(
        section_words(s) for d in docs for s in (d.get("sections") or [])
    )
    return len(docs), sections, words


# Print the document, section and word counts.
def report_normalized_counts(processed_dir):
    folder = processed_dir / "normalized"
    print("\n[normalized contents]")

    try:
        import ijson
    except ImportError:
        print("  ijson not installed, cannot count safely")
        return

    totals = {"documents": 0, "sections": 0, "empty_sections": 0, "words": 0}
    all_checks_pass = True

    print(f"  {'file':<24}{'docs':>9}{'sections':>11}{'empty':>9}"
          f"{'words':>13}  cross-check")

    for filename in KNOWN_FILES["normalized"]:
        path = folder / filename
        if not path.exists():
            print(f"  {filename:<24} missing")
            all_checks_pass = False
            continue

        counts = count_normalized_file(path)
        for key in totals:
            totals[key] += counts[key]

        if path.stat().st_size / (1024 * 1024) <= CROSS_CHECK_MAX_MB:
            same = cross_check_full_load(path) == (
                counts["documents"], counts["sections"], counts["words"]
            )
            check = "PASS" if same else "FAIL"
            all_checks_pass = all_checks_pass and same
        else:
            check = "skipped (too big)"

        print(f"  {filename:<24}{counts['documents']:>9,}"
              f"{counts['sections']:>11,}{counts['empty_sections']:>9,}"
              f"{counts['words']:>13,}  {check}")

    print(f"  {'TOTAL':<24}{totals['documents']:>9,}"
          f"{totals['sections']:>11,}{totals['empty_sections']:>9,}"
          f"{totals['words']:>13,}")

    print()
    print("  compared with known project facts:")
    for label, measured, known in [
        ("documents", totals["documents"], KNOWN_DOCUMENTS),
        ("sections", totals["sections"], KNOWN_SECTIONS),
        ("words", totals["words"], KNOWN_WORDS),
    ]:
        result = "MATCH" if measured == known else "DIFFERENT"
        print(f"    {label:<10} measured {measured:>12,}  "
              f"known {known:>12,}  {result}")
    print(f"  streaming vs full-load counts: "
          f"{'PASS' if all_checks_pass else 'FAIL'}")


# Print the shape of the vector file.
def report_vectors(processed_dir, chunk_count):
    path = processed_dir / "vectors" / "embeddings.npy"
    print("\n[vector count]")
    if not path.exists():
        print("  embeddings.npy missing, nothing to count")
        return

    try:
        import numpy as np
    except ImportError:
        print("  numpy not installed, cannot read shape")
        return

    array = np.load(path, mmap_mode="r")
    print(f"  shape : {array.shape}")
    print(f"  dtype : {array.dtype}")

    if chunk_count is not None and len(array.shape) == 2:
        ok = array.shape[0] == chunk_count
        print(f"  vectors match chunks : {'PASS' if ok else 'FAIL'}")


# Print all the numbers.
def main():
    root = PROJECT_ROOT
    if "--root" in sys.argv:
        root = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()

    processed_dir = root / "data" / "processed"

    print("=" * 60)
    print("DATA STATISTICS  (numbers only, read only)")
    print("=" * 60)

    if not processed_dir.exists():
        print("data/processed does not exist")
        sys.exit(1)

    for name in FOLDERS:
        report_folder(processed_dir, name)

    if "--quick" not in sys.argv:
        report_normalized_counts(processed_dir)

    chunk_count = report_chunks(processed_dir)
    report_vectors(processed_dir, chunk_count)
    print()


if __name__ == "__main__":
    main()
