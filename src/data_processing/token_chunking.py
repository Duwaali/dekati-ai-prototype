#!/usr/bin/env python3
# Cuts every section into chunks of at most 204 model tokens, without losing any text.


import hashlib
import json
import os
import re
import sys
import time
from array import array
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MODEL_WINDOW = 256
HEADER_BUDGET = 48
SPECIAL_ALLOWANCE = 4
TEXT_BUDGET = MODEL_WINDOW - HEADER_BUDGET - SPECIAL_ALLOWANCE

HEADER_SEPARATOR = "\n\n"

HEADER_FIELDS = [
    ("Document", 16),
    ("Type", 4),
    ("Location", 8),
    ("Section", 10),
    ("From", 8),
    ("To", 8),
    ("Date", 8),
    ("Content", 6),
]

NORMALIZED_FILES = [
    "ppt_normalized.json",
    "word_normalized.json",
    "excel_normalized.json",
    "ocr_normalized.json",
    "pdf_normalized.json",
    "email_normalized.json",
    "text_normalized.json",
]

KNOWN_DOCUMENTS = 8_672
KNOWN_SECTIONS = 166_711
KNOWN_SOURCE_WORDS = 21_575_929

FLAT_DOC_FIELDS = [
    "source_type",
    "title",
    "source_file",
    "origin",
    "author",
    "date",
    "parent_doc_id",
    "parent_source_file",
]

BOILERPLATE_MAX_WORDS = 15
BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*copyright\b", re.IGNORECASE),
    re.compile(r"^\s*©"),
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE),
    re.compile(r"^\s*www\.\S+\s*$", re.IGNORECASE),
]

WORD_PATTERN = re.compile(r"\S+")


# Load the embedding model's tokenizer.
def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = 10**9
    return tokenizer


# Count the tokens in a text, without extra start/end tokens.
def count_tokens(tokenizer, text):
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
    )


# Count the tokens exactly as the model will see them.
def count_model_input_tokens(tokenizer, text):
    return len(
        tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
    )


# Find where each token starts and ends in the text.
def token_offsets(tokenizer, text):
    return tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["offset_mapping"]


# Turn a value into one clean line of text.
def clean_one_line(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


# Return the first field that has a value.
def first_available(obj, fields):
    for field in fields:
        value = obj.get(field)
        if value not in (None, "", []):
            return value
    return None


# Check if a chunk is boilerplate, like a copyright line or page number.
def is_boilerplate(text):
    cleaned = clean_one_line(text)
    if not cleaned or len(cleaned.split()) > BOILERPLATE_MAX_WORDS:
        return False
    return any(p.search(cleaned) for p in BOILERPLATE_PATTERNS)


# Pick a percentile value from a sorted list.
def percentile(sorted_values, fraction):
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


# Shorten a text so it fits a token limit, cutting between tokens.
def cap_to_tokens(tokenizer, text, max_tokens):
    if not text:
        return "", False
    offsets = token_offsets(tokenizer, text)
    if len(offsets) <= max_tokens:
        return text, False
    keep = max_tokens
    while keep > 0:
        candidate = text[: offsets[keep - 1][1]].rstrip()
        if count_tokens(tokenizer, candidate) <= max_tokens:
            return candidate, True
        keep -= 1
    return "", True


# Collect the values for the provenance header (document, location, heading...).
def header_values(document, section):
    source_type = clean_one_line(document.get("source_type"))

    if source_type == "ocr_image":
        title = first_available(document, ["parent_source_file", "title", "source_file"])
    else:
        title = first_available(document, ["title", "source_file"])

    values = {
        "Document": title,
        "Type": source_type,
        "Location": first_available(section, ["location", "page", "sheet", "slide"]),
        "Section": first_available(section, ["heading", "title", "name"]),
        "Content": first_available(section, ["content_kind", "kind", "type"]),
    }

    if source_type == "email":
        values["From"] = first_available(document, ["sender", "author", "from_email"])
        values["To"] = first_available(document, ["recipient", "to", "to_email"])
        values["Date"] = document.get("date")

    return {k: clean_one_line(v) for k, v in values.items() if clean_one_line(v)}


# Build the short provenance header that is embedded with each chunk.
def build_embed_header(tokenizer, document, section):
    values = header_values(document, section)
    parts = []
    fields_cut = []
    fields_left_out = []

    for label, cap in HEADER_FIELDS:
        value = values.get(label)
        if not value:
            continue
        value, was_cut = cap_to_tokens(tokenizer, value, cap)
        if not value:
            fields_left_out.append(label)
            continue
        candidate = " | ".join(parts + [f"{label}: {value}"])
        if count_tokens(tokenizer, candidate) <= HEADER_BUDGET:
            parts.append(f"{label}: {value}")
            if was_cut:
                fields_cut.append(label)
        else:
            fields_left_out.append(label)

    header = " | ".join(parts)
    header_tokens = count_tokens(tokenizer, header) if header else 0

    if header_tokens > HEADER_BUDGET:
        raise RuntimeError("header over budget after capping, this is a bug")

    return header, header_tokens, fields_cut, fields_left_out


# Group the tokens by the word they belong to.
def word_spans_with_tokens(tokenizer, text):
    words = [[m.start(), m.end(), []] for m in WORD_PATTERN.finditer(text)]
    offsets = token_offsets(tokenizer, text)

    w = 0
    for start, end in offsets:
        if end <= start:
            continue
        while w < len(words) and words[w][1] <= start:
            w += 1
        if w == len(words) or start < words[w][0]:
            raise RuntimeError("token outside every word, tokenizer assumption broken")
        while end > words[w][1]:
            if w + 1 == len(words):
                raise RuntimeError("token runs past the end of the text")
            words[w][1] = words[w + 1][1]
            words[w][2].extend(words[w + 1][2])
            del words[w + 1]
        words[w][2].append((start, end))

    return [(s, e, toks) for s, e, toks in words]


# Split a word that is too long for one chunk into pieces at token boundaries.
def split_long_word(tokenizer, text, start, end, tokens, budget):
    pieces = []
    piece_start = start
    i = 0

    while i < len(tokens):
        take = min(budget, len(tokens) - i)
        while True:
            last = i + take
            piece_end = end if last == len(tokens) else tokens[last][0]
            piece_text = text[piece_start:piece_end]
            n = count_tokens(tokenizer, piece_text)
            if n <= budget:
                break
            take -= 1
            if take == 0:
                raise RuntimeError("cannot fit a single token into the budget")
        pieces.append((piece_start, piece_end, n))
        piece_start = piece_end
        i += take

    return pieces


# Cut one section into chunks that fit the token budget.
def chunk_section_text(tokenizer, text, budget=TEXT_BUDGET):
    words = word_spans_with_tokens(tokenizer, text)
    chunks = []

    current = []
    current_tokens = 0

    # Close the current chunk, moving words to the next one if it is too long.
    def emit(word_list):
        carry = []
        while word_list:
            s, e = word_list[0][0], word_list[-1][1]
            n = count_tokens(tokenizer, text[s:e])
            if n <= budget:
                chunks.append({
                    "char_start": s,
                    "char_end": e,
                    "text_tokens": n,
                    "split_word_part": None,
                })
                return carry
            if len(word_list) == 1:
                raise RuntimeError("single word over budget reached emit(), this is a bug")
            carry.insert(0, word_list.pop())
        return carry

    for s, e, toks in words:
        n = len(toks)

        if n > budget:
            pending = current
            while pending:
                pending = emit(pending)
            current, current_tokens = [], 0

            pieces = split_long_word(tokenizer, text, s, e, toks, budget)
            for number, (ps, pe, pn) in enumerate(pieces, start=1):
                chunks.append({
                    "char_start": ps,
                    "char_end": pe,
                    "text_tokens": pn,
                    "split_word_part": (number, len(pieces)),
                })
            continue

        if current and current_tokens + n > budget:
            carry = emit(current)
            current = carry
            current_tokens = sum(len(w[2]) for w in current)

        current.append((s, e, toks))
        current_tokens += n

    while current:
        current = emit(current)

    return chunks


# Check that the chunks rebuild the section text exactly, with nothing missing.
def verify_section_coverage(text, chunks):
    if not text.strip():
        return not chunks

    stripped_start = len(text) - len(text.lstrip())
    stripped_end = len(text.rstrip())

    if not chunks:
        return False
    if chunks[0]["char_start"] != stripped_start:
        return False
    if chunks[-1]["char_end"] != stripped_end:
        return False

    rebuilt = []
    previous_end = stripped_start
    for chunk in chunks:
        s, e = chunk["char_start"], chunk["char_end"]
        if e <= s or s < previous_end:
            return False
        gap = text[previous_end:s]
        if gap.strip():
            return False
        if chunk["split_word_part"] and chunk["split_word_part"][0] > 1 and gap:
            return False
        rebuilt.append(gap)
        rebuilt.append(text[s:e])
        previous_end = e

    if "".join(rebuilt) != text[stripped_start:stripped_end]:
        return False

    return "".join(rebuilt).split() == text.split()


# Read documents one at a time, so the whole file never has to fit in memory.
def iter_documents(path):
    import ijson

    with open(path, "rb") as f:
        head = f.read(4096).lstrip()[:1]
    prefix = "item" if head == b"[" else "documents.item"

    with open(path, "rb") as f:
        for document in ijson.items(f, prefix, use_float=True):
            yield document


# Make values safe to write as JSON.
def json_safe(value):
    return json.loads(json.dumps(value, default=str))


# Handles the output files: write to temporary names, rename only at the end.
class OutputFiles:
    NAMES = ["chunks.jsonl", "documents.jsonl", "empty_sections.jsonl", "chunking_meta.json"]

    # Prepare the temporary and final file names.
    def __init__(self, folder):
        self.folder = folder
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.building = {
            name: folder / f"{name}.building_{self.stamp}" for name in self.NAMES
        }
        self.final = {name: folder / name for name in self.NAMES}

    # Stop if output files already exist, so nothing is overwritten.
    def check_no_final_exists(self):
        existing = [name for name, path in self.final.items() if path.exists()]
        if existing:
            print("STOP: these output files already exist and will not be overwritten:")
            for name in existing:
                print(f"  {name}")
            print("Nothing was written. Move them yourself if you want a new run.")
            sys.exit(1)

    # Give the finished files their final names.
    def publish(self):
        self.check_no_final_exists()
        for name in self.NAMES:
            os.rename(self.building[name], self.final[name])


# Go through all normalized files and write the chunks.
def process_corpus(root, tokenizer):
    normalized_dir = root / "data" / "processed" / "normalized"
    out_dir = root / "data" / "processed" / "chunked"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [normalized_dir / name for name in NORMALIZED_FILES]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"STOP: {len(missing)} normalized file(s) missing: {missing}")
        sys.exit(1)

    files = OutputFiles(out_dir)
    files.check_no_final_exists()

    stats = {
        "documents": 0,
        "sections": 0,
        "empty_sections": 0,
        "source_words": 0,
        "chunks": 0,
        "split_long_words": 0,
        "split_word_pieces": 0,
        "duplicates_marked": 0,
        "boilerplate_marked": 0,
        "headers_with_cut_field": 0,
        "headers_with_left_out_field": 0,
        "coverage_failures": 0,
    }
    per_source = defaultdict(lambda: {"documents": 0, "sections": 0, "chunks": 0})
    text_tokens_all = array("H")
    input_tokens_all = array("H")
    header_tokens_all = array("H")

    doc_ids = set()
    section_ids = set()
    chunk_ids = set()
    first_chunk_for_hash = {}

    started = time.time()

    with open(files.building["chunks.jsonl"], "w", encoding="utf-8") as chunks_out, \
         open(files.building["documents.jsonl"], "w", encoding="utf-8") as docs_out, \
         open(files.building["empty_sections.jsonl"], "w", encoding="utf-8") as empty_out:
        for path in paths:
            file_label = path.name.replace("_normalized.json", "")
            print(f"  {file_label:<6} ...", end="", flush=True)
            file_chunks_before = stats["chunks"]

            for document in iter_documents(path):
                document = json_safe(document)
                doc_id = document.get("doc_id")
                if not doc_id:
                    raise RuntimeError(f"{path.name}: a document has no doc_id")
                if doc_id in doc_ids:
                    raise RuntimeError(f"{path.name}: duplicate doc_id found")
                doc_ids.add(doc_id)

                stats["documents"] += 1
                per_source[file_label]["documents"] += 1

                doc_record = {k: v for k, v in document.items() if k != "sections"}
                docs_out.write(json.dumps(doc_record, ensure_ascii=False) + "\n")

                flat = {f: document[f] for f in FLAT_DOC_FIELDS if f in document}

                for section in document.get("sections") or []:
                    stats["sections"] += 1
                    per_source[file_label]["sections"] += 1

                    section_id = section.get("section_id")
                    if not section_id:
                        raise RuntimeError(f"{path.name}: a section has no section_id")
                    if section_id in section_ids:
                        raise RuntimeError(f"{path.name}: duplicate section_id found")
                    section_ids.add(section_id)

                    raw = section.get("text")
                    text = "" if raw is None else str(raw)
                    stats["source_words"] += len(text.split())

                    section_meta = {k: v for k, v in section.items() if k != "text"}

                    if not text.strip():
                        stats["empty_sections"] += 1
                        empty_out.write(json.dumps({
                            **section_meta,
                            "doc_id": doc_id,
                            "source_type": document.get("source_type"),
                            "reason": "empty_text",
                        }, ensure_ascii=False) + "\n")
                        continue

                    header, header_tokens, cut, left_out = build_embed_header(
                        tokenizer, document, section
                    )
                    if cut:
                        stats["headers_with_cut_field"] += 1
                    if left_out:
                        stats["headers_with_left_out_field"] += 1

                    pieces = chunk_section_text(tokenizer, text)

                    if not verify_section_coverage(text, pieces):
                        stats["coverage_failures"] += 1
                        raise RuntimeError("section coverage check failed, stopping")

                    split_words_here = sum(
                        1 for p in pieces
                        if p["split_word_part"] and p["split_word_part"][0] == 1
                    )
                    stats["split_long_words"] += split_words_here
                    stats["split_word_pieces"] += sum(1 for p in pieces if p["split_word_part"])

                    section_word_total = len(text.split())

                    for index, piece in enumerate(pieces):
                        chunk_text = text[piece["char_start"]:piece["char_end"]]
                        chunk_id = f"{doc_id}_s{section_id}_c{index}"
                        if chunk_id in chunk_ids:
                            raise RuntimeError("duplicate chunk_id, stopping")
                        chunk_ids.add(chunk_id)

                        embed_input = f"{header}{HEADER_SEPARATOR}{chunk_text}" if header else chunk_text
                        input_tokens = count_model_input_tokens(tokenizer, embed_input)
                        if input_tokens > MODEL_WINDOW:
                            raise RuntimeError("a chunk is over the model window, stopping")

                        digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                        duplicate_of = first_chunk_for_hash.get(digest)
                        if duplicate_of is None:
                            first_chunk_for_hash[digest] = chunk_id
                        else:
                            stats["duplicates_marked"] += 1

                        boilerplate = is_boilerplate(chunk_text)
                        if boilerplate:
                            stats["boilerplate_marked"] += 1

                        chunk = {
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "section_id": section_id,
                            "chunk_index": index,
                            "total_chunks_in_section": len(pieces),
                            "text": chunk_text,
                            "embed_header": header,
                            "char_start": piece["char_start"],
                            "char_end": piece["char_end"],
                            "word_count": len(chunk_text.split()),
                            "section_word_count": section_word_total,
                            "text_tokens": piece["text_tokens"],
                            "header_tokens": header_tokens,
                            "input_tokens": input_tokens,
                            "split_word_part": piece["split_word_part"],
                            "header_fields_cut": cut,
                            "header_fields_left_out": left_out,
                            "text_hash": digest,
                            "is_duplicate": duplicate_of is not None,
                            "duplicate_of": duplicate_of,
                            "is_boilerplate": boilerplate,
                            "section_metadata": section_meta,
                            **flat,
                        }
                        for field in ("heading", "location", "content_kind"):
                            if section.get(field) is not None:
                                chunk[field] = section[field]

                        chunks_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

                        stats["chunks"] += 1
                        per_source[file_label]["chunks"] += 1
                        text_tokens_all.append(piece["text_tokens"])
                        input_tokens_all.append(input_tokens)
                        header_tokens_all.append(header_tokens)

            print(f" {stats['chunks'] - file_chunks_before:,} chunks")

    elapsed = time.time() - started
    return files, stats, dict(per_source), text_tokens_all, input_tokens_all, header_tokens_all, elapsed


# Re-read the written chunks and check every one again.
def validate_output(tokenizer, chunks_path, expected_chunks):
    lines = 0
    ids = set()
    mismatched_counts = 0
    over_window = 0
    over_text_budget = 0
    batch_inputs, batch_expected = [], []

    # Re-count a batch of chunks and compare with the stored counts.
    def check_batch():
        nonlocal mismatched_counts
        encoded = tokenizer(
            batch_inputs,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        for ids_, expected in zip(encoded, batch_expected):
            if len(ids_) != expected:
                mismatched_counts += 1
        batch_inputs.clear()
        batch_expected.clear()

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            lines += 1
            chunk = json.loads(line)
            ids.add(chunk["chunk_id"])
            if chunk["input_tokens"] > MODEL_WINDOW:
                over_window += 1
            if chunk["text_tokens"] > TEXT_BUDGET:
                over_text_budget += 1
            header = chunk["embed_header"]
            text = chunk["text"]
            batch_inputs.append(f"{header}{HEADER_SEPARATOR}{text}" if header else text)
            batch_expected.append(chunk["input_tokens"])
            if len(batch_inputs) >= 256:
                check_batch()
    if batch_inputs:
        check_batch()

    return {
        "lines": lines,
        "lines_match": lines == expected_chunks,
        "unique_ids": len(ids) == lines,
        "batched_equals_individual": mismatched_counts == 0,
        "batched_mismatches": mismatched_counts,
        "over_window": over_window,
        "over_text_budget": over_text_budget,
    }


# Turn a check result into PASS or FAIL.
def pass_fail(ok):
    return "PASS" if ok else "FAIL"


# Run the chunker from start to finish and print the results.
def main():
    root = PROJECT_ROOT
    if "--root" in sys.argv:
        root = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()
    real_corpus = root == PROJECT_ROOT

    print("=" * 70)
    print("TOKEN-BASED CHUNKING")
    print("=" * 70)
    print(f"window {MODEL_WINDOW} | header max {HEADER_BUDGET} | "
          f"specials {SPECIAL_ALLOWANCE} | text max {TEXT_BUDGET} tokens")
    print()

    tokenizer = load_tokenizer()

    print("Chunking:")
    (files, stats, per_source, text_tokens, input_tokens,
     header_tokens, elapsed) = process_corpus(root, tokenizer)

    print()
    print("Validating the written file (re-tokenizing every chunk in batches)...")
    v = validate_output(tokenizer, files.building["chunks.jsonl"], stats["chunks"])

    st, it, ht = sorted(text_tokens), sorted(input_tokens), sorted(header_tokens)

    # Summarise a list of numbers (min, median, max...).
    def dist(values):
        if not values:
            return {}
        return {
            "min": values[0],
            "median": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "max": values[-1],
            "mean": round(sum(values) / len(values), 1),
        }

    checks = {
        "every section fully covered": stats["coverage_failures"] == 0,
        "no chunk over model window": v["over_window"] == 0,
        "no chunk text over budget": v["over_text_budget"] == 0,
        "line count matches": v["lines_match"],
        "chunk ids unique": v["unique_ids"],
        "batched = individual token counts": v["batched_equals_individual"],
    }
    if real_corpus:
        checks["documents = 8,672"] = stats["documents"] == KNOWN_DOCUMENTS
        checks["sections = 166,711"] = stats["sections"] == KNOWN_SECTIONS
        checks["source words = 21,575,929"] = stats["source_words"] == KNOWN_SOURCE_WORDS

    meta = {
        "pipeline": "token_chunking",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "model_window": MODEL_WINDOW,
        "header_budget": HEADER_BUDGET,
        "special_allowance": SPECIAL_ALLOWANCE,
        "text_budget": TEXT_BUDGET,
        "overlap_tokens": 0,
        "header_fields": HEADER_FIELDS,
        "total_chunks": stats["chunks"],
        "stats": stats,
        "per_source": per_source,
        "text_tokens": dist(st),
        "input_tokens": dist(it),
        "header_tokens": dist(ht),
        "validation": v,
        "checks": {k: pass_fail(ok) for k, ok in checks.items()},
        "elapsed_seconds": round(elapsed, 1),
        "policies": {
            "duplicates": "mark_only",
            "boilerplate": "mark_only",
            "empty_sections": "kept in empty_sections.jsonl",
            "ocr_header_name": "parent_source_file when available",
            "long_words": "split at token boundaries, pieces rebuild the word exactly",
        },
    }
    with open(files.building["chunking_meta.json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print()
    print("=" * 70)
    print("RESULTS  (numbers only)")
    print("=" * 70)
    print(f"documents          : {stats['documents']:,}")
    print(f"sections           : {stats['sections']:,}  (empty kept: {stats['empty_sections']:,})")
    print(f"source words       : {stats['source_words']:,}")
    print(f"chunks             : {stats['chunks']:,}")
    for label in sorted(per_source):
        p = per_source[label]
        print(f"   {label:<6} {p['documents']:>6,} docs {p['sections']:>8,} sections {p['chunks']:>8,} chunks")
    print(f"long words split   : {stats['split_long_words']:,}  "
          f"(into {stats['split_word_pieces']:,} pieces)")
    print(f"duplicates marked  : {stats['duplicates_marked']:,}")
    print(f"boilerplate marked : {stats['boilerplate_marked']:,}")
    print(f"headers shortened  : {stats['headers_with_cut_field']:,}")
    print(f"headers missing a field (did not fit): {stats['headers_with_left_out_field']:,}")
    for name, values in (("text tokens", st), ("header tokens", ht), ("model input tokens", it)):
        d = dist(values)
        if d:
            print(f"{name:<19}: min {d['min']} | median {d['median']} | "
                  f"p95 {d['p95']} | max {d['max']} | mean {d['mean']}")
    print(f"time               : {elapsed:,.0f} s")
    print()
    print("CHECKS")
    for name, ok in checks.items():
        print(f"  {name:<36} {pass_fail(ok)}")

    if not all(checks.values()):
        print()
        print("A CHECK FAILED. Final files were NOT created.")
        print("The *.building files are left in place for inspection. Nothing was deleted.")
        sys.exit(1)

    files.publish()
    print()
    print("All checks passed. Files created in data/processed/chunked/:")
    for name in OutputFiles.NAMES:
        print(f"  {name}")


if __name__ == "__main__":
    main()
