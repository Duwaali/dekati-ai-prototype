# Converts extracted text files into the common format, keeping headers and data apart.

from pathlib import Path
import json
import hashlib
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEXT_JSON = PROJECT_ROOT / "data" / "processed" / "text" / "text_extracted_documents.json"

NORMALIZED_FOLDER = PROJECT_ROOT / "data" / "processed" / "normalized"
NORMALIZED_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = NORMALIZED_FOLDER / "text_normalized.json"

MAX_SECTION_CHARS = 3000


# Check if a line is just a row of numbers.
def is_numeric_line(line):
    stripped = line.strip()

    if not stripped:
        return False

    if "=" in stripped:
        label = stripped.split("=", 1)[0]
        if any(c.isalpha() for c in label):
            return False

    if stripped.startswith("[") and any(c.isalpha() for c in stripped):
        return False

    letters = re.sub(r"[eE](?=[+-]?\d)", "", stripped)
    letter_count = sum(1 for c in letters if c.isalpha())

    return letter_count / len(stripped) < 0.05


# Separate the readable header from the number rows.
def split_header_and_data(content):
    lines = content.split("\n")

    header_lines = []
    data_lines = []

    in_data = False
    numeric_streak = 0

    for line in lines:
        if not in_data:
            if is_numeric_line(line):
                numeric_streak += 1
                if numeric_streak >= 5:
                    in_data = True
                    data_lines.extend(header_lines[-numeric_streak:])
                    header_lines = header_lines[:-numeric_streak]
                    data_lines.append(line)
                    continue
            else:
                numeric_streak = 0

            header_lines.append(line)

        else:
            data_lines.append(line)

    return "\n".join(header_lines).strip(), "\n".join(data_lines).strip()


# Split long text at line breaks so rows stay whole.
def split_text(text, max_chars):
    if len(text) <= max_chars:
        return [text]

    parts = []
    buffer = []
    length = 0

    for line in text.split("\n"):
        if length + len(line) > max_chars and buffer:
            parts.append("\n".join(buffer))
            buffer = []
            length = 0

        buffer.append(line)
        length += len(line) + 1

    if buffer:
        parts.append("\n".join(buffer))

    return parts


# Turn one text file into the common document format.
def normalize_document(doc):
    source_file = doc.get("file_name", "unknown.txt")

    doc_id = hashlib.md5(source_file.encode("utf-8")).hexdigest()[:8]

    normalized = {
        "doc_id": doc_id,
        "source_file": source_file,
        "source_type": "text",
        "file_extension": doc.get("file_extension", ""),
        "origin": doc.get("source_type", "company_data"),
        "title": source_file,
        "author": "",
        "date": doc.get("processed_date") or "",
        "parent_doc_id": None,
        "sections": []
    }

    section_counter = 0

    # Add one section to the document.
    def add_section(text, location, heading="", content_kind="text"):
        nonlocal section_counter

        if not text.strip():
            return

        section_counter += 1

        normalized["sections"].append({
            "section_id": f"{doc_id}_s{section_counter}",
            "heading": heading,
            "text": text,
            "location": location,
            "content_kind": content_kind,
            "images": []
        })

    content = (doc.get("content") or "").strip()

    if not content:
        return normalized

    structure = doc.get("structure", "plain")


    if structure in ("xml", "csv", "xml-unparsed", "csv-unparsed"):
        pieces = split_text(content, MAX_SECTION_CHARS)

        for index, piece in enumerate(pieces, start=1):
            location = f"{structure} content"
            if len(pieces) > 1:
                location += f", part {index} of {len(pieces)}"

            add_section(
                text=f"File: {source_file}\n\n{piece}",
                location=location,
                heading=source_file,
                content_kind="labelled"
            )

        return normalized


    header, data = split_header_and_data(content)

    if header:
        pieces = split_text(header, MAX_SECTION_CHARS)

        for index, piece in enumerate(pieces, start=1):
            location = "file header"
            if len(pieces) > 1:
                location += f", part {index} of {len(pieces)}"

            add_section(
                text=f"File: {source_file}\n\n{piece}",
                location=location,
                heading=source_file,
                content_kind="header"
            )

    if data:
        pieces = split_text(data, MAX_SECTION_CHARS)

        for index, piece in enumerate(pieces, start=1):
            add_section(
                text=piece,
                location=f"measurement data, part {index} of {len(pieces)}",
                heading=source_file,
                content_kind="measurement"
            )

    return normalized


with open(TEXT_JSON, "r", encoding="utf-8") as f:
    raw_documents = json.load(f)

normalized_documents = [normalize_document(doc) for doc in raw_documents]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(normalized_documents, f, indent=4, ensure_ascii=False)

total_sections = sum(len(d["sections"]) for d in normalized_documents)
empty_docs = [d["source_file"] for d in normalized_documents if not d["sections"]]

by_kind = {}
for d in normalized_documents:
    for s in d["sections"]:
        key = s.get("content_kind", "unknown")
        by_kind[key] = by_kind.get(key, 0) + 1

by_ext = {}
for d in normalized_documents:
    key = d.get("file_extension", "unknown")
    by_ext[key] = by_ext.get(key, 0) + 1

header_chars = sum(
    len(s["text"]) for d in normalized_documents
    for s in d["sections"] if s.get("content_kind") == "header"
)
measurement_chars = sum(
    len(s["text"]) for d in normalized_documents
    for s in d["sections"] if s.get("content_kind") == "measurement"
)
labelled_chars = sum(
    len(s["text"]) for d in normalized_documents
    for s in d["sections"] if s.get("content_kind") == "labelled"
)

print("==============================")
print("TEXT normalization completed")
print("==============================")
print("Documents normalized  :", len(normalized_documents))
print("Total sections created:", f"{total_sections:,}")
print("Documents with no text:", len(empty_docs))

print("\nSections by kind:")
for kind, count in sorted(by_kind.items(), key=lambda x: -x[1]):
    print(f"  {kind:15} {count:,}")

print("\nDocuments by extension:")
for ext, count in sorted(by_ext.items(), key=lambda x: -x[1]):
    print(f"  {ext:8} {count}")

print("\nCharacters:")
print(f"  header (searchable)   : {header_chars:,}")
print(f"  labelled (searchable) : {labelled_chars:,}")
print(f"  measurement (numeric) : {measurement_chars:,}")

print("\nSaved to:", OUTPUT_FILE)
