# Converts extracted Word documents into the common format, one section per heading.

from pathlib import Path
import json
import hashlib


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WORD_JSON = PROJECT_ROOT / "data" / "processed" / "word" / "word_extracted_documents.json"

NORMALIZED_FOLDER = PROJECT_ROOT / "data" / "processed" / "normalized"
NORMALIZED_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = NORMALIZED_FOLDER / "word_normalized.json"

MAX_SECTION_CHARS = 3000


# Write a table as readable text rows.
def table_to_text(table):
    lines = []
    for row in table:
        cells = [str(cell) if cell is not None else "" for cell in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


# Turn one Word document into the common format, starting a section at each heading.
def normalize_document(doc):
    source_file = doc.get("file_name", "unknown.docx")
    metadata = doc.get("metadata") or {}

    doc_id = hashlib.md5(source_file.encode("utf-8")).hexdigest()[:8]

    normalized = {
        "doc_id": doc_id,
        "source_file": source_file,
        "source_type": "word",
        "origin": doc.get("source_type", "company_data"),
        "title": metadata.get("title") or source_file,
        "author": metadata.get("author") or "",
        "date": metadata.get("created") or "",
        "parent_doc_id": None,
        "sections": []
    }

    section_counter = 0
    current_heading = None
    buffer = []

    # Close the current section and start a new one.
    def flush():
        nonlocal section_counter, buffer, current_heading

        text = "\n\n".join(buffer).strip()

        if not text:
            buffer = []
            return

        section_counter += 1

        heading = current_heading or ""
        location = f"section {section_counter}"
        if heading:
            location = f"section {section_counter} - {heading}"

        normalized["sections"].append({
            "section_id": f"{doc_id}_s{section_counter}",
            "heading": heading,
            "text": text,
            "location": location,
            "images": []
        })

        buffer = []

    for block in doc.get("content", []):
        block_heading = block.get("heading") or ""

        if block_heading != current_heading:
            flush()
            current_heading = block_heading

        if block.get("type") == "table":
            table_text = table_to_text(block.get("data") or [])
            if table_text.strip():
                buffer.append("[TABLE]\n" + table_text)
        else:
            text = (block.get("text") or "").strip()
            if text:
                buffer.append(text)

        if sum(len(part) for part in buffer) > MAX_SECTION_CHARS:
            flush()

    flush()


    textboxes = [t for t in (doc.get("textboxes") or []) if t.strip()]

    if textboxes:
        section_counter += 1
        normalized["sections"].append({
            "section_id": f"{doc_id}_s{section_counter}",
            "heading": "Text boxes",
            "text": "\n\n".join(textboxes),
            "location": "text boxes",
            "images": []
        })


    comments = [c for c in (doc.get("comments") or []) if c.strip()]

    if comments:
        section_counter += 1
        normalized["sections"].append({
            "section_id": f"{doc_id}_s{section_counter}",
            "heading": "Comments",
            "text": "\n\n".join(comments),
            "location": "comments",
            "images": []
        })

    normalized["document_images"] = doc.get("images") or []

    return normalized


with open(WORD_JSON, "r", encoding="utf-8") as f:
    raw_documents = json.load(f)

normalized_documents = [normalize_document(doc) for doc in raw_documents]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(normalized_documents, f, indent=4, ensure_ascii=False)

total_sections = sum(len(d["sections"]) for d in normalized_documents)
empty_docs = [d["source_file"] for d in normalized_documents if not d["sections"]]

by_origin = {}
for d in normalized_documents:
    key = d.get("origin", "unknown")
    by_origin[key] = by_origin.get(key, 0) + 1

print("==============================")
print("WORD normalization completed")
print("==============================")
print("Documents normalized:", len(normalized_documents))
print("Total sections created:", total_sections)
print("Documents with no text:", len(empty_docs))

print("\nDocuments by origin:")
for origin, count in by_origin.items():
    print(f"  {origin:25} {count}")

if empty_docs:
    print("\nEmpty documents:")
    for name in empty_docs[:10]:
        print(" -", name)

print("\nSaved to:", OUTPUT_FILE)
