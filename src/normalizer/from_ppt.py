# Converts extracted presentations into the common document format, one section per slide.

from pathlib import Path
import json
import hashlib


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PPT_JSON = PROJECT_ROOT / "data" / "processed" / "ppt" / "ppt_extracted_documents.json"

NORMALIZED_FOLDER = PROJECT_ROOT / "data" / "processed" / "normalized"
NORMALIZED_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = NORMALIZED_FOLDER / "ppt_normalized.json"


# Write a table as readable text rows.
def table_to_text(table):
    lines = []
    for row in table:
        cells = [str(cell) if cell is not None else "" for cell in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


# Turn one presentation into the common format, one section per slide.
def normalize_document(doc):
    source_file = doc.get("source", "unknown.pptx")
    metadata = doc.get("metadata") or {}

    doc_id = hashlib.md5(source_file.encode("utf-8")).hexdigest()[:8]

    normalized = {
        "doc_id": doc_id,
        "source_file": source_file,
        "source_type": "ppt",
        "origin": doc.get("source_type", "company_data"),
        "title": source_file,
        "author": "",
        "date": metadata.get("processed_date") or "",
        "parent_doc_id": None,
        "sections": []
    }

    for slide in doc.get("slides", []):
        slide_number = slide.get("slide_number")

        parts = []

        for text_block in slide.get("text") or []:
            cleaned = (text_block or "").strip()
            if cleaned:
                parts.append(cleaned)

        for table in slide.get("tables") or []:
            table_text = table_to_text(table)
            if table_text.strip():
                parts.append("[TABLE]\n" + table_text)

        for chart_lines in slide.get("charts") or []:
            if chart_lines:
                parts.append("[CHART]\n" + "\n".join(chart_lines))

        notes = (slide.get("notes") or "").strip()
        if notes:
            parts.append("[SPEAKER NOTES]\n" + notes)

        full_text = "\n\n".join(parts)

        if not full_text.strip():
            continue

        normalized["sections"].append({
            "section_id": f"{doc_id}_s{slide_number}",
            "heading": slide.get("title") or "",
            "text": full_text,
            "location": f"slide {slide_number}",
            "images": slide.get("images") or []
        })

    return normalized


with open(PPT_JSON, "r", encoding="utf-8") as f:
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
print("PPT normalization completed")
print("==============================")
print("Documents normalized  :", len(normalized_documents))
print("Total sections created:", total_sections)
print("Documents with no text:", len(empty_docs))

print("\nDocuments by origin:")
for origin, count in by_origin.items():
    print(f"  {origin:25} {count}")

if empty_docs:
    print("\nDocuments with no extractable text (likely image-only):")
    for name in empty_docs:
        print(" -", name)

print("\nSaved to:", OUTPUT_FILE)
