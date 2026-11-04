# Converts extracted PDFs into the common document format, one section per page.

from pathlib import Path
import json
import hashlib


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_JSON = PROJECT_ROOT / "data" / "processed" / "pdf" / "pdf_extracted_documents.json"

NORMALIZED_FOLDER = PROJECT_ROOT / "data" / "processed" / "normalized"
NORMALIZED_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = NORMALIZED_FOLDER / "pdf_normalized.json"


# Write a table as readable text rows.
def table_to_text(table):
    lines = []
    for row in table:
        cells = [str(cell) if cell is not None else "" for cell in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


# Turn one PDF into the common format, one section per page.
def normalize_document(doc):
    source_file = doc.get("source", "unknown.pdf")
    metadata = doc.get("metadata") or {}

    doc_id = hashlib.md5(source_file.encode("utf-8")).hexdigest()[:8]

    normalized = {
        "doc_id": doc_id,
        "source_file": source_file,
        "source_type": "pdf",
        "origin": doc.get("source_type", "company_data"),
        "title": metadata.get("title") or source_file,
        "author": metadata.get("author") or "",
        "date": metadata.get("creationDate") or "",
        "parent_doc_id": None,
        "sections": []
    }

    for page in doc.get("pages", []):
        page_number = page.get("page_number")

        parts = []
        page_text = (page.get("text") or "").strip()
        if page_text:
            parts.append(page_text)

        for table in page.get("tables") or []:
            table_text = table_to_text(table)
            if table_text.strip():
                parts.append("[TABLE]\n" + table_text)

        full_text = "\n\n".join(parts)

        if not full_text.strip():
            continue

        section = {
            "section_id": f"{doc_id}_p{page_number}",
            "heading": "",
            "text": full_text,
            "location": f"page {page_number}",
            "images": page.get("images") or []
        }

        normalized["sections"].append(section)

    return normalized


with open(PDF_JSON, "r", encoding="utf-8") as f:
    raw_documents = json.load(f)

normalized_documents = [normalize_document(doc) for doc in raw_documents]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(normalized_documents, f, indent=4, ensure_ascii=False)

total_sections = sum(len(d["sections"]) for d in normalized_documents)

by_origin = {}
for d in normalized_documents:
    key = d.get("origin", "unknown")
    by_origin[key] = by_origin.get(key, 0) + 1

print("==============================")
print("PDF normalization completed")
print("==============================")
print("Documents normalized:", len(normalized_documents))
print("Total sections created:", total_sections)

print("\nDocuments by origin:")
for origin, count in by_origin.items():
    print(f"  {origin:25} {count}")

print("\nSaved to:", OUTPUT_FILE)
