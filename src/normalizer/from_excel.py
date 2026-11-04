# Converts extracted Excel sheets into the common document format, row by row.

from pathlib import Path
import json
import hashlib


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXCEL_JSON = PROJECT_ROOT / "data" / "processed" / "excel" / "excel_extracted_documents.json"

NORMALIZED_FOLDER = PROJECT_ROOT / "data" / "processed" / "normalized"
NORMALIZED_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = NORMALIZED_FOLDER / "excel_normalized.json"

ROWS_PER_SECTION = 20

HEADER_OVERRIDES = {
    "DekatiReferencelist_updated": 5,
}


# Check if a cell value is a number.
def looks_like_number(value):
    try:
        float(str(value).replace(",", ".").replace(" ", ""))
        return True
    except ValueError:
        return False


# Find the row that holds the column names.
def find_header_row(rows, search_limit=15):
    candidates = []

    for index, row in enumerate(rows[:search_limit]):
        filled = [c for c in row if str(c).strip()]

        if len(filled) < 2:
            continue

        text_cells = [c for c in filled if not looks_like_number(c)]

        if len(text_cells) / len(filled) < 0.7:
            continue

        if index + 1 >= len(rows):
            continue

        next_filled = [c for c in rows[index + 1] if str(c).strip()]

        if len(next_filled) < 2:
            continue

        candidates.append((index, len(text_cells)))

    if not candidates:
        return None

    return candidates[-1][0]


# Decide which row is the header, using a manual fix if there is one.
def resolve_header_index(sheet_name, rows):
    if sheet_name in HEADER_OVERRIDES:
        override = HEADER_OVERRIDES[sheet_name]

        if override is None:
            return None

        index = override - 1

        if 0 <= index < len(rows):
            return index

        return find_header_row(rows)

    return find_header_row(rows)


# Write a row as 'column: value' pairs so values keep their meaning.
def row_to_sentence(headers, row):
    parts = []

    for i, cell in enumerate(row):
        value = str(cell).strip()

        if not value:
            continue

        if headers and i < len(headers):
            label = str(headers[i]).strip()
        else:
            label = ""

        if label:
            parts.append(f"{label}: {value}")
        else:
            parts.append(value)

    return ", ".join(parts)


# Turn one workbook into the common document format.
def normalize_document(doc):
    source_file = doc.get("file_name", "unknown.xlsx")

    doc_id = hashlib.md5(source_file.encode("utf-8")).hexdigest()[:8]

    normalized = {
        "doc_id": doc_id,
        "source_file": source_file,
        "source_type": "excel",
        "origin": doc.get("source_type", "company_data"),
        "title": source_file,
        "author": "",
        "date": doc.get("processed_date") or "",
        "parent_doc_id": None,
        "sections": []
    }

    section_counter = 0

    # Add one section to the document.
    def add_section(heading, text, location):
        nonlocal section_counter

        if not text.strip():
            return

        section_counter += 1

        normalized["sections"].append({
            "section_id": f"{doc_id}_s{section_counter}",
            "heading": heading,
            "text": text,
            "location": location,
            "images": []
        })

    for sheet in doc.get("sheets", []):
        sheet_name = sheet.get("sheet_name", "")
        rows = sheet.get("rows") or []

        if not rows:
            continue

        header_index = resolve_header_index(sheet_name, rows)

        if header_index is not None and header_index > 0:
            note_lines = []

            for row in rows[:header_index]:
                line = " ".join(str(c).strip() for c in row if str(c).strip())
                if line:
                    note_lines.append(line)

            if note_lines:
                add_section(
                    heading=f"{sheet_name} - notes",
                    text="\n".join(note_lines),
                    location=f"sheet: {sheet_name}, rows above header"
                )

        if header_index is not None:
            headers = rows[header_index]
            data_rows = rows[header_index + 1:]
            first_data_row_number = header_index + 2
        else:
            headers = []
            data_rows = rows
            first_data_row_number = 1

        for start in range(0, len(data_rows), ROWS_PER_SECTION):
            chunk = data_rows[start:start + ROWS_PER_SECTION]

            lines = []

            for row in chunk:
                sentence = row_to_sentence(headers, row)
                if sentence:
                    lines.append(sentence)

            if not lines:
                continue

            row_from = first_data_row_number + start
            row_to = row_from + len(chunk) - 1

            add_section(
                heading=sheet_name,
                text="\n".join(lines),
                location=f"sheet: {sheet_name}, rows {row_from}-{row_to}"
            )

    comments = doc.get("comments") or []

    if comments:
        lines = []

        for c in comments:
            cell = c.get("cell", "")
            text = c.get("text", "").strip()
            if text:
                lines.append(f"{cell}: {text}" if cell else text)

        add_section(
            heading="Cell comments",
            text="\n".join(lines),
            location="cell comments"
        )

    return normalized


with open(EXCEL_JSON, "r", encoding="utf-8") as f:
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
print("EXCEL normalization completed")
print("==============================")
print("Documents normalized:", len(normalized_documents))
print("Total sections created:", total_sections)

print("\nDocuments by origin:")
for origin, count in by_origin.items():
    print(f"  {origin:25} {count}")

print("\nHeader row used per sheet (first 20):")
count_shown = 0
for doc in raw_documents:
    for sheet in doc.get("sheets", []):
        rows = sheet.get("rows") or []
        if not rows:
            continue
        name = sheet.get("sheet_name", "")
        idx = resolve_header_index(name, rows)
        source = "manual" if name in HEADER_OVERRIDES else "auto"
        label = f"row {idx + 1}" if idx is not None else "none (no headers)"
        print(f"  {doc.get('file_name')[:40]:40} / {name[:25]:25} {label} ({source})")
        count_shown += 1
        if count_shown >= 20:
            break
    if count_shown >= 20:
        break

print("\nSaved to:", OUTPUT_FILE)
