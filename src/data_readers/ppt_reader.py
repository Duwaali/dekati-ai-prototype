# Reads PowerPoint files: slide text, tables, charts, notes and images.

from pathlib import Path
import json
import hashlib
from datetime import datetime

from pptx import Presentation

from registry_utils import (
    calculate_file_hash,
    check_document_exists,
    add_document_record,
    check_image_exists,
    add_image_record,
    load_image_registry
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FOLDER = PROJECT_ROOT / "data" / "raw" / "company_data"

EMAIL_ATTACHMENT_FOLDER = PROJECT_ROOT / "data" / "processed" / "email" / "email_attachments"
EMAIL_UNZIPPED_FOLDER = PROJECT_ROOT / "data" / "processed" / "email" / "email_attachments_unzipped"

PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"

PPT_FOLDER = PROCESSED_FOLDER / "ppt"

IMAGE_FOLDER = PPT_FOLDER / "ppt_images"

JSON_OUTPUT = PPT_FOLDER / "ppt_extracted_documents.json"

FAILED_LOG = PPT_FOLDER / "ppt_failed_files.json"

PPT_FOLDER.mkdir(parents=True, exist_ok=True)
IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)


# Tell whether a file came from the company folder or an e-mail attachment.
def classify_source(file_path):
    try:
        if EMAIL_ATTACHMENT_FOLDER.exists() and file_path.is_relative_to(EMAIL_ATTACHMENT_FOLDER):
            return "email_attachment"
        if EMAIL_UNZIPPED_FOLDER.exists() and file_path.is_relative_to(EMAIL_UNZIPPED_FOLDER):
            return "email_attachment_zipped"
    except ValueError:
        pass
    return "company_data"


if JSON_OUTPUT.exists():
    with open(JSON_OUTPUT, "r", encoding="utf-8") as file:
        try:
            all_documents = json.load(file)
        except json.JSONDecodeError:
            all_documents = []
else:
    all_documents = []


image_registry = load_image_registry()

saved_image_hashes = {}

for image in image_registry["images"]:
    saved_image_hashes[image["image_hash"]] = image["file_name"]


# Make a fingerprint of an image to spot duplicates.
def calculate_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()


company_pptx = list(RAW_FOLDER.rglob("*.pptx"))
email_pptx = list(EMAIL_ATTACHMENT_FOLDER.glob("*.pptx")) if EMAIL_ATTACHMENT_FOLDER.exists() else []
zip_pptx = list(EMAIL_UNZIPPED_FOLDER.rglob("*.pptx")) if EMAIL_UNZIPPED_FOLDER.exists() else []

pptx_files = company_pptx + email_pptx + zip_pptx
pptx_files = [f for f in pptx_files if not f.name.startswith("~$")]

old_ppt_files = list(RAW_FOLDER.rglob("*.ppt"))
if EMAIL_ATTACHMENT_FOLDER.exists():
    old_ppt_files += list(EMAIL_ATTACHMENT_FOLDER.glob("*.ppt"))

print("\n==============================")
print("TOTAL .pptx FILES FOUND:", len(pptx_files))
print("==============================")
print("  from company_data       :", len(company_pptx))
print("  from email attachments  :", len(email_pptx))
print("  from unzipped emails    :", len(zip_pptx))

if old_ppt_files:
    print("\nOld .ppt files found (must be converted to .pptx):")
    for ppt in old_ppt_files:
        print("-", ppt.name)

if len(pptx_files) == 0:
    print("\nNo .pptx files found")
    exit()


PICTURE = 13
GROUP = 6


# Go through all shapes on a slide, including grouped ones.
def flatten_shapes(shapes):
    flat = []

    for shape in shapes:
        try:
            if shape.shape_type == GROUP:
                flat.extend(flatten_shapes(shape.shapes))
            else:
                flat.append(shape)
        except Exception:
            flat.append(shape)

    return flat


# Turn a chart into readable lines of text.
def extract_chart_data(chart):
    lines = []

    try:
        categories = [str(c) for c in chart.plots[0].categories]
    except Exception:
        categories = []

    if categories:
        lines.append("Categories: " + ", ".join(categories))

    try:
        for series in chart.series:
            values = [str(v) for v in series.values if v is not None]
            if values:
                lines.append(f"{series.name}: " + ", ".join(values))
    except Exception:
        pass

    return lines


# Read one presentation: text, tables, charts, notes and images per slide.
def extract_ppt(ppt_file):
    file_name = ppt_file.name

    ppt_hash = calculate_file_hash(ppt_file)

    if check_document_exists(file_name, ppt_hash):
        print("Skipping already processed PPT:", file_name)
        return "skipped"

    print("\nProcessing:", file_name)

    presentation = Presentation(ppt_file)

    document_data = {
        "source": file_name,
        "source_type": classify_source(ppt_file),
        "metadata": {
            "slides": len(presentation.slides),
            "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "slides": []
    }

    for slide_number, slide in enumerate(presentation.slides, start=1):
        slide_data = {
            "slide_number": slide_number,
            "title": "",
            "text": [],
            "bullets": [],
            "notes": "",
            "tables": [],
            "charts": [],
            "images": []
        }

        try:
            title_shape = slide.shapes.title
        except Exception:
            title_shape = None

        try:
            if slide.has_notes_slide:
                notes_text = slide.notes_slide.notes_text_frame.text.strip()
                if notes_text:
                    slide_data["notes"] = notes_text
        except Exception as e:
            print("  Notes extraction error:", e)

        all_shapes = flatten_shapes(slide.shapes)

        for shape in all_shapes:
            try:
                if shape.has_text_frame:
                    text = shape.text_frame.text.strip()

                    if text:
                        slide_data["text"].append(text)

                        for paragraph in shape.text_frame.paragraphs:
                            if paragraph.level > 0 and paragraph.text.strip():
                                slide_data["bullets"].append(paragraph.text)

                    if title_shape is not None and shape == title_shape:
                        slide_data["title"] = text
            except Exception as e:
                print("  Text extraction error:", e)

            try:
                if shape.has_table:
                    table_data = []

                    for row in shape.table.rows:
                        table_data.append([cell.text for cell in row.cells])

                    slide_data["tables"].append(table_data)
            except Exception as e:
                print("  Table extraction error:", e)

            try:
                if shape.has_chart:
                    chart_lines = extract_chart_data(shape.chart)

                    if chart_lines:
                        slide_data["charts"].append(chart_lines)
            except Exception as e:
                print("  Chart extraction error:", e)

            try:
                if shape.shape_type == PICTURE:
                    image = shape.image
                    image_bytes = image.blob
                    image_hash = calculate_image_hash(image_bytes)

                    if image_hash in saved_image_hashes or check_image_exists(image_hash):
                        image_name = saved_image_hashes.get(image_hash, "duplicate_skipped")
                    else:
                        image_name = f"ppt_image_{len(saved_image_hashes)+1}.{image.ext}"
                        image_path = IMAGE_FOLDER / image_name

                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        saved_image_hashes[image_hash] = image_name

                        add_image_record(
                            image_hash=image_hash,
                            file_name=image_name,
                            source_document=file_name,
                            page_number=slide_number
                        )

                    slide_data["images"].append({
                        "file": image_name,
                        "slide": slide_number
                    })
            except Exception as e:
                print("  Image extraction error:", e)

        document_data["slides"].append(slide_data)

    add_document_record(
        file_name=file_name,
        file_type="ppt",
        file_hash=ppt_hash
    )

    return document_data


new_documents = []
skipped_count = 0
failed_files = []

for ppt_file in pptx_files:
    try:
        result = extract_ppt(ppt_file)
    except Exception as e:
        print("FAILED:", ppt_file.name, "->", e)
        failed_files.append({
            "file": ppt_file.name,
            "path": str(ppt_file),
            "reason": str(e)
        })
        continue

    if result == "skipped":
        skipped_count += 1
    elif result:
        new_documents.append(result)

all_documents.extend(new_documents)


with open(JSON_OUTPUT, "w", encoding="utf-8") as file:
    json.dump(all_documents, file, indent=4, ensure_ascii=False)

for old_file in old_ppt_files:
    failed_files.append({
        "file": old_file.name,
        "path": str(old_file),
        "reason": "Legacy .ppt format - convert to .pptx"
    })

with open(FAILED_LOG, "w", encoding="utf-8") as file:
    json.dump(failed_files, file, indent=4, ensure_ascii=False)


by_source = {}
for d in new_documents:
    key = d.get("source_type", "unknown")
    by_source[key] = by_source.get(key, 0) + 1

slides_total = sum(len(d["slides"]) for d in new_documents)
notes_total = sum(1 for d in new_documents for s in d["slides"] if s["notes"])
charts_total = sum(len(s["charts"]) for d in new_documents for s in d["slides"])
tables_total = sum(len(s["tables"]) for d in new_documents for s in d["slides"])
text_slides = sum(1 for d in new_documents for s in d["slides"] if s["text"])

print("\n==============================")
print("PPT EXTRACTION SUMMARY")
print("==============================")
print(".pptx files found      :", len(pptx_files))
print("Newly processed        :", len(new_documents))
print("Already in registry    :", skipped_count)
print("Failed                 :", len(failed_files))
print("Slides extracted       :", slides_total)
print("Slides with text       :", text_slides)
print("Slides with notes      :", notes_total)
print("Tables extracted       :", tables_total)
print("Charts extracted       :", charts_total)
print("Total docs in JSON     :", len(all_documents))
print("Unique images so far   :", len(saved_image_hashes))

print("\nNewly processed by source:")
for source, count in by_source.items():
    print(f"  {source:25} {count}")

if failed_files:
    print("\nFiles needing attention:")
    for item in failed_files:
        print(" -", item["file"], "->", item["reason"])

print("\nJSON  :", JSON_OUTPUT)
print("Images:", IMAGE_FOLDER)
print("Failed:", FAILED_LOG)
