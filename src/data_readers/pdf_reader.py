# Reads PDF files: text, tables and images, page by page.

from pathlib import Path
import json
import hashlib
import fitz
import pdfplumber
from datetime import datetime

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

PDF_FOLDER = PROCESSED_FOLDER / "pdf"

IMAGE_FOLDER = PDF_FOLDER / "pdf_images"

JSON_OUTPUT = PDF_FOLDER / "pdf_extracted_documents.json"

FAILED_LOG = PDF_FOLDER / "pdf_failed_files.json"

PROCESSED_FOLDER.mkdir(exist_ok=True)
PDF_FOLDER.mkdir(exist_ok=True)
IMAGE_FOLDER.mkdir(exist_ok=True)


if JSON_OUTPUT.exists():
    with open(JSON_OUTPUT, "r", encoding="utf-8") as f:
        try:
            all_documents = json.load(f)
        except json.JSONDecodeError:
            all_documents = []
else:
    all_documents = []


saved_image_hashes = {}

image_registry = load_image_registry()
for img in image_registry["images"]:
    saved_image_hashes[img["image_hash"]] = img["file_name"]


# Make a fingerprint of an image to spot duplicates.
def get_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()


# Tell whether a PDF came from the company folder or an e-mail attachment.
def classify_source(pdf_file):
    try:
        if EMAIL_ATTACHMENT_FOLDER.exists() and pdf_file.is_relative_to(EMAIL_ATTACHMENT_FOLDER):
            return "email_attachment"
        if EMAIL_UNZIPPED_FOLDER.exists() and pdf_file.is_relative_to(EMAIL_UNZIPPED_FOLDER):
            return "email_attachment_zipped"
    except ValueError:
        pass
    return "company_data"


company_pdfs = list(RAW_FOLDER.rglob("*.pdf"))
email_pdfs = list(EMAIL_ATTACHMENT_FOLDER.glob("*.pdf")) if EMAIL_ATTACHMENT_FOLDER.exists() else []
zip_pdfs = list(EMAIL_UNZIPPED_FOLDER.rglob("*.pdf")) if EMAIL_UNZIPPED_FOLDER.exists() else []

pdf_files = company_pdfs + email_pdfs + zip_pdfs

print("\n==============================")
print("TOTAL PDF FILES FOUND:", len(pdf_files))
print("==============================")
print("  from company_data       :", len(company_pdfs))
print("  from email attachments  :", len(email_pdfs))
print("  from unzipped emails    :", len(zip_pdfs))

if len(pdf_files) == 0:
    print("No PDF files found")
    exit()


# Read one PDF: text, tables and images on every page.
def extract_pdf(pdf_file):
    pdf_hash = calculate_file_hash(pdf_file)
    file_name = pdf_file.name

    if check_document_exists(file_name, pdf_hash):
        print(f"Skipping already processed PDF: {file_name}")
        return "skipped"

    print("\nProcessing:", file_name)

    source_type = classify_source(pdf_file)

    document_data = {
        "source": file_name,
        "source_type": source_type,
        "source_path": str(pdf_file),
        "metadata": {},
        "pages": []
    }

    pdf = fitz.open(pdf_file)

    try:
        plumber_pdf = pdfplumber.open(pdf_file)
    except Exception as e:
        print("Could not open PDF for table extraction:", e)
        plumber_pdf = None

    document_data["metadata"] = pdf.metadata

    for page_number, page in enumerate(pdf):
        page_data = {
            "page_number": page_number + 1,
            "text": page.get_text(),
            "tables": [],
            "images": []
        }

        if plumber_pdf is not None:
            try:
                tables = plumber_pdf.pages[page_number].extract_tables()
                page_data["tables"] = tables
            except Exception as e:
                print("  Table extraction error:", e)

        images = page.get_images(full=True)
        for image_index, image in enumerate(images):
            xref = image[0]
            extracted = pdf.extract_image(xref)
            image_bytes = extracted["image"]
            image_hash = get_image_hash(image_bytes)

            if image_hash in saved_image_hashes or check_image_exists(image_hash):
                image_name = saved_image_hashes.get(image_hash, "duplicate_skipped")
            else:
                image_name = f"image_{len(saved_image_hashes) + 1}.{extracted['ext']}"
                image_path = IMAGE_FOLDER / image_name
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                saved_image_hashes[image_hash] = image_name
                add_image_record(
                    image_hash=image_hash,
                    file_name=image_name,
                    source_document=file_name,
                    page_number=page_number + 1
                )

            page_data["images"].append({
                "file": image_name,
                "page": page_number + 1
            })

        document_data["pages"].append(page_data)

    pdf.close()

    if plumber_pdf is not None:
        plumber_pdf.close()

    add_document_record(
        file_name=file_name,
        file_type="pdf",
        file_hash=pdf_hash
    )

    return document_data


new_documents = []
skipped_count = 0
failed_files = []

for index, pdf_file in enumerate(pdf_files, start=1):
    if index % 100 == 0:
        print(f"  processed {index} of {len(pdf_files)}")

    try:
        result = extract_pdf(pdf_file)
    except Exception as e:
        print("FAILED:", pdf_file.name, "->", e)
        failed_files.append({
            "file": pdf_file.name,
            "path": str(pdf_file),
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

with open(FAILED_LOG, "w", encoding="utf-8") as file:
    json.dump(failed_files, file, indent=4, ensure_ascii=False)


by_source = {}
for d in new_documents:
    key = d.get("source_type", "unknown")
    by_source[key] = by_source.get(key, 0) + 1

print("\n==============================")
print("PDF EXTRACTION SUMMARY")
print("==============================")
print("Files found        :", len(pdf_files))
print("Newly processed     :", len(new_documents))
print("Already in registry :", skipped_count)
print("Failed              :", len(failed_files))
print("Total docs in JSON  :", len(all_documents))
print("Unique images saved :", len(saved_image_hashes))

print("\nNewly processed by source:")
for source, count in by_source.items():
    print(f"  {source:25} {count}")

if failed_files:
    print("\nFiles needing attention:")
    for item in failed_files[:20]:
        print(" -", item["file"], "->", item["reason"])

print("\nJSON created :", JSON_OUTPUT)
print("Images stored:", IMAGE_FOLDER)
print("Failed       :", FAILED_LOG)
