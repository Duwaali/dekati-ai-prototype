#!/usr/bin/env python3
# Converts OCR text into the common format and links it to the document the image came from.


from pathlib import Path
from collections import Counter
from datetime import datetime
import json
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OCR_FILE = PROCESSED_DIR / "ocr" / "ocr_extracted_documents.json"
IMAGE_REGISTRY_FILE = PROCESSED_DIR / "registry" / "image_registry.json"
ATTACHMENT_REGISTRY_FILE = PROCESSED_DIR / "registry" / "attachment_registry.json"

NORMALIZED_DIR = PROCESSED_DIR / "normalized"
OUTPUT_FILE = NORMALIZED_DIR / "ocr_normalized.json"
TEMP_FILE = NORMALIZED_DIR / "ocr_normalized.building.json"

PARENT_FILES = {
    ".pdf": "pdf_normalized.json",
    ".msg": "email_normalized.json",
    ".docx": "word_normalized.json",
    ".doc": "word_normalized.json",
    ".dotx": "word_normalized.json",
    ".xlsx": "excel_normalized.json",
    ".xls": "excel_normalized.json",
    ".xlsm": "excel_normalized.json",
    ".pptx": "ppt_normalized.json",
    ".ppt": "ppt_normalized.json",
}


# Load a JSON file.
def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist:\n{path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Get the list of OCR results from the file.
def get_records(data):
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ("images", "attachments", "records", "items", "documents"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

    return []


# Find which document an image came from.
def get_source_document(record):
    for key in (
        "source_document",
        "source_file",
        "parent_source_file",
        "original_file",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


# Find which page an image was on.
def get_page_number(record):
    for key in ("page_number", "page", "page_num"):
        value = record.get(key)
        if value not in (None, ""):
            return value

    return None


# Load the list of known images.
def load_image_registry():
    data = load_json(IMAGE_REGISTRY_FILE)
    records = get_records(data)

    lookup = {}

    for record in records:
        file_name = record.get("file_name")

        if not file_name:
            continue

        if file_name in lookup:
            raise ValueError(
                f"Duplicate image filename in image_registry.json: {file_name}"
            )

        lookup[file_name] = record

    print(f"image_registry.json: {len(lookup)} usable records")

    return lookup


# Load the list of known e-mail attachments.
def load_attachment_registry():
    if not ATTACHMENT_REGISTRY_FILE.exists():
        print("attachment_registry.json: not found")
        return {}

    data = load_json(ATTACHMENT_REGISTRY_FILE)
    records = get_records(data)

    lookup = {}

    for record in records:
        saved_name = record.get("saved_name")

        if not saved_name:
            continue

        if saved_name in lookup:
            previous = lookup[saved_name]

            if previous.get("file_hash") != record.get("file_hash"):
                raise ValueError(
                    "Conflicting saved_name in attachment_registry.json:\n"
                    f"{saved_name}"
                )

            continue

        lookup[saved_name] = record

    print(f"attachment_registry.json: {len(lookup)} usable records")

    return lookup


# Load the documents that images can belong to.
def load_parent_documents():
    print("\nLoading normalized parent documents...")

    unique_files = set(PARENT_FILES.values())

    parent_lookup = {}

    total_documents = 0

    for filename in sorted(unique_files):
        path = NORMALIZED_DIR / filename
        data = load_json(path)

        if not isinstance(data, list):
            raise ValueError(f"{filename} must contain a list.")

        count = 0

        for document in data:
            doc_id = document.get("doc_id")
            source_file = document.get("source_file")

            if not doc_id:
                raise ValueError(
                    f"{filename} contains document without doc_id."
                )

            if not source_file:
                raise ValueError(
                    f"{filename} contains document without source_file."
                )

            extension = Path(source_file).suffix.lower()

            key = (extension, source_file)

            parent_lookup.setdefault(key, []).append(document)

            count += 1
            total_documents += 1

        print(f"  {filename}: {count} documents")

    print(f"Total normalized parent documents: {total_documents}")

    return parent_lookup


# Link images in e-mail attachments back to their e-mail.
def build_email_attachment_lookup():
    email_file = NORMALIZED_DIR / "email_normalized.json"

    emails = load_json(email_file)

    lookup = {}

    for email in emails:
        doc_id = email.get("doc_id")
        source_file = email.get("source_file")
        source_path = email.get("source_path")

        if not doc_id:
            raise ValueError("Email missing doc_id.")

        for attachment in email.get("attachments", []):
            if not isinstance(attachment, dict):
                continue

            saved_name = attachment.get("saved_name")

            if not saved_name:
                continue

            is_duplicate = attachment.get("duplicate") is True

            candidate = {
                "doc_id": doc_id,
                "source_file": source_file,
                "source_path": source_path,
                "attachment": attachment,
            }

            if saved_name not in lookup:
                lookup[saved_name] = candidate

                continue

            existing = lookup[saved_name]

            existing_duplicate = (
                existing["attachment"].get("duplicate") is True
            )

            if existing_duplicate and not is_duplicate:
                lookup[saved_name] = candidate

            elif not existing_duplicate and is_duplicate:
                pass

            elif existing["doc_id"] != doc_id:
                raise ValueError(
                    "Two non-duplicate emails claim the same image:\n"
                    f"{saved_name}\n"
                    f"{existing['doc_id']}\n"
                    f"{doc_id}"
                )

    return lookup


# Find the normalized document that an image belongs to.
def find_parent_document(source_document, parent_lookup):
    extension = Path(source_document).suffix.lower()

    candidates = parent_lookup.get(
        (extension, source_document),
        []
    )

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        return None

    doc_ids = {doc.get("doc_id") for doc in candidates}

    if len(doc_ids) == 1:
        return candidates[0]

    return None


# Turn one OCR result into the common document format.
def normalize_ocr_record(
    ocr_item,
    index,
    image_registry,
    attachment_registry,
    email_attachment_lookup,
    parent_lookup,
):
    file_name = ocr_item.get("file_name")

    if not file_name:
        return None, {
            "file_name": None,
            "reason": "missing_file_name",
        }

    registry_record = None
    registry_name = None

    if file_name in image_registry:
        registry_record = image_registry[file_name]
        registry_name = "image_registry"

    elif file_name in attachment_registry:
        registry_record = attachment_registry[file_name]
        registry_name = "attachment_registry"

    else:
        return None, {
            "file_name": file_name,
            "reason": "not_found_in_registry",
        }

    email_match = email_attachment_lookup.get(file_name)

    if email_match is not None:
        parent_doc_id = email_match["doc_id"]
        parent_source_file = email_match["source_file"]
        parent_source_type = "email"
        parent_source_path = email_match.get("source_path")

        page_number = None
        location = "email image"

    else:
        parent_source_file = get_source_document(
            registry_record
        )

        if not parent_source_file:
            return None, {
                "file_name": file_name,
                "reason": "registry_missing_source_document",
            }

        extension = Path(
            parent_source_file
        ).suffix.lower()

        if extension not in PARENT_FILES:
            return None, {
                "file_name": file_name,
                "source_document": parent_source_file,
                "reason": "unsupported_parent_extension",
                "extension": extension,
            }

        parent_document = find_parent_document(
            parent_source_file,
            parent_lookup,
        )

        if parent_document is None:
            return None, {
                "file_name": file_name,
                "source_document": parent_source_file,
                "reason": "parent_missing_or_ambiguous",
                "extension": extension,
            }

        parent_doc_id = parent_document["doc_id"]

        parent_source_type = parent_document.get(
            "source_type",
            "",
        )

        parent_source_path = parent_document.get(
            "source_path"
        )

        page_number = get_page_number(
            registry_record
        )

        if page_number is not None:
            location = f"page {page_number}"
        else:
            location = ocr_item.get(
                "location",
                "unknown",
            )

    ocr_metadata = dict(ocr_item)

    ocr_metadata.pop("text", None)

    doc_id = f"ocr_{index:06d}"

    normalized = {
        "doc_id": doc_id,
        "source_file": file_name,
        "source_type": "ocr_image",
        "origin": "ocr_extraction",
        "title": f"OCR: {file_name}",
        "author": "Tesseract OCR",
        "date": ocr_item.get("processed_date", ""),
        "parent_doc_id": parent_doc_id,
        "parent_source_file": parent_source_file,
        "parent_source_type": parent_source_type,
        "parent_source_path": parent_source_path,
        "page_number": page_number,
        "provenance_registry": registry_name,
        "ocr_metadata": ocr_metadata,
        "sections": [
            {
                "section_id": f"{doc_id}_s1",
                "heading": "Extracted Text",
                "text": ocr_item.get("text", ""),
                "location": location,
                "images": [],
            }
        ],
        "attachments": [],
    }

    return normalized, None


# Convert all OCR results and save them safely.
def main():
    print("=" * 46)
    print("OCR NORMALIZATION - FINAL PROVENANCE FIX")
    print("=" * 46)

    NORMALIZED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading OCR extraction results...")

    ocr_data = load_json(OCR_FILE)

    all_records = ocr_data.get(
        "documents",
        []
    )

    successful_records = [
        item
        for item in all_records
        if item.get("result") == "text_extracted"
    ]

    print(
        f"Total OCR records: {len(all_records)}"
    )

    print(
        f"Successful OCR records: "
        f"{len(successful_records)}"
    )

    print("\nLoading registries...")

    image_registry = load_image_registry()

    attachment_registry = load_attachment_registry()

    parent_lookup = load_parent_documents()

    email_attachment_lookup = (
        build_email_attachment_lookup()
    )

    print(
        "Email attachment -> exact parent mappings: "
        f"{len(email_attachment_lookup)}"
    )

    print("\nRestoring OCR provenance...")

    normalized_documents = []

    unresolved = []

    registry_usage = Counter()

    parent_type_usage = Counter()

    duplicate_email_images_skipped = 0

    for index, ocr_item in enumerate(
        successful_records
    ):
        file_name = ocr_item.get(
            "file_name"
        )

        if file_name in email_attachment_lookup:
            canonical = email_attachment_lookup[
                file_name
            ]

            if canonical["attachment"].get(
                "duplicate"
            ) is False:
                duplicate_email_images_skipped += 1

        normalized, error = normalize_ocr_record(
            ocr_item=ocr_item,
            index=index,
            image_registry=image_registry,
            attachment_registry=attachment_registry,
            email_attachment_lookup=email_attachment_lookup,
            parent_lookup=parent_lookup,
        )

        if error is not None:
            unresolved.append(error)

            continue

        normalized_documents.append(
            normalized
        )

        registry_usage[
            normalized["provenance_registry"]
        ] += 1

        parent_type_usage[
            normalized["parent_source_type"]
        ] += 1

    print(
        f"\nSuccessfully normalized: "
        f"{len(normalized_documents)}"
    )

    print(
        f"Unresolved records: "
        f"{len(unresolved)}"
    )

    if unresolved:
        print("\nUNRESOLVED RECORDS:")

        for item in unresolved:
            print("  ", item)

        raise RuntimeError(
            "\nPROVENANCE VALIDATION FAILED.\n"
            "Existing ocr_normalized.json was NOT changed."
        )

    print("\nRunning structural validation...")

    seen_doc_ids = set()
    seen_section_ids = set()

    for document in normalized_documents:
        doc_id = document.get(
            "doc_id"
        )

        if not doc_id:
            raise ValueError(
                "OCR document has no doc_id."
            )

        if doc_id in seen_doc_ids:
            raise ValueError(
                f"Duplicate OCR doc_id: {doc_id}"
            )

        seen_doc_ids.add(doc_id)

        required_fields = [
            "source_file",
            "source_type",
            "origin",
            "parent_doc_id",
            "parent_source_file",
            "parent_source_type",
            "provenance_registry",
        ]

        for field in required_fields:
            if not document.get(field):
                raise ValueError(
                    f"Missing required field "
                    f"'{field}' in {doc_id}"
                )

        sections = document.get(
            "sections"
        )

        if not isinstance(
            sections,
            list,
        ):
            raise ValueError(
                f"Sections is not a list: {doc_id}"
            )

        if len(sections) != 1:
            raise ValueError(
                f"OCR document must have exactly "
                f"one section: {doc_id}"
            )

        section = sections[0]

        section_id = section.get(
            "section_id"
        )

        if not section_id:
            raise ValueError(
                f"Missing section_id: {doc_id}"
            )

        if section_id in seen_section_ids:
            raise ValueError(
                f"Duplicate section_id: {section_id}"
            )

        seen_section_ids.add(
            section_id
        )

        if not section.get("text"):
            raise ValueError(
                f"Empty OCR text: {doc_id}"
            )

        if not section.get("location"):
            raise ValueError(
                f"Missing location: {doc_id}"
            )

    if len(normalized_documents) != len(
        successful_records
    ):
        raise ValueError(
            "Normalized count does not match "
            "successful OCR count."
        )

    print("\nValidating email-image provenance...")

    email_count = 0

    for document in normalized_documents:
        if document["parent_source_type"] != "email":
            continue

        email_count += 1

        if document.get("page_number") is not None:
            raise ValueError(
                "Email OCR image has unexpected "
                f"page number: {document['doc_id']}"
            )

        if document["sections"][0]["location"] != "email image":
            raise ValueError(
                "Email OCR image has incorrect "
                f"location: {document['doc_id']}"
            )

    print(
        f"Email-image records validated: "
        f"{email_count}"
    )

    print("\nValidating page-number provenance...")

    page_count = 0

    for document in normalized_documents:
        page_number = document.get(
            "page_number"
        )

        if page_number is None:
            continue

        page_count += 1

        expected = f"page {page_number}"

        actual = document[
            "sections"
        ][0]["location"]

        if actual != expected:
            raise ValueError(
                "Page location mismatch:\n"
                f"  file: {document['source_file']}\n"
                f"  expected: {expected}\n"
                f"  actual: {actual}"
            )

    print(
        f"Page-number records validated: "
        f"{page_count}"
    )

    print("\nAll validation passed.")

    print("Writing temporary normalized file...")

    if TEMP_FILE.exists():
        TEMP_FILE.unlink()

    with open(
        TEMP_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            normalized_documents,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("Re-reading temporary file...")

    verified = load_json(
        TEMP_FILE
    )

    if len(verified) != len(
        normalized_documents
    ):
        TEMP_FILE.unlink()

        raise ValueError(
            "Temporary file verification failed."
        )

    backup_file = None

    if OUTPUT_FILE.exists():
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        backup_file = (
            NORMALIZED_DIR
            / (
                "ocr_normalized.backup_"
                f"{timestamp}.json"
            )
        )

        print("\nCreating backup...")

        shutil.copy2(
            OUTPUT_FILE,
            backup_file,
        )

        print(
            f"Backup created: {backup_file}"
        )

    print(
        "\nReplacing ocr_normalized.json..."
    )

    TEMP_FILE.replace(
        OUTPUT_FILE
    )

    print("\n" + "=" * 46)
    print("OCR NORMALIZATION COMPLETE")
    print("=" * 46)

    print(
        f"Successful OCR records : "
        f"{len(successful_records)}"
    )

    print(
        f"Normalized documents   : "
        f"{len(normalized_documents)}"
    )

    print(
        f"Unique doc_ids         : "
        f"{len(seen_doc_ids)}"
    )

    print(
        f"Unique section_ids     : "
        f"{len(seen_section_ids)}"
    )

    print("\nRegistry usage:")

    for name, count in sorted(
        registry_usage.items()
    ):
        print(
            f"  {name}: {count}"
        )

    print("\nParent source types:")

    for source_type, count in sorted(
        parent_type_usage.items()
    ):
        print(
            f"  {source_type}: {count}"
        )

    print("\nValidation:")

    print("  Provenance validation    : PASSED")
    print("  Structural validation    : PASSED")
    print("  Email-image validation   : PASSED")
    print("  Page-location validation : PASSED")

    if backup_file:
        print("\nPrevious file backed up:")
        print(f"  {backup_file}")

    print("\nNew file:")
    print(f"  {OUTPUT_FILE}")

    print("\n" + "=" * 46)
    print("SAFE REPLACEMENT COMPLETED")
    print("=" * 46)


if __name__ == "__main__":
    main()
