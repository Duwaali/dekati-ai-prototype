# Converts extracted e-mails into the common document format.

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMAIL_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "email"
    / "email_extracted_documents.json"
)

NORMALIZED_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normalized"
)

NORMALIZED_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = NORMALIZED_FOLDER / "email_normalized.json"

MAX_SECTION_CHARS = 3000


# Split long e-mail text at paragraph breaks.
def split_text(text, max_chars):
    if len(text) <= max_chars:
        return [text]

    parts = []
    buffer = []
    length = 0

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        paragraph_length = len(paragraph)

        if length + paragraph_length > max_chars and buffer:
            parts.append("\n\n".join(buffer))
            buffer = []
            length = 0

        buffer.append(paragraph)
        length += paragraph_length

    if buffer:
        parts.append("\n\n".join(buffer))

    return parts


# Turn one e-mail into the common document format.
def normalize_document(doc):
    source_file = doc.get("file_name") or "unknown.msg"
    file_hash = (doc.get("file_hash") or "").strip()


    if not file_hash:
        raise ValueError(
            f"Email '{source_file}' has no file_hash. "
            "Every email must have a unique file_hash before normalization."
        )

    doc_id = file_hash


    subject = (doc.get("subject") or "").strip()
    sender = (doc.get("sender") or "").strip()
    recipient = (doc.get("to") or "").strip()
    cc = (doc.get("cc") or "").strip()
    date = (doc.get("date") or "").strip()

    body = (doc.get("body") or "").strip()


    normalized = {
        "doc_id": doc_id,
        "source_file": source_file,
        "source_type": "email",
        "origin": "email",

        "title": subject or source_file,
        "author": sender,
        "date": date,
        "parent_doc_id": None,

        "file_hash": file_hash,
        "file_type": doc.get("file_type", "email"),
        "source_path": doc.get("source_path", ""),
        "processed_date": doc.get("processed_date", ""),

        "archive_year": doc.get("archive_year", ""),
        "mailbox_folder": doc.get("mailbox_folder", ""),

        "sender": sender,
        "recipient": recipient,
        "cc": cc,

        "subject": subject,
        "body_source": doc.get("body_source", ""),
        "body_length": doc.get("body_length", len(body)),

        "body": body,

        "attachments": doc.get("attachments") or [],

        "sections": []
    }

    section_counter = 0

    # Add one section to the document.
    def add_section(text, location, heading=""):
        nonlocal section_counter

        if not text or not text.strip():
            return

        section_counter += 1

        normalized["sections"].append({
            "section_id": f"{doc_id}_s{section_counter}",
            "heading": heading,
            "text": text,
            "location": location,
            "images": []
        })


    if body:
        context_parts = []

        if sender:
            context_parts.append(f"From: {sender}")

        if recipient:
            context_parts.append(f"To: {recipient}")

        if cc:
            context_parts.append(f"CC: {cc}")

        if date:
            context_parts.append(f"Date: {date}")

        if subject:
            context_parts.append(f"Subject: {subject}")

        context = "\n".join(context_parts)

        pieces = split_text(body, MAX_SECTION_CHARS)

        for index, piece in enumerate(pieces, start=1):
            if len(pieces) > 1:
                location = (
                    f"email message, part {index} of {len(pieces)}"
                )
            else:
                location = "email message"

            if context:
                full_text = context + "\n\n" + piece
            else:
                full_text = piece

            add_section(
                text=full_text,
                location=location,
                heading=subject
            )


    attachments = doc.get("attachments") or []

    document_attachments = [
        attachment
        for attachment in attachments
        if attachment.get("type") == "document"
    ]

    if document_attachments:
        lines = []

        for attachment in document_attachments:
            original = attachment.get("original_name", "")
            saved = attachment.get("saved_name", "")

            if original and saved:
                lines.append(
                    f"{original} (stored as {saved})"
                )
            elif original:
                lines.append(original)
            elif saved:
                lines.append(saved)

        if lines:
            add_section(
                text=(
                    "Documents attached to this email:\n"
                    + "\n".join(lines)
                ),
                location="attachment list",
                heading="Attachments"
            )

    return normalized


print("=" * 80)
print("EMAIL NORMALIZATION")
print("=" * 80)

print("\nLoading:")
print(EMAIL_JSON)

with open(EMAIL_JSON, "r", encoding="utf-8") as f:
    raw_documents = json.load(f)

if not isinstance(raw_documents, list):
    raise ValueError(
        "Expected email_extracted_documents.json to contain a list."
    )

print("Extracted email documents:", len(raw_documents))


print("\nValidating file_hash values...")

missing_hash = []
duplicate_hashes = {}
seen_hashes = set()

for index, doc in enumerate(raw_documents):
    file_name = doc.get("file_name", "unknown.msg")
    file_hash = (doc.get("file_hash") or "").strip()

    if not file_hash:
        missing_hash.append(file_name)
        continue

    if file_hash in seen_hashes:
        duplicate_hashes.setdefault(file_hash, 1)
        duplicate_hashes[file_hash] += 1
    else:
        seen_hashes.add(file_hash)

if missing_hash:
    raise ValueError(
        f"Found {len(missing_hash)} email(s) without file_hash. "
        "Normalization stopped to prevent unstable document IDs."
    )

if duplicate_hashes:
    raise ValueError(
        f"Found {len(duplicate_hashes)} duplicate file_hash value(s). "
        "Normalization stopped because doc_id must be unique."
    )

print("Emails with file_hash:", len(seen_hashes))
print("Duplicate file_hash values:", 0)
print("Missing file_hash values:", 0)
print("✓ Source identity validation passed")


print("\nNormalizing emails...")

normalized_documents = []

for doc in raw_documents:
    normalized_documents.append(
        normalize_document(doc)
    )

print(
    "Normalized documents:",
    len(normalized_documents)
)


normalized_ids = [
    document["doc_id"]
    for document in normalized_documents
]

unique_normalized_ids = set(normalized_ids)

if len(normalized_ids) != len(unique_normalized_ids):
    raise ValueError(
        "Duplicate normalized doc_id values detected."
    )

print("Unique normalized doc_ids:", len(unique_normalized_ids))
print("Duplicate normalized doc_ids:", 0)
print("✓ Document ID validation passed")


all_section_ids = []
empty_documents = []

for document in normalized_documents:
    sections = document.get("sections", [])

    if not sections:
        empty_documents.append(
            document.get("source_file", "unknown.msg")
        )

    for section in sections:
        section_id = section.get("section_id")

        if not section_id:
            raise ValueError(
                f"Missing section_id in document "
                f"{document['doc_id']}"
            )

        if not section.get("text", "").strip():
            raise ValueError(
                f"Empty section text found in section "
                f"{section_id}"
            )

        all_section_ids.append(section_id)


unique_section_ids = set(all_section_ids)

if len(all_section_ids) != len(unique_section_ids):
    raise ValueError(
        "Duplicate section_id values detected."
    )

print("\nSection validation:")
print("Total sections:", len(all_section_ids))
print("Unique section IDs:", len(unique_section_ids))
print("Empty section IDs:", 0)
print("Documents with no sections:", len(empty_documents))
print("✓ Section validation passed")


total_sections = sum(
    len(document["sections"])
    for document in normalized_documents
)

split_documents = sum(
    1
    for document in normalized_documents
    if any(
        "part" in section.get("location", "")
        for section in document["sections"]
    )
)

documents_with_attachments = sum(
    1
    for document in normalized_documents
    if document.get("attachments")
)

by_year = {}

for document in normalized_documents:
    year = document.get("archive_year") or "unknown"

    by_year[year] = by_year.get(year, 0) + 1


print("\nSaving normalized email data...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        normalized_documents,
        f,
        indent=4,
        ensure_ascii=False
    )


print("\n" + "=" * 80)
print("EMAIL NORMALIZATION COMPLETED")
print("=" * 80)

print("Documents normalized      :", len(normalized_documents))
print("Unique doc_ids            :", len(unique_normalized_ids))
print("Total sections created    :", total_sections)
print("Documents with no sections:", len(empty_documents))
print("Long emails split         :", split_documents)
print("Emails with attachments   :", documents_with_attachments)

print("\nBy archive year:")

for year in sorted(by_year):
    print(f"  {year}: {by_year[year]}")

if empty_documents:
    print("\nDocuments with no sections (first 10):")
    for name in empty_documents[:10]:
        print(" -", name)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n✓ All email file_hash values are unique.")
print("✓ All normalized doc_ids are unique.")
print("✓ All section IDs are unique.")
print("✓ Original email metadata and attachments are preserved.")
print("=" * 80)
