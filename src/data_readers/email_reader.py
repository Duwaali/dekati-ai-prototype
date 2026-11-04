# Reads e-mails from the Outlook archives, cleans the text and saves attachments.

from registry_utils import add_document_record, check_document_exists
from pathlib import Path
import json
import hashlib
import re
import email
from email import policy
from datetime import datetime

import extract_msg
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FOLDER = PROJECT_ROOT / "data" / "raw" / "email_extracted"

EMAIL_OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed" / "email"

ATTACHMENT_FOLDER = EMAIL_OUTPUT_FOLDER / "email_attachments"

EMAIL_IMAGE_FOLDER = EMAIL_OUTPUT_FOLDER / "email_images"

JSON_OUTPUT = EMAIL_OUTPUT_FOLDER / "email_extracted_documents.json"

FAILED_LOG = EMAIL_OUTPUT_FOLDER / "email_failed_files.json"

REGISTRY_FOLDER = PROJECT_ROOT / "data" / "processed" / "registry"

IMAGE_REGISTRY_FILE = REGISTRY_FOLDER / "image_registry.json"

ATTACHMENT_REGISTRY_FILE = REGISTRY_FOLDER / "attachment_registry.json"

EMAIL_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
ATTACHMENT_FOLDER.mkdir(parents=True, exist_ok=True)
EMAIL_IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
REGISTRY_FOLDER.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}


# Make a fingerprint of a file to spot duplicates.
def file_hash(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()


# Make a fingerprint of some bytes to spot duplicates.
def bytes_hash(data):
    return hashlib.sha256(data).hexdigest()


# Clean up text from Outlook (stray characters and spaces).
def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")

    return str(value).replace("\x00", "").strip()


# Turn an HTML e-mail body into plain text.
def html_to_text(html):
    if not html:
        return ""

    if isinstance(html, (bytes, bytearray)):
        html = html.decode("utf-8", errors="ignore")

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "head", "meta", "title"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]

    return "\n".join(lines)


# Spot when the body is raw e-mail source instead of real text.
def looks_like_raw_mime(text):
    if not text:
        return False

    markers = [
        "boundary-LibPST",
        "Content-Transfer-Encoding: base64",
    ]

    if any(m in text for m in markers):
        return True

    if len(text) > 50000:
        space_ratio = text.count(" ") / len(text)
        if space_ratio < 0.05:
            return True

    return False


# Convert dates and other values so they can be saved as JSON.
def json_converter(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


# Load a JSON file, or return a default if it is missing.
def load_json(path, default):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return default
    return default


# Save data to a JSON file.
def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=json_converter)


image_registry = load_json(IMAGE_REGISTRY_FILE, {"images": []})
if "images" not in image_registry:
    image_registry = {"images": []}

attachment_registry = load_json(ATTACHMENT_REGISTRY_FILE, {"attachments": []})
if "attachments" not in attachment_registry:
    attachment_registry = {"attachments": []}

image_lookup = {i["image_hash"]: i["file_name"] for i in image_registry["images"]}
attachment_lookup = {a["file_hash"]: a["saved_name"] for a in attachment_registry["attachments"]}


QUOTE_MARKERS = [
    r"^-+\s*Original Message\s*-+",
    r"^-+\s*Alkuperäinen viesti\s*-+",
    r"^On .+ wrote:$",
    r"^_{10,}",
]

QUOTE_PATTERN = re.compile("|".join(QUOTE_MARKERS), re.IGNORECASE | re.MULTILINE)


# Keep only the new part of a reply, not the quoted older messages.
def strip_quoted_text(body):
    if not body:
        return ""

    match = QUOTE_PATTERN.search(body)

    if match and match.start() > 200:
        body = body[:match.start()]

    lines = [ln for ln in body.split("\n") if not ln.strip().startswith(">")]

    return "\n".join(lines).strip()


# Read the message text from a saved .eml file.
def read_body_from_eml(eml_file):
    try:
        msg = email.message_from_bytes(eml_file.read_bytes(), policy=policy.default)
    except Exception as e:
        print("  eml parse error:", e)
        return "", "error"

    plain_parts = []
    html_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                continue

            ctype = part.get_content_type()

            try:
                payload = part.get_payload(decode=True)
            except Exception:
                continue

            if not payload:
                continue

            if ctype == "text/plain":
                plain_parts.append(payload.decode("utf-8", errors="ignore"))
            elif ctype == "text/html":
                html_parts.append(payload.decode("utf-8", errors="ignore"))
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                if msg.get_content_type() == "text/html":
                    html_parts.append(payload.decode("utf-8", errors="ignore"))
                else:
                    plain_parts.append(payload.decode("utf-8", errors="ignore"))
        except Exception:
            pass

    plain_text = "\n".join(plain_parts).strip()
    html_text = html_to_text("\n".join(html_parts)) if html_parts else ""

    plain_ok = plain_text and not looks_like_raw_mime(plain_text)
    html_ok = html_text and not looks_like_raw_mime(html_text)

    if plain_ok:
        return plain_text, "plain"

    if html_ok:
        return html_text, "html"

    candidate = html_text if html_text else plain_text

    if candidate:
        cleaned = re.sub(r"[A-Za-z0-9+/=]{200,}", " ", candidate)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned[:50000], "cleaned"

    return "", "none"


# Save an attachment once, even if it appears in many e-mails.
def save_attachment(attachment, source_email):
    try:
        data = attachment.data
    except Exception as e:
        print("  Attachment read error:", e)
        return None

    if not isinstance(data, (bytes, bytearray)):
        return None

    raw_name = attachment.longFilename or attachment.shortFilename or "unnamed"

    name = clean_text(raw_name)
    name = Path(name).name
    name = re.sub(r"[/\\:]", "_", name)

    if not name:
        name = "unnamed"

    extension = Path(name).suffix.lower()

    content_hash = bytes_hash(data)

    is_image = extension in IMAGE_EXTENSIONS

    if is_image and content_hash in image_lookup:
        return {
            "original_name": name,
            "saved_name": image_lookup[content_hash],
            "type": "image",
            "duplicate": True
        }

    if not is_image and content_hash in attachment_lookup:
        return {
            "original_name": name,
            "saved_name": attachment_lookup[content_hash],
            "type": "document",
            "duplicate": True
        }

    if is_image:
        saved_name = f"email_image_{len(image_lookup) + 1}{extension}"
        saved_path = EMAIL_IMAGE_FOLDER / saved_name
    else:
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(name).stem)[:60]
        saved_name = f"email_att_{len(attachment_lookup) + 1}_{safe_stem}{extension}"
        saved_path = ATTACHMENT_FOLDER / saved_name

    try:
        with open(saved_path, "wb") as f:
            f.write(data)
    except Exception as e:
        print("  Attachment save error:", e)
        return None

    if is_image:
        image_lookup[content_hash] = saved_name
        image_registry["images"].append({
            "image_hash": content_hash,
            "file_name": saved_name,
            "source_document": source_email,
            "page_number": None
        })
    else:
        attachment_lookup[content_hash] = saved_name
        attachment_registry["attachments"].append({
            "file_hash": content_hash,
            "saved_name": saved_name,
            "original_name": name,
            "source_document": source_email
        })

    return {
        "original_name": name,
        "saved_name": saved_name,
        "type": "image" if is_image else "document",
        "duplicate": False
    }


# Read one e-mail: sender, date, subject, text and attachments.
def extract_email(msg_file):
    current_hash = file_hash(msg_file)

    file_name = msg_file.name

    if check_document_exists(file_name, current_hash):
        return "skipped"

    msg = extract_msg.Message(str(msg_file))

    relative = msg_file.relative_to(RAW_FOLDER)
    year_folder = relative.parts[0] if relative.parts else ""
    mailbox_folder = "/".join(relative.parts[1:-1])

    body = clean_text(msg.body)
    body_source = "msg" if body else ""

    if body and looks_like_raw_mime(body):
        body = ""
        body_source = ""

    if not body:
        eml_file = msg_file.with_suffix(".eml")
        if eml_file.exists():
            body, body_source = read_body_from_eml(eml_file)
        else:
            body_source = "no eml file"

    email_data = {
        "file_name": file_name,
        "file_hash": current_hash,
        "file_type": "email",
        "processed_date": datetime.now().isoformat(),

        "archive_year": year_folder,
        "mailbox_folder": mailbox_folder,
        "source_path": str(relative),

        "subject": clean_text(msg.subject),
        "sender": clean_text(msg.sender),
        "to": clean_text(msg.to),
        "cc": clean_text(msg.cc),
        "date": clean_text(msg.date),

        "body": strip_quoted_text(body),
        "body_source": body_source,
        "body_length": len(body),

        "attachments": []
    }

    for attachment in msg.attachments:
        result = save_attachment(attachment, file_name)
        if result:
            email_data["attachments"].append(result)

    msg.close()

    return email_data


msg_files = sorted(RAW_FOLDER.rglob("*.msg"))

print("\n==============================")
print("EMAIL FILES FOUND:", len(msg_files))
print("==============================")

if len(msg_files) == 0:
    print("No .msg files found in", RAW_FOLDER)
    exit()


documents = load_json(JSON_OUTPUT, [])

new_documents = []
skipped_count = 0
failed_files = []

for index, msg_file in enumerate(msg_files, start=1):
    if index % 100 == 0:
        print(f"  processed {index} of {len(msg_files)}")

    try:
        result = extract_email(msg_file)

    except Exception as e:
        print("FAILED:", msg_file.name, "->", e)
        failed_files.append({
            "file": msg_file.name,
            "path": str(msg_file),
            "reason": str(e)
        })
        continue

    if result == "skipped":
        skipped_count += 1
    elif result:
        new_documents.append(result)


new_names = {d.get("file_name") for d in new_documents}
documents = [d for d in documents if d.get("file_name") not in new_names]
documents.extend(new_documents)

save_json(JSON_OUTPUT, documents)
save_json(FAILED_LOG, failed_files)

for doc in new_documents:
    if not check_document_exists(doc["file_name"], doc["file_hash"]):
        add_document_record(
            file_name=doc["file_name"],
            file_type="email",
            file_hash=doc["file_hash"]
        )

save_json(IMAGE_REGISTRY_FILE, image_registry)
save_json(ATTACHMENT_REGISTRY_FILE, attachment_registry)


with_body = sum(1 for d in new_documents if d["body"].strip())
with_attachments = sum(1 for d in new_documents if d["attachments"])
total_attachments = sum(len(d["attachments"]) for d in new_documents)

new_docs_saved = sum(
    1 for d in new_documents for a in d["attachments"]
    if a["type"] == "document" and not a["duplicate"]
)
new_images_saved = sum(
    1 for d in new_documents for a in d["attachments"]
    if a["type"] == "image" and not a["duplicate"]
)
duplicates = sum(
    1 for d in new_documents for a in d["attachments"] if a["duplicate"]
)

body_sources = {}
for d in new_documents:
    key = d.get("body_source", "unknown")
    body_sources[key] = body_sources.get(key, 0) + 1

by_year = {}
for d in new_documents:
    by_year[d["archive_year"]] = by_year.get(d["archive_year"], 0) + 1

lengths = sorted(d["body_length"] for d in new_documents)

print("\n==============================")
print("EMAIL EXTRACTION SUMMARY")
print("==============================")
print("Messages found        :", len(msg_files))
print("Newly processed       :", len(new_documents))
print("Already in registry   :", skipped_count)
print("Failed                :", len(failed_files))
print("Messages with body    :", with_body)
print("Messages with attach. :", with_attachments)
print("Attachment links      :", total_attachments)
print("New documents saved   :", new_docs_saved)
print("New images saved      :", new_images_saved)
print("Duplicates skipped    :", duplicates)
print("Total in JSON         :", len(documents))

if lengths:
    print("\nBody length:")
    print("  shortest :", lengths[0])
    print("  median   :", lengths[len(lengths) // 2])
    print("  longest  :", lengths[-1])

print("\nBody came from:")
for source in sorted(body_sources):
    print(f"  {source:16} {body_sources[source]}")

print("\nBy archive year:")
for year in sorted(by_year):
    print(f"  {year}: {by_year[year]}")

if failed_files:
    print("\nFiles needing attention:")
    for item in failed_files[:20]:
        print(" -", item["file"], "->", item["reason"])

print("\nJSON       :", JSON_OUTPUT)
print("Attachments:", ATTACHMENT_FOLDER)
print("Images     :", EMAIL_IMAGE_FOLDER)
print("Failed     :", FAILED_LOG)
