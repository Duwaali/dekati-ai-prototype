# Reads plain text, CSV, HTML and XML files.

from registry_utils import add_document_record, check_document_exists
from pathlib import Path
import json
import hashlib
import csv
import io
from datetime import datetime
from collections import Counter

from lxml import etree
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FOLDER = PROJECT_ROOT / "data" / "raw" / "company_data"

EMAIL_ATTACHMENT_FOLDER = PROJECT_ROOT / "data" / "processed" / "email" / "email_attachments"
EMAIL_UNZIPPED_FOLDER = PROJECT_ROOT / "data" / "processed" / "email" / "email_attachments_unzipped"

TEXT_OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed" / "text"

JSON_OUTPUT = TEXT_OUTPUT_FOLDER / "text_extracted_documents.json"

FAILED_LOG = TEXT_OUTPUT_FOLDER / "text_failed_files.json"

TEXT_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# Make a fingerprint of a file to spot duplicates.
def file_hash(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()


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


BINARY_SIGNATURES = [
    (b"%PDF", "pdf"),
    (b"PK\x03\x04", "zip-based"),
    (b"\xd0\xcf\x11\xe0", "old-office"),
    (b"\xff\xd8\xff", "image"),
    (b"\x89PNG", "image"),
    (b"GIF8", "image"),
    (b"\x1f\x8b", "gzip"),
    (b"Rar!", "rar"),
    (b"MZ", "executable"),
    (b"\x7fELF", "executable"),
]


# Work out what kind of file it really is from its first bytes.
def detect_file_kind(file_path):
    try:
        head = file_path.read_bytes()[:512]
    except Exception:
        return "unreadable"

    if not head:
        return "empty"

    for signature, kind in BINARY_SIGNATURES:
        if head.startswith(signature):
            return kind

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            sample = head.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return "binary"

    lowered = sample.lstrip().lower()

    if lowered.startswith("<?xml"):
        return "xml"

    if lowered.startswith("<html") or lowered.startswith("<!doctype html"):
        return "html"

    if lowered.startswith("{\\rt"):
        return "rtf"

    control_chars = sum(
        1 for c in sample
        if ord(c) < 32 and c not in "\n\r\t"
    )

    if control_chars / max(len(sample), 1) > 0.1:
        return "binary"

    return "text"


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
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)


# Read a file as text, trying common encodings.
def read_raw_text(file_path):
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding), encoding
        except (UnicodeDecodeError, LookupError):
            continue

    return file_path.read_bytes().decode("utf-8", errors="ignore"), "utf-8-ignored"


# Read a plain text file.
def handle_plain_text(file_path):
    text, encoding = read_raw_text(file_path)
    return {"content": text.strip(), "encoding": encoding, "structure": "plain"}


# Read an HTML file as plain text.
def handle_html(file_path):
    text, encoding = read_raw_text(file_path)

    soup = BeautifulSoup(text, "html.parser")

    for tag in soup(["script", "style", "head", "meta", "title"]):
        tag.decompose()

    content = soup.get_text(separator="\n")

    lines = [ln.strip() for ln in content.split("\n")]
    lines = [ln for ln in lines if ln]

    return {"content": "\n".join(lines), "encoding": encoding, "structure": "html"}


# Read a CSV file as text rows.
def handle_csv(file_path):
    text, encoding = read_raw_text(file_path)

    try:
        dialect = csv.Sniffer().sniff(text[:4096])
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","

    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = [r for r in reader if any(str(c).strip() for c in r)]
    except Exception:
        return {"content": text.strip(), "encoding": encoding, "structure": "csv-unparsed"}

    if not rows:
        return {"content": "", "encoding": encoding, "structure": "csv-empty"}

    headers = rows[0]
    data_rows = rows[1:]

    numeric_first_row = sum(
        1 for c in headers
        if c.replace(".", "").replace("-", "").replace(",", "").strip().isdigit()
    )

    if headers and numeric_first_row > len(headers) / 2:
        headers = []
        data_rows = rows

    lines = []

    for row in data_rows:
        parts = []
        for i, cell in enumerate(row):
            value = str(cell).strip()
            if not value:
                continue
            if headers and i < len(headers) and str(headers[i]).strip():
                parts.append(f"{str(headers[i]).strip()}: {value}")
            else:
                parts.append(value)
        if parts:
            lines.append(", ".join(parts))

    return {
        "content": "\n".join(lines),
        "encoding": encoding,
        "structure": "csv",
        "row_count": len(data_rows),
        "headers": headers
    }


# Read an XML file as plain text.
def handle_xml(file_path):
    try:
        parser = etree.XMLParser(recover=True)
        tree = etree.parse(str(file_path), parser)
        root = tree.getroot()
    except Exception:
        text, encoding = read_raw_text(file_path)
        return {"content": text.strip(), "encoding": encoding, "structure": "xml-unparsed"}

    if root is None:
        text, encoding = read_raw_text(file_path)
        return {"content": text.strip(), "encoding": encoding, "structure": "xml-empty-root"}

    lines = []

    for element in root.iter():
        if not isinstance(element.tag, str):
            continue

        try:
            tag = etree.QName(element).localname
        except Exception:
            tag = str(element.tag)

        value = (element.text or "").strip()

        try:
            attributes = " ".join(f"{k}={v}" for k, v in element.attrib.items())
        except Exception:
            attributes = ""

        if value or attributes:
            parts = [tag]
            if attributes:
                parts.append(attributes)
            if value:
                parts.append(value)
            lines.append(": ".join(parts))

    return {
        "content": "\n".join(lines),
        "encoding": "xml",
        "structure": "xml",
        "element_count": len(lines)
    }


# Read one text-type file with the right handler.
def extract_text_file(file_path, detected_kind):
    current_hash = file_hash(file_path)

    file_name = file_path.name

    if check_document_exists(file_name, current_hash):
        return "skipped"

    extension = file_path.suffix.lower()

    if detected_kind == "xml" or extension == ".xml":
        result = handle_xml(file_path)
    elif detected_kind == "html" or extension in (".html", ".htm"):
        result = handle_html(file_path)
    elif extension == ".csv":
        result = handle_csv(file_path)
    else:
        result = handle_plain_text(file_path)

    document_data = {
        "file_name": file_name,
        "file_hash": current_hash,
        "file_type": "text",
        "file_extension": extension or "(none)",
        "detected_kind": detected_kind,
        "source_type": classify_source(file_path),
        "source_path": str(file_path),
        "processed_date": datetime.now().isoformat(),
        "character_count": len(result.get("content", "")),
        **result
    }

    add_document_record(
        file_name=file_name,
        file_type="text",
        file_hash=current_hash
    )

    return document_data


TEXT_EXTENSIONS = {".txt", ".csv", ".xml", ".dat", ".log", ".ini", ".html", ".htm"}

OTHER_READER_EXTENSIONS = {
    ".pdf", ".docx", ".dotx", ".doc", ".xlsx", ".xlsm", ".xls",
    ".pptx", ".ppt", ".msg", ".eml", ".zip", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".jfif", ".webp",
    ".mp4", ".mov", ".exe", ".hex", ".p7m", ".mat", ".stp",
}

folders = [RAW_FOLDER, EMAIL_ATTACHMENT_FOLDER, EMAIL_UNZIPPED_FOLDER]

candidates = []

for folder in folders:
    if not folder.exists():
        continue
    for f in folder.rglob("*"):
        if not f.is_file() or f.name.startswith("~$"):
            continue
        candidates.append(f)

all_files = []
detected_kinds = {}
skipped_kinds = Counter()

for f in candidates:
    extension = f.suffix.lower()

    if extension in TEXT_EXTENSIONS:
        kind = detect_file_kind(f)
        all_files.append(f)
        detected_kinds[f] = kind
        continue

    if extension in OTHER_READER_EXTENSIONS:
        continue

    kind = detect_file_kind(f)

    if kind in ("text", "xml", "html", "rtf"):
        all_files.append(f)
        detected_kinds[f] = kind
    else:
        skipped_kinds[kind] += 1

all_files = sorted(set(all_files))

print("\n==============================")
print("TEXT FILES FOUND:", len(all_files))
print("==============================")

for ext, count in Counter(f.suffix.lower() or "(no extension)" for f in all_files).most_common():
    print(f"  {count:5}  {ext}")

if skipped_kinds:
    print("\nSkipped (not text content):")
    for kind, count in skipped_kinds.most_common():
        print(f"  {count:5}  {kind}")

if len(all_files) == 0:
    print("\nNo text files found")
    exit()


documents = load_json(JSON_OUTPUT, [])

new_documents = []
skipped_count = 0
failed_files = []

for index, file_path in enumerate(all_files, start=1):
    if index % 50 == 0:
        print(f"  processed {index} of {len(all_files)}")

    try:
        result = extract_text_file(file_path, detected_kinds.get(file_path, "text"))

    except Exception as e:
        print("FAILED:", file_path.name, "->", e)
        failed_files.append({
            "file": file_path.name,
            "path": str(file_path),
            "reason": str(e)
        })
        continue

    if result == "skipped":
        skipped_count += 1
    elif result:
        new_documents.append(result)


new_names = {d["file_name"] for d in new_documents}
documents = [d for d in documents if d.get("file_name") not in new_names]
documents.extend(new_documents)

save_json(JSON_OUTPUT, documents)
save_json(FAILED_LOG, failed_files)


by_source = Counter(d.get("source_type", "unknown") for d in new_documents)
by_structure = Counter(d.get("structure", "unknown") for d in new_documents)
by_detected = Counter(d.get("detected_kind", "unknown") for d in new_documents)

empty_files = [d["file_name"] for d in new_documents if d["character_count"] == 0]
total_chars = sum(d["character_count"] for d in new_documents)

print("\n==============================")
print("TEXT EXTRACTION SUMMARY")
print("==============================")
print("Files found         :", len(all_files))
print("Newly processed     :", len(new_documents))
print("Already in registry :", skipped_count)
print("Failed              :", len(failed_files))
print("Empty files         :", len(empty_files))
print("Total characters    :", f"{total_chars:,}")
print("Total docs in JSON  :", len(documents))

print("\nBy source:")
for source, count in by_source.most_common():
    print(f"  {source:25} {count}")

print("\nBy structure:")
for structure, count in by_structure.most_common():
    print(f"  {structure:25} {count}")

print("\nBy detected content type:")
for kind, count in by_detected.most_common():
    print(f"  {kind:25} {count}")

if failed_files:
    print("\nFiles needing attention:")
    for item in failed_files[:20]:
        print(" -", item["file"], "->", item["reason"])

print("\nJSON  :", JSON_OUTPUT)
print("Failed:", FAILED_LOG)
