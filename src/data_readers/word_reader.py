# Reads Word documents: paragraphs, tables, headers, comments, text boxes, charts and images.

import subprocess
import tempfile
import shutil
from pathlib import Path
import json
import hashlib
import zipfile
from datetime import datetime

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
from lxml import etree

from registry_utils import add_document_record, check_document_exists


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FOLDER = PROJECT_ROOT / "data" / "raw" / "company_data"

EMAIL_ATTACHMENT_FOLDER = PROJECT_ROOT / "data" / "processed" / "email" / "email_attachments"
EMAIL_UNZIPPED_FOLDER = PROJECT_ROOT / "data" / "processed" / "email" / "email_attachments_unzipped"

WORD_OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed" / "word"

IMAGE_FOLDER = WORD_OUTPUT_FOLDER / "word_images"

JSON_OUTPUT = WORD_OUTPUT_FOLDER / "word_extracted_documents.json"

FAILED_LOG = WORD_OUTPUT_FOLDER / "word_failed_files.json"

REGISTRY_FOLDER = PROJECT_ROOT / "data" / "processed" / "registry"

IMAGE_REGISTRY_FILE = REGISTRY_FOLDER / "image_registry.json"

WORD_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
REGISTRY_FOLDER.mkdir(parents=True, exist_ok=True)


# Make a fingerprint of a file to spot duplicates.
def file_hash(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()


# Make a fingerprint of an image to spot duplicates.
def image_hash(data):
    return hashlib.sha256(data).hexdigest()


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
            print("Warning: corrupted JSON, starting fresh:", path)
            return default
    return default


# Save data to a JSON file.
def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)


# Read one internal XML part of a Word file.
def read_xml_from_docx(word_file, xml_file):
    try:
        with zipfile.ZipFile(word_file) as docx:
            if xml_file in docx.namelist():
                return etree.fromstring(docx.read(xml_file))
    except Exception:
        return None
    return None


# Read the document's title, author and dates.
def extract_metadata(document):
    p = document.core_properties
    return {
        "title": p.title,
        "author": p.author,
        "subject": p.subject,
        "keywords": p.keywords,
        "category": p.category,
        "comments": p.comments,
        "created": str(p.created),
        "modified": str(p.modified)
    }


# Save an image from a document once.
def save_word_image(image_bytes, extension, source_document, image_registry):
    img_hash = image_hash(image_bytes)

    for img in image_registry["images"]:
        if img.get("image_hash") == img_hash:
            return img.get("file_name")

    image_name = f"word_image_{len(image_registry['images']) + 1}.{extension}"

    image_path = IMAGE_FOLDER / image_name

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    image_registry["images"].append({
        "image_hash": img_hash,
        "file_name": image_name,
        "source_document": source_document,
        "page_number": None
    })

    return image_name


# Read the page headers.
def extract_headers(document):
    headers = []
    for section in document.sections:
        try:
            for paragraph in section.header.paragraphs:
                text = paragraph.text.strip()
                if text and text not in headers:
                    headers.append(text)
        except Exception:
            continue
    return headers


# Read the page footers.
def extract_footers(document):
    footers = []
    for section in document.sections:
        try:
            for paragraph in section.footer.paragraphs:
                text = paragraph.text.strip()
                if text and text not in footers:
                    footers.append(text)
        except Exception:
            continue
    return footers


# Read the links in the document.
def extract_hyperlinks(word_file):
    hyperlinks = []
    try:
        with zipfile.ZipFile(word_file) as docx:
            if "word/_rels/document.xml.rels" not in docx.namelist():
                return hyperlinks
            rels = etree.fromstring(docx.read("word/_rels/document.xml.rels"))
            for rel in rels:
                if "hyperlink" in rel.attrib.get("Type", ""):
                    hyperlinks.append({"url": rel.attrib.get("Target")})
    except Exception as e:
        print("  Hyperlink error:", e)
    return hyperlinks


# Read the review comments in the document.
def extract_comments(word_file):
    comments = []

    root = read_xml_from_docx(word_file, "word/comments.xml")

    if root is None:
        return comments

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    try:
        for comment in root.xpath("//w:comment", namespaces=ns):
            text = "".join(comment.xpath(".//w:t/text()", namespaces=ns)).strip()
            if text:
                comments.append(text)
    except Exception as e:
        print("  Comment error:", e)

    return comments


# Read the text inside text boxes.
def extract_textboxes(word_file):
    textboxes = []

    root = read_xml_from_docx(word_file, "word/document.xml")

    if root is None:
        return textboxes

    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "v": "urn:schemas-microsoft-com:vml",
        "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
    }

    try:
        for box in root.xpath("//v:textbox", namespaces=ns):
            text = " ".join(box.itertext()).strip()
            if text and text not in textboxes:
                textboxes.append(text)
    except Exception:
        pass

    try:
        for box in root.xpath("//wps:txbx", namespaces=ns):
            text = " ".join(box.itertext()).strip()
            if text and text not in textboxes:
                textboxes.append(text)
    except Exception:
        pass

    return textboxes


# Read the data inside charts.
def extract_charts(word_file):
    charts = []
    try:
        with zipfile.ZipFile(word_file) as docx:
            for item in docx.namelist():
                if item.startswith("word/charts/"):
                    charts.append(item)
    except Exception:
        pass
    return charts


# List files embedded inside the document.
def extract_embedded_objects(word_file):
    objects = []
    try:
        with zipfile.ZipFile(word_file) as docx:
            for item in docx.namelist():
                if item.startswith("word/embeddings/"):
                    objects.append(item)
    except Exception:
        pass
    return objects


# Save the images in the document.
def extract_images(document, file_name, image_registry):
    images = []

    for rel in document.part.rels.values():
        if "image" not in rel.target_ref:
            continue

        if rel.is_external:
            print("  External image skipped:", rel.target_ref)
            continue

        try:
            image_part = rel.target_part
        except Exception:
            print("  Invalid image relationship skipped")
            continue

        try:
            image_bytes = image_part.blob
            extension = image_part.content_type.split("/")[-1]
            image_name = save_word_image(image_bytes, extension, file_name, image_registry)
            images.append({"file": image_name})
        except Exception as e:
            print("  Image save error:", e)

    return images


SOFFICE_AVAILABLE = shutil.which("soffice") is not None


# Convert an old .doc file to .docx so it can be read.
def convert_to_docx(source_file):
    if not SOFFICE_AVAILABLE:
        raise RuntimeError("LibreOffice (soffice) not installed - cannot convert this file")

    temp_folder = Path(tempfile.mkdtemp())

    subprocess.run(
        [
            "soffice",
            "--headless",
            "--convert-to", "docx",
            "--outdir", str(temp_folder),
            str(source_file)
        ],
        check=True,
        capture_output=True,
        timeout=120
    )

    converted_file = temp_folder / (source_file.stem + ".docx")

    if not converted_file.exists():
        raise RuntimeError("Conversion produced no file")

    return converted_file


# Go through paragraphs and tables in their real order.
def iter_block_items(document):
    body = document.element.body

    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)

        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


# Read one Word document completely.
def extract_word(word_file, image_registry):
    current_hash = file_hash(word_file)

    file_name = word_file.name

    if check_document_exists(file_name, current_hash):
        print("Skipping already processed:", file_name)
        return "skipped"

    print("\nProcessing:", file_name)

    source_type = classify_source(word_file)

    input_file = word_file

    if word_file.suffix.lower() in (".dotx", ".doc"):
        print("  Converting to docx...")
        input_file = convert_to_docx(word_file)

    document = Document(input_file)

    document_data = {
        "file_name": file_name,
        "file_hash": current_hash,
        "file_type": "word",
        "source_type": source_type,
        "metadata": extract_metadata(document),
        "processed_date": datetime.now().isoformat(),
        "headers": extract_headers(document),
        "footers": extract_footers(document),
        "hyperlinks": extract_hyperlinks(input_file),
        "comments": extract_comments(input_file),
        "textboxes": extract_textboxes(input_file),
        "charts": extract_charts(input_file),
        "embedded_objects": extract_embedded_objects(input_file),
        "content": [],
        "images": []
    }


    current_heading = ""
    block_index = 0

    for block in iter_block_items(document):
        block_index += 1

        if isinstance(block, Paragraph):
            text = block.text.strip()

            if not text:
                continue

            style_name = ""
            try:
                style_name = block.style.name or ""
            except Exception:
                style_name = ""

            if style_name.lower().startswith("heading") or style_name.lower() == "title":
                current_heading = text

            document_data["content"].append({
                "order": block_index,
                "type": style_name if style_name else "Normal",
                "heading": current_heading,
                "text": text
            })

        else:
            rows = []

            try:
                for row in block.rows:
                    rows.append([cell.text.strip() for cell in row.cells])
            except Exception as e:
                print("  Table error:", e)
                continue

            if rows:
                document_data["content"].append({
                    "order": block_index,
                    "type": "table",
                    "heading": current_heading,
                    "data": rows
                })


    document_data["images"] = extract_images(document, file_name, image_registry)

    return document_data


company_docx = list(RAW_FOLDER.rglob("*.docx"))
company_dotx = list(RAW_FOLDER.rglob("*.dotx"))
company_doc = list(RAW_FOLDER.rglob("*.doc"))

email_docx = list(EMAIL_ATTACHMENT_FOLDER.glob("*.docx")) if EMAIL_ATTACHMENT_FOLDER.exists() else []
email_doc = list(EMAIL_ATTACHMENT_FOLDER.glob("*.doc")) if EMAIL_ATTACHMENT_FOLDER.exists() else []

zip_docx = list(EMAIL_UNZIPPED_FOLDER.rglob("*.docx")) if EMAIL_UNZIPPED_FOLDER.exists() else []
zip_doc = list(EMAIL_UNZIPPED_FOLDER.rglob("*.doc")) if EMAIL_UNZIPPED_FOLDER.exists() else []

all_files = company_docx + company_dotx + company_doc + email_docx + email_doc + zip_docx + zip_doc

all_files = [f for f in all_files if not f.name.startswith("~$")]

print("\n==============================")
print("WORD FILES FOUND:", len(all_files))
print("==============================")
print("  company .docx/.dotx/.doc :", len(company_docx) + len(company_dotx) + len(company_doc))
print("  email attachments        :", len(email_docx) + len(email_doc))
print("  from unzipped emails     :", len(zip_docx) + len(zip_doc))

if not SOFFICE_AVAILABLE and (company_dotx or company_doc or email_doc or zip_doc):
    print("\nWARNING: LibreOffice not found - .dotx/.doc files will fail")

if len(all_files) == 0:
    print("\nNo Word files found")
    exit()


documents = load_json(JSON_OUTPUT, [])
image_registry = load_json(IMAGE_REGISTRY_FILE, {"images": []})

if "images" not in image_registry:
    image_registry = {"images": []}

new_documents = []
skipped_count = 0
failed_files = []

for word_file in all_files:
    try:
        result = extract_word(word_file, image_registry)

    except Exception as e:
        reason = str(e)

        if "encrypted" in reason.lower() or "password" in reason.lower():
            reason = "Password protected / encrypted"

        print("FAILED:", word_file.name, "->", reason)

        failed_files.append({
            "file": word_file.name,
            "path": str(word_file),
            "reason": reason
        })
        continue

    if result == "skipped":
        skipped_count += 1
    elif result:
        new_documents.append(result)


for doc in new_documents:
    documents = [old for old in documents if old.get("file_name") != doc["file_name"]]
    documents.append(doc)


save_json(JSON_OUTPUT, documents)
save_json(FAILED_LOG, failed_files)

for doc in new_documents:
    if not check_document_exists(doc["file_name"], doc["file_hash"]):
        add_document_record(
            file_name=doc["file_name"],
            file_type="word",
            file_hash=doc["file_hash"]
        )

save_json(IMAGE_REGISTRY_FILE, image_registry)


by_source = {}
for d in new_documents:
    key = d.get("source_type", "unknown")
    by_source[key] = by_source.get(key, 0) + 1

paragraph_count = sum(
    1 for d in new_documents for c in d["content"] if c["type"] != "table"
)
table_count = sum(
    1 for d in new_documents for c in d["content"] if c["type"] == "table"
)
comment_count = sum(len(d["comments"]) for d in new_documents)
textbox_count = sum(len(d["textboxes"]) for d in new_documents)

print("\n==============================")
print("WORD EXTRACTION SUMMARY")
print("==============================")
print("Files found          :", len(all_files))
print("Newly processed      :", len(new_documents))
print("Already in registry  :", skipped_count)
print("Failed               :", len(failed_files))
print("Paragraphs extracted :", paragraph_count)
print("Tables extracted     :", table_count)
print("Comments extracted   :", comment_count)
print("Text boxes extracted :", textbox_count)
print("Total docs in JSON   :", len(documents))
print("Images in registry   :", len(image_registry["images"]))

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
