# Reads Excel workbooks: cell values, comments and images.

from registry_utils import add_document_record, check_document_exists
from pathlib import Path
import json
import hashlib
import zipfile
import subprocess
import tempfile
import shutil
from datetime import datetime

from lxml import etree
import openpyxl
from openpyxl.worksheet.worksheet import Worksheet


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FOLDER = PROJECT_ROOT / "data" / "raw" / "company_data"

EMAIL_ATTACHMENT_FOLDER = PROJECT_ROOT / "data" / "processed" / "email" / "email_attachments"
EMAIL_UNZIPPED_FOLDER = PROJECT_ROOT / "data" / "processed" / "email" / "email_attachments_unzipped"

EXCEL_OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed" / "excel"

IMAGE_FOLDER = EXCEL_OUTPUT_FOLDER / "excel_images"

JSON_OUTPUT = EXCEL_OUTPUT_FOLDER / "excel_extracted_documents.json"

FAILED_LOG = EXCEL_OUTPUT_FOLDER / "excel_failed_files.json"

REGISTRY_FOLDER = PROJECT_ROOT / "data" / "processed" / "registry"

IMAGE_REGISTRY_FILE = REGISTRY_FOLDER / "image_registry.json"

EXCEL_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
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


SOFFICE_AVAILABLE = shutil.which("soffice") is not None


# Convert an old .xls file to .xlsx so it can be read.
def convert_xls_to_xlsx(source_file):
    if not SOFFICE_AVAILABLE:
        raise RuntimeError("LibreOffice (soffice) not installed - cannot convert .xls")

    temp_folder = Path(tempfile.mkdtemp())

    subprocess.run(
        [
            "soffice",
            "--headless",
            "--convert-to", "xlsx",
            "--outdir", str(temp_folder),
            str(source_file)
        ],
        check=True,
        capture_output=True,
        timeout=120
    )

    converted = temp_folder / (source_file.stem + ".xlsx")

    if not converted.exists():
        raise RuntimeError("Conversion produced no file")

    return converted


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


# Look up an image that was already saved.
def get_existing_image(img_hash):
    for img in image_registry["images"]:
        if img.get("image_hash") == img_hash:
            return img.get("file_name")
    return None


# Save an image from a workbook once.
def save_excel_image(image, source_document):
    try:
        image.ref.seek(0)
        image_bytes = image.ref.read()
    except Exception as e:
        print("  Image read error:", e)
        return None

    img_hash = image_hash(image_bytes)

    existing = get_existing_image(img_hash)

    if existing:
        return existing

    extension = "png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        extension = "jpeg"
    elif image_bytes[:4] == b"GIF8":
        extension = "gif"

    image_name = f"excel_image_{len(image_registry['images']) + 1}.{extension}"

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


# Read the comments attached to cells.
def extract_cell_comments(file):
    comments = []

    try:
        with zipfile.ZipFile(file) as z:
            for name in z.namelist():
                if "comment" not in name.lower() or not name.endswith(".xml"):
                    continue

                try:
                    root = etree.fromstring(z.read(name))
                except Exception:
                    continue

                for node in root.iter():
                    if not isinstance(node.tag, str):
                        continue

                    try:
                        tag = etree.QName(node).localname
                    except Exception:
                        continue

                    if tag not in ("comment", "threadedComment"):
                        continue

                    text = " ".join(
                        t.strip() for t in node.itertext() if t.strip()
                    )

                    if text:
                        comments.append({
                            "cell": node.get("ref") or "",
                            "text": text
                        })

    except Exception as e:
        print("  Comment extraction error:", e)

    return comments


# Read one workbook: every sheet, its rows, comments and images.
def extract_excel(file):
    current_hash = file_hash(file)

    if check_document_exists(file.name, current_hash):
        return "skipped"

    print("\nProcessing:", file.name)

    source_type = classify_source(file)

    working_file = file

    if file.suffix.lower() == ".xls":
        print("  Converting .xls to .xlsx...")
        working_file = convert_xls_to_xlsx(file)

    workbook_data = {
        "file_name": file.name,
        "file_hash": current_hash,
        "file_type": "excel",
        "source_type": source_type,
        "processed_date": datetime.now().isoformat(),
        "comments": extract_cell_comments(working_file),
        "sheets": []
    }

    wb_values = openpyxl.load_workbook(working_file, data_only=True)
    wb_formulas = openpyxl.load_workbook(working_file, data_only=False)

    for sheet_name in wb_values.sheetnames:
        sheet = wb_values[sheet_name]

        sheet_data = {
            "sheet_name": sheet_name,
            "type": "worksheet",
            "hidden": False,
            "rows": [],
            "row_count": 0,
            "column_count": 0,
            "named_tables": [],
            "formulas": [],
            "images": [],
            "charts": []
        }

        try:
            sheet_data["hidden"] = sheet.sheet_state != "visible"
        except Exception:
            pass

        if not isinstance(sheet, Worksheet):
            sheet_data["type"] = "chartsheet"
            workbook_data["sheets"].append(sheet_data)
            continue

        rows = []

        for row in sheet.iter_rows(values_only=True):
            if all(cell is None or str(cell).strip() == "" for cell in row):
                continue

            rows.append([
                "" if cell is None else str(cell).strip()
                for cell in row
            ])

        sheet_data["rows"] = rows
        sheet_data["row_count"] = len(rows)
        sheet_data["column_count"] = max((len(r) for r in rows), default=0)

        try:
            formula_sheet = wb_formulas[sheet_name]

            for row in formula_sheet.iter_rows():
                for cell in row:
                    if cell.data_type == "f":
                        sheet_data["formulas"].append({
                            "cell": cell.coordinate,
                            "formula": str(cell.value)
                        })
        except Exception as e:
            print("  Formula error:", e)

        try:
            for name, table_ref in sheet.tables.items():
                sheet_data["named_tables"].append({
                    "table_name": name,
                    "reference": str(table_ref)
                })
        except Exception as e:
            print("  Named table error:", e)

        try:
            for image in sheet._images:
                name = save_excel_image(image, file.name)
                if name:
                    sheet_data["images"].append({"file": name})
        except Exception as e:
            print("  Image error:", e)

        try:
            for chart in sheet._charts:
                chart_title = ""
                try:
                    if chart.title is not None:
                        chart_title = str(chart.title)
                except Exception:
                    chart_title = ""

                sheet_data["charts"].append({
                    "chart_type": type(chart).__name__,
                    "title": chart_title
                })
        except Exception as e:
            print("  Chart error:", e)

        workbook_data["sheets"].append(sheet_data)

    wb_values.close()
    wb_formulas.close()

    add_document_record(
        file_name=file.name,
        file_type="excel",
        file_hash=current_hash
    )

    return workbook_data


company_files = (
    list(RAW_FOLDER.rglob("*.xlsx"))
    + list(RAW_FOLDER.rglob("*.xlsm"))
    + list(RAW_FOLDER.rglob("*.xls"))
)

email_files = []
if EMAIL_ATTACHMENT_FOLDER.exists():
    email_files = (
        list(EMAIL_ATTACHMENT_FOLDER.glob("*.xlsx"))
        + list(EMAIL_ATTACHMENT_FOLDER.glob("*.xlsm"))
        + list(EMAIL_ATTACHMENT_FOLDER.glob("*.xls"))
    )

zip_files = []
if EMAIL_UNZIPPED_FOLDER.exists():
    zip_files = (
        list(EMAIL_UNZIPPED_FOLDER.rglob("*.xlsx"))
        + list(EMAIL_UNZIPPED_FOLDER.rglob("*.xlsm"))
        + list(EMAIL_UNZIPPED_FOLDER.rglob("*.xls"))
    )

excel_files = company_files + email_files + zip_files
excel_files = [f for f in excel_files if not f.name.startswith("~$")]

legacy_count = sum(1 for f in excel_files if f.suffix.lower() == ".xls")

print("\n==============================")
print("EXCEL FILES FOUND:", len(excel_files))
print("==============================")
print("  from company_data       :", len(company_files))
print("  from email attachments  :", len(email_files))
print("  from unzipped emails    :", len(zip_files))
print("  legacy .xls to convert  :", legacy_count)

if legacy_count and not SOFFICE_AVAILABLE:
    print("\nWARNING: LibreOffice not found - .xls files will fail")

if len(excel_files) == 0:
    print("No Excel files found")
    exit()


documents = load_json(JSON_OUTPUT, [])

new_documents = []
skipped_count = 0
failed_files = []

for excel_file in excel_files:
    try:
        result = extract_excel(excel_file)

    except Exception as e:
        reason = str(e)

        if "password" in reason.lower() or "encrypt" in reason.lower():
            reason = "Password protected / encrypted"

        print("FAILED:", excel_file.name, "->", reason)

        failed_files.append({
            "file": excel_file.name,
            "path": str(excel_file),
            "reason": reason
        })
        continue

    if result == "skipped":
        skipped_count += 1
    elif result:
        new_documents.append(result)


for doc in new_documents:
    documents = [d for d in documents if d.get("file_name") != doc["file_name"]]
    documents.append(doc)

save_json(JSON_OUTPUT, documents)
save_json(FAILED_LOG, failed_files)
save_json(IMAGE_REGISTRY_FILE, image_registry)


by_source = {}
for d in new_documents:
    key = d.get("source_type", "unknown")
    by_source[key] = by_source.get(key, 0) + 1

sheet_count = sum(len(d["sheets"]) for d in new_documents)
row_count = sum(s["row_count"] for d in new_documents for s in d["sheets"])
formula_count = sum(len(s["formulas"]) for d in new_documents for s in d["sheets"])
comment_count = sum(len(d.get("comments", [])) for d in new_documents)

print("\n==============================")
print("EXCEL EXTRACTION SUMMARY")
print("==============================")
print("Files found         :", len(excel_files))
print("Newly processed     :", len(new_documents))
print("Already in registry :", skipped_count)
print("Failed              :", len(failed_files))
print("Sheets extracted    :", sheet_count)
print("Data rows extracted :", row_count)
print("Formulas found      :", formula_count)
print("Cell comments found :", comment_count)
print("Total docs in JSON  :", len(documents))

print("\nNewly processed by source:")
for source, count in by_source.items():
    print(f"  {source:25} {count}")

if failed_files:
    print("\nFiles needing attention:")
    for item in failed_files[:20]:
        print(" -", item["file"], "->", item["reason"])

print("\nJSON  :", JSON_OUTPUT)
print("Images:", IMAGE_FOLDER)
print("Failed:", FAILED_LOG)
