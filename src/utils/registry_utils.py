# Keeps track of which files and images were already processed.

from pathlib import Path
import json
import hashlib
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[2]


REGISTRY_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "registry"
)


DOCUMENT_REGISTRY_FILE = (
    REGISTRY_FOLDER
    / "document_registry.json"
)


IMAGE_REGISTRY_FILE = (
    REGISTRY_FOLDER
    / "image_registry.json"
)


# Make a fingerprint of a file to spot duplicates.
def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(4096),
            b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# Load a JSON file, or return a default if it is missing.
def load_json(file_path, default_data):
    if not file_path.exists():
        return default_data


    try:
        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:
            content = file.read().strip()


            if not content:
                return default_data


            return json.loads(content)


    except json.JSONDecodeError:
        return default_data


# Save data to a JSON file.
def save_json(file_path, data):
    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


# Load the list of processed files.
def load_document_registry():
    return load_json(
        DOCUMENT_REGISTRY_FILE,
        {
            "documents": []
        }
    )


# Save the list of processed files.
def save_document_registry(data):
    save_json(
        DOCUMENT_REGISTRY_FILE,
        data
    )


# Check if a file was already processed.
def check_document_exists(
        file_name,
        file_hash
):
    registry = load_document_registry()


    for document in registry["documents"]:
        if (
            document["file_name"] == file_name
            and
            document["file_hash"] == file_hash
        ):
            return True


    return False


# Add a file to the processed list.
def add_document_record(
        file_name,
        file_type,
        file_hash
):
    registry = load_document_registry()


    registry["documents"].append(
        {

            "file_name": file_name,

            "file_type": file_type,

            "file_hash": file_hash,

            "processed_date":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        }
    )


    save_document_registry(
        registry
    )


# Load the list of saved images.
def load_image_registry():
    return load_json(
        IMAGE_REGISTRY_FILE,
        {
            "images": []
        }
    )


# Save the list of saved images.
def save_image_registry(data):
    save_json(
        IMAGE_REGISTRY_FILE,
        data
    )


# Check if an image was already saved.
def check_image_exists(image_hash):
    registry = load_image_registry()


    for image in registry["images"]:
        if image["image_hash"] == image_hash:
            return True


    return False


# Add an image to the saved list.
def add_image_record(
        image_hash,
        file_name,
        source_document,
        page_number=None
):
    registry = load_image_registry()


    registry["images"].append(
        {

            "image_hash": image_hash,

            "file_name": file_name,

            "source_document": source_document,

            "page_number": page_number

        }
    )


    save_image_registry(
        registry
    )
