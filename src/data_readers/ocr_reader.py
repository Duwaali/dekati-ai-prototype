#!/usr/bin/env python3
# Reads text from images with OCR (Finnish and English) and drops unreadable results.


import json
import hashlib
from pathlib import Path
from PIL import Image
import pytesseract
from datetime import datetime
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED / "ocr"

MIN_TEXT_LENGTH = 10
MIN_WORD_COUNT = 2
MIN_MEANINGFUL_CHARS = 0.5

IMAGE_LOCATIONS = [
    ("ppt_images",        PROCESSED / "ppt" / "ppt_images"),
    ("pdf_images",        PROCESSED / "pdf" / "pdf_images"),
    ("word_images",       PROCESSED / "word" / "word_images"),
    ("excel_images",      PROCESSED / "excel" / "excel_images"),
    ("email_images",      PROCESSED / "email" / "email_images"),
    ("email_attachments", PROCESSED / "email" / "email_attachments"),
]

SAVE_EVERY = 100
LANGUAGES = "fin+eng"


# Make a fingerprint of an image file to spot duplicates.
def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

# Score how readable the OCR text is (0 to 1).
def get_image_quality_score(text):
    if not text or len(text) < MIN_TEXT_LENGTH:
        return 0.0

    meaningful = sum(1 for c in text if c.isalnum() or c.isspace())
    meaningful_ratio = meaningful / len(text) if text else 0

    words = text.split()
    word_count = len(words)

    scores = {
        'length': min(len(text) / 500, 1.0),
        'meaningful': meaningful_ratio,
        'words': min(word_count / 5, 1.0)
    }

    quality = (
        scores['length'] * 0.3 +
        scores['meaningful'] * 0.4 +
        scores['words'] * 0.3
    )

    return quality

# Keep OCR text only if it looks like real text.
def filter_ocr_quality(text):
    if not text or len(text) < MIN_TEXT_LENGTH:
        return False, 0.0, "too_short"

    words = text.split()
    if len(words) < MIN_WORD_COUNT:
        return False, 0.0, "too_few_words"

    meaningful = sum(1 for c in text if c.isalnum() or c.isspace())
    meaningful_ratio = meaningful / len(text) if text else 0

    if meaningful_ratio < MIN_MEANINGFUL_CHARS:
        return False, 0.0, "too_much_noise"

    quality = get_image_quality_score(text)

    if quality >= 0.3:
        return True, quality, "accepted"
    else:
        return False, quality, "low_quality"

# Read the text from one image with OCR.
def ocr_image(image_path, location_name, processed_hashes):
    try:
        file_hash = calculate_file_hash(image_path)

        if file_hash in processed_hashes:
            return {
                'file_name': image_path.name,
                'file_hash': file_hash,
                'location': location_name,
                'source_path': str(image_path),
                'processed_date': datetime.now().isoformat(),
                'result': 'duplicate_skipped',
                'reason': 'Already processed this image hash',
                'text': '',
                'character_count': 0,
                'quality_score': 0.0
            }

        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size

            if width < 60 or height < 60:
                return {
                    'file_name': image_path.name,
                    'file_hash': file_hash,
                    'location': location_name,
                    'source_path': str(image_path),
                    'processed_date': datetime.now().isoformat(),
                    'result': 'too_small',
                    'reason': f'Image too small: {width}x{height}',
                    'text': '',
                    'character_count': 0,
                    'quality_score': 0.0
                }

            text = pytesseract.image_to_string(img, lang=LANGUAGES)
            text = text.strip()

        keep, quality_score, filter_reason = filter_ocr_quality(text)

        if not keep:
            return {
                'file_name': image_path.name,
                'file_hash': file_hash,
                'location': location_name,
                'source_path': str(image_path),
                'processed_date': datetime.now().isoformat(),
                'result': 'low_quality_rejected',
                'reason': filter_reason,
                'text': '',
                'character_count': 0,
                'quality_score': quality_score
            }

        return {
            'file_name': image_path.name,
            'file_hash': file_hash,
            'location': location_name,
            'source_path': str(image_path),
            'processed_date': datetime.now().isoformat(),
            'result': 'text_extracted',
            'reason': 'Successfully extracted text',
            'text': text,
            'character_count': len(text),
            'quality_score': quality_score
        }

    except Exception as e:
        return {
            'file_name': image_path.name,
            'file_hash': '',
            'location': location_name,
            'source_path': str(image_path),
            'processed_date': datetime.now().isoformat(),
            'result': 'processing_error',
            'reason': str(e),
            'text': '',
            'character_count': 0,
            'quality_score': 0.0
        }


# Run OCR on all saved images and keep the good results.
def run_ocr_pipeline():
    print("=" * 70)
    print("OCR PIPELINE WITH QUALITY FILTERING")
    print("=" * 70)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_hashes = set()
    existing_results = []
    existing_file = OUTPUT_DIR / "ocr_extracted_documents.json"

    if existing_file.exists():
        print(f"Loading existing results from {existing_file.name}...")
        with open(existing_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'documents' in data:
                existing_results = data['documents']
            elif isinstance(data, list):
                existing_results = data

            for doc in existing_results:
                if doc.get('file_hash'):
                    existing_hashes.add(doc['file_hash'])

            print(f"  Found {len(existing_results)} existing results")
            print(f"  {len(existing_hashes)} unique file hashes")
    print()

    all_results = existing_results.copy()
    stats = defaultdict(int)

    for location_name, image_dir in IMAGE_LOCATIONS:
        if not image_dir.exists():
            print(f"Skipping {location_name}: directory not found")
            continue

        print(f"Processing {location_name}...")
        image_files = sorted([f for f in image_dir.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])

        if not image_files:
            print(f"  No images found in {location_name}")
            print()
            continue

        print(f"  Found {len(image_files)} images")

        location_results = []
        for i, image_file in enumerate(image_files, 1):
            result = ocr_image(image_file, location_name, existing_hashes)

            if result:
                location_results.append(result)
                all_results.append(result)

                stats[result['result']] += 1
                if result['result'] == 'duplicate_skipped':
                    existing_hashes.add(result['file_hash'])
                else:
                    existing_hashes.add(result['file_hash'])

                if i % 20 == 0:
                    print(f"    Processed {i}/{len(image_files)}")

            if i % SAVE_EVERY == 0:
                with open(existing_file, 'w', encoding='utf-8') as f:
                    json.dump({'documents': all_results}, f, indent=2, ensure_ascii=False)
                print(f"    Saved progress at image {i}")

        print(f"  ✓ Completed {len(location_results)} results")
        print()

    print("Saving final results...")
    with open(existing_file, 'w', encoding='utf-8') as f:
        json.dump({'documents': all_results}, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved to {existing_file}")
    print()

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for status, count in sorted(stats.items()):
        print(f"  {status}: {count}")
    print()
    print(f"Total documents in output: {len(all_results)}")
    print(f"Total unique file hashes: {len(existing_hashes)}")
    print()

if __name__ == "__main__":
    run_ocr_pipeline()
