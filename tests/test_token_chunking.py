#!/usr/bin/env python3
# Tests for the chunker, using made-up text only.


import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "src" / "data_processing" / "token_chunking.py"
sys.path.insert(0, str(SCRIPT.parent))

import token_chunking as tc

TOKENIZER = tc.load_tokenizer()


# Chunk a text with the real chunker.
def chunk(text, budget=tc.TEXT_BUDGET):
    return tc.chunk_section_text(TOKENIZER, text, budget)


# Get the text of each chunk.
def texts(text, chunks):
    return [text[c["char_start"]:c["char_end"]] for c in chunks]


# Join chunks back together the way they were cut.
def rebuild(text, chunks):
    out, previous_end = [], chunks[0]["char_start"] if chunks else 0
    for c in chunks:
        out.append(text[previous_end:c["char_start"]])
        out.append(text[c["char_start"]:c["char_end"]])
        previous_end = c["char_end"]
    return "".join(out)


# Tests for cutting single sections into chunks.
class ChunkingTests(unittest.TestCase):
    # Check every chunk fits and together they cover the text.
    def assert_valid(self, text, chunks, budget=tc.TEXT_BUDGET):
        self.assertTrue(tc.verify_section_coverage(text, chunks), "coverage failed")
        for c in chunks:
            piece = text[c["char_start"]:c["char_end"]]
            measured = tc.count_tokens(TOKENIZER, piece)
            self.assertEqual(measured, c["text_tokens"], "stored token count is wrong")
            self.assertLessEqual(measured, budget, "chunk over budget")


    # Empty text gives no chunks.
    def test_empty_text(self):
        for text in ["", "   ", "\n\t \n"]:
            chunks = chunk(text)
            self.assertEqual(chunks, [])
            self.assertTrue(tc.verify_section_coverage(text, chunks))

    # A short text stays one chunk.
    def test_short_text(self):
        text = "The cyclone has two connectors."
        chunks = chunk(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(texts(text, chunks), [text])
        self.assert_valid(text, chunks)

    # Text exactly at the limit stays one chunk.
    def test_exactly_at_boundary(self):
        text = " ".join(["hello"] * 10)
        self.assertEqual(tc.count_tokens(TOKENIZER, text), 10)
        chunks = chunk(text, budget=10)
        self.assertEqual(len(chunks), 1)
        self.assert_valid(text, chunks, budget=10)

    # One token over the limit makes a second chunk.
    def test_one_token_over_boundary(self):
        text = " ".join(["hello"] * 11)
        chunks = chunk(text, budget=10)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([c["text_tokens"] for c in chunks], [10, 1])
        self.assert_valid(text, chunks, budget=10)

    # The real 204-token limit works as expected.
    def test_real_budget_boundary(self):
        at = " ".join(["hello"] * tc.TEXT_BUDGET)
        over = " ".join(["hello"] * (tc.TEXT_BUDGET + 1))
        self.assertEqual(len(chunk(at)), 1)
        self.assertEqual(len(chunk(over)), 2)

    # A very long word is split and rejoins exactly.
    def test_very_long_word(self):
        word = "-".join(f"ab{i}" for i in range(300))
        self.assertGreater(tc.count_tokens(TOKENIZER, word), tc.TEXT_BUDGET)
        text = f"before {word} after"
        chunks = chunk(text)
        self.assert_valid(text, chunks)
        parts = [c for c in chunks if c["split_word_part"]]
        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(text[c["char_start"]:c["char_end"]] for c in parts), word)
        self.assertEqual(parts[0]["split_word_part"], (1, len(parts)))

    # Several long words in one text are handled.
    def test_multiple_long_words(self):
        w1 = "/".join(f"x{i}" for i in range(400))
        w2 = ".".join(f"y{i}" for i in range(350))
        text = f"start {w1} middle words here {w2} end"
        chunks = chunk(text)
        self.assert_valid(text, chunks)
        self.assertEqual(rebuild(text, chunks), text)

    # A word over 100 characters is handled.
    def test_word_over_100_characters(self):
        word = "a" * 500
        text = f"x {word} y"
        chunks = chunk(text)
        self.assert_valid(text, chunks)
        self.assertEqual(len(chunks), 1)

    # Text full of punctuation is handled.
    def test_punctuation_heavy(self):
        text = " ".join(["!!??..,,;;::--((]]"] * 60)
        chunks = chunk(text)
        self.assert_valid(text, chunks)
        self.assertGreater(len(chunks), 1)

    # Part numbers and codes are handled.
    def test_technical_identifiers(self):
        text = " ".join(
            f"PN-{i:05d}/REV.B_{i * 7}x{i % 13} DOFR-{i}A ELPI+{i}" for i in range(200)
        )
        chunks = chunk(text)
        self.assert_valid(text, chunks)

    # Finnish and other non-English letters are handled.
    def test_unicode(self):
        text = ("Mittauslaite päästöt höyry äöå ÄÖÅ "
                "粒子测量 размер частиц 🙂 naïve café ") * 40
        chunks = chunk(text)
        self.assert_valid(text, chunks)

    # Invisible and control characters are handled.
    def test_zero_width_and_control_characters(self):
        text = "alpha ​ beta \x07 gamma ﻿ delta"
        chunks = chunk(text)
        self.assert_valid(text, chunks)

    # Characters that Python and the tokenizer treat differently are handled.
    def test_characters_python_and_tokenizer_disagree_on(self):
        for odd in ["\x1c", "\x1d", "\x1e", "\x1f", "\x85"]:
            text = f"alpha{odd}beta gamma{odd}{odd}delta " * 80
            chunks = chunk(text)
            self.assert_valid(text, chunks)
            chunks_small = chunk(text, budget=5)
            self.assert_valid(text, chunks_small, budget=5)

    # Long repeated text is handled.
    def test_repeated_text(self):
        text = ("same sentence again. " * 500).strip()
        chunks = chunk(text)
        self.assert_valid(text, chunks)

    # Tabs, new lines and extra spaces are handled.
    def test_whitespace_kinds(self):
        text = "  one\ttwo\n\nthree\r\nfour   five  "
        chunks = chunk(text)
        self.assert_valid(text, chunks)
        self.assertEqual(" ".join(texts(text, chunks)).split(), text.split())

    # A large text is covered completely.
    def test_complete_coverage_large_text(self):
        words = [f"word{i}" if i % 37 else "-".join(["zz"] * 150) for i in range(5000)]
        text = " ".join(words)
        chunks = chunk(text)
        self.assert_valid(text, chunks)
        self.assertEqual(rebuild(text, chunks), text)
        self.assertEqual(rebuild(text, chunks).split(), text.split())

    # The coverage check notices a missing chunk.
    def test_coverage_check_catches_a_gap(self):
        text = "one two three four five"
        chunks = chunk(text, budget=2)
        broken = chunks[:1] + chunks[2:]
        self.assertFalse(tc.verify_section_coverage(text, broken))


    # A long, odd header is shortened but keeps the location.
    def test_header_with_unusual_tokenization(self):
        document = {
            "source_type": "pdf",
            "title": "-".join(f"ID{i}" for i in range(200)) + ".pdf",
        }
        section = {"heading": "§§ ¶¶ " * 30, "location": "page 12"}
        header, n, cut, left_out = tc.build_embed_header(TOKENIZER, document, section)
        self.assertLessEqual(n, tc.HEADER_BUDGET)
        self.assertEqual(n, tc.count_tokens(TOKENIZER, header))
        self.assertIn("Location: page 12", header)
        self.assertIn("Document", cut)

    # OCR chunks name the original document, not the image.
    def test_ocr_header_uses_parent_document(self):
        document = {
            "source_type": "ocr_image",
            "title": "OCR: image_0001.jpg",
            "source_file": "image_0001.jpg",
            "parent_source_file": "Parent Manual.pdf",
        }
        header, *_ = tc.build_embed_header(TOKENIZER, document, {"location": "page 3"})
        self.assertIn("Document: Parent Manual.pdf", header)
        self.assertNotIn("image_0001", header)

    # E-mail chunks show sender and date in the header.
    def test_email_header(self):
        document = {"source_type": "email", "title": "Quote request",
                    "author": "a.person@example.com", "date": "2024-05-01"}
        header, *_ = tc.build_embed_header(TOKENIZER, document, {"location": "body"})
        self.assertIn("From:", header)
        self.assertIn("Date: 2024-05-01", header)


    # Counting in batches gives the same numbers as one by one.
    def test_batched_equals_individual(self):
        samples = ["short", "a much longer sentence with more words in it",
                   "-".join(["x"] * 300), "äöå 粒子", ""]
        batched = TOKENIZER(samples, add_special_tokens=False, padding=False,
                            truncation=False)["input_ids"]
        single = [tc.count_tokens(TOKENIZER, s) for s in samples]
        self.assertEqual([len(b) for b in batched], single)
        padded = TOKENIZER(samples, add_special_tokens=False, padding=True)["input_ids"]
        self.assertNotEqual([len(p) for p in padded], single)


# Runs the whole chunker on a made-up project.
class WholePipelineTest(unittest.TestCase):
    # Create the made-up project and run the chunker once.
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        normalized = root / "data" / "processed" / "normalized"
        normalized.mkdir(parents=True)

        long_word = "-".join(f"q{i}" for i in range(400))
        long_text = " ".join(f"w{i}" for i in range(1500))
        cls.expected_words = 0

        for n, name in enumerate(tc.NORMALIZED_FILES):
            kind = name.replace("_normalized.json", "")
            docs = []
            for d in range(3):
                doc = {
                    "doc_id": f"{kind}-{d}",
                    "source_type": "ocr_image" if kind == "ocr" else kind,
                    "title": f"{kind} doc {d}",
                    "source_file": f"{kind}_{d}.bin",
                    "parent_source_file": "Parent.pdf" if kind == "ocr" else None,
                    "sections": [
                        {"section_id": f"{kind}-{d}-a", "text": "Copyright 2024",
                         "location": "page 1"},
                        {"section_id": f"{kind}-{d}-b", "text": "identical text here",
                         "location": "page 2"},
                        {"section_id": f"{kind}-{d}-c", "text": long_text,
                         "location": f"sheet {d}"},
                        {"section_id": f"{kind}-{d}-d", "text": f"x {long_word} y"},
                        {"section_id": f"{kind}-{d}-e", "text": "   "},
                    ],
                }
                for s in doc["sections"]:
                    cls.expected_words += len(s["text"].split())
                docs.append(doc)
            data = docs if n % 2 else {"documents": docs}
            (normalized / name).write_text(json.dumps(data), encoding="utf-8")

        cls.root = root
        cls.result = cls.run_script(root)
        cls.chunked = root / "data" / "processed" / "chunked"

    # Remove the made-up project afterwards.
    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # Run the chunker script.
    @staticmethod
    def run_script(root):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(root)],
            capture_output=True, text=True,
        )

    # Read the chunks the run produced.
    def chunks(self):
        with open(self.chunked / "chunks.jsonl", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    # The run finishes and all checks pass.
    def test_run_succeeded(self):
        self.assertEqual(self.result.returncode, 0, self.result.stdout + self.result.stderr)
        self.assertIn("All checks passed", self.result.stdout)

    # All output files are created and no temporary files are left.
    def test_all_outputs_exist(self):
        for name in tc.OutputFiles.NAMES:
            self.assertTrue((self.chunked / name).exists(), name)
        leftovers = [p for p in self.chunked.iterdir() if ".building_" in p.name]
        self.assertEqual(leftovers, [])

    # Document, section and word counts are right.
    def test_counts_and_word_coverage(self):
        meta = json.loads((self.chunked / "chunking_meta.json").read_text())
        self.assertEqual(meta["stats"]["documents"], 21)
        self.assertEqual(meta["stats"]["sections"], 105)
        self.assertEqual(meta["stats"]["source_words"], self.expected_words)
        self.assertEqual(meta["stats"]["empty_sections"], 21)
        self.assertEqual(meta["total_chunks"], len(self.chunks()))

    # No chunk is too long for the model.
    def test_every_chunk_within_window(self):
        for c in self.chunks():
            self.assertLessEqual(c["text_tokens"], tc.TEXT_BUDGET)
            self.assertLessEqual(c["header_tokens"], tc.HEADER_BUDGET)
            self.assertLessEqual(c["input_tokens"], tc.MODEL_WINDOW)

    # Each chunk keeps its source and location.
    def test_metadata_preserved(self):
        for c in self.chunks():
            for field in ("chunk_id", "doc_id", "section_id", "source_file",
                          "source_type", "embed_header", "text", "section_metadata"):
                self.assertIn(field, c)
        located = [c for c in self.chunks() if c["section_id"].endswith("-c")]
        self.assertTrue(all(c["location"].startswith("sheet") for c in located))
        self.assertTrue(all("Location: sheet" in c["embed_header"] for c in located))

    # Duplicates and boilerplate are marked, not deleted.
    def test_duplicates_and_boilerplate_marked_not_removed(self):
        same = [c for c in self.chunks() if c["text"] == "identical text here"]
        self.assertEqual(len(same), 21)
        self.assertEqual(sum(not c["is_duplicate"] for c in same), 1)
        self.assertTrue(all(c["duplicate_of"] for c in same if c["is_duplicate"]))
        boiler = [c for c in self.chunks() if c["text"] == "Copyright 2024"]
        self.assertEqual(len(boiler), 21)
        self.assertTrue(all(c["is_boilerplate"] for c in boiler))

    # Empty sections are kept.
    def test_empty_sections_kept(self):
        with open(self.chunked / "empty_sections.jsonl", encoding="utf-8") as f:
            self.assertEqual(len(f.readlines()), 21)

    # OCR chunks use the original document's name.
    def test_ocr_uses_parent_name(self):
        ocr = [c for c in self.chunks() if c["source_type"] == "ocr_image"]
        self.assertTrue(ocr)
        self.assertTrue(all("Document: Parent.pdf" in c["embed_header"] for c in ocr))

    # A split long word rejoins exactly.
    def test_long_word_rebuilt_exactly(self):
        long_word = "-".join(f"q{i}" for i in range(400))
        parts = [c for c in self.chunks()
                 if c["section_id"] == "pdf-0-d" and c["split_word_part"]]
        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(c["text"] for c in parts), long_word)

    # A second run refuses to overwrite the results.
    def test_second_run_refuses_to_overwrite(self):
        before = (self.chunked / "chunks.jsonl").stat().st_mtime
        again = self.run_script(self.root)
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("will not be overwritten", again.stdout)
        self.assertEqual((self.chunked / "chunks.jsonl").stat().st_mtime, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
