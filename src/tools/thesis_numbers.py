#!/usr/bin/env python3
# Collects the measured numbers and writes them into the thesis automatically.


import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT = PROJECT_ROOT / "thesis" / "generated" / "numbers.tex"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_stats import count_normalized_file, KNOWN_FILES

SOURCE_LABELS = {
    "pdf": "Pdf", "word": "Word", "excel": "Excel", "ppt": "Ppt",
    "email": "Email", "text": "Text", "ocr": "Ocr",
}

VERIFIED_HISTORY = {
    "NumOldChunks": 202_064,
    "NumOldChunkWords": 23_343_579,
    "NumOverlapWords": 1_767_650,
}

MEASURED_2026_10_08 = {
    "NumRawFiles": 437,
    "NumRawGB": "4.6",   # 4.57 GB (10^9 bytes), rechecked 2026-10-30; 4.26 was GiB
    "NumTextMeasurementChunks": 1_289_010,
    "NumTextHeaderChunks": 834_269,
    "NumTextLabelledChunks": 52_407,
    "NumNumericPctMeasurement": "97",
    "NumNumericPctHeader": "85",
    "NumWordsPerChunkMeasurement": "0.9",
    "NumWordsPerChunkHeader": "3.9",
    "NumOverWindowRecheck": 0,
    "NumWordsLost": 0,
    "NumSplitRejoinFail": 0,
    "NumDuplicateOriginals": 100_131,
    "NumDupWithoutAnyVector": 0,
    "NumRawPst": 3, "NumRawDocx": 195, "NumRawPdf": 166, "NumRawJpg": 65,
    "NumRawXlsx": 2, "NumRawPptx": 1, "NumRawOther": 5,
    "NumAttachmentDocs": 10 + 39 + 91 + 524 + 336,
    "NumUnreachable": 0, "NumHeaderOnlyChunks": 201,
    "NumFindKeywordPct": "92.7", "NumFindSemanticPct": "21.3",
    "NumFindEmailPct": "98", "NumFindWordPct": "98", "NumFindPptPct": "98",
    "NumFindPdfPct": "92", "NumFindOcrPct": "85",
    "NumFindExcelPct": "12", "NumFindTextPct": "8",
    "NumFinnishChunks": 2_779, "NumMixedChunks": 834, "NumEnglishChunks": 112_646,
    # answer evaluation, run_20261011_200557, judged by the author on 2026-10-12
    "NumEvalQuestions": "8", "NumEvalInCorpus": "7", "NumEvalFullyCorrect": 6, "NumEvalPartlyCorrect": 2, "NumEvalCorrectMean": '0.88', "NumEvalGrounded": 8, "NumEvalCitationPass": 6, "NumEvalRefusals": 1, "NumEvalMedianAnswerSec": 93, "NumEvalMinAnswerSec": 68, "NumEvalMaxAnswerSec": 301, "NumEvalMedianSearchSec": '0.20', "NumEvalSourcesShown": 25, "NumEvalSourcesRelated": 25, "NumEvalPooledPassages": 165, "NumLittleTextChunks": 181_351,
    # Blind comparison of Llama 3.2 3B and Qwen2.5 3B (author ratings, 18 Oct 2026, after the context fix)
    "NumCmpQuestions": 8, "NumCmpUseCases": 12, "NumCmpLlamaCorrect": 6, "NumCmpLlamaPartly": 2, "NumCmpLlamaWrong": 0, "NumCmpLlamaMean": '0.88', "NumCmpLlamaGrounded": 8, "NumCmpLlamaUseful": '1.88', "NumCmpLlamaBetter": 1, "NumCmpLlamaSec": 93, "NumCmpQwenCorrect": 6, "NumCmpQwenPartly": 2, "NumCmpQwenWrong": 0, "NumCmpQwenMean": '0.88', "NumCmpQwenGrounded": 8, "NumCmpQwenUseful": '1.75', "NumCmpQwenBetter": 4, "NumCmpQwenSec": 74, "NumCmpSame": 3, "NumUcLlamaCorrect": 3, "NumUcLlamaPartly": 8, "NumUcLlamaWrong": 1, "NumUcLlamaRefusal": 0, "NumUcLlamaMean": '0.58', "NumUcLlamaGrounded": 12, "NumUcLlamaAdded": 0, "NumUcLlamaUseful": '1.17', "NumUcLlamaBetter": 8, "NumUcLlamaSec": 68, "NumUcQwenRated": 12, "NumUcQwenCorrect": 2, "NumUcQwenPartly": 2, "NumUcQwenWrong": 7, "NumUcQwenRefusal": 0, "NumUcQwenMean": '0.25', "NumUcQwenGrounded": 7, "NumUcQwenAdded": 0, "NumUcQwenUseful": '0.42', "NumUcQwenBetter": 0, "NumUcQwenSec": 34, "NumUcSame": 4, "NumUcQwenMaxSec": 91, "NumCmpBetterLlamaTotal": 9, "NumCmpBetterQwenTotal": 4, "NumCmpSameTotal": 7, "NumUcQwenMissed": 1,
    # Blind rating by Dekati staff (21 Oct 2026): Q1-Q8 normal mode, U1-U12 advisor mode; agreement with the author on Q1-Q8
    "NumDekLlamaCorrect": 3, "NumDekLlamaPartly": 4, "NumDekLlamaWrong": 0, "NumDekLlamaRefusal": 1, "NumDekLlamaGrounded": 6, "NumDekLlamaGroundedOf": 7, "NumDekLlamaUseful": '1.25', "NumDekLlamaBetter": 4, "NumDekQwenCorrect": 1, "NumDekQwenPartly": 5, "NumDekQwenWrong": 1, "NumDekQwenRefusal": 1, "NumDekQwenGrounded": 5, "NumDekQwenGroundedOf": 7, "NumDekQwenUseful": '0.88', "NumDekQwenBetter": 1, "NumDekSame": 3, "NumDekAdvLlamaCorrect": 0, "NumDekAdvLlamaPartly": 3, "NumDekAdvLlamaWrong": 8, "NumDekAdvLlamaMissed": 1, "NumDekAdvLlamaGrounded": 10, "NumDekAdvLlamaGroundedOf": 11, "NumDekAdvLlamaUseful": '0.00', "NumDekAdvLlamaBetter": 2, "NumDekAdvLlamaSec": 251, "NumDekAdvQwenCorrect": 0, "NumDekAdvQwenPartly": 3, "NumDekAdvQwenWrong": 8, "NumDekAdvQwenRefusal": 1, "NumDekAdvQwenGrounded": 11, "NumDekAdvQwenGroundedOf": 11, "NumDekAdvQwenUseful": '0.08', "NumDekAdvQwenBetter": 4, "NumDekAdvQwenSec": 213, "NumDekAdvSame": 6, "NumDekAdded": 4, "NumDekAgreeCorrect": 7, "NumDekAgreeOf": 16, "NumDekStricter": 7, "NumDekAgreePref": 4, "NumDekAgreePrefOf": 8,
    # Context-assembly fix (18 Oct 2026): sources whose best chunk was not in the context
    "NumDropSources": 22, "NumDropTotalSources": 66, "NumDropPct": 33, "NumDropQuestions": 11, "NumDropAllQuestions": 20, "NumDropAdvSources": 37, "NumDropAdvTotal": 160, "NumDropAfter": 0,
    "NumMultilingualVectors": 3_613, "NumFinnishAllPct": "1", "NumFinnishWordPct": "68",
    "NumMultilingualMaxTokens": 472,
    "NumMistralGB": "5.7", "NumLlamaGB": "3.1",
    "NumCompareQuestions": 7, "NumSameTopDoc": 7, "NumTocOld": 1, "NumTocNew": 0,
    "NumWordsOldMin": 245, "NumWordsOldMax": "1,665",
    "NumWordsNewMin": 767, "NumWordsNewMax": "1,877",
    "NumManuStepsOld": 0, "NumManuStepsNew": 4, "NumManuStepsTotal": 6,
    "NumIndexBuildSec": 307, "NumMultiBuildMin": 18, "NumMultiBuildSec": 1_079,
    "NumLoadSearchSec": 40, "NumLoadModelSec": 12, "NumSearchSec": "2.4",
    "NumManuPromptTokens": "4,736", "NumManuReadSec": 154, "NumManuWriteSec": 88,
    "NumManuTotalSec": 244, "NumGenTokensPerSec": "9",
    "NumSynthQuestions": 4, "NumSynthCorrect": 3, "NumSynthRefusals": 1,
}

TEST_FILE = PROJECT_ROOT / "tests" / "test_token_chunking.py"


# Format a number for the thesis.
def fmt(value):
    if value is None:
        return r"\UNKNOWN"
    if isinstance(value, float):
        return f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# Load a results file if it exists.
def load(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Collect all measured numbers and write them for the thesis.
def main():
    numbers = {}

    folder = PROCESSED / "normalized"
    totals = {"documents": 0, "sections": 0, "words": 0}
    for filename in KNOWN_FILES["normalized"]:
        label = SOURCE_LABELS[filename.replace("_normalized.json", "")]
        path = folder / filename
        if not path.exists():
            for key in ("Docs", "Secs", "Words"):
                numbers[f"Num{key}{label}"] = None
            continue
        c = count_normalized_file(path)
        numbers[f"NumDocs{label}"] = c["documents"]
        numbers[f"NumSecs{label}"] = c["sections"]
        numbers[f"NumWords{label}"] = c["words"]
        totals["documents"] += c["documents"]
        totals["sections"] += c["sections"]
        totals["words"] += c["words"]
    numbers["NumDocuments"] = totals["documents"]
    numbers["NumSections"] = totals["sections"]
    numbers["NumSourceWords"] = totals["words"]

    numbers.update(VERIFIED_HISTORY)
    numbers.update(MEASURED_2026_10_08)
    # First search of each question in the final evaluation run (20261018_160322),
    # before the same search is repeated with the data in memory (NumEvalMedianSearchSec).
    numbers.update({"NumEvalFirstSearchSec": "1.11", "NumEvalFirstSearchMinSec": "0.23",
                    "NumEvalFirstSearchMaxSec": "2.57"})
    # Ten new held-out factual questions, rated blind by the author
    # (comparison_scores_20261029_163423.json; runs 20261029_140720 and 20261029_141912).
    numbers.update({"NumNewQuestions": 10, "NumNewAnswers": 20,
                    "NumNewLlamaCorrect": 9, "NumNewLlamaWrong": 1,
                    "NumNewQwenCorrect": 9, "NumNewQwenWrong": 1,
                    "NumNewGrounded": 20, "NumNewLlamaSec": 67, "NumNewQwenSec": 48})
    # RQ2 retrieval comparison: passages ranked in the top 3 by any method, judged by the author
    # for the 7 test questions whose answer is in the corpus (run 20261018_160322).
    numbers.update({"NumRetQuestions": 7, "NumRetDepth": 3,
                    "NumRetKwHitOne": 4, "NumRetSemHitOne": 4, "NumRetHybHitOne": 3,
                    "NumRetKwHitThree": 7, "NumRetSemHitThree": 6, "NumRetHybHitThree": 7,
                    "NumRetKwMrr": "0.76", "NumRetSemMrr": "0.71", "NumRetHybMrr": "0.69"})

    meta = load(PROCESSED / "chunked" / "chunking_meta.json")
    ok = meta is not None and all(v == "PASS" for v in meta.get("checks", {}).values())
    stats = meta.get("stats", {}) if ok else {}
    per_source = meta.get("per_source", {}) if ok else {}
    text_tokens = meta.get("text_tokens", {}) if ok else {}
    input_tokens = meta.get("input_tokens", {}) if ok else {}

    numbers["NumChunks"] = meta.get("total_chunks") if ok else None
    for key, label in SOURCE_LABELS.items():
        numbers[f"NumChunks{label}"] = per_source.get(key, {}).get("chunks")
    numbers["NumTextTokensMedian"] = text_tokens.get("median")
    numbers["NumTextTokensMax"] = text_tokens.get("max")
    numbers["NumInputTokensMax"] = input_tokens.get("max")
    numbers["NumSplitWords"] = stats.get("split_long_words")
    numbers["NumDuplicates"] = stats.get("duplicates_marked")
    numbers["NumBoilerplate"] = stats.get("boilerplate_marked")
    numbers["NumSplitPieces"] = stats.get("split_word_pieces")
    numbers["NumTextTokensMean"] = text_tokens.get("mean")
    numbers["NumInputTokensMedian"] = input_tokens.get("median")
    numbers["NumHeadersShortened"] = stats.get("headers_with_cut_field")
    total_chunks = meta.get("total_chunks") if ok else None
    numeric = MEASURED_2026_10_08["NumTextMeasurementChunks"] + MEASURED_2026_10_08["NumTextHeaderChunks"]
    numbers["NumNumericChunks"] = numeric
    numbers["NumNumericPct"] = f"{numeric * 100 / total_chunks:.0f}" if total_chunks else None
    elapsed = meta.get("elapsed_seconds") if ok else None
    numbers["NumChunkingMinutes"] = round(elapsed / 60, 1) if elapsed else None
    numbers["NumChunkingSec"] = round(elapsed) if elapsed else None

    vmeta = load(PROCESSED / "vectors" / "vectorization_meta.json")
    numbers["NumVectors"] = vmeta.get("total_vectors") if vmeta else None
    numbers["NumInputsOverWindow"] = vmeta.get("inputs_over_window") if vmeta else None
    not_emb = vmeta.get("not_embedded", {}) if vmeta else {}
    numbers["NumDupNotEmbedded"] = next((n for r, n in not_emb.items() if "duplicate" in r), None)
    # Chunks that are candidates for a vector: all chunks except the keyword-only numeric tier.
    candidates = total_chunks - numeric if total_chunks else None
    numbers["NumSemanticCandidates"] = candidates
    vector_check = (candidates is not None and numbers["NumDupNotEmbedded"] is not None
                    and candidates - numbers["NumDupNotEmbedded"] == numbers["NumVectors"])
    print(f"  candidates - duplicates = vectors: {'PASS' if vector_check else 'FAIL'}")
    secs = vmeta.get("elapsed_seconds_last_session") if vmeta else None
    numbers["NumEmbeddingHours"] = round(secs / 3600, 1) if secs else None
    numbers["NumEmbeddingSec"] = round(secs) if secs else None

    imeta = load(PROCESSED / "index" / "index_meta.json")
    numbers["NumSelfRetrieval"] = imeta.get("self_retrieval") if imeta else None

    if TEST_FILE.exists():
        numbers["NumChunkerTests"] = TEST_FILE.read_text(encoding="utf-8").count("    def test_")
    else:
        numbers["NumChunkerTests"] = None

    lines = [
        "% AUTO-GENERATED by src/tools/thesis_numbers.py. Do not edit by hand.",
        "% Every value is measured; \\UNKNOWN means not measured yet.",
    ]
    for name in sorted(numbers):
        lines.append(f"\\newcommand{{\\{name}}}{{{fmt(numbers[name])}}}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_name(OUTPUT.name + ".tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temp, OUTPUT)

    known = sum(v is not None for v in numbers.values())
    print(f"wrote {OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"  {len(numbers)} numbers: {known} measured, {len(numbers) - known} still unknown")
    if not ok:
        print("  chunking results missing or not all checks PASS: chunk numbers left as unknown")


if __name__ == "__main__":
    main()
