"""Measures real OCR accuracy (CER/WER) per page of a PDF, using the
PDF's own native text layer as ground truth (ocr/native_pdf_text.py) —
not a synthetic or hand-typed reference.

This only gives a meaningful number for a PDF confirmed to have a
clean, uncorrupted text layer: docs/phases/phase-2-ocr-pipeline.md
documents a real case where one of this project's own test PDFs has a
broken font encoding for certain letter sequences (its native text is
WRONG, not a valid ground truth) — that document must never be passed
to this script. Run scripts/check_pdf_text_layer_is_clean.py first if
unsure.

Usage:
    python scripts/measure_ocr_accuracy.py <pdf-path> [<pdf-path> ...]
    python scripts/measure_ocr_accuracy.py --tesseract-cmd "C:\\Program Files\\Tesseract-OCR\\tesseract.exe" doc.pdf
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pymupdf

from smart_text_extractor.ocr.engine import OcrEngine
from smart_text_extractor.ocr.locate import find_tessdata_dir, find_tesseract_cmd
from smart_text_extractor.ocr.native_pdf_text import extract_native_text_result

DEFAULT_RENDER_DPI = 300


def _levenshtein(reference: list, hypothesis: list) -> int:
    """Classic O(len(reference) * len(hypothesis)) edit distance over any
    sequence (characters or words) — a single implementation used for
    both CER and WER below rather than two near-duplicate ones."""
    if not reference:
        return len(hypothesis)
    previous_row = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current_row = [i] + [0] * len(hypothesis)
        for j, hyp_item in enumerate(hypothesis, start=1):
            current_row[j] = min(
                previous_row[j] + 1,  # deletion
                current_row[j - 1] + 1,  # insertion
                previous_row[j - 1] + (ref_item != hyp_item),  # substitution (0 if equal)
            )
        previous_row = current_row
    return previous_row[-1]


def _bag_of_words_error_rate(reference_words: list[str], hypothesis_words: list[str]) -> float:
    """Word error rate that ignores ORDER entirely — ordinary sequence-
    based WER conflates two different things: whether a word was
    recognized correctly at all, and whether it landed in the right
    position. A real table whose reading order gets scrambled (a known,
    separate, already-documented issue — docs/phases/phase-2-ocr-pipeline.md)
    makes sequence-based WER look catastrophic even when every individual
    word was read correctly, just reordered. This uses multiset overlap
    (Counter intersection) instead of edit distance, so moving a whole
    block of correctly-read words costs nothing here — only genuinely
    missing/extra/misrecognized words do.
    """
    from collections import Counter

    reference_counts = Counter(reference_words)
    hypothesis_counts = Counter(hypothesis_words)
    matched = sum((reference_counts & hypothesis_counts).values())
    errors = len(reference_words) - matched
    return errors / len(reference_words) if reference_words else 0.0


@dataclass
class PageAccuracy:
    page_number: int
    reference_char_count: int
    reference_word_count: int
    cer: float
    wer: float
    bag_of_words_wer: float


def _measure_page(reference_text: str, hypothesis_text: str, page_number: int) -> PageAccuracy:
    reference_chars = list(reference_text)
    hypothesis_chars = list(hypothesis_text)
    reference_words = reference_text.split()
    hypothesis_words = hypothesis_text.split()

    char_distance = _levenshtein(reference_chars, hypothesis_chars)
    word_distance = _levenshtein(reference_words, hypothesis_words)

    cer = char_distance / len(reference_chars) if reference_chars else 0.0
    wer = word_distance / len(reference_words) if reference_words else 0.0
    return PageAccuracy(
        page_number=page_number,
        reference_char_count=len(reference_chars),
        reference_word_count=len(reference_words),
        cer=cer,
        wer=wer,
        bag_of_words_wer=_bag_of_words_error_rate(reference_words, hypothesis_words),
    )


def measure_document(pdf_path: Path, engine: OcrEngine, dpi: int = DEFAULT_RENDER_DPI) -> list[PageAccuracy]:
    results: list[PageAccuracy] = []
    zoom = dpi / 72
    matrix = pymupdf.Matrix(zoom, zoom)
    with pymupdf.open(str(pdf_path)) as document:
        for page_index in range(len(document)):
            page = document.load_page(page_index)
            native_result = extract_native_text_result(page, render_dpi=dpi)
            if native_result is None:
                continue  # no usable ground truth for this page — skip, don't guess
            reference_text = native_result.raw_text

            pixmap = page.get_pixmap(matrix=matrix)
            image_bytes = pixmap.tobytes("png")
            import io

            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))
            ocr_result = engine.run(image)

            results.append(_measure_page(reference_text, ocr_result.raw_text, page_index + 1))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--tesseract-cmd", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=DEFAULT_RENDER_DPI)
    args = parser.parse_args()

    engine = OcrEngine(
        tesseract_cmd=args.tesseract_cmd or find_tesseract_cmd(),
        tessdata_dir=find_tessdata_dir(),
    )

    for pdf_path in args.pdfs:
        print(f"=== {pdf_path.name} ===")
        page_results = measure_document(pdf_path, engine, dpi=args.dpi)
        if not page_results:
            print("  no pages with a usable native text layer — nothing measured")
            continue

        total_char_errors = sum(round(p.cer * p.reference_char_count) for p in page_results)
        total_chars = sum(p.reference_char_count for p in page_results)
        total_word_errors = sum(round(p.wer * p.reference_word_count) for p in page_results)
        total_bow_errors = sum(round(p.bag_of_words_wer * p.reference_word_count) for p in page_results)
        total_words = sum(p.reference_word_count for p in page_results)

        for p in page_results:
            print(
                f"  page {p.page_number:3d}: CER={p.cer*100:5.1f}%  WER={p.wer*100:5.1f}%  "
                f"word-recognition-only={100-p.bag_of_words_wer*100:5.1f}%  (ref: {p.reference_char_count} chars, {p.reference_word_count} words)"
            )

        overall_cer = total_char_errors / total_chars if total_chars else 0.0
        overall_wer = total_word_errors / total_words if total_words else 0.0
        overall_bow_wer = total_bow_errors / total_words if total_words else 0.0
        print(
            f"  --- overall: CER={overall_cer*100:.1f}% (char accuracy={100-overall_cer*100:.1f}%)  "
            f"sequence-WER={overall_wer*100:.1f}% (order+recognition combined)  "
            f"word-recognition-only={100-overall_bow_wer*100:.1f}% (ignores order — see script docstring) ---"
        )
        print()


if __name__ == "__main__":
    main()
