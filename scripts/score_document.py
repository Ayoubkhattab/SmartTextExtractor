"""Scorecard: how close is a document's extraction to the real pages?

Reports, per page, the two things that matter and have always been
measured separately or not at all:

  text    — CER / WER / order-agnostic word recognition, against the PDF's
            own text layer, and only where that layer is trustworthy
            (ocr/native_pdf_text.MAX_CORRUPT_TOKEN_RATIO). A document that
            fails that gate has no reliable reference, so its text columns
            are reported as "-" rather than guessed at.
  visual  — how much the page LOOKS like its source, by rebuilding it from
            the extracted model alone and comparing ink maps
            (quality/visual_similarity.py).

The visual number is the one this project never had. It exists so a layout
change can be shown to help: twice already a change looked like an
improvement and measured worse.

Usage:
    python scripts/score_document.py docs/*.pdf
    python scripts/score_document.py --pages 3 "docs/دليل الاستخدام.pdf"
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pymupdf

from smart_text_extractor.core.models import Page, PdfPageSource
from smart_text_extractor.core.pdf_import import DEFAULT_RENDER_DPI, is_text_layer_trustworthy, render_pdf_to_images
from smart_text_extractor.export.page_render import render_result_to_image
from smart_text_extractor.ocr.engine import OcrEngine
from smart_text_extractor.ocr.locate import find_tessdata_dir, find_tesseract_cmd
from smart_text_extractor.ocr.native_pdf_text import extract_native_text_result
from smart_text_extractor.ocr.page_pipeline import run_page
from smart_text_extractor.quality.visual_similarity import compare_pages

from measure_ocr_accuracy import _measure_page


def score_document(pdf_path: Path, work_dir: Path, engine: OcrEngine, max_pages: int | None) -> None:
    images = render_pdf_to_images(pdf_path, work_dir / pdf_path.stem)
    trusted = is_text_layer_trustworthy(pdf_path)

    print(f"\n=== {pdf_path.name} ===")
    print(f"  text layer trusted: {trusted}")
    print(f"  {'page':>5}  {'CER':>7} {'WER':>7} {'word-rec':>9}   {'visual':>7} {'rows':>7} {'cols':>7} {'ink':>7}")

    totals = {"visual": 0.0, "vertical": 0.0, "horizontal": 0.0, "pages": 0}

    with pymupdf.open(str(pdf_path)) as document:
        count = min(max_pages or len(document), len(document))
        for index in range(count):
            page = Page(image_path=images[index], order_index=index)
            page.pdf_source = PdfPageSource(pdf_path, index, DEFAULT_RENDER_DPI, text_layer_trusted=trusted)
            result = run_page(page, engine)

            rebuilt = render_result_to_image(
                result, work_dir / pdf_path.stem / f"rebuilt_{index + 1:03d}.png", DEFAULT_RENDER_DPI
            )
            visual = compare_pages(images[index], rebuilt)

            text_columns = f"{'-':>7} {'-':>7} {'-':>9}"
            if trusted:
                reference = extract_native_text_result(document.load_page(index), render_dpi=DEFAULT_RENDER_DPI)
                if reference is not None:
                    accuracy = _measure_page(reference.raw_text, result.raw_text, index + 1)
                    text_columns = (
                        f"{accuracy.cer * 100:6.1f}% {accuracy.wer * 100:6.1f}% "
                        f"{100 - accuracy.bag_of_words_wer * 100:8.1f}%"
                    )

            print(
                f"  {index + 1:>5}  {text_columns}   "
                f"{visual.percent:6.1f}% {visual.vertical * 100:6.1f}% {visual.horizontal * 100:6.1f}% "
                f"{visual.ink_ratio:6.2f}x"
            )
            totals["visual"] += visual.overlap
            totals["vertical"] += visual.vertical
            totals["horizontal"] += visual.horizontal
            totals["pages"] += 1

    if totals["pages"]:
        pages = totals["pages"]
        print(
            f"  {'mean':>5}  {'':>25}   {totals['visual'] / pages * 100:6.1f}% "
            f"{totals['vertical'] / pages * 100:6.1f}% {totals['horizontal'] / pages * 100:6.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--pages", type=int, default=None, help="score only the first N pages of each document")
    parser.add_argument("--work-dir", type=Path, default=Path("build/scorecard"))
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    engine = OcrEngine(tesseract_cmd=find_tesseract_cmd(), tessdata_dir=find_tessdata_dir())

    for pdf_path in args.pdfs:
        score_document(pdf_path, args.work_dir, engine, args.pages)


if __name__ == "__main__":
    main()
