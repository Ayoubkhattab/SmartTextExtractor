"""Native PDF text-layer extraction (§7.1 extension): a PDF exported
from an authoring tool (Word, InDesign, a web page's "print to PDF", ...)
already contains real, embedded text — reading it directly is instant
and perfectly accurate, unlike rendering the page to an image and OCR'ing
it. A genuinely scanned PDF (a photographed/scanned paper document with
no embedded text) still needs the OCR pipeline; this module's job is
telling the two apart, and building the finished result directly when it
can.

Session-defining real finding (docs/phases/phase-2-ocr-pipeline.md): all
3 real test documents used throughout this project's OCR debugging
turned out to already have a full native text layer. Every character-
level OCR error chased this session (tessdata tuning, dual-pass merge,
known-misread correction, ...) was never present in the source text at
all — it was introduced by rendering that text to pixels and reading it
back with Tesseract. Reusing ocr/reorder.py's Line/reading-order/table/
heading machinery on the PDF's own words (instead of Tesseract OCR
words) gets all of that structure-aware handling for free, since it
already operates on generic (text, position, confidence) data, not
anything Tesseract-specific — verified end to end on the real,
previously-hardest table page: correct heading detection, correct
row-major table order, and (unlike OCR) zero character-level errors,
since there is no recognition step at all to make one.

KNOWN LIMITATION: a page that mixes native text with an embedded scanned
graphic (e.g. a digitally-typed cover memo with a scanned signature or
stamp image containing its own text) will silently lose whatever text is
inside that graphic — get_text("words") only ever sees the real text
runs, never pixels. Not something any of this project's real test
documents exercise, so left undetected rather than guessed at.
"""
from __future__ import annotations

import pymupdf

from smart_text_extractor.core.models import BoundingBox, OcrResult, Rect
from smart_text_extractor.ocr.reorder import (
    assemble_markdown,
    assemble_text_segments,
    classify_document_units,
    group_into_lines,
    order_lines_reading_order,
)

MIN_WORDS_FOR_NATIVE_TEXT = 10
"""Below this word count, treat the page as scanned (no usable text
layer) rather than native. Guards against a false positive on a
genuinely scanned page that happens to carry a few words of embedded
metadata/watermark text — real pages checked this session had 95-364
words, so this is a conservative, clearly-differentiating floor, not a
value tuned against edge cases we don't have real examples of."""

def extract_native_text_result(page: pymupdf.Page, render_dpi: int) -> OcrResult | None:
    """Returns a fully-formed OcrResult built directly from the PDF
    page's own embedded text, or None if the page doesn't have a usable
    text layer — callers fall back to the OCR pipeline in that case.

    render_dpi must match the DPI the caller renders this same page's
    preview image at (pdf_import.py's import_pdf_pages passes its own
    dpi through here) — PDF word coordinates are in points (1/72in), and
    scaling them by render_dpi/72 instead of an arbitrary factor puts
    them in the same numeric space as that rendered image's pixels.
    Every absolute-pixel threshold in ocr/reorder.py (cell-boundary
    gaps, column-gap floors, ...) was calibrated against that space, so
    this is what makes those thresholds behave consistently whether a
    word came from Tesseract or straight from the PDF — and it's also
    what keeps word_boxes aligned to the actual preview image pixels,
    should a future feature (e.g. the planned searchable-PDF export)
    need to highlight a word on it.
    """
    words = page.get_text("words")  # (x0, y0, x1, y1, text, block_no, line_no, word_no)
    if len(words) < MIN_WORDS_FOR_NATIVE_TEXT:
        return None

    points_to_pixels = render_dpi / 72
    tagged = [
        (
            BoundingBox(
                text=text,
                rect=Rect(
                    x=round(x0 * points_to_pixels),
                    y=round(y0 * points_to_pixels),
                    width=round((x1 - x0) * points_to_pixels),
                    height=round((y1 - y0) * points_to_pixels),
                ),
                confidence=100.0,  # not a recognition guess — this is the document's own real text
            ),
            block_no,
            0,
            line_no,
        )
        for x0, y0, x1, y1, text, block_no, line_no, word_no in words
    ]

    lines = group_into_lines(tagged)
    ordered_lines = order_lines_reading_order(lines)
    segments = assemble_text_segments(ordered_lines)
    raw_text = "".join(segment.text for segment in segments)

    return OcrResult(
        raw_text=raw_text,
        word_boxes=[box for box, *_ in tagged],
        segments=segments,
        markdown=assemble_markdown(ordered_lines),
        document_units=classify_document_units(ordered_lines),
        confidence_score=100.0,
    )
