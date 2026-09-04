"""Decides how one page's text is produced, given what that page actually is.

A page rendered from a PDF has two independent sources of truth for the
same words: the PDF's own embedded text, and an OCR reading of the
rendered pixels. Measured on this project's real documents, each is
reliably better than the other in a *different*, mechanically-detectable
way (docs/phases/phase-2-ocr-pipeline.md):

  - The embedded text is exact for the overwhelming majority of words —
    there is no recognition step to make an error in — while OCR sits at
    roughly 33-44% word error on the same pages.
  - But the embedded text stores some words with their letters
    transposed (a font-mapping bug present in ALL THREE real test PDFs,
    0.4% / 3.4% / 8.9% of words), and OCR reads exactly those words
    correctly, because it reads the rendered glyphs, which are fine.
  - And text baked into an image (a screenshot, a stamp) exists only in
    the pixels, so only OCR sees it at all.

So neither source is used alone: the embedded text is the base, OCR
repairs its transpositions and contributes the words it alone can see.
A page with no usable text layer (a genuine scan) falls back to plain OCR,
which is also what every non-PDF page uses.
"""
from __future__ import annotations

import pymupdf

from smart_text_extractor.core.models import OcrResult, Page
from smart_text_extractor.ocr.native_pdf_text import extract_native_text_result


def run_page(page: Page, engine) -> OcrResult:
    """Produces the finished OcrResult for one page. `engine` is anything
    with OcrEngine's .run(image) -> OcrResult shape."""
    ocr_result = engine.run(str(page.image_path))

    if page.pdf_source is None or not page.pdf_source.text_layer_trusted:
        return ocr_result

    try:
        with pymupdf.open(str(page.pdf_source.pdf_path)) as document:
            pdf_page = document.load_page(page.pdf_source.page_index)
            native_result = extract_native_text_result(
                pdf_page,
                render_dpi=page.pdf_source.render_dpi,
                ocr_word_boxes=ocr_result.word_boxes,
            )
    except Exception:  # noqa: BLE001 - the OCR result is already in hand; a missing/damaged source file must not lose it
        return ocr_result

    # None means this page has no usable text layer (a scanned page inside
    # an otherwise-digital PDF) — OCR is all there is for it.
    return native_result if native_result is not None else ocr_result
