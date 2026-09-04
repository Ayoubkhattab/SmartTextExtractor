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
from smart_text_extractor.ocr.layout_boxes import group_units_into_boxes
from smart_text_extractor.ocr.native_pdf_style import PageStyleIndex
from smart_text_extractor.ocr.native_pdf_text import extract_native_text_result, page_layout_of


def run_page(page: Page, engine) -> OcrResult:
    """Produces the finished OcrResult for one page. `engine` is anything
    with OcrEngine's .run(image) -> OcrResult shape."""
    if page.pdf_source is None:
        return engine.run(str(page.image_path))

    source = page.pdf_source
    try:
        with pymupdf.open(str(source.pdf_path)) as document:
            pdf_page = document.load_page(source.page_index)

            # How the page LOOKS is read from the PDF even when its text is
            # NOT trusted. The damage the trust gate screens for lives in
            # the font's character mapping; the sizes, colours, drawn shapes
            # and page geometry beside it are untouched and correct. So a
            # page whose words have to come from OCR still gets its real
            # heading sizes, its coloured boxes and its paper — instead of
            # the flat, full-width, unstyled dump it produced before, which
            # is exactly what a text layer failing the gate used to mean.
            style_index = PageStyleIndex(pdf_page, source.render_dpi)
            layout = page_layout_of(pdf_page)

            ocr_result = engine.run(str(page.image_path), style_index=style_index)

            if source.text_layer_trusted:
                native_result = extract_native_text_result(
                    pdf_page,
                    render_dpi=source.render_dpi,
                    ocr_word_boxes=ocr_result.word_boxes,
                )
                # None means this page has no usable text layer (a scanned
                # page inside an otherwise-digital PDF) — OCR is all there
                # is for it.
                if native_result is not None:
                    return native_result

            ocr_result.page_layout = layout
            ocr_result.document_units = group_units_into_boxes(
                ocr_result.document_units, style_index.container_boxes
            )
            return ocr_result
    except Exception:  # noqa: BLE001 - a missing/damaged source must still yield the page's text
        return engine.run(str(page.image_path))
