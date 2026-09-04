"""Tests for native PDF text-layer extraction (ocr/native_pdf_text.py).

Uses real PDFs generated with PyMuPDF itself (same pattern as
tests/core/test_pdf_import.py) — no OCR/Tesseract involved here at all,
since the whole point of this module is skipping OCR when a page
already has its own embedded text.
"""
from __future__ import annotations

import pymupdf
import pytest

from smart_text_extractor.ocr.native_pdf_text import MIN_WORDS_FOR_NATIVE_TEXT, extract_native_text_result


def _page_with_words(word_count: int) -> pymupdf.Page:
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=400)
    text = " ".join(f"word{i}" for i in range(word_count))
    # insert_text lays out along one line with no wrapping — a long
    # synthetic string silently overflows and gets clipped past the page
    # edge (confirmed real: dropped the last word's final characters).
    # insert_textbox wraps within the given rect instead.
    page.insert_textbox(pymupdf.Rect(20, 20, 580, 380), text, fontsize=14)
    return page


def test_returns_none_for_a_page_with_too_few_words() -> None:
    page = _page_with_words(MIN_WORDS_FOR_NATIVE_TEXT - 1)
    assert extract_native_text_result(page, render_dpi=300) is None


def test_returns_none_for_a_blank_page() -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=400)
    assert extract_native_text_result(page, render_dpi=300) is None


def test_extracts_a_real_result_for_a_page_with_enough_words() -> None:
    page = _page_with_words(MIN_WORDS_FOR_NATIVE_TEXT + 5)

    result = extract_native_text_result(page, render_dpi=300)

    assert result is not None
    assert result.confidence_score == 100.0
    assert len(result.word_boxes) == MIN_WORDS_FOR_NATIVE_TEXT + 5
    assert all(box.confidence == 100.0 for box in result.word_boxes)
    assert "word0" in result.raw_text and "word3" in result.raw_text
    assert result.segments and result.document_units  # both populated, same as the OCR path


def test_word_box_coordinates_scale_with_render_dpi() -> None:
    """word_boxes must land in the same pixel space as the image the
    caller renders this page's preview at (pdf_import.py passes its own
    dpi through) — verified here at two different DPIs directly, since a
    silent mismatch would misalign any feature that overlays word boxes
    on the preview image (a future searchable-PDF export, for instance).
    """
    page = _page_with_words(MIN_WORDS_FOR_NATIVE_TEXT + 2)

    result_150 = extract_native_text_result(page, render_dpi=150)
    result_300 = extract_native_text_result(page, render_dpi=300)

    assert result_150 is not None and result_300 is not None
    x_150 = result_150.word_boxes[0].rect.x
    x_300 = result_300.word_boxes[0].rect.x
    assert x_150 > 0 and x_300 > 0
    assert x_300 == pytest.approx(x_150 * 2, abs=2)  # 300 DPI is exactly 2x 150 DPI
