"""Tests for reading a PDF page's visual style (ocr/native_pdf_style.py)
and for the formatting decisions that depend on it.

The real-document assertions use the measured values from
docs/دليل الاستخدام.pdf: 24pt title, 18pt headings, 14pt body, 12pt small
print, and one highlighted line drawn in #f7d1d5.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from smart_text_extractor.core.models import BoundingBox, Rect, TextStyle
from smart_text_extractor.ocr.native_pdf_style import _fill_to_hex, _to_hex

DOCS = Path("docs")
REAL_PDF = DOCS / "دليل الاستخدام.pdf"

requires_real_pdf = pytest.mark.skipif(
    not REAL_PDF.exists(), reason="real test documents are not present in this checkout"
)


class TestColourConversion:
    def test_span_colour_integer_becomes_hex(self) -> None:
        assert _to_hex(0x333333) == "#333333"
        assert _to_hex(0x000000) == "#000000"

    def test_near_white_fill_is_not_a_highlight(self) -> None:
        """Every page has a white background rectangle; treating it as a
        highlight would paint the whole page."""
        assert _fill_to_hex((1.0, 1.0, 1.0)) is None
        assert _fill_to_hex((0.98, 0.99, 0.98)) is None

    def test_real_fill_becomes_hex(self) -> None:
        assert _fill_to_hex((0.97, 0.82, 0.84)) == "#f7d1d6"

    def test_no_fill_is_no_highlight(self) -> None:
        assert _fill_to_hex(None) is None


@requires_real_pdf
class TestStyleFromTheRealDocument:
    def _units(self):
        import pymupdf

        from smart_text_extractor.ocr.native_pdf_text import extract_native_text_result

        with pymupdf.open(str(REAL_PDF)) as document:
            return extract_native_text_result(document.load_page(0), render_dpi=300).document_units

    def test_font_sizes_are_read_from_the_document(self) -> None:
        sizes = {
            segment.style.font_size
            for unit in self._units()
            for segment in unit.segments
            if segment.style and segment.style.font_size
        }
        assert {24.0, 18.0, 14.0, 12.0} <= sizes

    def test_large_text_is_classified_as_a_heading(self) -> None:
        """Measured-height detection missed these: the body is 14pt and the
        headings 18pt, a 1.29x step well under _HEADING_HEIGHT_RATIO's
        1.75, so both headings came through as ordinary paragraphs before
        real font sizes were available."""
        headings = [u for u in self._units() if u.kind == "heading"]
        heading_text = ["".join(s.text for s in u.segments) for u in headings]

        assert any("مقدمة" in text for text in heading_text)
        assert any("كيفية استخدام" in text for text in heading_text)

    def test_the_centred_title_is_detected_as_centred_and_the_body_is_not(self) -> None:
        units = self._units()
        title = units[0]
        body = next(u for u in units if u.kind == "paragraph" and len("".join(s.text for s in u.segments)) > 80)

        assert title.alignment == "center"
        assert body.alignment == "natural"

    def test_the_highlighted_line_carries_its_fill_colour(self) -> None:
        highlights = {
            segment.style.highlight
            for unit in self._units()
            for segment in unit.segments
            if segment.style and segment.style.highlight
        }
        assert highlights == {"#f7d1d5"}


class TestStyleSurvivesTheRepair:
    def test_a_repaired_word_keeps_the_pdf_styling(self) -> None:
        """OCR supplies the letters for a transposition-corrupted word, but
        how that word LOOKS still comes from the PDF — OCR has no styling
        to offer."""
        from smart_text_extractor.ocr.native_text_repair import repair_native_words

        style = TextStyle(font_size=18.0, bold=True, color="#333333")
        native = [BoundingBox("الربمجة", Rect(100, 100, 80, 30), 100.0, style)]
        ocr = [BoundingBox("البرمجة", Rect(100, 105, 78, 20), 91.0, None)]

        repaired = repair_native_words(native, ocr).repaired[0]

        assert repaired.text == "البرمجة"
        assert repaired.style == style
