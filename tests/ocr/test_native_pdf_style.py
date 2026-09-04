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


@requires_real_pdf
class TestPageGeometry:
    """Reproducing the source page's paper size and margins is what keeps an
    export from re-flowing every line — measured, not assumed: one of this
    project's documents is A4 and another is US Letter."""

    def _result(self):
        import pymupdf

        from smart_text_extractor.ocr.native_pdf_text import extract_native_text_result

        with pymupdf.open(str(REAL_PDF)) as document:
            return extract_native_text_result(document.load_page(0), render_dpi=300)

    def test_page_size_is_the_real_a4_of_the_source(self) -> None:
        layout = self._result().page_layout

        assert layout.width_points == pytest.approx(595.3, abs=1)
        assert layout.height_points == pytest.approx(841.9, abs=1)

    def test_margins_come_from_where_the_text_actually_sits(self) -> None:
        layout = self._result().page_layout

        assert layout.margin_left == pytest.approx(70.8, abs=1)
        assert layout.margin_right == pytest.approx(56.6, abs=1)

    def test_a_real_vertical_gap_is_recorded_as_space_above_the_block(self) -> None:
        """The source page has a wide blank band between its title block and
        the introduction; without recording it the export spaces every
        block evenly and needs manual re-spacing."""
        units = self._result().document_units
        intro = next(u for u in units if "مقدمة" in "".join(s.text for s in u.segments))

        assert intro.space_before_points > 50

    def test_ordinary_line_spacing_is_not_mistaken_for_a_gap(self) -> None:
        units = self._result().document_units
        body = [u for u in units if u.kind == "paragraph" and u.space_before_points == 0]

        assert body, "consecutive body paragraphs should carry no extra space"


class TestPageLayoutInWord:
    def test_word_uses_the_sources_paper_and_margins(self, tmp_path) -> None:
        import docx

        from smart_text_extractor.core.models import PageLayout
        from smart_text_extractor.export.docx_export import export_docx

        layout = PageLayout(595.3, 841.9, 70.8, 56.6, 56.6, 56.6)
        out = tmp_path / "geometry.docx"

        export_docx([[]], out, layout)

        section = docx.Document(str(out)).sections[0]
        assert section.page_width.pt == pytest.approx(595.3, abs=1)
        assert section.page_height.pt == pytest.approx(841.9, abs=1)
        # the text column is what decides where lines break
        text_width = section.page_width.pt - section.left_margin.pt - section.right_margin.pt
        assert text_width == pytest.approx(595.3 - 70.8 - 56.6, abs=1)

    def test_without_a_layout_word_keeps_its_own_defaults(self, tmp_path) -> None:
        """A page with no known geometry (an image, an OCR'd scan) must not
        get a made-up page size imposed on it."""
        import docx

        from smart_text_extractor.export.docx_export import export_docx

        out = tmp_path / "default.docx"

        export_docx([[]], out)

        section = docx.Document(str(out)).sections[0]
        assert section.page_width.pt == pytest.approx(612, abs=1)  # Word's own Letter default
