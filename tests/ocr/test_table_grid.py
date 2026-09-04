"""Tests for reading a page's drawn table grids (ocr/table_grid.py).

The real-document numbers referenced here were measured on
docs/دليل الاستخدام.pdf: genuine tables of 9x2, 5x3 and 15x3, and one
highlighted line that the grid finder reports as a 1-row "table".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from smart_text_extractor.core.models import BoundingBox, Rect
from smart_text_extractor.ocr.table_grid import TableGrid, build_table_units, detect_table_grids

REAL_PDF = Path("docs/دليل الاستخدام.pdf")
requires_real_pdf = pytest.mark.skipif(
    not REAL_PDF.exists(), reason="real test documents are not present in this checkout"
)


def _word(text: str, x: int, y: int, width: int = 60, height: int = 20) -> BoundingBox:
    return BoundingBox(text=text, rect=Rect(x=x, y=y, width=width, height=height), confidence=100.0)


def _grid(rows: int, columns: int, cell_width: int = 100, cell_height: int = 40) -> TableGrid:
    cells = [
        [
            Rect(x=column * cell_width, y=row * cell_height, width=cell_width, height=cell_height)
            for column in range(columns)
        ]
        for row in range(rows)
    ]
    return TableGrid(bbox=Rect(0, 0, columns * cell_width, rows * cell_height), cells=cells)


class TestBuildTableUnits:
    def test_words_land_in_the_cell_they_sit_in(self) -> None:
        grid = _grid(rows=2, columns=2)
        words = [
            _word("يمين", x=110, y=10),  # row 0, column 1
            _word("يسار", x=10, y=10),  # row 0, column 0
            _word("ثان", x=10, y=50),  # row 1, column 0
        ]

        units, consumed = build_table_units([grid], words)

        assert len(units) == 1
        assert consumed == {0, 1, 2}
        # RTL: the cell recorded rightmost is the first one read
        assert "".join(s.text for s in units[0].rows[0][0]) == "يمين"
        assert "".join(s.text for s in units[0].rows[0][1]) == "يسار"

    def test_cell_order_is_left_to_right_for_latin(self) -> None:
        grid = _grid(rows=2, columns=2)
        words = [_word("Left", x=10, y=10), _word("Right", x=110, y=10)]

        units, _ = build_table_units([grid], words, rtl=False)

        assert "".join(s.text for s in units[0].rows[0][0]) == "Left"
        assert "".join(s.text for s in units[0].rows[0][1]) == "Right"

    def test_words_outside_the_grid_are_left_for_the_ordinary_flow(self) -> None:
        grid = _grid(rows=2, columns=2)
        words = [_word("داخل", x=10, y=10), _word("خارج", x=800, y=800)]

        units, consumed = build_table_units([grid], words)

        assert consumed == {0}
        assert "خارج" not in "".join(s.text for row in units[0].rows for cell in row for s in cell)

    def test_a_word_is_never_claimed_by_two_tables(self) -> None:
        """Consumed words must not reappear: a table's contents showing up
        again as loose paragraphs behind it is the duplication this guards."""
        overlapping = [_grid(rows=2, columns=2), _grid(rows=2, columns=2)]
        words = [_word("مرة", x=10, y=10)]

        units, consumed = build_table_units(overlapping, words)

        assert consumed == {0}
        assert len(units) == 1  # the second grid captured nothing, so it is not emitted

    def test_an_empty_grid_produces_no_unit(self) -> None:
        units, consumed = build_table_units([_grid(rows=2, columns=2)], [])

        assert units == []
        assert consumed == set()

    def test_word_styling_survives_into_the_cell(self) -> None:
        from smart_text_extractor.core.models import TextStyle

        grid = _grid(rows=2, columns=2)
        styled = BoundingBox("عنوان", Rect(10, 10, 60, 20), 100.0, TextStyle(font_size=12.0, bold=True))

        units, _ = build_table_units([grid], [styled])

        assert units[0].rows[0][1][0].style.bold is True


@requires_real_pdf
class TestDetectTableGridsOnRealPages:
    def _grids(self, page_index: int):
        import pymupdf

        with pymupdf.open(str(REAL_PDF)) as document:
            return detect_table_grids(document.load_page(page_index), render_dpi=300)

    def test_the_two_real_tables_on_page_two_are_found(self) -> None:
        grids = self._grids(1)

        assert len(grids) == 2
        assert [len(grid.cells) for grid in grids] == [9, 5]

    def test_a_highlighted_line_is_not_mistaken_for_a_table(self) -> None:
        """Measured false positive: the grid finder reports the pale band
        drawn behind a highlighted line as a 1-row, 4-column table."""
        assert self._grids(0) == []
