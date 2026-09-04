"""Reads a page's real table grids instead of inferring them from word gaps.

The tables in this project's documents are DRAWN: the file records their
rules as stroked lines (20 horizontal and 7 vertical on one real page), and
PyMuPDF reads that into an exact grid — 9x2 and 5x3 tables on one page,
15x3 on the next. Everything before this guessed at the same structure from
the horizontal distance between words, which is why a page that is plainly
a table could come out as a stack of paragraphs; the scorecard measured one
such page at 18.8% visual similarity.

GRID FROM THE FILE, TEXT FROM THE PIPELINE. Only the geometry is taken from
PyMuPDF. Its own cell text comes back in visual order — "الوصف المختصر"
arrives as "رصتخملا فصولا" — which is the same reversal ocr/native_pdf_text
already corrects, so cells are filled from words this pipeline has already
put in reading order and repaired. Taking the text too would reintroduce a
bug that is already fixed.

That split also means the grid is usable on a page whose text has to come
from OCR: the rules are drawn regardless of whether the text layer is
trustworthy.
"""
from __future__ import annotations

from dataclasses import dataclass

import pymupdf

from smart_text_extractor.core.models import BoundingBox, DocumentUnit, Rect, TextSegment

MIN_TABLE_ROWS = 2
MIN_TABLE_COLUMNS = 2
"""A table needs BOTH dimensions. Measured on a real page: the grid finder
reported a highlighted line as a 1-row, 4-column "table" — the pale
rectangle drawn behind the text looks like a bordered band — while the
document's genuine tables were 9x2, 5x3 and 15x3. Requiring two rows and
two columns separates them exactly, and a single band is not a table under
any reading."""


@dataclass(frozen=True)
class TableGrid:
    """One table's geometry, in the same pixel space as the page's words."""

    bbox: Rect
    cells: list[list[Rect | None]]  # row-major; None where the file records no cell


def _scaled(bbox, points_to_pixels: float) -> Rect:
    return Rect(
        x=round(bbox[0] * points_to_pixels),
        y=round(bbox[1] * points_to_pixels),
        width=round((bbox[2] - bbox[0]) * points_to_pixels),
        height=round((bbox[3] - bbox[1]) * points_to_pixels),
    )


def detect_table_grids(page: pymupdf.Page, render_dpi: int) -> list[TableGrid]:
    """The drawn tables on this page, or an empty list if it has none."""
    points_to_pixels = render_dpi / 72
    grids: list[TableGrid] = []

    try:
        found = page.find_tables()
    except Exception:  # noqa: BLE001 - a page whose grids cannot be read simply has none
        return []

    for table in found.tables:
        if table.row_count < MIN_TABLE_ROWS or table.col_count < MIN_TABLE_COLUMNS:
            continue
        cells = [[_scaled(cell, points_to_pixels) if cell else None for cell in row.cells] for row in table.rows]
        grids.append(TableGrid(bbox=_scaled(table.bbox, points_to_pixels), cells=cells))
    return grids


def _centre_inside(cell: Rect, word: Rect) -> bool:
    centre_x = word.x + word.width / 2
    centre_y = word.y + word.height / 2
    return cell.x <= centre_x <= cell.x + cell.width and cell.y <= centre_y <= cell.y + cell.height


def _cell_segments(words: list[BoundingBox], rtl: bool) -> list[TextSegment]:
    """One cell's words, in reading order, as segments."""
    ordered = sorted(words, key=lambda word: (word.rect.y, -word.rect.x if rtl else word.rect.x))
    segments: list[TextSegment] = []
    for index, word in enumerate(ordered):
        if index:
            segments.append(TextSegment(" ", None))
        segments.append(TextSegment(word.text, word.confidence, word.style))
    return segments


def build_table_units(
    grids: list[TableGrid], words: list[BoundingBox], rtl: bool = True
) -> tuple[list[DocumentUnit], set[int]]:
    """Fills each grid from `words` and reports which words it consumed.

    The returned index set lets the caller drop those words from the
    ordinary flow, so a table's contents are not emitted twice — once as
    the table and again as loose paragraphs behind it.
    """
    units: list[DocumentUnit] = []
    consumed: set[int] = set()

    for grid in grids:
        rows: list[list[list[TextSegment]]] = []
        used_here: set[int] = set()

        for row in grid.cells:
            row_segments: list[list[TextSegment]] = []
            for cell in row:
                if cell is None:
                    row_segments.append([])
                    continue
                inside = [
                    (index, word)
                    for index, word in enumerate(words)
                    if index not in consumed and _centre_inside(cell, word.rect)
                ]
                used_here.update(index for index, _ in inside)
                row_segments.append(_cell_segments([word for _, word in inside], rtl))
            # Cells are recorded left to right; Arabic reads the first cell
            # on the right, and every renderer downstream expects that
            # order (it reverses the visual column itself).
            rows.append(list(reversed(row_segments)) if rtl else row_segments)

        if not any(cell for row in rows for cell in row):
            continue

        consumed.update(used_here)
        units.append(DocumentUnit(kind="table", rows=rows, bbox=grid.bbox))

    return units, consumed
