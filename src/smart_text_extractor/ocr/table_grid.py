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
    stroked: bool = True  # False when the grid was read from shaded cells rather than drawn rules


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

    return _prefer_richer_grids(grids, _grid_from_fills(page, points_to_pixels))


def _contains(outer: Rect, inner: Rect) -> bool:
    return (
        outer.x <= inner.x + 1
        and outer.y <= inner.y + 1
        and outer.x + outer.width >= inner.x + inner.width - 1
        and outer.y + outer.height >= inner.y + inner.height - 1
    )


def _prefer_richer_grids(stroked: list[TableGrid], filled: list[TableGrid]) -> list[TableGrid]:
    """Keeps whichever reading of a region resolves more rows.

    Neither source wins outright. find_tables() is the better reading
    wherever a table's rules are actually stroked — on one real page it
    returns 9x2 and 5x3 where the fill reader collapses both into one 3x3.
    But on a page whose rules are drawn as fills it returned a 5x5
    FRAGMENT of a 19x5 table, and a fragment is worse than nothing: the
    remaining fourteen rows spill out as loose paragraphs behind it.

    So a fill-derived grid replaces the stroked grids it contains only
    when it resolves more rows than they do, and a stroked grid it does
    not contain is kept untouched alongside it.
    """
    if not filled:
        return stroked
    if not stroked:
        return filled

    # On a page where the stroked reader found nothing, the fill grid is
    # the only reading there is (the early return above). Where it found
    # something, the fill grid may only REPLACE a fragment it contains and
    # resolves better — never sit down beside it. Letting an uncontested
    # fill grid be added too was tried and measured: it costs 2.5 points
    # of visual similarity on the ruled document, where it lands a coarse
    # 3x3 over a page the stroked reader had already read as 9x2 and 5x3.
    kept: list[TableGrid] = []
    for candidate in filled:
        inside = [grid for grid in stroked if _contains(candidate.bbox, grid.bbox)]
        if inside and len(candidate.cells) > sum(len(grid.cells) for grid in inside):
            kept.append(candidate)
    if not kept:
        return stroked

    replaced = [grid for grid in stroked if not any(_contains(candidate.bbox, grid.bbox) for candidate in kept)]
    return replaced + kept


TOLERANCE_POINTS = 2.0
"""How far apart two edges can be and still be the same grid line."""

THIN_POINTS = 2.0
"""At or below this, a filled rectangle is a rule, not a cell."""

MIN_ROW_COVERAGE = 0.6
"""How much of the table's width the segments at one y must span together
before that y counts as a row boundary."""


def _cluster(values: list[float], tolerance: float = TOLERANCE_POINTS) -> list[list[float]]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _covered_fraction(segments: list[tuple[float, float]], low: float, high: float) -> float:
    """How much of [low, high] the union of `segments` covers."""
    if high <= low:
        return 0.0
    clipped = sorted((max(a, low), min(b, high)) for a, b in segments)
    merged: list[list[float]] = []
    for start, end in clipped:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged) / (high - low)


def _page_rectangles(page: pymupdf.Page) -> list[tuple[float, float, float, float]]:
    """Every drawn rectangle except the sheet-sized background."""
    page_area = page.rect.width * page.rect.height
    rectangles = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] != "re":
                continue
            rect = item[1]
            if (rect.x1 - rect.x0) * (rect.y1 - rect.y0) < 0.9 * page_area:
                rectangles.append((rect.x0, rect.y0, rect.x1, rect.y1))
    return rectangles


def _grid_from_fills(page: pymupdf.Page, points_to_pixels: float) -> list[TableGrid]:
    """Rebuilds a table drawn as FILLS rather than strokes.

    Measured on a real page: a full-page 19x5 staffing table reported
    strokes=0 — every rule and every shaded cell is a filled rectangle,
    and the height-1 fills that separate the rows are drawn per cell, not
    as one line across the table. find_tables() reads stroked rules, so on
    that page it returned a single 5x5 fragment and the other fourteen
    rows spilled out as 48 loose paragraphs.

    Two constraints keep this from firing on ordinary page decoration,
    both of which it did before they were added:

      columns come only from the SHADED CELLS, so a full-width header rule
      shares no edge with them and cannot invent a column; and

      a row boundary must be witnessed across MIN_ROW_COVERAGE of the
      table's width by the segments at that y taken TOGETHER — which is
      what a per-cell rule looks like, and what a decorative band under a
      heading does not.

    With both in place it finds nothing on either document's title page,
    and nothing behind the highlighted line that MIN_TABLE_ROWS was
    written for.
    """
    rectangles = _page_rectangles(page)
    shaded = [r for r in rectangles if (r[2] - r[0]) > THIN_POINTS and (r[3] - r[1]) > THIN_POINTS]
    if len(shaded) < MIN_TABLE_COLUMNS:
        return []

    columns = [
        sum(group) / len(group)
        for group in _cluster([r[0] for r in shaded] + [r[2] for r in shaded])
        if len(group) >= MIN_TABLE_ROWS
    ]
    if len(columns) - 1 < MIN_TABLE_COLUMNS:
        return []

    left, right = columns[0], columns[-1]
    top = min(r[1] for r in shaded)
    bottom = max(r[3] for r in shaded)

    edges = [(r[1], r[0], r[2]) for r in rectangles] + [(r[3], r[0], r[2]) for r in rectangles]
    edges = [edge for edge in edges if top - TOLERANCE_POINTS <= edge[0] <= bottom + TOLERANCE_POINTS]
    rows = []
    for group in _cluster([edge[0] for edge in edges]):
        y = sum(group) / len(group)
        segments = [(a, b) for edge_y, a, b in edges if abs(edge_y - y) <= TOLERANCE_POINTS]
        if _covered_fraction(segments, left, right) >= MIN_ROW_COVERAGE:
            rows.append(y)
    if len(rows) - 1 < MIN_TABLE_ROWS:
        return []

    cells = [
        [
            _scaled((columns[column], rows[row], columns[column + 1], rows[row + 1]), points_to_pixels)
            for column in range(len(columns) - 1)
        ]
        for row in range(len(rows) - 1)
    ]
    bbox = _scaled((left, rows[0], right, rows[-1]), points_to_pixels)
    return [TableGrid(bbox=bbox, cells=cells, stroked=False)]


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
        units.append(DocumentUnit(kind="table", rows=rows, bbox=grid.bbox, bordered=grid.stroked))

    return units, consumed
