"""Draws an extracted page back onto blank paper, from the model alone.

This is the honest test of how much of a page we actually captured: it
uses ONLY what extraction produced — unit positions, sizes, colours,
panels — and nothing from the source file. Whatever the model failed to
record simply does not appear.

That distinction is the whole point. The searchable-PDF export pastes the
original page image behind its text, so comparing THAT against the source
would score near-perfect no matter how little structure was understood.
Rendering from the model instead makes the comparison meaningful, and is
what scripts/score_document.py measures against the real page.

It doubles as a debugging view: anything misplaced here is misplaced in
the model, not in a particular renderer.
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

from smart_text_extractor.core.models import DocumentUnit, OcrResult, Rect, TextSegment
from smart_text_extractor.export.pdf_export import find_arabic_capable_font

_FALLBACK_FONTNAME = "helv"
_EMBEDDED_FONTNAME = "recon"
_DEFAULT_FONT_SIZE = 10.0

# Text is drawn from the top of its recorded box, so the baseline sits a
# little below it — roughly the cap height of a typical face.
_BASELINE_FRACTION = 0.8


def _to_points(value: float, dpi: int) -> float:
    return value * 72.0 / dpi


def _rect_to_points(rect: Rect, dpi: int) -> pymupdf.Rect:
    return pymupdf.Rect(
        _to_points(rect.x, dpi),
        _to_points(rect.y, dpi),
        _to_points(rect.x + rect.width, dpi),
        _to_points(rect.y + rect.height, dpi),
    )


def _segments_text(segments: list[TextSegment]) -> str:
    return "".join(segment.text for segment in segments).strip()


def _unit_text(unit: DocumentUnit) -> str:
    if unit.kind != "table":
        return _segments_text(unit.segments)
    lines = []
    for row in unit.rows:
        lines.append(" | ".join(_segments_text(cell) for cell in row))
    return "\n".join(lines)


def _dominant_font_size(unit: DocumentUnit, default: float) -> float:
    sizes = [
        segment.style.font_size
        for segment in (unit.segments or [s for row in unit.rows for cell in row for s in cell])
        if segment.style and segment.style.font_size
    ]
    return max(sizes) if sizes else default


def _dominant_colour(unit: DocumentUnit) -> tuple[float, float, float]:
    for segment in unit.segments or [s for row in unit.rows for cell in row for s in cell]:
        if segment.style and segment.style.color:
            value = segment.style.color.lstrip("#")
            return (int(value[0:2], 16) / 255, int(value[2:4], 16) / 255, int(value[4:6], 16) / 255)
    return (0.0, 0.0, 0.0)


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return (int(value[0:2], 16) / 255, int(value[2:4], 16) / 255, int(value[4:6], 16) / 255)


_MIN_RENDER_FONT_SIZE = 3.0
_SHRINK_STEP = 0.8


def _insert_text_that_fits(page, box, text, *, fontname, font_size, colour, align) -> None:
    """Draws the text, shrinking it until it fits the unit's own box.

    insert_textbox draws NOTHING and returns a negative overflow when the
    text is too big for the rectangle — silently, so a block whose recorded
    box is a little tight simply vanishes. The scorecard caught this
    immediately: whole pages came back with 0.00x the source's ink and a
    visual score of 0%, on pages that plainly had content.

    A reconstruction that drops a block is worse than one that draws it
    slightly small: the point of the render is to show what the model
    captured, and silence is indistinguishable from "we never extracted
    it".
    """
    size = max(font_size, _MIN_RENDER_FONT_SIZE)
    while size >= _MIN_RENDER_FONT_SIZE:
        overflow = page.insert_textbox(
            box, text, fontname=fontname, fontsize=size, color=colour, align=align
        )
        if overflow >= 0:
            return
        size *= _SHRINK_STEP

    # Still too much text for the box: draw what fits on one line rather
    # than nothing, so the block is visibly present and its absence never
    # gets mistaken for a extraction failure.
    page.insert_textbox(
        box, text, fontname=fontname, fontsize=_MIN_RENDER_FONT_SIZE, color=colour, align=align
    )


_GRID_COLOUR = (0.6, 0.6, 0.6)


def _column_edges(widths: list[float], column_count: int, left: float, right: float) -> list[float]:
    """The x of every column boundary, left to right.

    Uses the source's own proportions where the model recorded them — a
    table with a narrow count column beside wide description columns is
    not reproduced by an even split. Falls back to equal shares when the
    proportions are missing or do not match the number of columns, which
    is what the model promises when it cannot measure them.
    """
    span = right - left
    if len(widths) != column_count:
        return [left + span * index / column_count for index in range(column_count + 1)]

    edges, offset = [left], 0.0
    for width in widths:
        offset += width
        edges.append(left + span * offset)
    edges[-1] = right
    return edges


def _cell_shade(cell: list[TextSegment]) -> str | None:
    """The fill most of a cell's words sit on, or None if most sit on none.

    A majority rather than the first match, for the same reason the panel
    uses one: a header cell can hold a stray word the style index did not
    place inside the drawn band, and one such word should neither decide
    nor prevent the whole cell's colour.
    """
    fills = [segment.style.highlight for segment in cell if segment.style and segment.style.highlight]
    words = [segment for segment in cell if segment.confidence is not None]
    if not fills or len(fills) * 2 <= len(words):
        return None
    return max(set(fills), key=fills.count)


def _draw_table(page, box, unit: DocumentUnit, fontname: str, render_dpi: int) -> None:
    """Lays the table's cells out across its own recorded box.

    Cells are given equal shares of the box: the model records where the
    table sits but not its individual column widths, so an even split is
    the honest reconstruction of what is actually known — and it still puts
    ink in several columns, which is the property the source has and a
    flattened line does not.

    Columns run right to left, matching how the cells were read.
    """
    row_count = len(unit.rows)
    column_count = max(len(row) for row in unit.rows)
    if not row_count or not column_count:
        return

    row_height = (box.y1 - box.y0) / row_count
    edges = _column_edges(unit.column_widths, column_count, box.x0, box.x1)

    for row_index, row in enumerate(unit.rows):
        for logical_column, cell in enumerate(row):
            visual_column = column_count - 1 - logical_column
            cell_box = pymupdf.Rect(
                edges[visual_column],
                box.y0 + row_index * row_height,
                edges[visual_column + 1],
                box.y0 + (row_index + 1) * row_height,
            )
            # Reproduce how the page draws the table, not one fixed style:
            # a table found by its stroked rules gets rules, and one
            # recovered from shaded cells gets the shading instead.
            # Drawing a grid over the second kind adds ink the source
            # never had, which measured 34 points of visual similarity on
            # a real page even though the structure was finally right.
            if unit.bordered:
                page.draw_rect(cell_box, color=_GRID_COLOUR, width=0.4)
            else:
                shade = _cell_shade(cell)
                if shade:
                    page.draw_rect(cell_box, color=None, fill=_hex_to_rgb(shade))
            text = _segments_text(cell)
            if text:
                _insert_text_that_fits(
                    page,
                    cell_box,
                    text,
                    fontname=fontname,
                    font_size=_DEFAULT_FONT_SIZE,
                    colour=(0.0, 0.0, 0.0),
                    align=pymupdf.TEXT_ALIGN_RIGHT,
                )


def render_result_to_pdf(result: OcrResult, output_path: Path, render_dpi: int) -> Path:
    """Rebuilds the page from `result` alone and writes it as a one-page PDF.

    render_dpi is the DPI the unit positions were recorded at — the same
    number the rest of the pipeline carries for exactly this reason (§7.3).
    """
    output_path = Path(output_path)
    layout = result.page_layout
    width = layout.width_points if layout else 595.0
    height = layout.height_points if layout else 842.0

    document = pymupdf.open()
    page = document.new_page(width=width, height=height)

    arabic_font = find_arabic_capable_font()
    fontname = _FALLBACK_FONTNAME
    if arabic_font is not None:
        page.insert_font(fontname=_EMBEDDED_FONTNAME, fontfile=str(arabic_font))
        fontname = _EMBEDDED_FONTNAME

    for unit in result.document_units:
        if unit.bbox is None:
            continue
        box = _rect_to_points(unit.bbox, render_dpi)

        # A panel is drawn as the filled shape it is, before its text.
        if unit.box_fill:
            page.draw_rect(box, color=None, fill=_hex_to_rgb(unit.box_fill))

        # A table is drawn as the grid it is. Flattening it to "a | b" lines
        # was measurably wrong, not just untidy: on a table-heavy page the
        # rebuilt ink sat in one column while the source spread it across
        # several, and the page scored 16% when its model was actually
        # sound. The renderer was hiding fidelity the extraction had.
        if unit.kind == "table" and unit.rows and unit.box_fill is None:
            _draw_table(page, box, unit, fontname, render_dpi)
            continue

        text = _unit_text(unit)
        if not text:
            continue

        _insert_text_that_fits(
            page,
            box,
            text,
            fontname=fontname,
            font_size=_dominant_font_size(unit, _DEFAULT_FONT_SIZE),
            colour=_dominant_colour(unit),
            align=pymupdf.TEXT_ALIGN_CENTER if unit.alignment == "center" else pymupdf.TEXT_ALIGN_RIGHT,
        )

    document.save(str(output_path))
    document.close()
    return output_path


def render_result_to_image(result: OcrResult, output_path: Path, render_dpi: int, raster_dpi: int = 150) -> Path:
    """The same reconstruction, rasterised — what the visual score compares."""
    output_path = Path(output_path)
    pdf_path = output_path.with_suffix(".pdf")
    render_result_to_pdf(result, pdf_path, render_dpi)

    with pymupdf.open(str(pdf_path)) as document:
        zoom = raster_dpi / 72
        pixmap = document.load_page(0).get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        pixmap.save(str(output_path))
    return output_path
