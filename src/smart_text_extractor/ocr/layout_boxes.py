"""Turns the panels a page draws its content in back into real boxes.

A designed document does not lay everything out as one column. It puts
sidebar content in filled callout panels — the pale "الوحدات الداخلية" and
"المخرجات" panels, the dark "مؤشرات الأداء" panel — and reading order alone
flattens all of that into a single stream, which is the largest visible
difference between an extracted page and the page it came from.

The panels themselves are recorded in the PDF as filled rectangles
(ocr/native_pdf_style.py's container_boxes), so they do not have to be
inferred from the text at all. Every unit whose position falls inside one
becomes that box's content, and the box is emitted as a single unit
carrying its own fill colour.

A box is represented as a one-cell "table" unit rather than a new kind:
every renderer already draws a table cell with a background, in both the
live panel and the Word export, so a box gets a real filled container in
both without a second rendering path to keep in step.
"""
from __future__ import annotations

from smart_text_extractor.core.models import DocumentUnit, Rect, TextSegment

MIN_CONTAINMENT = 0.7
"""How much of a unit must sit inside a box before it counts as that box's
content. Not 1.0: a panel's text can overhang its rounded corners slightly,
and a heading's measured box often includes a descender past the fill."""


def _containment(box: Rect, unit: Rect) -> float:
    x0, y0 = max(box.x, unit.x), max(box.y, unit.y)
    x1 = min(box.x + box.width, unit.x + unit.width)
    y1 = min(box.y + box.height, unit.y + unit.height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    unit_area = unit.width * unit.height
    return ((x1 - x0) * (y1 - y0)) / unit_area if unit_area else 0.0


def _unit_segments(unit: DocumentUnit) -> list[TextSegment]:
    if unit.kind != "table":
        return unit.segments
    flattened: list[TextSegment] = []
    for row_index, row in enumerate(unit.rows):
        if row_index:
            flattened.append(TextSegment("\n", None))
        for cell_index, cell in enumerate(row):
            if cell_index:
                flattened.append(TextSegment(" | ", None))
            flattened.extend(cell)
    return flattened


def group_units_into_boxes(units: list[DocumentUnit], boxes: list[tuple[Rect, str]]) -> list[DocumentUnit]:
    """Replaces the units drawn inside each panel with one box unit.

    Units outside every panel are returned untouched and in their original
    order; a box takes the position of the first unit it captured, so the
    page's overall reading order is preserved rather than rebuilt.
    """
    if not boxes:
        return units

    # Smallest box first: panels can nest, and the innermost is the one a
    # unit actually belongs to.
    ordered_boxes = sorted(boxes, key=lambda item: item[0].width * item[0].height)

    box_contents: dict[int, list[DocumentUnit]] = {}
    result: list[DocumentUnit] = []
    box_placeholder: dict[int, int] = {}

    for unit in units:
        index = None
        if unit.bbox is not None:
            for candidate, (box_rect, _fill) in enumerate(ordered_boxes):
                if _containment(box_rect, unit.bbox) >= MIN_CONTAINMENT:
                    index = candidate
                    break

        if index is None:
            result.append(unit)
            continue

        if index not in box_contents:
            box_contents[index] = []
            box_placeholder[index] = len(result)
            result.append(unit)  # reserved slot, replaced below
        box_contents[index].append(unit)

    for index, contained in box_contents.items():
        box_rect, fill = ordered_boxes[index]
        segments: list[TextSegment] = []
        for position, unit in enumerate(contained):
            if position:
                segments.append(TextSegment("\n", None))
            segments.extend(_unit_segments(unit))
        result[box_placeholder[index]] = DocumentUnit(
            kind="table",
            rows=[[segments]],
            bbox=box_rect,
            alignment=contained[0].alignment,
            space_before_points=contained[0].space_before_points,
            box_fill=fill,
        )
    return result
