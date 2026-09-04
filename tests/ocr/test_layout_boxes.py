"""Tests for reproducing the panels a page draws its content in
(ocr/layout_boxes.py).

Fixture positions follow the real page this was built against: pale panels
at #f5f2e8 and dark ones at #17332f, laid out in a narrow left column.
"""
from __future__ import annotations

from smart_text_extractor.core.models import DocumentUnit, Rect, TextSegment
from smart_text_extractor.ocr.layout_boxes import group_units_into_boxes

PALE = "#f5f2e8"
DARK = "#17332f"


def _unit(text: str, rect: Rect, kind: str = "paragraph") -> DocumentUnit:
    return DocumentUnit(kind=kind, segments=[TextSegment(text, 100.0)], bbox=rect)


class TestGroupUnitsIntoBoxes:
    def test_units_inside_a_panel_become_one_box_carrying_its_fill(self) -> None:
        box = (Rect(50, 140, 205, 86), PALE)
        units = [
            _unit("الوحدات الداخلية", Rect(60, 150, 150, 20)),
            _unit("وحدة الخدمات الخلفية", Rect(60, 175, 180, 20)),
        ]

        result = group_units_into_boxes(units, [box])

        assert len(result) == 1
        assert result[0].kind == "table"
        assert result[0].box_fill == PALE
        assert "الوحدات الداخلية" in "".join(s.text for s in result[0].rows[0][0])
        assert "وحدة الخدمات الخلفية" in "".join(s.text for s in result[0].rows[0][0])

    def test_units_outside_every_panel_are_untouched_and_keep_their_order(self) -> None:
        box = (Rect(50, 140, 205, 86), PALE)
        prose = _unit("نص خارج الصندوق", Rect(300, 400, 250, 20))
        inside = _unit("داخل", Rect(60, 150, 150, 20))

        result = group_units_into_boxes([prose, inside], [box])

        assert result[0] is prose
        assert result[1].box_fill == PALE

    def test_a_box_keeps_the_position_of_the_first_unit_it_captured(self) -> None:
        """Reading order is preserved rather than rebuilt: the panel appears
        where its content appeared."""
        box = (Rect(50, 140, 205, 86), DARK)
        units = [
            _unit("قبل", Rect(300, 100, 250, 20)),
            _unit("داخل", Rect(60, 150, 150, 20)),
            _unit("بعد", Rect(300, 400, 250, 20)),
        ]

        result = group_units_into_boxes(units, [box])

        assert [u.box_fill for u in result] == [None, DARK, None]

    def test_separate_panels_stay_separate(self) -> None:
        boxes = [(Rect(50, 140, 205, 86), PALE), (Rect(50, 330, 206, 102), DARK)]
        units = [_unit("أول", Rect(60, 150, 150, 20)), _unit("ثاني", Rect(60, 340, 150, 20))]

        result = group_units_into_boxes(units, boxes)

        assert [u.box_fill for u in result] == [PALE, DARK]

    def test_a_unit_only_partly_over_a_panel_is_not_captured(self) -> None:
        """A unit next to a panel, overlapping its edge, belongs to the page
        rather than to the panel."""
        box = (Rect(50, 140, 205, 86), PALE)
        straddling = _unit("مجاور", Rect(230, 150, 200, 20))  # mostly outside

        result = group_units_into_boxes([straddling], [box])

        assert result[0] is straddling

    def test_the_innermost_panel_wins_when_they_nest(self) -> None:
        outer = (Rect(40, 130, 400, 300), PALE)
        inner = (Rect(50, 140, 205, 86), DARK)
        unit = _unit("داخل", Rect(60, 150, 150, 20))

        result = group_units_into_boxes([unit], [outer, inner])

        assert result[0].box_fill == DARK

    def test_a_table_inside_a_panel_keeps_its_cell_text(self) -> None:
        box = (Rect(50, 140, 205, 120), PALE)
        table = DocumentUnit(
            kind="table",
            rows=[[[TextSegment("أ", 100.0)], [TextSegment("ب", 100.0)]]],
            bbox=Rect(60, 150, 150, 60),
        )

        result = group_units_into_boxes([table], [box])

        text = "".join(s.text for s in result[0].rows[0][0])
        assert "أ" in text and "ب" in text

    def test_no_panels_leaves_everything_alone(self) -> None:
        units = [_unit("نص", Rect(60, 150, 150, 20))]

        assert group_units_into_boxes(units, []) is units
