"""Pure-logic unit tests for reorder.py — no real Tesseract needed, unlike
test_engine.py/test_reorder.py which exercise the real binary."""
from __future__ import annotations

from smart_text_extractor.core.models import BoundingBox, Rect
from smart_text_extractor.ocr.reorder import (
    Line,
    _is_majority_arabic,
    _split_line_into_column_runs,
    group_into_lines,
    words_from_tsv,
)


def _fake_tsv(rows: list[tuple[str, float, int, int, int, int, int, int]]) -> dict:
    """rows: (text, conf, left, top, width, height, block, line)"""
    return {
        "text": [r[0] for r in rows],
        "conf": [r[1] for r in rows],
        "left": [r[2] for r in rows],
        "top": [r[3] for r in rows],
        "width": [r[4] for r in rows],
        "height": [r[5] for r in rows],
        "block_num": [r[6] for r in rows],
        "par_num": [0 for _ in rows],
        "line_num": [r[7] for r in rows],
    }


def test_words_from_tsv_drops_empty_and_negative_confidence_rows() -> None:
    data = _fake_tsv(
        [
            ("", 95.0, 0, 0, 10, 10, 1, 1),  # empty text — a block/line summary row
            ("Hello", -1.0, 0, 0, 10, 10, 1, 1),  # negative conf — also a summary row
            ("World", 88.5, 20, 0, 30, 10, 1, 1),
        ]
    )
    result = words_from_tsv(data)
    assert len(result) == 1
    box, block_num, par_num, line_num = result[0]
    assert box.text == "World"
    assert box.confidence == 88.5
    assert (block_num, par_num, line_num) == (1, 0, 1)


def test_group_into_lines_preserves_first_seen_order() -> None:
    box_a = (BoundingBox("a", Rect(0, 0, 10, 10), 90.0), 1, 0, 1)
    box_b = (BoundingBox("b", Rect(20, 0, 10, 10), 90.0), 1, 0, 1)
    box_c = (BoundingBox("c", Rect(0, 50, 10, 10), 90.0), 1, 0, 2)

    lines = group_into_lines([box_a, box_b, box_c])

    assert len(lines) == 2
    assert [w.text for w in lines[0].words] == ["a", "b"]
    assert [w.text for w in lines[1].words] == ["c"]


def test_is_majority_arabic() -> None:
    assert _is_majority_arabic("مرحبا بكم") is True
    assert _is_majority_arabic("Hello World") is False
    assert _is_majority_arabic("123 456") is False  # no letters at all
    assert _is_majority_arabic("") is False


def test_split_line_into_column_runs_splits_on_a_wide_gap() -> None:
    line = Line(
        words=[
            BoundingBox("First", Rect(30, 0, 60, 20), 90.0),
            BoundingBox("Second", Rect(100, 0, 60, 20), 90.0),
            BoundingBox("far", Rect(600, 0, 40, 20), 90.0),
        ],
        block_num=1,
        par_num=0,
        line_num=1,
    )
    runs = _split_line_into_column_runs(line)
    assert len(runs) == 2
    assert [w.text for w in runs[0]] == ["First", "Second"]
    assert [w.text for w in runs[1]] == ["far"]


def test_split_line_into_column_runs_no_split_when_words_are_close() -> None:
    line = Line(
        words=[
            BoundingBox("First", Rect(30, 0, 60, 20), 90.0),
            BoundingBox("Second", Rect(100, 0, 60, 20), 90.0),
        ],
        block_num=1,
        par_num=0,
        line_num=1,
    )
    runs = _split_line_into_column_runs(line)
    assert len(runs) == 1
    assert [w.text for w in runs[0]] == ["First", "Second"]
