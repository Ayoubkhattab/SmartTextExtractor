"""Pure-logic unit tests for reorder.py — no real Tesseract needed, unlike
test_engine.py/test_reorder.py which exercise the real binary."""
from __future__ import annotations

from smart_text_extractor.core.models import BoundingBox, Rect
from smart_text_extractor.ocr.reorder import (
    Line,
    _is_majority_arabic,
    _is_mostly_latin,
    _split_line_into_column_runs,
    group_into_lines,
    merge_dual_language_passes,
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


def test_is_mostly_latin() -> None:
    assert _is_mostly_latin("Hello") is True
    assert _is_mostly_latin("مرحبا") is False
    assert _is_mostly_latin("") is False
    assert _is_mostly_latin("123") is False  # digits aren't letters


def _tagged(text: str, confidence: float, rect: Rect) -> tuple[BoundingBox, int, int, int]:
    """A words_from_tsv()-shaped entry, for tests that build fixtures
    directly instead of going through a fake TSV dict."""
    return (BoundingBox(text=text, rect=rect, confidence=confidence), 1, 0, 1)


class TestMergeDualLanguagePasses:
    """Regression tests for a real bug (docs/phases/phase-2-ocr-pipeline.md):
    running lang="ara+eng" sometimes misclassifies isolated Arabic words as
    Latin garbage. All fixtures use the exact real words/confidences
    captured from the document that exposed this."""

    def test_replaces_low_confidence_latin_word_with_higher_confidence_arabic_match(self) -> None:
        rect = Rect(100, 50, 30, 20)
        primary = [_tagged("Fro", 18.0, rect)]
        arabic_only = [_tagged("صريح", 27.0, rect)]

        merged = merge_dual_language_passes(primary, arabic_only)

        assert merged[0][0].text == "صريح"

    def test_keeps_high_confidence_english_word_over_a_lower_confidence_arabic_match(self) -> None:
        """The exact real regression: "Plan" (conf 96) has an ara-only
        "alternative" ("مقا", conf 52) that IS mostly-Arabic-script but is
        actually garbage from forcing Arabic classification onto English
        glyphs — lower confidence than the correct primary reading."""
        rect = Rect(200, 50, 40, 20)
        primary = [_tagged("Plan", 96.0, rect)]
        arabic_only = [_tagged("مقا", 52.0, rect)]

        merged = merge_dual_language_passes(primary, arabic_only)

        assert merged[0][0].text == "Plan"

    def test_keeps_english_word_when_arabic_pass_produces_non_arabic_garbage(self) -> None:
        """"Software" -> ara-only pass produces "5011100121" (digit soup,
        not Arabic script) — must never replace real English regardless
        of confidence, since it isn't a valid alternative at all."""
        rect = Rect(0, 0, 60, 20)
        primary = [_tagged("Software", 96.0, rect)]
        arabic_only = [_tagged("5011100121", 99.0, rect)]  # even if "confident", it's not Arabic

        merged = merge_dual_language_passes(primary, arabic_only)

        assert merged[0][0].text == "Software"

    def test_leaves_arabic_words_untouched(self) -> None:
        rect = Rect(0, 0, 50, 20)
        primary = [_tagged("مرحبا", 90.0, rect)]
        arabic_only = [_tagged("مرحبا", 10.0, rect)]  # irrelevant — primary was never Latin

        merged = merge_dual_language_passes(primary, arabic_only)

        assert merged[0][0].text == "مرحبا"
        assert merged[0][0].confidence == 90.0

    def test_no_overlapping_arabic_word_leaves_primary_untouched(self) -> None:
        primary = [_tagged("Fro", 18.0, Rect(100, 50, 30, 20))]
        arabic_only = [_tagged("شيء", 99.0, Rect(500, 500, 30, 20))]  # far away, no overlap

        merged = merge_dual_language_passes(primary, arabic_only)

        assert merged[0][0].text == "Fro"
