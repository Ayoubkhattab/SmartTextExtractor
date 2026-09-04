"""Calibration contract for the visual-similarity metric.

A similarity metric that has not been calibrated is not evidence. The
first version of this metric — ink-pixel overlap — passed every unit test
that checked its arithmetic and was still useless: measured against real
pages, a faithful reconstruction scored 8-21% while a COMPLETELY DIFFERENT
page scored 19-24%, so the number could not tell the two apart.

These tests assert the ordering that has to hold for the metric to mean
anything, on synthetic pages whose layouts are known by construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from smart_text_extractor.quality.visual_similarity import compare_pages

WIDTH, HEIGHT = 595, 842


def _page(bands: list[tuple[int, int, int, int]], path: Path) -> Path:
    """A white page with black bands at the given (x0, y0, x1, y1) boxes —
    a stand-in for blocks of text, which is all the metric looks at."""
    canvas = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in bands:
        canvas[y0:y1, x0:x1] = 0
    Image.fromarray(canvas).save(path)
    return path


@pytest.fixture()
def pages(tmp_path: Path) -> dict[str, Path]:
    single_column = [(70, 100, 520, 130), (70, 160, 520, 300), (70, 340, 520, 480)]
    return {
        # the same layout, drawn lighter/thinner — a reconstruction that
        # could not use the source's font
        "source": _page(single_column, tmp_path / "source.png"),
        "faithful": _page([(75, 105, 515, 125), (75, 165, 515, 295), (75, 345, 515, 475)], tmp_path / "faithful.png"),
        # content in entirely different places down the page
        "displaced": _page([(70, 500, 520, 530), (70, 600, 520, 740)], tmp_path / "displaced.png"),
        # two columns instead of one — the layout difference that matters most
        "two_column": _page([(70, 100, 280, 480), (320, 100, 520, 480)], tmp_path / "two_column.png"),
        "blank": _page([], tmp_path / "blank.png"),
    }


def test_a_page_against_itself_scores_full_marks(pages) -> None:
    assert compare_pages(pages["source"], pages["source"]).percent == pytest.approx(100.0, abs=0.5)


def test_a_faithful_reconstruction_scores_far_above_a_displaced_one(pages) -> None:
    """The ordering the first metric failed: content in the right places
    must beat content in the wrong places, even when the reconstruction
    carries less ink than the source."""
    faithful = compare_pages(pages["source"], pages["faithful"]).percent
    displaced = compare_pages(pages["source"], pages["displaced"]).percent

    assert faithful > displaced + 20, f"faithful={faithful:.1f}% displaced={displaced:.1f}%"


def test_less_ink_in_the_right_place_still_scores_well(pages) -> None:
    """A reconstruction cannot use the source's own font, so it will always
    carry a fraction of the ink — measured at 0.14x on a real page. That
    must not be punished as a layout error; an earlier version folded it
    into the score and buried faithful rebuilds below unrelated pages."""
    score = compare_pages(pages["source"], pages["faithful"])

    assert score.ink_ratio < 1.0  # genuinely lighter
    assert score.percent > 80


def test_collapsing_two_columns_into_one_is_detected(pages) -> None:
    """The metric's main job: a page whose sidebar was flattened into the
    flow has to score below one that kept its columns."""
    kept = compare_pages(pages["two_column"], pages["two_column"]).percent
    collapsed = compare_pages(pages["two_column"], pages["source"]).percent

    assert kept > collapsed
    assert collapsed < 90


def test_column_structure_is_what_the_horizontal_profile_sees(pages) -> None:
    same_columns = compare_pages(pages["two_column"], pages["two_column"]).horizontal
    different_columns = compare_pages(pages["two_column"], pages["source"]).horizontal

    assert same_columns > different_columns


def test_an_empty_reconstruction_scores_nothing(pages) -> None:
    """A page that produced no content must not score well by having no
    disagreements — silence is the worst outcome, not a neutral one."""
    score = compare_pages(pages["source"], pages["blank"])

    assert score.percent == 0.0
    assert score.ink_ratio == 0.0
