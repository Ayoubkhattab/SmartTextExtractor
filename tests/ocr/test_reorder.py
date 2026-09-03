"""Integration test for column-aware reading order (§7.1.1) against a real
two-column synthetic page, run through the real Tesseract engine.

This is the highest-risk, least-validated part of the OCR pipeline (see
reorder.py's module docstring) — one clean synthetic case is not proof the
heuristic holds on a real scanned multi-column document, but it is real
evidence the mechanism works at all, which a mocked test could not give.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tests.ocr.conftest import ARABIC_FONT, render_rtl_text, requires_arabic_font, requires_tesseract


def _two_column_image() -> np.ndarray:
    img = Image.new("RGB", (900, 260), "white")
    font = ImageFont.truetype(str(ARABIC_FONT), 28)
    draw = ImageDraw.Draw(img)

    english_lines = ["First English line", "Second English line", "Third English line"]
    arabic_lines = ["السطر العربي الأول", "السطر العربي الثاني", "السطر العربي الثالث"]

    for i, text in enumerate(english_lines):
        draw.text((30, 30 + i * 60), text, font=font, fill="black")
    for i, text in enumerate(arabic_lines):
        render_rtl_text(draw, (500, 30 + i * 60), text, font, "black")

    return np.array(img)[:, :, ::-1]


@requires_tesseract
@requires_arabic_font
def test_arabic_majority_page_reads_right_column_before_left_column(ocr_engine) -> None:
    image = _two_column_image()

    result = ocr_engine.run(image)

    arabic_first_pos = result.raw_text.find("الأول")
    english_first_pos = result.raw_text.find("First")
    assert arabic_first_pos != -1, f"Arabic text not found in: {result.raw_text!r}"
    assert english_first_pos != -1, f"English text not found in: {result.raw_text!r}"
    assert arabic_first_pos < english_first_pos, (
        "expected the right-hand (Arabic) column to be read before the left-hand "
        f"(English) column on a majority-Arabic page; got: {result.raw_text!r}"
    )


@requires_tesseract
@requires_arabic_font
def test_lines_within_each_column_stay_top_to_bottom(ocr_engine) -> None:
    image = _two_column_image()

    result = ocr_engine.run(image)

    first_pos = result.raw_text.find("الأول")
    second_pos = result.raw_text.find("الثاني")
    third_pos = result.raw_text.find("الثالث")
    assert -1 not in (first_pos, second_pos, third_pos), result.raw_text
    assert first_pos < second_pos < third_pos, result.raw_text
