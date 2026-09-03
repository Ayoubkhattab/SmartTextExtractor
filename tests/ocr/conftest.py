"""Shared fixtures for OCR integration tests.

These tests exercise the real Tesseract binary — that is the whole point
of Phase 2 (proving OCR quality empirically, not assuming it). They are
skipped, not faked, when no Tesseract install can be found.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from smart_text_extractor.ocr.locate import find_tessdata_dir, find_tesseract_cmd

_WINDOWS_FALLBACK_FONT = Path(r"C:\Windows\Fonts\arial.ttf")


def _find_arabic_capable_font() -> Path | None:
    if _WINDOWS_FALLBACK_FONT.exists():
        return _WINDOWS_FALLBACK_FONT
    fonts_dir = Path("/usr/share/fonts")
    if not fonts_dir.exists():
        return None
    # Prefer a font whose name says it covers Arabic — most Linux systems
    # ship several ttf files, and not all of them have Arabic glyphs.
    named_candidates = sorted(fonts_dir.rglob("*Arabic*.ttf")) + sorted(fonts_dir.rglob("*arabic*.ttf"))
    if named_candidates:
        return named_candidates[0]
    any_candidates = sorted(fonts_dir.rglob("*.ttf"))
    return any_candidates[0] if any_candidates else None


TESSERACT_CMD = find_tesseract_cmd()
ARABIC_FONT = _find_arabic_capable_font()

requires_tesseract = pytest.mark.skipif(
    TESSERACT_CMD is None, reason="no Tesseract install found on this machine"
)
requires_arabic_font = pytest.mark.skipif(
    ARABIC_FONT is None, reason="no Arabic-capable font found on this machine"
)


@pytest.fixture()
def ocr_engine():
    from smart_text_extractor.ocr.engine import OcrEngine

    return OcrEngine(lang="ara+eng", tesseract_cmd=TESSERACT_CMD, tessdata_dir=find_tessdata_dir())


def render_rtl_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.FreeTypeFont, fill) -> None:
    """Draw Arabic text correctly despite Pillow lacking libraqm shaping.

    Real scanned documents never need this — their ink is already shaped.
    This exists only so tests can synthesize a valid Arabic input image.
    """
    import arabic_reshaper
    from bidi.algorithm import get_display

    visual = get_display(arabic_reshaper.reshape(text))
    draw.text(xy, visual, font=font, fill=fill)


def make_text_image(
    lines: list[tuple[str, bool]], width: int = 800, height: int | None = None, font_size: int = 32
) -> np.ndarray:
    """lines: list of (text, is_rtl). Returns a BGR numpy array (OpenCV-style).

    width=800/font_size=32 is not an arbitrary default — bisecting this
    exact two-line layout (see docs/phases/phase-2-ocr-pipeline.md) found
    Arabic recognition success is NON-monotonic in font_size with this
    Tesseract build (32 and 40 fully correct; 36 and 44 returned nothing
    for the Arabic line entirely; 20/24/28 misread "2026" as "6"). There is
    no simple "bigger is safer" rule here — this combination is the one
    with the most empirical confirmation of working cleanly, not a proof
    it always will. Tests that depend on this function keep their
    assertions loose (word presence, not exact strings) precisely because
    this rendering path (Pillow without libraqm, synthetic glyphs) is
    known to be more fragile across environments than a real scanned
    document ever is — see the OcrEngine module docstring.
    """
    line_height = font_size + 20
    height = height or (line_height * len(lines) + 40)
    img = Image.new("RGB", (width, height), "white")
    font = ImageFont.truetype(str(ARABIC_FONT), font_size)
    draw = ImageDraw.Draw(img)
    for i, (text, is_rtl) in enumerate(lines):
        y = 20 + i * line_height
        if is_rtl:
            render_rtl_text(draw, (20, y), text, font, "black")
        else:
            draw.text((20, y), text, font=font, fill="black")
    return np.array(img)[:, :, ::-1]  # RGB -> BGR


def make_degraded_arabic_image(text: str, width: int = 1000, height: int = 200) -> np.ndarray:
    """A synthetic stand-in for a real poor-quality capture: skewed,
    low-contrast (gray-on-off-white, not black-on-white), and noisy.

    This is what actually caught the real bug it exists to regression-test:
    OcrEngine.run() used to send the raw image straight to Tesseract,
    skipping the deskew/contrast/denoise pipeline entirely even though
    that pipeline was built and tested — on a clean image the gap didn't
    show, but on an image shaped like this one, un-preprocessed OCR came
    back as three garbage characters instead of the sentence.
    """
    import cv2

    img = Image.new("RGB", (width, height), (235, 230, 220))
    font = ImageFont.truetype(str(ARABIC_FONT), 34)
    draw = ImageDraw.Draw(img)
    render_rtl_text(draw, (30, 60), text, font, (90, 85, 80))
    arr = np.array(img)[:, :, ::-1]

    rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), 6.0, 1.0)
    arr = cv2.warpAffine(arr, rotation_matrix, (width, height), borderValue=(235, 230, 220))

    rng = np.random.default_rng(seed=7)
    noise = rng.normal(0, 15, arr.shape).astype(np.int16)
    return np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
