"""Shared fixtures for OCR integration tests.

These tests exercise the real Tesseract binary — that is the whole point
of Phase 2 (proving OCR quality empirically, not assuming it). They are
skipped, not faked, when no Tesseract install can be found.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_TESSDATA = _PROJECT_ROOT / "tessdata"

_WINDOWS_FALLBACK_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
_WINDOWS_FALLBACK_FONT = Path(r"C:\Windows\Fonts\arial.ttf")


def _find_tesseract_cmd() -> str | None:
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    if _WINDOWS_FALLBACK_CMD.exists():
        return str(_WINDOWS_FALLBACK_CMD)
    return None


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


TESSERACT_CMD = _find_tesseract_cmd()
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

    tessdata_dir = _LOCAL_TESSDATA if _LOCAL_TESSDATA.exists() else None
    return OcrEngine(lang="ara+eng", tesseract_cmd=TESSERACT_CMD, tessdata_dir=tessdata_dir)


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
    lines: list[tuple[str, bool]], width: int = 900, height: int | None = None, font_size: int = 32
) -> np.ndarray:
    """lines: list of (text, is_rtl). Returns a BGR numpy array (OpenCV-style)."""
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
