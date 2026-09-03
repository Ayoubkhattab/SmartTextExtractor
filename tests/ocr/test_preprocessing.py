from __future__ import annotations

import cv2
import numpy as np
import pytest

from smart_text_extractor.ocr.preprocessing import deskew, denoise, enhance_contrast, preprocess, upscale_if_small
from tests.ocr.conftest import make_text_image, requires_arabic_font, requires_tesseract


def _measured_skew_angle(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    return -(90 + angle) if angle < -45 else -angle


def _synthetic_text_lines_image(angle_degrees: float = 0.0) -> np.ndarray:
    """White canvas with black horizontal bars simulating text lines, optionally rotated."""
    img = np.full((400, 600), 255, dtype=np.uint8)
    for y in range(40, 360, 40):
        cv2.rectangle(img, (60, y), (540, y + 15), 0, thickness=-1)
    if angle_degrees:
        height, width = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), -angle_degrees, 1.0)
        img = cv2.warpAffine(img, matrix, (width, height), borderValue=255)
    return img


def test_deskew_corrects_a_rotated_page() -> None:
    tilted = _synthetic_text_lines_image(angle_degrees=8.0)
    angle_before = abs(_measured_skew_angle(tilted))

    corrected = deskew(tilted)
    angle_after = abs(_measured_skew_angle(corrected))

    assert angle_after < angle_before
    assert angle_after < 1.0  # near-perfectly level


def test_deskew_leaves_an_already_level_page_alone() -> None:
    level = _synthetic_text_lines_image(angle_degrees=0.0)
    corrected = deskew(level)
    assert abs(_measured_skew_angle(corrected)) < 1.0


def _synthetic_structured_document_image() -> np.ndarray:
    """A level page with a title block, a rule line, and a body paragraph
    of a different width — shaped like the real document that exposed a
    genuine bug in the old deskew algorithm (minAreaRect over all ink
    pixels computed a spurious 8.5° on it, dropping Tesseract's word
    count from 94 to 22 on an already-level page — see
    preprocessing.py::_estimate_skew_angle's docstring and
    docs/phases/phase-2-ocr-pipeline.md for the real numbers).

    Honesty note: this synthetic shape does NOT reproduce that failure —
    manually confirmed the old minAreaRect code returns ~0° on this exact
    image too, so whatever in the real document (diacritic scatter and a
    plain reproduction attempt were both tried and also didn't trigger
    it) actually caused the spurious estimate remains uncharacterized.
    This test is a correctness sanity check for the new Hough-based
    algorithm on a structured layout, not a proven regression test for
    the old bug — the real evidence for that bug is the measured
    before/after numbers on the actual document, not this fixture.
    """
    img = np.full((500, 800), 255, dtype=np.uint8)
    # title: short, centered, thick bar (simulates a large-font heading)
    cv2.rectangle(img, (250, 40), (550, 70), 0, thickness=-1)
    # a horizontal rule line, as many document templates have
    cv2.line(img, (50, 110), (750, 110), 0, thickness=2)
    # body paragraph: full-width lines, much thinner (simulates body text)
    for y in range(150, 450, 35):
        cv2.rectangle(img, (50, y), (750, y + 12), 0, thickness=-1)
    return img


def test_deskew_does_not_introduce_spurious_rotation_on_a_structured_document() -> None:
    """See _synthetic_structured_document_image's docstring: a sanity
    check for the new algorithm, not a proven regression test."""
    level_structured = _synthetic_structured_document_image()
    corrected = deskew(level_structured)
    assert abs(_measured_skew_angle(corrected)) < 1.0


def test_denoise_reduces_variance_of_random_noise() -> None:
    rng = np.random.default_rng(seed=42)
    flat = np.full((200, 200), 200, dtype=np.uint8)
    noisy = flat.astype(np.int16) + rng.integers(-40, 40, size=flat.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    cleaned = denoise(noisy)

    assert float(np.std(cleaned)) < float(np.std(noisy))


def test_enhance_contrast_returns_grayscale_same_shape() -> None:
    color = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = enhance_contrast(color)
    assert result.shape == (100, 100)
    assert result.dtype == np.uint8


def test_full_pipeline_runs_end_to_end_without_error() -> None:
    """The 400x600 synthetic fixture is below upscale_if_small's default
    threshold, so the pipeline both upscales (3x, its max_scale cap) and
    converts to grayscale — output shape is scaled, not identical."""
    tilted_color = cv2.cvtColor(_synthetic_text_lines_image(angle_degrees=5.0), cv2.COLOR_GRAY2BGR)
    result = preprocess(tilted_color)
    assert result.shape == (1200, 1800)


def test_upscale_if_small_scales_up_a_low_resolution_image() -> None:
    small = np.full((400, 600, 3), 200, dtype=np.uint8)
    result = upscale_if_small(small, min_dimension=1600, max_scale=3.0)
    assert result.shape == (1200, 1800, 3)  # capped at max_scale=3.0, not the full 4x to reach 1600


def test_upscale_if_small_leaves_an_already_large_image_untouched() -> None:
    large = np.full((2000, 1700, 3), 200, dtype=np.uint8)
    result = upscale_if_small(large, min_dimension=1600, max_scale=3.0)
    assert result is large


def test_upscale_if_small_preserves_aspect_ratio() -> None:
    small = np.full((300, 900, 3), 200, dtype=np.uint8)  # smaller dimension = height (300)
    result = upscale_if_small(small, min_dimension=1500, max_scale=10.0)
    assert result.shape == (1500, 4500, 3)


@requires_tesseract
@requires_arabic_font
def test_upscale_if_small_real_world_regression_recovers_recognition_on_a_low_res_image() -> None:
    """Regression test for the "improve extraction from images" request:
    an uploaded image can be far lower resolution than a PDF page always
    rendered at 300 DPI. Measured live: a normal 800x144 synthetic
    mixed-script image OCRs cleanly (10/10 words); shrunk to 25% (200x36,
    simulating a low-res upload), Tesseract mangled it into 4 garbage
    tokens ('مرحيابكم', 'مستغرج', ...); running upscale_if_small on the
    shrunk image before OCR fully recovered all 10 words, exact match to
    the full-resolution result.
    """
    import pytesseract

    image = make_text_image(
        [
            ("مرحباً بكم في مستخرج النص الذكي", True),
            ("Smart Text Extractor 2026", False),
        ]
    )
    shrunk = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_AREA)

    def recognized_words(img: np.ndarray) -> list[str]:
        text = pytesseract.image_to_string(img, lang="ara+eng", config="--psm 6")
        return [w for w in text.split() if w.strip()]

    words_without_upscale = recognized_words(shrunk)
    words_with_upscale = recognized_words(upscale_if_small(shrunk, min_dimension=1600, max_scale=3.0))

    assert len(words_without_upscale) <= 5  # confirmed real: mangled into 4 garbage tokens
    assert len(words_with_upscale) == 10
