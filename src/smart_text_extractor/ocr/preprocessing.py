"""Pre-processing pipeline (§7.1): Deskew, Binarization/Contrast, Noise Removal.

Operates on numpy arrays (BGR or grayscale) so callers control I/O — this
module has no opinion about file formats.
"""
from __future__ import annotations

import cv2
import numpy as np


def denoise(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE (adaptive histogram equalization) — helps low-contrast/aged
    documents (the "د. أحمد" persona's old scanned papers, §7.1) without
    blowing out already-good scans the way global equalization would.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _estimate_skew_angle(image: np.ndarray) -> float:
    """Hough-line-based skew estimate, not minAreaRect-over-all-ink-pixels.

    The original implementation took the minAreaRect of every foreground
    pixel on the page. That works for a single uniform block of text
    lines (all our early synthetic tests) but breaks on a real structured
    document — confirmed live: a clean, perfectly level, digitally
    rendered page (title, subtitle, headings, a highlighted box, bulleted
    body text at different sizes) produced a spurious 8.5° estimate,
    because the *bounding shape* of scattered ink across such a layout
    doesn't track the actual text-baseline angle. Applying that rotation
    to an already-level page dropped Tesseract's recognized word count
    from 94 to 22 on that document (see
    docs/phases/phase-2-ocr-pipeline.md for the full trace).

    Hough line detection instead looks for actual near-horizontal line
    segments — dominated on a real page by text baselines and rule
    lines, both of which are horizontal when the page is level — and
    takes their median angle (robust to the handful of outliers a
    heterogeneous layout produces). Confirmed against three cases: the
    real document above (0.0° now, was 8.5°), a synthetic image tilted
    8° (detects 8.01°), and a level synthetic image (0.0°).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line.ravel()
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(angle) < 20:  # real scan skew is small; discard near-vertical noise
            angles.append(angle)

    return float(np.median(angles)) if angles else 0.0


def deskew(image: np.ndarray) -> np.ndarray:
    """Corrects page skew via a Hough-line-based angle estimate."""
    angle = _estimate_skew_angle(image)
    if angle == 0.0:
        return image

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess(image: np.ndarray) -> np.ndarray:
    """The full §7.1 pipeline: deskew -> denoise -> contrast.

    Deskew runs FIRST, not last — confirmed real bug: CLAHE contrast
    enhancement measurably degrades edge characteristics that
    _estimate_skew_angle's Hough line detection depends on. On a
    deliberately degraded (skewed + noisy + low-contrast) test image,
    running deskew after denoise/contrast made _estimate_skew_angle
    return 0.0 (its "no lines found" fallback) instead of the correct
    ~-6°, silently leaving the real skew completely uncorrected — same
    broken outcome as if deskew had never run at all. Running deskew
    first, on the least-processed edges available, fixed it: the same
    image went from garbled/fragmented OCR output to an exact match.
    """
    return enhance_contrast(denoise(deskew(image)))
