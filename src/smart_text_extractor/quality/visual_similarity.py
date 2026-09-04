"""Measures how much an extracted page LOOKS like the page it came from.

Text accuracy has been measurable in this project for a long time; visual
similarity never was, even though "the output should resemble the original
page" is the requirement that actually drives the work.

WHY NOT PIXEL OVERLAP. The obvious metric — how much ink lands in the same
place — was built first and calibrated against controls, which killed it:
a page scored against a COMPLETELY DIFFERENT page of the same document got
19-24%, while a faithful reconstruction of the page itself got 8-21%. The
reconstruction scored no better than an unrelated page, so the number
carried no signal at all. The reason is that ink overlap on a text page
mostly measures "is there text here", which is true of any text page,
while a reconstruction that cannot use the source's own font puts its
strokes a pixel or two off and loses the intersection anyway.

WHAT IS MEASURED INSTEAD. Layout is where content sits, not what its
glyphs look like, so the page is reduced to its ink PROFILES:

  vertical   — ink per row down the page. Captures the block rhythm: where
               text starts, the gaps between sections, how tall each band
               of content is.
  horizontal — ink per column across the page. This is what sees columns:
               a two-column page has two humps, a single-column page one,
               and a page whose sidebar was flattened into the flow loses
               a hump.

Each profile is compared by correlation, which is invariant to how dark or
dense the ink is — so a different font, or text rendered slightly heavier,
does not move the score, while a block in the wrong place does.

Calibration is part of the module's contract, not an afterthought: see
tests/quality/test_visual_similarity.py, which asserts the ordering that
must hold — a page against itself scores far above a faithful
reconstruction, which scores far above an unrelated page.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROFILE_RESOLUTION = 256
"""Buckets each ink profile is resampled to before comparison. Independent
of page size, so pages of different dimensions stay comparable."""

VERTICAL_SMOOTHING = 25
"""Smoothing for the down-the-page profile, in buckets.

Calibrated, not chosen: unsmoothed, a faithful reconstruction scored BELOW
an unrelated page, because the profile is spiky and a text line landing two
pixels off loses its correlation entirely. Widening the window recovers the
ordering (48.6 -> 74.4 for a faithful rebuild) while a reconstruction of
the WRONG page stays near 19 — it forgives placement noise without
forgiving real layout differences."""

HORIZONTAL_SMOOTHING = 5
"""Smoothing for the across-the-page profile — deliberately much narrower.

The two axes measure phenomena at different scales, and using one window
for both broke the metric's main job: at 25 buckets the gutter between two
columns is smoothed away, and a page whose columns had been collapsed into
a single flow still scored 92%. The gutter is the signal here, so it has
to survive."""

INK_THRESHOLD = 245
"""Below this grey level a pixel counts as ink. Permissive on purpose: a
pale panel fill is content too, and a page whose panels vanished should
score worse for it."""


@dataclass(frozen=True)
class VisualScore:
    vertical: float  # agreement on where content sits down the page
    horizontal: float  # agreement on column structure across the page
    ink_ratio: float  # ink in the output relative to the source, as a diagnostic

    @property
    def overlap(self) -> float:
        """The headline number: how well the two layouts agree.

        Deliberately NOT penalised by ink_ratio. That was tried and
        measured wrong: a reconstruction cannot use the source's own font,
        so it legitimately carries a fraction of the ink (0.14x on a real
        page) while placing it correctly, and the penalty buried a faithful
        rebuild below an unrelated page. ink_ratio is still reported,
        because a collapse in it does mean content was lost — it just
        cannot be folded into a layout score."""
        return (self.vertical + self.horizontal) / 2

    @property
    def percent(self) -> float:
        return self.overlap * 100


def _ink_image(image_path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not read image: {image_path}")
    return (image < INK_THRESHOLD).astype(np.float32)


def _profile(ink: np.ndarray, axis: int, smoothing: int) -> np.ndarray:
    """Ink per row (axis=1) or per column (axis=0), resampled to a fixed
    length so pages of any size compare directly."""
    raw = ink.sum(axis=axis).astype(np.float32)
    if raw.size == 0:
        return np.zeros(PROFILE_RESOLUTION, dtype=np.float32)
    positions = np.linspace(0, raw.size - 1, PROFILE_RESOLUTION)
    resampled = np.interp(positions, np.arange(raw.size), raw)
    smoothed = np.convolve(resampled, np.ones(smoothing) / smoothing, mode="same")
    # Normalised so the comparison is about WHERE the ink is, not how heavy
    # the font that drew it happened to be.
    peak = smoothed.max()
    return smoothed / peak if peak > 0 else smoothed


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation clamped to 0..1 — a negative correlation and no
    correlation are both simply "does not match"."""
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 1.0 if a.std() < 1e-6 and b.std() < 1e-6 else 0.0
    return float(max(0.0, np.corrcoef(a, b)[0, 1]))


def compare_pages(source_image: Path, rebuilt_image: Path) -> VisualScore:
    source_ink = _ink_image(Path(source_image))
    rebuilt_ink = _ink_image(Path(rebuilt_image))

    source_total = float(source_ink.sum())
    rebuilt_total = float(rebuilt_ink.sum())

    return VisualScore(
        vertical=_correlation(
            _profile(source_ink, axis=1, smoothing=VERTICAL_SMOOTHING),
            _profile(rebuilt_ink, axis=1, smoothing=VERTICAL_SMOOTHING),
        ),
        horizontal=_correlation(
            _profile(source_ink, axis=0, smoothing=HORIZONTAL_SMOOTHING),
            _profile(rebuilt_ink, axis=0, smoothing=HORIZONTAL_SMOOTHING),
        ),
        ink_ratio=(rebuilt_total / source_total) if source_total else 0.0,
    )
