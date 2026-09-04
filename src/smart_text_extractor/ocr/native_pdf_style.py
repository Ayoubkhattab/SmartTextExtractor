"""Reads how a PDF page's text LOOKS, not just what it says (§7.1.1 extension).

get_text("words") — what native_pdf_text.py builds on — gives position and
text and throws away everything visual. get_text("dict") keeps the font,
size, weight and colour of each span, and get_drawings() reports the filled
shapes that sit behind text (a highlighted line, a coloured callout box).
This module reads both and answers one question per word: what does it look
like?

The two are joined by position: a word's box sits inside the span that
produced it, and inside any shape drawn behind it. That is also why this
works at all for words whose text is unusable — the join never looks at
characters, so it is unaffected by the transposition corruption
ocr/native_text_repair.py deals with separately.

Only a PDF text layer carries any of this. OCR pages get TextStyle-free
segments and renderers fall back to their existing size heuristics.
"""
from __future__ import annotations

from dataclasses import dataclass

import pymupdf

from smart_text_extractor.core.models import Rect, TextStyle

_BOLD_FLAG = 1 << 4
_ITALIC_FLAG = 1 << 1

# A fill this close to white is the page itself, not a highlight — every
# document has a white or near-white background rectangle, and treating it
# as a highlight would paint the whole page.
_WHITE_THRESHOLD = 0.93

# A highlight has to actually sit behind the text rather than merely touch
# it: a table rule or an underline overlaps a word's box slightly, a real
# highlight covers most of it.
_MIN_HIGHLIGHT_COVERAGE = 0.5


@dataclass(frozen=True)
class _StyledRegion:
    rect: Rect
    style: TextStyle


def _to_hex(color: int) -> str:
    return f"#{color & 0xFFFFFF:06x}"


def _fill_to_hex(fill) -> str | None:
    if fill is None:
        return None
    red, green, blue = fill[0], fill[1], fill[2]
    if red >= _WHITE_THRESHOLD and green >= _WHITE_THRESHOLD and blue >= _WHITE_THRESHOLD:
        return None
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def _scaled(bbox, points_to_pixels: float) -> Rect:
    return Rect(
        x=round(bbox[0] * points_to_pixels),
        y=round(bbox[1] * points_to_pixels),
        width=round((bbox[2] - bbox[0]) * points_to_pixels),
        height=round((bbox[3] - bbox[1]) * points_to_pixels),
    )


def _contains_center(outer: Rect, inner: Rect) -> bool:
    center_x = inner.x + inner.width / 2
    center_y = inner.y + inner.height / 2
    return outer.x <= center_x <= outer.x + outer.width and outer.y <= center_y <= outer.y + outer.height


def _coverage(shape: Rect, word: Rect) -> float:
    x0, y0 = max(shape.x, word.x), max(shape.y, word.y)
    x1 = min(shape.x + shape.width, word.x + word.width)
    y1 = min(shape.y + shape.height, word.y + word.height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    word_area = word.width * word.height
    return ((x1 - x0) * (y1 - y0)) / word_area if word_area else 0.0


MIN_CONTAINER_SIZE_POINTS = 40
"""Smallest filled shape treated as a container box rather than a
highlight behind a word or a rule line."""

MAX_CONTAINER_PAGE_FRACTION = 0.8
"""A shape covering more of the page than this is the page's own
background, not a box on it — every one of this project's documents draws
a full-page white rectangle first."""


class PageStyleIndex:
    """Answers style_for(word_rect) for one page.

    Built once per page and queried per word, so the span/shape lists are
    walked once rather than re-read for every lookup.
    """

    def __init__(self, page: pymupdf.Page, render_dpi: int) -> None:
        points_to_pixels = render_dpi / 72
        self._spans: list[_StyledRegion] = []
        self._highlights: list[tuple[Rect, str]] = []

        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", ()):
                for span in line["spans"]:
                    self._spans.append(
                        _StyledRegion(
                            rect=_scaled(span["bbox"], points_to_pixels),
                            style=TextStyle(
                                font_size=span["size"],
                                bold=bool(span["flags"] & _BOLD_FLAG),
                                italic=bool(span["flags"] & _ITALIC_FLAG),
                                color=_to_hex(span["color"]),
                            ),
                        )
                    )

        self.container_boxes: list[tuple[Rect, str]] = []
        page_area = max(page.rect.width * page.rect.height, 1)

        for drawing in page.get_drawings():
            fill = _fill_to_hex(drawing.get("fill"))
            if fill is None:
                continue
            rect = drawing["rect"]
            scaled = _scaled(rect, points_to_pixels)
            self._highlights.append((scaled, fill))

            # A large filled shape is a box the page draws content inside —
            # the callout panels a document lays its sidebar content in —
            # rather than a mark behind a word. Reproducing those is what
            # makes an output page look like its source instead of a single
            # flat column of text.
            covers_page = (rect.width * rect.height) / page_area > MAX_CONTAINER_PAGE_FRACTION
            big_enough = rect.width >= MIN_CONTAINER_SIZE_POINTS and rect.height >= MIN_CONTAINER_SIZE_POINTS
            if big_enough and not covers_page:
                self.container_boxes.append((scaled, fill))

    def style_for(self, word_rect: Rect) -> TextStyle | None:
        span_style: TextStyle | None = None
        for span in self._spans:
            if _contains_center(span.rect, word_rect):
                span_style = span.style
                break

        highlight: str | None = None
        for shape_rect, fill in self._highlights:
            if _coverage(shape_rect, word_rect) >= _MIN_HIGHLIGHT_COVERAGE:
                highlight = fill
                break

        if span_style is None:
            return TextStyle(highlight=highlight) if highlight else None
        if highlight is None:
            return span_style
        return TextStyle(
            font_size=span_style.font_size,
            bold=span_style.bold,
            italic=span_style.italic,
            color=span_style.color,
            highlight=highlight,
        )
