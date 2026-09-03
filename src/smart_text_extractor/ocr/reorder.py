"""Bounding-box reordering (§7.1 step 4, §7.1.1).

Two empirical findings drove this design (see
docs/phases/phase-2-ocr-pipeline.md for the full investigation):

1. Tesseract's own per-line word order is already correct for RTL text
   when given a properly-shaped input image — no python-bidi
   post-processing is needed at word level.
2. Tesseract's layout analysis does NOT reliably separate close-together
   columns into distinct blocks/lines — even with psm 3/4/6, a two-column
   test page with a ~200px gutter got merged into single lines spanning
   both columns, with the LTR column's words placed before the RTL
   column's words regardless of which one should be read first. This
   means column detection cannot trust Tesseract's block_num/line_num
   grouping at all; _split_line_into_column_runs() re-detects real column
   boundaries from raw word x-positions and splits merged lines back
   apart, preserving each run's original (already-correct) word order.

KNOWN LIMITATION: the x-gap heuristics here (both the line-splitting and
the column-clustering that follows it) have only been validated against
one synthetic two-column image with clean, wide gutters and uniform line
lengths (tests/ocr/test_reorder.py) — not against a real scanned
multi-column document, which can have ragged line widths, skewed columns,
or a mix of column counts per page. Treat this as a first cut, not a
settled algorithm; it needs re-validation against real multi-column scans
before Phase 2 can be considered closed on this point.
"""
from __future__ import annotations

from dataclasses import dataclass

from smart_text_extractor.core.models import BoundingBox, Rect

_ARABIC_RANGE = range(0x0600, 0x0700)


@dataclass
class Line:
    words: list[BoundingBox]
    block_num: int
    par_num: int
    line_num: int

    @property
    def rect(self) -> Rect:
        x0 = min(w.rect.x for w in self.words)
        y0 = min(w.rect.y for w in self.words)
        x1 = max(w.rect.x + w.rect.width for w in self.words)
        y1 = max(w.rect.y + w.rect.height for w in self.words)
        return Rect(x=x0, y=y0, width=x1 - x0, height=y1 - y0)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def words_from_tsv(data: dict) -> list[tuple[BoundingBox, int, int, int]]:
    """Extract (word, block_num, par_num, line_num) from pytesseract's
    image_to_data(..., output_type=Output.DICT) result, dropping empty /
    non-text rows (conf == -1 marks block/par/line-level summary rows)."""
    results: list[tuple[BoundingBox, int, int, int]] = []
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        conf = float(data["conf"][i])
        if conf < 0:
            continue
        box = BoundingBox(
            text=text,
            rect=Rect(x=data["left"][i], y=data["top"][i], width=data["width"][i], height=data["height"][i]),
            confidence=conf,
        )
        results.append((box, data["block_num"][i], data["par_num"][i], data["line_num"][i]))
    return results


def group_into_lines(words: list[tuple[BoundingBox, int, int, int]]) -> list[Line]:
    """Groups by (block_num, par_num, line_num), preserving TSV order within
    each line — that order is Tesseract's own reading-order determination,
    which the empirical test confirms is correct even for RTL text."""
    groups: dict[tuple[int, int, int], list[BoundingBox]] = {}
    order: list[tuple[int, int, int]] = []
    for box, block_num, par_num, line_num in words:
        key = (block_num, par_num, line_num)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(box)
    return [Line(words=groups[k], block_num=k[0], par_num=k[1], line_num=k[2]) for k in order]


def _is_majority_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    arabic_count = sum(1 for c in letters if ord(c) in _ARABIC_RANGE)
    return arabic_count / len(letters) > 0.5


def _split_line_into_column_runs(line: Line, gap_multiplier: float = 3.0) -> list[list[BoundingBox]]:
    """A single Tesseract "line" can actually span two visually distinct
    columns: empirically (docs/phases/phase-2-ocr-pipeline.md), Tesseract's
    own layout analysis merges same-row multi-column text into one line
    when the column gap isn't large relative to the page, even with
    psm 3/4/6. Detect a wide gap between consecutive words (by x-position)
    and split there — into runs that each preserve the ORIGINAL TSV word
    order internally, so a correctly-recognized RTL run is not re-sorted
    and broken.
    """
    words = line.words
    if len(words) <= 1:
        return [words]

    avg_word_width = sum(w.rect.width for w in words) / len(words)
    gap_threshold = max(avg_word_width * gap_multiplier, 60)

    by_x = sorted(words, key=lambda w: w.rect.x)
    boundaries = [
        (a.rect.x + a.rect.width + b.rect.x) / 2
        for a, b in zip(by_x, by_x[1:])
        if (b.rect.x - (a.rect.x + a.rect.width)) > gap_threshold
    ]
    if not boundaries:
        return [words]

    boundaries.sort()
    runs: list[list[BoundingBox]] = [[] for _ in range(len(boundaries) + 1)]
    for word in words:  # preserve original TSV order within each run
        center = word.rect.x + word.rect.width / 2
        run_index = sum(1 for boundary in boundaries if center > boundary)
        runs[run_index].append(word)
    return [run for run in runs if run]


def order_lines_reading_order(lines: list[Line], column_gap_ratio: float = 0.08) -> list[Line]:
    """Column-aware reading order: split any line that actually spans
    multiple columns, cluster the resulting lines into columns by
    x-position, order columns right-to-left for a majority-Arabic page
    (else left-to-right), and keep lines within each column top-to-bottom.
    """
    if not lines:
        return []

    split_lines = [
        Line(words=run, block_num=line.block_num, par_num=line.par_num, line_num=line.line_num)
        for line in lines
        for run in _split_line_into_column_runs(line)
    ]

    rtl_page = _is_majority_arabic(" ".join(line.text for line in split_lines))

    sorted_by_x = sorted(split_lines, key=lambda line: line.rect.x)
    leftmost = min(line.rect.x for line in split_lines)
    rightmost = max(line.rect.x + line.rect.width for line in split_lines)
    gap_threshold = max((rightmost - leftmost) * column_gap_ratio, 20)

    columns: list[list[Line]] = [[sorted_by_x[0]]]
    current_max_x = sorted_by_x[0].rect.x + sorted_by_x[0].rect.width
    for line in sorted_by_x[1:]:
        if line.rect.x - current_max_x > gap_threshold:
            columns.append([line])
        else:
            columns[-1].append(line)
        current_max_x = max(current_max_x, line.rect.x + line.rect.width)

    columns.sort(key=lambda col: min(line.rect.x for line in col), reverse=rtl_page)

    ordered: list[Line] = []
    for col in columns:
        ordered.extend(sorted(col, key=lambda line: line.rect.y))
    return ordered


def assemble_text(lines: list[Line]) -> str:
    return "\n".join(line.text for line in lines)
