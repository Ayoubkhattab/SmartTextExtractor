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

from smart_text_extractor.core.models import BoundingBox, Rect, TextSegment

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


_BIDI_CONTROL_CHARS = ("‎", "‏")  # LRM, RLM


def words_from_tsv(data: dict) -> list[tuple[BoundingBox, int, int, int]]:
    """Extract (word, block_num, par_num, line_num) from pytesseract's
    image_to_data(..., output_type=Output.DICT) result, dropping empty /
    non-text rows (conf == -1 marks block/par/line-level summary rows).

    Also strips stray bidi control marks (LRM U+200E, RLM U+200F) that
    Tesseract's own bidi reordering leaves embedded in word text — real
    examples seen on every mixed-script page tested this session:
    'Jods‏', '٠‏', 'Smart‎'. These are zero-width, carry no
    visible glyph or semantic content, and their only effect is cluttering
    raw_text and confusing char-based analysis (e.g. a lone stray mark
    could otherwise slip through as a 1-character "word"). Stripped here,
    at the single point every word enters the pipeline, rather than only
    at final text assembly, so every downstream step (majority-Arabic
    detection, Latin detection, gap measurement) sees the clean text too.
    """
    results: list[tuple[BoundingBox, int, int, int]] = []
    for i, text in enumerate(data["text"]):
        for mark in _BIDI_CONTROL_CHARS:
            text = text.replace(mark, "")
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


_LATIN_RANGE_MAX = 0x024F  # rough upper bound of Latin script + extensions


def _is_mostly_latin(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    latin_count = sum(1 for c in letters if ord(c) <= _LATIN_RANGE_MAX)
    return latin_count / len(letters) > 0.5


def _rect_iou(a: Rect, b: Rect) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.width, b.x + b.width), min(a.y + a.height, b.y + b.height)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def merge_dual_language_passes(
    primary_words: list[tuple[BoundingBox, int, int, int]],
    arabic_only_words: list[tuple[BoundingBox, int, int, int]],
) -> list[tuple[BoundingBox, int, int, int]]:
    """Corrects a real, confirmed Tesseract failure mode (see
    docs/phases/phase-2-ocr-pipeline.md): running lang="ara+eng" together
    sometimes misclassifies isolated Arabic letter-shapes as Latin
    garbage — confirmed on a real document: "من" -> "OR", "صريح" -> "Fro",
    "مبنياً" -> "Liisa", "الساعة" -> "deludl" — while genuine English runs
    ("Software Department — Structure, Operating Model & Staffing Plan",
    "Agile") are read correctly by the same pass.

    A parallel ara-only pass gets those specific misread Arabic words
    right. For each primary (ara+eng) word that looks mostly-Latin, this
    looks for the spatially overlapping word in the ara-only pass and
    prefers it — but only if that alternative is mostly-Arabic (not
    digit/symbol soup — a genuine English run's ara-only alternative is
    unreadable garbage too, e.g. "Software" -> "5011100121", so it never
    passes this check) AND has strictly higher confidence than the
    primary word.

    That confidence comparison was added after real data showed the
    "mostly Arabic" check alone isn't sufficient: on the same real page,
    "Plan" (primary confidence 96) had an ara-only alternative "مقا"
    (confidence 52) that WAS mostly-Arabic and got wrongly substituted
    in, corrupting a correctly-read English word. Every genuine
    correction observed (e.g. "Fro" @18 -> "صريح" @27, "Liisa" @51 ->
    "مبنياً" @57) had the alternative's confidence exceed the primary's;
    the one bad substitution ("Plan" @96 -> "مقا" @52) did not. Requiring
    a strict improvement fixes the real errors and leaves "Plan" alone.
    """
    merged: list[tuple[BoundingBox, int, int, int]] = []
    for box, block, par, line in primary_words:
        if _is_mostly_latin(box.text):
            best_match: BoundingBox | None = None
            best_iou = 0.0
            for a_box, _a_block, _a_par, _a_line in arabic_only_words:
                iou = _rect_iou(box.rect, a_box.rect)
                if iou > best_iou:
                    best_iou = iou
                    best_match = a_box
            if (
                best_match is not None
                and best_iou > 0.3
                and _is_majority_arabic(best_match.text)
                and best_match.confidence > box.confidence
            ):
                merged.append((best_match, block, par, line))
                continue
        merged.append((box, block, par, line))
    return merged


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


def _line_segments_with_cell_separators(
    line: Line, gap_multiplier: float = 4.0, min_gap_threshold: float = 40.0, separator: str = " | "
) -> list[TextSegment]:
    """Renders one already-ordered row as TextSegments, inserting
    `separator` (confidence None — it's punctuation, not a recognized
    word) at table-cell boundaries instead of a plain space — addresses a
    real, evidence-confirmed complaint (user report +
    docs/phases/phase-2-ocr-pipeline.md): table rows were unreadable
    because every cell got mashed together with a single space,
    indistinguishable from the words within a cell.

    Evidence (real table, page 3 of هيكلية القسم والمكاتب.pdf, Tesseract
    TSV for one row — "تخطيط الدورة | بداية كل دورة ٠ ساعتان | المكاتب
    الخمسة + رئيس القسم | سجل الدورة مع تقديرات..."): within-cell word
    gaps measured 9-18px; the 3 real cell-boundary gaps measured 94px,
    101px, and 173px — roughly 7-12x the smallest within-line gap. A
    normal prose line from the same page (block 2, no table) had gaps
    clustered at 10-18px with NO outliers at all. This gives a reliable,
    self-calibrating signal per line: flag a gap as a cell boundary only
    if it's both far bigger than that line's own tightest word spacing
    (gap_multiplier) AND bigger than a small absolute floor
    (min_gap_threshold, guards short lines/near-zero gaps).

    Deliberately does NOT reuse _split_line_into_column_runs: that
    function re-sorts words by x because it targets a different bug
    (Tesseract merging two separate visual columns into one line, in the
    wrong order). Here the row's word order is already correct (per the
    module-level empirical finding), so gaps are measured directly on
    `line.words` in their existing order — re-sorting is unnecessary and
    would risk misreading which side of a gap is "first" for RTL rows.
    """
    words = line.words
    if len(words) <= 1:
        return [TextSegment(w.text, w.confidence) for w in words]

    gaps = [
        max(a.rect.x, b.rect.x) - min(a.rect.x + a.rect.width, b.rect.x + b.rect.width)
        for a, b in zip(words, words[1:])
    ]
    gap_threshold = max(min(gaps) * gap_multiplier, min_gap_threshold)

    segments = [TextSegment(words[0].text, words[0].confidence)]
    for word, gap in zip(words[1:], gaps):
        segments.append(TextSegment(separator if gap > gap_threshold else " ", None))
        segments.append(TextSegment(word.text, word.confidence))
    return segments


def _line_text_with_cell_separators(
    line: Line, gap_multiplier: float = 4.0, min_gap_threshold: float = 40.0, separator: str = " | "
) -> str:
    segments = _line_segments_with_cell_separators(line, gap_multiplier, min_gap_threshold, separator)
    return "".join(segment.text for segment in segments)


def _cluster_columns_and_order(lines: list[Line], column_gap_ratio: float, rtl_page: bool) -> list[Line]:
    """Column-cluster a set of lines that are already known to belong to
    one coherent region (see order_lines_reading_order), order columns
    right-to-left for a majority-Arabic page (else left-to-right), and
    keep lines within each column top-to-bottom.

    rtl_page is computed ONCE for the whole page and passed in, not
    recomputed per block/region — confirmed real bug: a small block can
    have a near-even word split (e.g. two rows of a 2-column diagram, one
    English line + one Arabic line each) whose *own* majority swings
    either way by coincidence, giving different blocks inconsistent
    reading directions on the same page. The page-level computation, over
    far more text, is far more reliable — and a document has one reading
    direction, not a different one per section.
    """
    if not lines:
        return []

    sorted_by_x = sorted(lines, key=lambda line: line.rect.x)
    leftmost = min(line.rect.x for line in lines)
    rightmost = max(line.rect.x + line.rect.width for line in lines)
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


def order_lines_reading_order(lines: list[Line], column_gap_ratio: float = 0.08) -> list[Line]:
    """Column-aware reading order: split any line that actually spans
    multiple columns, then cluster into columns and order right-to-left
    (Arabic) or left-to-right, keeping lines within a column top-to-bottom.

    Column-clustering is scoped to one Tesseract block at a time, not the
    whole page — confirmed real bug (docs/phases/phase-2-ocr-pipeline.md):
    doing it globally scrambled reading order on a page mixing full-width
    paragraphs with a 3-box side-by-side diagram, a data table, and
    colored side-by-side panels, because a full-width paragraph line's
    x-range overlaps every narrower box/column below it, so they all got
    lumped into "one column" together. Tesseract's own block segmentation
    (psm 3) already separates these regions correctly — verified: each
    row of a 3-box diagram became its own block — so it's trusted as the
    region boundary. Blocks are then stacked by their own top position,
    not assumed to already be in top-to-bottom order.
    """
    if not lines:
        return []

    split_lines = [
        Line(words=run, block_num=line.block_num, par_num=line.par_num, line_num=line.line_num)
        for line in lines
        for run in _split_line_into_column_runs(line)
    ]

    rtl_page = _is_majority_arabic(" ".join(line.text for line in split_lines))

    blocks: dict[int, list[Line]] = {}
    for line in split_lines:
        blocks.setdefault(line.block_num, []).append(line)

    block_order = sorted(blocks, key=lambda block_num: min(line.rect.y for line in blocks[block_num]))

    ordered: list[Line] = []
    for block_num in block_order:
        ordered.extend(_cluster_columns_and_order(blocks[block_num], column_gap_ratio, rtl_page))
    return ordered


def assemble_text_segments(lines: list[Line]) -> list[TextSegment]:
    """Joins lines within the same Tesseract block with a single newline,
    and separates blocks (paragraphs, table regions, diagram boxes...)
    with a blank line — addresses a real complaint that extracted text
    read as one undifferentiated wall of text with no paragraph structure.
    Relies on order_lines_reading_order's block_order pass already having
    made same-block lines contiguous, so a block_num change is a reliable
    paragraph boundary without needing to re-group here.

    Returns TextSegments (word segments carry confidence; the newlines/
    blank-lines/space/" | " separators inserted here and by
    _line_segments_with_cell_separators all carry confidence None) rather
    than a plain string — concatenating every segment's text reproduces
    exactly what assemble_text returns, but callers that need per-word
    confidence (e.g. highlighting uncertain words in the UI) can use the
    structure instead of re-deriving it from raw_text.
    """
    if not lines:
        return []

    segments: list[TextSegment] = []
    current_block = lines[0].block_num
    first_in_block = True
    for line in lines:
        if line.block_num != current_block:
            segments.append(TextSegment("\n\n", None))
            current_block = line.block_num
            first_in_block = True
        elif not first_in_block:
            segments.append(TextSegment("\n", None))
        segments.extend(_line_segments_with_cell_separators(line))
        first_in_block = False

    return segments


def assemble_text(lines: list[Line]) -> str:
    return "".join(segment.text for segment in assemble_text_segments(lines))


# Markdown export (§7.1.1 extension): a real, drawn table's header row and
# a document heading measure structurally identical on a real page — both
# are an isolated single-line Tesseract block, taller than surrounding
# body text (confirmed real, page 3 of هيكلية القسم والمكاتب.pdf: the
# genuine section heading "الإطار المنيجي (Agile) — ثانياً" measured
# 74px vs a page-median line height of 38px — a 1.95x ratio — while that
# SAME page's real table header row ("التواتر والمدة المُخْرَج
# المُلزِم") measured 60px — a 1.58x ratio: taller than body text too,
# but well short of the heading. _HEADING_HEIGHT_RATIO=1.75 sits in the
# gap between those two real, measured ratios, so it catches the genuine
# heading without also flagging the table header as one. This is a
# precision-first, deliberately conservative threshold: on that same
# page it also does NOT flag the page's own title (39px, ratio 1.03 —
# apparently bolded rather than enlarged in the source) or the
# "Definition of Done/Ready" box labels (30px — smaller than body text,
# emphasized by color/box rather than size) as headings. Those are real,
# accepted misses, not bugs: there is no font-weight or color signal in
# Tesseract's TSV output to catch them with, and a threshold loose enough
# to catch them would also catch the table header — see the ratios above.
_HEADING_HEIGHT_RATIO = 1.75


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[mid - 1] + ordered[mid]) / 2
    return ordered[mid]


def _line_median_height(line: Line) -> float:
    return _median([word.rect.height for word in line.words])


def _line_cells(line: Line, gap_multiplier: float = 4.0, min_gap_threshold: float = 40.0) -> list[str]:
    """Splits one line into per-cell strings using the same gap-based
    boundary detection as _line_segments_with_cell_separators — a line
    with no detected boundary comes back as a single-element list (its
    whole text), which callers use as the "not actually multi-cell"
    signal.
    """
    segments = _line_segments_with_cell_separators(line, gap_multiplier, min_gap_threshold)
    cells: list[str] = [""]
    for segment in segments:
        if segment.text == " | ":
            cells.append("")
        else:
            cells[-1] += segment.text
    return [cell.strip() for cell in cells]


def _block_is_tabular(block_lines: list[Line]) -> bool:
    """A block reads as a table when most of its rows actually split into
    cells — a majority rather than every row, because real, messy tables
    (confirmed on the same real table: its last couple of rows) don't
    always get a clean cell split on every single row (a known limitation
    of the gap-based boundary heuristic — see
    docs/phases/phase-2-ocr-pipeline.md); requiring 100% would silently
    demote an otherwise-obvious table to plain paragraph text over one or
    two ragged rows.
    """
    if len(block_lines) < 2:
        return False
    rows_with_a_cell_boundary = sum(1 for line in block_lines if len(_line_cells(line)) > 1)
    return rows_with_a_cell_boundary / len(block_lines) > 0.5


def _rows_to_markdown_table(rows: list[list[str]]) -> str:
    """Ragged rows (a real, messy table can have a different cell count
    per row — see _block_is_tabular) are padded to the widest row with
    empty cells rather than truncated, so no recognized text is dropped
    just to keep the grid rectangular."""
    column_count = max(len(row) for row in rows)
    padded_rows = [row + [""] * (column_count - len(row)) for row in rows]
    header, *data_rows = padded_rows
    table_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * column_count) + " |"]
    table_lines.extend("| " + " | ".join(row) + " |" for row in data_rows)
    return "\n".join(table_lines)


def _render_plain_block(block_lines: list[Line], page_median_height: float) -> list[str]:
    """Renders one non-tabular block as one or more top-level Markdown
    units: consecutive body-height lines join into a single paragraph
    (newline-separated, matching assemble_text), but a heading-height
    line is broken out as its own '## ...' unit so it gets the blank-line
    spacing Markdown headings need, instead of being folded into a
    paragraph's interior line — confirmed real: on the page this was
    calibrated against, the genuine heading is the second line of a
    2-line block (title line, then heading line), not alone in its own
    block, so heading detection has to work at the line level, not just
    flag single-line blocks.
    """
    units: list[str] = []
    paragraph_lines: list[str] = []
    for line in block_lines:
        if _line_median_height(line) > page_median_height * _HEADING_HEIGHT_RATIO:
            if paragraph_lines:
                units.append("\n".join(paragraph_lines))
                paragraph_lines = []
            units.append(f"## {line.text}")
        else:
            paragraph_lines.append(_line_text_with_cell_separators(line))
    if paragraph_lines:
        units.append("\n".join(paragraph_lines))
    return units


def assemble_markdown(lines: list[Line]) -> str:
    """Structured Markdown export: real tables become Markdown tables and
    a real heading becomes '## ...', instead of everything flattening
    into plain paragraphs. Builds entirely on already-validated pieces
    (the same per-block grouping and cell-boundary detection assemble_text
    uses) rather than new signal — deliberately does NOT attempt real
    table-gridline detection from the image: tried it during this
    session's investigation and found it unreliable on real documents
    (a naive OpenCV morphological line detector mostly picked up dense
    text strokes, not drawn borders — see
    docs/phases/phase-2-ocr-pipeline.md), so this reuses the
    already-proven word-gap heuristic instead.
    """
    if not lines:
        return ""

    page_median_height = _median([word.rect.height for line in lines for word in line.words])

    blocks: list[list[Line]] = [[]]
    current_block_num = lines[0].block_num
    for line in lines:
        if line.block_num != current_block_num:
            blocks.append([])
            current_block_num = line.block_num
        blocks[-1].append(line)

    rendered_units: list[str] = []
    i = 0
    while i < len(blocks):
        block_lines = blocks[i]
        is_isolated_tabular_line = len(block_lines) == 1 and len(_line_cells(block_lines[0])) > 1
        next_block_is_a_table = i + 1 < len(blocks) and _block_is_tabular(blocks[i + 1])

        if is_isolated_tabular_line and next_block_is_a_table:
            # A single cell-separated line immediately before a table
            # block reads as that table's header row, split into its own
            # Tesseract block — real tables can have a visually distinct
            # header (e.g. a colored bar) that segments separately from
            # the data rows below it. Fold it in as the header instead of
            # letting the table below fall back to using its own first
            # data row as a stand-in header.
            rows = [_line_cells(block_lines[0])] + [_line_cells(line) for line in blocks[i + 1]]
            rendered_units.append(_rows_to_markdown_table(rows))
            i += 2
            continue

        if _block_is_tabular(block_lines):
            rendered_units.append(_rows_to_markdown_table([_line_cells(line) for line in block_lines]))
        else:
            rendered_units.extend(_render_plain_block(block_lines, page_median_height))
        i += 1

    return "\n\n".join(rendered_units)
