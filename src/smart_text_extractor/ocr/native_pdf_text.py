"""Native PDF text-layer extraction (§7.1 extension): a PDF exported
from an authoring tool (Word, InDesign, a web page's "print to PDF", ...)
already contains real, embedded text — reading it directly is instant
and perfectly accurate, unlike rendering the page to an image and OCR'ing
it. A genuinely scanned PDF (a photographed/scanned paper document with
no embedded text) still needs the OCR pipeline; this module's job is
telling the two apart, and building the finished result directly when it
can.

Session-defining real finding (docs/phases/phase-2-ocr-pipeline.md): all
3 real test documents used throughout this project's OCR debugging
turned out to already have a full native text layer. Every character-
level OCR error chased this session (tessdata tuning, dual-pass merge,
known-misread correction, ...) was never present in the source text at
all — it was introduced by rendering that text to pixels and reading it
back with Tesseract. Reusing ocr/reorder.py's Line/reading-order/table/
heading machinery on the PDF's own words (instead of Tesseract OCR
words) gets all of that structure-aware handling for free, since it
already operates on generic (text, position, confidence) data, not
anything Tesseract-specific — verified end to end on the real,
previously-hardest table page: correct heading detection, correct
row-major table order, and (unlike OCR) zero character-level errors,
since there is no recognition step at all to make one.

KNOWN LIMITATION: a page that mixes native text with an embedded scanned
graphic (e.g. a digitally-typed cover memo with a scanned signature or
stamp image containing its own text) will silently lose whatever text is
inside that graphic — get_text("words") only ever sees the real text
runs, never pixels. Not something any of this project's real test
documents exercise, so left undetected rather than guessed at.
"""
from __future__ import annotations

import pymupdf

from smart_text_extractor.core.models import BoundingBox, DocumentUnit, OcrResult, PageLayout, Rect
from smart_text_extractor.ocr.native_pdf_style import PageStyleIndex
from smart_text_extractor.ocr.native_text_repair import find_orphan_ocr_words, repair_native_words
from smart_text_extractor.ocr.reorder import (
    _is_majority_arabic,
    _is_mostly_latin,
    assemble_markdown,
    assemble_text_segments,
    classify_document_units,
    group_into_lines,
    order_lines_reading_order,
)

MIN_WORDS_FOR_NATIVE_TEXT = 10
"""Below this word count, treat the page as scanned (no usable text
layer) rather than native. Guards against a false positive on a
genuinely scanned page that happens to carry a few words of embedded
metadata/watermark text — real pages checked this session had 95-364
words, so this is a conservative, clearly-differentiating floor, not a
value tuned against edge cases we don't have real examples of."""

MAX_CORRUPT_TOKEN_RATIO = 0.025
"""Above this fraction of visibly-corrupt tokens, a PDF's text layer is not
trusted at all and its pages fall back to OCR.

This is the verification mechanism the earlier native-text rollback said
was missing (docs/phases/phase-2-ocr-pipeline.md): the embedded text was
switched off wholesale because there was no way to tell a sound text layer
from a damaged one. There is now — a token starting with an Arabic
combining mark is unambiguously wrong, since a mark attaches to the letter
before it and can never begin a word, so the rate of those tokens measures
how scrambled the character stream is.

Measured WHOLE-DOCUMENT, not per page, because the damage comes from the
file's font/generator and so is a property of the document: 0.0%
(ODOKAN_UMA_8T10), 1.3% (دليل الاستخدام), 4.2% (هيكلية القسم والمكاتب) —
a clean separation. Per page the same measure overlaps and misclassifies
(a sound document's page at 2.73% would be rejected while a damaged one's
page at 2.65% would be accepted), which is what moved the decision up to
document level.

The first two documents are sound and their embedded text is far more
accurate than OCR of the same pages. The third is damaged well beyond the
transposition repair — its words are also split mid-word at ordinary
word-space distances and its diacritics are displaced, and attempts to
reconstruct it lost characters outright. Being fitted to three documents
this is a starting point, not a settled constant — but a measured one, and
the failure mode is safe: rejecting a sound layer only falls back to the
OCR path this project used exclusively before."""


def corrupt_token_ratio(boxes: list[BoundingBox]) -> float:
    """Fraction of tokens that begin with a combining mark — see
    MAX_CORRUPT_TOKEN_RATIO for why that measures text-layer damage."""
    if not boxes:
        return 0.0
    corrupt = sum(1 for box in boxes if box.text and box.text[0] in _COMBINING_MARKS)
    return corrupt / len(boxes)


def page_layout_of(page: pymupdf.Page) -> PageLayout:
    """The page's real size and the margins its text actually observes.

    Size comes straight from the page. Margins are measured from the text
    rather than from any page-box metadata, because it is the text's own
    extent that decides how wide a line may be — and reproducing that width
    is what stops an export from re-flowing every line somewhere else.

    A page whose text starts unusually far down (a title page) would give a
    misleadingly large top margin, so the top and bottom fall back to
    matching the horizontal margins, which are stable: vertical position
    within the page is reproduced by each unit's own space_before_points
    instead, where it belongs.
    """
    rect = page.rect
    words = page.get_text("words")
    if not words:
        default = min(rect.width, rect.height) * 0.1
        return PageLayout(rect.width, rect.height, default, default, default, default)

    left = min(word[0] for word in words)
    right = rect.width - max(word[2] for word in words)
    vertical = min(left, right)
    return PageLayout(
        width_points=rect.width,
        height_points=rect.height,
        margin_left=left,
        margin_right=right,
        margin_top=vertical,
        margin_bottom=vertical,
    )


def _apply_vertical_rhythm(units: list[DocumentUnit], points_to_pixels: float) -> None:
    """Records how much blank space sat above each unit on the page.

    Without this every block is separated by the exporter's own uniform
    paragraph gap, which is why an export of a page with real vertical
    structure (a title block, then a wide gap, then the body) comes out
    evenly spaced and needs manual re-spacing. The gap is stored in points
    so it survives whatever DPI the page was rendered at.

    Only clearly deliberate space is recorded: a gap no bigger than the
    unit's own line height is ordinary line spacing, not a break.
    """
    previous_bottom: float | None = None
    for unit in units:
        if unit.bbox is None:
            continue
        if previous_bottom is not None:
            gap_pixels = unit.bbox.y - previous_bottom
            if gap_pixels > unit.bbox.height:
                unit.space_before_points = round(gap_pixels / points_to_pixels, 1)
        previous_bottom = unit.bbox.y + unit.bbox.height


def _line_reading_order(boxes: list[BoundingBox]) -> list[BoundingBox]:
    """Puts one PDF line's words into reading order.

    Real bug this fixes: get_text("words") returns words in VISUAL order
    (left to right across the page), which for an Arabic line is exactly
    backwards — "مقترح هيكلية قسم البرمجة" came out as "البرمجة قسم
    هيكلية مقترح". ocr/reorder.py's machinery deliberately preserves the
    word order it is given, because Tesseract's order is already correct
    for RTL (see that module's docstring); PyMuPDF's is not, so the order
    has to be established here instead of relying on that guarantee.

    An embedded Latin run inside an Arabic line reads left-to-right within
    itself even though the line runs right-to-left, so consecutive Latin
    words are re-reversed after the line is flipped — without that,
    "Product Backlog" comes back as "Backlog Product".
    """
    if not boxes:
        return boxes
    if not _is_majority_arabic(" ".join(box.text for box in boxes)):
        return sorted(boxes, key=lambda box: box.rect.x)

    right_to_left = sorted(boxes, key=lambda box: -box.rect.x)

    ordered: list[BoundingBox] = []
    latin_run: list[BoundingBox] = []
    for box in right_to_left:
        if _is_mostly_latin(box.text):
            latin_run.append(box)
            continue
        ordered.extend(reversed(latin_run))
        latin_run = []
        ordered.append(box)
    ordered.extend(reversed(latin_run))
    return _merge_mark_initial_fragments(ordered)


_COMBINING_MARKS = frozenset("ًٌٍَُِّْٰ")

_FRAGMENT_MAX_GAP_RATIO = 0.08
"""Fraction of a word's height that two boxes may be apart and still count
as one word the PDF split in two. Expressed relative to height so it holds
at any render DPI. Measured over 246 real mark-initial pairs: ordinary
word spacing clusters at 0.13-0.20 of line height (p25 0.131, median
0.157, p75 0.196), so this sits below that cluster rather than inside it —
an earlier 0.15 cut landed mid-cluster and fused separate words."""


def _horizontal_gap(a: Rect, b: Rect) -> float:
    return max(a.x, b.x) - min(a.x + a.width, b.x + b.width)


def _merge_mark_initial_fragments(boxes: list[BoundingBox]) -> list[BoundingBox]:
    """Re-joins words this PDF split mid-word.

    Real, measured artifact: some PDFs emit a single word as several text
    runs broken around its diacritics, so get_text("words") hands back
    fragments — "محدَّدة" arrives as "،مح" + "َّددة", and the extracted text
    reads "مح َّددة". Measured prevalence: 0 fragments in one of this
    project's documents, 11 of 844 words in another, 149 of 3550 in the
    third.

    A token beginning with a combining mark is always wrong: a mark
    attaches to the letter before it and can never start a word. Its true
    position inside the word is not recoverable, so the stray leading mark
    is dropped — "تدار" instead of "ُتدار". Arabic reads correctly without
    optional diacritics, so this costs nothing but a pronunciation hint,
    and it never changes a letter.

    Rejoining such a token to the one before it was tried instead and
    removed: it duplicated characters ("موجّه" came out "موّجّجه") on files
    whose text layer already contains overlapping duplicate runs, and —
    when gated on box adjacency to avoid that — fused genuinely separate
    words whose spacing fell inside the same range.
    """
    cleaned: list[BoundingBox] = []
    for box in boxes:
        if not box.text or box.text[0] not in _COMBINING_MARKS:
            cleaned.append(box)
            continue
        stripped = box.text.lstrip("".join(_COMBINING_MARKS))
        if stripped:
            cleaned.append(BoundingBox(text=stripped, rect=box.rect, confidence=box.confidence, style=box.style))
    return cleaned


def _union_of(a: Rect, b: Rect) -> Rect:
    x0, y0 = min(a.x, b.x), min(a.y, b.y)
    x1 = max(a.x + a.width, b.x + b.width)
    y1 = max(a.y + a.height, b.y + b.height)
    return Rect(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def extract_native_text_result(
    page: pymupdf.Page, render_dpi: int, ocr_word_boxes: list[BoundingBox] | None = None
) -> OcrResult | None:
    """Returns a fully-formed OcrResult built directly from the PDF
    page's own embedded text, or None if the page doesn't have a usable
    text layer — callers fall back to the OCR pipeline in that case.

    ocr_word_boxes, when given, is an OCR pass of this same page used to
    repair the text layer's letter-transposition corruption
    (ocr/native_text_repair.py) — measured present in ALL THREE of this
    project's real PDFs (0.4% / 3.4% / 8.9% of words), so passing it is
    strongly recommended rather than optional-in-practice. Without it the
    embedded text is returned exactly as the PDF stores it, corruption
    included.

    render_dpi must match the DPI the caller renders this same page's
    preview image at (pdf_import.py's import_pdf_pages passes its own
    dpi through here) — PDF word coordinates are in points (1/72in), and
    scaling them by render_dpi/72 instead of an arbitrary factor puts
    them in the same numeric space as that rendered image's pixels.
    Every absolute-pixel threshold in ocr/reorder.py (cell-boundary
    gaps, column-gap floors, ...) was calibrated against that space, so
    this is what makes those thresholds behave consistently whether a
    word came from Tesseract or straight from the PDF — and it's also
    what keeps word_boxes aligned to the actual preview image pixels,
    should a future feature (e.g. the planned searchable-PDF export)
    need to highlight a word on it.
    """
    words = page.get_text("words")  # (x0, y0, x1, y1, text, block_no, line_no, word_no)
    if len(words) < MIN_WORDS_FOR_NATIVE_TEXT:
        return None

    points_to_pixels = render_dpi / 72
    # How each word LOOKS — size, weight, colour, and any shape drawn
    # behind it. Only a text layer carries this at all (ocr/native_pdf_style.py).
    style_index = PageStyleIndex(page, render_dpi)

    tagged = []
    for x0, y0, x1, y1, text, block_no, line_no, _word_no in words:
        rect = Rect(
            x=round(x0 * points_to_pixels),
            y=round(y0 * points_to_pixels),
            width=round((x1 - x0) * points_to_pixels),
            height=round((y1 - y0) * points_to_pixels),
        )
        tagged.append(
            (
                BoundingBox(
                    text=text,
                    rect=rect,
                    confidence=100.0,  # not a recognition guess — this is the document's own real text
                    style=style_index.style_for(rect),
                ),
                block_no,
                0,
                line_no,
            )
        )

    # Reading order first, before anything downstream consumes the order
    # (repair alignment is positional so it is unaffected, but line/segment
    # assembly is not) — see _line_reading_order.
    grouped: dict[tuple[int, int], list[BoundingBox]] = {}
    key_order: list[tuple[int, int]] = []
    for box, block_no, _par_no, line_no in tagged:
        key = (block_no, line_no)
        if key not in grouped:
            grouped[key] = []
            key_order.append(key)
        grouped[key].append(box)

    tagged = [
        (box, block_no, 0, line_no)
        for (block_no, line_no) in key_order
        for box in _line_reading_order(grouped[(block_no, line_no)])
    ]

    if ocr_word_boxes:
        native_boxes = [box for box, *_ in tagged]
        report = repair_native_words(native_boxes, ocr_word_boxes)
        tagged = [
            (repaired_box, block_no, par_no, line_no)
            for repaired_box, (_, block_no, par_no, line_no) in zip(report.repaired, tagged)
        ]

        # Text the page renders as pixels (inside a screenshot, a stamp, a
        # logo) never appears in get_text("words") at all — this module's
        # documented blind spot. Recovered from the OCR pass and appended
        # in blocks numbered past every native block, so
        # order_lines_reading_order still places them by their real
        # position on the page rather than by an id that would collide
        # with the PDF's own block numbering.
        next_block_num = max((block_no for _, block_no, _, _ in tagged), default=-1) + 1
        for offset, orphan in enumerate(find_orphan_ocr_words(native_boxes, ocr_word_boxes)):
            tagged.append((orphan, next_block_num + offset, 0, 0))

    lines = group_into_lines(tagged)
    ordered_lines = order_lines_reading_order(lines)
    segments = assemble_text_segments(ordered_lines)
    raw_text = "".join(segment.text for segment in segments)

    document_units = classify_document_units(ordered_lines)
    _apply_vertical_rhythm(document_units, points_to_pixels)

    return OcrResult(
        raw_text=raw_text,
        word_boxes=[box for box, *_ in tagged],
        segments=segments,
        markdown=assemble_markdown(ordered_lines),
        document_units=document_units,
        confidence_score=100.0,
        page_layout=page_layout_of(page),
    )
