"""Repairs the letter-transposition corruption found in real PDF text
layers, using an OCR pass of the same page as the corrective source.

THE BUG (measured, docs/phases/phase-2-ocr-pipeline.md): every one of this
project's three real test PDFs stores some words with their letters in the
wrong order — overwhelmingly a lam-alef ligature written back as
alef-lam ("وإفلاته" -> "وإفالته", "الأهداف" -> "األهداف"), plus other
adjacent-letter swaps ("البرمجة" -> "الربمجة", "المقترح" -> "المقرتح").
Measured incidence: 2.4% / 4.8% / 14.7% of spatially-alignable words
across the three documents. An earlier investigation concluded this
affected only one document; re-measuring the pattern itself (rather than
searching for specific known-bad words) showed it is in all three.

WHY OCR IS THE RIGHT CORRECTIVE SOURCE: OCR reads the rendered glyphs,
which are visually correct — the corruption lives only in the font's
character mapping, never in what the page actually shows. Confirmed on
every inspected instance: the PDF's own text is wrong and Tesseract's
reading of the same word is right.

THE SIGNATURE: the corruption REORDERS a word's letters, so the corrupt
word is an exact anagram of the correct one. An OCR misread instead
SUBSTITUTES letters (ه->ي, خ->ح), changing the letter multiset. That
asymmetry is what makes the two mechanically separable, and it is why
this repair does not need a dictionary, a model, or a quality score.

DELIBERATELY NARROW: only an exact anagram (same letter multiset,
different order) triggers a repair.
  - Words differing by an added/dropped letter are NOT repaired, even
    though some are real corruption (measured: 32 of 314 hits in the
    worst document). Dropping a letter is also a common OCR failure, so
    taking OCR there could delete a letter from a word the PDF had right
    — a real risk, avoided at the cost of leaving a minority of
    corruption unfixed.
  - Words differing only in diacritic placement ("تُستقبل" vs "ُتستقبل")
    are NOT repaired: both render identically, and OCR's diacritics are
    less reliable than the document's own.
"""
from __future__ import annotations

from dataclasses import dataclass

from smart_text_extractor.core.models import BoundingBox, Rect

MIN_OVERLAP_RATIO = 0.5
"""Overlap needed before a native word and an OCR word are treated as the
same word on the page — measured as intersection over the SMALLER of the
two boxes, not IoU.

IoU was tried first and aligned far too few words: a PDF word box spans
the font's full line height (ascender to descender) while Tesseract's box
hugs the actual glyph ink, so the same word on the same spot can score
well under 0.5 IoU purely from that height difference. Measured with IoU,
ordinary words plainly present in both sources ("مقدمة", "الذين") came
back "unaligned". Intersection-over-smaller is insensitive to that size
mismatch while still requiring the boxes to genuinely sit on top of each
other.

Loosening this is safe here specifically because the anagram test below
does the real work: two unrelated words would have to contain exactly the
same letters to trigger a repair, which effectively never happens by
accident."""

# Arabic combining marks (fatha/damma/kasra/shadda/sukun/tanween families
# plus the superscript alef) — stripped before comparison so a difference
# in where a mark landed is never mistaken for a letter transposition.
_DIACRITICS = frozenset("ًٌٍَُِّْٰ")


def _base_letters(text: str) -> str:
    return "".join(character for character in text if character not in _DIACRITICS)


def _overlap_ratio(a: Rect, b: Rect) -> float:
    """Intersection area over the smaller box's area — see MIN_OVERLAP_RATIO."""
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1 = min(a.x + a.width, b.x + b.width)
    y1 = min(a.y + a.height, b.y + b.height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    smaller = min(a.width * a.height, b.width * b.height)
    return intersection / smaller if smaller > 0 else 0.0


def is_transposition(native_text: str, ocr_text: str) -> bool:
    """True when the two words contain exactly the same letters in a
    different order — the corruption's signature (see module docstring)."""
    native_base = _base_letters(native_text)
    ocr_base = _base_letters(ocr_text)
    if native_base == ocr_base or not native_base:
        return False
    return sorted(native_base) == sorted(ocr_base)


ORPHAN_MIN_CONFIDENCE = 70.0
"""Confidence an OCR word must clear before it is treated as real text the
PDF's own text layer simply doesn't contain (see find_orphan_ocr_words)."""

ORPHAN_MIN_LENGTH = 2
"""Single characters are dropped from orphan recovery: measured, the
orphans below this length were page-decoration noise ('٠', '.', '#'),
never content, while the genuine finds were whole words."""


def find_orphan_ocr_words(
    native_boxes: list[BoundingBox],
    ocr_boxes: list[BoundingBox],
    min_confidence: float = ORPHAN_MIN_CONFIDENCE,
) -> list[BoundingBox]:
    """OCR words sitting where the PDF's text layer has nothing at all —
    i.e. text rendered as pixels rather than as text.

    This closes native extraction's one documented blind spot
    (native_pdf_text.py's KNOWN LIMITATION): text baked into an embedded
    image is invisible to get_text("words"). Measured on the real
    documents: 20 such words in the user manual, every one of them a label
    inside an app screenshot ("Statistics", "Correspondents", "Invoice
    #2026-001") — real content that native-only extraction would silently
    drop; 0 in one other document and 5 punctuation/digit fragments in the
    third, which the confidence and length floors discard.
    """
    orphans: list[BoundingBox] = []
    for ocr_box in ocr_boxes:
        if ocr_box.confidence < min_confidence or len(ocr_box.text.strip()) < ORPHAN_MIN_LENGTH:
            continue
        if any(_overlap_ratio(ocr_box.rect, native_box.rect) > MIN_OVERLAP_RATIO for native_box in native_boxes):
            continue
        orphans.append(ocr_box)
    return orphans


@dataclass
class RepairReport:
    repaired: list[BoundingBox]
    repair_count: int
    replacements: list[tuple[str, str]]  # (native_text, ocr_text) actually applied


def repair_native_words(native_boxes: list[BoundingBox], ocr_boxes: list[BoundingBox]) -> RepairReport:
    """Returns the native words with transposition-corrupted ones replaced
    by the OCR reading of the same position on the page.

    Native words with no confidently-overlapping OCR word are left exactly
    as they are: an unaligned word carries no evidence either way, and the
    PDF's own text is the better default everywhere the corruption
    signature does not fire.
    """
    repaired: list[BoundingBox] = []
    replacements: list[tuple[str, str]] = []

    for native_box in native_boxes:
        best_match: BoundingBox | None = None
        best_overlap = MIN_OVERLAP_RATIO
        for ocr_box in ocr_boxes:
            overlap = _overlap_ratio(native_box.rect, ocr_box.rect)
            if overlap > best_overlap:
                best_overlap, best_match = overlap, ocr_box

        if best_match is not None and is_transposition(native_box.text, best_match.text):
            repaired.append(
                BoundingBox(text=best_match.text, rect=native_box.rect, confidence=native_box.confidence)
            )
            replacements.append((native_box.text, best_match.text))
        else:
            repaired.append(native_box)

    return RepairReport(repaired=repaired, repair_count=len(replacements), replacements=replacements)
