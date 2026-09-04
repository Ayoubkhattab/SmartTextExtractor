"""Tests for the PDF text-layer transposition repair (ocr/native_text_repair.py).

Fixtures use the real corrupt/correct word pairs measured across this
project's actual PDFs (docs/phases/phase-2-ocr-pipeline.md), not invented
ones — every repair asserted here is one the rule really makes on real
documents.
"""
from __future__ import annotations

from pathlib import Path

from smart_text_extractor.core.models import BoundingBox, Rect
from smart_text_extractor.ocr.native_text_repair import (
    _overlap_ratio,
    is_transposition,
    repair_native_words,
)


def _box(text: str, x: int = 100, y: int = 100, width: int = 80, height: int = 30, confidence: float = 100.0):
    return BoundingBox(text=text, rect=Rect(x=x, y=y, width=width, height=height), confidence=confidence)


class TestIsTransposition:
    def test_real_ligature_corruption_is_detected(self) -> None:
        # Every one of these is a real (native -> OCR) pair measured on the
        # project's own PDFs.
        assert is_transposition("الربمجة", "البرمجة") is True
        assert is_transposition("وإفالته", "وإفلاته") is True
        assert is_transposition("األهداف", "الأهداف") is True
        assert is_transposition("يف", "في") is True
        assert is_transposition("إىل", "إلى") is True
        assert is_transposition("المالك", "الملاك") is True

    def test_ordinary_ocr_misread_is_not_flagged(self) -> None:
        """An OCR error substitutes a different letter, changing the letter
        multiset — the whole basis for telling the two apart. "ميندس" for
        "مهندس" (ه misread as ي) is a real, confirmed misread from this
        project's own OCR output."""
        assert is_transposition("مهندس", "ميندس") is False
        assert is_transposition("الافتراضية", "الفتراضية") is False  # dropped letter, not a reorder

    def test_identical_words_are_not_a_repair(self) -> None:
        assert is_transposition("البرمجة", "البرمجة") is False

    def test_diacritic_placement_difference_is_not_flagged(self) -> None:
        """Real measured pair: both render identically, and the document's
        own diacritics are more trustworthy than OCR's."""
        assert is_transposition("ُتستقبل", "تُستقبل") is False

    def test_empty_text_is_never_a_transposition(self) -> None:
        assert is_transposition("", "") is False
        assert is_transposition("", "في") is False


class TestOverlapRatio:
    def test_small_box_fully_inside_large_box_scores_one(self) -> None:
        """The case IoU got wrong: a PDF word box spans the full font line
        height while Tesseract's hugs the ink, so the same word scores low
        IoU but should still align."""
        native = Rect(x=100, y=100, width=80, height=40)
        ocr_ink = Rect(x=100, y=110, width=80, height=20)
        assert _overlap_ratio(native, ocr_ink) == 1.0

    def test_disjoint_boxes_score_zero(self) -> None:
        assert _overlap_ratio(Rect(0, 0, 10, 10), Rect(500, 500, 10, 10)) == 0.0


class TestRepairNativeWords:
    def test_corrupt_word_is_replaced_by_the_ocr_reading(self) -> None:
        native = [_box("الربمجة")]
        ocr = [_box("البرمجة", confidence=91.0)]

        report = repair_native_words(native, ocr)

        assert report.repair_count == 1
        assert report.repaired[0].text == "البرمجة"
        assert report.replacements == [("الربمجة", "البرمجة")]

    def test_repaired_word_keeps_the_native_position_and_confidence(self) -> None:
        """Position comes from the PDF, which is exact — the OCR box is only
        consulted for what the word says, never for where it sits."""
        native = [_box("يف", x=250, y=300, width=40, height=30)]
        ocr = [_box("في", x=248, y=305, width=38, height=20, confidence=88.0)]

        repaired = repair_native_words(native, ocr).repaired[0]

        assert repaired.rect == Rect(x=250, y=300, width=40, height=30)
        assert repaired.confidence == 100.0

    def test_word_the_ocr_merely_misread_is_left_alone(self) -> None:
        native = [_box("مهندس")]
        ocr = [_box("ميندس", confidence=39.0)]

        report = repair_native_words(native, ocr)

        assert report.repair_count == 0
        assert report.repaired[0].text == "مهندس"

    def test_word_with_no_overlapping_ocr_word_is_left_alone(self) -> None:
        native = [_box("الربمجة", x=100, y=100)]
        ocr = [_box("البرمجة", x=900, y=900)]  # same letters, but nowhere near it

        report = repair_native_words(native, ocr)

        assert report.repair_count == 0
        assert report.repaired[0].text == "الربمجة"

    def test_only_the_corrupt_words_change_others_pass_through_untouched(self) -> None:
        native = [_box("نظام", x=100), _box("الربمجة", x=200), _box("الحديث", x=300)]
        ocr = [_box("نظام", x=100), _box("البرمجة", x=200), _box("الحديث", x=300)]

        report = repair_native_words(native, ocr)

        assert [b.text for b in report.repaired] == ["نظام", "البرمجة", "الحديث"]
        assert report.repair_count == 1

    def test_empty_ocr_leaves_everything_untouched(self) -> None:
        native = [_box("الربمجة")]

        report = repair_native_words(native, [])

        assert report.repair_count == 0
        assert report.repaired[0].text == "الربمجة"


class TestReadingOrder:
    """PyMuPDF returns words in VISUAL (left-to-right) order, which is
    backwards for Arabic — a real bug that made "مقترح هيكلية قسم البرمجة"
    come out as "البرمجة قسم هيكلية مقترح"."""

    def test_arabic_line_is_flipped_into_reading_order(self) -> None:
        from smart_text_extractor.ocr.native_pdf_text import _line_reading_order

        # visual order: leftmost first, as get_text("words") returns them
        visual = [_box("البرمجة", x=100), _box("قسم", x=250), _box("هيكلية", x=350), _box("مقترح", x=500)]

        ordered = _line_reading_order(visual)

        assert [b.text for b in ordered] == ["مقترح", "هيكلية", "قسم", "البرمجة"]

    def test_latin_run_inside_an_arabic_line_keeps_its_own_direction(self) -> None:
        """Without this, "Product Backlog" comes back as "Backlog Product"."""
        from smart_text_extractor.ocr.native_pdf_text import _line_reading_order

        # Enough Arabic that the LINE is majority-Arabic (and so runs
        # right-to-left) while still embedding a two-word Latin phrase.
        visual = [
            _box("Product", x=100),
            _box("Backlog", x=200),
            _box("سجل", x=400),
            _box("من", x=500),
            _box("الخطة", x=600),
            _box("تنطلق", x=700),
        ]

        ordered = _line_reading_order(visual)

        assert [b.text for b in ordered] == ["تنطلق", "الخطة", "من", "سجل", "Product", "Backlog"]

    def test_english_line_is_left_to_right(self) -> None:
        from smart_text_extractor.ocr.native_pdf_text import _line_reading_order

        visual = [_box("Storage", x=100), _box("Path", x=250)]

        assert [b.text for b in _line_reading_order(visual)] == ["Storage", "Path"]

    def test_leading_combining_mark_is_dropped(self) -> None:
        """A mark attaches to the letter before it, so it can never begin a
        word — a token starting with one has it misplaced, and the true
        position is unrecoverable."""
        from smart_text_extractor.ocr.native_pdf_text import _line_reading_order

        ordered = _line_reading_order([_box("\u064fتدار", x=100), _box("وأعمالها", x=300)])

        assert [b.text for b in ordered] == ["وأعمالها", "تدار"]


class TestTextLayerTrustGate:
    def test_sound_token_stream_scores_zero(self) -> None:
        from smart_text_extractor.ocr.native_pdf_text import corrupt_token_ratio

        assert corrupt_token_ratio([_box("نظام"), _box("إدارة"), _box("المستندات")]) == 0.0

    def test_mark_initial_tokens_raise_the_score(self) -> None:
        from smart_text_extractor.ocr.native_pdf_text import corrupt_token_ratio

        boxes = [_box("نظام"), _box("\u064fتدار"), _box("إدارة"), _box("المستندات")]

        assert corrupt_token_ratio(boxes) == 0.25

    def test_empty_input_scores_zero_rather_than_dividing_by_zero(self) -> None:
        from smart_text_extractor.ocr.native_pdf_text import corrupt_token_ratio

        assert corrupt_token_ratio([]) == 0.0

    def test_real_documents_are_classified_as_measured(self) -> None:
        """The gate's whole purpose: keep the two sound documents' text
        layers and reject the damaged one (measured 0.0% / 1.3% / 4.2%)."""
        import pytest

        from smart_text_extractor.core.pdf_import import is_text_layer_trustworthy

        docs = Path("docs")
        if not (docs / "دليل الاستخدام.pdf").exists():
            pytest.skip("real test documents are not present in this checkout")

        assert is_text_layer_trustworthy(docs / "دليل الاستخدام.pdf") is True
        assert is_text_layer_trustworthy(docs / "ODOKAN_UMA_8T10.pdf") is True
        assert is_text_layer_trustworthy(docs / "هيكلية القسم والمكاتب.pdf") is False


class TestFindOrphanOcrWords:
    def test_ocr_word_with_no_native_counterpart_is_returned(self) -> None:
        """Text baked into an image (a screenshot label, a stamp) exists
        only in the pixels — measured: 20 such words in the user manual."""
        from smart_text_extractor.ocr.native_text_repair import find_orphan_ocr_words

        native = [_box("مقدمة", x=100)]
        ocr = [_box("مقدمة", x=100, confidence=90.0), _box("Statistics", x=900, confidence=92.0)]

        assert [b.text for b in find_orphan_ocr_words(native, ocr)] == ["Statistics"]

    def test_low_confidence_and_single_character_orphans_are_ignored(self) -> None:
        from smart_text_extractor.ocr.native_text_repair import find_orphan_ocr_words

        ocr = [_box("Statistics", x=900, confidence=40.0), _box("٠", x=800, confidence=95.0)]

        assert find_orphan_ocr_words([], ocr) == []
