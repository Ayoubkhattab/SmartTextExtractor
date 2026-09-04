"""Pure-logic tests for the hybrid OCR pipeline (ocr/hybrid_engine.py):
fake Tesseract-engine/Qari-engine stubs exercise the crop/replace/merge
and fallback logic without needing a real Tesseract binary, a GPU, or the
actual Qari model checkpoint.

Test images are 1700x1700 (matches test_engine.py's convention) —
comfortably above preprocessing.upscale_if_small's 1600px trigger, so
HybridOcrEngine's real preprocess_color() call (deliberately not mocked:
the crop-source array's exact shape is part of what's under test) leaves
dimensions unchanged and crop-boundary math stays predictable.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from smart_text_extractor.core.models import DocumentUnit, OcrResult, Rect, TextSegment
from smart_text_extractor.ocr.hybrid_engine import HybridOcrEngine


def _color_image() -> np.ndarray:
    return np.full((1700, 1700, 3), 255, dtype=np.uint8)


class _FakeTesseractEngine:
    """Stands in for OcrEngine — HybridOcrEngine.run() only ever calls
    run_on_color_preprocessed."""

    def __init__(self, result: OcrResult) -> None:
        self.result = result

    def run_on_color_preprocessed(self, color_preprocessed: np.ndarray, psm: int = 3) -> OcrResult:
        return self.result


class _FakeQariEngine:
    def __init__(self, text: str | None = "QARI_TEXT") -> None:
        self.text = text
        self.recognized_images: list[Image.Image] = []

    def recognize(self, image: Image.Image) -> str:
        self.recognized_images.append(image)
        return self.text


def _paragraph_unit(bbox: Rect, text: str = "TESSERACT_TEXT") -> DocumentUnit:
    return DocumentUnit(kind="paragraph", segments=[TextSegment(text, 91.0)], bbox=bbox)


def _heading_unit(bbox: Rect, text: str = "TESSERACT_HEADING") -> DocumentUnit:
    return DocumentUnit(kind="heading", segments=[TextSegment(text, 91.0)], bbox=bbox)


def _table_unit(bbox: Rect) -> DocumentUnit:
    return DocumentUnit(kind="table", rows=[[[TextSegment("CELL", 91.0)]]], bbox=bbox)


def test_qari_none_returns_tesseract_result_unchanged() -> None:
    tesseract_result = OcrResult(raw_text="tesseract only", document_units=[_paragraph_unit(Rect(0, 0, 50, 20))])
    tesseract = _FakeTesseractEngine(tesseract_result)
    engine = HybridOcrEngine(tesseract, qari_engine=None)

    result = engine.run(_color_image())

    assert result is tesseract_result


def test_paragraph_unit_is_replaced_with_qari_text() -> None:
    paragraph = _paragraph_unit(Rect(10, 10, 50, 20))
    tesseract = _FakeTesseractEngine(OcrResult(document_units=[paragraph]))
    qari = _FakeQariEngine(text="QARI_TEXT")
    engine = HybridOcrEngine(tesseract, qari)

    result = engine.run(_color_image())

    assert len(result.document_units) == 1
    updated = result.document_units[0]
    assert updated.kind == "paragraph"
    assert updated.segments == [TextSegment("QARI_TEXT", None)]
    assert updated.bbox == paragraph.bbox  # position is preserved even though the text was replaced
    assert "QARI_TEXT" in result.raw_text
    assert "QARI_TEXT" in result.markdown
    assert len(qari.recognized_images) == 1


def test_heading_unit_is_also_sent_to_qari() -> None:
    heading = _heading_unit(Rect(0, 0, 100, 40))
    tesseract = _FakeTesseractEngine(OcrResult(document_units=[heading]))
    qari = _FakeQariEngine(text="QARI_HEADING")
    engine = HybridOcrEngine(tesseract, qari)

    result = engine.run(_color_image())

    assert result.document_units[0].kind == "heading"
    assert result.document_units[0].segments == [TextSegment("QARI_HEADING", None)]


def test_table_unit_is_never_sent_to_qari() -> None:
    """The core safety property this whole module exists for
    (docs/phases/phase-2-ocr-pipeline.md): direct experiments found Qari
    fabricates content on tables, even a single table cropped in isolation
    with nothing else on the page — so a table unit's Tesseract content
    must be preserved exactly, and recognize() must never even be called
    with one.
    """
    table = _table_unit(Rect(0, 0, 200, 100))
    tesseract = _FakeTesseractEngine(OcrResult(document_units=[table]))
    qari = _FakeQariEngine(text="SHOULD_NEVER_APPEAR")
    engine = HybridOcrEngine(tesseract, qari)

    result = engine.run(_color_image())

    assert result.document_units[0] is table
    assert qari.recognized_images == []
    assert "SHOULD_NEVER_APPEAR" not in result.raw_text


def test_unit_without_bbox_is_left_untouched() -> None:
    unit = DocumentUnit(kind="paragraph", segments=[TextSegment("NO_BBOX", 91.0)], bbox=None)
    tesseract = _FakeTesseractEngine(OcrResult(document_units=[unit]))
    qari = _FakeQariEngine(text="SHOULD_NOT_BE_USED")
    engine = HybridOcrEngine(tesseract, qari)

    result = engine.run(_color_image())

    assert result.document_units[0] is unit
    assert qari.recognized_images == []


def test_qari_failure_on_one_unit_falls_back_to_tesseract_text_for_that_unit_only() -> None:
    """Skip-and-Continue at the unit level, mirroring OcrWorkerPool's
    existing page-level Skip-and-Continue: one crop's failure (a GPU
    error, a decode error) must not lose the whole page's already-good
    Tesseract text for every other unit."""
    good_unit = _paragraph_unit(Rect(0, 0, 50, 20), text="GOOD")
    second_unit = _paragraph_unit(Rect(0, 30, 50, 20), text="FALLBACK_TEXT")
    tesseract = _FakeTesseractEngine(OcrResult(document_units=[good_unit, second_unit]))

    class _PartialFailQari:
        def __init__(self) -> None:
            self.calls = 0

        def recognize(self, image: Image.Image) -> str:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated failure on the second crop")
            return "QARI_GOOD"

    engine = HybridOcrEngine(tesseract, _PartialFailQari())

    result = engine.run(_color_image())

    assert result.document_units[0].segments == [TextSegment("QARI_GOOD", None)]
    assert result.document_units[1] is second_unit  # untouched, Tesseract's own


def test_empty_qari_text_falls_back_to_tesseract_unit() -> None:
    unit = _paragraph_unit(Rect(0, 0, 50, 20), text="TESSERACT_FALLBACK")
    tesseract = _FakeTesseractEngine(OcrResult(document_units=[unit]))
    qari = _FakeQariEngine(text="")
    engine = HybridOcrEngine(tesseract, qari)

    result = engine.run(_color_image())

    assert result.document_units[0] is unit


def test_crop_is_taken_from_the_unit_bbox_with_margin_clamped_to_image_bounds() -> None:
    """A bbox flush with the image edge (a real, common case — a heading
    or paragraph often starts near x=0/y=0) must not crop out of bounds or
    crash when the crop margin is added."""
    image = _color_image()  # 1700x1700
    unit = _paragraph_unit(Rect(0, 0, 1700, 1700))  # fills the whole page
    tesseract = _FakeTesseractEngine(OcrResult(document_units=[unit]))
    qari = _FakeQariEngine(text="OK")
    engine = HybridOcrEngine(tesseract, qari)

    engine.run(image)

    cropped = qari.recognized_images[0]
    # the +12px margin would overflow past every edge — clamped to the
    # image's own bounds instead of going negative/out-of-bounds.
    assert cropped.size == (1700, 1700)
