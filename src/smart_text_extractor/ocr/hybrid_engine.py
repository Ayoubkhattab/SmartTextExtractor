"""Hybrid OCR pipeline (docs/phases/phase-2-ocr-pipeline.md): every page
still runs through Tesseract first exactly as before — word boxes, reading
order, table/heading/paragraph classification all come from there, unit
bboxes included (core/models.py DocumentUnit.bbox). Every non-table unit's
source region is then cropped straight from the page and re-recognized by
Qari-OCR (ocr/qari_engine.py), whose output replaces that unit's text —
real, direct comparisons found Qari far more accurate on prose/headings
than Tesseract. Table units are left exactly as Tesseract produced them:
the same comparisons found Qari fabricates content on tables, even a
single table cropped in isolation with nothing else on the page — a
categorically worse failure than a character-level misread for a tool
whose whole purpose is accuracy, so tables never reach Qari at all.

qari_engine=None (Qari unusable on this machine, or deliberately disabled)
makes run() behave exactly like the wrapped OcrEngine alone — Qari is an
enhancement layered on top of the always-available Tesseract pipeline,
never a hard requirement.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from smart_text_extractor.core.models import DocumentUnit, OcrResult, Rect, TextSegment
from smart_text_extractor.ocr.engine import OcrEngine, as_bgr_array
from smart_text_extractor.ocr.preprocessing import preprocess_color
from smart_text_extractor.ocr.qari_engine import QariEngine
from smart_text_extractor.ocr.reorder import document_units_to_markdown, document_units_to_segments

# Pixels of context kept around a unit's word-union bbox when cropping —
# a tight crop right at the glyph edges risks clipping a diacritic or a
# descender that extends slightly past the word boxes it was computed
# from.
_CROP_MARGIN = 12


def _crop_unit_image(color_image_bgr: np.ndarray, bbox: Rect) -> Image.Image:
    height, width = color_image_bgr.shape[:2]
    x0 = max(bbox.x - _CROP_MARGIN, 0)
    y0 = max(bbox.y - _CROP_MARGIN, 0)
    x1 = min(bbox.x + bbox.width + _CROP_MARGIN, width)
    y1 = min(bbox.y + bbox.height + _CROP_MARGIN, height)
    cropped_bgr = color_image_bgr[y0:y1, x0:x1]
    return Image.fromarray(cropped_bgr[:, :, ::-1])  # BGR -> RGB for PIL/Qari


class HybridOcrEngine:
    def __init__(self, tesseract_engine: OcrEngine, qari_engine: QariEngine | None) -> None:
        self._tesseract_engine = tesseract_engine
        self._qari_engine = qari_engine

    def run(self, image: np.ndarray | Image.Image | Path | str, psm: int = 3) -> OcrResult:
        # Same color-preprocessed array both feeds Tesseract (after its own
        # extra grayscale/CLAHE step — see OcrEngine.run_on_color_preprocessed)
        # and, unmodified, becomes the source Qari's crops are taken from —
        # computed once here so it isn't preprocessed twice.
        color_preprocessed = preprocess_color(as_bgr_array(image))
        result = self._tesseract_engine.run_on_color_preprocessed(color_preprocessed, psm=psm)

        if self._qari_engine is None:
            return result

        updated_units: list[DocumentUnit] = [
            self._reread_unit(unit, color_preprocessed) for unit in result.document_units
        ]

        segments = document_units_to_segments(updated_units)
        return OcrResult(
            raw_text="".join(segment.text for segment in segments),
            word_boxes=result.word_boxes,
            segments=segments,
            markdown=document_units_to_markdown(updated_units),
            document_units=updated_units,
            confidence_score=result.confidence_score,
        )

    def _reread_unit(self, unit: DocumentUnit, color_preprocessed: np.ndarray) -> DocumentUnit:
        if unit.kind == "table" or unit.bbox is None:
            return unit

        cropped = _crop_unit_image(color_preprocessed, unit.bbox)
        try:
            qari_text = self._qari_engine.recognize(cropped)
        except Exception:  # noqa: BLE001 - Skip-and-Continue for this one unit: a single bad crop (e.g. a GPU/decode error) must not lose the whole page's Tesseract text
            return unit

        if not qari_text:
            return unit
        return DocumentUnit(kind=unit.kind, segments=[TextSegment(qari_text, None)], rows=[], bbox=unit.bbox)
