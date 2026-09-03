"""Tesseract OCR execution and result assembly (§7.1 steps 3 and 5).

Wraps pytesseract so the rest of the app depends on OcrResult, never on
pytesseract's dict shape directly.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image

from smart_text_extractor.core.models import OcrResult
from smart_text_extractor.ocr.reorder import assemble_text, group_into_lines, order_lines_reading_order, words_from_tsv


class OcrEngine:
    def __init__(
        self,
        lang: str = "ara+eng",
        tesseract_cmd: str | Path | None = None,
        tessdata_dir: str | Path | None = None,
    ) -> None:
        self.lang = lang
        if tesseract_cmd is not None:
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)
        if tessdata_dir is not None:
            os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)

    def run(self, image: np.ndarray | Image.Image | Path | str, psm: int = 6) -> OcrResult:
        data = pytesseract.image_to_data(
            image, lang=self.lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
        )
        tagged_words = words_from_tsv(data)
        lines = group_into_lines(tagged_words)
        ordered_lines = order_lines_reading_order(lines)

        raw_text = assemble_text(ordered_lines)
        word_boxes = [word for line in ordered_lines for word in line.words]
        confidences = [word.confidence for word in word_boxes]
        confidence_score = sum(confidences) / len(confidences) if confidences else 0.0

        return OcrResult(raw_text=raw_text, word_boxes=word_boxes, confidence_score=confidence_score)
