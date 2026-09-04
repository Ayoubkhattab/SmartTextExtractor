"""Tesseract OCR execution and result assembly (§7.1 steps 2, 3, and 5).

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
from smart_text_extractor.ocr.preprocessing import preprocess
from smart_text_extractor.ocr.reorder import (
    assemble_markdown,
    assemble_text_segments,
    group_into_lines,
    merge_dual_language_passes,
    order_lines_reading_order,
    words_from_tsv,
)


def _as_bgr_array(image: np.ndarray | Image.Image | Path | str) -> np.ndarray:
    """Normalizes any of OcrEngine.run()'s accepted input types into the
    BGR numpy array preprocessing.py works on."""
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = Image.open(image)
    return np.array(pil_image.convert("RGB"))[:, :, ::-1]


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

    def run(self, image: np.ndarray | Image.Image | Path | str, psm: int = 3) -> OcrResult:
        # psm=3 (fully automatic page segmentation), not 6 (single uniform
        # block): confirmed against a real multi-section document (title,
        # subtitle, headings, highlighted box, bulleted body text at
        # different sizes) that psm=6 badly mis-segments the title/heading
        # regions entirely (garbage output) while psm=3 reads them
        # correctly — see docs/phases/phase-2-ocr-pipeline.md. This does
        # not undo the §7.1.1 multi-column fix: _split_line_into_column_runs
        # operates on Tesseract's line output regardless of which
        # auto-segmentation psm produced it.
        # §7.1 step 2 — this was previously skipped entirely: run() sent
        # the raw image straight to Tesseract, so deskew/contrast/denoise
        # existed as tested code that nothing ever actually called.
        preprocessed = preprocess(_as_bgr_array(image))
        data = pytesseract.image_to_data(
            preprocessed, lang=self.lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
        )
        tagged_words = words_from_tsv(data)

        if self.lang == "ara+eng":
            # §7.1.1 extension — confirmed real: ara+eng sometimes
            # misclassifies isolated Arabic words as Latin garbage. A
            # second ara-only pass gets those specific words right, and
            # merge_dual_language_passes' "must be mostly Arabic to
            # substitute" guard keeps it from touching genuine English
            # runs, whose ara-only alternative is unreadable garbage too.
            # This doubles OCR time for mixed-language pages — a real
            # cost, accepted because it fixes a confirmed accuracy bug.
            arabic_only_data = pytesseract.image_to_data(
                preprocessed, lang="ara", config=f"--psm {psm}", output_type=pytesseract.Output.DICT
            )
            arabic_only_words = words_from_tsv(arabic_only_data)
            tagged_words = merge_dual_language_passes(tagged_words, arabic_only_words)

        lines = group_into_lines(tagged_words)
        ordered_lines = order_lines_reading_order(lines)

        segments = assemble_text_segments(ordered_lines)
        raw_text = "".join(segment.text for segment in segments)
        markdown = assemble_markdown(ordered_lines)
        word_boxes = [word for line in ordered_lines for word in line.words]
        confidences = [word.confidence for word in word_boxes]
        confidence_score = sum(confidences) / len(confidences) if confidences else 0.0

        return OcrResult(
            raw_text=raw_text,
            word_boxes=word_boxes,
            segments=segments,
            markdown=markdown,
            confidence_score=confidence_score,
        )
