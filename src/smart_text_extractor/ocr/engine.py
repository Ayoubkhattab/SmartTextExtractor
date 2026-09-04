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

from smart_text_extractor.core.models import BoundingBox, OcrResult
from smart_text_extractor.ocr.preprocessing import enhance_contrast, preprocess_color
from smart_text_extractor.ocr.reorder import (
    assemble_markdown,
    assemble_text_segments,
    classify_document_units,
    correct_known_arabic_misreads,
    group_into_lines,
    merge_dual_language_passes,
    order_lines_reading_order,
    words_from_tsv,
)


def as_bgr_array(image: np.ndarray | Image.Image | Path | str) -> np.ndarray:
    """Normalizes any of OcrEngine.run()'s accepted input types into the
    BGR numpy array preprocessing.py works on. Public: ocr/hybrid_engine.py
    also needs it, to prepare the same input for preprocess_color()."""
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

    def run(self, image: np.ndarray | Image.Image | Path | str, psm: int = 3, style_index=None) -> OcrResult:
        # §7.1 step 2 — this was previously skipped entirely: run() sent
        # the raw image straight to Tesseract, so deskew/contrast/denoise
        # existed as tested code that nothing ever actually called.
        color_preprocessed = preprocess_color(as_bgr_array(image))
        return self.run_on_color_preprocessed(color_preprocessed, psm=psm, style_index=style_index)

    def run_on_color_preprocessed(self, color_preprocessed: np.ndarray, psm: int = 3, style_index=None) -> OcrResult:
        """The recognition half of run(), taking preprocessing.preprocess_color's
        output directly rather than a raw image — split out so the hybrid
        OCR engine (ocr/hybrid_engine.py) can run this exact Tesseract pass
        once and reuse the same color-preprocessed array afterwards to crop
        Qari-OCR's input regions, instead of preprocessing the page twice
        (once here, once more for the crops) or cropping from run()'s
        grayscale/CLAHE output, which is tuned for Tesseract specifically.
        """
        # psm=3 (fully automatic page segmentation), not 6 (single uniform
        # block): confirmed against a real multi-section document (title,
        # subtitle, headings, highlighted box, bulleted body text at
        # different sizes) that psm=6 badly mis-segments the title/heading
        # regions entirely (garbage output) while psm=3 reads them
        # correctly — see docs/phases/phase-2-ocr-pipeline.md. This does
        # not undo the §7.1.1 multi-column fix: _split_line_into_column_runs
        # operates on Tesseract's line output regardless of which
        # auto-segmentation psm produced it.
        preprocessed = enhance_contrast(color_preprocessed)
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

        tagged_words = correct_known_arabic_misreads(tagged_words)

        if style_index is not None:
            # OCR reads the glyphs; the source file still knows how they
            # look. Applied here, at the single point every word exists in
            # one list, so heading detection, alignment and every renderer
            # downstream see the real sizes and colours (ocr/page_pipeline.py).
            tagged_words = [
                (BoundingBox(box.text, box.rect, box.confidence, style_index.style_for(box.rect)), block, par, line)
                for box, block, par, line in tagged_words
            ]

        lines = group_into_lines(tagged_words)
        ordered_lines = order_lines_reading_order(lines)

        segments = assemble_text_segments(ordered_lines)
        raw_text = "".join(segment.text for segment in segments)
        markdown = assemble_markdown(ordered_lines)
        document_units = classify_document_units(ordered_lines)
        word_boxes = [word for line in ordered_lines for word in line.words]
        confidences = [word.confidence for word in word_boxes]
        confidence_score = sum(confidences) / len(confidences) if confidences else 0.0

        return OcrResult(
            raw_text=raw_text,
            word_boxes=word_boxes,
            segments=segments,
            markdown=markdown,
            document_units=document_units,
            confidence_score=confidence_score,
        )
