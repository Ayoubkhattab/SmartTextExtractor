"""Qari-OCR (NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct, a Qwen2-VL
fine-tune) inference for the hybrid pipeline's prose/heading regions.

docs/phases/phase-2-ocr-pipeline.md documents the real, direct comparisons
behind this module's scope: Qari reads Arabic prose and headings far more
accurately than Tesseract, but on a data table — even a single table
cropped in isolation, with nothing else on the page — it degrades past
recognizable errors into fabricated content (invented sentences, an
invented English name list, an invented "published by" attribution none of
which exist in the source document). That is a categorically worse failure
than Tesseract's character-level misreads for a tool whose whole purpose
is accuracy, so this module is only ever asked to recognize the specific
prose/heading crops ocr/hybrid_engine.py sends it — never a table region.

Heavy, GPU-dependent, optional dependency group (pyproject.toml
[project.optional-dependencies] "vlm"): imported lazily inside __init__
so the rest of the app works with torch/transformers/qwen-vl-utils absent
entirely. Construction raises QariUnavailableError for every "can't be
used on this machine" case (missing packages, no CUDA GPU, missing/broken
model files) — ocr/hybrid_engine.py's caller catches that once at startup
and falls back to Tesseract-only, per the user's explicit choice that Qari
is an enhancement layered on top of the always-available Tesseract
pipeline, never a hard requirement to open the app at all.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

from PIL import Image

# Keeps VRAM use in check on an 8GB GPU — matches the setting validated
# across this session's real test runs (scratchpad qari_eval/run_qari.py).
_MAX_IMAGE_DIMENSION = 1600

# A cropped single prose paragraph or heading, not a full dense page — the
# 4000-token budget used during whole-page table experiments was sized for
# a much larger, structured target and isn't needed here.
_MAX_NEW_TOKENS = 2000

_PROMPT = (
    "Below is the image of one region of a document page (a heading or a "
    "paragraph of text). Just return the plain text representation of "
    "this region as if you were reading it naturally, preserving the "
    "reading order. Do not use HTML or markdown markup."
)

# Real, observed failure mode (docs/phases/phase-2-ocr-pipeline.md): even
# on prose-only crops, Qari has occasionally wrapped output in HTML-ish
# tags. Stripped defensively so a stray <b>/<h1>/<br> can never leak into
# the user's document — see recognize()'s docstring.
_HTML_TAG_RE = re.compile(r"<[^>]+>")


class QariUnavailableError(Exception):
    """Qari-OCR can't be used on this machine — missing dependencies, no
    CUDA GPU, or a missing/broken model checkpoint. Callers treat this as
    "fall back to Tesseract-only", never as a crash."""


class QariEngine:
    def __init__(self, model_dir: str | Path) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise QariUnavailableError(
                f"Qari-OCR dependencies not installed (pip install '.[vlm]'): {exc}"
            ) from exc

        if not torch.cuda.is_available():
            raise QariUnavailableError("no CUDA GPU detected — Qari-OCR requires one")

        model_dir = str(model_dir)
        try:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16, device_map="auto"
            )
            self._processor = AutoProcessor.from_pretrained(model_dir)
        except (OSError, ValueError) as exc:
            raise QariUnavailableError(f"failed to load Qari-OCR model from {model_dir}: {exc}") from exc

        self._torch = torch
        # OcrWorkerPool runs up to several pages concurrently on separate
        # threads (§6.2.1); a single shared model instance is not safe for
        # concurrent .generate() calls (and concurrent generations would
        # also contend for the same, already-tight, 8GB of VRAM) — every
        # recognize() call is serialized through this lock regardless of
        # which page's worker thread it came from.
        self._lock = threading.Lock()

    def recognize(self, image: Image.Image) -> str:
        """Runs Qari-OCR on one already-cropped region and returns plain
        text — callers are responsible for cropping to a prose/heading
        region (see this module's docstring for why tables are excluded).
        """
        from qwen_vl_utils import process_vision_info

        image = image.convert("RGB")
        if max(image.size) > _MAX_IMAGE_DIMENSION:
            scale = _MAX_IMAGE_DIMENSION / max(image.size)
            image = image.resize((round(image.width * scale), round(image.height * scale)))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ]
        text_prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )

        with self._lock:
            inputs = inputs.to(self._model.device)
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                # Without these, Qari degenerates into infinite exact-token
                # repetition on complex regions (docs/phases/
                # phase-2-ocr-pipeline.md). They don't fix table handling
                # (a different, deeper failure — see this module's
                # docstring), which is exactly why tables never reach this
                # method at all; on the prose/heading crops that do, this
                # setting is the one validated across this session's tests.
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
                do_sample=False,
            )
            generated_ids_trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        return _HTML_TAG_RE.sub("", output_text[0]).strip()
