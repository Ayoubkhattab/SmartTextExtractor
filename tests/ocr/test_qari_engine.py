"""Tests for ocr/qari_engine.py that don't require a real GPU or the
several-GB model checkpoint. This project's own dev environment
deliberately does not install torch/transformers/qwen-vl-utils by default
(pyproject.toml's "vlm" optional-dependencies group) — QariEngine's
documented behavior on a machine like that (raise QariUnavailableError,
never crash) is exactly what's exercised here, on this actual machine's
real environment rather than a simulated one.
"""
from __future__ import annotations

import pytest

from smart_text_extractor.ocr.qari_engine import QariEngine, QariUnavailableError


def test_raises_qari_unavailable_when_dependencies_are_not_installed() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("torch is installed in this environment — the missing-dependency path isn't exercised here")

    with pytest.raises(QariUnavailableError):
        QariEngine("E:/hf_cache/Qari-OCR")


def test_unavailable_error_message_mentions_the_install_extra() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("torch is installed in this environment — the missing-dependency path isn't exercised here")

    with pytest.raises(QariUnavailableError, match=r"\.\[vlm\]"):
        QariEngine("E:/hf_cache/Qari-OCR")
