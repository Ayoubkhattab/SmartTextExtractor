"""Best-effort Tesseract binary/tessdata discovery for development runs.

Phase 5 packaging bundles Tesseract into the installer (§10) so end users
never hit this path — it only matters while running from source, on a
machine where Tesseract may or may not be on PATH.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

_WINDOWS_FALLBACK_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_TESSDATA = _PROJECT_ROOT / "tessdata"

# Qari-OCR (docs/phases/phase-2-ocr-pipeline.md, hybrid-engine findings):
# no installer/PATH convention exists for this yet — it's a several-GB
# model checkpoint the user downloads manually (HF Hub's unauthenticated
# rate limit made the automated download impractical). QARI_MODEL_DIR lets
# it live anywhere; the fallback matches where it's actually installed on
# this development machine, same spirit as _WINDOWS_FALLBACK_CMD above.
_QARI_MODEL_DIR_FALLBACK = Path(r"E:\hf_cache\Qari-OCR")


def find_tesseract_cmd() -> str | None:
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    if _WINDOWS_FALLBACK_CMD.exists():
        return str(_WINDOWS_FALLBACK_CMD)
    return None


def find_tessdata_dir() -> Path | None:
    return _LOCAL_TESSDATA if _LOCAL_TESSDATA.exists() else None


def find_qari_model_dir() -> Path | None:
    env_dir = os.environ.get("QARI_MODEL_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    if _QARI_MODEL_DIR_FALLBACK.exists():
        return _QARI_MODEL_DIR_FALLBACK
    return None
