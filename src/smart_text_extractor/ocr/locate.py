"""Best-effort Tesseract binary/tessdata discovery for development runs.

Phase 5 packaging bundles Tesseract into the installer (§10) so end users
never hit this path — it only matters while running from source, on a
machine where Tesseract may or may not be on PATH.
"""
from __future__ import annotations

import shutil
from pathlib import Path

_WINDOWS_FALLBACK_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_TESSDATA = _PROJECT_ROOT / "tessdata"


def find_tesseract_cmd() -> str | None:
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    if _WINDOWS_FALLBACK_CMD.exists():
        return str(_WINDOWS_FALLBACK_CMD)
    return None


def find_tessdata_dir() -> Path | None:
    return _LOCAL_TESSDATA if _LOCAL_TESSDATA.exists() else None
