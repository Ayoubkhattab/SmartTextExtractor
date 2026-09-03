"""Per-session temp workspace lifecycle (§9.1, §5.1: Document.temp_dir_path).

One TempWorkspace per app session. start() creates an isolated directory
under a dedicated base dir; cleanup() removes it on normal exit. Because a
clean exit always removes its own directory, anything left in the base dir
at the next startup is by definition an orphan from an abnormal exit (crash,
kill) — cleanup_orphaned() must run once at startup, before start().
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from uuid import uuid4


class TempWorkspace:
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or Path(tempfile.gettempdir()) / "smart_text_extractor_sessions"
        self._session_dir: Path | None = None

    def cleanup_orphaned(self) -> list[Path]:
        """Remove leftover session directories from a prior abnormal exit.

        Call once at app startup, before start(). Never raises on a single
        bad entry — best-effort, since a stray orphan is not worth crashing
        startup over.
        """
        removed: list[Path] = []
        if not self._base_dir.exists():
            return removed
        for child in self._base_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                removed.append(child)
        return removed

    def start(self) -> Path:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        session_dir = self._base_dir / uuid4().hex
        session_dir.mkdir()
        self._session_dir = session_dir
        return session_dir

    @property
    def path(self) -> Path:
        if self._session_dir is None:
            raise RuntimeError("start() must be called before accessing path")
        return self._session_dir

    def cleanup(self) -> None:
        """Normal-exit cleanup — call from aboutToQuit (Qt) or atexit."""
        if self._session_dir is not None and self._session_dir.exists():
            shutil.rmtree(self._session_dir, ignore_errors=True)
        self._session_dir = None
