"""Minimal Phase 1 domain entities (§5.1).

Page here is deliberately narrow — rotation, crop_box, ocr_status, and
ocr_result are added in Phase 2 once OCR exists (§14, Phase 2 task list).
Adding them now, before anything reads or writes them, would be exactly
the kind of speculative field the project avoids.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4


class SourceType(Enum):
    SCAN = "scan"
    UPLOAD_IMAGE = "upload_image"
    UPLOAD_PDF = "upload_pdf"


@dataclass
class Page:
    """One page within a Document. order_index in the list = display/export order (US-07)."""

    image_path: Path
    order_index: int
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class Document:
    """The document currently open in the session (§5.1)."""

    source_type: SourceType
    temp_dir_path: Path
    id: str = field(default_factory=lambda: uuid4().hex)
    pages: list[Page] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def add_page(self, image_path: Path) -> Page:
        page = Page(image_path=image_path, order_index=len(self.pages))
        self.pages.append(page)
        return page
