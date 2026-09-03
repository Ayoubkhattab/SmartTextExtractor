"""Domain entities (§5.1), extended in Phase 2 with the OCR-related shape.

Page now carries rotation/crop_box/ocr_status/ocr_result (§5.1) and the
DONE -> PENDING invariant from §5.3: any geometric edit on an already-OCR'd
page must reopen it for re-OCR, because word_boxes/raw_text were computed
against the pre-edit image. edited_text is untouched by this — it lives
separately in OcrResult and only the user ever replaces it (§5.2).
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


class OcrStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


ROTATION_DEGREES = (0, 90, 180, 270)


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class BoundingBox:
    """One recognized word and its position on the source image (§7.3).

    Coordinates are pixel offsets on the image as captured at Page.dpi —
    they are meaningless without that DPI, which is why Page stores it
    explicitly rather than assuming a fixed value (§7.3).
    """

    text: str
    rect: Rect
    confidence: float


@dataclass
class OcrResult:
    raw_text: str = ""
    edited_text: str | None = None  # None until the user edits (US-06); re-OCR never touches this
    word_boxes: list[BoundingBox] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class Page:
    """One page within a Document. order_index in the list = display/export order (US-07)."""

    image_path: Path
    order_index: int
    id: str = field(default_factory=lambda: uuid4().hex)
    dpi: int | None = None  # set at scan/import time (§7.3) — never assumed later
    rotation: int = 0
    crop_box: Rect | None = None
    ocr_status: OcrStatus = OcrStatus.PENDING
    ocr_result: OcrResult | None = None

    def set_rotation(self, degrees: int) -> None:
        if degrees not in ROTATION_DEGREES:
            raise ValueError(f"rotation must be one of {ROTATION_DEGREES}, got {degrees}")
        self.rotation = degrees
        self._invalidate_ocr_if_done()

    def set_crop_box(self, rect: Rect | None) -> None:
        self.crop_box = rect
        self._invalidate_ocr_if_done()

    def _invalidate_ocr_if_done(self) -> None:
        if self.ocr_status == OcrStatus.DONE:
            self.ocr_status = OcrStatus.PENDING


@dataclass
class Document:
    """The document currently open in the session (§5.1)."""

    source_type: SourceType
    temp_dir_path: Path
    id: str = field(default_factory=lambda: uuid4().hex)
    pages: list[Page] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def add_page(self, image_path: Path, dpi: int | None = None) -> Page:
        page = Page(image_path=image_path, order_index=len(self.pages), dpi=dpi)
        self.pages.append(page)
        return page
