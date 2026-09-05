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
class TextStyle:
    """How a piece of text actually looks on the page (§7.1.1 extension).

    Only a PDF's own text layer carries this — font size, weight and colour
    are recorded in the file. OCR cannot recover any of it from pixels, so
    every field is optional and an OCR-produced segment simply has none;
    renderers fall back to their existing size/heading heuristics there.

    highlight is not a font property: it is the fill colour of a drawn
    rectangle sitting behind the text (a highlighted line, a coloured
    callout box), matched to the text by position.
    """

    font_size: float | None = None  # points, as stored in the PDF
    bold: bool = False
    italic: bool = False
    color: str | None = None  # "#rrggbb"
    highlight: str | None = None  # "#rrggbb" of the shape behind the text


@dataclass(frozen=True)
class PageLayout:
    """The physical page an exported document should reproduce (§7.1.1
    extension) — its size and its margins, in PDF points (1/72in).

    Taken from the source page rather than assumed: measured on this
    project's own documents, one is exactly A4 (595.3 x 841.9pt) and
    another is US Letter (612 x 792pt), so hardcoding either would put half
    the exports on the wrong paper and re-flow every line.

    Margins are measured from where the text actually sits, which is what
    decides the line width and therefore where lines break — the single
    biggest source of "the export needs re-formatting" when it is wrong.
    """

    width_points: float
    height_points: float
    margin_left: float
    margin_right: float
    margin_top: float
    margin_bottom: float


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
    style: TextStyle | None = None  # only a PDF text layer supplies this; None for OCR words


@dataclass(frozen=True)
class TextSegment:
    """One piece of assembled OCR output text (§7.1.1): either a
    recognized word (confidence set) or a structural separator — a plain
    space, a table-cell " | ", a line break, or a blank-line paragraph
    break (confidence None, since a separator was never "recognized").

    Concatenating every segment's `text` in order reproduces
    OcrResult.raw_text exactly — this is what lets the UI render the
    same text with per-word confidence highlighting instead of needing a
    second, separately-formatted representation that could drift out of
    sync with raw_text.
    """

    text: str
    confidence: float | None
    style: TextStyle | None = None  # carried through from the source word, so renderers can match the page's look


@dataclass
class DocumentUnit:
    """One structural unit of a page's content, classified for structured
    output (§7.1.1) — a heading, a table, or a paragraph. Produced once
    (smart_text_extractor.ocr.reorder.classify_document_units) and shared
    by every structured renderer — Markdown export, Word/docx export, and
    the live extracted-text panel in the UI — so the table/heading
    classification logic lives in exactly one place instead of being
    re-implemented, or re-parsed back out of a rendered string, per
    consumer. Lives here rather than in reorder.py because OcrResult
    (below) needs to reference it without reorder.py's OCR-pipeline
    internals (Line, block_num grouping) becoming part of the domain
    model.

    Content is TextSegments, not plain strings, for the same reason
    OcrResult.segments is: it lets the live UI panel render real
    structure (an actual table grid, a visually distinct heading) while
    still coloring each word by its own confidence — a table row or
    heading is not automatically "fully trustworthy" just because it was
    recognized as one.
    """

    kind: str  # "heading" | "table" | "paragraph"
    segments: list[TextSegment] = field(default_factory=list)  # heading/paragraph content; unused for kind == "table"
    rows: list[list[list[TextSegment]]] = field(default_factory=list)  # table cell rows (each cell its own segment list); unused outside kind == "table"
    bbox: Rect | None = None  # union of this unit's source words' positions on the page image (pixels, same space as OcrResult.word_boxes) — lets the hybrid OCR engine (ocr/hybrid_engine.py) crop exactly this region for a second pass; None only for a unit built without positional data (e.g. constructed directly by a test)
    alignment: str = "natural"  # "natural" (follow the text's own direction) | "center" — measured from where the unit actually sits in the page's text column, so a centred title stays centred in the output
    box_fill: str | None = None  # "#rrggbb" when this unit is a panel the page draws its content inside (ocr/layout_boxes.py), so the container is reproduced and not just its text
    column_widths: list[float] = field(default_factory=list)  # each column's share of the table's width, LEFT TO RIGHT as the page lays them out, summing to 1. Empty when the source's own column geometry is not known, and renderers then split the width evenly — the honest reconstruction of what is actually known.
    bordered: bool = True  # whether the source drew this table's RULES. A table whose grid was recovered from shaded cells rather than stroked lines (ocr/table_grid._grid_from_fills) has none, and drawing them adds ink the page never had — measured as a 34-point drop in visual similarity on a real page.
    space_before_points: float = 0.0  # blank vertical space above this unit on the source page, so the export keeps the same rhythm instead of collapsing everything to a uniform gap


@dataclass
class OcrResult:
    raw_text: str = ""
    edited_text: str | None = None  # None until the user edits (US-06); re-OCR never touches this
    word_boxes: list[BoundingBox] = field(default_factory=list)
    segments: list[TextSegment] = field(default_factory=list)
    markdown: str = ""  # structured export (§7.1.1): real tables/headings, not just flat text
    document_units: list[DocumentUnit] = field(default_factory=list)  # same structure, for non-Markdown exports (Word)
    confidence_score: float = 0.0
    page_layout: PageLayout | None = None  # the source page's real size/margins, when it had them (a PDF does; a bare image does not)


@dataclass(frozen=True)
class PdfPageSource:
    """Where a page came from, when it came from a PDF rather than a scan
    or an image file — enough to re-open the source and read that page's
    own embedded text layer (ocr/native_pdf_text.py).

    render_dpi is the DPI image_path was rendered at, and must be carried
    rather than re-derived: PDF text coordinates are in points, and only
    this number puts them in the same pixel space as the rendered image
    (and therefore as the OCR word boxes they get compared against).
    """

    pdf_path: Path
    page_index: int
    render_dpi: int
    text_layer_trusted: bool = True
    """Decided once per document when the PDF is opened
    (core/pdf_import.is_text_layer_trustworthy), not per page: the damage
    it screens for comes from the file's font/generator, so it is a
    property of the whole document. False means this page is OCR'd exactly
    as it was before native text extraction existed."""


@dataclass
class Page:
    """One page within a Document. order_index in the list = display/export order (US-07)."""

    image_path: Path
    order_index: int
    id: str = field(default_factory=lambda: uuid4().hex)
    dpi: int | None = None  # set at scan/import time (§7.3) — never assumed later
    pdf_source: PdfPageSource | None = None  # None for scans/plain images: those have no text layer to read
    rotation: int = 0
    crop_box: Rect | None = None
    ocr_status: OcrStatus = OcrStatus.PENDING
    ocr_result: OcrResult | None = None
    included_in_range: bool = True  # US-08: page-range selection for export

    def set_rotation(self, degrees: int) -> None:
        if degrees not in ROTATION_DEGREES:
            raise ValueError(f"rotation must be one of {ROTATION_DEGREES}, got {degrees}")
        self.rotation = degrees
        self._invalidate_ocr_if_done()

    def set_crop_box(self, rect: Rect | None) -> None:
        self.crop_box = rect
        self._invalidate_ocr_if_done()

    def set_edited_text(self, text: str) -> None:
        """US-06. Never touches raw_text (§5.2) — edited_text is the only
        field a user edit ever writes to."""
        if self.ocr_result is None:
            self.ocr_result = OcrResult()
        self.ocr_result.edited_text = text

    @property
    def is_locked_for_reordering(self) -> bool:
        """§3.1: a page mid-OCR can't be dragged — the UI checks this before
        allowing a drag to start; Document.reorder_pages enforces it too."""
        return self.ocr_status == OcrStatus.PROCESSING

    def _invalidate_ocr_if_done(self) -> None:
        if self.ocr_status == OcrStatus.DONE:
            self.ocr_status = OcrStatus.PENDING


class PageLockedError(Exception):
    """Raised when a reorder would move a page that's currently mid-OCR (§3.1)."""


@dataclass
class Document:
    """The document currently open in the session (§5.1)."""

    source_type: SourceType
    temp_dir_path: Path
    id: str = field(default_factory=lambda: uuid4().hex)
    pages: list[Page] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    _reorder_history: list[list[str]] = field(default_factory=list, repr=False)

    def add_page(self, image_path: Path, dpi: int | None = None) -> Page:
        page = Page(image_path=image_path, order_index=len(self.pages), dpi=dpi)
        self.pages.append(page)
        return page

    def reorder_pages(self, new_order_page_ids: list[str]) -> None:
        """US-07 (drag-and-drop). Raises PageLockedError instead of silently
        allowing a reorder that would move a page mid-OCR (§3.1) — the UI is
        expected to have already prevented the drag, but the domain layer
        enforces the invariant regardless of what called it.
        """
        by_id = {page.id: page for page in self.pages}
        if set(new_order_page_ids) != set(by_id):
            raise ValueError("new_order_page_ids must be a permutation of the document's current page ids")

        current_order = [page.id for page in self.pages]
        for page_id, new_index in {pid: i for i, pid in enumerate(new_order_page_ids)}.items():
            page = by_id[page_id]
            if page.is_locked_for_reordering and current_order.index(page_id) != new_index:
                raise PageLockedError(f"page {page_id} is mid-OCR and cannot be reordered")

        self._reorder_history.append(current_order)
        self.pages = [by_id[pid] for pid in new_order_page_ids]
        for index, page in enumerate(self.pages):
            page.order_index = index

    def undo_reorder(self) -> bool:
        """Restores the order from before the last reorder_pages() call.
        Returns False (no-op) if there is nothing to undo."""
        if not self._reorder_history:
            return False
        previous_order = self._reorder_history.pop()
        by_id = {page.id: page for page in self.pages}
        self.pages = [by_id[pid] for pid in previous_order]
        for index, page in enumerate(self.pages):
            page.order_index = index
        return True
