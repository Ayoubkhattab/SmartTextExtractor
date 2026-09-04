"""JSON persistence for Document/Page state (§3.3 bullet 3).

Written periodically during a batch so it can resume from the first still-
PENDING page after an abnormal exit (OOM, crash, kill) instead of
restarting the whole batch.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from smart_text_extractor.core.models import (
    BoundingBox,
    Document,
    OcrResult,
    OcrStatus,
    Page,
    PdfPageSource,
    Rect,
    SourceType,
)


def _rect_to_dict(rect: Rect | None) -> dict | None:
    return None if rect is None else {"x": rect.x, "y": rect.y, "width": rect.width, "height": rect.height}


def _rect_from_dict(data: dict | None) -> Rect | None:
    return None if data is None else Rect(x=data["x"], y=data["y"], width=data["width"], height=data["height"])


def _ocr_result_to_dict(result: OcrResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "raw_text": result.raw_text,
        "edited_text": result.edited_text,
        "confidence_score": result.confidence_score,
        "word_boxes": [
            {"text": box.text, "rect": _rect_to_dict(box.rect), "confidence": box.confidence}
            for box in result.word_boxes
        ],
    }


def _ocr_result_from_dict(data: dict | None) -> OcrResult | None:
    if data is None:
        return None
    return OcrResult(
        raw_text=data["raw_text"],
        edited_text=data["edited_text"],
        confidence_score=data["confidence_score"],
        word_boxes=[
            BoundingBox(text=box["text"], rect=_rect_from_dict(box["rect"]), confidence=box["confidence"])
            for box in data["word_boxes"]
        ],
    )


def _pdf_source_to_dict(source: PdfPageSource | None) -> dict | None:
    if source is None:
        return None
    return {
        "pdf_path": str(source.pdf_path),
        "page_index": source.page_index,
        "render_dpi": source.render_dpi,
        "text_layer_trusted": source.text_layer_trusted,
    }


def _pdf_source_from_dict(data: dict | None) -> PdfPageSource | None:
    if data is None:
        return None
    return PdfPageSource(
        pdf_path=Path(data["pdf_path"]),
        page_index=data["page_index"],
        render_dpi=data["render_dpi"],
        text_layer_trusted=data.get("text_layer_trusted", True),
    )


def _page_to_dict(page: Page) -> dict:
    return {
        "id": page.id,
        "image_path": str(page.image_path),
        "order_index": page.order_index,
        "dpi": page.dpi,
        "rotation": page.rotation,
        "crop_box": _rect_to_dict(page.crop_box),
        "ocr_status": page.ocr_status.value,
        "ocr_result": _ocr_result_to_dict(page.ocr_result),
        "included_in_range": page.included_in_range,
        "pdf_source": _pdf_source_to_dict(page.pdf_source),
    }


def _page_from_dict(data: dict) -> Page:
    return Page(
        id=data["id"],
        image_path=Path(data["image_path"]),
        order_index=data["order_index"],
        dpi=data["dpi"],
        rotation=data["rotation"],
        crop_box=_rect_from_dict(data["crop_box"]),
        ocr_status=OcrStatus(data["ocr_status"]),
        ocr_result=_ocr_result_from_dict(data["ocr_result"]),
        included_in_range=data["included_in_range"],
        # .get, not [...]: a document saved before pdf_source existed must
        # still load — it just re-OCRs without its text layer.
        pdf_source=_pdf_source_from_dict(data.get("pdf_source")),
    )


def document_to_dict(document: Document) -> dict:
    return {
        "id": document.id,
        "source_type": document.source_type.value,
        "temp_dir_path": str(document.temp_dir_path),
        "created_at": document.created_at.isoformat(),
        "pages": [_page_to_dict(page) for page in document.pages],
    }


def document_from_dict(data: dict) -> Document:
    document = Document(
        source_type=SourceType(data["source_type"]),
        temp_dir_path=Path(data["temp_dir_path"]),
        id=data["id"],
        created_at=datetime.fromisoformat(data["created_at"]),
    )
    document.pages = [_page_from_dict(page) for page in data["pages"]]
    return document


def save_document(document: Document, path: Path) -> None:
    """Atomic write (temp file + rename) — a crash mid-write must never
    leave a half-written, unreadable state file behind (§3.3's whole point
    is surviving a crash, so the save mechanism itself has to survive one)."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(document_to_dict(document), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_document(path: Path) -> Document:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return document_from_dict(data)


def resume_pending_pages(document: Document) -> int:
    """§3.3: a page saved as PROCESSING was mid-OCR when the crash
    happened — that task no longer exists after restart, so it must go
    back to PENDING or BatchProcessor.run() would never pick it up again.
    Returns how many pages were reset.
    """
    reset_count = 0
    for page in document.pages:
        if page.ocr_status == OcrStatus.PROCESSING:
            page.ocr_status = OcrStatus.PENDING
            reset_count += 1
    return reset_count
