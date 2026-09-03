from __future__ import annotations

from pathlib import Path

from smart_text_extractor.core.models import (
    Document,
    OcrResult,
    OcrStatus,
    Page,
    PageLockedError,
    Rect,
    SourceType,
)


def test_add_page_assigns_sequential_order_index(tmp_path: Path) -> None:
    doc = Document(source_type=SourceType.SCAN, temp_dir_path=tmp_path)

    first = doc.add_page(tmp_path / "p1.bmp")
    second = doc.add_page(tmp_path / "p2.bmp")

    assert first.order_index == 0
    assert second.order_index == 1
    assert doc.pages == [first, second]


def test_each_document_and_page_gets_a_unique_id(tmp_path: Path) -> None:
    doc_a = Document(source_type=SourceType.SCAN, temp_dir_path=tmp_path)
    doc_b = Document(source_type=SourceType.SCAN, temp_dir_path=tmp_path)
    page = doc_a.add_page(tmp_path / "p1.bmp")

    assert doc_a.id != doc_b.id
    assert page.id


def _done_page(tmp_path: Path) -> Page:
    page = Page(image_path=tmp_path / "p1.bmp", order_index=0)
    page.ocr_status = OcrStatus.DONE
    page.ocr_result = OcrResult(raw_text="hello", edited_text="hello (edited)")
    return page


def test_rotating_a_done_page_reopens_it_for_ocr_but_keeps_edited_text(tmp_path: Path) -> None:
    page = _done_page(tmp_path)

    page.set_rotation(90)

    assert page.rotation == 90
    assert page.ocr_status is OcrStatus.PENDING
    assert page.ocr_result.edited_text == "hello (edited)"  # §5.2: re-OCR never touches this


def test_cropping_a_done_page_reopens_it_for_ocr(tmp_path: Path) -> None:
    page = _done_page(tmp_path)

    page.set_crop_box(Rect(x=0, y=0, width=100, height=100))

    assert page.crop_box == Rect(x=0, y=0, width=100, height=100)
    assert page.ocr_status is OcrStatus.PENDING


def test_rotating_a_pending_page_does_not_change_status(tmp_path: Path) -> None:
    page = Page(image_path=tmp_path / "p1.bmp", order_index=0)
    assert page.ocr_status is OcrStatus.PENDING

    page.set_rotation(180)

    assert page.ocr_status is OcrStatus.PENDING


def test_set_rotation_rejects_invalid_degrees(tmp_path: Path) -> None:
    page = Page(image_path=tmp_path / "p1.bmp", order_index=0)
    try:
        page.set_rotation(45)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-multiple-of-90 rotation")


def test_set_edited_text_never_touches_raw_text(tmp_path: Path) -> None:
    page = Page(image_path=tmp_path / "p1.bmp", order_index=0)
    page.ocr_result = OcrResult(raw_text="raw from tesseract")

    page.set_edited_text("corrected by user")

    assert page.ocr_result.raw_text == "raw from tesseract"
    assert page.ocr_result.edited_text == "corrected by user"


def test_set_edited_text_creates_ocr_result_if_missing(tmp_path: Path) -> None:
    page = Page(image_path=tmp_path / "p1.bmp", order_index=0)
    assert page.ocr_result is None

    page.set_edited_text("typed before OCR ever ran")

    assert page.ocr_result.edited_text == "typed before OCR ever ran"


def _three_page_document(tmp_path: Path) -> Document:
    doc = Document(source_type=SourceType.SCAN, temp_dir_path=tmp_path)
    doc.add_page(tmp_path / "p1.bmp")
    doc.add_page(tmp_path / "p2.bmp")
    doc.add_page(tmp_path / "p3.bmp")
    return doc


def test_reorder_pages_updates_order_index(tmp_path: Path) -> None:
    doc = _three_page_document(tmp_path)
    p1, p2, p3 = doc.pages

    doc.reorder_pages([p3.id, p1.id, p2.id])

    assert [p.id for p in doc.pages] == [p3.id, p1.id, p2.id]
    assert p3.order_index == 0
    assert p1.order_index == 1
    assert p2.order_index == 2


def test_reorder_pages_rejects_a_non_permutation(tmp_path: Path) -> None:
    doc = _three_page_document(tmp_path)
    p1, p2, _p3 = doc.pages
    try:
        doc.reorder_pages([p1.id, p2.id])  # missing p3
    except ValueError:
        return
    raise AssertionError("expected ValueError for an incomplete reorder list")


def test_reorder_pages_refuses_to_move_a_page_mid_ocr(tmp_path: Path) -> None:
    doc = _three_page_document(tmp_path)
    p1, p2, p3 = doc.pages
    p2.ocr_status = OcrStatus.PROCESSING

    try:
        doc.reorder_pages([p2.id, p1.id, p3.id])  # tries to move p2 from index 1 to 0
    except PageLockedError:
        pass
    else:
        raise AssertionError("expected PageLockedError")

    assert [p.id for p in doc.pages] == [p1.id, p2.id, p3.id]  # unchanged


def test_reorder_pages_allows_list_where_processing_page_keeps_its_slot(tmp_path: Path) -> None:
    doc = _three_page_document(tmp_path)
    p1, p2, p3 = doc.pages
    p2.ocr_status = OcrStatus.PROCESSING

    doc.reorder_pages([p1.id, p2.id, p3.id])  # p2 stays at index 1 — not actually moved

    assert [p.id for p in doc.pages] == [p1.id, p2.id, p3.id]


def test_undo_reorder_restores_previous_order(tmp_path: Path) -> None:
    doc = _three_page_document(tmp_path)
    p1, p2, p3 = doc.pages

    doc.reorder_pages([p3.id, p1.id, p2.id])
    undone = doc.undo_reorder()

    assert undone is True
    assert [p.id for p in doc.pages] == [p1.id, p2.id, p3.id]
    assert p1.order_index == 0
    assert p2.order_index == 1
    assert p3.order_index == 2


def test_undo_reorder_with_no_history_returns_false(tmp_path: Path) -> None:
    doc = _three_page_document(tmp_path)
    assert doc.undo_reorder() is False


def test_included_in_range_defaults_to_true(tmp_path: Path) -> None:
    page = Page(image_path=tmp_path / "p1.bmp", order_index=0)
    assert page.included_in_range is True
