from __future__ import annotations

from pathlib import Path

from smart_text_extractor.core.models import (
    Document,
    OcrResult,
    OcrStatus,
    Page,
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
