from __future__ import annotations

from pathlib import Path

from smart_text_extractor.core.models import (
    BoundingBox,
    Document,
    OcrResult,
    OcrStatus,
    Rect,
    SourceType,
)
from smart_text_extractor.core.persistence import (
    load_document,
    resume_pending_pages,
    save_document,
)


def test_save_and_load_round_trips_a_simple_document(tmp_path: Path) -> None:
    doc = Document(source_type=SourceType.SCAN, temp_dir_path=tmp_path)
    doc.add_page(tmp_path / "p1.bmp", dpi=300)
    doc.add_page(tmp_path / "p2.bmp", dpi=300)

    save_path = tmp_path / "state.json"
    save_document(doc, save_path)
    reloaded = load_document(save_path)

    assert reloaded.id == doc.id
    assert reloaded.source_type == SourceType.SCAN
    assert len(reloaded.pages) == 2
    assert reloaded.pages[0].dpi == 300
    assert reloaded.pages[0].image_path == doc.pages[0].image_path


def test_save_and_load_round_trips_full_ocr_result_and_geometry(tmp_path: Path) -> None:
    doc = Document(source_type=SourceType.UPLOAD_IMAGE, temp_dir_path=tmp_path)
    page = doc.add_page(tmp_path / "p1.bmp", dpi=300)
    page.set_rotation(90)
    page.set_crop_box(Rect(x=1, y=2, width=300, height=400))
    page.ocr_status = OcrStatus.DONE
    page.ocr_result = OcrResult(
        raw_text="مرحبا",
        edited_text="مرحبا (edited)",
        confidence_score=91.5,
        word_boxes=[BoundingBox(text="مرحبا", rect=Rect(1, 2, 3, 4), confidence=91.5)],
    )

    save_path = tmp_path / "state.json"
    save_document(doc, save_path)
    reloaded = load_document(save_path)

    reloaded_page = reloaded.pages[0]
    assert reloaded_page.rotation == 90
    assert reloaded_page.crop_box == Rect(x=1, y=2, width=300, height=400)
    assert reloaded_page.ocr_status is OcrStatus.DONE
    assert reloaded_page.ocr_result.raw_text == "مرحبا"
    assert reloaded_page.ocr_result.edited_text == "مرحبا (edited)"
    assert reloaded_page.ocr_result.word_boxes[0].text == "مرحبا"
    assert reloaded_page.ocr_result.word_boxes[0].rect == Rect(1, 2, 3, 4)


def test_save_is_atomic_no_tmp_file_left_behind(tmp_path: Path) -> None:
    doc = Document(source_type=SourceType.SCAN, temp_dir_path=tmp_path)
    save_path = tmp_path / "state.json"

    save_document(doc, save_path)

    assert save_path.exists()
    assert not save_path.with_suffix(".json.tmp").exists()


def test_resume_pending_pages_resets_processing_pages_and_counts_them(tmp_path: Path) -> None:
    doc = Document(source_type=SourceType.SCAN, temp_dir_path=tmp_path)
    doc.add_page(tmp_path / "p1.bmp")
    doc.add_page(tmp_path / "p2.bmp")
    doc.add_page(tmp_path / "p3.bmp")
    doc.pages[0].ocr_status = OcrStatus.DONE
    doc.pages[1].ocr_status = OcrStatus.PROCESSING  # was mid-OCR when the crash happened
    doc.pages[2].ocr_status = OcrStatus.PENDING

    reset_count = resume_pending_pages(doc)

    assert reset_count == 1
    assert doc.pages[0].ocr_status is OcrStatus.DONE  # untouched
    assert doc.pages[1].ocr_status is OcrStatus.PENDING  # reset
    assert doc.pages[2].ocr_status is OcrStatus.PENDING  # untouched


def test_full_crash_resume_scenario(tmp_path: Path) -> None:
    """Save mid-batch (one DONE, one PROCESSING when the crash happens, one
    still PENDING) -> reload -> resume_pending_pages -> only the genuinely
    unfinished pages are PENDING again, DONE work is never lost."""
    doc = Document(source_type=SourceType.UPLOAD_IMAGE, temp_dir_path=tmp_path)
    doc.add_page(tmp_path / "p1.png")
    doc.add_page(tmp_path / "p2.png")
    doc.add_page(tmp_path / "p3.png")
    doc.pages[0].ocr_status = OcrStatus.DONE
    doc.pages[0].ocr_result = OcrResult(raw_text="already extracted, must survive the crash")
    doc.pages[1].ocr_status = OcrStatus.PROCESSING  # simulated crash happens here

    save_path = tmp_path / "state.json"
    save_document(doc, save_path)

    reloaded = load_document(save_path)  # simulates restart after the crash
    reset_count = resume_pending_pages(reloaded)

    assert reset_count == 1
    assert reloaded.pages[0].ocr_status is OcrStatus.DONE
    assert reloaded.pages[0].ocr_result.raw_text == "already extracted, must survive the crash"
    assert reloaded.pages[1].ocr_status is OcrStatus.PENDING
    assert reloaded.pages[2].ocr_status is OcrStatus.PENDING
