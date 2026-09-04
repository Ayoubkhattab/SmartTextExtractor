"""MainWindow tests use a fake OcrWorkerPool/ScannerService (duck-typed,
same pattern as tests/scanner/test_service.py and
tests/concurrency/test_ocr_worker_pool.py) — real OCR is exercised by
tests/ocr/, real threading by tests/concurrency/; this suite is only
responsible for proving the Qt wiring itself is correct.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from PyQt6.QtGui import QTextCursor

from smart_text_extractor.core.models import Document, DocumentUnit, OcrResult, OcrStatus, SourceType, TextSegment
from smart_text_extractor.scanner.models import ScannerDeviceInfo
from smart_text_extractor.ui.main_window import MainWindow


class _FakeOcrPool:
    """submit() calls on_done synchronously — no threads, deterministic."""

    def __init__(self, result: OcrResult | None = None, error: Exception | None = None) -> None:
        self.result = result or OcrResult(raw_text="fake extracted text")
        self.error = error
        self.submitted_pages = []

    def submit(self, page, on_done):
        self.submitted_pages.append(page)
        if self.error is not None:
            page.ocr_status = OcrStatus.FAILED  # mirrors the real OcrWorkerPool's error path
            on_done(page, self.error)
        else:
            page.ocr_result = self.result
            page.ocr_status = OcrStatus.DONE
            on_done(page, None)


class _FakeScannerService:
    def __init__(self, devices: list[ScannerDeviceInfo] | None = None) -> None:
        self._devices = devices or []

    def discover(self):
        return self._devices


@pytest.fixture()
def document(tmp_path: Path) -> Document:
    return Document(source_type=SourceType.UPLOAD_IMAGE, temp_dir_path=tmp_path)


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "sample.png"
    Image.new("RGB", (100, 50), "white").save(path)
    return path


def test_window_starts_with_no_page_selected(qtbot, document) -> None:
    window = MainWindow(document, _FakeOcrPool(), _FakeScannerService())
    qtbot.addWidget(window)

    assert window._page_list.count() == 0
    assert "لا توجد صفحة محددة" in window._image_label.text()


def test_add_page_submits_to_pool_and_marks_done(qtbot, document, sample_image) -> None:
    pool = _FakeOcrPool(result=OcrResult(raw_text="hello world"))
    window = MainWindow(document, pool, _FakeScannerService())
    qtbot.addWidget(window)

    window._add_page(sample_image)

    assert len(document.pages) == 1
    assert document.pages[0] in pool.submitted_pages
    assert document.pages[0].ocr_status is OcrStatus.DONE
    assert window._page_list.count() == 1
    assert "تم" in window._page_list.item(0).text()


def test_add_page_shows_failure_in_list(qtbot, document, sample_image) -> None:
    pool = _FakeOcrPool(error=RuntimeError("boom"))
    window = MainWindow(document, pool, _FakeScannerService())
    qtbot.addWidget(window)

    window._add_page(sample_image)

    assert document.pages[0].ocr_status is OcrStatus.FAILED
    assert "فشل" in window._page_list.item(0).text()


def test_selecting_a_done_page_shows_its_extracted_text(qtbot, document, sample_image) -> None:
    pool = _FakeOcrPool(result=OcrResult(raw_text="the extracted text"))
    window = MainWindow(document, pool, _FakeScannerService())
    qtbot.addWidget(window)

    window._add_page(sample_image)
    window._page_list.setCurrentRow(0)

    assert window._text_edit.toPlainText() == "the extracted text"
    assert window._text_edit.isReadOnly() is False


def _char_format_at(text_edit, position: int):
    cursor = QTextCursor(text_edit.document())
    cursor.setPosition(position)
    cursor.setPosition(position + 1, QTextCursor.MoveMode.KeepAnchor)
    return cursor.charFormat()


def test_low_and_very_low_confidence_words_are_highlighted_distinctly(qtbot, document, sample_image) -> None:
    segments = [
        TextSegment("Good", 95.0),  # >= 75: no highlight
        TextSegment(" ", None),  # separator: never highlighted regardless of neighbors
        TextSegment("Iffy", 60.0),  # 50-75: low-confidence highlight
        TextSegment(" ", None),
        TextSegment("Bad", 40.0),  # < 50: very-low-confidence highlight
    ]
    result = OcrResult(raw_text="Good Iffy Bad", segments=segments)
    pool = _FakeOcrPool(result=result)
    window = MainWindow(document, pool, _FakeScannerService())
    qtbot.addWidget(window)

    window._add_page(sample_image)
    window._page_list.setCurrentRow(0)

    assert window._text_edit.toPlainText() == "Good Iffy Bad"
    from smart_text_extractor.ui.main_window import _LOW_CONFIDENCE_COLOR, _VERY_LOW_CONFIDENCE_COLOR

    assert _char_format_at(window._text_edit, 0).background().color().name() != _LOW_CONFIDENCE_COLOR.name()
    assert _char_format_at(window._text_edit, 5).background().color().name() == _LOW_CONFIDENCE_COLOR.name()
    assert _char_format_at(window._text_edit, 10).background().color().name() == _VERY_LOW_CONFIDENCE_COLOR.name()


def test_editing_text_calls_set_edited_text_not_raw_text(qtbot, document, sample_image) -> None:
    pool = _FakeOcrPool(result=OcrResult(raw_text="original"))
    window = MainWindow(document, pool, _FakeScannerService())
    qtbot.addWidget(window)

    window._add_page(sample_image)
    window._page_list.setCurrentRow(0)

    window._text_edit.setPlainText("edited by the user")

    page = document.pages[0]
    assert page.ocr_result.raw_text == "original"
    assert page.ocr_result.edited_text == "edited by the user"


def test_scan_with_no_devices_shows_information_message(qtbot, document, monkeypatch) -> None:
    window = MainWindow(document, _FakeOcrPool(), _FakeScannerService(devices=[]))
    qtbot.addWidget(window)

    shown = []
    monkeypatch.setattr(
        "smart_text_extractor.ui.main_window.QMessageBox.information",
        lambda *args, **kwargs: shown.append(args),
    )

    window._on_scan()

    assert len(shown) == 1


def test_scan_with_devices_shows_a_different_message(qtbot, document, monkeypatch) -> None:
    devices = [ScannerDeviceInfo(device_id="dev-1", name="Fake Scanner")]
    window = MainWindow(document, _FakeOcrPool(), _FakeScannerService(devices=devices))
    qtbot.addWidget(window)

    shown = []
    monkeypatch.setattr(
        "smart_text_extractor.ui.main_window.QMessageBox.information",
        lambda *args, **kwargs: shown.append(args),
    )

    window._on_scan()

    assert len(shown) == 1
    assert "1" in shown[0][2]  # message text mentions the device count


def test_export_markdown_with_no_done_pages_shows_information_message(qtbot, document, monkeypatch) -> None:
    window = MainWindow(document, _FakeOcrPool(), _FakeScannerService())
    qtbot.addWidget(window)

    shown = []
    monkeypatch.setattr(
        "smart_text_extractor.ui.main_window.QMessageBox.information",
        lambda *args, **kwargs: shown.append(args),
    )
    save_calls = []
    monkeypatch.setattr(
        "smart_text_extractor.ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: save_calls.append(args) or ("", ""),
    )

    window._on_export_markdown()

    assert len(shown) == 1
    assert save_calls == []  # never even prompted for a save path


def test_export_markdown_writes_markdown_for_unedited_pages_and_plain_text_for_edited_ones(
    qtbot, document, sample_image, tmp_path: Path, monkeypatch
) -> None:
    pool = _FakeOcrPool()  # submit() assigns the SAME OcrResult instance to every page
    window = MainWindow(document, pool, _FakeScannerService())
    qtbot.addWidget(window)

    window._add_page(sample_image)  # page 0: kept as a fresh, unedited OCR result below
    window._add_page(sample_image)  # page 1: will be edited by the user below

    # Give each page its own independent OcrResult — the fake pool shares
    # one instance across every submit() call, which isn't representative
    # of the real per-page OcrWorkerPool and would make the two pages'
    # edited_text alias each other if left as-is.
    document.pages[0].ocr_result = OcrResult(raw_text="RAW A", markdown="## RAW A (as markdown)")
    document.pages[1].ocr_result = OcrResult(raw_text="RAW B", markdown="## RAW B (as markdown)")

    window._page_list.setCurrentRow(1)
    window._text_edit.setPlainText("user's edited text")  # sets page 1's edited_text

    save_path = tmp_path / "export"  # deliberately no .md suffix
    monkeypatch.setattr(
        "smart_text_extractor.ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(save_path), ""),
    )

    window._on_export_markdown()

    written_path = tmp_path / "export.md"
    assert written_path.exists()
    content = written_path.read_text(encoding="utf-8")
    assert content == "## RAW A (as markdown)\n\n---\n\nuser's edited text"


def test_export_markdown_skips_pages_excluded_from_range(qtbot, document, sample_image, tmp_path: Path, monkeypatch) -> None:
    pool = _FakeOcrPool(result=OcrResult(raw_text="included", markdown="included"))
    window = MainWindow(document, pool, _FakeScannerService())
    qtbot.addWidget(window)

    window._add_page(sample_image)
    document.pages[0].included_in_range = False

    save_path = tmp_path / "export.md"
    monkeypatch.setattr(
        "smart_text_extractor.ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(save_path), ""),
    )
    shown = []
    monkeypatch.setattr(
        "smart_text_extractor.ui.main_window.QMessageBox.information",
        lambda *args, **kwargs: shown.append(args),
    )

    window._on_export_markdown()

    assert len(shown) == 1  # no exportable pages left once the only page is excluded
    assert not save_path.exists()


def test_export_word_writes_a_real_docx_using_document_units(qtbot, document, sample_image, tmp_path: Path, monkeypatch) -> None:
    import docx

    units = [DocumentUnit(kind="heading", text="عنوان"), DocumentUnit(kind="paragraph", text="نص الفقرة")]
    pool = _FakeOcrPool(result=OcrResult(raw_text="RAW", document_units=units))
    window = MainWindow(document, pool, _FakeScannerService())
    qtbot.addWidget(window)

    window._add_page(sample_image)

    save_path = tmp_path / "export"  # no .docx suffix — MainWindow must add one
    monkeypatch.setattr(
        "smart_text_extractor.ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(save_path), ""),
    )

    window._on_export_word()

    written_path = tmp_path / "export.docx"
    assert written_path.exists()
    reopened = docx.Document(str(written_path))
    assert any(p.text == "عنوان" and p.style.name == "Heading 2" for p in reopened.paragraphs)
    assert any(p.text == "نص الفقرة" for p in reopened.paragraphs)


def test_opening_a_multi_page_pdf_adds_one_page_per_pdf_page(qtbot, document, tmp_path: Path) -> None:
    import pymupdf

    pdf_path = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page(width=300, height=200)
        page.insert_text((20, 20), f"page {i + 1}", fontsize=16)
    doc.save(str(pdf_path))
    doc.close()

    pool = _FakeOcrPool()
    window = MainWindow(document, pool, _FakeScannerService())
    qtbot.addWidget(window)

    window._open_file(pdf_path)

    assert len(document.pages) == 3
    assert window._page_list.count() == 3
    assert len(pool.submitted_pages) == 3


def test_opening_a_broken_pdf_shows_a_warning_not_a_crash(qtbot, document, tmp_path: Path, monkeypatch) -> None:
    fake_pdf = tmp_path / "not_really_a_pdf.pdf"
    fake_pdf.write_bytes(b"this is not a valid pdf file")

    window = MainWindow(document, _FakeOcrPool(), _FakeScannerService())
    qtbot.addWidget(window)

    shown = []
    monkeypatch.setattr(
        "smart_text_extractor.ui.main_window.QMessageBox.warning",
        lambda *args, **kwargs: shown.append(args),
    )

    window._open_file(fake_pdf)

    assert len(shown) == 1
    assert len(document.pages) == 0
