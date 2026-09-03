"""MainWindow tests use a fake OcrWorkerPool/ScannerService (duck-typed,
same pattern as tests/scanner/test_service.py and
tests/concurrency/test_ocr_worker_pool.py) — real OCR is exercised by
tests/ocr/, real threading by tests/concurrency/; this suite is only
responsible for proving the Qt wiring itself is correct.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from smart_text_extractor.core.models import Document, OcrResult, OcrStatus, SourceType
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
