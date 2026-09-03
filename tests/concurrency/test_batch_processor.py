from __future__ import annotations

import threading
import time
from pathlib import Path

from smart_text_extractor.concurrency.batch_processor import BatchProcessor
from smart_text_extractor.concurrency.ocr_worker_pool import OcrWorkerPool
from smart_text_extractor.core.models import Document, OcrResult, OcrStatus, SourceType


class _FakeEngine:
    def __init__(self, fail_on: set[str] | None = None, delay: float = 0.0) -> None:
        self.fail_on = fail_on or set()
        self.delay = delay

    def run(self, image_path: str) -> OcrResult:
        if self.delay:
            time.sleep(self.delay)
        if image_path in self.fail_on:
            raise RuntimeError(f"simulated failure for {image_path}")
        return OcrResult(raw_text=f"text for {image_path}")


def _wait_for(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


def _batch_document(tmp_path: Path, n_pages: int) -> Document:
    doc = Document(source_type=SourceType.UPLOAD_IMAGE, temp_dir_path=tmp_path)
    for i in range(n_pages):
        doc.add_page(tmp_path / f"p{i}.png")
    return doc


def test_run_processes_all_pending_pages_successfully(tmp_path: Path) -> None:
    doc = _batch_document(tmp_path, 5)
    pool = OcrWorkerPool(_FakeEngine(), max_workers=2)
    completions = []

    processor = BatchProcessor(pool, on_complete=completions.append)
    processor.run(doc)

    _wait_for(lambda: len(completions) == 1)
    pool.shutdown()

    assert completions[0].total == 5
    assert completions[0].completed == 5
    assert completions[0].failed == 0
    assert all(page.ocr_status is OcrStatus.DONE for page in doc.pages)


def test_batch_completes_despite_one_failing_page_skip_and_continue(tmp_path: Path) -> None:
    doc = _batch_document(tmp_path, 5)
    failing_path = str(doc.pages[2].image_path)
    pool = OcrWorkerPool(_FakeEngine(fail_on={failing_path}), max_workers=2)
    completions = []

    processor = BatchProcessor(pool, on_complete=completions.append)
    processor.run(doc)

    _wait_for(lambda: len(completions) == 1)
    pool.shutdown()

    assert completions[0].total == 5
    assert completions[0].completed == 4
    assert completions[0].failed == 1
    assert doc.pages[2].ocr_status is OcrStatus.FAILED
    assert all(p.ocr_status is OcrStatus.DONE for p in doc.pages if p is not doc.pages[2])


def test_run_only_submits_pending_pages_not_already_done_or_processing(tmp_path: Path) -> None:
    doc = _batch_document(tmp_path, 3)
    doc.pages[0].ocr_status = OcrStatus.DONE
    doc.pages[0].ocr_result = OcrResult(raw_text="already done")
    doc.pages[1].ocr_status = OcrStatus.PROCESSING

    pool = OcrWorkerPool(_FakeEngine(), max_workers=2)
    completions = []
    processor = BatchProcessor(pool, on_complete=completions.append)
    processor.run(doc)

    _wait_for(lambda: len(completions) == 1)
    pool.shutdown()

    assert completions[0].total == 1  # only pages[2] was PENDING
    assert doc.pages[0].ocr_result.raw_text == "already done"  # untouched
    assert doc.pages[1].ocr_status is OcrStatus.PROCESSING  # untouched by this run


def test_empty_batch_calls_on_complete_immediately(tmp_path: Path) -> None:
    doc = Document(source_type=SourceType.UPLOAD_IMAGE, temp_dir_path=tmp_path)
    pool = OcrWorkerPool(_FakeEngine(), max_workers=2)
    completions = []

    processor = BatchProcessor(pool, on_complete=completions.append)
    processor.run(doc)

    assert len(completions) == 1
    assert completions[0].total == 0
    assert completions[0].is_complete is True
    pool.shutdown()


def test_progress_callback_fires_for_each_page(tmp_path: Path) -> None:
    doc = _batch_document(tmp_path, 4)
    pool = OcrWorkerPool(_FakeEngine(delay=0.02), max_workers=2)
    progress_events = []
    lock = threading.Lock()

    def on_progress(progress):
        with lock:
            progress_events.append(progress)

    processor = BatchProcessor(pool, on_progress=on_progress)
    processor.run(doc)

    _wait_for(lambda: len(progress_events) == 4)
    pool.shutdown()

    assert progress_events[-1].finished == 4
    assert progress_events[-1].is_complete
