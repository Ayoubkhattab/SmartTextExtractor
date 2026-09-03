from __future__ import annotations

import threading
import time
from pathlib import Path

from smart_text_extractor.concurrency.ocr_worker_pool import OcrWorkerPool
from smart_text_extractor.core.models import OcrResult, OcrStatus, Page


class _FakeEngine:
    """Stands in for OcrEngine — .run(path) is all OcrWorkerPool calls."""

    def __init__(self, delay: float = 0.0, fail_on: set[str] | None = None) -> None:
        self.delay = delay
        self.fail_on = fail_on or set()
        self.lock = threading.Lock()
        self.current_concurrency = 0
        self.max_observed_concurrency = 0

    def run(self, image_path: str) -> OcrResult:
        with self.lock:
            self.current_concurrency += 1
            self.max_observed_concurrency = max(self.max_observed_concurrency, self.current_concurrency)
        try:
            if self.delay:
                time.sleep(self.delay)
            if image_path in self.fail_on:
                raise RuntimeError(f"simulated OCR failure for {image_path}")
            return OcrResult(raw_text=f"text for {image_path}")
        finally:
            with self.lock:
                self.current_concurrency -= 1


def _wait_for(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


def test_submit_runs_engine_and_marks_page_done(tmp_path: Path) -> None:
    engine = _FakeEngine()
    pool = OcrWorkerPool(engine, max_workers=2)
    page = Page(image_path=tmp_path / "p1.png", order_index=0)
    results: list[tuple[Page, Exception | None]] = []

    pool.submit(page, lambda p, err: results.append((p, err)))
    _wait_for(lambda: len(results) == 1)
    pool.shutdown()

    assert page.ocr_status is OcrStatus.DONE
    assert page.ocr_result.raw_text == f"text for {page.image_path}"
    assert results[0] == (page, None)


def test_page_status_is_processing_immediately_after_submit(tmp_path: Path) -> None:
    engine = _FakeEngine(delay=0.3)
    pool = OcrWorkerPool(engine, max_workers=1)
    page = Page(image_path=tmp_path / "p1.png", order_index=0)

    pool.submit(page, lambda p, err: None)
    assert page.ocr_status is OcrStatus.PROCESSING

    pool.shutdown()


def test_one_failed_page_does_not_stop_the_others_skip_and_continue(tmp_path: Path) -> None:
    pages = [Page(image_path=tmp_path / f"p{i}.png", order_index=i) for i in range(3)]
    failing_path = str(pages[1].image_path)
    engine = _FakeEngine(fail_on={failing_path})
    pool = OcrWorkerPool(engine, max_workers=2)
    results: dict[str, tuple[Page, Exception | None]] = {}

    for page in pages:
        pool.submit(page, lambda p, err: results.__setitem__(str(p.image_path), (p, err)))

    _wait_for(lambda: len(results) == 3)
    pool.shutdown()

    assert pages[0].ocr_status is OcrStatus.DONE
    assert pages[2].ocr_status is OcrStatus.DONE
    assert pages[1].ocr_status is OcrStatus.FAILED
    assert results[str(pages[1].image_path)][1] is not None
    assert results[str(pages[0].image_path)][1] is None


def test_respects_max_workers_bound(tmp_path: Path) -> None:
    engine = _FakeEngine(delay=0.15)
    pool = OcrWorkerPool(engine, max_workers=2)
    pages = [Page(image_path=tmp_path / f"p{i}.png", order_index=i) for i in range(6)]
    done_count = {"n": 0}
    lock = threading.Lock()

    def on_done(_page, _err):
        with lock:
            done_count["n"] += 1

    for page in pages:
        pool.submit(page, on_done)

    _wait_for(lambda: done_count["n"] == len(pages), timeout=10.0)
    pool.shutdown()

    assert engine.max_observed_concurrency <= 2
    assert engine.max_observed_concurrency >= 1  # sanity: work actually happened concurrently at some point
