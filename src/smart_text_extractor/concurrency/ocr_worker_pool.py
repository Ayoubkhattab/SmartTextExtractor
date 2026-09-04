"""OcrWorkerPool: bounded parallel OCR, one task per page (§6.2, §6.2.1).

Each page's OCR is an independent future — one page's failure never
touches another's task, which is what makes Skip-and-Continue (§3.2
option B) mechanically possible: the pool doesn't stop just because one
future raised.

Default max_workers = min(cpu_count - 1, 4), per §6.2.1 — a bound exists
specifically to cap how many full-resolution images sit in memory at once
(§3.3's OOM mitigation), not just to control CPU contention.
"""
from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from smart_text_extractor.core.models import OcrStatus, Page
from smart_text_extractor.ocr.engine import OcrEngine
from smart_text_extractor.ocr.page_pipeline import run_page

OnPageDone = Callable[[Page, Exception | None], None]


def _default_max_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return min(max(cpu_count - 1, 1), 4)


class OcrWorkerPool:
    def __init__(self, engine: OcrEngine, max_workers: int | None = None) -> None:
        self._engine = engine
        self.max_workers = max_workers or _default_max_workers()
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="OcrWorker")

    def submit(self, page: Page, on_done: OnPageDone) -> Future:
        page.ocr_status = OcrStatus.PROCESSING
        # run_page, not engine.run directly: a page rendered from a PDF also
        # has that PDF's own embedded text, which is combined with the OCR
        # reading here (ocr/page_pipeline.py). Pages that aren't from a PDF
        # go straight through to the engine exactly as before.
        future = self._executor.submit(run_page, page, self._engine)
        future.add_done_callback(lambda f: self._handle_result(page, f, on_done))
        return future

    def _handle_result(self, page: Page, future: Future, on_done: OnPageDone) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - Skip-and-Continue: reported, never re-raised here
            page.ocr_status = OcrStatus.FAILED
            on_done(page, exc)
        else:
            page.ocr_result = result
            page.ocr_status = OcrStatus.DONE
            on_done(page, None)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
