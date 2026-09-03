"""BatchProcessor: functional requirement 14 — process a whole batch of
pages through OcrWorkerPool with Skip-and-Continue (§3.2 option b).

The pool already isolates one page's failure from another's task; this
module adds the batch-level bookkeeping — knowing when every submitted
page has reached a terminal state, and reporting how many succeeded vs.
failed, without ever treating "some pages failed" as a reason to stop.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from smart_text_extractor.concurrency.ocr_worker_pool import OcrWorkerPool
from smart_text_extractor.core.models import Document, OcrStatus, Page

OnProgress = Callable[["BatchProgress"], None]


@dataclass(frozen=True)
class BatchProgress:
    total: int
    completed: int
    failed: int

    @property
    def finished(self) -> int:
        return self.completed + self.failed

    @property
    def is_complete(self) -> bool:
        return self.finished >= self.total


class BatchProcessor:
    def __init__(
        self,
        pool: OcrWorkerPool,
        on_progress: OnProgress | None = None,
        on_complete: OnProgress | None = None,
    ) -> None:
        self._pool = pool
        self._on_progress = on_progress
        self._on_complete = on_complete

    def run(self, document: Document) -> BatchProgress:
        """Submits every PENDING page in the document to the pool.

        Pages already DONE/PROCESSING/FAILED are left alone — this is also
        how batch resume works (§3.3): a document reloaded after a crash
        just has run() called again, and only the pages still PENDING get
        (re-)submitted.
        """
        pending_pages = [page for page in document.pages if page.ocr_status == OcrStatus.PENDING]
        total = len(pending_pages)
        state = {"completed": 0, "failed": 0}
        lock = threading.Lock()

        if total == 0:
            progress = BatchProgress(total=0, completed=0, failed=0)
            if self._on_complete:
                self._on_complete(progress)
            return progress

        def _on_page_done(_page: Page, error: Exception | None) -> None:
            with lock:
                if error is not None:
                    state["failed"] += 1
                else:
                    state["completed"] += 1
                progress = BatchProgress(total=total, completed=state["completed"], failed=state["failed"])
                if self._on_progress:
                    self._on_progress(progress)
                if progress.is_complete and self._on_complete:
                    self._on_complete(progress)

        for page in pending_pages:
            self._pool.submit(page, _on_page_done)

        return BatchProgress(total=total, completed=0, failed=0)
