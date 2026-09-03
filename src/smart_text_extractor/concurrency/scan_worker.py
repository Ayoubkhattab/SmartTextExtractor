"""ScanWorker: single dedicated thread, serial queue (§6.2.1).

A scanner cannot serve two concurrent scan operations — the queue exists
so callers (eventually: the UI thread) never call ScannerService directly
and never block on a scan, but the underlying work is still one-at-a-time
by construction, not a thread pool.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable

from smart_text_extractor.scanner.models import ScannedImage, ScanSettings
from smart_text_extractor.scanner.service import ScannerService

OnScanDone = Callable[[ScannedImage | None, Exception | None], None]


@dataclass
class _ScanJob:
    device_id: str
    settings: ScanSettings
    on_done: OnScanDone


class ScanWorker:
    def __init__(self, scanner_service: ScannerService) -> None:
        self._service = scanner_service
        self._queue: queue.Queue[_ScanJob] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="ScanWorker", daemon=True)
        self._thread.start()

    def submit(self, device_id: str, settings: ScanSettings, on_done: OnScanDone) -> None:
        self._queue.put(_ScanJob(device_id=device_id, settings=settings, on_done=on_done))

    def stop(self, wait: bool = True) -> None:
        self._stop_event.set()
        if wait and self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._execute(job)

    def _execute(self, job: _ScanJob) -> None:
        try:
            handle = self._service.open(job.device_id)
            try:
                image = self._service.scan(handle, job.settings)
            finally:
                self._service.close(handle)
        except Exception as exc:  # noqa: BLE001 - forwarded to caller, never raised into the worker thread
            job.on_done(None, exc)
        else:
            job.on_done(image, None)
