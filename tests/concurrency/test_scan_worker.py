from __future__ import annotations

import threading
import time

from smart_text_extractor.concurrency.scan_worker import ScanWorker
from smart_text_extractor.scanner.base import ScannerDriver
from smart_text_extractor.scanner.errors import DeviceDisconnectedError
from smart_text_extractor.scanner.models import (
    ScannedImage,
    ScannerCapabilities,
    ScannerDeviceInfo,
    ScannerHandle,
    ScanSettings,
)
from smart_text_extractor.scanner.service import ScannerService


class _FakeDriver(ScannerDriver):
    """Real threading behavior against a fake device — no mocks for the
    concurrency logic itself, only for the hardware underneath it."""

    def __init__(self, scan_delay: float = 0.0, fail_devices: set[str] | None = None) -> None:
        self.scan_delay = scan_delay
        self.fail_devices = fail_devices or set()
        self.lock = threading.Lock()
        self.currently_scanning = False
        self.overlap_detected = False

    def discover(self) -> list[ScannerDeviceInfo]:
        return [ScannerDeviceInfo(device_id="dev-1", name="Fake")]

    def open(self, device_id: str) -> ScannerHandle:
        if device_id in self.fail_devices:
            raise DeviceDisconnectedError(f"{device_id} is offline")
        return ScannerHandle(device_id=device_id, native_ref=object())

    def capabilities(self, handle: ScannerHandle) -> ScannerCapabilities:
        return ScannerCapabilities(supported_dpi=(300,), supports_color=True, supports_grayscale=True, supports_adf=False)

    def scan(self, handle: ScannerHandle, settings: ScanSettings) -> ScannedImage:
        with self.lock:
            if self.currently_scanning:
                self.overlap_detected = True
            self.currently_scanning = True
        if self.scan_delay:
            time.sleep(self.scan_delay)
        with self.lock:
            self.currently_scanning = False
        return ScannedImage(file_path="scan.bmp", dpi=settings.dpi, width_px=100, height_px=100)  # type: ignore[arg-type]

    def close(self, handle: ScannerHandle) -> None:
        pass


def _wait_for(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


def test_scan_success_calls_on_done_with_image() -> None:
    driver = _FakeDriver()
    worker = ScanWorker(ScannerService(driver=driver))
    worker.start()
    results = []

    worker.submit("dev-1", ScanSettings(dpi=300), lambda image, err: results.append((image, err)))
    _wait_for(lambda: len(results) == 1)
    worker.stop()

    image, err = results[0]
    assert err is None
    assert image.dpi == 300


def test_scan_failure_calls_on_done_with_exception_and_worker_keeps_running() -> None:
    driver = _FakeDriver(fail_devices={"bad-device"})
    worker = ScanWorker(ScannerService(driver=driver))
    worker.start()
    results = []

    worker.submit("bad-device", ScanSettings(dpi=300), lambda image, err: results.append((image, err)))
    _wait_for(lambda: len(results) == 1)

    # the worker thread must have survived the exception — prove it by
    # submitting a second, valid job and confirming it still completes.
    worker.submit("dev-1", ScanSettings(dpi=300), lambda image, err: results.append((image, err)))
    _wait_for(lambda: len(results) == 2)
    worker.stop()

    assert results[0][1] is not None  # first job failed
    assert results[1][1] is None  # second job succeeded — worker thread is still alive


def test_scans_never_overlap_even_when_submitted_back_to_back() -> None:
    driver = _FakeDriver(scan_delay=0.1)
    worker = ScanWorker(ScannerService(driver=driver))
    worker.start()
    results = []

    for _ in range(4):
        worker.submit("dev-1", ScanSettings(dpi=300), lambda image, err: results.append((image, err)))

    _wait_for(lambda: len(results) == 4, timeout=5.0)
    worker.stop()

    assert driver.overlap_detected is False
