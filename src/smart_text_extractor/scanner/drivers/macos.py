"""macOS scanner driver: ImageCaptureCore via pyobjc (§4.2).

STATUS: HIGH UNCERTAINTY. Written entirely from documented
ImageCaptureCore concepts with ZERO ability to verify against the real
API — no macOS environment was available to the session that wrote
this, not even to import the framework and check a symbol exists (see
docs/phases/phase-0-spike.md).

ImageCaptureCore is delegate-based and asynchronous (NSRunLoop-driven),
a fundamentally different model from WIA/SANE's synchronous calls. This
file adapts it to the synchronous ScannerDriver contract by pumping the
run loop with a timeout (_pump_run_loop_until) — a pattern used in real
pyobjc scripts, but never exercised here even once.

Treat this as a starting sketch to correct on real hardware, not a
finished driver. Most likely wrong: exact delegate method selector
names/signatures, functional-unit selection, and the scanned-file URL
handoff in scan().
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

from smart_text_extractor.scanner.base import ScannerDriver
from smart_text_extractor.scanner.errors import (
    DeviceDisconnectedError,
    DeviceNotFoundError,
    ScannerError,
)
from smart_text_extractor.scanner.models import (
    ScannedImage,
    ScannerCapabilities,
    ScannerDeviceInfo,
    ScannerHandle,
    ScanSettings,
)

_DISCOVERY_TIMEOUT_SECONDS = 5.0
_OPEN_SESSION_TIMEOUT_SECONDS = 5.0
_SCAN_TIMEOUT_SECONDS = 120.0


def _pump_run_loop_until(condition, timeout_seconds: float) -> bool:
    from Foundation import NSDate, NSRunLoop

    deadline = time.monotonic() + timeout_seconds
    run_loop = NSRunLoop.currentRunLoop()
    while not condition() and time.monotonic() < deadline:
        run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
    return condition()


class _BrowserDelegate:
    """ICDeviceBrowserDelegate: collects devices as ICDeviceBrowser announces them."""

    def __init__(self) -> None:
        self.devices: list = []

    def deviceBrowser_didAddDevice_moreComing_(self, browser, device, more_coming):
        self.devices.append(device)

    def deviceBrowser_didRemoveDevice_moreComing_(self, browser, device, more_coming):
        if device in self.devices:
            self.devices.remove(device)


class _ScannerDeviceDelegate:
    """ICScannerDeviceDelegate: session-open and scan-completion callbacks."""

    def __init__(self) -> None:
        self.session_opened = False
        self.session_error = None
        self.scan_finished = False
        self.scan_error = None
        self.scanned_url = None

    def device_didOpenSessionWithError_(self, device, error):
        self.session_opened = error is None
        self.session_error = error

    def scannerDevice_didScanToURL_error_(self, device, url, error):
        self.scanned_url = url
        self.scan_error = error
        self.scan_finished = True


class IcaDriver(ScannerDriver):
    def __init__(self) -> None:
        import ImageCaptureCore  # deferred: only installable on macOS

        self._icc = ImageCaptureCore
        self._last_discovered: dict[str, object] = {}

    def discover(self) -> list[ScannerDeviceInfo]:
        delegate = _BrowserDelegate()
        browser = self._icc.ICDeviceBrowser.alloc().init()
        browser.setDelegate_(delegate)
        browser.setBrowsedDeviceTypeMask_(self._icc.ICDeviceTypeMaskScanner)
        browser.start()
        _pump_run_loop_until(lambda: len(delegate.devices) > 0, _DISCOVERY_TIMEOUT_SECONDS)
        browser.stop()

        self._last_discovered = {str(d.UUIDString()): d for d in delegate.devices}
        return [
            ScannerDeviceInfo(
                device_id=device_id,
                name=str(device.name()),
                manufacturer=str(device.usbVendorName()) if hasattr(device, "usbVendorName") else "",
            )
            for device_id, device in self._last_discovered.items()
        ]

    def open(self, device_id: str) -> ScannerHandle:
        device = self._last_discovered.get(device_id)
        if device is None:
            raise DeviceNotFoundError(f"No device with id {device_id!r} — call discover() again first")

        delegate = _ScannerDeviceDelegate()
        device.setDelegate_(delegate)
        device.requestOpenSession()
        _pump_run_loop_until(
            lambda: delegate.session_opened or delegate.session_error is not None,
            _OPEN_SESSION_TIMEOUT_SECONDS,
        )
        if not delegate.session_opened:
            raise DeviceDisconnectedError(f"Could not open session: {delegate.session_error}")
        return ScannerHandle(device_id=device_id, native_ref=(device, delegate))

    def capabilities(self, handle: ScannerHandle) -> ScannerCapabilities:
        device, _delegate = handle.native_ref
        try:
            available_units = device.availableFunctionalUnitTypes() or []
            supports_adf = getattr(self._icc, "ICScannerFunctionalUnitTypeDocumentFeeder", -1) in available_units
            return ScannerCapabilities(
                supported_dpi=(150, 200, 300, 600),
                supports_color=True,
                supports_grayscale=True,
                supports_adf=bool(supports_adf),
            )
        except Exception as exc:
            raise ScannerError(f"Could not read capabilities: {exc}") from exc

    def scan(self, handle: ScannerHandle, settings: ScanSettings) -> ScannedImage:
        device, delegate = handle.native_ref
        try:
            unit = device.selectedFunctionalUnit()
            unit.setResolution_(settings.dpi)

            delegate.scan_finished = False
            delegate.scan_error = None
            delegate.scanned_url = None
            device.requestScan()
            _pump_run_loop_until(lambda: delegate.scan_finished, _SCAN_TIMEOUT_SECONDS)

            if not delegate.scan_finished:
                raise ScannerError("Scan timed out waiting for scannerDevice:didScanToURL:error:")
            if delegate.scan_error is not None:
                raise ScannerError(f"Scan failed: {delegate.scan_error}")

            out_path = Path(gettempdir()) / f"scan_{uuid4().hex}.tiff"
            shutil.copyfile(str(delegate.scanned_url.path()), out_path)
            # Pixel dimensions need reading back from the saved file (e.g. via
            # Pillow) once Pre-processing (Phase 2) is available — left at 0
            # here rather than guessed.
            return ScannedImage(file_path=out_path, dpi=settings.dpi, width_px=0, height_px=0)
        except ScannerError:
            raise
        except Exception as exc:
            raise ScannerError(f"Scan failed: {exc}") from exc

    def close(self, handle: ScannerHandle) -> None:
        device, _delegate = handle.native_ref
        try:
            device.requestCloseSession()
        except Exception:
            pass  # closing a handle should never raise into the caller
        handle.native_ref = None
