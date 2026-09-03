"""ScannerService: the only thing the rest of the app talks to (§4.2).

Selects the right ScannerDriver for the running platform and forwards
every call. No caller outside this module should ever import a concrete
driver directly.
"""
from __future__ import annotations

import platform

from smart_text_extractor.scanner.base import ScannerDriver
from smart_text_extractor.scanner.errors import UnsupportedPlatformError
from smart_text_extractor.scanner.models import (
    ScannedImage,
    ScannerCapabilities,
    ScannerDeviceInfo,
    ScannerHandle,
    ScanSettings,
)


def _driver_for_current_platform() -> ScannerDriver:
    system = platform.system()
    if system == "Windows":
        from smart_text_extractor.scanner.drivers.windows import WiaDriver

        return WiaDriver()
    if system == "Linux":
        from smart_text_extractor.scanner.drivers.linux import SaneDriver

        return SaneDriver()
    if system == "Darwin":
        from smart_text_extractor.scanner.drivers.macos import IcaDriver

        return IcaDriver()
    raise UnsupportedPlatformError(f"No scanner driver registered for platform {system!r}")


class ScannerService:
    """Facade over the active platform driver (Strategy + Facade, §4.2)."""

    def __init__(self, driver: ScannerDriver | None = None) -> None:
        self._driver = driver or _driver_for_current_platform()

    def discover(self) -> list[ScannerDeviceInfo]:
        return self._driver.discover()

    def open(self, device_id: str) -> ScannerHandle:
        return self._driver.open(device_id)

    def capabilities(self, handle: ScannerHandle) -> ScannerCapabilities:
        return self._driver.capabilities(handle)

    def scan(self, handle: ScannerHandle, settings: ScanSettings) -> ScannedImage:
        return self._driver.scan(handle, settings)

    def close(self, handle: ScannerHandle) -> None:
        self._driver.close(handle)
