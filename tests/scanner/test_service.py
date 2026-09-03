from __future__ import annotations

import pytest

from smart_text_extractor.scanner.base import ScannerDriver
from smart_text_extractor.scanner.errors import UnsupportedPlatformError
from smart_text_extractor.scanner.models import (
    ScannedImage,
    ScannerCapabilities,
    ScannerDeviceInfo,
    ScannerHandle,
    ScanSettings,
)
from smart_text_extractor.scanner.service import ScannerService, _driver_for_current_platform


class _FakeDriver(ScannerDriver):
    """Records calls so tests can assert the facade forwards correctly."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def discover(self) -> list[ScannerDeviceInfo]:
        self.calls.append("discover")
        return [ScannerDeviceInfo(device_id="dev-1", name="Fake Scanner")]

    def open(self, device_id: str) -> ScannerHandle:
        self.calls.append(f"open:{device_id}")
        return ScannerHandle(device_id=device_id, native_ref=object())

    def capabilities(self, handle: ScannerHandle) -> ScannerCapabilities:
        self.calls.append("capabilities")
        return ScannerCapabilities(
            supported_dpi=(300,), supports_color=True, supports_grayscale=True, supports_adf=False
        )

    def scan(self, handle: ScannerHandle, settings: ScanSettings) -> ScannedImage:
        self.calls.append(f"scan:{settings.dpi}")
        return ScannedImage(file_path=__file__, dpi=settings.dpi, width_px=1, height_px=1)  # type: ignore[arg-type]

    def close(self, handle: ScannerHandle) -> None:
        self.calls.append("close")


def test_service_forwards_every_call_to_the_injected_driver() -> None:
    driver = _FakeDriver()
    service = ScannerService(driver=driver)

    devices = service.discover()
    handle = service.open(devices[0].device_id)
    service.capabilities(handle)
    service.scan(handle, ScanSettings(dpi=300))
    service.close(handle)

    assert driver.calls == ["discover", "open:dev-1", "capabilities", "scan:300", "close"]


def test_unsupported_platform_raises_unified_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "PlayStation")

    with pytest.raises(UnsupportedPlatformError):
        _driver_for_current_platform()
