"""macOS scanner driver: ImageCaptureCore via pyobjc (§4.2).

STATUS: NOT IMPLEMENTED — same blocker as linux.py: no macOS machine is
available in this development environment. pyobjc-framework-imagecapturecore
only installs and runs on macOS, so this cannot even be stubbed against a
real API surface from here, let alone tested.
"""
from __future__ import annotations

from smart_text_extractor.scanner.base import ScannerDriver
from smart_text_extractor.scanner.models import (
    ScannedImage,
    ScannerCapabilities,
    ScannerDeviceInfo,
    ScannerHandle,
    ScanSettings,
)


class IcaDriver(ScannerDriver):
    def discover(self) -> list[ScannerDeviceInfo]:
        raise NotImplementedError("IcaDriver is not implemented yet — see module docstring")

    def open(self, device_id: str) -> ScannerHandle:
        raise NotImplementedError("IcaDriver is not implemented yet — see module docstring")

    def capabilities(self, handle: ScannerHandle) -> ScannerCapabilities:
        raise NotImplementedError("IcaDriver is not implemented yet — see module docstring")

    def scan(self, handle: ScannerHandle, settings: ScanSettings) -> ScannedImage:
        raise NotImplementedError("IcaDriver is not implemented yet — see module docstring")

    def close(self, handle: ScannerHandle) -> None:
        raise NotImplementedError("IcaDriver is not implemented yet — see module docstring")
