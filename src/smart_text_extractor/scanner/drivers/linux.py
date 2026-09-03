"""Linux scanner driver: SANE via python-sane (§4.2).

STATUS: NOT IMPLEMENTED. This module only exists so ScannerService can
import a Linux entry without a hard crash; there is no Linux machine
available in the environment this project is currently developed in, so
this driver has not been written or tested against real SANE hardware.
Writing it requires a Linux environment with python-sane installed
against an actual libsane installation — that is a hard blocker for
completing Phase 0's cross-platform spike (see
docs/phases/phase-0-spike.md) and needs either a Linux dev machine or a
CI runner with real (or SANE test-backend) hardware.
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


class SaneDriver(ScannerDriver):
    def discover(self) -> list[ScannerDeviceInfo]:
        raise NotImplementedError("SaneDriver is not implemented yet — see module docstring")

    def open(self, device_id: str) -> ScannerHandle:
        raise NotImplementedError("SaneDriver is not implemented yet — see module docstring")

    def capabilities(self, handle: ScannerHandle) -> ScannerCapabilities:
        raise NotImplementedError("SaneDriver is not implemented yet — see module docstring")

    def scan(self, handle: ScannerHandle, settings: ScanSettings) -> ScannedImage:
        raise NotImplementedError("SaneDriver is not implemented yet — see module docstring")

    def close(self, handle: ScannerHandle) -> None:
        raise NotImplementedError("SaneDriver is not implemented yet — see module docstring")
