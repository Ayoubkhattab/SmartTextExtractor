"""The one contract every platform driver must satisfy (§4.2)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from smart_text_extractor.scanner.models import (
    ScannedImage,
    ScannerCapabilities,
    ScannerDeviceInfo,
    ScannerHandle,
    ScanSettings,
)


class ScannerDriver(ABC):
    """Strategy interface (§4.2). One concrete subclass per platform.

    Every method must translate platform-native exceptions into the
    scanner.errors taxonomy before they leave the driver — callers of
    ScannerService never catch a platform-specific exception type.
    """

    @abstractmethod
    def discover(self) -> list[ScannerDeviceInfo]:
        """List scanners currently reachable on this machine."""

    @abstractmethod
    def open(self, device_id: str) -> ScannerHandle:
        """Acquire a handle to one device, ready for capabilities()/scan()."""

    @abstractmethod
    def capabilities(self, handle: ScannerHandle) -> ScannerCapabilities:
        """Query what the opened device actually supports."""

    @abstractmethod
    def scan(self, handle: ScannerHandle, settings: ScanSettings) -> ScannedImage:
        """Perform one scan and return the resulting image."""

    @abstractmethod
    def close(self, handle: ScannerHandle) -> None:
        """Release the device handle."""
