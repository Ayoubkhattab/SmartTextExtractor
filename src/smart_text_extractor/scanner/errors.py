"""Scanner-specific errors, all mapping onto the app-wide taxonomy (core/errors.py).

Each driver is responsible for catching its own platform-native exception
(WIA HRESULT, SANE status code, ICA NSError, ...) and raising one of these
instead — the rest of the app never sees a platform-specific error type.
"""
from __future__ import annotations

from smart_text_extractor.core.errors import DriverError, UserFixableHardwareError


class ScannerError(Exception):
    """Base class for every scanner-layer error."""


class DeviceDiscoveryError(ScannerError, DriverError):
    """Enumerating devices failed (service down, permissions, ...)."""


class DeviceNotFoundError(ScannerError, DriverError):
    """open() called with a device_id that no longer exists."""


class DeviceBusyError(ScannerError, UserFixableHardwareError):
    """Scanner is in use by another application (§2.2)."""


class DeviceDisconnectedError(ScannerError, UserFixableHardwareError):
    """Scanner went offline / USB unplugged mid-operation."""


class PaperJamError(ScannerError, UserFixableHardwareError):
    pass


class PaperEmptyError(ScannerError, UserFixableHardwareError):
    """ADF ran out of paper mid-batch (§3.1)."""


class CoverOpenError(ScannerError, UserFixableHardwareError):
    pass


class DriverConflictError(ScannerError, DriverError):
    """Two scanners from the same vendor register conflicting drivers (§3.1)."""


class UnsupportedPlatformError(ScannerError, DriverError):
    """No driver is registered for the current OS."""
