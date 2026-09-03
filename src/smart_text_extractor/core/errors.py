"""Unified error taxonomy shared by every subsystem (SYSTEM_ANALYSIS.md §8.1).

Each subsystem (scanner, OCR, export, ...) translates its own native
exceptions into one of these categories so the UI layer only ever has to
handle four cases, never a driver-specific exception type.
"""
from __future__ import annotations


class AppError(Exception):
    """Base class for every error the application raises deliberately."""


class UserFixableHardwareError(AppError):
    """Recoverable by the user immediately (paper jam, cover open, USB unplugged)."""


class DriverError(AppError):
    """Driver/OS-level fault the app cannot resolve on its own (§8.1)."""


class InternalProcessingError(AppError):
    """Internal failure isolated to one unit of work (one page, one task)."""


class FileSystemError(AppError):
    """Disk full, target file locked, etc. (§8.1)."""
