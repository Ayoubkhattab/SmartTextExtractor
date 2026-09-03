"""Data contracts shared by ScannerService and every driver (§4.2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ScannerDeviceInfo:
    """One discoverable scanner, as reported by ScannerService.discover()."""

    device_id: str
    name: str
    manufacturer: str = ""


@dataclass(frozen=True)
class ScannerCapabilities:
    """What a given device supports, read via ScannerService.capabilities()."""

    supported_dpi: tuple[int, ...]
    supports_color: bool
    supports_grayscale: bool
    supports_adf: bool


@dataclass(frozen=True)
class ScanSettings:
    """Requested acquisition parameters for one ScannerService.scan() call."""

    dpi: int = 300
    color_mode: str = "color"  # "color" | "grayscale" | "bw"


@dataclass(frozen=True)
class ScannedImage:
    """Result of a successful scan — a temp file plus the DPI it was captured at.

    dpi is carried explicitly rather than re-read from the image file later,
    per the §7.3 decision: original DPI must be stored, never assumed, so
    that word-box coordinates stay aligned with the source image at export.
    """

    file_path: Path
    dpi: int
    width_px: int
    height_px: int


@dataclass
class ScannerHandle:
    """Opaque handle returned by open(); passed back into capabilities()/scan()/close().

    native_ref holds whatever the underlying driver needs (a COM object for
    WIA, a device handle for SANE, ...) — opaque to everything above the
    driver layer.
    """

    device_id: str
    native_ref: object = field(repr=False)
