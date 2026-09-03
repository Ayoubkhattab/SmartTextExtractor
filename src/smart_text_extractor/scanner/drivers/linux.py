"""Linux scanner driver: SANE via the `python-sane` package (§4.2).

STATUS: written against python-sane's documented API but NOT VERIFIED —
no Linux machine was available in the session that wrote this (see
docs/phases/phase-0-spike.md). The user has separate access to a Linux
machine and will test this directly there; expect this file to need
correction once real results come back (SANE status strings in
particular — see _translate_sane_error — are matched by substring
because python-sane does not expose a clean status enum in Python, only
the human-readable message from libsane).
"""
from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

from smart_text_extractor.scanner.base import ScannerDriver
from smart_text_extractor.scanner.errors import (
    CoverOpenError,
    DeviceBusyError,
    DeviceDisconnectedError,
    DeviceDiscoveryError,
    DeviceNotFoundError,
    PaperEmptyError,
    PaperJamError,
    ScannerError,
)
from smart_text_extractor.scanner.models import (
    ScannedImage,
    ScannerCapabilities,
    ScannerDeviceInfo,
    ScannerHandle,
    ScanSettings,
)

_COLOR_MODE_TO_SANE_MODE = {
    "color": "Color",
    "grayscale": "Gray",
    "bw": "Lineart",
}

# SANE status messages (as python-sane surfaces them in exception text),
# from sane.h SANE_Status. Matched by substring since python-sane gives no
# machine-readable status code in Python — UNVERIFIED, see module docstring.
_SANE_ERROR_SUBSTRING_MAP: list[tuple[str, type[ScannerError]]] = [
    ("jammed", PaperJamError),
    ("out of documents", PaperEmptyError),
    ("cover is open", CoverOpenError),
    ("device busy", DeviceBusyError),
    ("device has been disconnected", DeviceDisconnectedError),
    ("invalid argument", ScannerError),
]


def _translate_sane_error(exc: Exception) -> ScannerError:
    message = str(exc).lower()
    for substring, mapped in _SANE_ERROR_SUBSTRING_MAP:
        if substring in message:
            return mapped(f"SANE error: {exc}")
    return ScannerError(f"Unrecognized SANE failure: {exc}")


class SaneDriver(ScannerDriver):
    """ScannerDriver implementation for Linux, backed by python-sane."""

    def __init__(self) -> None:
        import sane  # deferred: only installable on Linux against libsane

        self._sane = sane
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            try:
                self._sane.init()
                self._initialized = True
            except Exception as exc:
                raise DeviceDiscoveryError(f"sane.init() failed: {exc}") from exc

    def discover(self) -> list[ScannerDeviceInfo]:
        self._ensure_initialized()
        try:
            devices = self._sane.get_devices()
        except Exception as exc:
            raise _translate_sane_error(exc) from exc
        # Each entry: (device_name, vendor, model, type)
        return [
            ScannerDeviceInfo(device_id=name, name=model or name, manufacturer=vendor)
            for name, vendor, model, _type in devices
        ]

    def open(self, device_id: str) -> ScannerHandle:
        self._ensure_initialized()
        try:
            handle = self._sane.open(device_id)
        except Exception as exc:
            if "invalid argument" in str(exc).lower():
                raise DeviceNotFoundError(f"No SANE device with id {device_id!r}") from exc
            raise _translate_sane_error(exc) from exc
        return ScannerHandle(device_id=device_id, native_ref=handle)

    def capabilities(self, handle: ScannerHandle) -> ScannerCapabilities:
        dev = handle.native_ref
        try:
            dpi_values: tuple[int, ...] = (150, 200, 300, 600)
            try:
                constraint = dev.opt["resolution"].constraint
                if isinstance(constraint, (list, tuple)) and constraint:
                    dpi_values = tuple(sorted(int(v) for v in constraint))
            except Exception:
                pass  # some backends expose a (min, max, step) range instead of a list
            supports_adf = False
            try:
                supports_adf = "ADF" in (dev.opt["source"].constraint or [])
            except Exception:
                pass
            return ScannerCapabilities(
                supported_dpi=dpi_values,
                supports_color=True,
                supports_grayscale=True,
                supports_adf=supports_adf,
            )
        except Exception as exc:
            raise _translate_sane_error(exc) from exc

    def scan(self, handle: ScannerHandle, settings: ScanSettings) -> ScannedImage:
        dev = handle.native_ref
        try:
            dev.resolution = settings.dpi
            dev.mode = _COLOR_MODE_TO_SANE_MODE.get(settings.color_mode, "Color")
            dev.start()
            image = dev.snap()  # returns a PIL.Image

            out_path = Path(gettempdir()) / f"scan_{uuid4().hex}.png"
            image.save(out_path)

            width, height = image.size
            return ScannedImage(file_path=out_path, dpi=settings.dpi, width_px=width, height_px=height)
        except Exception as exc:
            raise _translate_sane_error(exc) from exc

    def close(self, handle: ScannerHandle) -> None:
        dev = handle.native_ref
        try:
            if dev is not None:
                dev.close()
        except Exception:
            pass  # closing a handle should never raise into the caller
        handle.native_ref = None
