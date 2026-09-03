"""Windows scanner driver: WIA (Windows Image Acquisition) via pywin32 (§4.2).

STATUS: implemented against the WIA automation API but NOT YET VERIFIED
against real hardware — no scanner was attached during Phase 0 (see
docs/phases/phase-0-spike.md). In particular the WIA_ERROR_* HRESULT
mapping in _translate_com_error below is transcribed from Microsoft's
wiaerror.h from memory and must be confirmed by actually triggering each
error condition (paper jam, paper empty, cover open, USB unplug) on a
real device before this driver is trusted in production.
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

# WIA device Type value for scanners (as opposed to cameras / streaming video).
_WIA_DEVICE_TYPE_SCANNER = 1

# WIA item property IDs (wiaprop.h) used to drive a scan without showing the
# built-in acquisition UI.
_PROP_HORIZONTAL_RESOLUTION = "6147"
_PROP_VERTICAL_RESOLUTION = "6148"
_PROP_CURRENT_INTENT = "6146"

_INTENT_COLOR = 1
_INTENT_GRAYSCALE = 2
_INTENT_TEXT_BW = 4

_COLOR_MODE_TO_INTENT = {
    "color": _INTENT_COLOR,
    "grayscale": _INTENT_GRAYSCALE,
    "bw": _INTENT_TEXT_BW,
}

# WIA_FORMAT_BMP — uncompressed, simplest transfer format to depend on.
_WIA_FORMAT_BMP = "{B96B3CAB-0728-11D3-9D7B-0000F81EF32E}"

# HRESULT -> exception mapping, transcribed from wiaerror.h.
# UNVERIFIED against real hardware — see module docstring.
_COM_ERROR_MAP: dict[int, type[ScannerError]] = {
    0x80210001: DeviceDisconnectedError,  # WIA_ERROR_GENERAL_ERROR
    0x80210002: PaperJamError,  # WIA_ERROR_PAPER_JAM
    0x80210003: PaperEmptyError,  # WIA_ERROR_PAPER_EMPTY
    0x80210005: DeviceDisconnectedError,  # WIA_ERROR_OFFLINE
    0x80210006: DeviceBusyError,  # WIA_ERROR_BUSY
    0x80210008: CoverOpenError,  # WIA_ERROR_USER_INTERVENTION (commonly: cover open)
    0x8021000A: DeviceDisconnectedError,  # WIA_ERROR_DEVICE_COMMUNICATION
    0x8021000D: DeviceBusyError,  # WIA_ERROR_DEVICE_LOCKED
}


def _translate_com_error(exc: Exception) -> ScannerError:
    """Map a pywin32 com_error to our unified taxonomy; fall back to a generic wrap."""
    hresult = getattr(exc, "hresult", None) or getattr(exc, "args", [None])[0]
    if hresult is not None:
        # pywin32 reports HRESULTs as signed 32-bit ints; WIA docs use unsigned.
        unsigned = hresult & 0xFFFFFFFF
        mapped = _COM_ERROR_MAP.get(unsigned)
        if mapped is not None:
            return mapped(f"WIA error 0x{unsigned:08X}: {exc}")
    return ScannerError(f"Unrecognized WIA failure: {exc}")


class WiaDriver(ScannerDriver):
    """ScannerDriver implementation for Windows, backed by the WIA automation API."""

    def __init__(self) -> None:
        import win32com.client  # deferred: only importable on Windows

        self._win32com = win32com.client

    def _device_manager(self):
        try:
            return self._win32com.Dispatch("WIA.DeviceManager")
        except Exception as exc:  # pywin32 com_error
            raise DeviceDiscoveryError(f"Could not start WIA.DeviceManager: {exc}") from exc

    def discover(self) -> list[ScannerDeviceInfo]:
        manager = self._device_manager()
        result: list[ScannerDeviceInfo] = []
        try:
            count = manager.DeviceInfos.Count
            for i in range(1, count + 1):
                info = manager.DeviceInfos(i)
                if info.Type != _WIA_DEVICE_TYPE_SCANNER:
                    continue
                name = ""
                manufacturer = ""
                for j in range(1, info.Properties.Count + 1):
                    prop = info.Properties(j)
                    if prop.Name == "Name":
                        name = str(prop.Value)
                    elif prop.Name == "Manufacturer":
                        manufacturer = str(prop.Value)
                result.append(
                    ScannerDeviceInfo(
                        device_id=info.DeviceID, name=name or info.DeviceID, manufacturer=manufacturer
                    )
                )
        except Exception as exc:
            raise _translate_com_error(exc) from exc
        return result

    def open(self, device_id: str) -> ScannerHandle:
        manager = self._device_manager()
        try:
            count = manager.DeviceInfos.Count
            for i in range(1, count + 1):
                info = manager.DeviceInfos(i)
                if info.DeviceID == device_id:
                    device = info.Connect()
                    return ScannerHandle(device_id=device_id, native_ref=device)
        except Exception as exc:
            raise _translate_com_error(exc) from exc
        raise DeviceNotFoundError(f"No WIA device with id {device_id!r}")

    def capabilities(self, handle: ScannerHandle) -> ScannerCapabilities:
        device = handle.native_ref
        try:
            item = device.Items(1)
            dpi_values: tuple[int, ...] = (150, 200, 300, 600)
            try:
                res_prop = item.Properties(_PROP_HORIZONTAL_RESOLUTION)
                legal = res_prop.SubTypeValues
                if legal is not None and len(list(legal)) > 0:
                    dpi_values = tuple(sorted(int(v) for v in legal))
            except Exception:
                pass  # driver doesn't expose a discrete legal-value list; keep the default
            return ScannerCapabilities(
                supported_dpi=dpi_values,
                supports_color=True,
                supports_grayscale=True,
                supports_adf=False,  # ADF detection needs WIA_DPS_DOCUMENT_HANDLING_CAPABILITIES; deferred
            )
        except Exception as exc:
            raise _translate_com_error(exc) from exc

    def scan(self, handle: ScannerHandle, settings: ScanSettings) -> ScannedImage:
        device = handle.native_ref
        try:
            item = device.Items(1)
            self._set_property(item, _PROP_HORIZONTAL_RESOLUTION, settings.dpi)
            self._set_property(item, _PROP_VERTICAL_RESOLUTION, settings.dpi)
            intent = _COLOR_MODE_TO_INTENT.get(settings.color_mode, _INTENT_COLOR)
            self._set_property(item, _PROP_CURRENT_INTENT, intent)

            image = item.Transfer(_WIA_FORMAT_BMP)
            out_path = Path(gettempdir()) / f"scan_{uuid4().hex}.bmp"
            image.SaveFile(str(out_path))

            return ScannedImage(
                file_path=out_path,
                dpi=settings.dpi,
                width_px=int(image.Width) if hasattr(image, "Width") else 0,
                height_px=int(image.Height) if hasattr(image, "Height") else 0,
            )
        except Exception as exc:
            raise _translate_com_error(exc) from exc

    def close(self, handle: ScannerHandle) -> None:
        handle.native_ref = None

    @staticmethod
    def _set_property(item, prop_id: str, value) -> None:
        """Best-effort property set — some devices don't support every property."""
        try:
            item.Properties(prop_id).Value = value
        except Exception:
            pass
