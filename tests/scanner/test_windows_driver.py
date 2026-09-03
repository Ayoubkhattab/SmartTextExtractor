"""Unit tests for WiaDriver logic against a FAKE WIA COM layer.

IMPORTANT: these tests prove the driver's parsing/translation logic is
correct against the shape of the WIA API as documented — they do NOT
prove the driver works against a real scanner, because no scanner was
available in this environment (see docs/phases/phase-0-spike.md). Real
hardware validation is a separate, still-open task.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="WIA is Windows-only")


class _FakeCollection:
    """Mimics a 1-based COM collection: obj.Count and obj(i)."""

    def __init__(self, items: list) -> None:
        self._items = items

    @property
    def Count(self) -> int:  # noqa: N802 - COM naming convention
        return len(self._items)

    def __call__(self, index: int):
        return self._items[index - 1]


class _FakeProperty:
    def __init__(self, name: str, value) -> None:
        self.Name = name
        self.Value = value
        self.SubTypeValues = []


class _FakeDeviceInfo:
    def __init__(self, device_id: str, name: str, manufacturer: str, device) -> None:
        self.Type = 1  # scanner
        self.DeviceID = device_id
        self.Properties = _FakeCollection(
            [_FakeProperty("Name", name), _FakeProperty("Manufacturer", manufacturer)]
        )
        self._device = device

    def Connect(self):  # noqa: N802
        return self._device


class _FakeImage:
    def __init__(self) -> None:
        self.Width = 2480
        self.Height = 3508
        self.saved_to: str | None = None

    def SaveFile(self, path: str) -> None:  # noqa: N802
        self.saved_to = path
        Path(path).write_bytes(b"fake-bmp")


class _FakeItem:
    def __init__(self) -> None:
        self._props: dict[str, _FakeProperty] = {}
        self.transferred_format: str | None = None

    def Properties(self, prop_id: str):  # noqa: N802
        return self._props.setdefault(prop_id, _FakeProperty(prop_id, None))

    def Transfer(self, format_id: str):  # noqa: N802
        self.transferred_format = format_id
        return _FakeImage()


class _FakeDevice:
    def __init__(self) -> None:
        self.item = _FakeItem()

    def Items(self, index: int):  # noqa: N802
        assert index == 1
        return self.item


@pytest.fixture()
def fake_win32com(monkeypatch: pytest.MonkeyPatch):
    device = _FakeDevice()
    device_info = _FakeDeviceInfo("dev-123", "Canon LiDE 300", "Canon", device)
    manager = types.SimpleNamespace(DeviceInfos=_FakeCollection([device_info]))

    fake_module = types.SimpleNamespace(Dispatch=MagicMock(return_value=manager))
    monkeypatch.setitem(sys.modules, "win32com.client", fake_module)
    monkeypatch.setitem(sys.modules, "win32com", types.SimpleNamespace(client=fake_module))
    return types.SimpleNamespace(manager=manager, device=device, device_info=device_info)


def test_discover_lists_scanner_devices_only(fake_win32com) -> None:
    from smart_text_extractor.scanner.drivers.windows import WiaDriver

    driver = WiaDriver()
    devices = driver.discover()

    assert len(devices) == 1
    assert devices[0].device_id == "dev-123"
    assert devices[0].name == "Canon LiDE 300"
    assert devices[0].manufacturer == "Canon"


def test_open_returns_handle_wrapping_the_connected_device(fake_win32com) -> None:
    from smart_text_extractor.scanner.drivers.windows import WiaDriver

    driver = WiaDriver()
    handle = driver.open("dev-123")

    assert handle.device_id == "dev-123"
    assert handle.native_ref is fake_win32com.device


def test_open_unknown_device_id_raises_device_not_found(fake_win32com) -> None:
    from smart_text_extractor.scanner.drivers.windows import WiaDriver
    from smart_text_extractor.scanner.errors import DeviceNotFoundError

    driver = WiaDriver()
    with pytest.raises(DeviceNotFoundError):
        driver.open("does-not-exist")


def test_scan_sets_requested_dpi_and_saves_a_file(fake_win32com, tmp_path) -> None:
    from smart_text_extractor.scanner.drivers.windows import WiaDriver
    from smart_text_extractor.scanner.models import ScanSettings

    driver = WiaDriver()
    handle = driver.open("dev-123")
    result = driver.scan(handle, ScanSettings(dpi=300, color_mode="color"))

    item = fake_win32com.device.item
    assert item.Properties("6147").Value == 300  # horizontal resolution
    assert item.Properties("6148").Value == 300  # vertical resolution
    assert result.dpi == 300
    assert result.file_path.exists()


def test_com_error_is_translated_to_unified_taxonomy(fake_win32com, monkeypatch) -> None:
    from smart_text_extractor.scanner.drivers.windows import WiaDriver
    from smart_text_extractor.scanner.errors import PaperEmptyError

    class _FakeComError(Exception):
        hresult = 0x80210003 - 0x100000000  # WIA_ERROR_PAPER_EMPTY, as pywin32 signs it

    def _raise(*_args, **_kwargs):
        raise _FakeComError("paper empty")

    driver = WiaDriver()
    handle = driver.open("dev-123")
    monkeypatch.setattr(fake_win32com.device.item, "Transfer", _raise)

    from smart_text_extractor.scanner.models import ScanSettings

    with pytest.raises(PaperEmptyError):
        driver.scan(handle, ScanSettings(dpi=300))
