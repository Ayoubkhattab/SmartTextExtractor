"""Unit tests for SaneDriver logic against a FAKE python-sane module.

Like test_windows_driver.py, these prove the driver's parsing/translation
logic is correct against the shape of python-sane's documented API — they
do NOT prove it works against a real scanner. Run inside
docker/linux-dev.Dockerfile to at least confirm the code imports and runs
against a real libsane install (still with zero physical scanner attached).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


class _FakeOption:
    def __init__(self, constraint) -> None:
        self.constraint = constraint


class _FakeImage:
    def __init__(self) -> None:
        self.size = (2480, 3508)
        self.saved_to: str | None = None

    def save(self, path) -> None:
        self.saved_to = str(path)
        Path(path).write_bytes(b"fake-png")


class _FakeSaneDevice:
    def __init__(self, name: str) -> None:
        self.name = name
        self.resolution = None
        self.mode = None
        self.started = False
        self.closed = False
        self.opt = {
            "resolution": _FakeOption([150, 200, 300, 600]),
            "source": _FakeOption(["Flatbed", "ADF"]),
        }

    def start(self) -> None:
        self.started = True

    def snap(self) -> _FakeImage:
        return _FakeImage()

    def close(self) -> None:
        self.closed = True


class _FakeSaneError(Exception):
    pass


@pytest.fixture()
def fake_sane(monkeypatch: pytest.MonkeyPatch):
    device = _FakeSaneDevice("device-1")
    devices_list = [("device-1", "Canon", "LiDE 300", "flatbed")]

    fake_module = types.SimpleNamespace(
        init=lambda: None,
        get_devices=lambda: devices_list,
        open=lambda name: device if name == "device-1" else (_ for _ in ()).throw(
            _FakeSaneError("Error: Invalid argument")
        ),
        error=_FakeSaneError,
    )
    monkeypatch.setitem(sys.modules, "sane", fake_module)
    return types.SimpleNamespace(device=device, module=fake_module)


def test_discover_lists_devices(fake_sane) -> None:
    from smart_text_extractor.scanner.drivers.linux import SaneDriver

    driver = SaneDriver()
    devices = driver.discover()

    assert len(devices) == 1
    assert devices[0].device_id == "device-1"
    assert devices[0].name == "LiDE 300"
    assert devices[0].manufacturer == "Canon"


def test_open_unknown_device_raises_device_not_found(fake_sane) -> None:
    from smart_text_extractor.scanner.drivers.linux import SaneDriver
    from smart_text_extractor.scanner.errors import DeviceNotFoundError

    driver = SaneDriver()
    with pytest.raises(DeviceNotFoundError):
        driver.open("does-not-exist")


def test_scan_sets_dpi_and_mode_and_saves_file(fake_sane) -> None:
    from smart_text_extractor.scanner.drivers.linux import SaneDriver
    from smart_text_extractor.scanner.models import ScanSettings

    driver = SaneDriver()
    handle = driver.open("device-1")
    result = driver.scan(handle, ScanSettings(dpi=300, color_mode="color"))

    assert fake_sane.device.resolution == 300
    assert fake_sane.device.mode == "Color"
    assert fake_sane.device.started is True
    assert result.dpi == 300
    assert result.file_path.exists()


def test_capabilities_reads_resolution_constraint_and_adf(fake_sane) -> None:
    from smart_text_extractor.scanner.drivers.linux import SaneDriver

    driver = SaneDriver()
    handle = driver.open("device-1")
    caps = driver.capabilities(handle)

    assert caps.supported_dpi == (150, 200, 300, 600)
    assert caps.supports_adf is True


def test_sane_error_is_translated_to_unified_taxonomy(fake_sane, monkeypatch) -> None:
    from smart_text_extractor.scanner.drivers.linux import SaneDriver
    from smart_text_extractor.scanner.errors import PaperEmptyError
    from smart_text_extractor.scanner.models import ScanSettings

    def _raise_paper_empty():
        raise _FakeSaneError("Error: Document feeder out of documents")

    driver = SaneDriver()
    handle = driver.open("device-1")
    monkeypatch.setattr(fake_sane.device, "start", _raise_paper_empty)

    with pytest.raises(PaperEmptyError):
        driver.scan(handle, ScanSettings(dpi=300))
