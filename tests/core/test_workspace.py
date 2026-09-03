from __future__ import annotations

from pathlib import Path

from smart_text_extractor.core.workspace import TempWorkspace


def test_start_creates_an_isolated_session_directory(tmp_path: Path) -> None:
    workspace = TempWorkspace(base_dir=tmp_path / "sessions")
    session_dir = workspace.start()

    assert session_dir.exists()
    assert session_dir.parent == tmp_path / "sessions"
    assert workspace.path == session_dir


def test_cleanup_removes_the_session_directory(tmp_path: Path) -> None:
    workspace = TempWorkspace(base_dir=tmp_path / "sessions")
    session_dir = workspace.start()
    (session_dir / "scan_0001.bmp").write_bytes(b"fake")

    workspace.cleanup()

    assert not session_dir.exists()


def test_cleanup_orphaned_removes_leftovers_from_a_prior_crash(tmp_path: Path) -> None:
    base_dir = tmp_path / "sessions"
    base_dir.mkdir()
    orphan = base_dir / "leftover-from-crash"
    orphan.mkdir()
    (orphan / "scan_0001.bmp").write_bytes(b"fake")

    workspace = TempWorkspace(base_dir=base_dir)
    removed = workspace.cleanup_orphaned()

    assert removed == [orphan]
    assert not orphan.exists()


def test_cleanup_orphaned_is_a_noop_when_base_dir_does_not_exist_yet(tmp_path: Path) -> None:
    workspace = TempWorkspace(base_dir=tmp_path / "never-created")
    assert workspace.cleanup_orphaned() == []


def test_path_before_start_raises() -> None:
    workspace = TempWorkspace()
    try:
        workspace.path
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError before start() is called")
