"""Static enforcement of SYSTEM_ANALYSIS.md §9.1: zero network access.

Not a matter of manual discipline — this test fails the build if any
network-capable module is ever imported from src/, on any code path that
runs during normal operation.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"

_FORBIDDEN_MODULES = {
    "requests",
    "urllib",
    "urllib2",
    "urllib3",
    "http",
    "socket",
    "ftplib",
    "smtplib",
    "telnetlib",
    "xmlrpc",
    "aiohttp",
    "httpx",
    "websockets",
    "websocket",
}


def _iter_python_files() -> list[Path]:
    return sorted(_SRC_ROOT.rglob("*.py"))


def _imported_module_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize(
    "path", _iter_python_files(), ids=lambda p: str(p.relative_to(_SRC_ROOT))
)
def test_no_network_library_imported(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_found = _imported_module_roots(tree) & _FORBIDDEN_MODULES
    assert not forbidden_found, (
        f"{path} imports network module(s) {forbidden_found} — forbidden by "
        f"SYSTEM_ANALYSIS.md §9.1 (zero internet connection)"
    )
