"""Runs the whole tests/ui/ suite against Qt's offscreen platform plugin so
it needs no real display — works locally and in headless CI alike."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
