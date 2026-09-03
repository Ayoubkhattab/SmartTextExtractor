"""Desktop app entry point.

Wires §9.1's temp-workspace lifecycle (sweep orphans at startup, delete on
normal exit) around a MainWindow backed by the real scanner/OCR/concurrency
layers built in Phases 0-3 — this is the first place all of them are
actually connected to something runnable.
"""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from smart_text_extractor.concurrency.ocr_worker_pool import OcrWorkerPool
from smart_text_extractor.core.models import Document, SourceType
from smart_text_extractor.core.workspace import TempWorkspace
from smart_text_extractor.ocr.engine import OcrEngine
from smart_text_extractor.ocr.locate import find_tessdata_dir, find_tesseract_cmd
from smart_text_extractor.scanner.service import ScannerService
from smart_text_extractor.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)

    workspace = TempWorkspace()
    workspace.cleanup_orphaned()  # §9.1: sweep any leftovers from a prior crash
    session_dir = workspace.start()
    app.aboutToQuit.connect(workspace.cleanup)  # §9.1: delete on normal exit

    document = Document(source_type=SourceType.UPLOAD_IMAGE, temp_dir_path=session_dir)
    engine = OcrEngine(tesseract_cmd=find_tesseract_cmd(), tessdata_dir=find_tessdata_dir())
    ocr_pool = OcrWorkerPool(engine)
    scanner_service = ScannerService()

    window = MainWindow(document, ocr_pool, scanner_service)
    window.show()

    exit_code = app.exec()
    ocr_pool.shutdown(wait=False)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
