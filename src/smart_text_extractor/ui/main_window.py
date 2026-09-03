"""Main application window (§14 Phase 1/3 remaining UI tasks).

MVP scope: open image file(s) -> run through the real OCR pipeline
(OcrEngine, via OcrWorkerPool) -> show extracted text, editable (US-06).
The Scan button is wired to the real ScannerService — with no scanner
currently available to test against, it exercises the same
"discover() returns []" path already validated in the Phase 0 spike.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolBar,
)

from smart_text_extractor.concurrency.ocr_worker_pool import OcrWorkerPool
from smart_text_extractor.core.models import Document, OcrStatus, Page
from smart_text_extractor.scanner.service import ScannerService


class _OcrBridge(QObject):
    """Bridges OcrWorkerPool's worker-thread callback to the Qt GUI thread.

    pyqtSignal.emit() called from a non-GUI thread is queued automatically
    to the receiver's thread when the receiver lives on it — this is the
    one piece of genuinely PyQt-specific plumbing this window needs; the
    OCR/concurrency logic itself has no Qt dependency at all.
    """

    page_done = pyqtSignal(object, object)


class MainWindow(QMainWindow):
    def __init__(self, document: Document, ocr_pool: OcrWorkerPool, scanner_service: ScannerService) -> None:
        super().__init__()
        self._document = document
        self._ocr_pool = ocr_pool
        self._scanner_service = scanner_service
        self._suppress_text_changed = False

        self._bridge = _OcrBridge()
        self._bridge.page_done.connect(self._on_page_done)

        self.setWindowTitle("مستخرج النص الذكي — Smart Text Extractor")
        self.resize(1100, 700)

        self._build_toolbar()
        self._build_central_widget()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("جاهز")

    # --- layout --------------------------------------------------------

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("الأدوات")
        self.addToolBar(toolbar)

        open_action = QAction("افتح صورة...", self)
        open_action.triggered.connect(self._on_open_images)
        toolbar.addAction(open_action)

        scan_action = QAction("مسح ضوئي...", self)
        scan_action.triggered.connect(self._on_scan)
        toolbar.addAction(scan_action)

    def _build_central_widget(self) -> None:
        self._page_list = QListWidget()
        self._page_list.currentRowChanged.connect(self._on_page_selected)

        self._image_label = QLabel("لا توجد صفحة محددة")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumWidth(400)

        self._text_edit = QTextEdit()
        self._text_edit.setPlaceholderText("سيظهر النص المستخرج هنا بعد المعالجة...")
        self._text_edit.textChanged.connect(self._on_text_edited)

        right_split = QSplitter(Qt.Orientation.Horizontal)
        right_split.addWidget(self._image_label)
        right_split.addWidget(self._text_edit)
        right_split.setSizes([500, 500])

        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.addWidget(self._page_list)
        main_split.addWidget(right_split)
        main_split.setSizes([200, 900])

        self.setCentralWidget(main_split)

    # --- actions ---------------------------------------------------------

    def _on_open_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "افتح صورة أو أكثر", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        for path_str in paths:
            self._add_page(Path(path_str))

    def _on_scan(self) -> None:
        devices = self._scanner_service.discover()
        if not devices:
            QMessageBox.information(
                self, "لا يوجد ماسح ضوئي", "لم يتم العثور على أي ماسح ضوئي متصل بهذا الجهاز."
            )
            return
        # Full scan flow (device picker -> ScanWorker.submit -> add_page)
        # needs real hardware to build against meaningfully — deferred
        # until the Phase 0 spike's hardware validation is unblocked.
        QMessageBox.information(
            self, "أجهزة موجودة", f"تم العثور على {len(devices)} جهاز — دعم المسح الكامل قيد الإنجاز."
        )

    def _add_page(self, image_path: Path) -> None:
        page = self._document.add_page(image_path)
        item = QListWidgetItem(f"{len(self._document.pages)}. {image_path.name} — قيد المعالجة")
        self._page_list.addItem(item)
        self._page_list.setCurrentRow(self._page_list.count() - 1)
        self.statusBar().showMessage(f"تجري معالجة {image_path.name}...")
        self._ocr_pool.submit(page, lambda p, err: self._bridge.page_done.emit(p, err))

    def _on_page_done(self, page: Page, error: Exception | None) -> None:
        index = self._document.pages.index(page)
        item = self._page_list.item(index)
        if error is not None:
            item.setText(f"{index + 1}. {page.image_path.name} — فشل ({error})")
            self.statusBar().showMessage(f"فشلت معالجة {page.image_path.name}")
        else:
            item.setText(f"{index + 1}. {page.image_path.name} — تم")
            self.statusBar().showMessage(f"اكتملت معالجة {page.image_path.name}")
        if self._page_list.currentRow() == index:
            self._refresh_detail_panel(page)

    def _on_page_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._document.pages):
            return
        self._refresh_detail_panel(self._document.pages[row])

    def _refresh_detail_panel(self, page: Page) -> None:
        pixmap = QPixmap(str(page.image_path))
        if not pixmap.isNull():
            self._image_label.setPixmap(
                pixmap.scaled(
                    max(self._image_label.width(), 1),
                    max(self._image_label.height(), 1),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self._image_label.setText("تعذّر عرض الصورة")

        self._suppress_text_changed = True
        if page.ocr_status == OcrStatus.DONE and page.ocr_result is not None:
            self._text_edit.setPlainText(page.ocr_result.edited_text or page.ocr_result.raw_text)
            self._text_edit.setReadOnly(False)
        elif page.ocr_status == OcrStatus.FAILED:
            self._text_edit.setPlainText("")
            self._text_edit.setReadOnly(True)
        else:
            self._text_edit.setPlainText("... جارٍ الاستخلاص")
            self._text_edit.setReadOnly(True)
        self._suppress_text_changed = False

    def _on_text_edited(self) -> None:
        if self._suppress_text_changed:
            return
        row = self._page_list.currentRow()
        if row < 0:
            return
        page = self._document.pages[row]
        if page.ocr_status == OcrStatus.DONE:
            page.set_edited_text(self._text_edit.toPlainText())
