"""Main application window (§14 Phase 1/3 remaining UI tasks).

MVP scope: open image/PDF file(s) -> run through the real OCR pipeline
(OcrEngine, via OcrWorkerPool) -> show extracted text, editable (US-06).
The Scan button is wired to the real ScannerService — with no scanner
currently available to test against, it exercises the same
"discover() returns []" path already validated in the Phase 0 spike.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QIcon, QPixmap, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from smart_text_extractor.concurrency.ocr_worker_pool import OcrWorkerPool
from smart_text_extractor.core.models import Document, OcrResult, OcrStatus, Page, TextSegment
from smart_text_extractor.core.pdf_import import import_pdf_pages
from smart_text_extractor.export.docx_export import PageContent, export_docx
from smart_text_extractor.scanner.service import ScannerService

_TEXT_COLOR = "#1a1d21"

# Confidence-highlighting thresholds (§7.1.1): a word below
# _LOW_CONFIDENCE_THRESHOLD is flagged yellow, below
# _VERY_LOW_CONFIDENCE_THRESHOLD flagged red — lets the user spot-check
# exactly the uncertain spots instead of proofreading the whole page.
# Calibrated against real confidence numbers seen across this session's
# three real test documents: correctly-recognized words clustered 85-96%,
# while confirmed real misreads measured 33-66% (see
# docs/phases/phase-2-ocr-pipeline.md) — 75%/50% sit between those
# clusters without needing to be exact, since this is a review aid for
# the user, not an automatic correction.
_LOW_CONFIDENCE_THRESHOLD = 75.0
_VERY_LOW_CONFIDENCE_THRESHOLD = 50.0
_LOW_CONFIDENCE_COLOR = QColor("#fdf0b5")
_VERY_LOW_CONFIDENCE_COLOR = QColor("#f8c9c9")

_STYLESHEET = f"""
QMainWindow, QWidget {{ background-color: #f4f5f7; color: {_TEXT_COLOR}; font-family: "Segoe UI", "Tahoma", sans-serif; font-size: 13px; }}
QToolBar {{ background-color: #ffffff; border-bottom: 1px solid #dde1e6; padding: 8px; spacing: 8px; }}
QToolButton {{ background-color: #2f6fed; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: 600; }}
QToolButton:hover {{ background-color: #1f5adb; }}
QToolButton:pressed {{ background-color: #17439f; }}
QStatusBar {{ background-color: #ffffff; border-top: 1px solid #dde1e6; color: #5b6472; padding: 4px 8px; }}
QListWidget {{ background-color: #ffffff; color: {_TEXT_COLOR}; border: 1px solid #dde1e6; border-radius: 8px; padding: 4px; outline: none; }}
QListWidget::item {{ border-radius: 6px; padding: 8px; margin: 2px; color: {_TEXT_COLOR}; }}
QListWidget::item:selected {{ background-color: #e8effe; color: #17439f; }}
QListWidget::item:hover:!selected {{ background-color: #f0f2f5; }}
QFrame#card {{ background-color: #ffffff; border: 1px solid #dde1e6; border-radius: 8px; }}
QLabel#cardTitle {{ color: #5b6472; font-weight: 600; padding: 8px 12px; border-bottom: 1px solid #eceff2; }}
QLabel#imagePreview {{ color: #9aa2ad; }}
QTextEdit {{ background-color: #ffffff; color: {_TEXT_COLOR}; border: none; padding: 12px; font-size: 14px; selection-background-color: #b9d0fb; }}
QLabel#confidenceLegend {{ color: #5b6472; font-size: 12px; padding: 6px 12px; border-bottom: 1px solid #eceff2; }}
QSplitter::handle {{ background-color: #f4f5f7; width: 6px; }}
"""


class _OcrBridge(QObject):
    """Bridges OcrWorkerPool's worker-thread callback to the Qt GUI thread.

    pyqtSignal.emit() called from a non-GUI thread is queued automatically
    to the receiver's thread when the receiver lives on it — this is the
    one piece of genuinely PyQt-specific plumbing this window needs; the
    OCR/concurrency logic itself has no Qt dependency at all.
    """

    page_done = pyqtSignal(object, object)


def _card(title: str, content: QWidget) -> QFrame:
    """A titled panel — used for the image preview and the text editor so
    the window reads as distinct sections instead of two bare widgets
    glued together."""
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    title_label = QLabel(title)
    title_label.setObjectName("cardTitle")
    layout.addWidget(title_label)
    layout.addWidget(content, stretch=1)
    return frame


class MainWindow(QMainWindow):
    _THUMBNAIL_SIZE = 56

    def __init__(self, document: Document, ocr_pool: OcrWorkerPool, scanner_service: ScannerService) -> None:
        super().__init__()
        self._document = document
        self._ocr_pool = ocr_pool
        self._scanner_service = scanner_service
        self._suppress_text_changed = False

        self._bridge = _OcrBridge()
        self._bridge.page_done.connect(self._on_page_done)

        self.setWindowTitle("مستخرج النص الذكي — Smart Text Extractor")
        self.resize(1200, 750)
        self.setStyleSheet(_STYLESHEET)

        self._build_toolbar()
        self._build_central_widget()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("جاهز")

    # --- layout --------------------------------------------------------

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("الأدوات")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
        self.addToolBar(toolbar)

        open_action = QAction("افتح صورة أو PDF...", self)
        open_action.triggered.connect(self._on_open_files)
        toolbar.addAction(open_action)

        scan_action = QAction("مسح ضوئي...", self)
        scan_action.triggered.connect(self._on_scan)
        toolbar.addAction(scan_action)

        export_word_action = QAction("تصدير كملف Word...", self)
        export_word_action.triggered.connect(self._on_export_word)
        toolbar.addAction(export_word_action)

        export_markdown_action = QAction("تصدير كـ Markdown...", self)
        export_markdown_action.triggered.connect(self._on_export_markdown)
        toolbar.addAction(export_markdown_action)

    def _build_central_widget(self) -> None:
        self._page_list = QListWidget()
        self._page_list.setIconSize(QSize(self._THUMBNAIL_SIZE, self._THUMBNAIL_SIZE))
        self._page_list.currentRowChanged.connect(self._on_page_selected)
        pages_card = _card("الصفحات", self._page_list)
        pages_card.setMinimumWidth(220)
        pages_card.setMaximumWidth(320)

        self._image_label = QLabel("لا توجد صفحة محددة")
        self._image_label.setObjectName("imagePreview")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumWidth(380)
        image_card = _card("معاينة الصورة", self._image_label)

        self._text_edit = QTextEdit()
        self._text_edit.setPlaceholderText("سيظهر النص المستخرج هنا بعد المعالجة...")
        self._text_edit.textChanged.connect(self._on_text_edited)

        legend = QLabel(
            f'<span style="background-color:{_VERY_LOW_CONFIDENCE_COLOR.name()};">&nbsp;&nbsp;</span>'
            "&nbsp;ثقة منخفضة جداً&nbsp;&nbsp;&nbsp;"
            f'<span style="background-color:{_LOW_CONFIDENCE_COLOR.name()};">&nbsp;&nbsp;</span>'
            "&nbsp;ثقة منخفضة — يُفضَّل مراجعتها"
        )
        legend.setObjectName("confidenceLegend")

        text_panel = QWidget()
        text_panel_layout = QVBoxLayout(text_panel)
        text_panel_layout.setContentsMargins(0, 0, 0, 0)
        text_panel_layout.setSpacing(0)
        text_panel_layout.addWidget(legend)
        text_panel_layout.addWidget(self._text_edit, stretch=1)

        text_card = _card("النص المستخرج", text_panel)

        right_split = QSplitter(Qt.Orientation.Horizontal)
        right_split.addWidget(image_card)
        right_split.addWidget(text_card)
        right_split.setSizes([550, 550])

        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.addWidget(pages_card)
        main_split.addWidget(right_split)
        main_split.setSizes([240, 960])

        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(12, 12, 12, 12)
        central_layout.addWidget(main_split)
        self.setCentralWidget(central)

    # --- actions ---------------------------------------------------------

    def _on_open_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "افتح صورة أو ملف PDF",
            "",
            "الملفات المدعومة (*.png *.jpg *.jpeg *.bmp *.tiff *.pdf);;صور (*.png *.jpg *.jpeg *.bmp *.tiff);;PDF (*.pdf)",
        )
        for path_str in paths:
            self._open_file(Path(path_str))

    def _open_file(self, path: Path) -> None:
        if path.suffix.lower() == ".pdf":
            try:
                page_imports = import_pdf_pages(path, self._document.temp_dir_path)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user, not a crash
                QMessageBox.warning(self, "تعذّرت قراءة الملف", f"تعذّر فتح {path.name}:\n{exc}")
                return
            for page_import in page_imports:
                self._add_page(page_import.image_path, page_import.native_result)
        else:
            self._add_page(path)

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

    def _exportable_pages(self) -> list[Page]:
        # US-08's included_in_range already exists for exactly this: which
        # pages belong in an export. A page whose OCR isn't DONE (still
        # processing, failed, or never run) has no text worth exporting.
        return [
            page
            for page in self._document.pages
            if page.included_in_range and page.ocr_status == OcrStatus.DONE and page.ocr_result is not None
        ]

    def _on_export_markdown(self) -> None:
        exportable_pages = self._exportable_pages()
        if not exportable_pages:
            QMessageBox.information(self, "لا يوجد نص لتصديره", "لا توجد صفحات مكتملة المعالجة لتصديرها.")
            return

        path_str, _ = QFileDialog.getSaveFileName(self, "تصدير كملف Markdown", "", "ملفات Markdown (*.md)")
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() != ".md":
            path = path.with_suffix(".md")

        # A page the user has edited only has edited_text — a plain
        # string with no positional/height data left to detect headings
        # or tables from, so it's included as-is rather than reformatted.
        page_texts = [
            page.ocr_result.edited_text
            if page.ocr_result.edited_text is not None
            else (page.ocr_result.markdown or page.ocr_result.raw_text)
            for page in exportable_pages
        ]

        try:
            path.write_text("\n\n---\n\n".join(page_texts), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "تعذّر الحفظ", f"تعذّر حفظ الملف:\n{exc}")
            return

        self.statusBar().showMessage(f"تم تصدير {len(exportable_pages)} صفحة إلى {path.name}")

    def _on_export_word(self) -> None:
        exportable_pages = self._exportable_pages()
        if not exportable_pages:
            QMessageBox.information(self, "لا يوجد نص لتصديره", "لا توجد صفحات مكتملة المعالجة لتصديرها.")
            return

        path_str, _ = QFileDialog.getSaveFileName(self, "تصدير كملف Word", "", "مستندات Word (*.docx)")
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() != ".docx":
            path = path.with_suffix(".docx")

        # Same edited-vs-structured distinction as Markdown export: a
        # page the user has edited only has edited_text left (a plain
        # string), so it's added as plain paragraphs instead of being
        # reformatted through headings/tables it no longer has positional
        # data for.
        pages: list[PageContent] = [
            page.ocr_result.edited_text if page.ocr_result.edited_text is not None else page.ocr_result.document_units
            for page in exportable_pages
        ]

        try:
            export_docx(pages, path)
        except OSError as exc:
            QMessageBox.warning(self, "تعذّر الحفظ", f"تعذّر حفظ الملف:\n{exc}")
            return

        self.statusBar().showMessage(f"تم تصدير {len(exportable_pages)} صفحة إلى {path.name}")

    def _add_page(self, image_path: Path, native_result: OcrResult | None = None) -> None:
        page = self._document.add_page(image_path)
        item = QListWidgetItem(self._thumbnail_icon(image_path), f"{image_path.name}\nقيد المعالجة")
        self._page_list.addItem(item)
        index = self._page_list.count() - 1
        self._page_list.setCurrentRow(index)

        if native_result is not None:
            # A PDF page with its own real, embedded text layer (§7.1
            # extension) — extracted directly, no OCR ever run on it, so
            # it's DONE immediately instead of going through the pool.
            page.ocr_result = native_result
            page.ocr_status = OcrStatus.DONE
            item.setText(f"{image_path.name}\nتم ✓ (نص أصلي)")
            self.statusBar().showMessage(f"تم استخراج {image_path.name} مباشرة من نص الملف")
            if self._page_list.currentRow() == index:
                self._refresh_detail_panel(page)
            return

        self.statusBar().showMessage(f"تجري معالجة {image_path.name}...")
        self._ocr_pool.submit(page, lambda p, err: self._bridge.page_done.emit(p, err))

    def _thumbnail_icon(self, image_path: Path) -> QIcon:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return QIcon()
        scaled = pixmap.scaled(
            self._THUMBNAIL_SIZE,
            self._THUMBNAIL_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return QIcon(scaled)

    def _on_page_done(self, page: Page, error: Exception | None) -> None:
        index = self._document.pages.index(page)
        item = self._page_list.item(index)
        if error is not None:
            item.setText(f"{page.image_path.name}\nفشل ({error})")
        else:
            item.setText(f"{page.image_path.name}\nتم ✓")
        self.statusBar().showMessage(
            f"اكتملت معالجة {page.image_path.name}" if error is None else f"فشلت معالجة {page.image_path.name}"
        )
        if self._page_list.currentRow() == index:
            self._refresh_detail_panel(page)

    def _on_page_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._document.pages):
            return
        self._refresh_detail_panel(self._document.pages[row])

    def _render_segments(self, segments: list[TextSegment]) -> None:
        """Paints OcrResult.segments into the text editor, giving each
        word a background color keyed to its recognition confidence.
        Concatenating segment.text in order is exactly raw_text (see
        TextSegment's docstring), so this is purely a formatting pass —
        it does not change what toPlainText() returns once the user
        starts editing."""
        self._text_edit.clear()
        cursor = QTextCursor(self._text_edit.document())
        plain_format = QTextCharFormat()
        low_format = QTextCharFormat()
        low_format.setBackground(_LOW_CONFIDENCE_COLOR)
        very_low_format = QTextCharFormat()
        very_low_format.setBackground(_VERY_LOW_CONFIDENCE_COLOR)
        for segment in segments:
            if segment.confidence is None or segment.confidence >= _LOW_CONFIDENCE_THRESHOLD:
                char_format = plain_format
            elif segment.confidence >= _VERY_LOW_CONFIDENCE_THRESHOLD:
                char_format = low_format
            else:
                char_format = very_low_format
            cursor.insertText(segment.text, char_format)

    def _refresh_detail_panel(self, page: Page) -> None:
        pixmap = QPixmap(str(page.image_path))
        if not pixmap.isNull():
            self._image_label.setPixmap(
                pixmap.scaled(
                    max(self._image_label.width() - 24, 1),
                    max(self._image_label.height() - 24, 1),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self._image_label.setText("تعذّر عرض الصورة")

        self._suppress_text_changed = True
        if page.ocr_status == OcrStatus.DONE and page.ocr_result is not None:
            if page.ocr_result.edited_text is not None:
                # Once the user has edited the text, segments/raw_text no
                # longer correspond to what's on screen — show their text
                # plainly rather than highlighting stale confidence data.
                self._text_edit.setPlainText(page.ocr_result.edited_text)
            elif page.ocr_result.segments:
                self._render_segments(page.ocr_result.segments)
            else:
                # segments is only populated by OcrEngine.run() — a
                # caller that builds OcrResult directly (tests, or any
                # future non-OCR text source) still gets its raw_text
                # shown, just without confidence highlighting.
                self._text_edit.setPlainText(page.ocr_result.raw_text)
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
