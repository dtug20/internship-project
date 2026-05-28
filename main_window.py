"""
Main application window: PyQt5 UI with start/stop controls,
FPS display, plate count, and a plate log panel on the right.
"""

import queue
from datetime import datetime

from PyQt5.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
                              QWidget, QPushButton, QFrame, QSizePolicy,
                              QListWidget, QListWidgetItem, QSplitter)
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QPixmap, QFont, QImage, QColor

from tracker import PlateTracker
from capture_thread import CaptureThread
from detection_thread import DetectionThread
from ocr_worker import OCRWorker
from detector import DEFAULT_VEHICLE_MODEL, DEFAULT_PLATE_MODEL


# ── Styles ──────────────────────────────────────────────────────────────
DARK_BG = "#0c0c0f"
PANEL_BG = "#111116"
BORDER_COLOR = "#222"
TEXT_PRIMARY = "#e0e0e0"
TEXT_SECONDARY = "#888"
ACCENT_GREEN = "#50e660"
ACCENT_ORANGE = "#f0b429"

LOG_PANEL_STYLE = f"""
    QListWidget {{
        background: {PANEL_BG};
        border: none;
        border-left: 1px solid {BORDER_COLOR};
        color: {TEXT_PRIMARY};
        font: 11px 'Courier New';
        padding: 4px;
    }}
    QListWidget::item {{
        padding: 6px 8px;
        border-bottom: 1px solid #1a1a20;
    }}
    QListWidget::item:selected {{
        background: #1e1e2a;
    }}
"""

LOG_HEADER_STYLE = f"""
    background: #16161c;
    color: {TEXT_SECONDARY};
    font: bold 11px 'Courier New';
    padding: 8px 10px;
    border-left: 1px solid {BORDER_COLOR};
    border-bottom: 1px solid {BORDER_COLOR};
"""


class MainWindow(QMainWindow):
    def __init__(self, source, width, height, vehicle_model, plate_model,
                 yolo_conf, plate_conf, yolo_imgsz):
        super().__init__()
        self.source = source
        self.width = width
        self.height = height
        self.vehicle_model = vehicle_model
        self.plate_model = plate_model
        self.yolo_conf = yolo_conf
        self.plate_conf = plate_conf
        self.yolo_imgsz = yolo_imgsz
        self.tracker = PlateTracker()
        self.frame_queue = queue.Queue(maxsize=4)
        self.ocr_queue = queue.Queue(maxsize=4)
        self.capture_thread = None
        self.detect_thread = None
        self.ocr_worker = None
        self._plates = []
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("Vehicle YOLO Detector")
        self.setMinimumSize(1100, 600)
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Top area: video + log panel ─────────────────────────────
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Video canvas (left, stretches)
        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setStyleSheet(f"background: {DARK_BG}; color: #666;")
        self.canvas.setText("● PRESS START")
        self.canvas.setFont(QFont("Courier New", 20, QFont.Bold))
        content_layout.addWidget(self.canvas, 3)

        # Log panel (right sidebar)
        log_container = QWidget()
        log_container.setFixedWidth(280)
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(0)

        # Log header
        log_header = QLabel("📋  DETECTED PLATES")
        log_header.setStyleSheet(LOG_HEADER_STYLE)
        log_layout.addWidget(log_header)

        # Plate list
        self.plate_list = QListWidget()
        self.plate_list.setStyleSheet(LOG_PANEL_STYLE)
        self.plate_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plate_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        log_layout.addWidget(self.plate_list, 1)

        # Log footer with count
        self.log_footer = QLabel("Total: 0")
        self.log_footer.setStyleSheet(
            f"background: #16161c; color: {ACCENT_GREEN}; "
            f"font: bold 12px 'Courier New'; padding: 8px 10px; "
            f"border-left: 1px solid {BORDER_COLOR}; "
            f"border-top: 1px solid {BORDER_COLOR};")
        log_layout.addWidget(self.log_footer)

        content_layout.addWidget(log_container)
        main_layout.addLayout(content_layout, 1)

        # ── Bottom bar: controls ────────────────────────────────────
        bar = QFrame()
        bar.setFixedHeight(50)
        bar.setStyleSheet(
            f"background: #131318; border-top: 1px solid {BORDER_COLOR};")
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(14, 0, 14, 0)
        hbox.setSpacing(12)

        self.btn = QPushButton("START")
        self.btn.setFixedSize(100, 34)
        self.btn.setStyleSheet(
            "font: bold 12px 'Courier New'; background: #1e1e26; "
            "color: #ccc; border: 1px solid #333; border-radius: 5px;")
        self.btn.clicked.connect(self._toggle)

        btn_clear = QPushButton("CLEAR")
        btn_clear.setFixedSize(80, 34)
        btn_clear.setStyleSheet(
            "font: bold 12px 'Courier New'; background: #1e1e26; "
            "color: #e64040; border: 1px solid #333; border-radius: 5px;")
        btn_clear.clicked.connect(self._clear_log)

        self.lbl_fps = QLabel("FPS —")
        self.lbl_ocr = QLabel("OCR: loading…")
        self.lbl_total = QLabel("Total: 0")
        self.lbl_info = QLabel(f"GStreamer | {self.width}x{self.height}")
        for lbl in (self.lbl_fps, self.lbl_ocr, self.lbl_total, self.lbl_info):
            lbl.setFont(QFont("Courier New", 10))
            lbl.setStyleSheet(f"color: {TEXT_SECONDARY};")

        hbox.addWidget(self.btn)
        hbox.addWidget(btn_clear)
        hbox.addSpacing(8)
        hbox.addWidget(self.lbl_fps)
        hbox.addWidget(self.lbl_info)
        hbox.addWidget(self.lbl_ocr)
        hbox.addStretch()
        hbox.addWidget(self.lbl_total)
        main_layout.addWidget(bar)

    def _toggle(self):
        if self.capture_thread and self.capture_thread.isRunning():
            self._stop()
        else:
            self._start()

    def _start(self):
        self.tracker = PlateTracker()
        self._plates.clear()
        self.plate_list.clear()
        self.log_footer.setText("Total: 0")
        # Clear queues
        for q in (self.frame_queue, self.ocr_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Thread 3: OCR
        self.ocr_worker = OCRWorker(self.ocr_queue)
        self.ocr_worker.result.connect(self._on_ocr)
        self.ocr_worker.ready.connect(
            lambda ok: self.lbl_ocr.setText(
                "OCR ✓" if ok else "OCR demo"))
        self.ocr_worker.start()

        # Thread 2: Detection
        self.detect_thread = DetectionThread(
            self.frame_queue, self.tracker, self.ocr_queue,
            self.vehicle_model, self.plate_model, self.yolo_conf,
            self.plate_conf, self.yolo_imgsz)
        self.detect_thread.frame_ready.connect(self._on_frame)
        self.detect_thread.fps_update.connect(self._on_fps)
        self.detect_thread.start()

        # Thread 1: Capture
        self.capture_thread = CaptureThread(
            self.source, self.frame_queue, self.width, self.height)
        self.capture_thread.status.connect(
            lambda m: self.lbl_ocr.setText(m))
        self.capture_thread.start()

        self.btn.setText("STOP")

    def _stop(self):
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread = None
        if self.detect_thread:
            self.detect_thread.stop()
            self.detect_thread = None
        if self.ocr_worker:
            self.ocr_worker.stop()
            self.ocr_worker = None
        self.btn.setText("START")
        self.canvas.setText("● PRESS START")

    @pyqtSlot(QImage)
    def _on_frame(self, qimg):
        self.canvas.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.canvas.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation))

    @pyqtSlot(int, str, float)
    def _on_ocr(self, tid, text, conf):
        self.tracker.set_text(tid, text, conf)
        vehicle_label = self.tracker.get_vehicle_label(tid)

        # Add to log if not duplicate
        if text not in self._plates:
            self._plates.append(text)
            timestamp = datetime.now().strftime("%H:%M:%S")
            conf_pct = int(conf * 100)
            vtype = vehicle_label.upper() if vehicle_label else "—"

            # Create styled list item
            item = QListWidgetItem()
            item.setText(
                f"  {timestamp}  │ {vtype:>10}  │  {text}  │ {conf_pct}%")

            # Color by confidence
            if conf_pct >= 80:
                item.setForeground(QColor(ACCENT_GREEN))
            elif conf_pct >= 50:
                item.setForeground(QColor(ACCENT_ORANGE))
            else:
                item.setForeground(QColor("#e64040"))

            self.plate_list.insertItem(0, item)  # newest on top
            self.plate_list.scrollToTop()

            # Update counters
            count = len(self._plates)
            self.log_footer.setText(f"Total: {count}")
            self.lbl_total.setText(f"Total: {count}")

        self.lbl_ocr.setText(f"Last: {text} ({int(conf*100)}%)")

    def _clear_log(self):
        self._plates.clear()
        self.plate_list.clear()
        self.log_footer.setText("Total: 0")
        self.lbl_total.setText("Total: 0")
    @pyqtSlot(float)
    def _on_fps(self, fps):
        if fps > 22:
            c = ACCENT_GREEN
        elif fps > 12:
            c = ACCENT_ORANGE
        else:
            c = "#e64040"
        self.lbl_fps.setText(f"FPS {fps:4.1f}")
        self.lbl_fps.setStyleSheet(f"color: {c};")

    def closeEvent(self, e):
        self._stop()
        e.accept()
