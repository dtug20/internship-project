import argparse
import json
import subprocess
import sys
import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


RTSP_URL = "rtsp://192.168.1.198:28537/den33"


def ffprobe_stream(url: str, timeout: int = 15) -> tuple[int, int, float]:
    cmd = [
        "ffprobe",
        "-rtsp_transport", "tcp",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        stream = json.loads(result.stdout)["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        num, den = map(int, stream["r_frame_rate"].split("/"))
        return width, height, num / max(den, 1)
    except Exception as exc:
        print(f"[ffprobe] Failed: {exc}. Using defaults 1920x1080@25fps.")
        return 1920, 1080, 25.0


def start_ffmpeg(url: str, width: int, height: int) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-hwaccel", "videotoolbox",
        "-i", url,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-an",
        "-sn",
        "-loglevel", "error",
        "-",
    ]
    print(f"[FFmpeg] cmd: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=width * height * 3 * 2,
    )


class CaptureThread(QThread):
    frame_ready = pyqtSignal(QImage)
    status = pyqtSignal(str)
    fps_update = pyqtSignal(float)
    stream_info = pyqtSignal(int, int, float)
    stopped = pyqtSignal()

    def __init__(self, source: str, width: int = 0, height: int = 0):
        super().__init__()
        self.source = source
        self.width = width
        self.height = height
        self._running = False
        self._proc = None

    def run(self):
        self._running = True
        width, height = self.width, self.height

        if not width or not height:
            self.status.emit("[FFmpeg] Probing stream...")
            width, height, fps = ffprobe_stream(self.source)
        else:
            fps = 25.0

        self.stream_info.emit(width, height, fps)
        self.status.emit("[FFmpeg] Connecting...")
        self._proc = start_ffmpeg(self.source, width, height)
        frame_size = width * height * 3
        t0, frames = time.time(), 0
        connected = False

        while self._running:
            if not self._proc or not self._proc.stdout:
                break

            raw = self._proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                err = ""
                if self._proc.stderr:
                    try:
                        err = self._proc.stderr.read(1000).decode(errors="replace")
                    except Exception:
                        pass
                self.status.emit(f"[FFmpeg] Stream ended. {err[:120]}")
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (height, width, 3)).copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(
                rgb.data.tobytes(),
                width,
                height,
                width * 3,
                QImage.Format_RGB888,
            ).copy()
            if not connected:
                self.status.emit("[FFmpeg] Connected")
                connected = True
            self.frame_ready.emit(qimg)

            frames += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                self.fps_update.emit(frames / elapsed)
                frames, t0 = 0, time.time()

        self._cleanup()
        self.stopped.emit()

    def stop(self):
        self._running = False
        self._cleanup()
        self.wait(5000)

    def _cleanup(self):
        proc = self._proc
        self._proc = None
        if not proc:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


class MainWindow(QMainWindow):
    def __init__(self, source: str, width: int = 0, height: int = 0):
        super().__init__()
        self.source = source
        self.width = width
        self.height = height
        self.capture_thread = None
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("RTSP Viewer - FFmpeg")
        self.setMinimumSize(900, 600)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        top = QFrame()
        top.setFixedHeight(58)
        top.setStyleSheet("background: #141418; border-bottom: 1px solid #262630;")
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(14, 10, 14, 10)
        top_layout.setSpacing(10)

        self.source_input = QLineEdit(self.source)
        self.source_input.setStyleSheet(
            "QLineEdit { background: #0d0d10; color: #dddddd; "
            "border: 1px solid #343440; border-radius: 4px; padding: 7px; }"
        )

        self.btn = QPushButton("START")
        self.btn.setFixedSize(100, 36)
        self.btn.setStyleSheet(
            "QPushButton { font: bold 12px 'Courier New'; background: #202028; "
            "color: #dddddd; border: 1px solid #3b3b46; border-radius: 4px; }"
            "QPushButton:hover { background: #2a2a34; }"
        )
        self.btn.clicked.connect(self._toggle)

        top_layout.addWidget(QLabel("Source"))
        top_layout.itemAt(0).widget().setStyleSheet("color: #9a9aa3;")
        top_layout.addWidget(self.source_input, 1)
        top_layout.addWidget(self.btn)
        layout.addWidget(top)

        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setStyleSheet("background: #09090c; color: #666670;")
        self.canvas.setText("PRESS START")
        self.canvas.setFont(QFont("Courier New", 20, QFont.Bold))
        layout.addWidget(self.canvas, 1)

        bottom = QFrame()
        bottom.setFixedHeight(44)
        bottom.setStyleSheet("background: #141418; border-top: 1px solid #262630;")
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(14, 0, 14, 0)
        bottom_layout.setSpacing(18)

        self.lbl_status = QLabel("Ready")
        self.lbl_fps = QLabel("FPS --")
        self.lbl_info = QLabel("Resolution --")
        for label in (self.lbl_status, self.lbl_fps, self.lbl_info):
            label.setFont(QFont("Courier New", 10))
            label.setStyleSheet("color: #8f8f99;")

        bottom_layout.addWidget(self.lbl_status, 1)
        bottom_layout.addWidget(self.lbl_info)
        bottom_layout.addWidget(self.lbl_fps)
        layout.addWidget(bottom)

    def _toggle(self):
        if self.capture_thread and self.capture_thread.isRunning():
            self._stop()
        else:
            self._start()

    def _start(self):
        source = self.source_input.text().strip()
        if not source:
            self.lbl_status.setText("Source is empty")
            return

        self.source_input.setEnabled(False)
        self.btn.setText("STOP")
        self.canvas.setText("CONNECTING...")
        self.lbl_status.setText("Starting FFmpeg...")

        self.capture_thread = CaptureThread(source, self.width, self.height)
        self.capture_thread.frame_ready.connect(self._on_frame)
        self.capture_thread.status.connect(self.lbl_status.setText)
        self.capture_thread.fps_update.connect(self._on_fps)
        self.capture_thread.stream_info.connect(self._on_stream_info)
        self.capture_thread.stopped.connect(self._on_stopped)
        self.capture_thread.start()

    def _stop(self):
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread = None
        self._on_stopped()

    @pyqtSlot(QImage)
    def _on_frame(self, qimg):
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.canvas.setPixmap(pixmap)

    @pyqtSlot(float)
    def _on_fps(self, fps):
        color = "#50e660" if fps > 22 else "#f0b429" if fps > 12 else "#e64040"
        self.lbl_fps.setText(f"FPS {fps:4.1f}")
        self.lbl_fps.setStyleSheet(f"color: {color}; font: 10pt 'Courier New';")

    @pyqtSlot(int, int, float)
    def _on_stream_info(self, width, height, fps):
        self.lbl_info.setText(f"{width}x{height} @ {fps:.1f}fps")

    @pyqtSlot()
    def _on_stopped(self):
        self.source_input.setEnabled(True)
        self.btn.setText("START")
        if not self.canvas.pixmap():
            self.canvas.setText("PRESS START")
        self.lbl_status.setText("Stopped")

    def closeEvent(self, event):
        self._stop()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="PyQt5 RTSP viewer using FFmpeg.")
    parser.add_argument("--source", default=RTSP_URL)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(args.source, args.width, args.height)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
