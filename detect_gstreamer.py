"""
Uses gst-launch-1.0 subprocess to decode RTSP → raw BGR frames piped
to stdout via filesink.  No Python GI / PyGObject bindings required.

"""

import sys, os, subprocess, json
from pathlib import Path
import cv2, numpy as np, queue, threading, time, argparse, re, copy

from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel,
                              QVBoxLayout, QHBoxLayout, QWidget,
                              QPushButton, QFrame, QSizePolicy)
from PyQt5.QtCore  import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui   import QImage, QPixmap, QFont, QColor


# ──────────────────────────────────────────────────────────────────────────
#  GSTREAMER HELPERS
# ──────────────────────────────────────────────────────────────────────────

def gst_discover(url: str, timeout: int = 15) -> tuple[int, int, float]:
    """Probe stream dimensions using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-rtsp_transport', 'tcp',
            '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate',
            '-of', 'json', url
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        info = json.loads(r.stdout)
        s = info['streams'][0]
        w, h = int(s['width']), int(s['height'])
        num, den = map(int, s['r_frame_rate'].split('/'))
        fps = num / max(den, 1)
        return w, h, fps
    except Exception as e:
        print(f"[probe] ffprobe failed: {e}. Using defaults 1920x1080@25fps.")
        return 1920, 1080, 25.0


def start_gstreamer(url: str, width: int, height: int) -> subprocess.Popen:
    """
    Start gst-launch-1.0 pipeline that pipes raw BGR24 frames to stdout.
    Each element/property is a separate arg for gst-launch-1.0.
    """
    cmd = [
        'gst-launch-1.0', '-q', '-e',
        'rtspsrc', f'location={url}', 'latency=100', 'protocols=tcp', '!',
        'rtph264depay', '!',
        'h264parse', '!',
        'vtdec', '!',
        'videoconvert', '!',
        'video/x-raw,format=BGR', '!',
        'filesink', 'location=/dev/stdout', 'sync=false'
    ]
    print(f"[GStreamer] cmd: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=width * height * 3 * 2
    )


# ──────────────────────────────────────────────────────────────────────────
#  PLATE TRACKER
# ──────────────────────────────────────────────────────────────────────────

class Track:
    _id = 0
    def __init__(self, bbox):
        Track._id += 1
        self.id = Track._id
        self.bbox = bbox
        self.text = ""
        self.confidence = 0.0
        self.confirmed = False
        self.age = 0
        self.missed = 0
        self.ocr_sent = False


class PlateTracker:
    def __init__(self, iou_thresh=0.25, max_missed=10):
        self.tracks = {}
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self._lock = threading.Lock()

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a;  bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        union = aw*ah + bw*bh - inter
        return inter / max(union, 1e-6)

    def update(self, detections):
        with self._lock:
            unmatched = set(self.tracks.keys())
            for det in detections:
                best_iou, best_id = 0.0, None
                for tid, t in self.tracks.items():
                    iou = self._iou(det, t.bbox)
                    if iou > best_iou:
                        best_iou, best_id = iou, tid
                if best_iou >= self.iou_thresh and best_id is not None:
                    t = self.tracks[best_id]
                    t.bbox, t.missed, t.age = det, 0, t.age + 1
                    unmatched.discard(best_id)
                else:
                    t = Track(det)
                    self.tracks[t.id] = t
            dead = []
            for tid in unmatched:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missed:
                    dead.append(tid)
            for tid in dead:
                del self.tracks[tid]

    def set_text(self, tid, text, conf):
        with self._lock:
            if tid in self.tracks:
                self.tracks[tid].text = text
                self.tracks[tid].confidence = conf
                self.tracks[tid].confirmed = True

    def get_unrecognized(self):
        with self._lock:
            return [t for t in self.tracks.values()
                    if not t.confirmed and not t.ocr_sent and t.age >= 2]

    def mark_ocr_sent(self, tid):
        with self._lock:
            if tid in self.tracks:
                self.tracks[tid].ocr_sent = True

    def snapshot(self):
        with self._lock:
            return copy.copy(list(self.tracks.values()))


# ──────────────────────────────────────────────────────────────────────────
#  VEHICLE + PLATE DETECTOR (YOLO)
# ──────────────────────────────────────────────────────────────────────────

VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # COCO: car, motorcycle, bus, truck
DEFAULT_VEHICLE_MODEL = "yolov8n.pt"
DEFAULT_PLATE_MODEL = "license_plate_detector.pt"
BASE_DIR = Path(__file__).resolve().parent


def resolve_model_path(model_path):
    if not model_path:
        return None

    path = Path(model_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([
            BASE_DIR / path,
            BASE_DIR / "models" / path,
            BASE_DIR / "weights" / path,
        ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return str((BASE_DIR / path).resolve()) if not path.is_absolute() else str(path)


class YoloDetector:
    def __init__(self, vehicle_model=DEFAULT_VEHICLE_MODEL,
                 plate_model=DEFAULT_PLATE_MODEL, conf=0.35,
                 plate_conf=0.25, imgsz=640):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: install ultralytics to use YOLO detection."
            ) from exc

        self.vehicle_model_path = resolve_model_path(vehicle_model)
        self.plate_model_path = resolve_model_path(plate_model)
        self.vehicle_yolo = YOLO(self.vehicle_model_path)
        self.plate_yolo = None
        self.conf = conf
        self.plate_conf = plate_conf
        self.imgsz = imgsz

        if self.plate_model_path:
            if os.path.exists(self.plate_model_path):
                self.plate_yolo = YOLO(self.plate_model_path)
                print(f"[YOLO] Plate model loaded: {self.plate_model_path}")
            else:
                print(
                    f"[YOLO] Plate model not found: {self.plate_model_path}. "
                    "Put license_plate_detector.pt beside detect_gstreamer.py "
                    "or pass --plate-model /path/to/license_plate_detector.pt."
                )

    @staticmethod
    def _clip_xyxy(frame, xyxy):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(0, min(x2, w - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _xyxy_to_xywh(xyxy):
        x1, y1, x2, y2 = xyxy
        return x1, y1, x2 - x1, y2 - y1

    def detect(self, frame):
        if frame is None:
            return [], []

        vehicles = []
        for result in self.vehicle_yolo.predict(
                frame, conf=self.conf, imgsz=self.imgsz, verbose=False):
            names = result.names or {}
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue
                xyxy = self._clip_xyxy(frame, box.xyxy[0].tolist())
                if not xyxy:
                    continue
                vehicles.append({
                    "bbox": self._xyxy_to_xywh(xyxy),
                    "label": names.get(cls_id, f"class-{cls_id}"),
                    "confidence": float(box.conf[0]),
                })

        plate_boxes = []
        if self.plate_yolo:
            for result in self.plate_yolo.predict(
                    frame, conf=self.plate_conf, imgsz=self.imgsz, verbose=False):
                for box in result.boxes:
                    xyxy = self._clip_xyxy(frame, box.xyxy[0].tolist())
                    if xyxy:
                        plate_boxes.append(self._xyxy_to_xywh(xyxy))

        return plate_boxes, vehicles

    @staticmethod
    def preprocess(plate_img):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        dn = cv2.fastNlMeansDenoising(up, h=10)
        return cv2.adaptiveThreshold(dn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)


# ──────────────────────────────────────────────────────────────────────────
#  OVERLAY RENDERER
# ──────────────────────────────────────────────────────────────────────────

FONT = cv2.FONT_HERSHEY_DUPLEX
COL_OK = (50, 230, 80)
COL_SCAN = (30, 190, 255)
COL_VEHICLE = (255, 170, 40)


def _draw_label(frame, label, x, y, color, scale=0.55, thickness=1):
    (tw, th), bl = cv2.getTextSize(label, FONT, scale, thickness)
    pad = 5
    py = y - th - pad * 2 - 4
    if py < 2:
        py = y + 4
    cv2.rectangle(frame, (x, py), (x + tw + pad * 2, py + th + pad * 2 + bl),
                  (10, 10, 10), -1)
    cv2.rectangle(frame, (x, py), (x + tw + pad * 2, py + th + pad * 2 + bl),
                  color, 1, cv2.LINE_AA)
    cv2.putText(frame, label, (x + pad, py + pad + th), FONT, scale, color,
                thickness, cv2.LINE_AA)


def draw_overlay(frame, tracks, vehicles=None):
    vehicles = vehicles or []
    if not tracks and not vehicles:
        return frame
    overlay = frame.copy()
    H, W = frame.shape[:2]

    for det in vehicles:
        x, y, w, h = det["bbox"]
        x, y = max(0, min(x, W - 2)), max(0, min(y, H - 2))
        w, h = min(w, W - x), min(h, H - y)
        if w < 15 or h < 15:
            continue
        label = f"{det['label']} {int(det['confidence'] * 100)}%"
        cv2.rectangle(frame, (x, y), (x + w, y + h), COL_VEHICLE, 2, cv2.LINE_AA)
        _draw_label(frame, label, x, y, COL_VEHICLE)

    for t in tracks:
        x, y, w, h = t.bbox
        x, y = max(0, min(x, W-2)), max(0, min(y, H-2))
        w, h = min(w, W-x), min(h, H-y)
        if w < 10 or h < 4:
            continue
        color = COL_OK if t.confirmed else COL_SCAN
        label = t.text if t.confirmed else "SCANNING..."
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2, cv2.LINE_AA)
        (tw, th), bl = cv2.getTextSize(label, FONT, 0.65, 2)
        pad = 6
        py = y - th - pad*2 - 4
        if py < 2:
            py = y + h + 4
        px = x
        cv2.rectangle(frame, (px, py), (px+tw+pad*2, py+th+pad*2+bl), (10,10,10), -1)
        cv2.rectangle(frame, (px, py), (px+tw+pad*2, py+th+pad*2+bl), color, 1, cv2.LINE_AA)
        cv2.putText(frame, label, (px+pad, py+pad+th), FONT, 0.65, color, 2, cv2.LINE_AA)
        if t.confirmed and t.confidence > 0:
            badge = f"{int(t.confidence*100)}%"
            cv2.putText(frame, badge, (px+tw+pad*2-40, py+th+pad*2+bl-3),
                        FONT, 0.35, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    return frame


# ──────────────────────────────────────────────────────────────────────────
#  THREAD 1: CAPTURE (reads raw frames from GStreamer pipe → frame_queue)
# ──────────────────────────────────────────────────────────────────────────

class CaptureThread(QThread):
    """
    Pure I/O thread.  Reads raw BGR frames from the GStreamer subprocess
    pipe and pushes them into frame_queue.  Drops old frames if the
    detection thread can't keep up (keeps UI smooth).
    """
    status  = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, source, frame_queue, width, height):
        super().__init__()
        self.source = source
        self.frame_queue = frame_queue
        self.width = width
        self.height = height
        self._running = False

    def run(self):
        self._running = True
        frame_size = self.width * self.height * 3

        self.status.emit("[Capture] Connecting via GStreamer...")
        proc = start_gstreamer(self.source, self.width, self.height)

        # Give GStreamer time to connect & preroll
        time.sleep(2.0)

        # Check if process died immediately
        if proc.poll() is not None:
            err = proc.stderr.read().decode(errors='replace') if proc.stderr else ""
            print(f"[GStreamer] FAILED (exit {proc.returncode}):\n{err}")
            self.status.emit(f"[Capture] GStreamer failed: {err[:80]}")
            self.stopped.emit()
            return

        self.status.emit("[Capture] GStreamer connected ✓")
        print("[CaptureThread] Started.")

        while self._running:
            raw = proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                err = ""
                if proc.stderr:
                    try:
                        err = proc.stderr.read(2000).decode(errors='replace')
                    except Exception:
                        pass
                print(f"[GStreamer] Stream ended. Got {len(raw)}/{frame_size} bytes")
                if err:
                    print(f"[GStreamer] stderr: {err[:500]}")
                self.status.emit("[Capture] Stream ended or broken.")
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)).copy()

            # Non-blocking put — drop oldest frame if queue is full
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()  # discard oldest
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait(frame)

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        self.stopped.emit()
        print("[CaptureThread] Stopped.")

    def stop(self):
        self._running = False
        self.wait(5000)


# ──────────────────────────────────────────────────────────────────────────
#  THREAD 2: DETECTION (frame_queue → detect → overlay → UI signal)
# ──────────────────────────────────────────────────────────────────────────

class DetectionThread(QThread):
    """
    Consumes frames from frame_queue, runs plate detection every N frames,
    draws the AR overlay, and emits the annotated QImage to the UI.
    """
    frame_ready = pyqtSignal(QImage)
    fps_update  = pyqtSignal(float)

    def __init__(self, frame_queue, tracker, ocr_queue,
                 vehicle_model=DEFAULT_VEHICLE_MODEL,
                 plate_model=DEFAULT_PLATE_MODEL,
                 yolo_conf=0.35, plate_conf=0.25, yolo_imgsz=640):
        super().__init__()
        self.frame_queue = frame_queue
        self.tracker = tracker
        self.ocr_queue = ocr_queue
        self._running = False
        self.detector = None
        self.vehicles = []
        self.vehicle_model = vehicle_model
        self.plate_model = plate_model
        self.yolo_conf = yolo_conf
        self.plate_conf = plate_conf
        self.yolo_imgsz = yolo_imgsz

    def run(self):
        self._running = True
        t0, n_frames = time.time(), 0
        skip, SKIP_N = 0, 2
        print("[DetectionThread] Started.")
        try:
            self.detector = YoloDetector(
                self.vehicle_model, self.plate_model, self.yolo_conf,
                self.plate_conf, self.yolo_imgsz)
        except Exception as e:
            print(f"[YOLO] Detector failed to load: {e}")

        while self._running:
            # Block briefly waiting for a frame
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            skip += 1

            # Detection every SKIP_N frames to save CPU
            if self.detector and skip % SKIP_N == 0:
                boxes, self.vehicles = self.detector.detect(frame)
                self.tracker.update(boxes)

                # Queue unrecognized plates for OCR
                for t in self.tracker.get_unrecognized():
                    x, y, w, h = t.bbox
                    crop = frame[max(0, y):y+h, max(0, x):x+w]
                    if crop.size == 0:
                        continue
                    processed = YoloDetector.preprocess(crop)
                    try:
                        self.ocr_queue.put_nowait((t.id, processed))
                        self.tracker.mark_ocr_sent(t.id)
                    except queue.Full:
                        pass

            # Draw overlay on every frame for smooth visuals
            rendered = draw_overlay(frame.copy(), self.tracker.snapshot(), self.vehicles)

            # Emit QImage
            rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
            h_, w_, ch = rgb.shape
            qimg = QImage(rgb.data.tobytes(), w_, h_, ch * w_, QImage.Format_RGB888)
            self.frame_ready.emit(qimg)

            # FPS counter
            n_frames += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                self.fps_update.emit(n_frames / elapsed)
                n_frames, t0 = 0, time.time()

        print("[DetectionThread] Stopped.")

    def stop(self):
        self._running = False
        self.wait(5000)


# ──────────────────────────────────────────────────────────────────────────
#  THREAD 3: OCR WORKER (ocr_queue → EasyOCR → result signal)
# ──────────────────────────────────────────────────────────────────────────

class OCRWorker(QThread):
    result = pyqtSignal(int, str, float)
    ready  = pyqtSignal(bool)

    def __init__(self, ocr_queue):
        super().__init__()
        self.q = ocr_queue
        self._running = False

    def run(self):
        self._running = True
        reader = None
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            self.ready.emit(True)
            print("[OCRWorker] EasyOCR loaded ✓")
        except ImportError:
            self.ready.emit(False)
            print("[OCRWorker] EasyOCR not available — demo mode.")

        while self._running:
            try:
                tid, img = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            text, conf = "", 0.0
            if reader:
                try:
                    results = reader.readtext(img, allowlist='0123456789ABCDEFGHKLMNPRSTUVXYZ-. ')
                    parts, tc = [], 0.0
                    for (_, t, c) in results:
                        if c > 0.25:
                            cleaned = re.sub(r'[^A-Z0-9\-]', '', t.upper().strip())
                            if cleaned:
                                parts.append(cleaned); tc += c
                    if parts:
                        text, conf = ' '.join(parts), tc / len(parts)
                except Exception:
                    pass
            else:
                import random
                time.sleep(0.08)
                text = f"{random.choice(['51','29','43'])}-{random.choice('ABCDEFG')}{random.randint(1,9)} {random.randint(1000,9999)}"
                conf = round(random.uniform(0.78, 0.97), 2)
            if text:
                self.result.emit(tid, text, conf)

    def stop(self):
        self._running = False
        self.wait(5000)


# ──────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ──────────────────────────────────────────────────────────────────────────

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
        self.setMinimumSize(860, 560)
        root = QWidget(); self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0); vbox.setSpacing(0)

        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setStyleSheet("background: #0c0c0f; color: #666;")
        self.canvas.setText("● PRESS START")
        self.canvas.setFont(QFont("Courier New", 20, QFont.Bold))
        vbox.addWidget(self.canvas, 1)

        bar = QFrame(); bar.setFixedHeight(50)
        bar.setStyleSheet("background: #131318; border-top: 1px solid #222;")
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(14, 0, 14, 0); hbox.setSpacing(12)

        self.btn = QPushButton("START")
        self.btn.setFixedSize(100, 34)
        self.btn.setStyleSheet(
            "font: bold 12px 'Courier New'; background: #1e1e26; "
            "color: #ccc; border: 1px solid #333; border-radius: 5px;")
        self.btn.clicked.connect(self._toggle)

        self.lbl_fps = QLabel("FPS —")
        self.lbl_ocr = QLabel("OCR: loading…")
        self.lbl_total = QLabel("Total: 0")
        self.lbl_info = QLabel(f"GStreamer | {self.width}x{self.height}")
        for lbl in (self.lbl_fps, self.lbl_ocr, self.lbl_total, self.lbl_info):
            lbl.setFont(QFont("Courier New", 10))
            lbl.setStyleSheet("color: #888;")

        hbox.addWidget(self.btn)
        hbox.addSpacing(8)
        hbox.addWidget(self.lbl_fps)
        hbox.addWidget(self.lbl_info)
        hbox.addWidget(self.lbl_ocr)
        hbox.addStretch()
        hbox.addWidget(self.lbl_total)
        vbox.addWidget(bar)

    def _toggle(self):
        if self.capture_thread and self.capture_thread.isRunning():
            self._stop()
        else:
            self._start()

    def _start(self):
        self.tracker = PlateTracker()
        # Clear queues
        for q in (self.frame_queue, self.ocr_queue):
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break

        # Thread 3: OCR
        self.ocr_worker = OCRWorker(self.ocr_queue)
        self.ocr_worker.result.connect(self._on_ocr)
        self.ocr_worker.ready.connect(
            lambda ok: self.lbl_ocr.setText("OCR ✓" if ok else "OCR demo"))
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
        self.capture_thread.status.connect(lambda m: self.lbl_ocr.setText(m))
        self.capture_thread.start()

        self.btn.setText("STOP")

    def _stop(self):
        if self.capture_thread:
            self.capture_thread.stop(); self.capture_thread = None
        if self.detect_thread:
            self.detect_thread.stop(); self.detect_thread = None
        if self.ocr_worker:
            self.ocr_worker.stop(); self.ocr_worker = None
        self.btn.setText("START")
        self.canvas.setText("● PRESS START")

    @pyqtSlot(QImage)
    def _on_frame(self, qimg):
        self.canvas.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @pyqtSlot(int, str, float)
    def _on_ocr(self, tid, text, conf):
        self.tracker.set_text(tid, text, conf)
        if text not in self._plates:
            self._plates.append(text)
        self.lbl_total.setText(f"Total: {len(self._plates)}")
        self.lbl_ocr.setText(f"Last: {text} ({int(conf*100)}%)")

    @pyqtSlot(float)
    def _on_fps(self, fps):
        c = "#50e660" if fps > 22 else "#f0b429" if fps > 12 else "#e64040"
        self.lbl_fps.setText(f"FPS {fps:4.1f}")
        self.lbl_fps.setStyleSheet(f"color: {c};")

    def closeEvent(self, e):
        self._stop(); e.accept()


# ──────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="YOLO vehicle + plate detector with GStreamer RTSP backend")
    ap.add_argument("--source", default="rtsp://192.168.1.198:28537/den33")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--vehicle-model", default=DEFAULT_VEHICLE_MODEL,
                    help="YOLO model for COCO vehicle detection")
    ap.add_argument("--plate-model", default=DEFAULT_PLATE_MODEL,
                    help="Custom YOLO license plate model path")
    ap.add_argument("--yolo-conf", type=float, default=0.35,
                    help="YOLO vehicle confidence threshold")
    ap.add_argument("--plate-conf", type=float, default=0.25,
                    help="YOLO license plate confidence threshold")
    ap.add_argument("--yolo-imgsz", type=int, default=640,
                    help="YOLO inference image size")
    args = ap.parse_args()

    src = args.source

    if args.width and args.height:
        w, h = args.width, args.height
    else:
        print(f"[probe] Discovering stream {src} ...")
        w, h, fps = gst_discover(src)
    print(f"[probe] Stream: {w}x{h}")

    app = QApplication(sys.argv)
    win = MainWindow(
        source=src, width=w, height=h, vehicle_model=args.vehicle_model,
        plate_model=args.plate_model, yolo_conf=args.yolo_conf,
        plate_conf=args.plate_conf, yolo_imgsz=args.yolo_imgsz)
    win.show()
    sys.exit(app.exec_())
