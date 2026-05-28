"""
Detection thread: consumes frames from frame_queue, runs YOLO plate
detection every N frames, draws the AR overlay, and emits the annotated
QImage to the UI.
"""

import queue
import time

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

from detector import YoloDetector, DEFAULT_VEHICLE_MODEL, DEFAULT_PLATE_MODEL
from overlay import draw_overlay


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

                # Associate each plate with its containing vehicle
                plate_labels = []
                for px, py, pw, ph in boxes:
                    pcx, pcy = px + pw / 2, py + ph / 2
                    label = ""
                    for v in self.vehicles:
                        vx, vy, vw, vh = v["bbox"]
                        if vx <= pcx <= vx + vw and vy <= pcy <= vy + vh:
                            label = v["label"]
                            break
                    plate_labels.append(label)

                self.tracker.update(boxes, plate_labels)

                # Queue unrecognized plates for OCR (raw crop)
                for t in self.tracker.get_unrecognized():
                    x, y, w, h = t.bbox
                    crop = frame[max(0, y):y+h, max(0, x):x+w]
                    if crop.size == 0:
                        continue
                    try:
                        self.ocr_queue.put_nowait((t.id, crop))
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
