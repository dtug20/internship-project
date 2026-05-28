"""
OCR worker thread: consumes raw plate image crops from ocr_queue,
preprocesses them, runs PaddleOCR recognition, and emits results to UI.
"""

import queue
import re
import time

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

PLATE_CHARS = re.compile(r'[^A-Z0-9\-.]')


def _preprocess(img):
    """Upscale + CLAHE contrast enhancement for plate crops."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(up)
    # PaddleOCR expects BGR 3-channel image
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


class OCRWorker(QThread):
    result = pyqtSignal(int, str, float)
    ready  = pyqtSignal(bool)

    def __init__(self, ocr_queue):
        super().__init__()
        self.q = ocr_queue
        self._running = False

    def run(self):
        self._running = True
        ocr = None
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(
                use_angle_cls=False,   # plates are horizontal
                lang='en',
                use_gpu=False,
            )
            self.ready.emit(True)
            print("[OCRWorker] PaddleOCR loaded ✓")
        except Exception as e:
            self.ready.emit(False)
            print(f"[OCRWorker] PaddleOCR failed: {e} — demo mode.")

        while self._running:
            try:
                tid, img = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            text, conf = "", 0.0
            if ocr:
                try:
                    processed = _preprocess(img)
                    result = ocr.ocr(processed, cls=False)

                    parts, tc = [], 0.0
                    for line in (result or []):
                        for item in (line or []):
                            txt = item[1][0]
                            c = item[1][1]
                            if c > 0.3:
                                cleaned = PLATE_CHARS.sub(
                                    '', txt.upper().strip())
                                if cleaned:
                                    parts.append(cleaned)
                                    tc += c
                    if parts:
                        text = ' '.join(parts)
                        conf = tc / len(parts)
                except Exception:
                    pass
            else:
                # Demo mode fallback
                import random
                time.sleep(0.08)
                plate_prefix = random.choice(['51', '29', '43'])
                plate_letter = random.choice('ABCDEFG')
                plate_num1 = random.randint(1, 9)
                plate_num2 = random.randint(1000, 9999)
                text = f"{plate_prefix}-{plate_letter}{plate_num1} {plate_num2}"
                conf = round(random.uniform(0.78, 0.97), 2)
            if text:
                self.result.emit(tid, text, conf)

    def stop(self):
        self._running = False
        self.wait(5000)
