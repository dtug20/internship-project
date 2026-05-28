"""
Entry point: YOLO vehicle + plate detector with GStreamer RTSP backend.

Modules:
  gstreamer_utils.py   – GStreamer probe & pipeline
  tracker.py           – Track + PlateTracker
  detector.py          – YoloDetector + model resolution
  overlay.py           – Frame overlay rendering
  capture_thread.py    – CaptureThread (raw frame I/O)
  detection_thread.py  – DetectionThread (YOLO + tracking)
  ocr_worker.py        – OCRWorker (EasyOCR)
  main_window.py       – MainWindow (PyQt5 UI)
"""

import sys
import argparse

from PyQt5.QtWidgets import QApplication

from gstreamer_utils import gst_discover
from detector import DEFAULT_VEHICLE_MODEL, DEFAULT_PLATE_MODEL
from main_window import MainWindow


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
        source=src, width=w, height=h,
        vehicle_model=args.vehicle_model,
        plate_model=args.plate_model,
        yolo_conf=args.yolo_conf,
        plate_conf=args.plate_conf,
        yolo_imgsz=args.yolo_imgsz)
    win.show()
    sys.exit(app.exec_())
