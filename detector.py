"""
YOLO vehicle + license plate detector.
- Detects vehicles (car, motorcycle, bus, truck) using COCO model
- Detects license plates using a custom YOLO model
- Preprocesses plate crops for OCR
"""

import os
from pathlib import Path

import cv2

VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # COCO: car, motorcycle, bus, truck

# Minimum vehicle size for cascaded plate detection (skip tiny vehicles)
# Similar to DeepStream SGIE1 input-object-min-width/height
MIN_VEHICLE_W = 40
MIN_VEHICLE_H = 30
DEFAULT_VEHICLE_MODEL = "yolov8n.pt"
DEFAULT_PLATE_MODEL = "best.pt"
BASE_DIR = Path(__file__).resolve().parent


def _is_hf_model_id(model_str: str) -> bool:
    """Check if a model string looks like a HuggingFace model ID (org/repo)."""
    # HF IDs look like "org/model-name" — not absolute paths, not .pt/.onnx files
    if "/" not in model_str:
        return False
    # Absolute paths or relative paths with extensions are local files
    if model_str.startswith("/") or model_str.startswith("./"):
        return False
    if Path(model_str).suffix in (".pt", ".onnx", ".torchscript"):
        return False
    # If the file exists locally, treat as local path
    if Path(model_str).exists():
        return False
    return True


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


def _load_yolo_model(model_str: str):
    """Load a YOLO model from local path or HuggingFace Hub."""
    from ultralytics import YOLO

    # HuggingFace model ID (e.g. "morsetechlab/yolov11-license-plate-detection")
    if _is_hf_model_id(model_str):
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            print(f"[YOLO] Downloading from HuggingFace: {model_str} ...")
            # Find the first .pt file in the repo
            repo_files = list_repo_files(model_str)
            pt_files = [f for f in repo_files if f.endswith(".pt")]
            if not pt_files:
                print(f"[YOLO] No .pt files found in {model_str}")
                return None
            # Prefer nano (n) variant for speed, fallback to first .pt
            target = pt_files[0]
            for f in pt_files:
                if f.endswith("v1n.pt") or f.endswith("n.pt"):
                    target = f
                    break
            local_path = hf_hub_download(repo_id=model_str, filename=target)
            model = YOLO(local_path)
            print(f"[YOLO] HuggingFace model loaded ✓ ({target})")
            return model
        except Exception as e:
            print(f"[YOLO] Failed to load HF model '{model_str}': {e}")
            return None
        except Exception as e:
            print(f"[YOLO] Failed to load HF model '{model_str}': {e}")
            return None

    # Local file path
    resolved = resolve_model_path(model_str)
    if resolved and os.path.exists(resolved):
        model = YOLO(resolved)
        print(f"[YOLO] Local model loaded: {resolved}")
        return model

    print(f"[YOLO] Model not found: {model_str}")
    return None


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

        self.conf = conf
        self.plate_conf = plate_conf
        self.imgsz = imgsz

        # Load vehicle detection model (COCO)
        self.vehicle_yolo = YOLO(resolve_model_path(vehicle_model))

        # Load plate detection model (HuggingFace or local)
        self.plate_yolo = _load_yolo_model(plate_model) if plate_model else None

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

        # ── Cascaded plate detection (on vehicle crops) ──────────────
        # Like DeepStream SGIE: detect plates only within detected vehicles
        plate_boxes = []
        if self.plate_yolo:
            if vehicles:
                for v in vehicles:
                    vx, vy, vw, vh = v["bbox"]
                    if vw < MIN_VEHICLE_W or vh < MIN_VEHICLE_H:
                        continue
                    crop = frame[vy:vy+vh, vx:vx+vw]
                    if crop.size == 0:
                        continue
                    for result in self.plate_yolo.predict(
                            crop, conf=self.plate_conf,
                            imgsz=self.imgsz, verbose=False):
                        for box in result.boxes:
                            xyxy = self._clip_xyxy(
                                crop, box.xyxy[0].tolist())
                            if xyxy:
                                px, py, pw, ph = self._xyxy_to_xywh(xyxy)
                                # Map crop coords → frame coords
                                plate_boxes.append(
                                    (px + vx, py + vy, pw, ph))
            else:
                # Fallback: no vehicles → scan full frame
                for result in self.plate_yolo.predict(
                        frame, conf=self.plate_conf,
                        imgsz=self.imgsz, verbose=False):
                    for box in result.boxes:
                        xyxy = self._clip_xyxy(
                            frame, box.xyxy[0].tolist())
                        if xyxy:
                            plate_boxes.append(
                                self._xyxy_to_xywh(xyxy))

        return plate_boxes, vehicles

    @staticmethod
    def preprocess(plate_img):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        dn = cv2.fastNlMeansDenoising(up, h=10)
        return cv2.adaptiveThreshold(dn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
