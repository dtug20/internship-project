"""
Overlay renderer: draws vehicle bounding boxes, plate annotations,
and OCR labels on video frames.
"""

import cv2

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
    H, W = frame.shape[:2]

    # Draw vehicle bounding boxes
    for det in vehicles:
        x, y, w, h = det["bbox"]
        x, y = max(0, min(x, W - 2)), max(0, min(y, H - 2))
        w, h = min(w, W - x), min(h, H - y)
        if w < 15 or h < 15:
            continue
        label = f"{det['label']} {int(det['confidence'] * 100)}%"
        cv2.rectangle(frame, (x, y), (x + w, y + h), COL_VEHICLE, 2, cv2.LINE_AA)
        _draw_label(frame, label, x, y, COL_VEHICLE)

    # Draw plate bounding boxes (same style as vehicles)
    for t in tracks:
        x, y, w, h = t.bbox
        x, y = max(0, min(x, W - 2)), max(0, min(y, H - 2))
        w, h = min(w, W - x), min(h, H - y)
        if w < 10 or h < 4:
            continue

        if t.confirmed:
            # Confirmed plate: green box + recognized text
            color = COL_OK
            label = f"{t.text} {int(t.confidence * 100)}%"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
            _draw_label(frame, label, x, y, color)
        else:
            # Unconfirmed plate: thin orange border only, no label
            cv2.rectangle(frame, (x, y), (x + w, y + h), COL_SCAN, 1, cv2.LINE_AA)

    return frame
