"""
Plate tracker: associates detected plate bounding boxes across frames
using IoU + center-distance matching with EMA smoothing.
"""

import copy
import threading


class Track:
    _id = 0
    def __init__(self, bbox, vehicle_label=""):
        Track._id += 1
        self.id = Track._id
        self.bbox = bbox
        self.vehicle_label = vehicle_label
        self.text = ""
        self.confidence = 0.0
        self.confirmed = False
        self.age = 0
        self.missed = 0
        self.ocr_sent = False


class PlateTracker:
    def __init__(self, iou_thresh=0.15, max_missed=5, smooth=0.5):
        """
        Args:
            iou_thresh: minimum IoU to consider a match
            max_missed: frames before removing an unmatched track
            smooth: EMA factor for bbox smoothing (0=no smooth, 1=freeze)
        """
        self.tracks = {}
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.smooth = smooth
        self._lock = threading.Lock()

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a;  bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        union = aw*ah + bw*bh - inter
        return inter / max(union, 1e-6)

    @staticmethod
    def _center_dist(a, b):
        """Euclidean distance between bbox centers."""
        acx, acy = a[0] + a[2] / 2, a[1] + a[3] / 2
        bcx, bcy = b[0] + b[2] / 2, b[1] + b[3] / 2
        return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5

    def _match_score(self, det, track_bbox):
        """Combined score: IoU + proximity bonus. Higher = better match."""
        iou = self._iou(det, track_bbox)
        cdist = self._center_dist(det, track_bbox)
        max_dim = max(det[2], det[3], track_bbox[2], track_bbox[3], 1)
        # Proximity: 1.0 when centers overlap, 0.0 when dist >= max_dim
        proximity = max(0.0, 1.0 - cdist / max_dim)
        return iou + proximity * 0.5  # weighted combination

    def _smooth_bbox(self, old, new):
        """EMA smooth: blend old bbox with new detection."""
        s = self.smooth
        return (
            int(old[0] * s + new[0] * (1 - s)),
            int(old[1] * s + new[1] * (1 - s)),
            int(old[2] * s + new[2] * (1 - s)),
            int(old[3] * s + new[3] * (1 - s)),
        )

    def update(self, detections, vehicle_labels=None):
        with self._lock:
            labels = vehicle_labels or [""] * len(detections)

            # Build all candidate matches with scores
            candidates = []
            for di, det in enumerate(detections):
                for tid, t in self.tracks.items():
                    score = self._match_score(det, t.bbox)
                    # Accept if IoU alone passes OR combined score is good
                    iou = self._iou(det, t.bbox)
                    if iou >= self.iou_thresh or score >= 0.6:
                        candidates.append((score, di, tid))

            # Greedy assignment: best score first, no double-matching
            candidates.sort(reverse=True)
            matched_dets = set()
            matched_tracks = set()

            for score, di, tid in candidates:
                if di in matched_dets or tid in matched_tracks:
                    continue
                t = self.tracks[tid]
                t.bbox = self._smooth_bbox(t.bbox, detections[di])
                t.missed = 0
                t.age += 1
                if di < len(labels) and labels[di]:
                    t.vehicle_label = labels[di]
                matched_dets.add(di)
                matched_tracks.add(tid)

            # Create new tracks for unmatched detections
            for di, det in enumerate(detections):
                if di not in matched_dets:
                    lbl = labels[di] if di < len(labels) else ""
                    t = Track(det, vehicle_label=lbl)
                    self.tracks[t.id] = t

            # Age unmatched tracks, remove dead ones
            dead = []
            for tid in self.tracks:
                if tid not in matched_tracks:
                    self.tracks[tid].missed += 1
                    t = self.tracks[tid]
                    # Unconfirmed plates: remove after 1 miss (no trailing boxes)
                    # Confirmed plates (OCR done): keep longer
                    limit = self.max_missed if t.confirmed else 1
                    if t.missed > limit:
                        dead.append(tid)
            for tid in dead:
                del self.tracks[tid]

    def set_text(self, tid, text, conf):
        with self._lock:
            if tid in self.tracks:
                self.tracks[tid].text = text
                self.tracks[tid].confidence = conf
                self.tracks[tid].confirmed = True

    def get_vehicle_label(self, tid):
        with self._lock:
            if tid in self.tracks:
                return self.tracks[tid].vehicle_label
            return ""

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
