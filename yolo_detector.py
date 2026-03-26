"""
yolo_detector.py — YOLO dart detection + scoring.

Uses a 63-class YOLO model that directly classifies each dart into its
board segment (S1-S20, D1-D20, T1-T20, S_BULL, D_BULL, MISS).
Geometry (homography) is used as a confidence check, not primary classification.
"""

import math
from pathlib import Path

import cv2
import numpy as np

import config
from board import classify_dart, apply_homography
from classes_v2 import ID_TO_CLASS, parse_class_name, NUM_CLASSES


RUNS_DIR = config.PROJECT_ROOT / "runs"

def find_best_weights(frame_width=None, frame_height=None):
    """Find the best YOLO weights.

    Searches runs/detect/dartscorer*/weights/best.pt, preferring the
    most recently modified model.

    Returns:
        Path to best.pt, or None if no model found.
    """
    detect_dir = RUNS_DIR / "detect"
    if not detect_dir.exists():
        return None

    candidates = []
    for d in sorted(detect_dir.iterdir()):
        if d.is_dir() and d.name.startswith("dartscorer"):
            best = d / "weights" / "best.pt"
            if best.exists():
                candidates.append(best)

    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    return None


DEFAULT_WEIGHTS = find_best_weights() or (RUNS_DIR / "detect" / "dartscorer" / "weights" / "best.pt")


def _estimate_tip(bbox, board_center):
    """Estimate dart tip position from bounding box.

    The tip is the point on the bbox edge closest to the board center,
    since darts point roughly toward the center.

    Args:
        bbox: (x1, y1, x2, y2) bounding box.
        board_center: (cx, cy) board center in pixel coordinates.

    Returns:
        (tip_x, tip_y) integer pixel coordinates.
    """
    x1, y1, x2, y2 = bbox
    bcx, bcy = board_center

    # Bbox center
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Direction from bbox center toward board center
    dx = bcx - cx
    dy = bcy - cy
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1:
        return int(cx), int(cy)

    dx /= dist
    dy /= dist

    # Walk from bbox center toward board center until we hit the bbox edge
    # The tip is the closest edge point to the board center
    # Find intersection with bbox boundary
    candidates = []

    # Right/left edge
    if dx > 0:
        t = (x2 - cx) / dx if dx != 0 else float('inf')
    elif dx < 0:
        t = (x1 - cx) / dx if dx != 0 else float('inf')
    else:
        t = float('inf')
    # Wait — we want to go TOWARD center, not away. The tip is the edge closest
    # to center. So we go in the direction of (dx, dy) and find the bbox edge.
    # Actually, the tip IS the part closest to the board center.
    # Simplest approach: find the bbox corner/edge point closest to board center.

    # Sample points along bbox edges and find closest to board center
    best_pt = (int(cx), int(cy))
    best_dist = dist

    # Check midpoints of each edge + corners
    edge_points = [
        ((x1 + x2) / 2, y1),  # top mid
        ((x1 + x2) / 2, y2),  # bottom mid
        (x1, (y1 + y2) / 2),  # left mid
        (x2, (y1 + y2) / 2),  # right mid
        (x1, y1), (x2, y1),   # top corners
        (x1, y2), (x2, y2),   # bottom corners
    ]

    for px, py in edge_points:
        d = math.sqrt((px - bcx) ** 2 + (py - bcy) ** 2)
        if d < best_dist:
            best_dist = d
            best_pt = (int(px), int(py))

    return best_pt


class YOLODartDetector:
    def __init__(self, weights=None, conf=config.YOLO_DEFAULT_CONF,
                 iou=config.YOLO_DEFAULT_IOU, device=None,
                 homography=None, crop_offset=(0, 0)):
        """
        Args:
            weights: Path to YOLO weights file (1-class model).
            conf: Confidence threshold for detections.
            iou: IoU threshold for NMS.
            device: Inference device ('cpu', '0', etc).
            homography: 3x3 numpy array for board homography.
            crop_offset: (x, y) offset from crop ROI to full frame.
        """
        from ultralytics import YOLO

        weights = weights or DEFAULT_WEIGHTS
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {weights}.\n"
                "Run 'python train.py' or check runs/detect/dartscorer_v2/"
            )

        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.device = device
        self.homography = homography
        self.crop_offset = crop_offset

        # Board center in crop-pixel space (for tip estimation)
        self.board_center = None
        if homography is not None:
            try:
                H_inv = np.linalg.inv(homography)
                cx, cy = config.CANONICAL_CENTER
                pts = np.array([[[cx, cy]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(pts, H_inv)
                bcx = float(transformed[0][0][0]) - crop_offset[0]
                bcy = float(transformed[0][0][1]) - crop_offset[1]
                self.board_center = (bcx, bcy)
            except Exception:
                pass

        # Track confirmed darts in current round
        self.confirmed_darts = {}  # ordinal (1-3) -> detection dict
        self.round_dart_count = 0

    def process_frame(self, frame):
        """Run YOLO inference + geometry classification on a frame.

        Args:
            frame: BGR uint8 image (cropped).

        Returns:
            List of detection dicts, each with:
                tip: (x, y) estimated tip position
                bbox: (x1, y1, x2, y2)
                yolo_confidence: YOLO detection confidence
                classification: dict from board.classify_dart (or None)
                segment: segment name (e.g., "T20") or "unknown"
                score: point value
                label: human-readable label
                confidence: geometry confidence (0-1)
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            yolo_conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = (int(x1), int(y1), int(x2), int(y2))

            # Estimate tip position
            if self.board_center is not None:
                tip = _estimate_tip(bbox, self.board_center)
            else:
                tip = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Primary classification from YOLO (63-class model)
            cls_name = ID_TO_CLASS.get(cls_id, "unknown")
            info = parse_class_name(cls_name) if cls_name != "unknown" else None
            segment = info["segment"] if info else "unknown"
            score = info["score"] if info else 0
            label = info["label"] if info else "Unknown"

            # Geometry as confidence check (not primary classification)
            geo_classification = None
            geo_confidence = 1.0  # default: trust YOLO
            geo_agrees = True

            if self.homography is not None:
                tip_full = (tip[0] + self.crop_offset[0],
                            tip[1] + self.crop_offset[1])
                try:
                    can = apply_homography(tip_full, self.homography)
                    geo_classification = classify_dart(can[0], can[1])
                    geo_confidence = geo_classification["confidence"]
                    geo_agrees = geo_classification["segment"] == segment
                except Exception:
                    pass

            detections.append({
                "tip": tip,
                "bbox": bbox,
                "yolo_confidence": yolo_conf,
                "yolo_class": cls_name,
                "classification": geo_classification,
                "segment": segment,
                "score": score,
                "label": label,
                "confidence": geo_confidence,
                "geo_agrees": geo_agrees,
            })

        return detections

    def get_new_darts(self, detections):
        """Filter detections to find newly thrown darts.

        Uses detection count: if YOLO sees more darts than we've confirmed,
        the extras are new. This works even when darts are in a tight cluster
        (a few pixels apart) where proximity matching would fail.

        To identify WHICH detections are new, we match confirmed darts to
        their nearest detection and return the unmatched ones.

        Args:
            detections: List from process_frame().

        Returns:
            List of new detection dicts (subset of input).
        """
        if len(detections) <= self.round_dart_count:
            return []  # no new darts (same or fewer than confirmed)

        if self.round_dart_count == 0:
            # No confirmed darts yet — all detections are new
            return list(detections)

        # Match each confirmed dart to its nearest detection (greedy)
        import math
        matched_indices = set()
        for confirmed in self.confirmed_darts.values():
            cx, cy = confirmed["tip"]
            best_idx = None
            best_dist = float('inf')
            for i, det in enumerate(detections):
                if i in matched_indices:
                    continue
                tx, ty = det["tip"]
                d = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx is not None:
                matched_indices.add(best_idx)

        # Unmatched detections are new darts
        return [det for i, det in enumerate(detections)
                if i not in matched_indices]

    def confirm_dart(self, detection):
        """Mark a detection as a confirmed dart in this round.

        Assigns the next ordinal (1, 2, or 3).
        """
        ordinal = self.round_dart_count + 1
        detection["ordinal"] = ordinal
        self.confirmed_darts[ordinal] = detection
        self.round_dart_count = ordinal

    def reset(self):
        """Reset for a new round (darts pulled from board)."""
        self.confirmed_darts = {}
        self.round_dart_count = 0

    def get_debug_frame(self, frame, detections):
        """Annotate frame with detection boxes, tips, and classifications."""
        out = frame.copy()

        # Draw board center
        if self.board_center is not None:
            bcx, bcy = int(self.board_center[0]), int(self.board_center[1])
            cv2.drawMarker(out, (bcx, bcy), (255, 0, 255),
                          cv2.MARKER_CROSS, 20, 1)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["yolo_confidence"]
            geo_conf = det["confidence"]
            segment = det["segment"]
            geo_agrees = det.get("geo_agrees", True)
            label_text = f"{segment} ({conf:.0%})"
            if not geo_agrees:
                geo_seg = det["classification"]["segment"] if det.get("classification") else "?"
                label_text += f" [geo:{geo_seg}]"

            # Color: green if YOLO+geo agree, yellow if geo unsure, red if disagree
            if geo_agrees and geo_conf > config.GEO_CONF_MODERATE:
                color = (0, 255, 0)    # green — YOLO + geo agree
            elif not geo_agrees:
                color = (0, 0, 255)    # red — YOLO and geo disagree
            else:
                color = (0, 255, 255)  # yellow — geo unsure

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label_text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Tip marker
            tx, ty = det["tip"]
            cv2.circle(out, (tx, ty), 4, color, -1)

            # Score text
            if det["score"] > 0:
                cv2.putText(out, str(det["score"]), (tx + 8, ty - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw confirmed darts
        for ordinal, dart in self.confirmed_darts.items():
            tx, ty = dart["tip"]
            color = config.DART_ORDINAL_COLORS.get(ordinal, (255, 255, 255))
            cv2.circle(out, (tx, ty), 8, color, 2)

        return out
