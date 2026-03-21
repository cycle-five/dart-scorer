"""
yolo_detector.py — YOLO-based dart detection and scoring.

Drop-in replacement for detector.py. Uses a trained YOLOv8 model to detect
dart tips and classify them by ordinal (1st/2nd/3rd) and board segment.
No homography or board calibration needed.
"""

from pathlib import Path

import cv2
import numpy as np

import config
from classes import ID_TO_CLASS, parse_class_name


RUNS_DIR = config.PROJECT_ROOT / "runs"
DEFAULT_WEIGHTS = RUNS_DIR / "detect" / "dartscorer" / "weights" / "best.pt"


class YOLODartDetector:
    def __init__(self, weights=None, conf=0.25, iou=0.45, device=None):
        """
        Args:
            weights: Path to YOLO weights file. Defaults to best.pt from training.
            conf: Confidence threshold for detections.
            iou: IoU threshold for NMS.
            device: Inference device ('cpu', '0', etc).
        """
        from ultralytics import YOLO

        weights = weights or DEFAULT_WEIGHTS
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {weights}.\n"
                "Run 'python train.py' first to train a model."
            )

        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.device = device

        # Track confirmed darts in current round
        self.confirmed_darts = {}  # ordinal -> detection dict
        self.round_dart_count = 0

    def process_frame(self, frame):
        """Run YOLO inference on a frame.

        Args:
            frame: BGR uint8 image from camera.

        Returns:
            List of detection dicts, each with:
                tip: (x, y) center of bounding box
                bbox: (x1, y1, x2, y2)
                class_name: e.g. "d1_T20"
                class_id: integer class index
                confidence: float 0-1
                score_info: parsed score dict from classes.parse_class_name
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
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = ID_TO_CLASS.get(cls_id, f"unknown_{cls_id}")
            score_info = parse_class_name(class_name)

            tip_x = int((x1 + x2) / 2)
            tip_y = int((y1 + y2) / 2)

            detections.append({
                "tip": (tip_x, tip_y),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "class_name": class_name,
                "class_id": cls_id,
                "confidence": conf,
                "score_info": score_info,
            })

        return detections

    def get_new_darts(self, detections):
        """Filter detections to find newly thrown darts.

        Compares current detections against previously confirmed darts
        and returns only new ones (higher ordinal than what's already known).

        Args:
            detections: List from process_frame().

        Returns:
            List of new detection dicts (subset of input).
        """
        new = []
        for det in detections:
            ordinal = det["score_info"]["ordinal"]
            if ordinal not in self.confirmed_darts:
                new.append(det)
        return new

    def confirm_dart(self, detection):
        """Mark a detection as a confirmed dart in this round."""
        ordinal = detection["score_info"]["ordinal"]
        self.confirmed_darts[ordinal] = detection
        self.round_dart_count = len(self.confirmed_darts)

    def reset(self):
        """Reset for a new round (darts pulled from board)."""
        self.confirmed_darts = {}
        self.round_dart_count = 0

    def get_debug_frame(self, frame, detections):
        """Annotate frame with detection boxes and labels."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            info = det["score_info"]
            label = f"d{info['ordinal']} {info['label']} ({conf:.2f})"

            # Color by ordinal
            colors = {1: (0, 255, 0), 2: (0, 255, 255), 3: (0, 0, 255)}
            color = colors.get(info["ordinal"], (255, 255, 255))

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Tip marker
            tx, ty = det["tip"]
            cv2.circle(out, (tx, ty), 4, color, -1)

        return out
