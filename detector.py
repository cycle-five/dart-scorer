"""
detector.py — Dart detection via background subtraction and frame differencing.

Maintains a rolling median background model and detects new darts by finding
persistent blobs that appear in the difference image. Localizes the dart tip
as the point within each blob closest to the board center.
"""

import cv2
import numpy as np
import config


class DartDetector:
    def __init__(self, board_center=None, debug=False):
        """
        Args:
            board_center: (x, y) approximate center of the dartboard in
                         undistorted camera pixel space. Used for tip
                         localization. If None, uses frame center.
            debug: If True, show intermediate CV windows.
        """
        self.debug = debug
        self.board_center = board_center
        self.bg_buffer = []
        self.background = None
        self.candidate_blobs = {}
        self.confirmed_darts = {}
        self.next_blob_id = 0
        self.cooldown_counter = 0
        self.prev_mean_intensity = None
        self.paused = False

    def update_background(self, gray_frame):
        """Update the rolling median background model with a new grayscale frame.

        Does nothing if a motion cooldown is active.

        Args:
            gray_frame: Grayscale uint8 frame.
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        self.bg_buffer.append(gray_frame.copy())
        if len(self.bg_buffer) > config.BACKGROUND_HISTORY:
            self.bg_buffer.pop(0)
        if len(self.bg_buffer) >= 5:
            self.background = np.median(np.array(self.bg_buffer), axis=0).astype(np.uint8)

    def check_illumination(self, gray_frame):
        """Detect sudden illumination changes and pause detection if one occurs.

        Args:
            gray_frame: Grayscale uint8 frame.

        Returns:
            True if an illumination change was detected, False otherwise.
        """
        current_mean = float(np.mean(gray_frame))
        if self.prev_mean_intensity is not None:
            if abs(current_mean - self.prev_mean_intensity) > config.ILLUMINATION_CHANGE_THRESHOLD:
                print("Illumination change detected — pausing detection")
                self.paused = True
                self.bg_buffer = []
                self.background = None
                self.candidate_blobs = {}
                self.prev_mean_intensity = current_mean
                return True
        self.prev_mean_intensity = current_mean
        if self.paused and len(self.bg_buffer) >= 10:
            self.paused = False
            print("Resuming detection")
        return False

    def get_diff_mask(self, gray_frame):
        """Compute a binary mask of pixels that differ from the background.

        Applies Gaussian blur before differencing to suppress wire-level
        detail and camera sensor noise, then uses morphological opening
        (to remove small speckles) followed by closing (to fill gaps in
        dart-shaped blobs).

        Args:
            gray_frame: Grayscale uint8 frame.

        Returns:
            Binary uint8 mask, or None if no background model exists yet.
        """
        if self.background is None:
            return None
        # Blur both frames to suppress wire detail and sensor noise
        bg_blur = cv2.GaussianBlur(self.background, (9, 9), 0)
        fr_blur = cv2.GaussianBlur(gray_frame, (9, 9), 0)
        diff = cv2.absdiff(bg_blur, fr_blur)
        _, thresh = cv2.threshold(diff, config.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        # Opening removes small noise specks (wire glints)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        # Closing fills gaps within dart-shaped blobs
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        if self.debug:
            cv2.imshow("Diff Mask", mask)
        return mask

    def find_blobs(self, mask):
        """Find contours in the diff mask that fall within the valid blob area range.

        Args:
            mask: Binary uint8 mask from get_diff_mask.

        Returns:
            List of dicts: [{"contour": cnt, "centroid": (cx, cy),
                             "bbox": (x, y, w, h), "area": area}, ...]
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (config.MIN_BLOB_AREA <= area <= config.MAX_BLOB_AREA):
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            bbox = cv2.boundingRect(cnt)
            blobs.append({
                "contour": cnt,
                "centroid": (cx, cy),
                "bbox": bbox,
                "area": area,
            })
        return blobs

    def find_dart_tip(self, contour, blob_bbox):
        """Find the point in the contour closest to the board center.

        This approximates the dart tip since the dart points toward the center.

        Args:
            contour: OpenCV contour array.
            blob_bbox: (x, y, w, h) bounding rect (unused, kept for API clarity).

        Returns:
            (tip_x, tip_y) integer pixel coordinates.
        """
        center = self.board_center
        pts = contour.reshape(-1, 2)
        cx, cy = center
        dists = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
        idx = int(np.argmin(dists))
        return (int(pts[idx, 0]), int(pts[idx, 1]))

    def match_blobs(self, current_blobs):
        """Match current blobs to existing candidates and yield newly confirmed darts.

        Updates candidate_blobs in place. Candidates that have been seen for at
        least PERSISTENCE_FRAMES consecutive frames are promoted to confirmed_darts.

        Args:
            current_blobs: List of blob dicts from find_blobs.

        Yields:
            Dict {"tip": (x, y), "blob_id": int, "bbox": (x, y, w, h)} for each
            newly confirmed dart.
        """
        MATCH_DISTANCE = 50  # pixels

        matched_candidate_ids = set()
        matched_blob_indices = set()

        # Match current blobs to existing candidates
        for blob_idx, blob in enumerate(current_blobs):
            bx, by = blob["centroid"]
            best_id = None
            best_dist = MATCH_DISTANCE

            for cid, cand in self.candidate_blobs.items():
                cx, cy = cand["centroid"]
                dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id = cid

            if best_id is not None:
                cand = self.candidate_blobs[best_id]
                cand["centroid"] = blob["centroid"]
                cand["frames_seen"] += 1
                cand["tip"] = blob.get("tip", blob["centroid"])
                cand["bbox"] = blob["bbox"]
                matched_candidate_ids.add(best_id)
                matched_blob_indices.add(blob_idx)
            else:
                # New candidate
                new_id = self.next_blob_id
                self.next_blob_id += 1
                self.candidate_blobs[new_id] = {
                    "centroid": blob["centroid"],
                    "frames_seen": 1,
                    "tip": blob.get("tip", blob["centroid"]),
                    "bbox": blob["bbox"],
                }
                matched_blob_indices.add(blob_idx)

        # Remove candidates not seen this frame
        gone = [cid for cid in self.candidate_blobs if cid not in matched_candidate_ids
                and not any(
                    ((self.candidate_blobs[cid]["centroid"][0] - b["centroid"][0]) ** 2 +
                     (self.candidate_blobs[cid]["centroid"][1] - b["centroid"][1]) ** 2) ** 0.5 < MATCH_DISTANCE
                    for b in current_blobs
                )]
        for cid in gone:
            del self.candidate_blobs[cid]

        # Promote persistent candidates to confirmed darts
        for cid, cand in list(self.candidate_blobs.items()):
            if cand["frames_seen"] >= config.PERSISTENCE_FRAMES and cid not in self.confirmed_darts:
                self.confirmed_darts[cid] = {
                    "tip": cand["tip"],
                    "bbox": cand["bbox"],
                    "blob_id": cid,
                }
                yield {"tip": cand["tip"], "blob_id": cid, "bbox": cand["bbox"]}

    def process_frame(self, frame):
        """Main entry point. Process one undistorted BGR frame.

        Args:
            frame: BGR uint8 image (undistorted camera frame).

        Returns:
            List of new dart detection dicts:
            [{"tip": (x, y), "blob_id": int, "bbox": (x, y, w, h)}, ...]
        """
        if self.board_center is None:
            h, w = frame.shape[:2]
            self.board_center = (w // 2, h // 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.check_illumination(gray):
            return []

        if self.paused:
            self.bg_buffer.append(gray.copy())
            if len(self.bg_buffer) > config.BACKGROUND_HISTORY:
                self.bg_buffer.pop(0)
            return []

        mask = self.get_diff_mask(gray)
        if mask is None:
            self.update_background(gray)
            return []

        blobs = self.find_blobs(mask)

        # Compute tip for each blob before matching
        for blob in blobs:
            blob["tip"] = self.find_dart_tip(blob["contour"], blob["bbox"])

        new_detections = list(self.match_blobs(blobs))

        if len(blobs) == 0 and self.cooldown_counter == 0:
            self.update_background(gray)
        else:
            if len(blobs) > 0:
                self.cooldown_counter = config.MOTION_COOLDOWN_FRAMES

        return new_detections

    def dart_removed(self, gray_frame):
        """Check if any confirmed dart has been removed from the board.

        Args:
            gray_frame: Current grayscale frame.

        Returns:
            List of blob_ids that are no longer present in the diff mask.
        """
        mask = self.get_diff_mask(gray_frame)
        if mask is None or not self.confirmed_darts:
            return []

        removed = []
        for blob_id, dart in list(self.confirmed_darts.items()):
            tx, ty = dart["tip"]
            x, y, w, h = dart["bbox"]
            # Check if the bbox region in the mask still has significant signal
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(mask.shape[1], x + w)
            y2 = min(mask.shape[0], y + h)
            roi = mask[y1:y2, x1:x2]
            if roi.size == 0 or np.sum(roi) == 0:
                removed.append(blob_id)
                del self.confirmed_darts[blob_id]
        return removed

    def reset(self):
        """Clear all detector state, resetting to initial conditions."""
        self.bg_buffer = []
        self.background = None
        self.candidate_blobs = {}
        self.confirmed_darts = {}
        self.next_blob_id = 0
        self.cooldown_counter = 0
        self.prev_mean_intensity = None
        self.paused = False

    def get_debug_frame(self, frame, detections):
        """Annotate a frame copy with detection debug info.

        Draws:
        - Green bounding boxes around each confirmed dart
        - Red circle at each confirmed dart tip
        - Yellow circle at each newly detected dart tip
        - Small inset of the diff mask in the top-left corner (if background exists)

        Args:
            frame: BGR uint8 frame to annotate.
            detections: List of new detection dicts from process_frame.

        Returns:
            Annotated BGR frame copy.
        """
        out = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Draw confirmed darts
        for blob_id, dart in self.confirmed_darts.items():
            x, y, w, h = dart["bbox"]
            tx, ty = dart["tip"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(out, (tx, ty), 5, (0, 0, 255), -1)
            cv2.putText(out, str(blob_id), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw new detections
        for det in detections:
            tx, ty = det["tip"]
            cv2.circle(out, (tx, ty), 9, (0, 255, 255), 2)

        # Diff mask inset
        if self.background is not None:
            mask = self.get_diff_mask(gray)
            if mask is not None:
                inset_h, inset_w = 120, 213
                inset = cv2.resize(mask, (inset_w, inset_h))
                inset_bgr = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
                out[0:inset_h, 0:inset_w] = inset_bgr

        return out
