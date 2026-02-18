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
        bk = config.BLUR_KSIZE
        bg_blur = cv2.GaussianBlur(self.background, (bk, bk), 0)
        fr_blur = cv2.GaussianBlur(gray_frame, (bk, bk), 0)
        diff = cv2.absdiff(bg_blur, fr_blur)
        _, thresh = cv2.threshold(diff, config.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        # Opening removes small noise specks (wire glints)
        ok = config.OPEN_KSIZE
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        # Closing fills gaps within dart-shaped blobs
        ck = config.CLOSE_KSIZE
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        if self.debug:
            cv2.imshow("Diff Mask", mask)
        return mask

    def find_blobs(self, mask):
        """Find contours in the diff mask, merge nearby ones, and filter by area.

        Dart shafts and flights often produce separate blobs in the diff mask.
        This method merges blobs whose centroids are within BLOB_MERGE_DISTANCE
        so that each physical dart becomes a single blob with a correct tip.

        Args:
            mask: Binary uint8 mask from get_diff_mask.

        Returns:
            List of dicts: [{"contour": cnt, "centroid": (cx, cy),
                             "bbox": (x, y, w, h), "area": area}, ...]
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Build raw blob list (no area filter yet — small fragments get merged)
        raw = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:  # skip single-pixel noise
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            raw.append({"contour": cnt, "centroid": (cx, cy), "area": area})

        # Merge nearby blobs using union-find
        n = len(raw)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj

        merge_dist = config.BLOB_MERGE_DISTANCE
        for i in range(n):
            for j in range(i + 1, n):
                ci = raw[i]["centroid"]
                cj = raw[j]["centroid"]
                dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
                if dist < merge_dist:
                    union(i, j)

        # Group by root and combine contours
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        blobs = []
        for indices in groups.values():
            merged_cnt = np.vstack([raw[i]["contour"] for i in indices])
            total_area = sum(raw[i]["area"] for i in indices)
            if not (config.MIN_BLOB_AREA <= total_area <= config.MAX_BLOB_AREA):
                continue
            M = cv2.moments(merged_cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            bbox = cv2.boundingRect(merged_cnt)
            blobs.append({
                "contour": merged_cnt,
                "centroid": (cx, cy),
                "bbox": bbox,
                "area": total_area,
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

    def _near_confirmed(self, tip, radius=60):
        """Check if a tip is near any already-confirmed dart.

        Returns the blob_id of the nearby confirmed dart, or None.
        """
        tx, ty = tip
        for cid, dart in self.confirmed_darts.items():
            dx, dy = dart["tip"]
            if ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5 < radius:
                return cid
        return None

    def match_blobs(self, current_blobs):
        """Match current blobs to existing candidates and yield newly confirmed darts.

        Uses a two-stage approach:
        1. Match current blobs to candidates by centroid proximity.
        2. Before yielding a "new" dart, dedup against confirmed_darts by tip
           proximity so that tracking instability doesn't cause re-detections.

        Args:
            current_blobs: List of blob dicts from find_blobs.

        Yields:
            Dict {"tip": (x, y), "blob_id": int, "bbox": (x, y, w, h)} for each
            newly confirmed dart.
        """
        # Use a generous match distance — merged blob centroids shift a lot
        MATCH_DISTANCE = 100  # pixels

        matched_candidate_ids = set()

        # First try to match each blob to an existing candidate
        for blob in current_blobs:
            bx, by = blob["centroid"]
            best_id = None
            best_dist = MATCH_DISTANCE

            for cid, cand in self.candidate_blobs.items():
                if cid in matched_candidate_ids:
                    continue  # already claimed this frame
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
            else:
                # New candidate — but only if not near a confirmed dart
                tip = blob.get("tip", blob["centroid"])
                if self._near_confirmed(tip) is not None:
                    continue  # already known dart, skip
                new_id = self.next_blob_id
                self.next_blob_id += 1
                self.candidate_blobs[new_id] = {
                    "centroid": blob["centroid"],
                    "frames_seen": 1,
                    "frames_missed": 0,
                    "tip": tip,
                    "bbox": blob["bbox"],
                }

        # Allow candidates a grace period of 3 missed frames before removal
        gone = []
        for cid in self.candidate_blobs:
            if cid not in matched_candidate_ids:
                self.candidate_blobs[cid]["frames_missed"] += 1
                if self.candidate_blobs[cid]["frames_missed"] > 3:
                    gone.append(cid)
            else:
                self.candidate_blobs[cid]["frames_missed"] = 0
        for cid in gone:
            del self.candidate_blobs[cid]

        # Promote persistent candidates to confirmed darts
        for cid, cand in list(self.candidate_blobs.items()):
            if cand["frames_seen"] >= config.PERSISTENCE_FRAMES and cid not in self.confirmed_darts:
                tip = cand["tip"]
                # Final dedup: don't promote if near an existing confirmed dart
                if self._near_confirmed(tip) is not None:
                    continue
                self.confirmed_darts[cid] = {
                    "tip": tip,
                    "bbox": cand["bbox"],
                    "blob_id": cid,
                }
                yield {"tip": tip, "blob_id": cid, "bbox": cand["bbox"]}

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
