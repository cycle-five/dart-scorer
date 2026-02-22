#!/usr/bin/env python3
"""
optimize.py — Bayesian optimization of dart detection parameters.

Uses Optuna to search for the best combination of detection thresholds
by evaluating the classical CV pipeline against labeled training data.

Usage:
    python optimize.py                          # Run 200 trials
    python optimize.py --trials 500             # More trials
    python optimize.py --datadir data/training  # Custom data dir
    python optimize.py --apply                  # Apply best params to config.py
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Suppress optuna's verbose logging
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Evaluation engine — runs detection pipeline on static frames
# ---------------------------------------------------------------------------

def build_background(img_dir, annotations, blur_ksize):
    """Build a median background from all background frames."""
    bg_frames = []
    for entry in annotations:
        if not entry.get("is_background", False):
            continue
        path = img_dir / entry["filename"]
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg_frames.append(gray)

    if not bg_frames:
        # No backgrounds labeled — use frames with 0 darts
        for entry in annotations:
            if entry.get("n_darts", 0) == 0:
                path = img_dir / entry["filename"]
                img = cv2.imread(str(path))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                bg_frames.append(gray)

    if not bg_frames:
        return None

    background = np.median(np.array(bg_frames), axis=0).astype(np.uint8)
    return background


def detect_tips_static(gray_frame, background, board_center, params):
    """Run the detection pipeline on a single frame with given parameters.

    Returns list of detected tip (x, y) positions.
    """
    blur_ksize = params["blur_ksize"]
    diff_threshold = params["diff_threshold"]
    open_ksize = params["open_ksize"]
    close_ksize = params["close_ksize"]
    min_blob_area = params["min_blob_area"]
    max_blob_area = params["max_blob_area"]
    merge_distance = params["merge_distance"]

    # Blur
    bg_blur = cv2.GaussianBlur(background, (blur_ksize, blur_ksize), 0)
    fr_blur = cv2.GaussianBlur(gray_frame, (blur_ksize, blur_ksize), 0)

    # Diff + threshold
    diff = cv2.absdiff(bg_blur, fr_blur)
    _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    # Morphology
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        raw.append({"contour": cnt, "centroid": (cx, cy), "area": area})

    # Merge nearby blobs (union-find)
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

    for i in range(n):
        for j in range(i + 1, n):
            ci = raw[i]["centroid"]
            cj = raw[j]["centroid"]
            dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
            if dist < merge_distance:
                union(i, j)

    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    # Find tips
    tips = []
    bcx, bcy = board_center
    for indices in groups.values():
        merged_cnt = np.vstack([raw[i]["contour"] for i in indices])
        total_area = sum(raw[i]["area"] for i in indices)
        if not (min_blob_area <= total_area <= max_blob_area):
            continue
        # Tip = closest point to board center
        pts = merged_cnt.reshape(-1, 2)
        dists = (pts[:, 0] - bcx) ** 2 + (pts[:, 1] - bcy) ** 2
        idx = int(np.argmin(dists))
        tips.append((int(pts[idx, 0]), int(pts[idx, 1])))

    return tips


def score_detections(detected_tips, ground_truth_tips, match_radius=30):
    """Score detected tips against ground truth.

    Returns (true_positives, false_positives, false_negatives, mean_error).
    A detected tip matches a ground truth tip if within match_radius pixels.
    Each ground truth tip can only be matched once (greedy nearest).
    """
    if not ground_truth_tips:
        # No darts in frame — any detection is a false positive
        return 0, len(detected_tips), 0, 0.0

    if not detected_tips:
        return 0, 0, len(ground_truth_tips), 0.0

    gt = list(ground_truth_tips)
    det = list(detected_tips)

    # Build distance matrix
    distances = []
    for di, (dx, dy) in enumerate(det):
        for gi, (gx, gy) in enumerate(gt):
            dist = ((dx - gx) ** 2 + (dy - gy) ** 2) ** 0.5
            distances.append((dist, di, gi))
    distances.sort()

    matched_det = set()
    matched_gt = set()
    errors = []

    for dist, di, gi in distances:
        if di in matched_det or gi in matched_gt:
            continue
        if dist > match_radius:
            break
        matched_det.add(di)
        matched_gt.add(gi)
        errors.append(dist)

    tp = len(matched_det)
    fp = len(det) - tp
    fn = len(gt) - len(matched_gt)
    mean_err = float(np.mean(errors)) if errors else 0.0

    return tp, fp, fn, mean_err


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _group_into_rounds(annotations, img_dir, inject_backgrounds=False):
    """Group annotations into rounds of sequential throws.

    A round starts when dart count resets (drops to 0, or decreases from
    previous frame indicating darts were pulled).  Each round is a sequence
    of frames with non-decreasing dart counts.

    Rounds that start without a background frame (e.g. 3→1 transitions)
    get a background injected from the collected background frames, since
    the board and lighting are largely static.

    Returns list of rounds, where each round is a list of
    {"gray": ndarray, "tips": [(x,y),...], "n_darts": int}.
    """
    # Sort by timestamp to get capture order
    sorted_ann = sorted(annotations, key=lambda a: a.get("timestamp", 0))

    # Collect all background frames for injection
    bg_grays = []
    for entry in sorted_ann:
        if entry.get("n_darts", 0) == 0:
            path = img_dir / entry["filename"]
            img = cv2.imread(str(path))
            if img is not None:
                bg_grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # Build a median background if we have multiple (filter to most common size)
    bg_inject = None
    if bg_grays:
        from collections import Counter
        sizes = [g.shape for g in bg_grays]
        most_common_size = Counter(sizes).most_common(1)[0][0]
        bg_same = [g for g in bg_grays if g.shape == most_common_size]
        if bg_same:
            bg_inject = np.median(np.array(bg_same), axis=0).astype(np.uint8)

    rounds = []
    current_round = []
    prev_n = -1

    for entry in sorted_ann:
        path = img_dir / entry["filename"]
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        n = entry.get("n_darts", 0)
        tips = [tuple(t) for t in entry.get("tips", [])]

        item = {"gray": gray, "tips": tips, "n_darts": n}

        if n < prev_n or (n == 0 and prev_n == 0):
            # Dart count decreased (darts pulled) or consecutive backgrounds
            # — start a new round
            if current_round:
                rounds.append(current_round)
            current_round = [item]
        else:
            current_round.append(item)

        prev_n = n

    if current_round:
        rounds.append(current_round)

    # Optionally inject background into rounds that lack one
    if inject_backgrounds and bg_inject is not None:
        bg_item = {"gray": bg_inject, "tips": [], "n_darts": 0}
        for i, rnd in enumerate(rounds):
            if rnd[0]["n_darts"] > 0:
                rounds[i] = [bg_item] + rnd

    return rounds


def _find_new_tips(prev_tips, curr_tips, radius=40):
    """Find tips in curr_tips that aren't near any tip in prev_tips."""
    new = []
    for (cx, cy) in curr_tips:
        is_old = False
        for (px, py) in prev_tips:
            if ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 < radius:
                is_old = True
                break
        if not is_old:
            new.append((cx, cy))
    return new


def make_objective(annotations, img_dir, background_frames, board_center):
    """Create an Optuna objective function using sequential evaluation.

    Simulates real play: for each round of throws, the background starts
    as the empty board and is updated after each detected dart, so the
    detector only needs to find the ONE new dart per frame.
    """
    rounds = _group_into_rounds(annotations, img_dir)

    total_gt_tips = 0
    total_frames = 0
    for r in rounds:
        for item in r:
            if item["n_darts"] > 0:
                total_frames += 1
        # Count new tips per step
        prev_tips = []
        for item in r:
            new = _find_new_tips(prev_tips, item["tips"])
            total_gt_tips += len(new)
            prev_tips = item["tips"]

    print(f"Loaded {len(rounds)} rounds, {total_frames} frames with darts, {total_gt_tips} new-dart events")

    def objective(trial):
        params = {
            "blur_ksize": trial.suggest_int("blur_ksize", 3, 15, step=2),
            "diff_threshold": trial.suggest_int("diff_threshold", 20, 70),
            "open_ksize": trial.suggest_int("open_ksize", 3, 11, step=2),
            "close_ksize": trial.suggest_int("close_ksize", 5, 21, step=2),
            "min_blob_area": trial.suggest_int("min_blob_area", 50, 600, step=25),
            "max_blob_area": trial.suggest_int("max_blob_area", 3000, 20000, step=500),
            "merge_distance": trial.suggest_int("merge_distance", 40, 200, step=10),
        }

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_err = 0.0
        n_matched = 0

        for rnd in rounds:
            # First frame in round should be background (or use it to init)
            background = rnd[0]["gray"].copy()
            prev_tips = rnd[0]["tips"]  # usually empty

            for item in rnd[1:]:
                # What's new in this frame?
                new_gt_tips = _find_new_tips(prev_tips, item["tips"])
                if not new_gt_tips and item["n_darts"] == len(prev_tips):
                    # No new darts, skip (duplicate frame)
                    prev_tips = item["tips"]
                    continue

                # Detect against current background
                detected = detect_tips_static(
                    item["gray"], background, board_center, params
                )

                # Score only the NEW tips (not all tips in frame)
                tp, fp, fn, mean_err = score_detections(
                    detected, new_gt_tips, match_radius=40
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn
                if tp > 0:
                    total_err += mean_err * tp
                    n_matched += tp

                # Absorb: update background to current frame (simulates absorb_current_scene)
                background = item["gray"].copy()
                prev_tips = item["tips"]

        # F1-based score with error penalty
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Penalize positional error
        mean_error = total_err / n_matched if n_matched > 0 else 40.0
        error_penalty = max(0, 1.0 - mean_error / 40.0)

        score = 0.8 * f1 + 0.2 * error_penalty
        return score

    return objective, total_frames


# ---------------------------------------------------------------------------
# Estimate board center from annotations
# ---------------------------------------------------------------------------

def estimate_board_center(annotations):
    """Estimate the board center as the centroid of all annotated tips."""
    all_tips = []
    for entry in annotations:
        for tip in entry.get("tips", []):
            all_tips.append(tip)
    if not all_tips:
        return (960, 540)  # fallback to frame center
    arr = np.array(all_tips)
    return (int(np.mean(arr[:, 0])), int(np.mean(arr[:, 1])))


# ---------------------------------------------------------------------------
# Apply results to config.py
# ---------------------------------------------------------------------------

def apply_to_config(best_params, config_path="config.py"):
    """Update config.py with optimized parameters."""
    mapping = {
        "diff_threshold": ("DIFF_THRESHOLD", int),
        "min_blob_area": ("MIN_BLOB_AREA", int),
        "max_blob_area": ("MAX_BLOB_AREA", int),
        "merge_distance": ("BLOB_MERGE_DISTANCE", int),
        "blur_ksize": ("BLUR_KSIZE", int),
        "open_ksize": ("OPEN_KSIZE", int),
        "close_ksize": ("CLOSE_KSIZE", int),
    }

    with open(config_path) as f:
        lines = f.readlines()

    for param_key, (config_name, cast) in mapping.items():
        if param_key not in best_params:
            continue
        value = cast(best_params[param_key])
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{config_name} =") or line.strip().startswith(f"{config_name}="):
                # Preserve the comment
                parts = line.split("#", 1)
                comment = f"  # {parts[1].strip()}" if len(parts) > 1 else ""
                lines[i] = f"{config_name} = {value}{comment}\n"
                break

    with open(config_path, "w") as f:
        f.writelines(lines)

    print(f"Updated {config_path} with optimized parameters")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optimize dart detection parameters")
    parser.add_argument("--datadir", default="data/training",
                        help="Training data directory")
    parser.add_argument("--trials", type=int, default=200,
                        help="Number of Optuna trials (default: 200)")
    parser.add_argument("--match-radius", type=int, default=40,
                        help="Max pixels between detected and ground truth tip to count as match")
    parser.add_argument("--apply", action="store_true",
                        help="Apply best parameters to config.py")
    parser.add_argument("--show-best", action="store_true",
                        help="Show best parameters from previous run without optimizing")
    args = parser.parse_args()

    datadir = Path(args.datadir)
    annotations_path = datadir / "annotations.jsonl"
    img_dir = datadir / "images"

    if not annotations_path.exists():
        print(f"No annotations found at {annotations_path}")
        print("Run 'python collect.py' first to collect training data")
        sys.exit(1)

    # Load annotations
    annotations = []
    with open(annotations_path) as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))

    n_with_darts = sum(1 for a in annotations if a.get("n_darts", 0) > 0)
    n_bg = sum(1 for a in annotations if a.get("n_darts", 0) == 0)
    total_tips = sum(a.get("n_darts", 0) for a in annotations)
    print(f"Dataset: {len(annotations)} frames ({n_with_darts} with darts, {n_bg} backgrounds)")
    print(f"Total dart tips: {total_tips}")

    board_center = estimate_board_center(annotations)
    print(f"Estimated board center: {board_center}")

    # Create objective
    objective, n_frames = make_objective(annotations, img_dir, annotations, board_center)

    if n_frames == 0:
        print("No labeled frames with darts found!")
        sys.exit(1)

    # Run optimization
    print(f"\nRunning {args.trials} optimization trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # Results
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"BEST SCORE: {best.value:.4f}")
    print(f"{'='*60}")
    print("Parameters:")
    for key, value in sorted(best.params.items()):
        print(f"  {key:20s} = {value}")

    # Run final sequential evaluation with best params
    rounds = _group_into_rounds(annotations, img_dir)
    total_tp = total_fp = total_fn = 0
    for rnd in rounds:
        background = rnd[0]["gray"].copy()
        prev_tips = rnd[0]["tips"]
        for item in rnd[1:]:
            new_gt_tips = _find_new_tips(prev_tips, item["tips"])
            if not new_gt_tips and item["n_darts"] == len(prev_tips):
                prev_tips = item["tips"]
                continue
            detected = detect_tips_static(
                item["gray"], background, board_center, best.params
            )
            tp, fp, fn, _ = score_detections(
                detected, new_gt_tips, match_radius=args.match_radius
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
            background = item["gray"].copy()
            prev_tips = item["tips"]

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nDetailed metrics (match_radius={args.match_radius}px):")
    print(f"  True positives:  {total_tp}")
    print(f"  False positives: {total_fp}")
    print(f"  False negatives: {total_fn}")
    print(f"  Precision:       {precision:.3f}")
    print(f"  Recall:          {recall:.3f}")
    print(f"  F1:              {f1:.3f}")

    if args.apply:
        apply_to_config(best.params)

    # Save study results
    results_path = datadir / "optimization_results.json"
    results = {
        "best_score": best.value,
        "best_params": best.params,
        "n_trials": args.trials,
        "n_frames": len(annotations),
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
        },
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
