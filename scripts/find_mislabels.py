#!/usr/bin/env python3
"""
find_mislabels.py — Find likely mislabeled training data.

Runs the current model against the training set and flags frames where
the model's prediction disagrees with the label. High-confidence
disagreements are the most likely mislabels.

Usage:
    uv run python find_mislabels.py                # Summary report
    uv run python find_mislabels.py --browse       # Interactive browser
    uv run python find_mislabels.py --fix          # Auto-fix high-confidence mismatches
"""

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

from dartscorer import config
from dartscorer.classes_v2 import CLASS_NAMES, CLASS_TO_ID, ID_TO_CLASS, NUM_CLASSES, parse_class_name


def find_disagreements(conf_threshold=0.5):
    """Compare model predictions against labels for all training data.

    Returns list of dicts sorted by severity (highest confidence
    disagreements first).
    """
    from ultralytics import YOLO

    best_pt = config.PROJECT_ROOT / "runs" / "detect" / "dartscorer" / "weights" / "best.pt"
    if not best_pt.exists():
        print("ERROR: No model found. Train first.")
        return []

    model = YOLO(str(best_pt))
    img_dir = config.DATASET_IMAGES_DIR
    label_dir = config.DATASET_LABELS_DIR

    disagreements = []
    total_annotations = 0
    total_matches = 0

    image_files = sorted(img_dir.glob("*.png"))
    print(f"Scanning {len(image_files)} images...")

    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        # Read labels
        labels = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy = float(parts[1]), float(parts[2])
                    labels.append({
                        "class_id": cid,
                        "class_name": CLASS_NAMES[cid] if cid < NUM_CLASSES else f"?{cid}",
                        "cx": cx, "cy": cy,
                    })

        if not labels:
            continue

        # Run model
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        results = model.predict(img, conf=conf_threshold, verbose=False)
        if not results or results[0].boxes is None:
            # Model found nothing but labels exist
            for lab in labels:
                total_annotations += 1
                disagreements.append({
                    "file": img_path.name,
                    "stem": img_path.stem,
                    "label_class": lab["class_name"],
                    "label_id": lab["class_id"],
                    "pred_class": "(not detected)",
                    "pred_id": -1,
                    "pred_conf": 0.0,
                    "type": "undetected",
                    "cx": lab["cx"], "cy": lab["cy"],
                })
            continue

        # Match predictions to labels by proximity
        preds = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pcx = ((x1 + x2) / 2) / w
            pcy = ((y1 + y2) / 2) / h
            pcid = int(box.cls[0])
            pconf = float(box.conf[0])
            preds.append({
                "class_id": pcid,
                "class_name": CLASS_NAMES[pcid] if pcid < NUM_CLASSES else f"?{pcid}",
                "conf": pconf,
                "cx": pcx, "cy": pcy,
            })

        # Optimal matching using Hungarian algorithm (scipy)
        # Greedy nearest-neighbor fails when darts are close together
        from scipy.optimize import linear_sum_assignment

        n_lab = len(labels)
        n_pred = len(preds)

        if n_pred == 0:
            for lab in labels:
                total_annotations += 1
                disagreements.append({
                    "file": img_path.name,
                    "stem": img_path.stem,
                    "label_class": lab["class_name"],
                    "label_id": lab["class_id"],
                    "pred_class": "(not detected)",
                    "pred_id": -1,
                    "pred_conf": 0.0,
                    "type": "undetected",
                    "cx": lab["cx"], "cy": lab["cy"],
                })
            continue

        # Build cost matrix (distance between each label and prediction)
        cost = np.zeros((n_lab, n_pred))
        for li, lab in enumerate(labels):
            for pi, pred in enumerate(preds):
                cost[li, pi] = ((lab["cx"] - pred["cx"]) ** 2 +
                                (lab["cy"] - pred["cy"]) ** 2) ** 0.5

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_labels = set()
        for li, pi in zip(row_ind, col_ind):
            lab = labels[li]
            pred = preds[pi]
            dist = cost[li, pi]
            total_annotations += 1
            matched_labels.add(li)

            if dist > 0.15:
                # Matched but too far — treat as undetected
                disagreements.append({
                    "file": img_path.name,
                    "stem": img_path.stem,
                    "label_class": lab["class_name"],
                    "label_id": lab["class_id"],
                    "pred_class": "(not detected)",
                    "pred_id": -1,
                    "pred_conf": 0.0,
                    "type": "undetected",
                    "cx": lab["cx"], "cy": lab["cy"],
                })
            elif pred["class_id"] != lab["class_id"]:
                disagreements.append({
                    "file": img_path.name,
                    "stem": img_path.stem,
                    "label_class": lab["class_name"],
                    "label_id": lab["class_id"],
                    "pred_class": pred["class_name"],
                    "pred_id": pred["class_id"],
                    "pred_conf": pred["conf"],
                    "type": "mismatch",
                    "cx": lab["cx"], "cy": lab["cy"],
                })
            else:
                total_matches += 1

        # Unmatched labels (more labels than predictions)
        for li, lab in enumerate(labels):
            if li not in matched_labels:
                total_annotations += 1
                disagreements.append({
                    "file": img_path.name,
                    "stem": img_path.stem,
                    "label_class": lab["class_name"],
                    "label_id": lab["class_id"],
                    "pred_class": "(not detected)",
                    "pred_id": -1,
                    "pred_conf": 0.0,
                    "type": "undetected",
                    "cx": lab["cx"], "cy": lab["cy"],
                })

    # Sort by confidence (highest confidence disagreements = most likely mislabels)
    disagreements.sort(key=lambda d: -d["pred_conf"])

    print(f"\nTotal annotations: {total_annotations}")
    print(f"Matches: {total_matches} ({total_matches / total_annotations * 100:.1f}%)")
    print(f"Disagreements: {len(disagreements)} ({len(disagreements) / total_annotations * 100:.1f}%)")

    return disagreements


def print_report(disagreements):
    """Print a summary of likely mislabels."""
    if not disagreements:
        print("No disagreements found.")
        return

    mismatches = [d for d in disagreements if d["type"] == "mismatch"]
    undetected = [d for d in disagreements if d["type"] == "undetected"]

    print(f"\n{'=' * 70}")
    print(f"LIKELY MISLABELS (model confident but label disagrees)")
    print(f"{'=' * 70}")

    # High-confidence mismatches are the most suspicious
    high_conf = [d for d in mismatches if d["pred_conf"] > 0.8]
    med_conf = [d for d in mismatches if 0.5 < d["pred_conf"] <= 0.8]

    print(f"\n  High confidence (>80%) mismatches: {len(high_conf)}")
    print(f"  Medium confidence (50-80%) mismatches: {len(med_conf)}")
    print(f"  Undetected (label exists, model sees nothing): {len(undetected)}")

    if high_conf:
        print(f"\n  --- Most likely mislabels (fix these first) ---")
        for d in high_conf[:20]:
            print(f"  {d['file']:50s}  label={d['label_class']:>8s}  "
                  f"model={d['pred_class']:>8s} ({d['pred_conf']:.0%})")

    # What classes are most confused?
    confusion = Counter()
    for d in mismatches:
        confusion[(d["label_class"], d["pred_class"])] += 1

    if confusion:
        print(f"\n  --- Most common confusions ---")
        for (label, pred), count in confusion.most_common(15):
            print(f"    {label:>8s} labeled as → model says {pred:>8s}  ({count}x)")

    # Which classes have the most issues?
    class_issues = Counter()
    for d in disagreements:
        class_issues[d["label_class"]] += 1

    print(f"\n  --- Classes with most disagreements ---")
    for cls, count in class_issues.most_common(10):
        print(f"    {cls:>8s}: {count} disagreements")


def browse_disagreements(disagreements):
    """Interactive browser for reviewing disagreements."""
    if not disagreements:
        print("No disagreements to browse.")
        return

    # Filter to mismatches only (more actionable than undetected)
    mismatches = [d for d in disagreements if d["type"] == "mismatch"]
    if not mismatches:
        print("No class mismatches found.")
        return

    print(f"\nBrowsing {len(mismatches)} mismatches (highest confidence first)")
    print("Controls: SPACE=next  B=back  F=fix label to model's prediction  Q=quit\n")

    img_dir = config.DATASET_IMAGES_DIR
    label_dir = config.DATASET_LABELS_DIR

    win = "Mislabel Browser"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    idx = 0
    fixed = 0

    while 0 <= idx < len(mismatches):
        d = mismatches[idx]
        img_path = img_dir / d["file"]
        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue

        h, w = img.shape[:2]
        display = img.copy()

        # Draw the annotation location
        px, py = int(d["cx"] * w), int(d["cy"] * h)
        cv2.circle(display, (px, py), 12, (0, 0, 255), 2)
        cv2.circle(display, (px, py), 3, (0, 0, 255), -1)

        # Label info
        label_info = parse_class_name(d["label_class"])
        pred_info = parse_class_name(d["pred_class"]) if d["pred_class"] != "(not detected)" else None

        text_y = 30
        cv2.putText(display, f"[{idx + 1}/{len(mismatches)}] {d['file']}",
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        text_y += 30
        cv2.putText(display, f"Label: {d['label_class']} ({label_info['label']})",
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        text_y += 30
        if pred_info:
            cv2.putText(display, f"Model: {d['pred_class']} ({pred_info['label']}) @ {d['pred_conf']:.0%}",
                        (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        text_y += 30
        cv2.putText(display, "SPACE=next  B=back  F=fix to model  Q=quit",
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow(win, display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            idx += 1
        elif key == ord("b"):
            idx = max(0, idx - 1)
        elif key == ord("f"):
            # Fix: replace label class with model's prediction
            label_path = label_dir / f"{d['stem']}.txt"
            if label_path.exists():
                lines = []
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            lcx, lcy = float(parts[1]), float(parts[2])
                            # Match by position
                            if (abs(lcx - d["cx"]) < 0.01 and
                                    abs(lcy - d["cy"]) < 0.01 and
                                    int(parts[0]) == d["label_id"]):
                                parts[0] = str(d["pred_id"])
                            lines.append(" ".join(parts))
                        else:
                            lines.append(line.strip())
                with open(label_path, "w") as f:
                    f.write("\n".join(lines) + "\n")
                print(f"  Fixed: {d['file']} {d['label_class']} → {d['pred_class']}")
                fixed += 1
            idx += 1

    cv2.destroyAllWindows()
    print(f"\nFixed {fixed} labels.")


def auto_fix(disagreements, min_conf=0.9):
    """Automatically fix labels where model is very confident."""
    mismatches = [d for d in disagreements
                  if d["type"] == "mismatch" and d["pred_conf"] >= min_conf]

    if not mismatches:
        print(f"No mismatches above {min_conf:.0%} confidence to auto-fix.")
        return

    label_dir = config.DATASET_LABELS_DIR
    fixed = 0

    for d in mismatches:
        label_path = label_dir / f"{d['stem']}.txt"
        if not label_path.exists():
            continue

        lines = []
        changed = False
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    lcx, lcy = float(parts[1]), float(parts[2])
                    if (abs(lcx - d["cx"]) < 0.01 and
                            abs(lcy - d["cy"]) < 0.01 and
                            int(parts[0]) == d["label_id"]):
                        parts[0] = str(d["pred_id"])
                        changed = True
                    lines.append(" ".join(parts))
                else:
                    lines.append(line.strip())

        if changed:
            with open(label_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            fixed += 1

    print(f"Auto-fixed {fixed} labels (model confidence >= {min_conf:.0%})")
    if fixed:
        print("Review with --browse to verify, then retrain.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find likely mislabeled training data")
    parser.add_argument("--browse", action="store_true",
                        help="Interactive browser for reviewing disagreements")
    parser.add_argument("--fix", action="store_true",
                        help="Auto-fix labels where model is >90%% confident")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Minimum model confidence to consider (default: 0.5)")
    args = parser.parse_args()

    disagreements = find_disagreements(conf_threshold=args.conf)

    if args.fix:
        auto_fix(disagreements)
    elif args.browse:
        browse_disagreements(disagreements)
    else:
        print_report(disagreements)
