#!/usr/bin/env python3
"""
train.py — YOLO training for dart tip detection + scoring.

Wraps ultralytics YOLOv8 training with the dartscorer dataset.

Usage:
    python train.py                         # Train from scratch
    python train.py --resume                # Resume from last checkpoint
    python train.py --weights best          # Fine-tune from best.pt
    python train.py --balanced              # Balanced pre-train then fine-tune on all
    python train.py --epochs 200            # Custom epoch count
    python train.py --eval                  # Evaluate current model on training set
    python train.py --snapshot "v1 before balanced"  # Save model snapshot
    python train.py --list-snapshots        # List saved snapshots
    python train.py --restore v1            # Restore a snapshot
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import config
from classes import NUM_CLASSES, ID_TO_CLASS, CLASS_TO_ID, parse_class_name


DATASET_YAML = config.PROJECT_ROOT / "data" / "training" / "dataset.yaml"
RUNS_DIR = config.PROJECT_ROOT / "runs"
SNAPSHOTS_DIR = config.PROJECT_ROOT / "runs" / "snapshots"


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

def snapshot(name, description=""):
    """Save a snapshot of the current best model."""
    best_pt = RUNS_DIR / "detect" / "dartscorer" / "weights" / "best.pt"
    results_csv = RUNS_DIR / "detect" / "dartscorer" / "results.csv"

    if not best_pt.exists():
        print("ERROR: No best.pt found to snapshot")
        return

    # Sanitize name
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    snap_dir = SNAPSHOTS_DIR / safe_name
    snap_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(best_pt, snap_dir / "best.pt")
    if results_csv.exists():
        shutil.copy2(results_csv, snap_dir / "results.csv")

    # Read last line of results for metrics
    metrics = {}
    if results_csv.exists():
        with open(results_csv) as f:
            lines = f.readlines()
            if lines:
                parts = lines[-1].strip().split(",")
                if len(parts) >= 8:
                    metrics = {
                        "epoch": parts[0],
                        "precision": parts[5],
                        "recall": parts[6],
                        "mAP50": parts[7],
                        "mAP50-95": parts[8] if len(parts) > 8 else "",
                    }

    meta = {
        "name": name,
        "description": description,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "n_images": len(list((DATASET_YAML.parent / "images").glob("*.png"))),
        "n_labels": len(list((DATASET_YAML.parent / "labels").glob("*.txt"))),
    }
    with open(snap_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Snapshot saved: {safe_name}/")
    print(f"  {meta['timestamp']} | {meta['n_images']} images")
    if metrics:
        print(f"  P={metrics.get('precision', '?')} R={metrics.get('recall', '?')} mAP50={metrics.get('mAP50', '?')}")


def list_snapshots():
    """List all saved snapshots."""
    if not SNAPSHOTS_DIR.exists():
        print("No snapshots found.")
        return

    snaps = sorted(SNAPSHOTS_DIR.iterdir())
    if not snaps:
        print("No snapshots found.")
        return

    print(f"\n{'Name':<25s} {'Date':<20s} {'Images':>7s} {'P':>8s} {'R':>8s} {'mAP50':>8s} Description")
    print("-" * 95)
    for snap_dir in snaps:
        meta_path = snap_dir / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        m = meta.get("metrics", {})
        print(f"{meta['name']:<25s} {meta['timestamp']:<20s} {meta.get('n_images', '?'):>7} "
              f"{m.get('precision', '?'):>8s} {m.get('recall', '?'):>8s} {m.get('mAP50', '?'):>8s} "
              f"{meta.get('description', '')}")


def restore_snapshot(name):
    """Restore a snapshot as the current best model."""
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    snap_dir = SNAPSHOTS_DIR / safe_name
    if not snap_dir.exists():
        # Try partial match
        matches = [d for d in SNAPSHOTS_DIR.iterdir() if safe_name in d.name]
        if len(matches) == 1:
            snap_dir = matches[0]
        elif matches:
            print(f"Ambiguous name, matches: {[d.name for d in matches]}")
            return
        else:
            print(f"Snapshot '{name}' not found")
            return

    best_src = snap_dir / "best.pt"
    if not best_src.exists():
        print(f"ERROR: No best.pt in snapshot {snap_dir.name}")
        return

    dest_dir = RUNS_DIR / "detect" / "dartscorer" / "weights"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_src, dest_dir / "best.pt")

    with open(snap_dir / "meta.json") as f:
        meta = json.load(f)
    print(f"Restored: {meta['name']} ({meta['timestamp']})")


# ---------------------------------------------------------------------------
# Balanced training
# ---------------------------------------------------------------------------

def create_balanced_subset(outdir="data/training", samples_per_class=5):
    """Create a balanced subset of the training data.

    Returns path to a temporary dataset.yaml for the balanced subset.
    """
    outdir = Path(outdir)
    img_dir = outdir / "images"
    label_dir = outdir / "labels"
    bal_dir = outdir / "_balanced"
    bal_img = bal_dir / "images"
    bal_lbl = bal_dir / "labels"

    # Clean previous
    if bal_dir.exists():
        shutil.rmtree(bal_dir)
    bal_img.mkdir(parents=True)
    bal_lbl.mkdir(parents=True)

    # Count classes per label file
    file_classes = {}
    for lp in sorted(label_dir.glob("*.txt")):
        classes_in_file = set()
        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    classes_in_file.add(int(parts[0]))
        file_classes[lp.stem] = classes_in_file

    # Track how many times each class has been selected
    class_counts = Counter()
    selected = set()

    # Priority: select files that contain rare classes first
    # Sort files by rarest class they contain
    all_class_counts = Counter()
    for classes in file_classes.values():
        for c in classes:
            all_class_counts[c] += 1

    def file_priority(stem):
        classes = file_classes[stem]
        if not classes:
            return 999999
        return min(all_class_counts.get(c, 0) for c in classes)

    sorted_files = sorted(file_classes.keys(), key=file_priority)

    for stem in sorted_files:
        classes = file_classes[stem]
        # Select if any class in this file needs more samples
        needs_more = any(class_counts[c] < samples_per_class for c in classes)
        if needs_more:
            selected.add(stem)
            for c in classes:
                class_counts[c] += 1

    # Copy selected files
    for stem in selected:
        src_img = img_dir / f"{stem}.png"
        src_lbl = label_dir / f"{stem}.txt"
        if src_img.exists() and src_lbl.exists():
            shutil.copy2(src_img, bal_img / f"{stem}.png")
            shutil.copy2(src_lbl, bal_lbl / f"{stem}.txt")

    # Create balanced dataset.yaml
    bal_yaml = bal_dir / "dataset.yaml"
    from classes import CLASS_NAMES
    lines = [
        f"# Balanced subset — {len(selected)} images",
        f"path: {bal_dir.resolve()}",
        "train: images",
        "val: images",
        "",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")
    with open(bal_yaml, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Stats
    n_classes_covered = sum(1 for c in range(NUM_CLASSES) if class_counts[c] > 0)
    print(f"Balanced subset: {len(selected)} images, {n_classes_covered}/{NUM_CLASSES} classes covered")
    print(f"  Saved to {bal_dir}")

    return str(bal_yaml)


def train_balanced(epochs=100, batch=16, imgsz=640, device=None, samples_per_class=5):
    """Two-phase training: balanced pre-train then fine-tune on full dataset."""
    from ultralytics import YOLO

    print("=== Phase 1: Balanced pre-training ===")
    bal_yaml = create_balanced_subset(samples_per_class=samples_per_class)

    model = YOLO("yolov8n.pt")
    model.train(
        data=bal_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(RUNS_DIR / "detect"),
        name="dartscorer_balanced",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
        **({"device": device} if device else {}),
    )

    balanced_best = RUNS_DIR / "detect" / "dartscorer_balanced" / "weights" / "best.pt"
    if not balanced_best.exists():
        print("ERROR: Balanced training failed to produce weights")
        return

    print(f"\n=== Phase 2: Fine-tune on full dataset ===")
    model = YOLO(str(balanced_best))
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(RUNS_DIR / "detect"),
        name="dartscorer",
        exist_ok=True,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        **({"device": device} if device else {}),
    )

    print(f"\nBalanced training complete. Results in {RUNS_DIR / 'detect' / 'dartscorer'}")
    return results


# ---------------------------------------------------------------------------
# Standard training
# ---------------------------------------------------------------------------

def train(epochs=100, batch=16, imgsz=640, resume=False, weights=None, device=None):
    """Train YOLOv8 on the dartscorer dataset."""
    from ultralytics import YOLO

    if not DATASET_YAML.exists():
        print(f"ERROR: Dataset config not found at {DATASET_YAML}")
        sys.exit(1)

    img_dir = DATASET_YAML.parent / "images"
    label_dir = DATASET_YAML.parent / "labels"
    n_images = len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpg")))
    n_labels = len(list(label_dir.glob("*.txt")))
    print(f"Dataset: {n_images} images, {n_labels} label files")

    if n_images == 0:
        print("ERROR: No training images found. Run collect.py first.")
        sys.exit(1)

    if resume:
        last_pt = RUNS_DIR / "detect" / "dartscorer" / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"ERROR: No checkpoint found at {last_pt}")
            sys.exit(1)
        model = YOLO(str(last_pt))
        print(f"Resuming from {last_pt}")
    elif weights:
        if weights == "best":
            weights_path = RUNS_DIR / "detect" / "dartscorer" / "weights" / "best.pt"
        else:
            weights_path = Path(weights)
        if not weights_path.exists():
            print(f"ERROR: Weights not found at {weights_path}")
            sys.exit(1)
        model = YOLO(str(weights_path))
        print(f"Fine-tuning from {weights_path}")
    else:
        model = YOLO("yolov8n.pt")
        print("Training from pretrained YOLOv8n")

    train_args = dict(
        data=str(DATASET_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(RUNS_DIR / "detect"),
        name="dartscorer",
        exist_ok=True,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )
    if device is not None:
        train_args["device"] = device
    if resume:
        train_args["resume"] = True

    results = model.train(**train_args)
    print(f"\nTraining complete. Results saved to {RUNS_DIR / 'detect' / 'dartscorer'}")
    return results


def evaluate():
    """Evaluate the current best model on the training set."""
    from ultralytics import YOLO

    best_pt = RUNS_DIR / "detect" / "dartscorer" / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"ERROR: No trained model found at {best_pt}")
        print("Run 'python train.py' first.")
        sys.exit(1)

    model = YOLO(str(best_pt))
    print(f"Evaluating {best_pt}")

    results = model.val(data=str(DATASET_YAML))

    if hasattr(results, 'results_dict'):
        print(f"\n{'Class':<20} {'P':>8} {'R':>8} {'mAP50':>8}")
        print("-" * 50)
        print(f"{'all':<20} "
              f"{results.results_dict.get('metrics/precision(B)', 0):.3f}   "
              f"{results.results_dict.get('metrics/recall(B)', 0):.3f}   "
              f"{results.results_dict.get('metrics/mAP50(B)', 0):.3f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train YOLO dart detector")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights, or 'best' for best.pt")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (e.g. 'cpu', '0', '0,1')")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate model instead of training")
    parser.add_argument("--balanced", action="store_true",
                        help="Two-phase: balanced pre-train then fine-tune on all")
    parser.add_argument("--snapshot", type=str, default=None,
                        help="Save a named snapshot of current best model")
    parser.add_argument("--list-snapshots", action="store_true",
                        help="List saved model snapshots")
    parser.add_argument("--restore", type=str, default=None,
                        help="Restore a snapshot as current best model")
    args = parser.parse_args()

    if args.list_snapshots:
        list_snapshots()
    elif args.snapshot:
        snapshot(args.snapshot)
    elif args.restore:
        restore_snapshot(args.restore)
    elif args.eval:
        evaluate()
    elif args.balanced:
        train_balanced(
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
        )
    else:
        train(
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            resume=args.resume,
            weights=args.weights,
            device=args.device,
        )


if __name__ == "__main__":
    main()
