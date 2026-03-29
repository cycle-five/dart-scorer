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
from classes_v2 import NUM_CLASSES, ID_TO_CLASS, CLASS_TO_ID, CLASS_NAMES, parse_class_name


DATASET_YAML = config.DATASET_YAML_PATH
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

def create_balanced_subset(outdir="data/training", samples_per_class=50):
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
# Focused training (specific classes)
# ---------------------------------------------------------------------------

def create_focused_subset(focus_classes, min_others=0):
    """Create a subset containing frames with specific classes.

    Selects all frames that contain at least one of the focus classes.
    Optionally includes a small number of other frames for stability.

    Args:
        focus_classes: List of class names (e.g. ["D3", "D11", "T10"]).
        min_others: Minimum other frames to include (0 = focus only).

    Returns:
        Path to temporary dataset.yaml, or None on error.
    """
    import random

    data_dir = DATASET_YAML.parent
    img_dir = data_dir / "images"
    label_dir = data_dir / "labels"
    focus_dir = data_dir / "_focused"
    focus_img = focus_dir / "images"
    focus_lbl = focus_dir / "labels"

    # Map focus class names to IDs
    focus_ids = set()
    for name in focus_classes:
        cid = CLASS_TO_ID.get(name)
        if cid is None:
            print(f"WARNING: Unknown class '{name}', skipping")
        else:
            focus_ids.add(cid)

    if not focus_ids:
        print("ERROR: No valid focus classes specified")
        return None

    # Clean previous
    if focus_dir.exists():
        import shutil
        shutil.rmtree(focus_dir)
    focus_img.mkdir(parents=True)
    focus_lbl.mkdir(parents=True)

    # Scan labels to find frames containing focus classes
    focus_stems = []
    other_stems = []

    for lp in sorted(label_dir.glob("*.txt")):
        classes_in_file = set()
        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    classes_in_file.add(int(parts[0]))

        if classes_in_file & focus_ids:
            focus_stems.append(lp.stem)
        else:
            other_stems.append(lp.stem)

    # Add some background frames for stability
    if min_others > 0 and other_stems:
        random.shuffle(other_stems)
        selected_others = other_stems[:min_others]
    else:
        selected_others = []

    all_selected = focus_stems + selected_others

    # Symlink files
    for stem in all_selected:
        for ext, src_dir, dst_dir in [(".png", img_dir, focus_img),
                                       (".txt", label_dir, focus_lbl)]:
            src = src_dir / f"{stem}{ext}"
            dst = dst_dir / f"{stem}{ext}"
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve())

    # Create dataset.yaml
    focus_yaml = focus_dir / "dataset.yaml"
    lines = [
        f"# Focused subset: {', '.join(focus_classes)}",
        f"# {len(focus_stems)} focus frames + {len(selected_others)} background",
        "",
        f"path: {focus_dir.resolve()}",
        "train: images",
        "val: images",
        "",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")
    with open(focus_yaml, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Show stats
    focus_counts = Counter()
    for stem in focus_stems:
        lp = label_dir / f"{stem}.txt"
        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    if cid in focus_ids:
                        focus_counts[CLASS_NAMES[cid]] += 1

    print(f"\nFocused subset: {len(all_selected)} frames "
          f"({len(focus_stems)} with target classes, {len(selected_others)} background)")
    for name in focus_classes:
        print(f"  {name}: {focus_counts.get(name, 0)} annotations")

    return str(focus_yaml)


def focus_train(focus_classes, epochs=100, batch=16, imgsz=640, device=None,
                patience=50, min_background=20):
    """Train focused on specific underrepresented classes."""
    from ultralytics import YOLO

    focus_yaml = create_focused_subset(focus_classes, min_others=min_background)
    if focus_yaml is None:
        return

    # Start from current best
    best_pt = RUNS_DIR / "detect" / "dartscorer" / "weights" / "best.pt"
    if not best_pt.exists():
        print("ERROR: No best.pt found. Run full training first.")
        return

    print(f"\nFine-tuning on focused subset (patience={patience})...")
    train(
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        weights="best",
        device=device,
        patience=patience,
        dataset_yaml=focus_yaml,
    )


class ModelInfo:
    """Parsed info about a trained YOLO model."""

    def __init__(self, weights_path=None):
        self.weights_path = weights_path
        self.size_mb = 0.0
        self.last_modified = 0.0
        self.num_classes = 0
        self.num_parameters = 0
        self.class_names = {}

        # Training config (from args.yaml)
        self.dataset = ""
        self.epochs = 0
        self.imgsz = 0
        self.batch = 0
        self.patience = 0

        # Last training metrics (from results.csv)
        self.actual_epochs = 0
        self.precision = 0.0
        self.recall = 0.0
        self.map50 = 0.0
        self.map50_95 = 0.0

        self.has_stale_best = False  # last.pt newer than best.pt

        if weights_path and Path(weights_path).exists():
            self._load(Path(weights_path))

    def _load(self, best_pt):
        import time as _time
        import yaml

        self.weights_path = best_pt
        self.size_mb = best_pt.stat().st_size / 1024 / 1024
        self.last_modified = best_pt.stat().st_mtime

        from ultralytics import YOLO
        model = YOLO(str(best_pt))
        self.num_classes = len(model.names)
        self.class_names = dict(model.names)
        self.num_parameters = sum(p.numel() for p in model.model.parameters())

        # Training config
        args_yaml = best_pt.parent.parent / "args.yaml"
        if args_yaml.exists():
            with open(args_yaml) as f:
                args = yaml.safe_load(f)
            self.dataset = args.get("data", "")
            self.epochs = args.get("epochs", 0)
            self.imgsz = args.get("imgsz", 0)
            self.batch = args.get("batch", 0)
            self.patience = args.get("patience", 0)

        # Last training metrics
        results_csv = best_pt.parent.parent / "results.csv"
        if results_csv.exists():
            with open(results_csv) as f:
                lines = f.readlines()
            if len(lines) >= 2:
                header = lines[0].strip().split(",")
                last = lines[-1].strip().split(",")
                vals = dict(zip(header, last))
                self.actual_epochs = len(lines) - 1
                self.precision = float(vals.get("metrics/precision(B)", 0))
                self.recall = float(vals.get("metrics/recall(B)", 0))
                self.map50 = float(vals.get("metrics/mAP50(B)", 0))
                self.map50_95 = float(vals.get("metrics/mAP50-95(B)", 0))

        # Check if training continued past best
        last_pt = best_pt.parent / "last.pt"
        if last_pt.exists() and last_pt.stat().st_mtime > best_pt.stat().st_mtime:
            self.has_stale_best = True

    @property
    def exists(self):
        return self.weights_path is not None and Path(self.weights_path).exists()


class DatasetInfo:
    """Parsed info about a training dataset."""

    def __init__(self, dataset_yaml=None):
        self.path = None
        self.yaml_path = dataset_yaml
        self.n_images = 0
        self.n_labels = 0
        self.n_annotations = 0
        self.classes_covered = 0
        self.class_counts = Counter()  # class_id -> count
        self.snapshots = []  # list of dicts with name, timestamp, metrics

        if dataset_yaml and Path(dataset_yaml).exists():
            self._load(Path(dataset_yaml))

    def _load(self, yaml_path):
        self.yaml_path = yaml_path
        self.path = yaml_path.parent

        img_dir = self.path / "images"
        label_dir = self.path / "labels"
        self.n_images = (len(list(img_dir.glob("*.png")))
                         + len(list(img_dir.glob("*.jpg"))))
        self.n_labels = len(list(label_dir.glob("*.txt")))

        self.class_counts = Counter()
        self.n_annotations = 0
        for lp in label_dir.glob("*.txt"):
            with open(lp) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        self.class_counts[int(parts[0])] += 1
                        self.n_annotations += 1

        self.classes_covered = sum(
            1 for i in range(NUM_CLASSES) if self.class_counts.get(i, 0) > 0
        )

        # Load snapshots
        self.snapshots = []
        if SNAPSHOTS_DIR.exists():
            for s in sorted(SNAPSHOTS_DIR.iterdir()):
                meta_path = s / "meta.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        self.snapshots.append(json.load(f))

    @property
    def avg_per_class(self):
        return self.n_annotations / NUM_CLASSES if NUM_CLASSES > 0 else 0

    @property
    def min_per_class(self):
        return min((self.class_counts.get(i, 0) for i in range(NUM_CLASSES)),
                    default=0)

    @property
    def max_per_class(self):
        return max((self.class_counts.get(i, 0) for i in range(NUM_CLASSES)),
                    default=0)

    def weakest(self, n=5):
        """Return the n classes with fewest samples as (count, name) pairs."""
        all_counts = [(self.class_counts.get(i, 0), CLASS_NAMES[i])
                      for i in range(NUM_CLASSES)]
        all_counts.sort()
        return all_counts[:n]

    def strongest(self, n=5):
        """Return the n classes with most samples as (count, name) pairs."""
        all_counts = [(self.class_counts.get(i, 0), CLASS_NAMES[i])
                      for i in range(NUM_CLASSES)]
        all_counts.sort(reverse=True)
        return all_counts[:n]


def get_model_info():
    """Load and return ModelInfo for the current best model."""
    best_pt = RUNS_DIR / "detect" / "dartscorer" / "weights" / "best.pt"
    return ModelInfo(best_pt)


def get_dataset_info():
    """Load and return DatasetInfo for the current dataset."""
    return DatasetInfo(DATASET_YAML)


def print_info():
    """Print detailed info about the current model and dataset."""
    import time as _time

    mi = get_model_info()
    di = get_dataset_info()

    print("=" * 60)
    print("MODEL INFO")
    print("=" * 60)

    if not mi.exists:
        print("\n  No trained model found.")
        print(f"  Expected at: {RUNS_DIR / 'detect' / 'dartscorer' / 'weights' / 'best.pt'}")
        print("  Run 'python train.py' to train.")
    else:
        print(f"\n  Weights:      {mi.weights_path}")
        print(f"  Size:         {mi.size_mb:.1f} MB")
        print(f"  Last updated: {_time.ctime(mi.last_modified)}")
        print(f"  Classes:      {mi.num_classes}")
        print(f"  Parameters:   {mi.num_parameters:,}")

        if mi.dataset:
            print(f"\n  Training config:")
            print(f"    Dataset:    {mi.dataset}")
            print(f"    Epochs:     {mi.epochs}")
            print(f"    Image size: {mi.imgsz}")
            print(f"    Batch:      {mi.batch}")
            print(f"    Patience:   {mi.patience}")

        if mi.actual_epochs > 0:
            print(f"\n  Last training results (epoch {mi.actual_epochs}):")
            print(f"    Precision:  {mi.precision:.3f}")
            print(f"    Recall:     {mi.recall:.3f}")
            print(f"    mAP50:      {mi.map50:.3f}")
            print(f"    mAP50-95:   {mi.map50_95:.3f}")

        if mi.has_stale_best:
            print(f"\n  NOTE: last.pt is newer than best.pt — training may have")
            print(f"        continued past the best checkpoint.")

    print(f"\n{'=' * 60}")
    print("DATASET INFO")
    print("=" * 60)
    print(f"\n  Path:     {di.path}")
    print(f"  YAML:     {di.yaml_path}")
    print(f"  Images:   {di.n_images}")
    print(f"  Labels:   {di.n_labels}")
    print(f"  Annotations: {di.n_annotations}")
    print(f"  Classes covered: {di.classes_covered}/{NUM_CLASSES}")

    if di.n_annotations > 0:
        print(f"  Per class: min={di.min_per_class}  avg={di.avg_per_class:.0f}  max={di.max_per_class}")

        print(f"\n  5 weakest classes:")
        for count, name in di.weakest(5):
            print(f"    {name:>8s}: {count}")

        print(f"\n  5 strongest classes:")
        for count, name in di.strongest(5):
            print(f"    {name:>8s}: {count}")

    if di.snapshots:
        print(f"\n  Snapshots: {len(di.snapshots)}")
        for snap in di.snapshots:
            m = snap.get("metrics", {})
            print(f"    {snap['name']:<25s} {snap['timestamp']:<20s} mAP50={m.get('mAP50', '?')}")


def auto_focus(n_weakest=5, **kwargs):
    """Automatically focus on the N classes with fewest samples."""
    label_dir = DATASET_YAML.parent / "labels"
    counts = Counter()
    for lp in label_dir.glob("*.txt"):
        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    counts[int(parts[0])] += 1

    # Find weakest classes
    all_counts = [(counts.get(i, 0), CLASS_NAMES[i]) for i in range(NUM_CLASSES)]
    all_counts.sort()
    weakest = [name for _, name in all_counts[:n_weakest]]

    print(f"Auto-focus: targeting {n_weakest} weakest classes:")
    for count, name in all_counts[:n_weakest]:
        print(f"  {name}: {count} samples")

    focus_train(weakest, **kwargs)


# ---------------------------------------------------------------------------
# Standard training
# ---------------------------------------------------------------------------

def train(epochs=100, batch=16, imgsz=640, resume=False, weights=None, device=None,
          patience=20, dataset_yaml=None):
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
        data=str(dataset_yaml or DATASET_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(RUNS_DIR / "detect"),
        name="dartscorer",
        exist_ok=True,
        patience=patience,
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
    parser.add_argument("--info", action="store_true",
                        help="Show model and dataset info")
    parser.add_argument("--balanced", action="store_true",
                        help="Two-phase: balanced pre-train then fine-tune on all")
    parser.add_argument("--snapshot", type=str, default=None,
                        help="Save a named snapshot of current best model")
    parser.add_argument("--list-snapshots", action="store_true",
                        help="List saved model snapshots")
    parser.add_argument("--restore", type=str, default=None,
                        help="Restore a snapshot as current best model")
    parser.add_argument("--focus", type=str, default=None,
                        help="Focus training on specific classes (comma-separated, e.g. D3,D11,T10)")
    parser.add_argument("--auto-focus", type=int, default=None, metavar="N",
                        help="Auto-focus on the N weakest classes")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20, use higher for focus training)")
    args = parser.parse_args()

    if args.list_snapshots:
        list_snapshots()
    elif args.snapshot:
        snapshot(args.snapshot)
    elif args.restore:
        restore_snapshot(args.restore)
    elif args.info:
        print_info()
    elif args.eval:
        evaluate()
    elif args.focus:
        classes = [c.strip() for c in args.focus.split(",")]
        focus_train(
            classes,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            patience=args.patience,
        )
    elif args.auto_focus is not None:
        auto_focus(
            n_weakest=args.auto_focus,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            patience=args.patience,
        )
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
            patience=args.patience,
        )


if __name__ == "__main__":
    main()
