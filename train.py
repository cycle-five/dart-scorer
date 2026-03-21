#!/usr/bin/env python3
"""
train.py — YOLO training for dart tip detection + scoring.

Wraps ultralytics YOLOv8 training with the dartscorer dataset.

Usage:
    python train.py                         # Train from scratch
    python train.py --resume                # Resume from last checkpoint
    python train.py --weights best          # Fine-tune from best.pt
    python train.py --epochs 200            # Custom epoch count
    python train.py --eval                  # Evaluate current model on training set
"""

import argparse
import sys
from pathlib import Path

import config
from classes import NUM_CLASSES, ID_TO_CLASS, parse_class_name


DATASET_YAML = config.PROJECT_ROOT / "data" / "training" / "dataset.yaml"
RUNS_DIR = config.PROJECT_ROOT / "runs"


def train(epochs=100, batch=16, imgsz=640, resume=False, weights=None, device=None):
    """Train YOLOv8 on the dartscorer dataset."""
    from ultralytics import YOLO

    if not DATASET_YAML.exists():
        print(f"ERROR: Dataset config not found at {DATASET_YAML}")
        sys.exit(1)

    # Check we have training data
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
        model = YOLO("yolov8n.pt")  # start from pretrained nano
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

    # Print per-class summary for classes that have data
    print(f"\n{'Class':<20} {'Images':>7} {'Instances':>10} {'P':>8} {'R':>8} {'mAP50':>8}")
    print("-" * 65)

    if hasattr(results, 'results_dict'):
        print(f"{'all':<20} {'':>7} {'':>10} "
              f"{results.results_dict.get('metrics/precision(B)', 0):.3f}   "
              f"{results.results_dict.get('metrics/recall(B)', 0):.3f}   "
              f"{results.results_dict.get('metrics/mAP50(B)', 0):.3f}")

    return results


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
    args = parser.parse_args()

    if args.eval:
        evaluate()
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
