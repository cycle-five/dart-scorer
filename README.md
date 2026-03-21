# Dartscorer

Personal dart scoring system using a webcam and YOLO object detection. A camera watches a dartboard, detects where darts land, classifies the board segment, and scores the game automatically.

## Architecture

**186-class YOLO model** — each dart is classified by ordinal (1st/2nd/3rd in the round) and board segment (e.g. `d1_T20` = first dart, triple 20). The model learns the board layout implicitly from labeled training data — no board calibration or homography needed for scoring.

### Classes
- 20 sectors × 3 rings (single/double/triple) = 60 segments
- 2 bulls (single bull, double bull) = 2 segments
- 3 dart ordinals × 62 segments = **186 classes**

### Why 186 classes instead of 62?
The model needs to distinguish which dart is new. With 3 darts on the board, it predicts all three with their ordinals, and we know which one just landed. Previous approaches (frame diffing, tracking between frames) were tried and found insufficient.

## Hardware

- **Camera**: eMeet C950 webcam at 1920×1080 (MJPG via V4L2)
- **Mic**: eMeet C950 built-in mic (for audio dart detection trigger)
- **Board**: Standard WDF dartboard (Viper Razorback)
- **Mount**: Camera above/angled at the board

## Setup

```bash
# Dependencies (uses uv, not pip)
uv sync

# 1. (Optional) Lens calibration — checkerboard
uv run python calibrate.py --lens

# 2. Crop to dartboard region
uv run python calibrate.py --crop

# 3. (Optional) Board homography — helps auto-guess segments during annotation
uv run python calibrate.py --board

# 4. (Optional) Train audio trigger — ML classifier for dart impact sounds
uv run python audio_trigger.py --record     # collect dart/noise samples
uv run python audio_trigger.py --train      # train classifier
uv run python audio_trigger.py --test       # verify live

# Note: eMeet mic may be muted by default in PipeWire:
wpctl set-mute 61 0 && wpctl set-volume 61 1.0
```

## Data Collection

```bash
# Video trigger (frame differencing — recommended)
uv run python collect.py --trigger video --no-undistort

# Audio trigger (ML dart sound classifier)
uv run python collect.py --trigger audio --no-undistort

# Manual only
uv run python collect.py --trigger manual --no-undistort
```

### Collection workflow
1. Throw dart → auto-capture (video diff or audio thud) → "DART! Settling..." countdown
2. Frame freezes → click dart tip → segment guess appears (if homography calibrated)
3. ENTER to accept guess, or type correction (e.g. `t20`, `s5`, `dbull`) + ENTER
4. Dart ordinal auto-advances (1→2→3)
5. ENTER to save frame with YOLO labels
6. After 3rd dart: trigger suppressed, "PULL DARTS" shown → pull darts → auto-resumes

### Controls
| Key | Action |
|-----|--------|
| SPACE | Manual capture |
| Click | Mark dart tip |
| ENTER | Accept guess / save frame |
| ESC | Discard frame / cancel click |
| Z | Undo last annotation |
| R | Reset round to dart 1 |
| +/- | Adjust trigger threshold |
| Q | Quit |

### Output
- `data/training/images/` — PNG frames
- `data/training/labels/` — YOLO format label files
- `data/training/annotations.jsonl` — bookkeeping

## Training

```bash
# Train YOLOv8 on collected data
uv run python train.py --epochs 100

# Resume from checkpoint
uv run python train.py --resume

# Evaluate current model
uv run python train.py --eval
```

Recommended: collect ~100 frames → train → evaluate → collect more → retrain.

## Live Scoring

```bash
uv run python scorer.py --debug
uv run python scorer.py --game 501
uv run python scorer.py --game 301 --conf 0.3
```

## File Map

| File | Purpose |
|------|---------|
| `classes.py` | 186 class definitions, shorthand parsing |
| `audio_trigger.py` | ML dart sound classifier (record/train/test, online learning) |
| `collect.py` | Data collection UI with audio/video/manual triggers |
| `train.py` | YOLOv8 training wrapper |
| `yolo_detector.py` | YOLO inference for live scoring |
| `scorer.py` | Live scoring loop with game tracking |
| `calibrate.py` | Lens calibration, crop ROI, board homography |
| `config.py` | All paths, camera settings, board geometry constants |
| `window_manager.py` | Sticky OpenCV window sizes |
| `board.py` | Board geometry + score mapping (used for annotation guessing) |
| `detector.py` | Legacy classical CV detector (reference) |
| `optimize.py` | Legacy Bayesian parameter optimizer (reference) |

## Data Directories

```
data/
├── training/
│   ├── images/          # Training frames (PNG)
│   ├── labels/          # YOLO label files
│   ├── dataset.yaml     # YOLO dataset config
│   └── annotations.jsonl
├── audio/
│   ├── samples/         # Audio clips (.npy)
│   ├── samples.jsonl    # Audio sample metadata
│   └── dart_classifier.pkl  # Trained audio model
├── lens_params.npz      # Camera intrinsics
├── board_homography.npz # Perspective transform
├── crop_roi.npz         # Crop region
└── .window_cache.json   # Window size cache
```
