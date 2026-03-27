# Dartscorer

Automatic dart scoring system using a single USB webcam (~$20) and YOLO object detection. A camera watches a dartboard, a 63-class YOLO model classifies where each dart lands, and the system scores the game in real time. Board geometry provides a confidence check via homography.

Built as a personal project with the potential to serve local dart leagues.

## How It Works

```
Camera Frame → Crop → YOLO (63-class) → Segment + Score
                                ↓
                        Geometry Check (homography → polar → wire distance)
                                ↓
                        Confidence: YOLO + geometry agree? → display
```

Each board segment is its own YOLO class — 20 sectors x 3 rings (single/double/triple) + bulls + miss = 63 classes. The model directly predicts the segment from the dart's visual appearance and board position. No ordinal prediction — dart order comes from frame sequence.

## Hardware

- **Camera**: Any USB webcam (~$20). Developed with eMeet C950 at 1024x576 MJPG
- **Board**: Standard WDF dartboard (e.g. Viper Razorback)
- **GPU**: NVIDIA GPU recommended for training (RTX 3080 trains in ~17 min)
- **Optional**: Side camera at oblique angle for improved detection (future)

## Quick Start

```bash
# Install dependencies (uses uv, not pip)
uv sync

# 1. Set crop region (dartboard area only)
uv run python calibrate.py --crop

# 2. Calibrate board homography (for annotation guessing + confidence checks)
uv run python calibrate.py --board --no-undistort

# 3. Collect training data (throw darts, click tips, labels auto-guessed)
uv run python collect.py collect --trigger video --no-undistort

# 4. Train the model
uv run python train.py --epochs 100

# 5. Score!
uv run python scorer.py --debug
```

## Calibration

```bash
# Lens distortion (optional — skip if using --no-undistort everywhere)
uv run python calibrate.py --lens

# Crop to dartboard region (required)
uv run python calibrate.py --crop

# Board homography (21-point click calibration)
# MUST match undistort setting used in collection
uv run python calibrate.py --board --no-undistort

# Audio trigger training (optional — alternative to video trigger)
uv run python audio_trigger.py --record     # collect dart/noise samples
uv run python audio_trigger.py --train      # train classifier
uv run python audio_trigger.py --test       # verify live
```

## Data Collection

```bash
# Full collection: capture + annotate (recommended)
uv run python collect.py collect --trigger video --no-undistort

# Capture only (label later)
uv run python collect.py capture --trigger video --no-undistort

# Label previously captured frames
uv run python collect.py label
```

### Trigger modes
- **video** (recommended): Frame differencing detects dart arrival
- **audio**: ML classifier on microphone detects impact sound
- **manual**: SPACE key only

### Collection workflow
1. Throw dart → auto-capture → settle delay
2. Frame freezes → click dart tip → segment auto-guessed from homography
3. Accept guess or type correction (e.g. `t20`, `s5`, `dbull`) + ENTER
4. Auto-advances to next frame; after 3 darts → review screen
5. Pull darts → auto-detects board is clear → next round

### Controls (annotation)
| Key | Action |
|-----|--------|
| Click | Mark dart tip |
| ENTER | Accept guess / save |
| ESC | Discard entire batch (deletes saved files) |
| B | Mark dart as bounced out (skip frame) |
| Z | Undo last annotation |
| X | Edit mode (in review): select dart 1/2/3, then type new label or click to reposition tip |
| +/- | Adjust trigger sensitivity |
| Q | Quit |

### Control panel
The control panel displays:
- Trigger waveform and thresholds
- Current state and dart count
- Files saved this batch (with full paths)
- **3 classes most in need of training data** (updates live)

### Filenames
Frames use timestamp-based naming to prevent overwrites:
```
20260325_143052_123_raw_674x569.png
YYYYMMDD_HHMMSS_fff_{undistort|raw}_{W}x{H}.ext
```

## Training

```bash
# Train from scratch (pretrained YOLOv8n backbone)
uv run python train.py --epochs 100

# Fine-tune from current best model
uv run python train.py --weights best --epochs 100

# Focus on specific underperforming classes
uv run python train.py --focus D3,D11,T10 --weights best --patience 50

# Auto-focus on the 10 weakest classes
uv run python train.py --auto-focus 10 --weights best --patience 50

# Balanced pre-training (equal representation per class)
uv run python train.py --balanced

# Evaluate current model
uv run python train.py --eval

# Resume interrupted training
uv run python train.py --resume
```

### Snapshots
```bash
uv run python train.py --snapshot "before-doubles-push"
uv run python train.py --list-snapshots
uv run python train.py --restore "before-doubles-push"
```

### Training targets
- **Minimum**: ~20 examples per class (model learns the class exists)
- **Functional**: ~50 per class (reasonable detection)
- **Solid**: ~100+ per class (good generalization)

## Live Scoring

```bash
# Raw detection (no game tracking)
uv run python scorer.py --debug

# 501 game
uv run python scorer.py --game 501

# 301 game with custom confidence
uv run python scorer.py --game 301 --conf 0.3
```

### Scoring display
- Green: YOLO and geometry agree (high confidence)
- Yellow: Geometry uncertain (near wire)
- Red: YOLO and geometry disagree
- `[!geo]` / `[?]` / `[~]` markers indicate confidence level

### Controls (scoring)
| Key | Action |
|-----|--------|
| Q | Quit |
| R | Reset game |
| D | Toggle debug overlay |

## Analysis Tools

```bash
# Class distribution heatmap
uv run python analyze_data.py
uv run python analyze_data.py --save

# Resolution quality, bbox calibration, confidence model analysis
uv run python analyze_resolution.py
uv run python analyze_resolution.py --sharpness
uv run python analyze_resolution.py --confidence

# Compare homography predictions vs labels
uv run python compare_homography.py --browse
uv run python compare_homography.py --report

# Label browser (web UI)
uv run python label_analyzer.py
# Opens http://localhost:8765
```

## Data Migration

```bash
# Convert old v1 data (189-class) to v3 (63-class)
uv run python convert_v1_to_v3.py                   # dry run
uv run python convert_v1_to_v3.py --convert --skip-bad-frames

# Migrate all data to new directory structure
uv run python migrate_data.py              # dry run
uv run python migrate_data.py --migrate

# Regenerate dataset.yaml after changes
uv run python generate_dataset_yaml.py
```

## Project Structure

```
dartscorer/
├── config.py                 # Central configuration (paths, constants, geometry)
├── classes_v2.py             # 63-class definitions (segments only, no ordinals)
├── board.py                  # Board geometry, scoring, confidence model (arc-length)
│
├── collect.py                # Data collection UI (video/audio/manual triggers)
├── train.py                  # YOLO training (standard, balanced, focused)
├── scorer.py                 # Live scoring with game tracking (301/501)
├── yolo_detector.py          # YOLO inference + geometry confidence check
├── calibrate.py              # Lens, crop, and board homography calibration
│
├── analyze_data.py           # Class distribution analysis + heatmap
├── analyze_resolution.py     # Resolution quality and confidence model validation
├── compare_homography.py     # Homography accuracy comparison
├── label_analyzer.py         # Web-based label browser (localhost:8765)
│
├── audio_trigger.py          # ML audio dart classifier (MFCC + RandomForest)
├── window_manager.py         # Persistent OpenCV window sizes
│
├── convert_v1_to_v3.py       # V1 (189-class) → V3 (63-class) label conversion
├── convert_v1_data.py        # V1 → V2 conversion (historical)
├── migrate_data.py           # Migrate data to timestamp-based directory structure
├── generate_dataset_yaml.py  # Regenerate YOLO dataset config
│
├── detector.py               # Legacy classical CV detector (reference only)
├── optimize.py               # Legacy parameter optimizer (reference only)
├── classes.py                # Legacy 189-class definitions (reference only)
│
└── data/
    ├── v3/                   # Active training dataset
    │   ├── images/           # YYYYMMDD_HHMMSS_fff_raw_WxH.png
    │   ├── labels/           # Matching YOLO label files
    │   ├── dataset.yaml      # YOLO dataset config
    │   └── annotations.jsonl # Collection metadata
    ├── audio/
    │   ├── samples/          # Audio clips (.npy)
    │   └── dart_classifier.pkl → .joblib
    ├── lens_params.npz       # Camera intrinsics
    ├── board_homography.npz  # Perspective transform
    └── crop_roi.npz          # Crop region
```

## Architecture History

| Version | Classes | Approach | Status |
|---------|---------|----------|--------|
| V1 | 189 (3 ordinals x 63 segments) | Direct YOLO classification | Ordinal confusion, stuck classes |
| V2 | 1 ("dart") | YOLO detection + geometry classification | Overcomplicated, bbox issues |
| **V3** | **63 (segments only)** | **Direct YOLO classification + geometry confidence** | **Active** |

The key insight: V1's model could classify segments well — the only problem was fragmenting data across 3 ordinals. V3 collapses ordinals (determined by frame order), giving 3x more data per class.

## Configuration

All tunable constants live in `config.py`:
- Camera settings (device, resolution)
- Board geometry (WDF standard dimensions in mm)
- Classification confidence (arc-length wire proximity model)
- YOLO defaults (confidence, IoU thresholds)
- Video trigger parameters
- Bounding box expansion fractions (resolution-relative)
- File paths (dataset, calibration, logs)
