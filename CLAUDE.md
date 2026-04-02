# Dartscorer — AI Context

## What this is

A personal dart scoring system. A webcam watches a dartboard, a YOLO model detects where darts land (62-class direct classification), and scores the game.

## Current state (April 2026)

- **All tooling is built** — calibration, data collection, training, inference, scoring
- **Data collection is in progress** — the user is actively collecting YOLO training data
- **YOLO model training in progress** — need ~50+ labeled frames for *each class* (the rarer ones around still around 20.)
- **Audio dart classifier is trained** (100% F1 on 145 samples) but video trigger is preferred
- **Board homography works** — rotation bug fixed, calibrated in cropped space, good segment guesses, parallax(?) homography(??) used for perspective distorted metal separator pieces jutting out of the board.
- **Collection UI is streamlined** — throw 3, click 3 tips, pull darts, repeat

## Project layout

```
src/dartscorer/     Core package (installed via uv)
  config.py         All paths, constants, thresholds
  board.py          Board geometry, scoring math
  classes_v2.py     62-class definitions (segment-only, no ordinals)
  calibrate.py      Camera lens + board homography calibration
  collect.py        Training data collection workflow
  train.py          YOLO training wrapper
  scorer.py         Live scoring (YOLO + geometry)
  yolo_detector.py  YOLO inference + dart tracking
  audio_trigger.py  ML audio dart impact detection
  window_manager.py Sticky OpenCV window sizes

scripts/            Ad-hoc utilities (run with uv run python scripts/...)
  analyze_data.py        Training data distribution heatmap
  analyze_resolution.py  Image quality analysis
  compare_homography.py  Homography vs label comparison
  find_mislabels.py      Model-vs-label disagreement finder
  generate_dataset_yaml.py  Regenerate YOLO dataset.yaml
  label_analyzer.py      Web-based label debugger (port 8765)

docs/               Documentation (ARCHITECTURE.md, V2_PLAN.md, prompt.md)
old/                Legacy/migration code (v1 classes, converters, classical CV)
test_board.py       Unit tests (root level)
```

## Key commands

```bash
uv run calibrate --crop                                     # Set crop region
uv run calibrate --board --no-undistort                     # Board homography
uv run collect collect --trigger video --no-undistort        # Full collection
uv run collect capture --trigger video --no-undistort        # Capture-only
uv run collect label                                        # Label previously captured frames
uv run train --epochs 100                                   # Train YOLO
uv run train --eval                                         # Evaluate
uv run scorer --debug                                       # Live scoring
uv run python scripts/compare_homography.py --browse        # Compare homography vs labels
uv run python scripts/generate_dataset_yaml.py              # Regenerate dataset.yaml
uv run python -m pytest test_board.py -v                    # Run unit tests
```

## Important conventions

- Uses `uv` exclusively (no pip, no conda)
- Package installed from `src/dartscorer/` via pyproject.toml entry points
- `QT_QPA_PLATFORM=xcb` must be set before importing cv2
- eMeet C950 mic must be unmuted in PipeWire before audio features work
- `sounddevice` playback is broken (PipeWire routing) — use `paplay` via subprocess
- Camera device changes — check `src/dartscorer/config.py` for current setting
- All data lives under `data/` which is gitignored
- Board homography MUST be calibrated with same undistort setting as collection (both --no-undistort or both without)
- When classes change, run `scripts/generate_dataset_yaml.py` to update dataset.yaml

## Architecture decisions to respect

- 62 classes (segment-only, no ordinal encoding) — MISS handled by geometry, not YOLO
- Board homography used for annotation guessing + geometry confidence check
- Homography rotation fix: canonical destinations use sector MIDPOINTS (i*18°), not boundary wires (i*18°-9°), because users click sector centers even when told to click wires
- Legacy code in `old/` — don't modify or build on it
- collect.py has an explicit UIState enum: WARMUP → LISTENING → SETTLING → COLLECTING → ANNOTATING → REVIEW → PULL_DARTS
- Only LISTENING state checks triggers — all other states ignore them
