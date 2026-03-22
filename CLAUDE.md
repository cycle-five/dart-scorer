# Dartscorer — AI Context

## What this is
A personal dart scoring system. A webcam watches a dartboard, a YOLO model detects where darts land (189 classes = 3 ordinals × 63 board segments including MISS), and scores the game.

## Current state (March 2026)
- **All tooling is built** — calibration, data collection, training, inference, scoring
- **Data collection is in progress** — the user is actively collecting YOLO training data
- **No trained YOLO model yet** — need ~100+ labeled frames before first training run
- **Audio dart classifier is trained** (100% F1 on 145 samples) but video trigger is preferred
- **Board homography works** — rotation bug fixed, calibrated in cropped space, good segment guesses
- **Collection UI is streamlined** — throw 3, click 3 tips, pull darts, repeat

## Key commands
```bash
uv run python calibrate.py --crop                          # Set crop region
uv run python calibrate.py --board --no-undistort           # Board homography (match collection space)
uv run python collect.py collect --trigger video --no-undistort  # Full collection (batch + annotate)
uv run python collect.py capture --trigger video --no-undistort  # Capture-only (no labeling)
uv run python collect.py label                              # Label previously captured frames
uv run python generate_dataset_yaml.py                      # Regenerate dataset.yaml after class changes
uv run python train.py --epochs 100                         # Train YOLO
uv run python train.py --eval                               # Evaluate
uv run python scorer.py --debug                             # Live scoring
uv run python compare_homography.py --browse                # Compare homography vs labels
uv run python -m pytest test_board.py -v                    # Run unit tests
```

## Important conventions
- Uses `uv` exclusively (no pip, no conda)
- `QT_QPA_PLATFORM=xcb` must be set before importing cv2
- eMeet C950 mic must be unmuted in PipeWire before audio features work
- `sounddevice` playback is broken (PipeWire routing) — use `paplay` via subprocess
- Camera device changes — check `config.py` for current setting
- All data lives under `data/` which is gitignored
- Board homography MUST be calibrated with same undistort setting as collection (both --no-undistort or both without)
- When classes change, run `generate_dataset_yaml.py` to update dataset.yaml

## Architecture decisions to respect
- 189 classes (3 ordinals × 63 segments) — ordinal matters, MISS is a first-class label
- Board homography is ONLY for annotation guessing, NOT for scoring
- Homography rotation fix: canonical destinations use sector MIDPOINTS (i*18°), not boundary wires (i*18°-9°), because users click sector centers even when told to click wires
- Classical CV files (detector.py, optimize.py) are legacy reference — don't modify or build on them
- collect.py has an explicit UIState enum: WARMUP → LISTENING → SETTLING → COLLECTING → ANNOTATING → REVIEW → PULL_DARTS
- Only LISTENING state checks triggers — all other states ignore them
