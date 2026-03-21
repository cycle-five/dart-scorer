# Dartscorer — AI Context

## What this is
A personal dart scoring system. A webcam watches a dartboard, a YOLO model detects where darts land (186 classes = 3 ordinals × 62 board segments), and scores the game.

## Current state (March 2026)
- **All tooling is built** — calibration, data collection, training, inference, scoring
- **Data collection is in progress** — the user is actively collecting YOLO training data
- **No trained YOLO model yet** — need ~100+ labeled frames before first training run
- **Audio dart classifier is trained** (100% F1 on 145 samples) but video trigger is preferred for reliability
- **Collection UI has rough edges** being iterated on — see memory for known issues

## Key commands
```bash
uv run python calibrate.py --crop        # Set crop region
uv run python calibrate.py --board       # Board homography (for segment guessing)
uv run python collect.py --trigger video --no-undistort   # Collect training data
uv run python train.py --epochs 100      # Train YOLO
uv run python train.py --eval            # Evaluate
uv run python scorer.py --debug          # Live scoring
uv run python audio_trigger.py --record  # Collect audio samples
```

## Important conventions
- Uses `uv` exclusively (no pip, no conda)
- `QT_QPA_PLATFORM=xcb` must be set before importing cv2
- eMeet C950 mic must be unmuted in PipeWire before audio features work
- `sounddevice` playback is broken (PipeWire routing) — use `paplay` via subprocess
- Camera device is `/dev/video2` (may change, check `config.py`)
- All data lives under `data/` which is gitignored
- OpenCV has two packages installed (opencv-python + opencv-contrib-python) due to ultralytics — don't try to resolve, it works

## Architecture decisions to respect
- 186 classes, not 62 — the ordinal matters, this was a deliberate decision after trying alternatives
- Board homography is ONLY for annotation guessing, NOT for scoring — it was never accurate enough
- Classical CV files (detector.py, optimize.py) are legacy reference — don't modify or build on them
- The collect.py state machine (WARMUP → LISTENING → SETTLING → FROZEN → COOLDOWN → SUPPRESSED) is complex — test interactively, not just with imports
