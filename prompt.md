**Project: Webcam Dart Scoring System**

Build a dart scoring system in Python using OpenCV that uses a ceiling-mounted webcam (eMeet C950) pointed down at a dartboard.

**Hardware context:**
- Camera: eMeet C950 HD webcam at `/dev/video2` (verify with `v4l2-ctl --list-devices`)
- Mounting: ceiling overhead, significant perspective distortion expected
- Board: Viper Razorback standard bristle dartboard

**Project structure to create:**
```
dartscorer/
├── calibrate.py       # One-time camera + board calibration
├── scorer.py          # Main scoring loop
├── board.py           # Dartboard geometry and score mapping
├── detector.py        # Dart detection via frame differencing
├── config.py          # Paths, constants, tunable parameters
└── data/
    ├── lens_params.npz    # Saved camera intrinsics
    └── board_homography.npz  # Saved perspective transform
```

**Phase 1 — Lens calibration (`calibrate.py`):**
Implement OpenCV checkerboard lens calibration using `cv2.calibrateCamera`. Print a standard checkerboard, hold it at various angles in front of the camera, collect ~20 frames, compute intrinsics and distortion coefficients, save to `data/lens_params.npz`. Add a `--lens` flag to trigger this mode.

**Phase 2 — Board calibration (`calibrate.py` continued):**
After lens calibration, detect the dartboard ring structure. Undistort the frame first. Then detect the outer wire ellipse using Canny edge detection + `cv2.fitEllipse` on contours filtered by area and aspect ratio. Use the known dartboard geometry (bullseye at center, double ring at radius=170mm, triple at 107mm, outer bull at 16mm, inner bull at 6.35mm — standard WDF dimensions) to build a homography from camera space to a canonical 340px-diameter top-down circle. Save homography to `data/board_homography.npz`. Include a visual debug mode (`--debug`) that draws the detected ellipse and overlays the canonical grid so the user can verify accuracy.

**Phase 3 — Dart detection (`detector.py`):**
Implement background subtraction via frame differencing:
- Maintain a rolling median background model (use the last N undistorted frames with no detected motion)
- On each new frame, compute `cv2.absdiff(background, frame)`, convert to grayscale, threshold, morphological close to fill gaps
- If a significant new blob appears and persists for >3 frames (to reject false positives), declare a dart throw detected
- Localize the dart tip: within the diff blob region, find the point closest to the board center — this is the tip coordinate
- Return the tip in camera pixel space

**Phase 4 — Score mapping (`board.py`):**
Given a tip coordinate in camera pixels:
1. Apply the homography to get canonical circle coordinates
2. Compute polar coords: `r = distance from center`, `theta = atan2(y, x)` adjusted for board rotation (20 at top)
3. Map `r` to ring: bullseye (<6.35mm equivalent), bull25 (<16mm), miss (>170mm), triple band (107-115mm), double band (162-170mm), otherwise single
4. Map `theta` to sector using the standard clockwise order: 20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5
5. Return `(sector, multiplier, score)` and a human-readable string like "Triple 20 → 60"

**Phase 5 — Main loop (`scorer.py`):**
- Load lens params and homography on startup, abort with clear error if not found (tell user to run calibrate first)
- Open the camera with `cv2.VideoCapture`, set resolution to max supported
- Run detector in a loop; on dart detection, compute and print the score, display the frame with the tip marked and score overlaid
- Implement a simple state machine: `WAITING` (no darts) → `DART_DETECTED` → `WAITING_FOR_REMOVAL` (wait until dart is gone before next detection)
- Support a `--game 501` or `--game 301` flag that tracks running totals and prints remaining score
- Press `r` to reset game, `q` to quit, `c` to recalibrate board homography without redoing lens cal

**Implementation requirements:**
- Use Python 3.11+, OpenCV (`opencv-python-headless` + `opencv-contrib-python`), numpy
- All tunable thresholds (diff threshold, min blob area, persistence frames, etc.) in `config.py` as named constants with comments
- `--debug` flag on all scripts shows intermediate CV steps in windows (edge detection, diff mask, detected tip, etc.)
- Handle camera not found, calibration files missing, and low-light conditions gracefully with clear error messages
- Log detections with timestamp to a CSV for later analysis

**Known challenges to handle:**
- Perspective distortion is significant (ceiling mount) — homography must be applied before any geometry math
- Barrel distortion from the C950 — always undistort before processing
- Dart shaft and flight occlude the tip — tip-finding should search along the dart axis direction toward center, not just take the blob centroid
- Multiple darts in the board simultaneously — track blobs individually, only score newly appeared blobs
- Lighting changes (ceiling light turning on/off) — background model should adapt slowly; detect sudden global illumination changes and pause detection

Start by implementing Phase 1 and Phase 2 with the debug visualization, then confirm with the user before proceeding to Phase 3.
