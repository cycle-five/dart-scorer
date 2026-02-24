"""
config.py — Tunable constants for the webcam dart scoring system.

All file paths are resolved relative to the project root (the directory
containing this file) so the project can be run from any working directory.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — every path below is anchored here
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

# V4L2 device node for the eMeet C950 HD webcam (ceiling-mounted)
CAMERA_DEVICE = "/dev/video0"

# Native capture resolution of the eMeet C950
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# ---------------------------------------------------------------------------
# Data / calibration file paths
# ---------------------------------------------------------------------------

# OpenCV camera-intrinsics (lens distortion) produced by checkerboard calibration
LENS_PARAMS_PATH = PROJECT_ROOT / "data" / "lens_params.npz"

# Homography matrix that maps the distortion-corrected frame to the canonical
# top-down dartboard view
BOARD_HOMOGRAPHY_PATH = PROJECT_ROOT / "data" / "board_homography.npz"

# ---------------------------------------------------------------------------
# Checkerboard calibration
# ---------------------------------------------------------------------------

# Number of *inner* corner intersections (columns, rows) on the calibration
# checkerboard — not the number of squares
CHECKERBOARD_SIZE = (9, 6)

# Physical size of each checkerboard square in millimetres; used to recover
# real-world scale for the lens-distortion model
CHECKERBOARD_SQUARE_SIZE = 25.0  # mm

# Minimum number of accepted checkerboard frames required before the
# calibration is considered valid
MIN_CALIBRATION_FRAMES = 20

# Minimum time in seconds that must elapse between successive automatic
# frame captures during the calibration procedure
CALIBRATION_DELAY = 1.0  # seconds

# ---------------------------------------------------------------------------
# Dartboard geometry  (WDF standard dimensions, all in millimetres unless
# noted)
# ---------------------------------------------------------------------------

# Radius of the inner bull (double bull / bullseye) — scores 50 points
INNER_BULL_RADIUS = 6.35  # mm

# Radius of the outer bull (single bull) — scores 25 points
OUTER_BULL_RADIUS = 15.9  # mm

# Inside edge of the triple (treble) ring
TRIPLE_INNER_RADIUS = 99.0  # mm

# Outside edge of the triple ring
TRIPLE_OUTER_RADIUS = 107.0  # mm

# Inside edge of the double ring
DOUBLE_INNER_RADIUS = 162.0  # mm

# Outside edge of the double ring — also the playable outer boundary of the
# board (radius = 170 mm → diameter = 340 mm)
DOUBLE_OUTER_RADIUS = 170.0  # mm

# Side length of the square canonical top-down image in pixels.
# Chosen so that 1 pixel ≈ 1 mm at the board surface, giving
# CANONICAL_DIAMETER = 2 × DOUBLE_OUTER_RADIUS = 340 px.
CANONICAL_DIAMETER = 340  # pixels

# Pixel coordinates of the bullseye in the canonical top-down view
CANONICAL_CENTER = (170, 170)  # (x, y) pixels

# ---------------------------------------------------------------------------
# Sector order
# ---------------------------------------------------------------------------

# Standard dartboard sector sequence, listed clockwise starting from the top
# (12 o'clock position).  There are 20 sectors; index 0 is the topmost sector.
SECTOR_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# ---------------------------------------------------------------------------
# Detection constants
# ---------------------------------------------------------------------------

# Minimum per-pixel intensity difference (0–255) between the current frame
# and the background model for a pixel to be classified as "changed".
# Raised from 30 to reject wire reflections and camera sensor noise.
DIFF_THRESHOLD = 51

# Minimum contour area in pixels for a detected blob to be considered a
# potential dart tip.  At 1080p a dart shaft produces a blob of ~300–2000 px²;
# wire glints are typically < 200 px².
MIN_BLOB_AREA = 150  # pixels²

# Maximum contour area in pixels; blobs larger than this are rejected as
# environmental noise or large motion events (e.g. a hand passing through)
MAX_BLOB_AREA = 4500  # pixels²

# Number of consecutive frames a candidate blob must appear in before it is
# accepted as a real dart.  Wire noise can persist for a few frames due to
# camera auto-exposure adjustments, so require more frames.
PERSISTENCE_FRAMES = 4  # frames

# Maximum distance (pixels) between blob centroids to merge them into one
# dart.  Dart shafts and flights often appear as separate blobs; merging
# prevents double-counting and ensures the tip is found on the shaft, not
# the flight.
BLOB_MERGE_DISTANCE = 150  # pixels

# Gaussian blur kernel size (must be odd).  Applied to both background and
# current frame before differencing to suppress wire detail and sensor noise.
BLUR_KSIZE = 3

# Morphological opening kernel size (must be odd).  Removes small noise
# speckles (wire glints) from the diff mask.
OPEN_KSIZE = 3

# Morphological closing kernel size (must be odd).  Fills gaps within
# dart-shaped blobs in the diff mask.
CLOSE_KSIZE = 19

# Number of frames kept in the rolling buffer used to compute the median
# background image.  Larger values produce a more stable background at the
# cost of slower adaptation to slow illumination changes.
BACKGROUND_HISTORY = 30  # frames

# Mean frame-to-frame intensity change (across all pixels) that triggers a
# "lighting shift" event, causing the background model to be reset.
ILLUMINATION_CHANGE_THRESHOLD = 20.0  # mean pixel intensity units (0–255)

# Number of frames to ignore after a motion event before resuming background
# model updates, preventing motion blur from corrupting the background.
MOTION_COOLDOWN_FRAMES = 5  # frames

# ---------------------------------------------------------------------------
# ROI (Region of Interest) cropping
# ---------------------------------------------------------------------------

# Pixels of padding around the board bounding box when cropping the
# undistorted frame to just the dartboard region.  Keeps a margin so darts
# at the outer edge are not clipped.
ROI_PADDING = 60  # pixels

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------

# Default game mode at startup.  None means the system runs in raw-detection
# mode without enforcing any game rules.
DEFAULT_GAME = None

# Checkout / start-score variants that the game engine supports
SUPPORTED_GAMES = [301, 501]

# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

# Path to the CSV file where every detected dart throw is appended for
# offline review and model training
LOG_FILE = PROJECT_ROOT / "data" / "detections.csv"

# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

# Set to True to enable verbose console output, intermediate frame windows,
# and other developer aids.  Keep False in production.
DEBUG = True
