"""
classes_v2.py — YOLO class definitions for 63-class dart detection (v2).

V2 drops the ordinal dimension. Each class represents a board segment only.
Format: {segment}  e.g. "S20", "D5", "T1", "S_BULL", "D_BULL", "MISS"

Segments (63 total):
  S1..S20   — single 1-20
  D1..D20   — double 1-20
  T1..T20   — triple 1-20
  S_BULL    — single bull (25 points)
  D_BULL    — double bull / bullseye (50 points)
  MISS      — outside scoring area (0 points)

Total: 63 classes (no ordinal dimension).
"""

SECTORS = list(range(1, 21))
RINGS = ["S", "D", "T"]  # single, double, triple
BULLS = ["S_BULL", "D_BULL"]

# Clockwise sector order on a standard dartboard, starting from the top.
SECTOR_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# Maps each sector number to its two clockwise neighbors [prev, next].
# Example: SECTOR_ADJACENCY[20] = [5, 1]
SECTOR_ADJACENCY = {
    SECTOR_ORDER[i]: [
        SECTOR_ORDER[(i - 1) % 20],
        SECTOR_ORDER[(i + 1) % 20],
    ]
    for i in range(20)
}

# Build segment list (62 segments — MISS is handled by geometry, not YOLO).
SEGMENTS = []
for s in SECTORS:
    for r in RINGS:
        SEGMENTS.append(f"{r}{s}")
SEGMENTS.extend(BULLS)
# MISS is NOT a YOLO class — it's determined by geometry (r > DOUBLE_OUTER_RADIUS).
# Keeping it out of the class list prevents training data imbalance.

# Class list — each segment is its own class (index == class ID).
CLASS_NAMES = list(SEGMENTS)

# Lookup dicts
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

NUM_CLASSES = len(CLASS_NAMES)  # 62


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def ring_from_segment(segment: str) -> str:
    """Return the ring code for a segment.

    Returns one of: "S", "D", "T", "S_BULL", "D_BULL", "MISS".
    """
    if segment == "MISS":
        return "MISS"
    if segment in ("S_BULL", "D_BULL"):
        return segment
    return segment[0]  # "S", "D", or "T"


def sector_from_segment(segment: str) -> int:
    """Return the sector number for a segment.

    Returns:
        1-20  for normal segments
        25    for bulls (S_BULL / D_BULL)
        0     for MISS
    """
    if segment == "MISS":
        return 0
    if segment in ("S_BULL", "D_BULL"):
        return 25
    return int(segment[1:])


def parse_class_name(name: str) -> dict:
    """Parse a segment class name into scoring components.

    Returns a dict with keys:
        segment, sector, ring, multiplier, score, label
    """
    segment = name  # v2: name IS the segment

    if segment == "MISS":
        return {
            "segment": segment,
            "sector": 0,
            "ring": "miss",
            "multiplier": 0,
            "score": 0,
            "label": "Miss → 0",
        }
    if segment == "S_BULL":
        return {
            "segment": segment,
            "sector": 25,
            "ring": "S-BULL",
            "multiplier": 1,
            "score": 25,
            "label": "Single Bull → 25",
        }
    if segment == "D_BULL":
        return {
            "segment": segment,
            "sector": 25,
            "ring": "D-BULL",
            "multiplier": 2,
            "score": 50,
            "label": "Double Bull → 50",
        }

    # e.g. "T20" -> ring_code="T", sector=20
    ring_code = segment[0]
    sector = int(segment[1:])

    ring_names = {"S": "single", "D": "double", "T": "triple"}
    multipliers = {"S": 1, "D": 2, "T": 3}

    ring = ring_names[ring_code]
    multiplier = multipliers[ring_code]
    score = sector * multiplier
    label = f"{ring.capitalize()} {sector} → {score}"

    return {
        "segment": segment,
        "sector": sector,
        "ring": ring,
        "multiplier": multiplier,
        "score": score,
        "label": label,
    }


def make_class_name(segment: str) -> str:
    """Return the class name for a segment (identity in v2 — no ordinal)."""
    return segment


def segment_shorthand(text: str) -> str | None:
    """Parse user input like 't20', 's5', 'dbull', 'sbull' into a segment name.

    Case insensitive. Returns None if the input is not a valid segment.
    """
    text = text.strip().upper()

    if text in ("MISS", "M", "OUT"):
        return "MISS"
    if text in ("DBULL", "DB"):
        return "D_BULL"
    if text in ("SBULL", "SB", "BULL"):
        return "S_BULL"

    if len(text) < 2:
        return None

    ring = text[0]
    if ring not in ("S", "D", "T"):
        return None

    try:
        sector = int(text[1:])
    except ValueError:
        return None

    if sector < 1 or sector > 20:
        return None

    return f"{ring}{sector}"
