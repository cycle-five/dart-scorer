"""
classes.py — YOLO class definitions for 189-class dart detection.

Classes encode dart ordinal (1st, 2nd, 3rd) and board segment.
Format: d{ordinal}_{segment}

Segments (63 total):
  S1..S20   — single 1-20
  D1..D20   — double 1-20
  T1..T20   — triple 1-20
  S_BULL    — single bull (25)
  D_BULL    — double bull (50)
  MISS      — outside scoring area (0 points)

Total: 3 ordinals × 63 segments = 189 classes.
"""

SECTORS = list(range(1, 21))
RINGS = ["S", "D", "T"]  # single, double, triple
BULLS = ["S_BULL", "D_BULL"]
ORDINALS = [1, 2, 3]

# Build segment list (63 segments)
SEGMENTS = []
for s in SECTORS:
    for r in RINGS:
        SEGMENTS.append(f"{r}{s}")
SEGMENTS.extend(BULLS)
SEGMENTS.append("MISS")

# Build full class list (186 classes)
CLASS_NAMES = []
for d in ORDINALS:
    for seg in SEGMENTS:
        CLASS_NAMES.append(f"d{d}_{seg}")

# Lookup dicts
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

NUM_CLASSES = len(CLASS_NAMES)  # 186


def parse_class_name(name):
    """Parse a class name into (ordinal, segment, sector, ring, multiplier, score).

    Returns:
        dict with keys: ordinal, segment, sector, ring, multiplier, score, label
    """
    parts = name.split("_", 1)
    ordinal = int(parts[0][1])  # d1 -> 1
    segment = parts[1]

    if segment == "MISS":
        return {
            "ordinal": ordinal,
            "segment": segment,
            "sector": 0,
            "ring": "miss",
            "multiplier": 0,
            "score": 0,
            "label": "Miss → 0",
        }
    elif segment == "S_BULL":
        return {
            "ordinal": ordinal,
            "segment": segment,
            "sector": 25,
            "ring": "S-BULL",
            "multiplier": 1,
            "score": 25,
            "label": "Single Bull → 25",
        }
    elif segment == "D_BULL":
        return {
            "ordinal": ordinal,
            "segment": segment,
            "sector": 25,
            "ring": "D-BULL",
            "multiplier": 2,
            "score": 50,
            "label": "Double Bull → 50",
        }

    # e.g. "T20" -> ring="T", sector=20
    ring_code = segment[0]
    sector = int(segment[1:])

    ring_names = {"S": "single", "D": "double", "T": "triple"}
    multipliers = {"S": 1, "D": 2, "T": 3}

    ring = ring_names[ring_code]
    multiplier = multipliers[ring_code]
    score = sector * multiplier
    label = f"{ring.capitalize()} {sector} → {score}"

    return {
        "ordinal": ordinal,
        "segment": segment,
        "sector": sector,
        "ring": ring,
        "multiplier": multiplier,
        "score": score,
        "label": label,
    }


def segment_shorthand(text):
    """Parse user input like 't20', 's5', 'dbull', 'sbull' into a segment name.

    Case insensitive. Returns None if invalid.
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


def make_class_name(ordinal, segment):
    """Build a class name from ordinal (1-3) and segment string."""
    return f"d{ordinal}_{segment}"
