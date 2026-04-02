"""
board.py — Dartboard geometry and score mapping.

Converts pixel coordinates (in canonical top-down space) to dart scores
using WDF standard dartboard dimensions.
"""

import math
import cv2
import numpy as np
from dartscorer import config


def apply_homography(point, homography):
    """Transform a point from camera pixel space to canonical space.

    Args:
        point: (x, y) tuple in camera pixel coordinates.
        homography: 3x3 homography matrix (numpy array).

    Returns:
        (x, y) tuple in canonical pixel coordinates.
    """
    pts = np.array([[[point[0], point[1]]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pts, homography)
    return (float(transformed[0][0][0]), float(transformed[0][0][1]))


def pixel_to_polar(x, y):
    """Convert canonical pixel coordinates to polar coordinates.

    Center is config.CANONICAL_CENTER. Radius equals distance in mm
    (1 px = 1 mm). Angle is measured clockwise from 12 o'clock (top).

    Args:
        x: Pixel x-coordinate in canonical space.
        y: Pixel y-coordinate in canonical space.

    Returns:
        (r, theta) where r is radius in mm and theta is degrees [0, 360).
    """
    cx, cy = config.CANONICAL_CENTER
    dx = x - cx
    dy = y - cy
    r = math.sqrt(dx * dx + dy * dy)
    # atan2(dx, -dy): dx gives East-West, -dy flips Y so top is 0°.
    # Result is clockwise from top, matching dartboard convention.
    theta = (math.degrees(math.atan2(dx, -dy)) + 360) % 360
    return r, theta


def get_sector(theta):
    """Return the dartboard sector number for a given angle.

    Args:
        theta: Angle in degrees, clockwise from top (0–360).

    Returns:
        Integer sector number (1–20).
    """
    # Each sector spans SECTOR_SPAN_DEG. Adding SECTOR_BOUNDARY_OFFSET
    # so sector 20 (centred at 0°) covers 351°–9°.
    sector_index = int((theta + config.SECTOR_BOUNDARY_OFFSET) % 360 / config.SECTOR_SPAN_DEG)
    return config.SECTOR_ORDER[sector_index]


def get_ring(r):
    """Return ring name and score multiplier for a given radius.

    Args:
        r: Radius from board centre in mm.

    Returns:
        Tuple of (ring_name: str, multiplier: int).
    """
    if r < config.INNER_BULL_RADIUS:
        return ("D-BULL", 2)
    if r < config.OUTER_BULL_RADIUS:
        return ("S-BULL", 1)
    if r < config.TRIPLE_INNER_RADIUS:
        return ("single", 1)
    if r < config.TRIPLE_OUTER_RADIUS:
        return ("triple", 3)
    if r < config.DOUBLE_INNER_RADIUS:
        return ("single", 1)
    if r < config.DOUBLE_OUTER_RADIUS:
        return ("double", 2)
    return ("miss", 0)


def classify_dart(x, y):
    """Classify a dart with confidence based on proximity to wires.

    Args:
        x: Canonical pixel x-coordinate (1px ≈ 1mm).
        y: Canonical pixel y-coordinate.

    Returns:
        Dict with keys: segment, sector, ring, ring_code, score, label,
        r, theta, confidence, sector_candidates, ring_candidates,
        sector_wire_dist_deg, ring_wire_dist_mm.

        confidence is 0.0–1.0 based on distance from nearest wire:
          - 1.0 = far from any wire (high confidence)
          - 0.0 = right on a wire (ambiguous)
    """
    r, theta = pixel_to_polar(x, y)
    ring_name, multiplier = get_ring(r)
    sector = get_sector(theta) if ring_name not in ("D-BULL", "S-BULL", "miss") else 0

    # --- Sector confidence (arc-length wire proximity) ---
    # Each sector spans SECTOR_SPAN_DEG. Distance to nearest sector wire
    # is converted to arc length (mm) at the dart's radius so that sector
    # and ring confidence are in the same physical units.
    sector_wire_dist = config.SECTOR_BOUNDARY_OFFSET  # max = center of sector (degrees)
    sector_wire_arc_mm = float('inf')  # arc-length in mm
    if ring_name not in ("D-BULL", "S-BULL", "miss"):
        offset = (theta + config.SECTOR_BOUNDARY_OFFSET) % config.SECTOR_SPAN_DEG
        sector_wire_dist = min(offset, config.SECTOR_SPAN_DEG - offset)
        sector_wire_arc_mm = r * math.radians(sector_wire_dist)

    # --- Ring confidence (radial wire proximity) ---
    ring_boundaries = [
        config.INNER_BULL_RADIUS,
        config.OUTER_BULL_RADIUS,
        config.TRIPLE_INNER_RADIUS,
        config.TRIPLE_OUTER_RADIUS,
        config.DOUBLE_INNER_RADIUS,
        config.DOUBLE_OUTER_RADIUS,
    ]
    ring_wire_dist = min(abs(r - b) for b in ring_boundaries)

    # --- Combined confidence ---
    # Both sector and ring use mm-from-wire with the same divisor.
    # Wire width is ~1.6mm; at WIRE_CONF_DIVISOR_MM the confidence is 1.0.
    sector_conf = min(sector_wire_arc_mm / config.WIRE_CONF_DIVISOR_MM, 1.0)
    ring_conf = min(ring_wire_dist / config.WIRE_CONF_DIVISOR_MM, 1.0)
    confidence = min(sector_conf, ring_conf)

    # --- Sector candidates (when near angular wire) ---
    sector_candidates = [sector] if sector else []
    if sector and sector_wire_arc_mm < config.WIRE_AMBIGUITY_THRESHOLD_MM:
        # Find adjacent sectors
        idx = config.SECTOR_ORDER.index(sector)
        left = config.SECTOR_ORDER[(idx - 1) % config.NUM_SECTORS]
        right = config.SECTOR_ORDER[(idx + 1) % config.NUM_SECTORS]
        # Which side is closer?
        offset = (theta + config.SECTOR_BOUNDARY_OFFSET) % config.SECTOR_SPAN_DEG
        if offset < config.SECTOR_BOUNDARY_OFFSET:
            # Closer to the wire on the "left" (clockwise previous)
            sector_candidates.append(config.SECTOR_ORDER[(idx - 1) % config.NUM_SECTORS])
        else:
            sector_candidates.append(config.SECTOR_ORDER[(idx + 1) % config.NUM_SECTORS])

    # --- Ring candidates (when near radial boundary) ---
    RING_MARGIN = config.RING_MARGIN_MM
    ring_candidates = [ring_name]
    if ring_name == "D-BULL" and abs(r - config.INNER_BULL_RADIUS) < RING_MARGIN:
        ring_candidates.append("S-BULL")
    elif ring_name == "S-BULL":
        if abs(r - config.INNER_BULL_RADIUS) < RING_MARGIN:
            ring_candidates.append("D-BULL")
        if abs(r - config.OUTER_BULL_RADIUS) < RING_MARGIN:
            ring_candidates.append("single")
    elif ring_name == "single":
        if abs(r - config.OUTER_BULL_RADIUS) < RING_MARGIN:
            ring_candidates.append("S-BULL")
        if abs(r - config.TRIPLE_INNER_RADIUS) < RING_MARGIN:
            ring_candidates.append("triple")
        if abs(r - config.DOUBLE_INNER_RADIUS) < RING_MARGIN:
            ring_candidates.append("double")
    elif ring_name == "triple":
        if abs(r - config.TRIPLE_INNER_RADIUS) < RING_MARGIN:
            ring_candidates.append("single")
        if abs(r - config.TRIPLE_OUTER_RADIUS) < RING_MARGIN:
            ring_candidates.append("single")
    elif ring_name == "double":
        if abs(r - config.DOUBLE_INNER_RADIUS) < RING_MARGIN:
            ring_candidates.append("single")
        if abs(r - config.DOUBLE_OUTER_RADIUS) < RING_MARGIN:
            ring_candidates.append("miss")
    elif ring_name == "miss" and abs(r - config.DOUBLE_OUTER_RADIUS) < RING_MARGIN:
        ring_candidates.append("double")

    # --- Build segment name ---
    ring_code_map = {"single": "S", "double": "D", "triple": "T"}
    if ring_name == "D-BULL":
        segment = "D_BULL"
        ring_code = "D_BULL"
        label = "Double Bull → 50"
        score = 50
    elif ring_name == "S-BULL":
        segment = "S_BULL"
        ring_code = "S_BULL"
        label = "Single Bull → 25"
        score = 25
    elif ring_name == "miss":
        segment = "MISS"
        ring_code = "MISS"
        label = "Miss → 0"
        score = 0
    else:
        ring_code = ring_code_map[ring_name]
        segment = f"{ring_code}{sector}"
        score = sector * multiplier
        label = f"{ring_name.capitalize()} {sector} → {score}"

    return {
        "segment": segment,
        "sector": sector,
        "ring": ring_name,
        "ring_code": ring_code,
        "multiplier": multiplier,
        "score": score,
        "label": label,
        "r": r,
        "theta": theta,
        "confidence": confidence,
        "sector_candidates": sector_candidates,
        "ring_candidates": ring_candidates,
        "sector_wire_dist_deg": sector_wire_dist,
        "ring_wire_dist_mm": ring_wire_dist,
    }


def classify_from_camera(point, homography):
    """Classify a dart from camera pixel coordinates with confidence.

    Args:
        point: (x, y) tuple in camera pixel space.
        homography: 3x3 homography matrix.

    Returns:
        Classification dict as returned by classify_dart, or None on failure.
    """
    try:
        canonical = apply_homography(point, homography)
        return classify_dart(canonical[0], canonical[1])
    except Exception:
        return None


def score_dart(x, y):
    """Score a dart given its canonical pixel coordinates.

    Args:
        x: Canonical pixel x-coordinate.
        y: Canonical pixel y-coordinate.

    Returns:
        Dict with keys: x, y, r, theta, sector, ring, multiplier,
        base_score, score, label.
    """
    r, theta = pixel_to_polar(x, y)
    ring_name, multiplier = get_ring(r)

    if ring_name == "D-BULL":
        sector = 25  # conventional placeholder
        base_score = 50
        score = 50
        label = "Double Bull → 50"
    elif ring_name == "S-BULL":
        sector = 25
        base_score = 25
        score = 25
        label = "Single Bull → 25"
    elif ring_name == "miss":
        sector = 0
        base_score = 0
        score = 0
        label = "Miss"
    else:
        sector = get_sector(theta)
        base_score = sector
        score = sector * multiplier
        ring_display = ring_name.capitalize()
        label = f"{ring_display} {sector} → {score}"

    return {
        "x": x,
        "y": y,
        "r": r,
        "theta": theta,
        "sector": sector,
        "ring": ring_name,
        "multiplier": multiplier,
        "base_score": base_score,
        "score": score,
        "label": label,
    }


def score_from_camera(point, homography):
    """Score a dart from raw camera pixel coordinates.

    Applies the homography transform then delegates to score_dart.

    Args:
        point: (x, y) tuple in camera pixel space.
        homography: 3x3 homography matrix (numpy array).

    Returns:
        Score dict as returned by score_dart.
    """
    canonical = apply_homography(point, homography)
    return score_dart(canonical[0], canonical[1])


def draw_board_overlay(img, alpha=0.3):
    """Draw a dartboard grid overlay on an image for debug visualisation.

    Draws semi-transparent filled ring bands (double, triple, bull) so the
    user can see whether the overlay aligns with the actual colored bands on
    the physical board.  Also draws sector boundary lines and ring edge circles.

    Args:
        img: Input image (numpy array, BGR, any dtype).
        alpha: Blend factor for the overlay (0 = invisible, 1 = opaque).

    Returns:
        Blended image as a numpy array with the same shape and dtype as img.
    """
    overlay = img.copy()
    cx, cy = config.CANONICAL_CENTER

    # --- Filled ring bands (draw outer first, then inner to punch out) ---
    # Double ring band — red filled
    cv2.circle(overlay, (cx, cy), int(round(config.DOUBLE_OUTER_RADIUS)),
               (0, 0, 200), -1)
    cv2.circle(overlay, (cx, cy), int(round(config.DOUBLE_INNER_RADIUS)),
               (0, 0, 0), -1)  # punch out inner

    # Triple ring band — red filled
    cv2.circle(overlay, (cx, cy), int(round(config.TRIPLE_OUTER_RADIUS)),
               (0, 0, 200), -1)
    cv2.circle(overlay, (cx, cy), int(round(config.TRIPLE_INNER_RADIUS)),
               (0, 0, 0), -1)  # punch out inner

    # Outer bull band — green filled
    cv2.circle(overlay, (cx, cy), int(round(config.OUTER_BULL_RADIUS)),
               (0, 180, 0), -1)

    # Inner bull (bullseye) — brighter green/red filled
    cv2.circle(overlay, (cx, cy), int(round(config.INNER_BULL_RADIUS)),
               (0, 0, 220), -1)

    # Blend the filled bands: use a mask so only the ring pixels blend
    ring_mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    # The "punch out" areas are black (0), ring bands have colour
    _, ring_mask = cv2.threshold(ring_mask, 1, 255, cv2.THRESH_BINARY)

    # Blend only where rings are drawn
    blended = img.copy()
    ring_region = cv2.bitwise_and(overlay, overlay, mask=ring_mask)
    bg_region = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(ring_mask))
    combined_rings = cv2.add(ring_region, bg_region)
    blended = cv2.addWeighted(combined_rings, alpha, img, 1 - alpha, 0)

    # --- Ring edge circles (thin lines for precision) ---
    ring_edges = [
        (config.INNER_BULL_RADIUS,   (0, 255, 0),   1),
        (config.OUTER_BULL_RADIUS,   (0, 255, 0),   1),
        (config.TRIPLE_INNER_RADIUS, (0, 0, 255),   1),
        (config.TRIPLE_OUTER_RADIUS, (0, 0, 255),   1),
        (config.DOUBLE_INNER_RADIUS, (0, 0, 255),   1),
        (config.DOUBLE_OUTER_RADIUS, (0, 0, 255),   1),
    ]
    for radius_mm, colour, thickness in ring_edges:
        cv2.circle(blended, (cx, cy), int(round(radius_mm)), colour, thickness)

    # --- Sector dividing lines ---
    line_length = config.DOUBLE_OUTER_RADIUS
    for i in range(config.NUM_SECTORS):
        boundary_deg = (i * config.SECTOR_SPAN_DEG - config.SECTOR_BOUNDARY_OFFSET) % 360
        boundary_rad = math.radians(boundary_deg)
        end_x = int(round(cx + line_length * math.sin(boundary_rad)))
        end_y = int(round(cy - line_length * math.cos(boundary_rad)))
        cv2.line(blended, (cx, cy), (end_x, end_y), (255, 255, 255), 1)

    # --- Sector number labels ---
    label_r = config.DOUBLE_OUTER_RADIUS + 8  # just outside the board
    for i, sector_num in enumerate(config.SECTOR_ORDER):
        angle_deg = i * config.SECTOR_SPAN_DEG  # center of each sector
        angle_rad = math.radians(angle_deg)
        lx = int(round(cx + label_r * math.sin(angle_rad)))
        ly = int(round(cy - label_r * math.cos(angle_rad)))
        cv2.putText(blended, str(sector_num), (lx - 8, ly + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return blended
