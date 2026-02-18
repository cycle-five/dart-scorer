"""
board.py — Dartboard geometry and score mapping.

Converts pixel coordinates (in canonical top-down space) to dart scores
using WDF standard dartboard dimensions.
"""

import math
import cv2
import numpy as np
import config


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
    # Each sector spans 18°. Adding 9° offsets so sector 20 (centred at 0°)
    # covers 351°–9°.
    sector_index = int((theta + 9) % 360 / 18)
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
    for i in range(20):
        boundary_deg = (i * 18 - 9) % 360
        boundary_rad = math.radians(boundary_deg)
        end_x = int(round(cx + line_length * math.sin(boundary_rad)))
        end_y = int(round(cy - line_length * math.cos(boundary_rad)))
        cv2.line(blended, (cx, cy), (end_x, end_y), (255, 255, 255), 1)

    # --- Sector number labels ---
    label_r = config.DOUBLE_OUTER_RADIUS + 8  # just outside the board
    for i, sector_num in enumerate(config.SECTOR_ORDER):
        angle_deg = i * 18  # center of each sector
        angle_rad = math.radians(angle_deg)
        lx = int(round(cx + label_r * math.sin(angle_rad)))
        ly = int(round(cy - label_r * math.cos(angle_rad)))
        cv2.putText(blended, str(sector_num), (lx - 8, ly + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return blended
