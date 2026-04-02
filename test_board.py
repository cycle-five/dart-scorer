#!/usr/bin/env python3
"""
test_board.py — Unit tests for board geometry, scoring, and homography.

Run: uv run python -m pytest test_board.py -v
"""

import math
import pytest
import numpy as np
from dartscorer import config
from dartscorer.board import pixel_to_polar, get_sector, get_ring, score_dart, apply_homography
from dartscorer.classes_v2 import segment_shorthand, parse_class_name, make_class_name, CLASS_TO_ID, NUM_CLASSES


# ---------------------------------------------------------------------------
# pixel_to_polar
# ---------------------------------------------------------------------------

class TestPixelToPolar:
    def test_center_is_zero(self):
        cx, cy = config.CANONICAL_CENTER
        r, theta = pixel_to_polar(cx, cy)
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_straight_up(self):
        cx, cy = config.CANONICAL_CENTER
        r, theta = pixel_to_polar(cx, cy - 50)  # 50px above center
        assert r == pytest.approx(50.0, abs=1e-6)
        assert theta == pytest.approx(0.0, abs=1.0)  # top = 0 degrees

    def test_straight_right(self):
        cx, cy = config.CANONICAL_CENTER
        r, theta = pixel_to_polar(cx + 50, cy)
        assert r == pytest.approx(50.0, abs=1e-6)
        assert theta == pytest.approx(90.0, abs=1.0)

    def test_straight_down(self):
        cx, cy = config.CANONICAL_CENTER
        r, theta = pixel_to_polar(cx, cy + 50)
        assert r == pytest.approx(50.0, abs=1e-6)
        assert theta == pytest.approx(180.0, abs=1.0)

    def test_straight_left(self):
        cx, cy = config.CANONICAL_CENTER
        r, theta = pixel_to_polar(cx - 50, cy)
        assert r == pytest.approx(50.0, abs=1e-6)
        assert theta == pytest.approx(270.0, abs=1.0)

    def test_diagonal(self):
        cx, cy = config.CANONICAL_CENTER
        r, theta = pixel_to_polar(cx + 50, cy - 50)  # up-right
        assert r == pytest.approx(50 * math.sqrt(2), abs=0.1)
        assert theta == pytest.approx(45.0, abs=1.0)


# ---------------------------------------------------------------------------
# get_sector
# ---------------------------------------------------------------------------

class TestGetSector:
    def test_sector_20_at_top(self):
        assert get_sector(0.0) == 20   # top center = sector 20

    def test_sector_20_near_boundary(self):
        assert get_sector(8.9) == 20   # just inside 20
        assert get_sector(351.1) == 20  # just inside 20 (other side)

    def test_sector_1_clockwise_from_20(self):
        assert get_sector(18.0) == 1   # 18 degrees = center of sector 1

    def test_sector_5_at_bottom_left(self):
        assert get_sector(342.0) == 5  # center of sector 5

    def test_all_sectors_reachable(self):
        """Every sector should be reachable at its center angle."""
        for i, sector in enumerate(config.SECTOR_ORDER):
            angle = i * 18.0  # center of each sector
            assert get_sector(angle) == sector, f"Sector {sector} at {angle}°"

    def test_boundary_between_sectors(self):
        # Boundary at 9° is between sector 20 and sector 1
        # At exactly 9°, should be sector 1 (boundary goes to next)
        result = get_sector(9.0)
        assert result in (20, 1)  # boundary — either is acceptable


# ---------------------------------------------------------------------------
# get_ring
# ---------------------------------------------------------------------------

class TestGetRing:
    def test_inner_bull(self):
        ring, mult = get_ring(3.0)
        assert ring == "D-BULL"
        assert mult == 2

    def test_outer_bull(self):
        ring, mult = get_ring(10.0)
        assert ring == "S-BULL"
        assert mult == 1

    def test_inner_single(self):
        ring, mult = get_ring(50.0)
        assert ring == "single"
        assert mult == 1

    def test_triple(self):
        ring, mult = get_ring(103.0)
        assert ring == "triple"
        assert mult == 3

    def test_outer_single(self):
        ring, mult = get_ring(130.0)
        assert ring == "single"
        assert mult == 1

    def test_double(self):
        ring, mult = get_ring(166.0)
        assert ring == "double"
        assert mult == 2

    def test_miss(self):
        ring, mult = get_ring(175.0)
        assert ring == "miss"
        assert mult == 0

    def test_ring_boundaries(self):
        """Test values at exact ring boundaries."""
        # Inner bull boundary
        _, m = get_ring(config.INNER_BULL_RADIUS - 0.01)
        assert m == 2  # still inner bull
        _, m = get_ring(config.INNER_BULL_RADIUS + 0.01)
        assert m == 1  # outer bull

        # Triple boundaries
        _, m = get_ring(config.TRIPLE_INNER_RADIUS - 0.01)
        assert m == 1  # single
        _, m = get_ring(config.TRIPLE_INNER_RADIUS + 0.01)
        assert m == 3  # triple

        # Double boundaries
        _, m = get_ring(config.DOUBLE_OUTER_RADIUS - 0.01)
        assert m == 2  # double
        _, m = get_ring(config.DOUBLE_OUTER_RADIUS + 0.01)
        assert m == 0  # miss


# ---------------------------------------------------------------------------
# score_dart
# ---------------------------------------------------------------------------

class TestScoreDart:
    def test_bullseye(self):
        cx, cy = config.CANONICAL_CENTER
        result = score_dart(cx, cy)
        assert result["score"] == 50
        assert result["ring"] == "D-BULL"

    def test_single_bull(self):
        cx, cy = config.CANONICAL_CENTER
        result = score_dart(cx, cy - 10)  # 10mm above center
        assert result["score"] == 25
        assert result["ring"] == "S-BULL"

    def test_triple_20(self):
        cx, cy = config.CANONICAL_CENTER
        # Sector 20 is at top (0°), triple ring at ~103mm
        result = score_dart(cx, cy - 103)
        assert result["sector"] == 20
        assert result["ring"] == "triple"
        assert result["score"] == 60

    def test_miss(self):
        cx, cy = config.CANONICAL_CENTER
        result = score_dart(cx, cy - 200)  # way above board
        assert result["score"] == 0
        assert result["ring"] == "miss"

    def test_known_positions(self):
        """Test a few known positions."""
        cx, cy = config.CANONICAL_CENTER
        # Single 20 inner (50mm from center, straight up)
        r = score_dart(cx, cy - 50)
        assert r["sector"] == 20
        assert r["multiplier"] == 1
        assert r["score"] == 20


# ---------------------------------------------------------------------------
# apply_homography
# ---------------------------------------------------------------------------

class TestApplyHomography:
    def test_identity(self):
        H = np.eye(3, dtype=np.float64)
        x, y = apply_homography((100, 200), H)
        assert x == pytest.approx(100.0, abs=0.01)
        assert y == pytest.approx(200.0, abs=0.01)

    def test_translation(self):
        # Homography that translates by (10, 20)
        H = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]], dtype=np.float64)
        x, y = apply_homography((100, 200), H)
        assert x == pytest.approx(110.0, abs=0.01)
        assert y == pytest.approx(220.0, abs=0.01)

    def test_scale(self):
        H = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)
        x, y = apply_homography((50, 50), H)
        assert x == pytest.approx(100.0, abs=0.01)
        assert y == pytest.approx(100.0, abs=0.01)


# ---------------------------------------------------------------------------
# classes.py
# ---------------------------------------------------------------------------

class TestClasses:
    def test_segment_shorthand(self):
        assert segment_shorthand("t20") == "T20"
        assert segment_shorthand("s5") == "S5"
        assert segment_shorthand("dbull") == "D_BULL"
        assert segment_shorthand("sbull") == "S_BULL"
        assert segment_shorthand("D16") == "D16"
        assert segment_shorthand("bad") is None
        assert segment_shorthand("s0") is None
        assert segment_shorthand("s21") is None

    def test_parse_class_name(self):
        info = parse_class_name("T20")
        assert info["sector"] == 20
        assert info["ring"] == "triple"
        assert info["score"] == 60

    def test_parse_bull(self):
        info = parse_class_name("D_BULL")
        assert info["score"] == 50
        assert info["ring"] == "D-BULL"

    def test_parse_miss(self):
        info = parse_class_name("MISS")
        assert info["score"] == 0
        assert info["ring"] == "miss"

    def test_make_class_name(self):
        assert make_class_name("T20") == "T20"
        assert make_class_name("S_BULL") == "S_BULL"

    def test_all_classes_in_lookup(self):
        for s in range(1, 21):
            for r in ("S", "D", "T"):
                name = f"{r}{s}"
                assert name in CLASS_TO_ID, f"{name} not in CLASS_TO_ID"
        assert "S_BULL" in CLASS_TO_ID
        assert "D_BULL" in CLASS_TO_ID

    def test_miss_not_in_yolo_classes(self):
        """MISS is handled by geometry, not YOLO — should not be a class."""
        assert "MISS" not in CLASS_TO_ID

    def test_total_classes(self):
        assert len(CLASS_TO_ID) == 62
        assert NUM_CLASSES == 62


# ---------------------------------------------------------------------------
# Parallax correction (config.apply_parallax_correction)
# ---------------------------------------------------------------------------

class TestParallaxCorrection:
    """Tests for the wire parallax correction that adjusts ring radii
    based on camera viewing angle estimated from the homography."""

    def _reset_radii(self):
        """Reset global radii to standard values before each test."""
        config.INNER_BULL_RADIUS = config.INNER_BULL_RADIUS_STANDARD
        config.OUTER_BULL_RADIUS = config.OUTER_BULL_RADIUS_STANDARD
        config.TRIPLE_INNER_RADIUS = config.TRIPLE_INNER_RADIUS_STANDARD
        config.TRIPLE_OUTER_RADIUS = config.TRIPLE_OUTER_RADIUS_STANDARD
        config.DOUBLE_INNER_RADIUS = config.DOUBLE_INNER_RADIUS_STANDARD
        config.DOUBLE_OUTER_RADIUS = config.DOUBLE_OUTER_RADIUS_STANDARD

    @pytest.fixture(autouse=True)
    def reset(self):
        self._reset_radii()
        yield
        self._reset_radii()

    def test_none_homography_returns_zero(self):
        shift = config.apply_parallax_correction(None)
        assert shift == 0.0
        # Radii unchanged
        assert config.INNER_BULL_RADIUS == config.INNER_BULL_RADIUS_STANDARD

    def test_identity_homography_no_shift(self):
        """A perpendicular camera (identity homography) should produce
        zero or near-zero parallax shift."""
        H = np.eye(3, dtype=np.float64)
        shift = config.apply_parallax_correction(H)
        assert shift == pytest.approx(0.0, abs=0.01)
        assert config.DOUBLE_OUTER_RADIUS == pytest.approx(
            config.DOUBLE_OUTER_RADIUS_STANDARD, abs=0.01
        )

    def test_tilted_homography_positive_shift(self):
        """A perspective homography simulating a tilted camera should
        produce a positive outward shift on all radii."""
        # Simulate a camera tilted ~20° vertically: the top of the board
        # appears smaller than the bottom in camera space.
        # Build homography from 4-point correspondence.
        import cv2

        cx, cy = config.CANONICAL_CENTER
        r = config.DOUBLE_OUTER_RADIUS_STANDARD
        # Canonical (top-down) corners
        src = np.array(
            [[cx - r, cy - r], [cx + r, cy - r],
             [cx + r, cy + r], [cx - r, cy + r]],
            dtype=np.float32,
        )
        # Camera-space corners: top edge narrower (perspective foreshortening)
        squeeze = 0.8  # top is 80% the width of bottom → noticeable tilt
        dst = np.array(
            [[cx - r * squeeze, cy - r], [cx + r * squeeze, cy - r],
             [cx + r, cy + r], [cx - r, cy + r]],
            dtype=np.float32,
        )
        H, _ = cv2.findHomography(src, dst)

        shift = config.apply_parallax_correction(H)
        assert shift > 0.0
        # All radii should have increased by the same shift
        assert config.INNER_BULL_RADIUS == pytest.approx(
            config.INNER_BULL_RADIUS_STANDARD + shift
        )
        assert config.DOUBLE_OUTER_RADIUS == pytest.approx(
            config.DOUBLE_OUTER_RADIUS_STANDARD + shift
        )

    def test_shift_increases_with_tilt(self):
        """More extreme tilt should produce a larger parallax shift."""
        import cv2

        cx, cy = config.CANONICAL_CENTER
        r = config.DOUBLE_OUTER_RADIUS_STANDARD
        src = np.array(
            [[cx - r, cy - r], [cx + r, cy - r],
             [cx + r, cy + r], [cx - r, cy + r]],
            dtype=np.float32,
        )

        shifts = []
        for squeeze in (0.95, 0.80, 0.60):
            self._reset_radii()
            dst = np.array(
                [[cx - r * squeeze, cy - r], [cx + r * squeeze, cy - r],
                 [cx + r, cy + r], [cx - r, cy + r]],
                dtype=np.float32,
            )
            H, _ = cv2.findHomography(src, dst)
            shifts.append(config.apply_parallax_correction(H))

        # Each tilt should produce a larger shift than the previous
        assert shifts[0] < shifts[1] < shifts[2]

    def test_zero_wire_height_disables_correction(self):
        """With WIRE_HEIGHT_MM=0, no correction should be applied."""
        original_wh = config.WIRE_HEIGHT_MM
        try:
            config.WIRE_HEIGHT_MM = 0.0
            H = np.eye(3, dtype=np.float64)
            shift = config.apply_parallax_correction(H)
            assert shift == 0.0
        finally:
            config.WIRE_HEIGHT_MM = original_wh

    def test_all_six_radii_updated(self):
        """All six ring boundary radii should be updated by the same shift."""
        import cv2

        cx, cy = config.CANONICAL_CENTER
        r = config.DOUBLE_OUTER_RADIUS_STANDARD
        src = np.array(
            [[cx - r, cy - r], [cx + r, cy - r],
             [cx + r, cy + r], [cx - r, cy + r]],
            dtype=np.float32,
        )
        dst = np.array(
            [[cx - r * 0.85, cy - r], [cx + r * 0.85, cy - r],
             [cx + r, cy + r], [cx - r, cy + r]],
            dtype=np.float32,
        )
        H, _ = cv2.findHomography(src, dst)
        shift = config.apply_parallax_correction(H)
        assert shift > 0

        pairs = [
            (config.INNER_BULL_RADIUS, config.INNER_BULL_RADIUS_STANDARD),
            (config.OUTER_BULL_RADIUS, config.OUTER_BULL_RADIUS_STANDARD),
            (config.TRIPLE_INNER_RADIUS, config.TRIPLE_INNER_RADIUS_STANDARD),
            (config.TRIPLE_OUTER_RADIUS, config.TRIPLE_OUTER_RADIUS_STANDARD),
            (config.DOUBLE_INNER_RADIUS, config.DOUBLE_INNER_RADIUS_STANDARD),
            (config.DOUBLE_OUTER_RADIUS, config.DOUBLE_OUTER_RADIUS_STANDARD),
        ]
        for corrected, standard in pairs:
            assert corrected == pytest.approx(standard + shift)

    def test_shift_magnitude_reasonable(self):
        """For a typical ~15-25° tilt, shift should be roughly 0.3-0.7mm
        with the default 1.2mm wire height."""
        import cv2

        cx, cy = config.CANONICAL_CENTER
        r = config.DOUBLE_OUTER_RADIUS_STANDARD
        src = np.array(
            [[cx - r, cy - r], [cx + r, cy - r],
             [cx + r, cy + r], [cx - r, cy + r]],
            dtype=np.float32,
        )
        # ~20° tilt
        squeeze = 0.85
        dst = np.array(
            [[cx - r * squeeze, cy - r], [cx + r * squeeze, cy - r],
             [cx + r, cy + r], [cx - r, cy + r]],
            dtype=np.float32,
        )
        H, _ = cv2.findHomography(src, dst)
        shift = config.apply_parallax_correction(H)
        # Should be small but non-trivial — sub-millimeter to a few mm
        assert 0.1 < shift < 5.0

    def test_singular_homography_returns_zero(self):
        """A degenerate/singular homography should not crash."""
        H = np.zeros((3, 3), dtype=np.float64)
        shift = config.apply_parallax_correction(H)
        assert shift == 0.0
