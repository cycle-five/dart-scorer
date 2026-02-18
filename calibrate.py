#!/usr/bin/env python3
"""
calibrate.py — Camera lens calibration and dartboard homography setup.

Usage:
    python calibrate.py --lens           # Lens distortion calibration (Phase 1)
    python calibrate.py --board          # Board homography calibration (Phase 2)
    python calibrate.py --lens --board   # Both in sequence
    python calibrate.py --debug          # Enable debug visualization (combine with above)
"""

import argparse
import os
import sys
import time

# Suppress Qt/Wayland warnings from pip-installed OpenCV on Linux
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import config
import board


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def open_camera():
    """Open the configured camera device and set resolution.

    Uses the V4L2 backend explicitly to ensure resolution control works
    on Linux systems.
    """
    # Use V4L2 backend for reliable resolution setting on Linux
    cap = cv2.VideoCapture(config.CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback to default backend
        cap = cv2.VideoCapture(config.CAMERA_DEVICE)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera at {config.CAMERA_DEVICE}")
        print("Check that the device exists and is not in use by another process.")
        sys.exit(1)
    # Set FOURCC to MJPG first — many USB cameras only support high res via MJPEG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w != config.CAMERA_WIDTH or actual_h != config.CAMERA_HEIGHT:
        print(f"WARNING: Requested {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT} "
              f"but got {actual_w}x{actual_h}")
    print(f"Camera opened: {config.CAMERA_DEVICE} at {actual_w}x{actual_h}")
    return cap


# ---------------------------------------------------------------------------
# Lens parameter I/O
# ---------------------------------------------------------------------------

def load_lens_params():
    """Load and return (camera_matrix, dist_coeffs, new_camera_matrix, roi)."""
    path = config.LENS_PARAMS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Lens parameters not found at {path}.\n"
            "Run 'python calibrate.py --lens' to generate them."
        )
    data = np.load(str(path))
    return (
        data["camera_matrix"],
        data["dist_coeffs"],
        data["new_camera_matrix"],
        data["roi"],
    )


def undistort_frame(frame, camera_matrix, dist_coeffs, new_camera_matrix, roi):
    """Undistort a frame using the given lens parameters.

    Applies cv2.undistort and optionally crops to the valid ROI.
    Returns the undistorted (and cropped) frame.
    """
    dst = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = [int(v) for v in roi]
    if w > 0 and h > 0:
        dst = dst[y:y + h, x:x + w]
    return dst


# ---------------------------------------------------------------------------
# Phase 1 — Lens calibration
# ---------------------------------------------------------------------------

def calibrate_lens(cap, debug=False):
    """Collect checkerboard frames and compute lens distortion parameters."""

    # Object points: 3-D coordinates of each inner corner (Z=0 plane).
    # Scaled by the physical square size so units are in millimetres.
    cols, rows = config.CHECKERBOARD_SIZE
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= config.CHECKERBOARD_SQUARE_SIZE

    obj_points = []   # 3-D points in world space
    img_points = []   # Corresponding 2-D points in image space

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    last_capture_time = 0.0
    captured = 0
    target = config.MIN_CALIBRATION_FRAMES
    gray = None  # will be set on first frame

    print("\n=== Lens Calibration ===")
    print(f"Hold the {cols}x{rows} checkerboard in front of the camera.")
    print(f"Need {target} frames. Auto-capture every {config.CALIBRATION_DELAY}s when detected.")
    print("Controls: SPACE = force capture  |  q = abort\n")

    window = "Lens Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while captured < target:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from camera.")
                break

            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            found, corners = cv2.findChessboardCorners(gray, config.CHECKERBOARD_SIZE, None)

            now = time.monotonic()
            do_capture = False

            if found:
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                if debug:
                    cv2.drawChessboardCorners(display, config.CHECKERBOARD_SIZE, corners_refined, found)

                # Auto-capture when enough time has elapsed
                if now - last_capture_time >= config.CALIBRATION_DELAY:
                    do_capture = True

                status_text = "Checkerboard detected"
                status_color = (0, 200, 0)
            else:
                corners_refined = None
                status_text = "No checkerboard detected"
                status_color = (0, 0, 220)

            # Overlay HUD
            h_frame, w_frame = display.shape[:2]
            overlay_y = 30
            cv2.putText(display, status_text, (10, overlay_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            overlay_y += 35
            cv2.putText(display, f"Captured: {captured} / {target}", (10, overlay_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            overlay_y += 35
            cv2.putText(display,
                        "Hold checkerboard steady - press SPACE to capture, or auto-capture when detected",
                        (10, overlay_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Aborted by user.")
                cv2.destroyWindow(window)
                return False
            elif key == ord(' ') and found and corners_refined is not None:
                # Manual capture via SPACE
                do_capture = True

            if do_capture and found and corners_refined is not None:
                obj_points.append(objp)
                img_points.append(corners_refined)
                last_capture_time = now
                captured += 1
                print(f"  Captured frame {captured}/{target}")

    except KeyboardInterrupt:
        print("\nCalibration interrupted.")
        cv2.destroyWindow(window)
        return False

    cv2.destroyWindow(window)

    if captured < target:
        print(f"ERROR: Only captured {captured}/{target} frames. Calibration aborted.")
        return False

    # ---------------------------------------------------------------------------
    # Compute calibration
    # ---------------------------------------------------------------------------
    print(f"\nComputing calibration from {captured} frames...")
    h_img, w_img = gray.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w_img, h_img), None, None
    )

    print(f"Reprojection error: {ret:.4f} px")
    if ret > 1.0:
        print("WARNING: Reprojection error > 1.0 px — calibration may be poor.")
        print("Try again with more varied checkerboard poses and better lighting.")

    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w_img, h_img), 1, (w_img, h_img))

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    save_path = config.LENS_PARAMS_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(save_path),
        camera_matrix=mtx,
        dist_coeffs=dist,
        new_camera_matrix=new_mtx,
        roi=roi,
    )
    print(f"Lens parameters saved to: {save_path}")

    # ---------------------------------------------------------------------------
    # Debug: side-by-side comparison on last frame
    # ---------------------------------------------------------------------------
    if debug:
        # Re-read one more frame to show undistorted preview
        ok, sample = cap.read()
        if ok:
            undist = undistort_frame(sample, mtx, dist, new_mtx, roi)
            # Resize both to the same height for side-by-side display
            h_s = min(sample.shape[0], undist.shape[0])
            def _resize_h(img, target_h):
                scale = target_h / img.shape[0]
                return cv2.resize(img, (int(img.shape[1] * scale), target_h))

            left = _resize_h(sample, h_s)
            right = _resize_h(undist, h_s)

            # Label each side
            cv2.putText(left,  "Original",    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(right, "Undistorted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            combined = np.hstack([left, right])
            cmp_win = "Distortion comparison (any key to close)"
            cv2.namedWindow(cmp_win, cv2.WINDOW_NORMAL)
            cv2.imshow(cmp_win, combined)
            print("Showing distortion comparison — press any key to close.")
            cv2.waitKey(0)
            cv2.destroyWindow(cmp_win)

    print("Lens calibration complete.\n")
    return True


# ---------------------------------------------------------------------------
# Phase 2 — Board calibration (manual 5-point click)
# ---------------------------------------------------------------------------

# Mouse callback state for point collection
_click_points = []
_click_frame = None


def _mouse_callback(event, x, y, flags, param):
    """Mouse callback for collecting calibration clicks."""
    global _click_points, _click_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_points.append((x, y))


def calibrate_board(cap, debug=False):
    """Phase 2: Manual 5-point board calibration.

    The user clicks 5 known landmarks on the dartboard to establish both
    the board's shape (perspective) and rotational orientation (where 20 is).

    Points to click (in order):
    1. Bullseye center
    2. Top of board — outer double wire at 12 o'clock (sector 20)
    3. Right of board — outer double wire at 3 o'clock (sector 6)
    4. Bottom of board — outer double wire at 6 o'clock (sector 3)
    5. Left of board — outer double wire at 9 o'clock (sector 11)

    These 5 points map to known canonical positions, giving us a homography
    that captures both perspective distortion AND board rotation.
    """
    global _click_points, _click_frame

    # --- Load lens calibration ---
    try:
        cam_mtx, dist, new_mtx, roi = load_lens_params()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return False

    print("\n=== Board Calibration (Manual 5-Point) ===")
    print("Position camera so the full dartboard is visible.")
    print("Press SPACE to capture a frame, then click 5 points.\n")

    window = "Board Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    captured_frame = None

    # --- Step 1: Capture a frame ---
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from camera.")
                return False

            undistorted = undistort_frame(frame, cam_mtx, dist, new_mtx, roi)
            display = undistorted.copy()

            cv2.putText(display, "Press SPACE to capture frame for calibration",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "Ensure full dartboard is visible | q = abort",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Aborted by user.")
                cv2.destroyWindow(window)
                return False
            elif key == ord(' '):
                captured_frame = undistorted.copy()
                print("Frame captured.\n")
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
        cv2.destroyWindow(window)
        return False

    # --- Step 2: Collect 5 clicks ---
    _click_points = []
    _click_frame = captured_frame.copy()
    cv2.setMouseCallback(window, _mouse_callback)

    point_labels = [
        "1/5: Click the BULLSEYE (center of board)",
        "2/5: Click 12 o'clock — outer wire at TOP (sector 20)",
        "3/5: Click 3 o'clock — outer wire at RIGHT (sector 6)",
        "4/5: Click 6 o'clock — outer wire at BOTTOM (sector 3)",
        "5/5: Click 9 o'clock — outer wire at LEFT (sector 11)",
    ]

    # Canonical destinations for each click:
    #   center    → (170, 170)
    #   top (20)  → (170, 0)     — 12 o'clock on outer ring
    #   right (6) → (340, 170)   — 3 o'clock
    #   bottom(3) → (170, 340)   — 6 o'clock
    #   left (11) → (0, 170)     — 9 o'clock
    canonical_r = config.DOUBLE_OUTER_RADIUS  # 170
    cx, cy = config.CANONICAL_CENTER          # (170, 170)
    dst_positions = [
        (float(cx), float(cy)),            # center
        (float(cx), float(cy - canonical_r)),  # top
        (float(cx + canonical_r), float(cy)),  # right
        (float(cx), float(cy + canonical_r)),  # bottom
        (float(cx - canonical_r), float(cy)),  # left
    ]

    print("Click 5 points on the board in order:")
    for lbl in point_labels:
        print(f"  {lbl}")
    print("Press 'u' to undo last click, 'q' to abort.\n")

    colors = [
        (0, 0, 255),    # center: red
        (0, 255, 255),  # top: yellow
        (0, 255, 0),    # right: green
        (255, 0, 0),    # bottom: blue
        (255, 0, 255),  # left: magenta
    ]

    try:
        while len(_click_points) < 5:
            display = captured_frame.copy()

            # Draw instruction
            idx = len(_click_points)
            cv2.putText(display, point_labels[idx],
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "u = undo last | q = abort",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Draw already-clicked points with labels
            click_labels = ["Center", "Top(20)", "Right(6)", "Bottom(3)", "Left(11)"]
            for i, pt in enumerate(_click_points):
                cv2.circle(display, pt, 8, colors[i], 2)
                cv2.circle(display, pt, 2, colors[i], -1)
                cv2.putText(display, click_labels[i], (pt[0] + 12, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)

            # Draw crosshair lines between cardinal points if we have them
            if len(_click_points) >= 2:
                # Draw line from center to each clicked cardinal point
                for i in range(1, len(_click_points)):
                    cv2.line(display, _click_points[0], _click_points[i],
                             colors[i], 1, cv2.LINE_AA)

            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("Aborted by user.")
                cv2.destroyWindow(window)
                return False
            elif key == ord('u') and _click_points:
                removed = _click_points.pop()
                print(f"  Undid click at ({removed[0]}, {removed[1]})")
    except KeyboardInterrupt:
        print("\nInterrupted.")
        cv2.destroyWindow(window)
        return False

    print(f"\nCollected {len(_click_points)} points:")
    click_labels = ["Center", "Top(20)", "Right(6)", "Bottom(3)", "Left(11)"]
    center_px = _click_points[0]
    for i, pt in enumerate(_click_points):
        if i == 0:
            print(f"  {click_labels[i]}: ({pt[0]}, {pt[1]})")
        else:
            # Compute pixel distance from center to this cardinal point
            dx = pt[0] - center_px[0]
            dy = pt[1] - center_px[1]
            px_dist = (dx**2 + dy**2) ** 0.5
            print(f"  {click_labels[i]}: ({pt[0]}, {pt[1]})  "
                  f"[{px_dist:.1f}px from center]")

    # Diagnostic: show the canonical mapping parameters
    print(f"\n--- Calibration diagnostics ---")
    print(f"  Canonical circle: {config.CANONICAL_DIAMETER}x{config.CANONICAL_DIAMETER}px, "
          f"center=({cx},{cy})")
    print(f"  Outer double radius (canonical): {canonical_r}px = {canonical_r}mm")
    print(f"  Click-derived radii (camera pixels):")
    for i in range(1, 5):
        dx = _click_points[i][0] - center_px[0]
        dy = _click_points[i][1] - center_px[1]
        print(f"    {click_labels[i]}: {(dx**2 + dy**2)**0.5:.1f}px")
    avg_radius = np.mean([
        ((p[0]-center_px[0])**2 + (p[1]-center_px[1])**2)**0.5
        for p in _click_points[1:]
    ])
    print(f"  Average click radius: {avg_radius:.1f}px")
    print(f"  Implied scale: {avg_radius:.1f}px → {canonical_r}px canonical "
          f"({canonical_r/avg_radius:.4f} px/px)")
    print(f"---")

    # --- Step 3: Compute homography ---
    # Use only the 5 clicked points. A homography has 8 DOF; 5 points give
    # 10 constraints, which is a clean slightly-overdetermined least-squares fit.
    # Do NOT interpolate midpoints — averaging camera-space coords assumes
    # linear distortion, but perspective distortion is projective, so
    # interpolated points would be wrong and poison the fit.
    src_pts = np.array(_click_points, dtype=np.float32)
    dst_pts = np.array(dst_positions, dtype=np.float32)

    H, mask = cv2.findHomography(src_pts, dst_pts, 0)

    if H is None:
        print("ERROR: Failed to compute homography.")
        cv2.destroyWindow(window)
        return False

    # Diagnostic: warp the original 5 click points through H and verify
    print(f"\n--- Homography verification ---")
    print(f"  Warping clicked points through H to check mapping accuracy:")
    orig_5 = np.array(_click_points, dtype=np.float32).reshape(-1, 1, 2)
    warped_pts = cv2.perspectiveTransform(orig_5, H)
    expected = dst_positions
    for i in range(5):
        wx, wy = warped_pts[i][0]
        ex, ey = expected[i]
        err = ((wx - ex)**2 + (wy - ey)**2) ** 0.5
        print(f"    {click_labels[i]}: clicked ({_click_points[i][0]}, {_click_points[i][1]}) "
              f"→ canonical ({wx:.1f}, {wy:.1f}), "
              f"expected ({ex:.0f}, {ey:.0f}), error={err:.1f}px")
    print(f"---")

    # --- Interactive scale adjustment + verification ---
    # Let the user press [ / ] to shrink/grow the mapping and visually align
    # the overlay rings with the actual board bands.
    scale_factor = 1.0
    scale_step = 0.005  # 0.5% per keypress

    canonical_size = config.CANONICAL_DIAMETER
    canon_win = "Canonical View — [ ] to scale, ENTER to accept, q to abort"
    cv2.namedWindow(canon_win, cv2.WINDOW_NORMAL)

    # Draw the clicked points on the original frame
    debug_frame = captured_frame.copy()
    for i, pt in enumerate(_click_points):
        cv2.circle(debug_frame, pt, 8, colors[min(i, len(colors)-1)], 2)
        cv2.circle(debug_frame, pt, 2, colors[min(i, len(colors)-1)], -1)
    for i in range(1, 5):
        cv2.line(debug_frame, _click_points[0], _click_points[i],
                 colors[i], 1, cv2.LINE_AA)
    cv2.imshow(window, debug_frame)

    print("\nVerify the canonical view: ring bands should align with the board.")
    print("Controls:")
    print("  ] = grow rings (if overlay is too small)")
    print("  [ = shrink rings (if overlay is too large)")
    print("  ENTER = accept and save")
    print("  q = abort and redo\n")

    while True:
        # Recompute homography with current scale
        # Scaling works by adjusting the canonical destination radius
        scaled_r = canonical_r * scale_factor
        scaled_cx, scaled_cy = cx, cy

        # Rebuild destination points with scaled radius (5 points only)
        scaled_dst = np.array([
            [float(scaled_cx), float(scaled_cy)],
            [float(scaled_cx), float(scaled_cy - scaled_r)],
            [float(scaled_cx + scaled_r), float(scaled_cy)],
            [float(scaled_cx), float(scaled_cy + scaled_r)],
            [float(scaled_cx - scaled_r), float(scaled_cy)],
        ], dtype=np.float32)

        H_scaled, _ = cv2.findHomography(src_pts, scaled_dst, 0)

        if H_scaled is None:
            print("ERROR: Homography failed at this scale.")
            break

        warped = cv2.warpPerspective(captured_frame, H_scaled,
                                      (canonical_size, canonical_size))
        warped_overlay = board.draw_board_overlay(warped, alpha=0.4)

        # Show scale info on the overlay
        scale_text = f"Scale: {scale_factor:.3f}  [ ] to adjust, ENTER to accept"
        cv2.putText(warped_overlay, scale_text, (5, canonical_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow(canon_win, warped_overlay)

        key = cv2.waitKey(30) & 0xFF
        if key == ord(']'):
            scale_factor += scale_step
            print(f"  Scale: {scale_factor:.3f} (growing)")
        elif key == ord('['):
            scale_factor -= scale_step
            if scale_factor < 0.5:
                scale_factor = 0.5
            print(f"  Scale: {scale_factor:.3f} (shrinking)")
        elif key == 13 or key == 10:  # ENTER
            H = H_scaled
            break
        elif key == ord('q'):
            print("Calibration aborted.")
            cv2.destroyWindow(window)
            cv2.destroyWindow(canon_win)
            return False

    cv2.destroyWindow(window)
    cv2.destroyWindow(canon_win)

    if scale_factor != 1.0:
        print(f"Applied scale adjustment: {scale_factor:.3f}")

    # --- Save ---
    save_path = config.BOARD_HOMOGRAPHY_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    center_pt = _click_points[0]
    np.savez(str(save_path), homography=H,
             ellipse_center=np.array(center_pt, dtype=np.float64),
             click_points=np.array(_click_points, dtype=np.float64),
             scale_factor=scale_factor)
    print(f"Board homography saved to: {save_path}")

    print("Board calibration complete.\n")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dartboard calibration tool")
    parser.add_argument("--lens",  action="store_true", help="Run lens distortion calibration")
    parser.add_argument("--board", action="store_true", help="Run board homography calibration")
    parser.add_argument("--debug", action="store_true", help="Show debug visualizations")
    args = parser.parse_args()

    if not args.lens and not args.board:
        parser.print_help()
        sys.exit(1)

    cap = open_camera()

    try:
        if args.lens:
            if not calibrate_lens(cap, debug=args.debug):
                print("Lens calibration failed. Aborting.")
                return
        if args.board:
            calibrate_board(cap, debug=args.debug)
    except KeyboardInterrupt:
        print("\nInterrupted — exiting cleanly.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
