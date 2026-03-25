"""
label_analyzer.py — Web-based label analysis/debugging tool for YOLO dart training data.

Run with: uv run python label_analyzer.py
Then open: http://localhost:8765
"""

import io
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Must set before importing cv2
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from classes import ID_TO_CLASS, CLASS_TO_ID, SEGMENTS, ORDINALS

PORT = 8765
DATA_DIR = Path(__file__).parent / "data" / "training"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_label_index():
    """Return dict: class_id -> list of (stem, boxes) where boxes is list of
    (class_id, cx, cy, w, h) for all annotations in that image."""
    index = {cid: [] for cid in ID_TO_CLASS}
    label_files = sorted(LABELS_DIR.glob("*.txt"))
    for lf in label_files:
        stem = lf.stem
        boxes_by_class = {}
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes_by_class.setdefault(cid, []).append((cid, cx, cy, w, h))
        for cid, boxes in boxes_by_class.items():
            if cid in index:
                index[cid].append((stem, boxes))
    return index


# Cache at module level — loaded once on first request
_label_index = None

def get_index():
    global _label_index
    if _label_index is None:
        _label_index = load_label_index()
    return _label_index


def count_per_class():
    """Return dict: class_id -> sample count."""
    idx = get_index()
    return {cid: len(entries) for cid, entries in idx.items()}

# ---------------------------------------------------------------------------
# Image annotation (OpenCV)
# ---------------------------------------------------------------------------

def render_annotated_thumbnail(stem, target_class_id, thumb_w=300):
    """Load image, draw all bboxes, highlight target class. Return JPEG bytes."""
    img_path = IMAGES_DIR / f"{stem}.png"
    if not img_path.exists():
        # Try jpg
        img_path = IMAGES_DIR / f"{stem}.jpg"
    if not img_path.exists():
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]

    label_path = LABELS_DIR / f"{stem}.txt"
    boxes = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                boxes.append((cid, x1, y1, x2, y2))

    # Draw dimmed boxes first
    overlay = img.copy()
    for cid, x1, y1, x2, y2 in boxes:
        if cid != target_class_id:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 80, 80), 1)
            label = ID_TO_CLASS.get(cid, str(cid))
            cv2.putText(overlay, label, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    # Draw highlighted box on top
    for cid, x1, y1, x2, y2 in boxes:
        if cid == target_class_id:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = ID_TO_CLASS.get(cid, str(cid))
            # Background for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(y1 - 4, th + 2)
            cv2.rectangle(overlay, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), (0, 200, 0), -1)
            cv2.putText(overlay, label, (x1 + 1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    img = cv2.addWeighted(overlay, 1.0, img, 0.0, 0)

    # Resize to thumbnail width
    scale = thumb_w / w
    new_h = int(h * scale)
    img = cv2.resize(img, (thumb_w, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return bytes(buf)

# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

COLOR_RED    = "#c0392b"
COLOR_ORANGE = "#e67e22"
COLOR_YELLOW = "#f1c40f"
COLOR_GREEN  = "#27ae60"
COLOR_NONE   = "#2c3e50"

def count_color(n):
    if n == 0:    return COLOR_RED
    if n <= 3:    return COLOR_ORANGE
    if n <= 10:   return COLOR_YELLOW
    return COLOR_GREEN

def text_color(n):
    if n <= 10:   return "#000"
    return "#fff"

HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dart Label Analyzer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: monospace; background: #1a1a2e; color: #eee; padding: 16px; }
  h1 { font-size: 1.4rem; margin-bottom: 12px; color: #a0c4ff; }
  h2 { font-size: 1.1rem; margin-bottom: 10px; color: #a0c4ff; }
  a { color: #a0c4ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .back { display: inline-block; margin-bottom: 16px; padding: 6px 12px;
          background: #333; border-radius: 4px; }
  .stats { margin-bottom: 12px; font-size: 0.85rem; color: #aaa; }
  .legend { display: flex; gap: 12px; margin-bottom: 14px; font-size: 0.8rem; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 5px; }
  .legend-swatch { width: 16px; height: 16px; border-radius: 3px; flex-shrink: 0; }
  table.heatmap { border-collapse: collapse; font-size: 0.75rem; }
  table.heatmap th { padding: 4px 8px; background: #2a2a4a; color: #ccc; font-weight: normal; }
  table.heatmap td { padding: 0; }
  table.heatmap td a { display: block; width: 58px; height: 32px; line-height: 32px;
                        text-align: center; font-weight: bold; border: 1px solid #1a1a2e; }
  table.heatmap td a:hover { opacity: 0.8; filter: brightness(1.2); text-decoration: none; }
  .segment-label { padding: 4px 8px; background: #2a2a4a; color: #ccc;
                   text-align: right; white-space: nowrap; font-size: 0.75rem; }
  .thumb-grid { display: flex; flex-wrap: wrap; gap: 12px; }
  .thumb-card { background: #2a2a4a; border-radius: 6px; overflow: hidden;
                border: 1px solid #444; }
  .thumb-card img { display: block; width: 300px; height: auto; }
  .thumb-label { padding: 4px 8px; font-size: 0.7rem; color: #aaa;
                 white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                 max-width: 300px; }
  .empty { color: #888; font-style: italic; padding: 20px 0; }
</style>
</head>
<body>
"""

HTML_FOOTER = "</body></html>\n"


def heatmap_page(counts):
    total_images = len(list(LABELS_DIR.glob("*.txt")))
    total_labeled = sum(counts.values())

    parts = [HTML_HEADER]
    parts.append(f"<h1>Dart Label Analyzer — Heatmap</h1>\n")
    parts.append(f'<div class="stats">'
                 f'{total_images} images &nbsp;|&nbsp; '
                 f'{total_labeled} labeled instances across {len([v for v in counts.values() if v>0])} classes'
                 f'</div>\n')
    parts.append('<div class="legend">\n')
    for color, label in [
        (COLOR_RED,    "0 samples"),
        (COLOR_ORANGE, "1–3 samples"),
        (COLOR_YELLOW, "4–10 samples"),
        (COLOR_GREEN,  "10+ samples"),
    ]:
        parts.append(f'<div class="legend-item">'
                     f'<div class="legend-swatch" style="background:{color}"></div>'
                     f'{label}</div>\n')
    parts.append('</div>\n')

    parts.append('<table class="heatmap">\n')
    # Header row: ordinals
    parts.append('<thead><tr><th>Segment</th>')
    for d in ORDINALS:
        parts.append(f'<th>d{d}</th>')
    parts.append('</tr></thead>\n<tbody>\n')

    for seg in SEGMENTS:
        parts.append('<tr>')
        parts.append(f'<td class="segment-label">{seg}</td>')
        for d in ORDINALS:
            class_name = f"d{d}_{seg}"
            cid = CLASS_TO_ID.get(class_name)
            n = counts.get(cid, 0) if cid is not None else 0
            bg = count_color(n)
            fg = text_color(n)
            if cid is not None and n > 0:
                href = f"/class/{cid}"
                cell = (f'<a href="{href}" style="background:{bg};color:{fg}">'
                        f'{n}</a>')
            else:
                cell = (f'<a href="#" style="background:{bg};color:{fg};cursor:default">'
                        f'{n}</a>')
            parts.append(f'<td>{cell}</td>')
        parts.append('</tr>\n')

    parts.append('</tbody></table>\n')
    parts.append(HTML_FOOTER)
    return "".join(parts).encode("utf-8")


def class_detail_page(class_id):
    class_name = ID_TO_CLASS.get(class_id, f"class_{class_id}")
    entries = get_index().get(class_id, [])

    parts = [HTML_HEADER]
    parts.append(f'<a class="back" href="/">← Back to heatmap</a>\n')
    parts.append(f'<h2>Class: {class_name} &nbsp;({len(entries)} images)</h2>\n')

    if not entries:
        parts.append('<p class="empty">No labeled images for this class.</p>\n')
    else:
        parts.append('<div class="thumb-grid">\n')
        for stem, _boxes in entries:
            img_url = f"/thumb/{class_id}/{urllib.parse.quote(stem)}"
            parts.append(
                f'<div class="thumb-card">'
                f'<img src="{img_url}" loading="lazy" width="300">'
                f'<div class="thumb-label">{stem}</div>'
                f'</div>\n'
            )
        parts.append('</div>\n')

    parts.append(HTML_FOOTER)
    return "".join(parts).encode("utf-8")

# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Suppress default access log spam; print only errors
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(fmt, *args)

    def send_response_with_body(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        # ----- Heatmap (root) -----
        if path == "/":
            counts = count_per_class()
            body = heatmap_page(counts)
            self.send_response_with_body(200, "text/html; charset=utf-8", body)
            return

        # ----- Class detail: /class/<id> -----
        if path.startswith("/class/"):
            try:
                cid = int(path[len("/class/"):])
            except ValueError:
                self.send_error(400, "Bad class id")
                return
            if cid not in ID_TO_CLASS:
                self.send_error(404, "Unknown class id")
                return
            body = class_detail_page(cid)
            self.send_response_with_body(200, "text/html; charset=utf-8", body)
            return

        # ----- Annotated thumbnail: /thumb/<class_id>/<stem> -----
        if path.startswith("/thumb/"):
            rest = path[len("/thumb/"):]
            slash = rest.find("/")
            if slash < 0:
                self.send_error(400, "Bad thumb path")
                return
            try:
                cid = int(rest[:slash])
            except ValueError:
                self.send_error(400, "Bad class id")
                return
            stem = urllib.parse.unquote(rest[slash + 1:])
            jpeg = render_annotated_thumbnail(stem, cid)
            if jpeg is None:
                self.send_error(404, "Image not found or could not render")
                return
            self.send_response_with_body(200, "image/jpeg", jpeg)
            return

        # ----- Raw source image: /image/<stem> -----
        if path.startswith("/image/"):
            stem = urllib.parse.unquote(path[len("/image/"):])
            img_path = IMAGES_DIR / f"{stem}.png"
            if not img_path.exists():
                img_path = IMAGES_DIR / f"{stem}.jpg"
            if not img_path.exists():
                self.send_error(404, "Image not found")
                return
            data = img_path.read_bytes()
            ct = "image/png" if str(img_path).endswith(".png") else "image/jpeg"
            self.send_response_with_body(200, ct, data)
            return

        self.send_error(404, "Not found")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading label index from {LABELS_DIR} ...", flush=True)
    # Pre-load so first request is fast
    counts = count_per_class()
    n_classes = sum(1 for v in counts.values() if v > 0)
    total = sum(counts.values())
    print(f"  {total} labeled instances across {n_classes} / {len(ID_TO_CLASS)} classes", flush=True)

    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"\nServing at http://localhost:{PORT}/", flush=True)
    print("Press Ctrl+C to stop.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
