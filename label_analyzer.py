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
from classes_v2 import ID_TO_CLASS, CLASS_TO_ID, SEGMENTS, NUM_CLASSES, parse_class_name

PORT = 8765
import config as _config
DATA_DIR = _config.DATASET_DIR
IMAGES_DIR = _config.DATASET_IMAGES_DIR
LABELS_DIR = _config.DATASET_LABELS_DIR

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_label_index():
    """Return dict: class_id -> list of (stem, boxes) where boxes is list of
    (class_id, cx, cy, w, h) for all annotations in that image."""
    index = {cid: [] for cid in range(NUM_CLASSES)}
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

def render_annotated_image(stem, target_class_id, max_width=None):
    """Load image, draw all bboxes, highlight target class. Return JPEG bytes.

    If max_width is None, return full-resolution image.
    """
    img_path = IMAGES_DIR / f"{stem}.png"
    if not img_path.exists():
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
            info = parse_class_name(label)
            display = f"{label} ({info['label']})"
            (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(y1 - 4, th + 2)
            cv2.rectangle(overlay, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), (0, 200, 0), -1)
            cv2.putText(overlay, display, (x1 + 1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    img = overlay

    # Resize if max_width specified
    if max_width and w > max_width:
        scale = max_width / w
        new_h = int(h * scale)
        img = cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)

    quality = 85 if max_width else 92
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
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
COLOR_DARK_GREEN = "#1e8449"

def count_color(n):
    if n == 0:    return COLOR_RED
    if n <= 5:    return COLOR_ORANGE
    if n <= 20:   return COLOR_YELLOW
    if n <= 50:   return COLOR_GREEN
    return COLOR_DARK_GREEN

def text_color(n):
    if n <= 20:   return "#000"
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
  table.heatmap { border-collapse: collapse; font-size: 0.8rem; }
  table.heatmap th { padding: 6px 10px; background: #2a2a4a; color: #ccc; font-weight: normal; }
  table.heatmap td { padding: 0; }
  table.heatmap td a { display: block; width: 64px; height: 36px; line-height: 36px;
                        text-align: center; font-weight: bold; border: 1px solid #1a1a2e; }
  table.heatmap td a:hover { opacity: 0.8; filter: brightness(1.2); text-decoration: none; }
  .ring-label { padding: 6px 10px; background: #2a2a4a; color: #ccc;
                text-align: left; white-space: nowrap; }
  .thumb-grid { display: flex; flex-wrap: wrap; gap: 12px; }
  .thumb-card { background: #2a2a4a; border-radius: 6px; overflow: hidden;
                border: 1px solid #444; cursor: pointer; }
  .thumb-card:hover { border-color: #a0c4ff; }
  .thumb-card img { display: block; width: 300px; height: auto; }
  .thumb-label { padding: 4px 8px; font-size: 0.7rem; color: #aaa;
                 white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                 max-width: 300px; }
  .empty { color: #888; font-style: italic; padding: 20px 0; }

  /* Lightbox for full-size image zoom */
  .lightbox { display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
              background: rgba(0,0,0,0.9); z-index: 1000; justify-content: center;
              align-items: center; cursor: zoom-out; }
  .lightbox.active { display: flex; }
  .lightbox img { max-width: 95vw; max-height: 95vh; object-fit: contain; }
  .lightbox-caption { position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
                      color: #aaa; font-size: 0.85rem; background: rgba(0,0,0,0.7);
                      padding: 6px 16px; border-radius: 4px; }
</style>
</head>
<body>
"""

HTML_FOOTER = "</body></html>\n"

LIGHTBOX_JS = """
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <img id="lightbox-img" src="">
  <div class="lightbox-caption" id="lightbox-caption"></div>
</div>
<script>
function openLightbox(thumbUrl, stem, classId) {
  // Request full-size image
  var fullUrl = '/image/' + classId + '/' + encodeURIComponent(stem);
  document.getElementById('lightbox-img').src = fullUrl;
  document.getElementById('lightbox-caption').textContent = stem;
  document.getElementById('lightbox').classList.add('active');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('active');
  document.getElementById('lightbox-img').src = '';
}
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeLightbox();
});
</script>
"""


def heatmap_page(counts):
    total_images = len(list(LABELS_DIR.glob("*.txt")))
    total_labeled = sum(counts.values())
    covered = sum(1 for v in counts.values() if v > 0)

    parts = [HTML_HEADER]
    parts.append(f"<h1>Dart Label Analyzer — {NUM_CLASSES} classes</h1>\n")
    parts.append(f'<div class="stats">'
                 f'{total_images} images &nbsp;|&nbsp; '
                 f'{total_labeled} annotations &nbsp;|&nbsp; '
                 f'{covered}/{NUM_CLASSES} classes covered'
                 f'</div>\n')
    parts.append('<div class="legend">\n')
    for color, label in [
        (COLOR_RED,        "0"),
        (COLOR_ORANGE,     "1–5"),
        (COLOR_YELLOW,     "6–20"),
        (COLOR_GREEN,      "21–50"),
        (COLOR_DARK_GREEN, "50+"),
    ]:
        parts.append(f'<div class="legend-item">'
                     f'<div class="legend-swatch" style="background:{color}"></div>'
                     f'{label}</div>\n')
    parts.append('</div>\n')

    # Build heatmap: rows = sectors (1-20, then bulls), columns = rings (S, D, T)
    RINGS = ["S", "D", "T"]
    SECTORS = list(range(1, 21))

    parts.append('<table class="heatmap">\n')
    parts.append('<thead><tr><th>Sector</th>')
    for ring in RINGS:
        parts.append(f'<th>{ring}</th>')
    parts.append('</tr></thead>\n<tbody>\n')

    for sector in SECTORS:
        parts.append('<tr>')
        parts.append(f'<td class="ring-label">{sector}</td>')
        for ring in RINGS:
            seg = f"{ring}{sector}"
            cid = CLASS_TO_ID.get(seg)
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

    # Bulls row
    parts.append('<tr>')
    parts.append(f'<td class="ring-label">Bull</td>')
    for seg in ["S_BULL", "D_BULL"]:
        cid = CLASS_TO_ID.get(seg)
        n = counts.get(cid, 0) if cid is not None else 0
        bg = count_color(n)
        fg = text_color(n)
        if cid is not None and n > 0:
            cell = f'<a href="/class/{cid}" style="background:{bg};color:{fg}">{n}</a>'
        else:
            cell = f'<a href="#" style="background:{bg};color:{fg};cursor:default">{n}</a>'
        parts.append(f'<td>{cell}</td>')
    parts.append('<td></td>')  # empty cell for T column
    parts.append('</tr>\n')

    parts.append('</tbody></table>\n')
    parts.append(HTML_FOOTER)
    return "".join(parts).encode("utf-8")


def class_detail_page(class_id):
    class_name = ID_TO_CLASS.get(class_id, f"class_{class_id}")
    info = parse_class_name(class_name)
    entries = get_index().get(class_id, [])

    parts = [HTML_HEADER]
    parts.append(f'<a class="back" href="/">← Back to heatmap</a>\n')
    parts.append(f'<h2>{class_name} — {info["label"]} &nbsp;({len(entries)} images)</h2>\n')

    if not entries:
        parts.append('<p class="empty">No labeled images for this class.</p>\n')
    else:
        parts.append('<div class="thumb-grid">\n')
        for stem, _boxes in entries:
            thumb_url = f"/thumb/{class_id}/{urllib.parse.quote(stem)}"
            parts.append(
                f'<div class="thumb-card" onclick="openLightbox(\'{thumb_url}\', '
                f'\'{stem}\', {class_id})">'
                f'<img src="{thumb_url}" loading="lazy" width="300">'
                f'<div class="thumb-label">{stem}</div>'
                f'</div>\n'
            )
        parts.append('</div>\n')

    parts.append(LIGHTBOX_JS)
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
        path = self.path.split("?")[0]  # strip query string

        # ----- Root: heatmap -----
        if path == "/":
            counts = count_per_class()
            body = heatmap_page(counts)
            self.send_response_with_body(200, "text/html; charset=utf-8", body)
            return

        # ----- Class detail: /class/<id> -----
        if path.startswith("/class/"):
            try:
                class_id = int(path[len("/class/"):])
            except ValueError:
                self.send_error(400, "Invalid class ID")
                return
            body = class_detail_page(class_id)
            self.send_response_with_body(200, "text/html; charset=utf-8", body)
            return

        # ----- Thumbnail: /thumb/<class_id>/<stem> -----
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
            # Sanitize: reject path traversal attempts
            if ".." in stem or "/" in stem or "\\" in stem or "\x00" in stem:
                self.send_error(400, "Invalid filename")
                return
            jpeg = render_annotated_image(stem, cid, max_width=300)
            if jpeg is None:
                self.send_error(404, "Image not found or could not render")
                return
            self.send_response_with_body(200, "image/jpeg", jpeg)
            return

        # ----- Full-size image: /image/<class_id>/<stem> -----
        if path.startswith("/image/"):
            rest = path[len("/image/"):]
            slash = rest.find("/")
            if slash < 0:
                self.send_error(400, "Bad image path")
                return
            try:
                cid = int(rest[:slash])
            except ValueError:
                self.send_error(400, "Bad class id")
                return
            stem = urllib.parse.unquote(rest[slash + 1:])
            # Sanitize: reject path traversal attempts
            if ".." in stem or "/" in stem or "\\" in stem or "\x00" in stem:
                self.send_error(400, "Invalid filename")
                return
            jpeg = render_annotated_image(stem, cid, max_width=None)
            if jpeg is None:
                self.send_error(404, "Image not found")
                return
            self.send_response_with_body(200, "image/jpeg", jpeg)
            return

        self.send_error(404, "Not found")


def main():
    print(f"Loading label index from {LABELS_DIR}...")
    get_index()
    counts = count_per_class()
    total = sum(counts.values())
    covered = sum(1 for v in counts.values() if v > 0)
    print(f"  {total} annotations across {covered}/{NUM_CLASSES} classes")

    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"\nLabel Analyzer running at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
