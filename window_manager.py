"""
window_manager.py — Sticky window sizes for OpenCV windows.

Saves and restores window positions/sizes to a JSON cache file
so windows remember their layout between sessions.
"""

import json
from pathlib import Path

import cv2

CACHE_PATH = Path(__file__).parent / "data" / ".window_cache.json"


def _load_cache():
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def create_window(name, default_width=800, default_height=600):
    """Create a named OpenCV window with saved size, or default."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cache = _load_cache()
    if name in cache:
        w = cache[name].get("width", default_width)
        h = cache[name].get("height", default_height)
    else:
        w, h = default_width, default_height
    cv2.resizeWindow(name, w, h)


def save_window_sizes(names):
    """Save current sizes for the given window names."""
    cache = _load_cache()
    for name in names:
        try:
            rect = cv2.getWindowImageRect(name)
            if rect is not None and rect[2] > 0 and rect[3] > 0:
                cache[name] = {
                    "width": rect[2],
                    "height": rect[3],
                }
        except cv2.error:
            pass
    _save_cache(cache)
