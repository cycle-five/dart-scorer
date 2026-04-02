# Dartscorer V2 — Implementation Plan

## Problem Statement

V1 uses 189 YOLO classes (3 ordinals × 63 segments) with tiny fixed-size bounding boxes (22×28px). This causes:
- **Ordinal confusion**: The model can't distinguish d1_S5 from d2_S5 visually — ordinal is contextual, not visual
- **Stuck classes**: 10+ well-represented classes have 0.0 AP despite 20-159 training samples
- **Wasted model capacity**: 189 classes where 126 share identical visual features (only differ by ordinal)
- **Tiny feature patches**: Point annotations give the model almost nothing to work with

## V2 Architecture

Decompose the monolithic 189-class YOLO into cooperating stages:

```
Camera Frame
    │
    ▼
┌──────────────────────────┐
│  YOLO: 1-class "dart"    │  ← Find darts, full-dart bounding boxes
│  (detection only)        │
└──────────┬───────────────┘
           │ For each detection:
           ▼
┌──────────────────────────┐
│  Tip Extraction          │  ← Estimate tip point from bbox (closest to center)
└──────────┬───────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌──────────┐  ┌──────────────────┐
│ Geometry │  │ Ring Classifier   │
│ Pipeline │  │ (CNN on dart crop)│
└────┬─────┘  └────────┬─────────┘
     │                 │
     ▼                 ▼
  sector (θ)      ring (S/D/T/BULL/MISS)
     │                 │
     └────────┬────────┘
              ▼
     ┌────────────────┐
     │ Combine + Score │  ← sector + ring → segment → score
     └────────────────┘
```

**Ordinal**: Assigned by detection order within the frame batch (1st new dart = d1, etc.), NOT predicted by the model.

---

## Key Design Decisions

### 1. Single-Class Detection
All training data pools into one class ("dart"). With 2299 frames and ~3884 annotations, that's ~3884 positive examples for ONE class instead of ~20 per class average.

Benefits:
- Robust detection — the model only needs to answer "is there a dart here?"
- All existing tip-coordinate data is directly reusable
- More data per class = better generalization

### 2. Full-Dart Bounding Boxes
Replace the fixed 30×30px point annotation with a box that covers the visible dart (tip + shaft + flights where visible).

Benefits:
- Dart angle, color, shaft length provide rich features
- Larger box = more context for the model
- Dart angle correlates with board position (darts roughly point toward center)

Annotation approach: Click two points — (1) dart tip, (2) end of visible shaft/flight. Generate axis-aligned bounding box from those two points with padding. Store tip point separately for geometry pipeline.

### 3. Geometry-First Classification
The existing homography + `board.py` already maps pixel coordinates to (radius, angle):
- **Angle → Sector**: `get_sector(theta)` already works. Near-wire tolerance: accept ±1 adjacent sector.
- **Radius → Ring**: `get_ring(r)` maps radius to ring type. Ring boundaries are well-defined (WDF standard, in `config.py`).

Geometry handles ~85-90% of classification confidently (darts that land clearly within a segment). Only ambiguous cases (near wires) need ML assistance.

### 4. Ring Classifier (Second Stage)
A small CNN classifier on the cropped dart image resolves ring ambiguity near boundaries.

Input features:
- Dart crop (from YOLO bbox), resized to fixed square (e.g., 64×64)
- **Radius from center** (normalized, from homography) — the key new feature
- **Angle** (could be useful for bulls near certain sectors)

Output: 5 classes — Single, Double, Triple, Bull, Miss
(Or 6 if we split S_BULL / D_BULL)

This classifier only needs to resolve near-boundary cases. Much simpler than 63 or 189 classes.

### 5. Geometry as Hard Constraint
At inference time, NEVER allow a classification that contradicts geometry by more than 1 adjacent segment. This prevents catastrophic misclassification (e.g., calling a T20 a T1).

```python
# Pseudocode for final classification
sector_candidates = geometry.get_sector_candidates(theta)  # 1-3 sectors
ring_candidates = geometry.get_ring_candidates(radius)       # 1-2 rings
model_ring = ring_classifier.predict(crop, radius, theta)

# Ring: prefer model if geometry is ambiguous, else trust geometry
if len(ring_candidates) == 1:
    ring = ring_candidates[0]  # geometry is confident
else:
    ring = model_ring  # model resolves ambiguity

# Sector: always trust geometry (angle is highly reliable)
sector = sector_candidates[0]  # primary candidate

segment = f"{ring}{sector}"  # e.g., "T20"
```

---

## Implementation Phases

### Phase 0: Data Audit & Salvage Assessment
**Goal**: Determine how much v1 data can be reused.

**Tasks**:
1. Verify tip coordinates are accurate using the label analyzer tool
2. For salvageable data: strip ordinal from class labels (d1_T20 → T20)
3. Assess whether auto-expanding point annotations to full-dart boxes is viable:
   - Use dart angle heuristic (tip points roughly toward board center)
   - Extend box 80-100px away from center along radial line
   - Validate a sample manually — if >80% look reasonable, bulk-convert
4. If auto-expansion fails, data still useful as "tip location ground truth" for geometry validation

**Deliverables**: Decision on data reuse, converted dataset if viable.

### Phase 1: Collection Pipeline V2
**Goal**: New collection workflow that captures full-dart bounding boxes.

**Changes to `collect.py`**:
1. **Two-click annotation**: First click = dart tip, second click = end of visible shaft
   - Generate axis-aligned bounding box with padding
   - Store tip coordinates separately in JSONL metadata
   - YOLO label uses the full bbox, not the point annotation
2. **Geometry auto-classification**: After clicking tip, compute (r, θ) from homography
   - Auto-assign sector + ring → segment name
   - Show segment guess on screen for user confirmation
   - This already partially exists (homography-assisted guessing) — extend it
3. **No ordinal in class label**: Label is just the segment (e.g., "T20", "S5", "MISS")
   - 63 classes total
   - Ordinal tracked in metadata only (frame index within batch)
4. **Background frame collection**: Continue saving empty-board frames (useful for detection training as negatives)

**New `classes_v2.py`**:
```python
SEGMENTS = [f"{r}{s}" for s in range(1, 21) for r in ["S", "D", "T"]]
SEGMENTS += ["S_BULL", "D_BULL", "MISS"]
# 63 classes, no ordinal
CLASS_NAMES = SEGMENTS
```

**UI flow changes**:
- ANNOTATING state: click tip → click shaft end → auto-classify → confirm/override → next dart
- Show geometry guess prominently (sector + ring + confidence level)
- Color-code confidence: green (far from wires), yellow (near wire), red (very ambiguous)

### Phase 2: YOLO Detection Model (1-Class)
**Goal**: Train robust dart detection with full-dart bounding boxes.

**Tasks**:
1. Generate `dataset_v2.yaml` with 1 class: "dart"
2. Train YOLOv8n (or v11n) with standard settings
   - `imgsz=640`, `batch=16`, `epochs=100`
   - Expect much better detection since all data is one class
3. Evaluate: focus on detection metrics (precision, recall, mAP) — NOT classification
4. The model's job: find all darts in the frame, provide good bounding boxes

**Expected improvement**: With ~3884 annotations for 1 class (vs ~20 avg for 189), detection recall should jump significantly.

### Phase 3: Geometry Pipeline Enhancement
**Goal**: Make geometry classification production-ready.

**Tasks**:
1. **Refine `board.py`** functions for classification:
   - `classify_dart(pixel_x, pixel_y, H)` → `(sector, ring, confidence)`
   - Confidence based on distance from nearest wire (angular and radial)
   - Return `sector_candidates` when near angular wire
   - Return `ring_candidates` when near radial boundary
2. **Sector adjacency map**: Pre-compute for the dartboard layout
   ```python
   SECTOR_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
   ADJACENT = {20: [5, 1], 1: [20, 18], 18: [1, 4], ...}
   ```
3. **Ring boundary margins**: Define "ambiguous zone" width for each ring boundary
   - Inner/outer bull boundary: ±2mm
   - Triple inner/outer: ±3mm
   - Double inner/outer: ±3mm
   - Outside board (MISS): anything beyond double outer + margin
4. **Validate geometry accuracy**: Run geometry classification on all existing tip coordinates, compare to human labels. Measure % agreement.

### Phase 4: Ring Classifier (Optional, If Needed)
**Goal**: ML classifier to resolve ring ambiguity near boundaries.

**Only build this if** Phase 3 geometry alone isn't accurate enough (test with labeled data first).

**Design**:
- Input: 64×64 crop centered on dart tip, normalized radius, normalized angle
- Architecture: Small CNN (MobileNetV3-small or even a few conv layers) + MLP head for radius/angle
- Output: 6 classes (single, double, triple, s_bull, d_bull, miss)
- Training data: existing labeled darts with ring labels (extract from segment names)
- Loss: Cross-entropy, possibly weighted by inverse frequency (triples/doubles are rare)

**Feature engineering**:
```python
# For each detected dart:
tip_pixel = yolo_bbox_tip(detection)
board_x, board_y = apply_homography(tip_pixel, H)
r, theta = pixel_to_polar(board_x, board_y)

# Normalized features
r_norm = r / DOUBLE_OUTER_RADIUS  # 0=center, 1=board edge
theta_norm = theta / 360.0
wire_dist = min_distance_to_nearest_ring_boundary(r)

# If wire_dist < AMBIGUITY_THRESHOLD:
#     Use ring classifier
# Else:
#     Trust geometry
```

### Phase 5: Scoring Pipeline V2
**Goal**: Updated live scoring that uses the new detection + geometry pipeline.

**Updated `scorer.py`**:
```python
# Detection
detections = yolo_model.predict(frame)  # 1-class: just finds darts

for det in detections:
    # Tip extraction (closest bbox point to board center)
    tip = extract_tip(det.bbox, board_center)

    # Geometry classification
    r, theta = pixel_to_polar(*apply_homography(tip, H))
    sector, ring, confidence = classify_dart(r, theta)

    # Optional: ring classifier for ambiguous cases
    if confidence < CONFIDENCE_THRESHOLD:
        crop = extract_crop(frame, det.bbox)
        ring = ring_classifier.predict(crop, r, theta)

    segment = make_segment_name(ring, sector)
    score = compute_score(segment)
```

**Ordinal tracking**:
- Compare detections to previous frame's confirmed darts
- New dart = detection not matched to any existing (by position proximity)
- Assign ordinal by order of appearance, not model prediction

### Phase 6: Validation & Iteration
**Goal**: Measure end-to-end accuracy, iterate.

**Tests**:
1. **Detection recall**: % of darts correctly localized (any class)
2. **Geometry sector accuracy**: % of darts with correct sector from geometry alone
3. **Ring accuracy**: % correct ring from geometry alone vs geometry + classifier
4. **End-to-end accuracy**: % of darts with fully correct segment classification
5. **Comparison to v1**: Same test frames, v1 (189-class) vs v2 (detect + geometry)

**Iteration targets**:
- Detection recall > 95%
- Sector accuracy > 98% (geometry is very reliable for angle)
- Ring accuracy > 90% (hardest part — near boundaries)
- End-to-end > 88%

---

## Migration Checklist

### Files to Create/Modify
| File | Action | Description |
|------|--------|-------------|
| `classes_v2.py` | Create | 63 segment classes, no ordinal |
| `collect.py` | Modify | Two-click bbox, geometry auto-classify, no ordinal |
| `board.py` | Modify | Add `classify_dart()`, confidence, adjacency |
| `config.py` | Modify | Add ring boundary margins, ambiguity thresholds |
| `train.py` | Modify | Support 1-class and 63-class modes |
| `scorer.py` | Modify | Detection + geometry pipeline |
| `yolo_detector.py` | Modify | 1-class detection, no segment parsing |
| `ring_classifier.py` | Create | Optional ring disambiguation model |
| `generate_dataset_yaml.py` | Modify | Support v2 class schemes |
| `convert_v1_data.py` | Create | Tool to convert v1 labels → v2 format |
| `geometry_accuracy.py` | Create | Validate geometry classification vs labels |

### Files Unchanged
| File | Reason |
|------|--------|
| `calibrate.py` | Homography calibration stays the same |
| `audio_trigger.py` | Trigger system is independent |
| `window_manager.py` | UI utilities unchanged |
| `label_analyzer.py` | Update to support v2 classes |

### Data
- Existing v1 tip coordinates: **reusable** (strip ordinal, tip positions are accurate)
- Existing v1 bounding boxes: **need expansion** (auto-expand from tip or re-collect)
- Homography calibration: **reusable as-is**
- New data collection: Recommended to build up v2 dataset from scratch alongside salvaged v1 data

---

## Resolved Decisions

1. **Bounding box annotation UX**: Auto-detect full-dart bbox from frame differencing or edge detection. User only intervenes to redraw if the auto-detected box is wrong. Single-click tip + auto-expand, not two clicks.

2. **V1 data salvage**: Yes — convert existing v1 data. Strip ordinal from class labels, auto-expand point annotations to full-dart boxes using radial direction heuristic (darts point roughly toward board center).

3. **MISS handling**: Pure geometry. Radius > DOUBLE_OUTER_RADIUS → MISS. No ML needed for this.

4. **Camera resolution**: **1024x576 MJPG for everything** (collection, training, and inference). The eMeet C950 upscales to 1080p — no real detail gain. 1024x576 is near 1:1 with YOLO's 640px input, cutting decode work ~3.5x. Crop ROI and homography will need recalibration at new resolution.

5. **Tip extraction from bbox**: Estimate tip as the bbox edge closest to board center (darts angle inward).

---

## Camera Resolution Change

### Why 1024x576
| | 1920x1080 (v1) | 1024x576 (v2) |
|---|---|---|
| MJPG decode | 2.07 MP | 0.59 MP (3.5x less) |
| After crop | ~1371x1080 | ~731x576 |
| YOLO downscale | 2.1x (wasted) | 1.1x (near 1:1) |
| Annotation precision | 4.0 px/mm | 2.1 px/mm |

The 1080p "precision" is illusory — the sensor upscales. At 1024x576 the crop feeds YOLO almost directly with minimal resize, reducing latency and decode overhead.

### Migration impact
- `config.py`: Change `CAMERA_WIDTH=1024, CAMERA_HEIGHT=576`
- `calibrate.py --crop`: Re-run to set crop ROI at new resolution
- `calibrate.py --board`: Re-run homography at new resolution
- Existing v1 frames (1080p): Downscale during conversion, normalize coordinates stay valid

---

## Recommended Implementation Order

```
Phase 0 (data salvage) → Phase 1 (collection v2) → Phase 2 (YOLO 1-class)
    → Phase 3 (geometry pipeline) → Phase 5 (scorer v2) → Phase 6 (validation)
                                        ↓
                                    Phase 4 (ring classifier — only if geometry isn't enough)
```

Phase 0 first: salvage v1 data so we have a training set immediately for the 1-class detector.
