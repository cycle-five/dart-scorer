# Dartscorer — Target Architecture

## System Overview

```
                    ┌─────────────────────────────────────────────────┐
                    │              Camera Manager                     │
                    │                                                 │
                    │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
                    │  │ Front    │  │ Side     │  │ Top      │     │
                    │  │ Camera   │  │ Camera   │  │ Camera   │     │
                    │  │ (primary)│  │ (oblique)│  │ (future) │     │
                    │  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
                    │       │             │             │            │
                    │       ▼             ▼             ▼            │
                    │  ┌──────────────────────────────────────┐      │
                    │  │         Frame Synchronizer           │      │
                    │  │   (timestamp-aligned frame pairs)    │      │
                    │  └──────────────────┬───────────────────┘      │
                    └─────────────────────┼─────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────────┐
                    │                     ▼                          │
                    │           Trigger Fusion                       │
                    │                                                │
                    │  ┌────────────┐ ┌────────────┐ ┌──────────┐  │
                    │  │ Video Diff │ │ Side-cam   │ │ Audio    │  │
                    │  │ (front)    │ │ Protrusion │ │ Impact   │  │
                    │  └─────┬──────┘ └─────┬──────┘ └────┬─────┘  │
                    │        │              │             │         │
                    │        ▼              ▼             ▼         │
                    │  ┌──────────────────────────────────────┐     │
                    │  │     Trigger Arbiter                   │     │
                    │  │  (any 2 of 3 agree = dart arrived)   │     │
                    │  └──────────────────┬───────────────────┘     │
                    └─────────────────────┼─────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────┐
                    │              Detection Pipeline                  │
                    │                                                  │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │         Environment Profiler              │   │
                    │  │  (lighting, angle, board type, contrast)  │   │
                    │  └──────────────────┬───────────────────────┘   │
                    │                     │                            │
                    │                     ▼                            │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │           Model Zoo                       │   │
                    │  │                                           │   │
                    │  │  base_bar_warm.pt     — warm bar lighting │   │
                    │  │  base_bar_cool.pt     — cool fluorescent  │   │
                    │  │  base_outdoor.pt      — natural light     │   │
                    │  │  base_electronic.pt   — soft-tip boards   │   │
                    │  │  site_joes_bar.pt     — fine-tuned        │   │
                    │  │                                           │   │
                    │  │  select_model(profile) → best weights     │   │
                    │  └──────────────────┬───────────────────────┘   │
                    │                     │                            │
                    │                     ▼                            │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │      YOLO (63-class segment)             │   │
                    │  │      + Geometry confidence check         │   │
                    │  └──────────────────┬───────────────────────┘   │
                    │                     │                            │
                    │                     ▼                            │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │      Multi-Camera Fusion (optional)       │   │
                    │  │                                           │   │
                    │  │  Front: segment classification + tip      │   │
                    │  │  Side:  dart count + vertical position    │   │
                    │  │  Fuse:  agreement → high confidence       │   │
                    │  │         disagree → flag for review        │   │
                    │  └──────────────────┬───────────────────────┘   │
                    └─────────────────────┼─────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────┐
                    │              Game Engine                         │
                    │                                                  │
                    │  Dart ordinal (frame order)                      │
                    │  Score accumulation (301/501/cricket)            │
                    │  Checkout logic                                  │
                    │  Round management                                │
                    │  CSV logging                                     │
                    └─────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Camera Manager

Manages one or more USB cameras. Each camera has a role and configuration.

```python
class CameraRole(Enum):
    FRONT = "front"       # face-on view of board (primary, required)
    SIDE = "side"         # oblique/parallel to wall (dart protrusion)
    TOP = "top"           # overhead (future — best for tip localization)

class CameraConfig:
    role: CameraRole
    device: str           # /dev/video0, /dev/video2, etc.
    width: int
    height: int
    crop_roi: tuple       # (x, y, w, h) or None
    homography: ndarray   # 3x3 matrix or None (front camera only)
    calibrated: bool

class CameraManager:
    cameras: dict[CameraRole, CameraConfig]

    def discover() -> list[CameraConfig]
        """Auto-discover connected USB cameras."""

    def add(role, device, ...) -> CameraConfig
        """Register a camera with a role."""

    def read_synced() -> dict[CameraRole, Frame]
        """Read timestamp-aligned frames from all cameras."""
```

**Single-camera deployment** (current, MVP): Only FRONT camera. Everything works
as it does today.

**Two-camera deployment** (next): FRONT + SIDE. Side camera provides:
- Reliable dart count (protrusions from board surface)
- Dart arrival trigger (new protrusion appears)
- Vertical position constraint (narrows YOLO's search space)
- Bounce-out detection (protrusion appears then disappears)

**Three-camera** (future): FRONT + SIDE + TOP for maximum accuracy.

### 2. Side Camera Signal

The side camera sees darts as linear features protruding from the board plane.
Detection is classical CV — no YOLO needed.

```python
class SideCameraDetector:
    """Detect darts from oblique/side camera view."""

    def process_frame(frame) -> SideDetection:
        """
        Returns:
            dart_count: int — number of protrusions visible
            protrusions: list of {
                y_position: float    — vertical position on board (pixels)
                depth: float         — how far dart protrudes (pixels)
                angle: float         — shaft angle relative to board surface
                confidence: float
            }
            is_stable: bool — scene hasn't changed recently
        """
        # Background subtraction on edge-detected frame
        # Find horizontal line segments protruding from board edge
        # Each segment = one dart
        # y_position = where segment meets board edge
        # depth = length of segment

    def detect_arrival() -> bool:
        """New protrusion appeared since last stable frame."""

    def detect_removal() -> bool:
        """Protrusions disappeared (darts pulled)."""

    def detect_bounce() -> bool:
        """Brief protrusion that disappeared (bounce-out)."""
```

**Key insight**: The side camera's signal is complementary to the front camera.
Front gives (x, y) on board face. Side gives reliable count + y + depth.
Neither is sufficient alone, but together they're very robust.

### 3. Trigger Fusion

Multiple trigger sources vote on "did a dart just arrive?"

```python
class TriggerArbiter:
    """Fuse multiple trigger signals to reduce false positives."""

    sources: dict[str, TriggerSource]  # "video", "side_cam", "audio"

    def update(signals: dict[str, bool]) -> TriggerResult:
        """
        Policy options:
          ANY    — any source fires (most sensitive, current behavior)
          AGREE  — 2+ sources agree within time window (fewer false triggers)
          PRIMARY_CONFIRM — primary fires, secondary confirms within N ms
        """

    # For single-camera: degrades to current behavior (video OR audio)
    # For two-camera: side_cam arrival + front_cam video diff = high confidence
```

### 4. Environment Profiler

Automatically characterizes the deployment environment from a few frames.

```python
class EnvironmentProfile:
    # Lighting
    color_temperature: float    # Kelvin estimate (warm=2700, cool=6500)
    brightness: float           # mean intensity 0-255
    contrast: float             # std of intensity
    light_uniformity: float     # how even across the board

    # Board
    board_type: str             # "bristle", "electronic", "unknown"
    board_wear: float           # 0=new, 1=heavily worn
    board_colors: dict          # dominant color clusters

    # Geometry
    camera_angle: float         # degrees from perpendicular
    board_coverage: float       # fraction of frame occupied by board
    resolution_at_board: float  # px/mm at board surface

    # Darts
    dart_contrast: float        # how visible darts are against board

    @classmethod
    def from_frames(cls, frames: list[ndarray], homography=None):
        """Profile environment from a few captured frames."""
        # Color temperature from gray-world assumption or Planckian locus
        # Board type from color histogram (electronic = bright plastic colors)
        # Camera angle from homography decomposition
        # Dart contrast from foreground/background separation

    def similarity(self, other: 'EnvironmentProfile') -> float:
        """How similar two environments are (0-1)."""

    def best_cluster(self, profiles: list) -> str:
        """Which environment cluster this profile belongs to."""
```

### 5. Model Zoo

Manages multiple base models and site-specific fine-tuned models.

```python
class ModelEntry:
    name: str                     # "base_bar_warm", "site_joes_bar"
    weights_path: Path
    environment: EnvironmentProfile  # what it was trained on
    metrics: dict                 # mAP50, precision, recall
    created: datetime
    parent: str                   # which base model it was fine-tuned from
    training_frames: int          # how much data went into it
    is_base: bool                 # base model vs site-specific

class ModelZoo:
    models_dir: Path              # data/models/
    registry: dict[str, ModelEntry]

    def select_model(profile: EnvironmentProfile) -> ModelEntry:
        """Pick the best model for a given environment.

        Strategy:
          1. If a site-specific model exists and env hasn't changed → use it
          2. Find base model whose training environment is most similar
          3. Fall back to the most general base model
        """

    def register(name, weights, profile, metrics, parent=None):
        """Add a model to the zoo."""

    def fine_tune(base: ModelEntry, site_data: Path,
                  profile: EnvironmentProfile) -> ModelEntry:
        """Fine-tune a base model for a specific site.

        The standard deployment workflow:
          1. Install camera, run calibration
          2. Auto-profile environment
          3. Select best base model
          4. User throws 30-50 rounds (~100-150 frames)
          5. Fine-tune → site-specific model
          6. System is ready
        """

    def list_models() -> list[ModelEntry]:
        """List all available models with metrics."""
```

**Storage layout:**
```
data/models/
├── registry.json           # model metadata index
├── base/
│   ├── bar_warm/
│   │   ├── best.pt
│   │   ├── profile.json    # environment it was trained on
│   │   └── metrics.json
│   ├── bar_cool/
│   └── general/            # trained on diverse data
├── sites/
│   ├── joes_bar/
│   │   ├── best.pt
│   │   ├── profile.json
│   │   ├── metrics.json
│   │   └── calibration/    # homography, crop, lens
│   └── league_hall/
```

### 6. Multi-Camera Fusion

When multiple cameras provide detection results, fuse them.

```python
class FusedDetection:
    segment: str              # final classification
    score: int
    confidence: float         # fused confidence
    sources: dict             # which cameras contributed what

    front_segment: str        # what front camera YOLO said
    front_confidence: float
    side_y_position: float    # vertical position from side camera
    side_dart_count: int      # how many darts side camera sees
    geo_segment: str          # what geometry says
    geo_confidence: float

    agreement: str            # "all_agree", "front_side_agree", "conflict"

class DetectionFusion:
    def fuse(front: list[Detection],
             side: SideDetection,
             geometry: list[Classification]) -> list[FusedDetection]:
        """
        Fusion strategy:
          1. Start with front camera YOLO classification (primary)
          2. Validate dart count against side camera
          3. Validate segment against geometry
          4. If all agree → high confidence
          5. If front + side agree but geo disagrees → trust cameras
             (geo is weak near wires)
          6. If front disagrees with side count → investigate
             (possible occlusion, false detection, or bounce-out)
        """
```

---

## Deployment Workflow

### First-time site setup
```
1. Mount camera(s)
2. Run calibration wizard:
   - Auto-detect cameras, assign roles
   - Set crop region
   - Board homography (21-point click)
   - Auto-profile environment
3. Select base model from zoo
4. Collection phase: "Throw 30 rounds of darts"
   - Model-assisted labeling with base model
   - Human corrects mistakes
   - ~90-150 annotated frames
5. Fine-tune: ~5 min on any NVIDIA GPU, or ship frames to cloud
6. Deploy fine-tuned model
7. System operational
```

### Ongoing improvement
```
- System logs every detection with confidence
- Low-confidence detections flagged for human review
- Periodic re-training with corrected data
- Environment drift detection (lighting changed? board replaced?)
  → alert operator, suggest recalibration
```

---

## Migration Path from Current Code

### Phase 1 (current): Single camera, 63-class YOLO
- Everything working today
- collect.py, train.py, scorer.py

### Phase 2: Model zoo + environment profiler
- Add EnvironmentProfile class
- Add ModelZoo with registry
- Refactor train.py to register models
- Add --profile flag to calibrate.py
- No camera changes needed

### Phase 3: Side camera support
- Add CameraManager with role-based cameras
- Add SideCameraDetector (classical CV)
- Add TriggerArbiter for trigger fusion
- scorer.py uses fused detections
- collect.py optionally uses side camera for better triggering

### Phase 4: Multi-camera fusion + deployment wizard
- DetectionFusion for combining camera signals
- First-run setup wizard
- Cloud fine-tuning option
- Site management dashboard
