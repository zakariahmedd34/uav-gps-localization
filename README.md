# uav-gps-localization

**ICMTC 2026 — UAVC-9 Fixed-Wing Challenge**  
AI Team — Eagles, Nile University

Convert a YOLO pixel detection on a UAV video frame into a GPS coordinate (lat, lon) accurate to within 20 m — the threshold for full competition points.

---

## Problem

A fixed-wing UAV flies at 75–100 m AGL and detects flags on the ground with a downward-facing GoPro. For each flag, we must report a GPS coordinate within **20 m** of the real position and submit it on USB within 10 minutes of landing.

**Scoring:**

| GPS error | Points |
|---|---|
| < 20 m | 15 |
| 20–30 m | 12 |
| 30–60 m | 3–9 |
| > 60 m | 0 |

---

## Pipeline

```
Video Frame
    ↓
[YOLO Detection]  →  bounding box center (u, v)
    ↓
[Undistortion]    →  corrected pixel (Brown-Conrady)
    ↓
[Camera Ray]      →  normalized ray in camera frame
    ↓
[IMU Rotation]    →  ray in NED world frame  (roll, pitch, yaw)
    ↓
[Ray-Ground]      →  ground offset (ΔE, ΔN) in metres
    ↓
[GPS Conversion]  →  (Δlat, Δlon)
    ↓
[Multi-Frame Aggregation]  →  final (lat_flag, lon_flag)
```

---

## Three Approaches

| Approach | Description | Expected error |
|---|---|---|
| `src/approach1.py` | Baseline — nadir assumption, no IMU | 15–25 m |
| `src/approach2.py` | **Recommended** — full IMU rotation + undistortion | 4–8 m |
| `src/approach3.py` | Advanced — DBSCAN clustering + SAHI + GPS COG | 2–5 m |

---

## Repository Structure

```
localization_uav/
├── doc/                          # Full engineering docs
│   ├── FINAL_Pipeline_Document.md
│   ├── UAV_GPS_Localization_Engineering_Report.md
│   └── MOBILE_TEST_IMPLEMENTATION_PLAN.md
├── notebook/
│   └── UAV_GPS_Localization.ipynb   # Interactive demo + unit tests
├── pipeline/                     # Integration layer (in progress)
│   ├── calibrate_camera.py
│   ├── mobile_telemetry_parser.py
│   ├── video_sync.py
│   ├── detector.py
│   └── run_localization.py
├── src/                          # Core localization approaches
│   ├── approach1.py
│   ├── approach2.py
│   └── approach3.py
├── tests/                        # Unit + integration tests
├── configs/                      # Camera parameter YAMLs
└── data/                         # Flight recordings (gitignored)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Calibrate camera

```bash
python pipeline/calibrate_camera.py \
    --images data/checkerboard/ \
    --out configs/camera_params_phone.yaml
```

### 3. Run localization on a recording

```bash
python pipeline/run_localization.py \
    --video      data/test_flight/recording.mp4 \
    --telemetry  data/test_flight/sensors.csv \
    --camera     configs/camera_params_phone.yaml \
    --start-ms   1716640000000 \
    --out        results.json
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Mobile Phone Testing

The GoPro can be substituted with any smartphone for ground testing:

1. Install **Sensor Logger** (Android) or **SensorLog** (iOS)
2. Disable all video stabilization (EIS, optical stabilization)
3. Place a red/orange 40×40 cm target on the ground
4. Hold phone at ~1.5 m height pointing straight down, walk slowly over target
5. Export video + CSV — run the pipeline

See [`doc/MOBILE_TEST_IMPLEMENTATION_PLAN.md`](doc/MOBILE_TEST_IMPLEMENTATION_PLAN.md) for full instructions.

---

## Team

| Member | Role |
|---|---|
| A | Camera calibration + telemetry parser |
| B | Video sync + end-to-end runner |
| C | Flag detection + validation |

---

## References

- Barber & Redding (2006) — UAV camera attitude rotation convention
- OpenCV Brown-Conrady distortion model
- ICMTC 2026 Official Rules, Section 6.2.1
