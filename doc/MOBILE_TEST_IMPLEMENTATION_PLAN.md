# ICMTC 2026 — UAVC-9 Fixed Wing Challenge
# Mobile Phone Pipeline: Implementation & Test Plan
### AI Team — Eagles, Nile University

**Version:** 1.0  
**Date:** 2026-05-25  
**Purpose:** Full pipeline integration and validation using a mobile phone as a GoPro substitute while the actual camera is unavailable  
**Target:** Replace GoPro-specific components with phone-equivalent components; keep all geometry code (Approach 2) untouched

---

## Overview

The three localization approaches (`src/approach1.py`, `src/approach2.py`, `src/approach3.py`) and the full notebook are mathematically complete. What is missing is the **integration layer** — the code that connects real sensor data to those approaches and validates the output against a known ground truth.

Since the GoPro is unavailable, a mobile phone covers every missing component:

| GoPro + ArduPilot | Mobile substitute | Same math? |
|---|---|---|
| GoPro `.mp4` video | Phone `.mp4` (EIS disabled) | ✅ Yes |
| GoPro K matrix + distortion | Calibrated phone K matrix | ✅ Yes |
| ArduPilot `.bin` / `.tlog` | Sensor Logger app CSV | ✅ Yes |
| ArduPilot GPS | Phone GPS (same app) | ✅ Yes |
| Time-sync via boot event | App timestamps both streams | ✅ Yes |

The only swap needed at competition time: drop in `camera_params_gopro.yaml` instead of `camera_params_phone.yaml`. Everything else is reused exactly.

---

## Repository Layout After This Sprint

```
localization_uav/
├── doc/
│   ├── FINAL_Pipeline_Document.md
│   ├── UAV_GPS_Localization_Engineering_Report.md
│   └── MOBILE_TEST_IMPLEMENTATION_PLAN.md       ← this file
├── notebook/
│   └── UAV_GPS_Localization.ipynb
├── pipeline/                                     ← NEW (this sprint)
│   ├── __init__.py
│   ├── calibrate_camera.py                       ← Member A
│   ├── mobile_telemetry_parser.py                ← Member A
│   ├── video_sync.py                             ← Member B
│   ├── detector.py                               ← Member C
│   └── run_localization.py                       ← Member B
├── src/
│   ├── approach1.py
│   ├── approach2.py
│   └── approach3.py
├── tests/                                        ← NEW (this sprint)
│   ├── test_calibration.py                       ← Member A
│   ├── test_telemetry_parser.py                  ← Member A
│   ├── test_video_sync.py                        ← Member B
│   ├── test_detector.py                          ← Member C
│   ├── test_run_localization.py                  ← Member B + C
│   └── test_end_to_end.py                        ← all three (integration)
├── data/
│   ├── checkerboard/                             ← Member A fills this
│   ├── test_flight/                              ← Member B fills this
│   └── ground_truth/                             ← Member C fills this
└── configs/
    ├── camera_params_phone.yaml                  ← Member A produces this
    └── camera_params_gopro.yaml                  ← placeholder for later
```

---

## Member Distribution

| Member | Role | Files owned | Deadline |
|---|---|---|---|
| **A** | Camera & Telemetry | `calibrate_camera.py`, `mobile_telemetry_parser.py`, `test_calibration.py`, `test_telemetry_parser.py`, `camera_params_phone.yaml` | Week 1 |
| **B** | Video Sync & Runner | `video_sync.py`, `run_localization.py`, `test_video_sync.py`, `test_run_localization.py` | Week 2 |
| **C** | Detection & Validation | `detector.py`, `test_detector.py`, `test_end_to_end.py`, ground-truth survey | Week 2–3 |

Dependencies: B depends on A finishing the telemetry parser. C's end-to-end test depends on A + B. A has no dependencies — start immediately.

---

---

# Member A — Camera Calibration & Telemetry Parser

**Deadline:** End of Week 1  
**No dependencies on B or C**

---

## A.1 — Phone Setup

Before writing any code:

1. Install **Sensor Logger** (Android) or **SensorLog** (iOS)
2. In phone camera settings: **disable all stabilization** (EIS, optical stabilization, horizon lock)
3. Set video resolution to the highest available (1080p or 4K)
4. In the sensor app: enable GPS, accelerometer, gyroscope, magnetometer — set rate to 50 Hz or higher
5. Verify the app exports a CSV with at minimum these columns:
   ```
   timestamp_ms, latitude, longitude, altitude_m, roll_deg, pitch_deg, yaw_deg
   ```

---

## A.2 — `pipeline/calibrate_camera.py`

**Purpose:** Compute the intrinsic matrix K and distortion coefficients `dist` for the phone camera. Output a `camera_params_phone.yaml` that the rest of the pipeline reads.

**Input:** ~30 photos of a 9×6 checkerboard at varied angles and distances.  
**Output:** `configs/camera_params_phone.yaml`

### Implementation spec

```python
"""
pipeline/calibrate_camera.py

Usage:
    python pipeline/calibrate_camera.py --images data/checkerboard/ --out configs/camera_params_phone.yaml
"""

import cv2
import numpy as np
import yaml
import glob
import argparse

CHECKERBOARD = (9, 6)      # inner corners: columns × rows
SQUARE_SIZE_MM = 25.0      # physical size of one square in millimetres


def collect_object_points(board_shape):
    """
    Build the 3-D world coordinates of checkerboard corners.
    Z = 0 for all corners (flat board).
    Returns shape (N_corners, 3) float32.
    """
    objp = np.zeros((board_shape[0] * board_shape[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_shape[0], 0:board_shape[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    return objp


def find_corners(image_paths, board_shape):
    """
    Detect checkerboard corners in each image.
    Returns (obj_points, img_points, image_size).
    Skips images where detection fails.
    """
    obj_template = collect_object_points(board_shape)
    obj_points = []
    img_points = []
    image_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])   # (width, height)

        ret, corners = cv2.findChessboardCorners(gray, board_shape, None)

        if ret:
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(obj_template)
            img_points.append(corners_sub)
            print(f"  OK  {path}")
        else:
            print(f"  SKIP (corners not found): {path}")

    return obj_points, img_points, image_size


def calibrate(obj_points, img_points, image_size):
    """
    Run cv2.calibrateCamera.
    Returns (rms, K, dist).
    """
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    return rms, K, dist


def save_yaml(path, K, dist, rms, image_size):
    data = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "rms_px": float(rms),
        "image_size": list(image_size),
        "model": "brown_conrady",
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Saved: {path}  (RMS = {rms:.3f} px)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Folder of checkerboard images")
    parser.add_argument("--out",    required=True, help="Output YAML path")
    args = parser.parse_args()

    paths = sorted(glob.glob(f"{args.images}/*.jpg") + glob.glob(f"{args.images}/*.png"))
    print(f"Found {len(paths)} images")

    obj_pts, img_pts, img_size = find_corners(paths, CHECKERBOARD)
    print(f"Usable images: {len(obj_pts)} / {len(paths)}")

    rms, K, dist = calibrate(obj_pts, img_pts, img_size)
    print(f"RMS reprojection error: {rms:.4f} px  (target < 1.0)")

    if rms > 1.5:
        print("WARNING: RMS > 1.5 px — retake photos with more angle variety")

    save_yaml(args.out, K, dist, rms, img_size)


if __name__ == "__main__":
    main()
```

### Acceptance criteria
- RMS < 1.0 px on ≥ 20 usable images
- `camera_params_phone.yaml` present in `configs/`
- Verify visually: undistort one checkerboard image with the saved params — lines must be straight

---

## A.3 — `pipeline/mobile_telemetry_parser.py`

**Purpose:** Read the CSV exported by the sensor app. Return a DataFrame indexed by time that the video sync module can interpolate.

**Input:** CSV file from Sensor Logger / SensorLog  
**Output:** `pandas.DataFrame` with columns `[timestamp_ms, lat, lon, alt_m, roll_rad, pitch_rad, yaw_rad]`

### Implementation spec

```python
"""
pipeline/mobile_telemetry_parser.py

Reads the CSV exported by Sensor Logger (Android) or SensorLog (iOS).
Both apps export slightly different column names — this parser handles both.
"""

import pandas as pd
import numpy as np


# Column name aliases — maps app-specific names → canonical names
COLUMN_ALIASES = {
    # Sensor Logger (Android)
    "time":          "timestamp_ms",
    "Time":          "timestamp_ms",
    "timestamp":     "timestamp_ms",
    "lat":           "lat",
    "latitude":      "lat",
    "lon":           "lon",
    "longitude":     "lon",
    "alt":           "alt_m",
    "altitude":      "alt_m",
    "roll":          "roll_deg",
    "pitch":         "pitch_deg",
    "yaw":           "yaw_deg",
    "azimuth":       "yaw_deg",
    "heading":       "yaw_deg",
}

REQUIRED_COLUMNS = ["timestamp_ms", "lat", "lon", "alt_m", "roll_deg", "pitch_deg", "yaw_deg"]


def load_telemetry(csv_path: str) -> pd.DataFrame:
    """
    Load and normalize a sensor-app CSV.

    Returns
    -------
    df : DataFrame with columns [timestamp_ms, lat, lon, alt_m, roll_rad, pitch_rad, yaw_rad]
         Sorted by timestamp_ms. Degrees converted to radians.
    """
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.rename(columns={c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES},
              inplace=True)

    # Validate
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns after alias mapping: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].dropna().sort_values("timestamp_ms").reset_index(drop=True)

    # Convert degrees → radians for angles
    for col_deg, col_rad in [("roll_deg", "roll_rad"),
                              ("pitch_deg", "pitch_rad"),
                              ("yaw_deg", "yaw_rad")]:
        df[col_rad] = np.deg2rad(df[col_deg])

    df.drop(columns=["roll_deg", "pitch_deg", "yaw_deg"], inplace=True)

    return df


def get_telemetry_at(df: pd.DataFrame, query_ms: float) -> dict:
    """
    Linearly interpolate the telemetry row at an arbitrary timestamp.

    Parameters
    ----------
    df       : output of load_telemetry()
    query_ms : target timestamp in milliseconds

    Returns
    -------
    dict with keys: lat, lon, alt_m, roll_rad, pitch_rad, yaw_rad
    """
    t = df["timestamp_ms"].values

    if query_ms <= t[0]:
        row = df.iloc[0]
    elif query_ms >= t[-1]:
        row = df.iloc[-1]
    else:
        idx = np.searchsorted(t, query_ms) - 1
        t0, t1 = t[idx], t[idx + 1]
        alpha = (query_ms - t0) / (t1 - t0)

        row = {}
        for col in ["lat", "lon", "alt_m", "roll_rad", "pitch_rad", "yaw_rad"]:
            v0 = df.iloc[idx][col]
            v1 = df.iloc[idx + 1][col]
            row[col] = float(v0 + alpha * (v1 - v0))

        return row

    return {c: float(row[c]) for c in ["lat", "lon", "alt_m", "roll_rad", "pitch_rad", "yaw_rad"]}
```

---

## A.4 — Tests for Member A

### `tests/test_calibration.py`

```python
"""
Tests for calibrate_camera.py — run without real images using synthetic data.
"""
import numpy as np
import pytest
from pipeline.calibrate_camera import collect_object_points, CHECKERBOARD, SQUARE_SIZE_MM


def test_object_points_shape():
    pts = collect_object_points(CHECKERBOARD)
    expected_n = CHECKERBOARD[0] * CHECKERBOARD[1]
    assert pts.shape == (expected_n, 3)


def test_object_points_z_zero():
    pts = collect_object_points(CHECKERBOARD)
    assert np.all(pts[:, 2] == 0.0)


def test_object_points_spacing():
    pts = collect_object_points(CHECKERBOARD)
    # First two corners differ by one square in X
    dx = pts[1, 0] - pts[0, 0]
    assert abs(dx - SQUARE_SIZE_MM) < 1e-6
```

### `tests/test_telemetry_parser.py`

```python
"""
Tests for mobile_telemetry_parser.py using synthetic CSV data.
"""
import io
import math
import numpy as np
import pandas as pd
import pytest

from pipeline.mobile_telemetry_parser import load_telemetry, get_telemetry_at


SYNTHETIC_CSV = """time,latitude,longitude,altitude,roll,pitch,yaw
0,30.0000,31.0000,80.0,0.0,0.0,0.0
1000,30.0001,31.0001,80.5,1.0,0.5,90.0
2000,30.0002,31.0002,81.0,2.0,1.0,180.0
"""


@pytest.fixture
def df():
    return load_telemetry(io.StringIO(SYNTHETIC_CSV))


def test_load_columns(df):
    expected = {"timestamp_ms", "lat", "lon", "alt_m", "roll_rad", "pitch_rad", "yaw_rad"}
    assert expected.issubset(set(df.columns))


def test_load_sorted(df):
    assert df["timestamp_ms"].is_monotonic_increasing


def test_degrees_to_radians(df):
    # Row 1: yaw_deg = 90 → yaw_rad ≈ π/2
    row1 = df[df["timestamp_ms"] == 1000].iloc[0]
    assert abs(row1["yaw_rad"] - math.pi / 2) < 1e-6


def test_interpolation_at_midpoint(df):
    result = get_telemetry_at(df, query_ms=500)
    # At t=500 ms, midpoint between t=0 and t=1000
    assert abs(result["lat"] - 30.00005) < 1e-8
    assert abs(result["alt_m"] - 80.25) < 1e-6


def test_interpolation_clamps_before_start(df):
    result = get_telemetry_at(df, query_ms=-100)
    assert result["lat"] == pytest.approx(30.0000)


def test_interpolation_clamps_after_end(df):
    result = get_telemetry_at(df, query_ms=9999)
    assert result["lat"] == pytest.approx(30.0002)
```

---

---

# Member B — Video Sync & End-to-End Runner

**Deadline:** End of Week 2  
**Depends on:** Member A delivering `mobile_telemetry_parser.py`

---

## B.1 — `pipeline/video_sync.py`

**Purpose:** Extract frames from a `.mp4` with per-frame wall-clock timestamps, then align those timestamps to the telemetry log so each frame can be paired with the correct IMU + GPS row.

**Input:** Video `.mp4` + telemetry DataFrame (from A) + sync offset (seconds)  
**Output:** Iterator that yields `(frame_np, telemetry_dict)` pairs

### Implementation spec

```python
"""
pipeline/video_sync.py

Aligns video frames to telemetry data via a time offset.

Sync strategy:
    The sensor app records a unified clock (milliseconds since epoch).
    The video file records duration only (no absolute time).
    At the START of recording, note the app's displayed timestamp in ms.
    Pass this as `video_start_ms` to VideoTelemetrySync.
    Every frame's absolute time = video_start_ms + (frame_index / fps) * 1000
"""

import cv2
import numpy as np
from pipeline.mobile_telemetry_parser import get_telemetry_at


class VideoTelemetrySync:
    """
    Pairs video frames with interpolated telemetry.

    Parameters
    ----------
    video_path      : path to .mp4 file
    telemetry_df    : DataFrame from load_telemetry()
    video_start_ms  : absolute timestamp (ms) of video frame 0
                      read from the sensor app's session start time
    skip_frames     : process every N-th frame (1 = every frame)
    """

    def __init__(self, video_path, telemetry_df, video_start_ms, skip_frames=1):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.df = telemetry_df
        self.video_start_ms = video_start_ms
        self.skip_frames = skip_frames
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_idx % self.skip_frames == 0:
                frame_ms = self.video_start_ms + (frame_idx / self.fps) * 1000.0
                telemetry = get_telemetry_at(self.df, frame_ms)
                yield frame, telemetry, frame_idx, frame_ms

            frame_idx += 1

    def release(self):
        self.cap.release()

    def __len__(self):
        return self.total_frames // self.skip_frames
```

---

## B.2 — `pipeline/run_localization.py`

**Purpose:** The end-to-end script. Takes a video + telemetry CSV + camera params → outputs `results.json` with GPS coordinates per detected flag.

```python
"""
pipeline/run_localization.py

End-to-end localization runner.

Usage:
    python pipeline/run_localization.py \
        --video       data/test_flight/recording.mp4 \
        --telemetry   data/test_flight/sensors.csv \
        --camera      configs/camera_params_phone.yaml \
        --start-ms    1716640000000 \
        --out         results.json

Output JSON format:
    [
        {"flag_id": 0, "lat": 30.0123, "lon": 31.0456, "n_frames": 47, "mean_confidence": 0.82},
        ...
    ]
"""

import argparse
import json
import yaml
import numpy as np
import cv2

from pipeline.mobile_telemetry_parser import load_telemetry
from pipeline.video_sync import VideoTelemetrySync
from pipeline.detector import FlagDetector
from src.approach2 import localize_full, aggregate_approach2


R_MOUNT = np.array([
    [0,  1, 0],
    [1,  0, 0],
    [0,  0, 1],
], dtype=float)


def load_camera_params(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    K    = np.array(data["camera_matrix"], dtype=float)
    dist = np.array(data["dist_coeffs"],   dtype=float)
    return K, dist


def run(video_path, telemetry_path, camera_yaml, video_start_ms, output_path,
        conf_threshold=0.4, skip_frames=3):

    K, dist = load_camera_params(camera_yaml)
    df_telem = load_telemetry(telemetry_path)
    detector = FlagDetector(conf_threshold=conf_threshold)

    sync = VideoTelemetrySync(video_path, df_telem, video_start_ms,
                              skip_frames=skip_frames)

    # {flag_id: [(lat_estimate, lon_estimate, confidence), ...]}
    all_estimates = {}

    for frame, telem, frame_idx, frame_ms in sync:
        detections = detector.detect(frame)

        for flag_id, u, v, conf in detections:
            est = localize_full(
                u=u, v=v,
                K=K, dist=dist,
                h=telem["alt_m"],
                lat_uav=telem["lat"],
                lon_uav=telem["lon"],
                roll_rad=telem["roll_rad"],
                pitch_rad=telem["pitch_rad"],
                yaw_true_rad=telem["yaw_rad"],
                R_mount=R_MOUNT,
            )

            if est is not None:
                all_estimates.setdefault(flag_id, []).append(
                    (est[0], est[1], conf)
                )

    sync.release()

    results = []
    for flag_id, estimates in all_estimates.items():
        agg = aggregate_approach2(estimates, min_frames=5)
        if agg is not None:
            results.append({
                "flag_id": int(flag_id),
                "lat": agg["lat"],
                "lon": agg["lon"],
                "n_frames": len(estimates),
                "mean_confidence": float(np.mean([e[2] for e in estimates])),
            })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
    for r in results:
        print(f"  Flag {r['flag_id']}: ({r['lat']:.6f}, {r['lon']:.6f})  "
              f"n={r['n_frames']}  conf={r['mean_confidence']:.2f}")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video",      required=True)
    p.add_argument("--telemetry",  required=True)
    p.add_argument("--camera",     required=True)
    p.add_argument("--start-ms",   required=True, type=float,
                   help="Absolute timestamp (ms) of video frame 0 from sensor app")
    p.add_argument("--out",        default="results.json")
    p.add_argument("--skip",       default=3, type=int,
                   help="Process every N-th frame (default 3)")
    args = p.parse_args()

    run(args.video, args.telemetry, args.camera,
        args.start_ms, args.out, skip_frames=args.skip)


if __name__ == "__main__":
    main()
```

---

## B.3 — Tests for Member B

### `tests/test_video_sync.py`

```python
"""
Tests for VideoTelemetrySync using a synthetic video.
"""
import numpy as np
import cv2
import tempfile
import os
import pytest
import pandas as pd

from pipeline.video_sync import VideoTelemetrySync


def make_synthetic_video(path, n_frames=10, fps=30, width=640, height=480):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), i * 25, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def make_synthetic_telemetry(n_rows=100):
    t = np.arange(n_rows) * 100           # 0, 100, 200, ... ms
    df = pd.DataFrame({
        "timestamp_ms": t,
        "lat":          30.0 + t * 1e-6,
        "lon":          31.0 + t * 1e-6,
        "alt_m":        80.0 * np.ones(n_rows),
        "roll_rad":     np.zeros(n_rows),
        "pitch_rad":    np.zeros(n_rows),
        "yaw_rad":      np.zeros(n_rows),
    })
    return df


@pytest.fixture
def sync_fixture(tmp_path):
    video_path = str(tmp_path / "test.mp4")
    make_synthetic_video(video_path, n_frames=10, fps=10)
    df = make_synthetic_telemetry()
    return VideoTelemetrySync(video_path, df, video_start_ms=0.0)


def test_yields_correct_number_of_frames(sync_fixture):
    frames = list(sync_fixture)
    assert len(frames) == 10


def test_frame_is_numpy_array(sync_fixture):
    frames = list(sync_fixture)
    frame, telem, idx, ms = frames[0]
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3


def test_telemetry_dict_has_required_keys(sync_fixture):
    frames = list(sync_fixture)
    _, telem, _, _ = frames[0]
    for key in ["lat", "lon", "alt_m", "roll_rad", "pitch_rad", "yaw_rad"]:
        assert key in telem


def test_frame_timestamps_increase(sync_fixture):
    frames = list(sync_fixture)
    timestamps = [ms for _, _, _, ms in frames]
    assert all(t1 > t0 for t0, t1 in zip(timestamps, timestamps[1:]))


def test_skip_frames_reduces_output():
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v.mp4")
        make_synthetic_video(path, n_frames=30, fps=10)
        df = make_synthetic_telemetry(n_rows=200)
        sync = VideoTelemetrySync(path, df, video_start_ms=0.0, skip_frames=3)
        frames = list(sync)
        assert len(frames) == 10   # 30 / 3
```

---

---

# Member C — Flag Detector & End-to-End Validation

**Deadline:** End of Week 2–3  
**Depends on:** Member A (telemetry), Member B (runner) for integration test

---

## C.1 — `pipeline/detector.py`

**Purpose:** Wrap a detection model (YOLO or color-based fallback) into a common interface. The runner always calls `detector.detect(frame)` regardless of the underlying model.

```python
"""
pipeline/detector.py

FlagDetector wraps either:
  (a) YOLOv8 model          — use when a trained model is available
  (b) Color segmentation    — use for testing with colored targets on the ground

Interface (same for both):
    detector = FlagDetector(model_path="best.pt")   # YOLO
    detector = FlagDetector(use_color=True)          # Color fallback

    detections = detector.detect(frame)
    # returns list of (flag_id, u, v, confidence)
    #   flag_id     : int cluster index (0, 1, ...)
    #   u, v        : pixel center of bbox
    #   confidence  : float 0–1
"""

import cv2
import numpy as np


class FlagDetector:

    def __init__(self, model_path=None, conf_threshold=0.4, use_color=False,
                 color_lower=(0, 120, 70), color_upper=(10, 255, 255)):
        """
        Parameters
        ----------
        model_path      : path to YOLOv8 .pt weights (None → color fallback)
        conf_threshold  : minimum detection confidence
        use_color       : force color-segmentation fallback even if model_path given
        color_lower     : lower HSV bound for target color (default: red)
        color_upper     : upper HSV bound for target color (default: red)
        """
        self.conf_threshold = conf_threshold
        self.use_color = use_color or (model_path is None)
        self.color_lower = np.array(color_lower)
        self.color_upper = np.array(color_upper)
        self.model = None

        if not self.use_color:
            from ultralytics import YOLO
            self.model = YOLO(model_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, frame):
        """
        Detect flags in one frame.

        Returns
        -------
        list of (flag_id, u, v, confidence)
        """
        if self.use_color:
            return self._detect_color(frame)
        return self._detect_yolo(frame)

    # ------------------------------------------------------------------
    # YOLO backend
    # ------------------------------------------------------------------

    def _detect_yolo(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for i, box in enumerate(results.boxes):
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0
            detections.append((i, u, v, conf))

        return detections

    # ------------------------------------------------------------------
    # Color segmentation fallback (for ground tests with colored target)
    # ------------------------------------------------------------------

    def _detect_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)

        # Handle red hue wrap-around (170–180 + 0–10)
        mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask, mask2)

        # Clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 100:      # ignore tiny blobs
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            u = M["m10"] / M["m00"]
            v = M["m01"] / M["m00"]
            confidence = min(1.0, area / 5000.0)    # proxy confidence from blob size
            detections.append((i, u, v, confidence))

        return detections
```

---

## C.2 — Ground Test Protocol

**Goal:** Validate the full pipeline against a known GPS ground truth using the phone at low altitude.

### Setup

```
Materials:
  - 1× colored target: 40×40 cm red/orange cloth pinned flat on ground
  - 1× phone (sensor app running, EIS disabled, recording)
  - 1× measuring tape or second phone for ground truth

Steps:
  1. Place the target flat on open ground (no shadows, good lighting)
  2. Stand directly over the center — record the GPS on the SAME phone.
     This is your ground truth: (lat_true, lon_true)
  3. Hold the phone at exactly 1.5 m height, pointing straight down.
     Walk slowly in an oval around the target while recording (~60 seconds).
  4. Export video + CSV from the sensor app.
  5. Note the session start timestamp from the app (this is video_start_ms).
```

### Expected distances at 1.5 m height

| Flag size | Expected pixel width | Detectable? |
|---|---|---|
| 40×40 cm | ~60–80 px at 1.5 m | ✅ Yes (color detector) |
| 30×30 cm | ~45–60 px | ✅ Yes |
| 20×20 cm | ~30–40 px | ✅ Yes |

At 1.5 m height the pipeline still uses the same math. The output GPS should match ground truth within 0.3 m (phone GPS noise floor).

---

## C.3 — Tests for Member C

### `tests/test_detector.py`

```python
"""
Tests for FlagDetector using synthetic frames with known targets.
"""
import numpy as np
import cv2
import pytest

from pipeline.detector import FlagDetector


def make_red_target_frame(height=480, width=640, cx=320, cy=240, size=60):
    """Create a black frame with a red square at (cx, cy)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    half = size // 2
    frame[cy - half:cy + half, cx - half:cx + half] = (0, 0, 200)   # BGR red
    return frame


@pytest.fixture
def color_detector():
    return FlagDetector(use_color=True)


def test_detects_red_target(color_detector):
    frame = make_red_target_frame(cx=320, cy=240)
    dets = color_detector.detect(frame)
    assert len(dets) >= 1


def test_pixel_center_accuracy(color_detector):
    frame = make_red_target_frame(cx=320, cy=240, size=60)
    dets = color_detector.detect(frame)
    assert len(dets) >= 1
    _, u, v, _ = dets[0]
    assert abs(u - 320) < 5
    assert abs(v - 240) < 5


def test_no_detection_on_black_frame(color_detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = color_detector.detect(frame)
    assert len(dets) == 0


def test_confidence_is_between_0_and_1(color_detector):
    frame = make_red_target_frame()
    dets = color_detector.detect(frame)
    for _, u, v, conf in dets:
        assert 0.0 <= conf <= 1.0


def test_small_noise_not_detected(color_detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Place a 3×3 red dot — below area threshold
    frame[200:203, 300:303] = (0, 0, 200)
    dets = color_detector.detect(frame)
    assert len(dets) == 0
```

### `tests/test_end_to_end.py`

```python
"""
End-to-end integration test.

Uses synthetic video + synthetic telemetry + known target position.
Verifies that run_localization.py outputs a GPS coordinate within
0.5 m of the planted ground truth.
"""
import numpy as np
import pandas as pd
import cv2
import json
import math
import tempfile
import os
import pytest

from pipeline.run_localization import run


# ------------------------------------------------------------------ helpers

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def make_synthetic_telemetry_csv(path, lat=30.0, lon=31.0, alt=2.0, n=60):
    """Stationary UAV directly above the target."""
    t = np.arange(n) * 100
    df = pd.DataFrame({
        "time":      t,
        "latitude":  lat,
        "longitude": lon,
        "altitude":  alt,
        "roll":      0.0,
        "pitch":     0.0,
        "yaw":       0.0,
    })
    df.to_csv(path, index=False)


def make_synthetic_video(path, cx=320, cy=240, n_frames=60, fps=30):
    """Video with a red square at (cx, cy) in every frame — simulates nadir target."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (640, 480))
    for _ in range(n_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[cy - 20:cy + 20, cx - 20:cx + 20] = (0, 0, 200)
        writer.write(frame)
    writer.release()


def make_camera_yaml(path, fx=500, fy=500, cx=320, cy=240):
    import yaml
    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    dist = [[0.0, 0.0, 0.0, 0.0, 0.0]]
    with open(path, "w") as f:
        yaml.dump({"camera_matrix": K, "dist_coeffs": dist,
                   "rms_px": 0.5, "image_size": [640, 480],
                   "model": "brown_conrady"}, f)


# ------------------------------------------------------------------ test

def test_nadir_flag_below_uav(tmp_path):
    """
    UAV hovering directly above target (lat=30.0, lon=31.0) at 2 m height.
    Red square is at image center (cx=320, cy=240).
    Expected output: pipeline reports GPS ≈ (30.0, 31.0) within 0.5 m.
    """
    video_path   = str(tmp_path / "test.mp4")
    telem_path   = str(tmp_path / "sensors.csv")
    camera_path  = str(tmp_path / "cam.yaml")
    results_path = str(tmp_path / "results.json")

    make_synthetic_video(video_path, cx=320, cy=240, n_frames=60)
    make_synthetic_telemetry_csv(telem_path, lat=30.0, lon=31.0, alt=2.0)
    make_synthetic_camera_yaml(camera_path)

    run(
        video_path=video_path,
        telemetry_path=telem_path,
        camera_yaml=camera_path,
        video_start_ms=0.0,
        output_path=results_path,
        conf_threshold=0.1,
        skip_frames=1,
    )

    with open(results_path) as f:
        results = json.load(f)

    assert len(results) >= 1
    r = results[0]
    error_m = haversine_m(30.0, 31.0, r["lat"], r["lon"])
    assert error_m < 0.5, f"GPS error too large: {error_m:.2f} m"
```

---

---

# Integration & Delivery Checklist

## Week 1 — Member A delivers
- [ ] `data/checkerboard/` — ≥ 30 calibration photos
- [ ] `configs/camera_params_phone.yaml` — RMS < 1.0 px
- [ ] `pipeline/calibrate_camera.py`
- [ ] `pipeline/mobile_telemetry_parser.py`
- [ ] `tests/test_calibration.py` — all pass
- [ ] `tests/test_telemetry_parser.py` — all pass

## Week 2 — Member B delivers
- [ ] `pipeline/video_sync.py`
- [ ] `pipeline/run_localization.py`
- [ ] `tests/test_video_sync.py` — all pass
- [ ] `tests/test_run_localization.py` — all pass

## Week 2 — Member C delivers
- [ ] `pipeline/detector.py`
- [ ] `tests/test_detector.py` — all pass
- [ ] Ground truth survey: `data/ground_truth/target_gps.json`

## Week 3 — All three together
- [ ] Real ground test recording in `data/test_flight/`
- [ ] Run `run_localization.py` on real data
- [ ] `tests/test_end_to_end.py` passes on real data
- [ ] Report GPS error in metres vs surveyed ground truth
- [ ] Error < 1.0 m (at 2 m height test) confirms pipeline is correct

---

## Swap-in Guide (GoPro Arrives)

When the GoPro becomes available, only two files change:

1. **Recalibrate:** Run `calibrate_camera.py` on GoPro checkerboard photos → `configs/camera_params_gopro.yaml`
2. **Replace telemetry parser:** Swap `mobile_telemetry_parser.py` for `ardupilot_telemetry_parser.py` that reads `.bin` / `.tlog` via `pymavlink`
3. Pass `--camera configs/camera_params_gopro.yaml` to `run_localization.py`

All geometry code (`approach2.py`), video sync, detector, runner, and tests remain **identical**.

---

*Document end — AI Team Eagles, Nile University, ICMTC 2026*
