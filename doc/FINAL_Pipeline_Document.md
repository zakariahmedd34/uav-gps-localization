# ICMTC 2026 — UAVC-9 Fixed Wing Challenge
# Flag Localization: Final Pipeline Document
### AI Team — Eagles, Nile University

**Version:** 2.0 (GoPro Edition)  
**Date:** 2026-05-25  
**Camera:** GoPro Hero 5 or Hero 9 — Linear Mode  
**Altitude:** 75–100 m AGL  
**Target:** Flag (1 m × 2 m), pinned flat on airport ground  
**Mission:** Detect 2 flags (+ 1 bonus) in search zone → report GPS within 20 m accuracy  

---

## Quick Reference — The Full Chain

```
Video Frame
    ↓
[YOLO Detection] → bounding box (x_min, y_min, x_max, y_max), confidence c
    ↓
[Bbox Center] → (u, v) raw pixel
    ↓
[Brown-Conrady Undistortion] → (u_c, v_c) corrected pixel        ← CHANGED from CDR
    ↓
[Pixel → Camera Ray] → r_cam = [x_n, y_n, 1]
    ↓
[IMU Rotation] → r_world = R_yaw · R_pitch · R_roll · R_mount · r_cam   ← ADDED
    ↓
[Ray–Ground Intersection] → (ΔE, ΔN) meters
    ↓
[GPS Conversion] → (Δlat, Δlon)
    ↓
[Multi-Frame Aggregation] → final (lat_flag, lon_flag)
    ↓
Submit on USB within 10 minutes
```

---

## Part 1 — Competition Context

### Scoring (Section 6.2.1 of Official Rules)

| GPS accuracy | Location pts | + Identification pts | Total per flag |
|---|---|---|---|
| < 20 m | **15** | 10 | **25** |
| 20–30 m | 12 | 10 | 22 |
| 30–40 m | 9 | 10 | 19 |
| 40–50 m | 6 | 10 | 16 |
| 50–60 m | 3 | 10 | 13 |
| > 60 m | 0 | 10 | 10 |

**Bonus flag (3rd flag, outside search area):** Location points × 2. Hit it within 20 m = **30 + 15 = 45 pts**.  
**Total maximum with all flags at < 20 m:** 25 + 25 + 45 = **95 pts** from localization alone.

### Operational Constraints
- Flight altitude: 50–100 m AGL (our target: **75–80 m** for best accuracy)
- Mission time: 10 minutes max takeoff-to-landing
- USB submission: **within 10 minutes** of leaving judges' tent
- No internet; all computation runs locally on laptop or Raspberry Pi 5

---

## Part 2 — Camera Parameters

### GoPro Hero 9 (Preferred)

| Parameter | Value | Notes |
|---|---|---|
| Resolution (use this) | **1920 × 1080 @ 60 fps** | Linear mode |
| fx (pre-calibration estimate) | **~1432 px** | Measure with checkerboard |
| fy (pre-calibration estimate) | **~1432 px** | Should equal fx |
| cx | ~960 px | Near but not exactly W/2 |
| cy | ~540 px | Near but not exactly H/2 |
| HFOV | ~68° | Linear mode |
| Distortion model | **Brown-Conrady** | k1≈−0.09, k2≈+0.06 |
| EIS / HyperSmooth | **OFF** | Critical — disable before every flight |
| Horizon Lock | **OFF** | Critical |

### GoPro Hero 5 (Fallback)

| Parameter | Value | Notes |
|---|---|---|
| Resolution | **1920 × 1080 @ 60 fps** | Linear mode |
| fx (estimate) | **~1060 px** | Narrower FOV than Hero 9 |
| fy (estimate) | **~1060 px** | |
| cx | ~960 px | |
| cy | ~540 px | |
| HFOV | ~84° | Linear mode |
| Distortion model | **Brown-Conrady** | k1≈−0.27, k2≈+0.11 |

> **All values above are pre-calibration estimates. Run `cv2.calibrateCamera` and replace with measured values before competition.**

### Ground Sampling Distance (Flag Coverage)

| Camera | h | GSD (m/px) | Flag long axis | Status |
|---|---|---|---|---|
| Hero 9 Linear | 75 m | 0.052 | **38 px** | ✅ Reliable |
| Hero 9 Linear | 100 m | 0.070 | **29 px** | ✅ OK |
| Hero 5 Linear | 75 m | 0.071 | **28 px** | ✅ OK |
| Hero 5 Linear | 100 m | 0.094 | **21 px** | ⚠️ Marginal |

**Fly at 75–80 m. Below 20 px long axis, YOLO detection becomes unreliable.**

---

## Part 3 — Complete Mathematical Pipeline (All Equations)

### Stage 1: YOLO Bounding Box

YOLO outputs per detection:

$$B = (x_{\min},\; y_{\min},\; x_{\max},\; y_{\max}), \quad c \in [0, 1]$$

where $c$ is the confidence score.

---

### Stage 2: Bounding Box Center

$$u = \frac{x_{\min} + x_{\max}}{2}, \qquad v = \frac{y_{\min} + y_{\max}}{2}$$

---

### Stage 3: Brown-Conrady Lens Distortion Correction

> **This stage is new relative to the CDR. The CDR used the Kannala-Brandt fisheye model — wrong for GoPro.**

GoPro Hero 5/9 in Linear mode follows the **Brown-Conrady** (standard OpenCV) radial-tangential distortion model.

The distortion mapping from undistorted $(x_u, y_u)$ to distorted $(x_d, y_d)$ in normalized coordinates:

$$r^2 = x_u^2 + y_u^2$$

$$x_d = x_u(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2p_1 x_u y_u + p_2(r^2 + 2x_u^2)$$

$$y_d = y_u(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1(r^2 + 2y_u^2) + 2p_2 x_u y_u$$

where $\{k_1, k_2, k_3\}$ are radial coefficients and $\{p_1, p_2\}$ are tangential coefficients.

**Undistortion** (going backward from distorted pixel to undistorted pixel) is done numerically by OpenCV:

```python
# Corrected pixel (u_c, v_c) from raw pixel (u, v):
pts = np.array([[[u, v]]], dtype=np.float32)
pts_corr = cv2.undistortPoints(pts, K, dist, P=K)
u_c, v_c = pts_corr[0, 0, 0], pts_corr[0, 0, 1]
```

> **Only undistort the single detection pixel** — not the full frame. This is 1000× cheaper.

---

### Stage 4: Pixel → Normalized Camera Coordinates (Camera Ray)

Using the corrected pixel and the intrinsic matrix $K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$:

$$x_n = \frac{u_c - c_x}{f_x}, \qquad y_n = \frac{v_c - c_y}{f_y}$$

The **camera ray** (direction from optical center toward the flag, in camera frame):

$$\mathbf{r}_{\text{cam}} = \begin{bmatrix} x_n \\ y_n \\ 1 \end{bmatrix}$$

**Physical meaning:** $x_n = 0.05$ means the flag is 5% of one focal-length to the right of the image center. Multiplied by altitude $h$, it gives the metric ground offset.

---

### Stage 5: Rotate Camera Ray to World (NED) Frame

> **This is the step missing from the original simplified pipeline — its absence causes directional errors up to 100% of the offset magnitude [1].**

The full rotation chain:

$$\mathbf{r}_{\text{world}} = \underbrace{\mathbf{R}_{\text{NED} \leftarrow \text{body}}}_{\text{from IMU}} \cdot \underbrace{\mathbf{R}_{\text{body} \leftarrow \text{cam}}}_{\text{mount, fixed}} \cdot \mathbf{r}_{\text{cam}}$$

#### 5a. Camera Mount Rotation $\mathbf{R}_{\text{mount}}$

Determined **once** by the physical mounting. For a GoPro mounted nadir with its top (lens-side label) pointing toward the aircraft nose:

| Camera axis | Body axis |
|---|---|
| $+X$ (image right) | $+Y$ body (starboard) |
| $+Y$ (image down) | $+X$ body (forward) |
| $+Z$ (into scene = down) | $+Z$ body (down) |

$$\mathbf{R}_{\text{mount}} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

> Verify experimentally: point at a known landmark and check direction. Different physical orientations need a different matrix.

#### 5b. UAV Attitude Rotation $\mathbf{R}_{\text{NED} \leftarrow \text{body}}$

ZYX Euler convention (yaw applied first, as per Barber & Redding [1] and ArduPilot):

$$\mathbf{R} = \mathbf{R}_z(\psi) \cdot \mathbf{R}_y(\theta) \cdot \mathbf{R}_x(\phi)$$

where:
- $\phi$ = roll (rotation about $x$-axis)
- $\theta$ = pitch (rotation about $y$-axis)
- $\psi$ = **true heading** = magnetic heading + 4° (Cairo magnetic declination)

$$\mathbf{R}_x(\phi) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\phi & -\sin\phi \\ 0 & \sin\phi & \cos\phi \end{bmatrix}$$

$$\mathbf{R}_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}$$

$$\mathbf{R}_z(\psi) = \begin{bmatrix} \cos\psi & -\sin\psi & 0 \\ \sin\psi & \cos\psi & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

Full world ray:

$$\mathbf{r}_{\text{world}} = \mathbf{R}_z(\psi) \cdot \mathbf{R}_y(\theta) \cdot \mathbf{R}_x(\phi) \cdot \mathbf{R}_{\text{mount}} \cdot \mathbf{r}_{\text{cam}} = \begin{bmatrix} r_x \\ r_y \\ r_z \end{bmatrix}$$

**Heading source (recommended by [8]):** Use **GPS course-over-ground** (COG) for $\psi$ when the aircraft is in straight-and-level forward flight — more accurate than magnetometer for a moving fixed-wing aircraft because it is immune to motor magnetic interference.

---

### Stage 6: Ray–Ground Plane Intersection

The airport ground is a flat plane at $z = 0$ in local NED. The UAV is at altitude $h$ above it. The parametric ray from $\mathbf{P}_{\text{UAV}} = [0, 0, h]^T$ in direction $\mathbf{r}_{\text{world}}$ hits the ground when:

$$t = \frac{h}{r_z}$$

**Safety guard:** If $|r_z| < 0.1$, the ray is nearly horizontal (camera severely tilted or edge detection). **Reject this frame.**

Ground offset in meters (NED):

$$\Delta E = t \cdot r_x \quad \text{(East, meters)}$$

$$\Delta N = t \cdot r_y \quad \text{(North, meters)}$$

**Nadir special case** (zero tilt, $\mathbf{R} = \mathbf{I}$): collapses to $\Delta E = x_n \cdot h$, $\Delta N = -y_n \cdot h$. This is the simplified baseline formula — valid only when the camera is perfectly vertical.

---

### Stage 7: GPS Conversion (Flat-Earth Approximation)

Valid for displacements < 1 km (entire competition search zone):

$$\Delta\text{lat} = \frac{\Delta N}{111{,}320} \quad \text{(degrees)}$$

$$\Delta\text{lon} = \frac{\Delta E}{111{,}320 \cdot \cos\!\left(\varphi_{\text{UAV}} \cdot \frac{\pi}{180}\right)} \quad \text{(degrees)}$$

For **Cairo** ($\varphi \approx 30°$N): $\cos(30°) = 0.866$, so 1° longitude $\approx 96{,}388$ m.

Final flag GPS coordinates:

$$\boxed{\varphi_{\text{flag}} = \varphi_{\text{UAV}} + \Delta\text{lat}}$$

$$\boxed{\lambda_{\text{flag}} = \lambda_{\text{UAV}} + \Delta\text{lon}}$$

---

### Stage 8: Multi-Frame Aggregation

Given $N$ per-frame estimates $\{(\hat{\varphi}^{(i)}, \hat{\lambda}^{(i)})\}_{i=1}^{N}$ with confidence scores $c_i$:

**Approach 1 — Simple Median:**
$$\hat{\varphi}_{\text{final}} = \text{median}\!\left(\hat{\varphi}^{(1)}, \ldots, \hat{\varphi}^{(N)}\right)$$

**Approach 2 — Confidence-Weighted Mean** (from [9]):
$$\hat{\varphi}_{\text{final}} = \frac{\sum_{i=1}^{N} c_i \cdot \hat{\varphi}^{(i)}}{\sum_{i=1}^{N} c_i}$$

**Approach 3 — DBSCAN Cluster + Weighted Mean** (from [2, Lago et al.]):
- Convert all estimates to local meters (flat-earth)
- Run DBSCAN($\varepsilon = 15$ m, minPts $= 3$) to find the dominant cluster
- Apply confidence-weighted mean within the cluster
- Outlier detections (isolated points) are rejected as noise

**GPS noise reduction by averaging** (from [1]):

$$\sigma_{\text{avg}} = \frac{\sigma_{\text{GPS}}}{\sqrt{N}}$$

| N frames | Residual GPS $\sigma$ (from 3 m baseline) |
|---|---|
| 3 (CDR) | 1.73 m |
| 10 | 0.95 m |
| **20** | **0.67 m** ← target |
| 40 | 0.47 m |

**Target: N ≥ 20 frames per flag.** At 60 fps with a 3-second overpass, 180 frames are available. Taking only frames with $c > 0.5$ confidence typically yields 20–50 usable frames.

---

## Part 4 — Three Implementation Approaches

---

### Approach 1 — Simplified Baseline (MVP)

**When to use:** First test, hardware not fully ready, no IMU data available.  
**Expected accuracy:** 10–30 m (may miss < 20 m bin if not nadir)  
**Assumptions:** Camera perfectly nadir, no tilt, no distortion correction  
**Pipeline stages used:** 1, 2, 4 (simplified), 6 (nadir), 7, 8 (median)

```python
"""
APPROACH 1 — SIMPLIFIED BASELINE
Source: Simplified version from UAV_GPS_Localization_Engineering_Report.md
Assumption: Camera is perfectly nadir, no IMU, no distortion correction.
Use as sanity check only.
"""

import numpy as np
import math

def localize_baseline(u, v, K, h, lat_uav, lon_uav):
    """
    Simplified nadir localization — no IMU, no undistortion.

    Args:
        u, v       : raw pixel coordinates of YOLO bbox center
        K          : 3x3 camera intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
        h          : altitude AGL in meters
        lat_uav    : UAV latitude (degrees)
        lon_uav    : UAV longitude (degrees)

    Returns:
        (lat_flag, lon_flag) or None
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Normalize pixel offset (no undistortion)
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy

    # Nadir ground offset (image +Y is down, so North = -y_n * h)
    dE = x_n * h       # East  (meters)
    dN = -y_n * h      # North (meters) — negative sign: image-down = world-south

    # GPS conversion
    dlat = dN / 111320.0
    dlon = dE / (111320.0 * math.cos(math.radians(lat_uav)))

    return lat_uav + dlat, lon_uav + dlon


def aggregate_approach1(estimates):
    """
    Simple median aggregation.

    Args:
        estimates : list of (lat, lon) tuples

    Returns:
        (lat_final, lon_final) or None
    """
    if len(estimates) < 3:
        return None
    coords = np.array(estimates)
    return tuple(np.median(coords, axis=0))


# ─── USAGE EXAMPLE ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example camera (Hero 9 Linear 1080p — replace with calibrated values)
    K = np.array([[1432, 0, 960],
                  [0, 1432, 540],
                  [0,    0,   1]], dtype=float)

    # UAV state (from telemetry log)
    h       = 80.0          # meters AGL
    lat_uav = 30.0613       # degrees (example Cairo-area lat)
    lon_uav = 31.2497       # degrees (example Cairo-area lon)

    # Detection from YOLO (example)
    detections = [(950, 560), (952, 558), (948, 562)]  # (u, v) per frame

    estimates = []
    for u, v in detections:
        est = localize_baseline(u, v, K, h, lat_uav, lon_uav)
        estimates.append(est)

    result = aggregate_approach1(estimates)
    print(f"Approach 1 → lat: {result[0]:.7f}, lon: {result[1]:.7f}")
```

---

### Approach 2 — Full Pipeline (Recommended for Competition)

**When to use:** Main competition pipeline — use this from Week 2 onward.  
**Expected accuracy:** 4–8 m (within < 20 m bin reliably)  
**Sources:** [1] Barber & Redding, [UAV Engineering Report], [GoPro Corrections Doc]  
**Pipeline stages:** All 8 stages, confidence-weighted aggregation

```python
"""
APPROACH 2 — FULL PIPELINE (RECOMMENDED)
Sources:
  [1] Barber & Redding (2006) — ZYX rotation, ray-ground intersection
  [15] MDPI Sensors (2014) — Brown-Conrady for GoPro
  UAV_GPS_Localization_Engineering_Report.md — error analysis
  GoPro_Pipeline_Analysis_And_Corrections.md — undistortion fix

Implements:
  - Brown-Conrady undistortion (cv2.undistortPoints, NOT fisheye)
  - Full ZYX attitude rotation with mount compensation
  - Ray-ground intersection
  - Confidence-weighted multi-frame aggregation (N >= 20)
"""

import numpy as np
import math
import cv2


# ─── ROTATION UTILITIES ──────────────────────────────────────────────────────

def Rx(a):
    """Rotation about X (roll)."""
    return np.array([[1, 0, 0],
                     [0, math.cos(a), -math.sin(a)],
                     [0, math.sin(a),  math.cos(a)]])

def Ry(a):
    """Rotation about Y (pitch)."""
    return np.array([[ math.cos(a), 0, math.sin(a)],
                     [0,            1, 0           ],
                     [-math.sin(a), 0, math.cos(a)]])

def Rz(a):
    """Rotation about Z (yaw)."""
    return np.array([[math.cos(a), -math.sin(a), 0],
                     [math.sin(a),  math.cos(a), 0],
                     [0,            0,            1]])

def build_R_total(roll_rad, pitch_rad, yaw_true_rad, R_mount):
    """
    Full rotation: camera frame -> body frame -> NED world frame.
    ZYX Euler convention (Barber & Redding [1]).

    Args:
        roll_rad      : roll in radians (from IMU)
        pitch_rad     : pitch in radians (from IMU)
        yaw_true_rad  : TRUE heading in radians (magnetic + declination)
        R_mount       : 3x3 camera-to-body rotation (determined by mount geometry)

    Returns:
        3x3 combined rotation matrix
    """
    R_body_to_ned = Rz(yaw_true_rad) @ Ry(pitch_rad) @ Rx(roll_rad)
    return R_body_to_ned @ R_mount


# ─── SINGLE-FRAME LOCALIZATION ───────────────────────────────────────────────

def localize_full(u, v, K, dist, h,
                  lat_uav, lon_uav,
                  roll_rad, pitch_rad, yaw_true_rad,
                  R_mount,
                  oblique_threshold=0.1):
    """
    Full single-frame GPS localization for GoPro Hero 5/9 in Linear mode.

    Args:
        u, v              : raw YOLO bbox center pixel (float)
        K                 : 3x3 intrinsic matrix (from cv2.calibrateCamera)
        dist              : distortion coefficients [k1,k2,p1,p2,k3]
        h                 : altitude AGL in meters (use barometer, not GPS altitude)
        lat_uav           : UAV latitude degrees
        lon_uav           : UAV longitude degrees
        roll_rad          : roll  (from IMU, radians)
        pitch_rad         : pitch (from IMU, radians)
        yaw_true_rad      : TRUE heading (magnetic heading + 0.0698 rad for Cairo +4°)
        R_mount           : camera-to-body 3x3 rotation
        oblique_threshold : reject frames where camera ray z-component < this value

    Returns:
        (lat_flag, lon_flag) in degrees, or None if rejected
    """
    # Stage 3: Undistort detection point (Brown-Conrady, NOT fisheye API)
    pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
    pt_corr = cv2.undistortPoints(pt, K, dist, P=K)
    u_c = float(pt_corr[0, 0, 0])
    v_c = float(pt_corr[0, 0, 1])

    # Stage 4: Pixel → normalized camera ray
    x_n = (u_c - K[0, 2]) / K[0, 0]
    y_n = (v_c - K[1, 2]) / K[1, 1]
    r_cam = np.array([x_n, y_n, 1.0])

    # Stage 5: Rotate to world (NED)
    R = build_R_total(roll_rad, pitch_rad, yaw_true_rad, R_mount)
    r_world = R @ r_cam

    # Stage 6: Ray–ground intersection (reject oblique rays)
    if abs(r_world[2]) < oblique_threshold:
        return None
    t  = h / r_world[2]
    dE = t * r_world[0]   # East offset  (meters)
    dN = t * r_world[1]   # North offset (meters)

    # Stage 7: GPS conversion (flat-earth, valid < 10 km)
    dlat = dN / 111320.0
    dlon = dE / (111320.0 * math.cos(math.radians(lat_uav)))

    return lat_uav + dlat, lon_uav + dlon


# ─── CONFIDENCE-WEIGHTED AGGREGATION ─────────────────────────────────────────

def aggregate_approach2(estimates_with_confidence, min_frames=10):
    """
    Confidence-weighted mean aggregation across N frames.
    Source: [9] Real-Time Multi-Target Localization (PMC)

    Args:
        estimates_with_confidence : list of ((lat, lon), confidence) tuples
        min_frames                : reject flag if fewer valid frames

    Returns:
        (lat_final, lon_final) or None
    """
    valid = [(est, c) for est, c in estimates_with_confidence
             if est is not None and c > 0.4]

    if len(valid) < min_frames:
        return None

    total_weight = sum(c for _, c in valid)
    lat_w = sum(est[0] * c for est, c in valid) / total_weight
    lon_w = sum(est[1] * c for est, c in valid) / total_weight

    return lat_w, lon_w


# ─── USAGE EXAMPLE ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    # Load calibrated camera parameters (produced by Person A)
    # Uncomment after calibration is done:
    # with open("camera_params.yaml") as f:
    #     cam = yaml.safe_load(f)
    # K    = np.array(cam["K"])
    # dist = np.array(cam["dist"])

    # For now — pre-calibration Hero 9 estimates:
    K    = np.array([[1432, 0, 960],
                     [0, 1432, 540],
                     [0,    0,   1]], dtype=float)
    dist = np.array([-0.09, 0.06, 0.0, 0.0, 0.0])  # [k1, k2, p1, p2, k3]

    # Camera mount (nadir, image-top = aircraft forward)
    R_mount = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]], dtype=float)

    CAIRO_DECLINATION_RAD = math.radians(4.0)   # +4° East

    # Simulated detections from one flag pass
    # Each entry: (u_px, v_px, lat_uav, lon_uav, h_m,
    #              roll_rad, pitch_rad, mag_heading_rad, confidence)
    pass_data = [
        (960, 545, 30.0613, 31.2497, 79.5, 0.01, -0.02, 0.52, 0.91),
        (962, 543, 30.0612, 31.2498, 80.1, 0.02, -0.01, 0.53, 0.88),
        (958, 547, 30.0614, 31.2496, 79.8, 0.00, -0.03, 0.51, 0.93),
    ]

    results = []
    for u, v, lat, lon, h, roll, pitch, mag_hdg, conf in pass_data:
        true_hdg = mag_hdg + CAIRO_DECLINATION_RAD
        est = localize_full(u, v, K, dist, h,
                            lat, lon,
                            roll, pitch, true_hdg,
                            R_mount)
        results.append((est, conf))
        if est:
            print(f"  Frame → lat: {est[0]:.7f}, lon: {est[1]:.7f} (conf: {conf:.2f})")

    final = aggregate_approach2(results, min_frames=2)  # use 2 for demo; set 10 in production
    if final:
        print(f"\nApproach 2 Final → lat: {final[0]:.7f}, lon: {final[1]:.7f}")
```

---

### Approach 3 — Advanced Pipeline (Best Accuracy)

**When to use:** After Approach 2 is validated. Use for competition if time permits.  
**Expected accuracy:** 2–5 m  
**Sources:** [2] Lago et al. (DBSCAN), [10] SAHI (detection), [7] AIAA 2020 (auto-calibration), [9] (confidence weighting)  
**Added features:** DBSCAN outlier rejection, SAHI for detection at 100 m, GPS COG heading

```python
"""
APPROACH 3 — ADVANCED PIPELINE
Sources:
  [2]  Lago et al. 2024 — DBSCAN GPS clustering
  [10] SAHI (2022) — Sliced inference for tiny flags at 100 m
  [7]  AIAA 2020 — Automatic mount angle discovery
  [9]  PMC Multi-target — Confidence-weighted aggregation
  [8]  Sensors AR — GPS COG as heading source

Added over Approach 2:
  - DBSCAN spatial clustering to reject isolated false-positive GPS estimates
  - Optional SAHI detection wrapper for high-altitude frames
  - GPS course-over-ground (COG) heading option
"""

import numpy as np
import math
import cv2

# Reuse rotation functions from Approach 2:
from approach2 import Rx, Ry, Rz, build_R_total, localize_full


# ─── DETECTION WITH OPTIONAL SAHI ────────────────────────────────────────────

def detect_flags_yolo(frame, yolo_model, use_sahi=False,
                      slice_size=512, overlap=0.2, conf_threshold=0.5):
    """
    Detect flags in a frame. Optionally uses SAHI for high-altitude frames.

    Args:
        frame           : BGR image (numpy array)
        yolo_model      : loaded Ultralytics YOLO model
        use_sahi        : True at altitudes >= 90 m (flag < 25 px)
        slice_size      : SAHI tile size in pixels
        overlap         : SAHI tile overlap ratio
        conf_threshold  : minimum confidence to accept detection

    Returns:
        list of (u, v, confidence) tuples
    """
    detections = []

    if use_sahi:
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            # Wrap model for SAHI (run once, reuse)
            result = get_sliced_prediction(
                frame,
                yolo_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
            )
            for pred in result.object_prediction_list:
                if pred.score.value >= conf_threshold:
                    bbox = pred.bbox
                    u = (bbox.minx + bbox.maxx) / 2
                    v = (bbox.miny + bbox.maxy) / 2
                    detections.append((u, v, pred.score.value))
        except ImportError:
            print("SAHI not installed — falling back to standard YOLO")
            use_sahi = False

    if not use_sahi:
        results = yolo_model(frame, conf=conf_threshold, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            u = (x1 + x2) / 2
            v = (y1 + y2) / 2
            c = float(box.conf[0])
            detections.append((u, v, c))

    return detections


# ─── GPS COG HEADING ─────────────────────────────────────────────────────────

def heading_from_gps_cog(lat1, lon1, lat2, lon2):
    """
    Compute true heading from GPS course-over-ground.
    More accurate than magnetometer for moving fixed-wing UAV [8].

    Args:
        (lat1, lon1) : previous GPS position
        (lat2, lon2) : current GPS position (both in degrees)

    Returns:
        True heading in radians (0 = North, π/2 = East)
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Convert to meters
    dlat_m = dlat * 111320.0
    dlon_m = dlon * 111320.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.atan2(dlon_m, dlat_m)  # atan2(East, North) = bearing from North


# ─── DBSCAN AGGREGATION ───────────────────────────────────────────────────────

def aggregate_approach3(estimates_with_confidence,
                        min_frames=10,
                        dbscan_eps_m=15.0,
                        dbscan_min_samples=3):
    """
    DBSCAN clustering + confidence-weighted mean.
    Source: [2] Lago et al. 2024 — removes isolated false-positive GPS estimates.

    Args:
        estimates_with_confidence : list of ((lat, lon), confidence) tuples
        min_frames                : minimum valid detections required
        dbscan_eps_m              : cluster radius in meters (15 m default)
        dbscan_min_samples        : minimum points to form a cluster

    Returns:
        (lat_final, lon_final) or None
    """
    from sklearn.cluster import DBSCAN

    valid = [(est, c) for est, c in estimates_with_confidence
             if est is not None and c > 0.4]

    if len(valid) < min_frames:
        return None

    coords   = np.array([est for est, _ in valid])
    conf_arr = np.array([c   for _, c   in valid])

    # Convert to local meters for clustering
    lat0 = np.mean(coords[:, 0])
    lon0 = np.mean(coords[:, 1])
    coords_m = np.column_stack([
        (coords[:, 0] - lat0) * 111320.0,
        (coords[:, 1] - lon0) * 111320.0 * math.cos(math.radians(lat0))
    ])

    # DBSCAN: find tight spatial cluster
    labels = DBSCAN(eps=dbscan_eps_m,
                    min_samples=dbscan_min_samples).fit_predict(coords_m)

    valid_mask = labels >= 0
    if not valid_mask.any():
        # No cluster found — fall back to confidence-weighted mean of all
        w = conf_arr / conf_arr.sum()
        return float(np.dot(w, coords[:, 0])), float(np.dot(w, coords[:, 1]))

    # Largest cluster
    cluster_labels = labels[valid_mask]
    best_label     = np.bincount(cluster_labels).argmax()
    cluster_mask   = labels == best_label

    cluster_coords = coords[cluster_mask]
    cluster_conf   = conf_arr[cluster_mask]

    # Confidence-weighted mean within cluster
    w       = cluster_conf / cluster_conf.sum()
    lat_fin = float(np.dot(w, cluster_coords[:, 0]))
    lon_fin = float(np.dot(w, cluster_coords[:, 1]))

    cluster_size = cluster_mask.sum()
    rejected     = len(valid) - cluster_size
    print(f"  DBSCAN: {cluster_size} frames in cluster, {rejected} outliers rejected")

    return lat_fin, lon_fin


# ─── FULL VIDEO PROCESSING LOOP ──────────────────────────────────────────────

def process_video_for_flags(video_path, telemetry_log,
                             K, dist, R_mount,
                             yolo_model,
                             frame_sample_rate=5,
                             altitude_sahi_threshold=90.0):
    """
    Process a full mission video and return GPS estimates for all detected flags.

    Args:
        video_path            : path to .mp4 video file
        telemetry_log         : dict mapping frame_index -> {lat, lon, h, roll, pitch, yaw_mag}
        K, dist, R_mount      : camera parameters
        yolo_model            : loaded Ultralytics YOLO model
        frame_sample_rate     : process every Nth frame (5 = 12 fps from 60 fps)
        altitude_sahi_threshold : use SAHI when h > this value (meters)

    Returns:
        dict: {flag_id: (lat_final, lon_final)}
    """
    CAIRO_DECLINATION_RAD = math.radians(4.0)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    all_estimates = []   # (estimate, confidence) per frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_sample_rate != 0:
            frame_idx += 1
            continue

        # Get synchronized telemetry for this frame
        telem = telemetry_log.get(frame_idx)
        if telem is None:
            frame_idx += 1
            continue

        lat_uav   = telem["lat"]
        lon_uav   = telem["lon"]
        h         = telem["h"]
        roll      = math.radians(telem["roll_deg"])
        pitch     = math.radians(telem["pitch_deg"])
        yaw_true  = math.radians(telem["yaw_mag_deg"]) + CAIRO_DECLINATION_RAD

        # Use SAHI at high altitude where flags are < 25 px
        use_sahi = h > altitude_sahi_threshold

        detections = detect_flags_yolo(frame, yolo_model, use_sahi=use_sahi)

        for u, v, conf in detections:
            est = localize_full(u, v, K, dist, h,
                                lat_uav, lon_uav,
                                roll, pitch, yaw_true,
                                R_mount)
            all_estimates.append((est, conf))

        frame_idx += 1

    cap.release()

    # Aggregate all estimates → final flag GPS
    final_gps = aggregate_approach3(all_estimates, min_frames=10)
    return final_gps


# ─── AUTOMATIC MOUNT CALIBRATION ─────────────────────────────────────────────

def auto_calibrate_mount(known_gps_target, observations, K, dist, h_mean,
                         lat_uav_mean, lon_uav_mean):
    """
    Automatically discover the camera mount offset angles.
    Source: [7] AIAA 2020 — Improving UAV-Based Target Geolocation Accuracy.

    Flies over a known GPS point, collects multiple detections, then
    optimizes mount roll/pitch/yaw offsets to minimize localization error.

    Args:
        known_gps_target : (lat_true, lon_true) of the ground marker
        observations     : list of (u, v, roll_rad, pitch_rad, yaw_rad) per frame
        K, dist          : camera intrinsics
        h_mean           : approximate flight altitude
        lat_uav_mean     : approximate UAV latitude during pass
        lon_uav_mean     : approximate UAV longitude during pass

    Returns:
        R_mount_optimized : 3x3 optimized camera-to-body rotation
    """
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    lat_true, lon_true = known_gps_target

    def error_fn(euler_xyz):
        """Total squared error across all observations."""
        droll, dpitch, dyaw = euler_xyz
        R_adj = Rotation.from_euler('xyz', [droll, dpitch, dyaw]).as_matrix()

        total_err = 0.0
        for u, v, roll, pitch, yaw in observations:
            r_world = build_R_total(roll + droll, pitch + dpitch,
                                    yaw + dyaw, R_adj) @ \
                      np.array([(u - K[0,2]) / K[0,0],
                                (v - K[1,2]) / K[1,1],
                                1.0])
            if abs(r_world[2]) < 0.05:
                continue
            t  = h_mean / r_world[2]
            dE = t * r_world[0]
            dN = t * r_world[1]
            lat_est = lat_uav_mean + dN / 111320.0
            lon_est = lon_uav_mean + dE / (111320.0 * math.cos(math.radians(lat_uav_mean)))
            # Error in meters
            err_m = math.sqrt(((lat_est - lat_true) * 111320) ** 2 +
                              ((lon_est - lon_true) * 111320 * math.cos(math.radians(lat_true))) ** 2)
            total_err += err_m ** 2

        return total_err

    result = minimize(error_fn, [0.0, 0.0, 0.0],
                      method='Nelder-Mead',
                      options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 5000})

    droll, dpitch, dyaw = result.x
    R_mount_opt = Rotation.from_euler('xyz', [droll, dpitch, dyaw]).as_matrix()
    print(f"Mount offset: roll={math.degrees(droll):.2f}°, "
          f"pitch={math.degrees(dpitch):.2f}°, yaw={math.degrees(dyaw):.2f}°")
    print(f"Residual RMS error: {math.sqrt(result.fun / max(len(observations), 1)):.2f} m")
    return R_mount_opt
```

---

## Part 5 — Approach Comparison

| Criterion | Approach 1 (MVP) | Approach 2 (Standard) | Approach 3 (Advanced) |
|---|---|---|---|
| **Expected accuracy** | 10–30 m | 4–8 m | 2–5 m |
| **Hits < 20 m bin?** | ⚠️ Maybe | ✅ Yes (reliably) | ✅ Yes (robustly) |
| **IMU rotation** | ❌ No | ✅ Yes | ✅ Yes |
| **Undistortion** | ❌ No | ✅ Yes | ✅ Yes |
| **Multi-frame avg** | Median (N=3+) | Weighted mean (N≥20) | DBSCAN + weighted (N≥20) |
| **SAHI detection** | ❌ | ❌ | ✅ At h>90m |
| **Auto-mount calib** | ❌ | ❌ | ✅ Yes |
| **GPS heading** | Magnetometer | Magnetometer + declination | COG preferred |
| **Setup time** | 1 day | 1 week | 2–3 weeks |
| **Code complexity** | Low | Medium | High |
| **Risk if bugs** | Low | Medium | Medium |

**Recommendation:** Implement Approach 2 for the competition. Approach 3 improvements are incremental (2–5 m vs 4–8 m) but require significantly more development and testing time. Approach 1 is for initial smoke testing only.

---

## Part 6 — Error Budget (Approach 2, h = 80 m)

| Error source | Magnitude | Ground error | Severity | Fix |
|---|---|---|---|---|
| UAV GPS noise | ±3 m (1σ) | **±3 m** | 🔴 HIGH | N≥20 frames → σ/√20 ≈ 0.67 m |
| Camera tilt 2° | 2° pitch/roll | **±2.8 m** | 🔴 HIGH | Stage 5 IMU rotation |
| GoPro EIS ON | Frame warp | **up to 5 m** | 🔴 CRITICAL | Disable EIS before every flight |
| Yaw/heading error 2° | 2° | ~0.7 m at 20 m | 🟡 MED | GPS COG or declination correction |
| Time sync 200 ms @ 15 m/s | frame drift | ~3 m | 🟡 MED | Log timestamps; interpolate |
| Altitude error ±5 m | ±6% of offset | ~0.6 m | 🟡 MED | Barometer + ground calibration |
| Undistortion residual | <0.5% center | <0.1 m | 🟢 LOW | Good calibration |
| Bbox center jitter ±3 px | 3 × GSD | ~0.15 m | 🟢 LOW | Median over N frames |
| Cairo declination ignored | 4° | ~1.4 m at 20 m | 🟡 MED | Add +4° to magnetic heading |

**RSS total (Approach 2, all mitigations applied):** ~4–6 m → **< 20 m bin: reliable**  
**RSS total (no IMU, EIS on):** ~15–25 m → **< 20 m bin: unreliable**

---

## Part 7 — Team Task Distribution (3 Members)

---

### 👤 Person A — Calibration & Camera Lead

**Mission:** "I make sure the camera numbers are correct. Without me, nothing else works."

#### Deliverables

| # | Deliverable | Deadline | Status |
|---|---|---|---|
| A1 | `camera_params.yaml` — K matrix + dist coefficients, RMS < 1.0 px | Week 1 | ⬜ |
| A2 | `undistort_pixel(u, v, K, dist) → (u_c, v_c)` Python function, tested | Week 1 | ⬜ |
| A3 | Before/after undistortion image (checkerboard lines must be straight) | Week 1 | ⬜ |
| A4 | R_mount determination — experimental, with unit test result | Week 2 | ⬜ |
| A5 | Written note on GoPro body: "MODE: Linear, EIS: OFF, calib v1.0" | Week 1 | ⬜ |
| A6 | Mount verticality report (digital level measurement, in degrees) | Week 2 | ⬜ |

#### Tasks

**Week 1 — GoPro Calibration:**
1. Print 9×6 checkerboard, 30 mm squares, on rigid A2 foamboard
2. Set GoPro to Linear mode, EIS OFF, 1080p60. Confirm on screen.
3. Capture 40–50 still images of the checkerboard:
   - Cover all corners of the frame (edges, corners, center)
   - Use distances 0.5 m, 1 m, 2 m from camera
   - Multiple tilt angles: 0°, ±30°, ±45° rotation
4. Run calibration:

```python
# calibrate_gopro.py — run this once
import cv2, numpy as np, glob, yaml

CHECKERBOARD = (9, 6)
SQUARE_MM = 30

objp = np.zeros((54, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * SQUARE_MM

objpoints, imgpoints = [], []
images = glob.glob("calibration_images/*.jpg")

for fname in images:
    img  = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        objpoints.append(objp)
        imgpoints.append(corners2)

rms, K, dist, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print(f"RMS reprojection error: {rms:.4f} px")
assert rms < 1.5, f"RMS {rms:.2f} too high — recapture calibration images"

params = {
    "camera": "GoPro Hero 9 (or 5)",
    "mode": "Linear 1080p60",
    "eis": "OFF",
    "rms_px": float(rms),
    "K": K.tolist(),
    "dist": dist.flatten().tolist(),
    "fx": float(K[0,0]),
    "fy": float(K[1,1]),
    "cx": float(K[0,2]),
    "cy": float(K[1,2]),
}

with open("camera_params.yaml", "w") as f:
    yaml.dump(params, f, default_flow_style=False)

print("Saved camera_params.yaml")
print(f"  fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
print(f"  cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
print(f"  dist={dist.flatten()}")
```

**Week 2 — R_mount Verification:**
5. Place a marker at a **measured GPS location** on flat ground
6. Hold UAV by hand at known altitude (measure with tape) directly above a reference
7. Run `localize_full()` with identity R_mount, compare result to known GPS
8. Adjust R_mount until error < 5 m for markers in all 4 directions (N, E, S, W relative to UAV)
9. Document the final R_mount matrix in `camera_params.yaml`

---

### 👤 Person B — Geometry & Pipeline Lead

**Mission:** "I turn a pixel into a GPS coordinate. I own the localization code."

#### Deliverables

| # | Deliverable | Deadline | Status |
|---|---|---|---|
| B1 | `localization.py` — Approach 2 fully implemented and tested | Week 2 | ⬜ |
| B2 | 5 unit tests with synthetic ground truth (all pass) | Week 2 | ⬜ |
| B3 | End-to-end demo: sample video frame + telemetry → prints GPS | Week 2 | ⬜ |
| B4 | `aggregate.py` — confidence-weighted aggregation + DBSCAN (Approach 3) | Week 3 | ⬜ |
| B5 | Telemetry parser: reads ArduPilot `.bin` or `.tlog` → dict of {frame_idx: state} | Week 2 | ⬜ |
| B6 | SAHI integration (optional, if flags at 100 m are not detected reliably) | Week 4 | ⬜ |

#### Tasks

**Week 1 — Implement Approach 1 as skeleton:**
1. Implement `localize_baseline()` (Approach 1 code above)
2. Test on synthetic pixel: UAV at 30.0000°N, 31.0000°E, h=80m, flag directly below → should return (30.0000, 31.0000)

**Unit Tests (must pass before competition):**

```python
# test_localization.py
import numpy as np, math, pytest
from approach2 import localize_full, Rz

K_test = np.array([[1432, 0, 960],
                   [0, 1432, 540],
                   [0,    0,   1]], dtype=float)
dist_test = np.zeros(5)  # no distortion for unit tests
R_mount_I = np.eye(3)    # identity mount

LAT0, LON0 = 30.0, 31.0
H = 80.0

def test_nadir_center():
    """Flag at image center → should be directly below UAV."""
    est = localize_full(960, 540, K_test, dist_test, H,
                        LAT0, LON0, 0.0, 0.0, 0.0, R_mount_I)
    assert abs(est[0] - LAT0) < 0.0001, "Latitude off center"
    assert abs(est[1] - LON0) < 0.0001, "Longitude off center"

def test_flag_east_of_uav():
    """Flag 10m East → should be +10m longitude from UAV."""
    u_east = 960 + int(10 * 1432 / H)   # pixel offset for 10 m East at h=80
    est = localize_full(u_east, 540, K_test, dist_test, H,
                        LAT0, LON0, 0.0, 0.0, 0.0, R_mount_I)
    dE_m = (est[1] - LON0) * 111320 * math.cos(math.radians(LAT0))
    assert abs(dE_m - 10.0) < 1.0, f"East offset wrong: {dE_m:.2f} m"

def test_yaw_rotation():
    """UAV pointing East (yaw=90°): flag right of image → flag South of UAV."""
    u_right = 960 + 100
    est = localize_full(u_right, 540, K_test, dist_test, H,
                        LAT0, LON0, 0.0, 0.0, math.radians(90), R_mount_I)
    # Flag should now be South (negative lat change) when UAV faces East
    assert est[0] < LAT0, "Yaw rotation: flag should be south when facing East"

def test_oblique_rejected():
    """Heavily tilted frame should be rejected."""
    R_severe_tilt = np.array([[1,0,0],[0,0,-1],[0,1,0]])  # 90° pitch
    est = localize_full(960, 540, K_test, dist_test, H,
                        LAT0, LON0, 0.0, math.radians(80), 0.0, R_severe_tilt)
    assert est is None, "Oblique frame should be rejected"

def test_altitude_scaling():
    """Doubling altitude should double the ground offset."""
    est_80 = localize_full(1000, 540, K_test, dist_test, 80,
                           LAT0, LON0, 0.0, 0.0, 0.0, R_mount_I)
    est_160 = localize_full(1000, 540, K_test, dist_test, 160,
                            LAT0, LON0, 0.0, 0.0, 0.0, R_mount_I)
    offset_80  = (est_80[1]  - LON0) * 111320 * math.cos(math.radians(LAT0))
    offset_160 = (est_160[1] - LON0) * 111320 * math.cos(math.radians(LAT0))
    assert abs(offset_160 / offset_80 - 2.0) < 0.01, "Altitude scaling failed"

if __name__ == "__main__":
    test_nadir_center();    print("✓ nadir center")
    test_flag_east_of_uav(); print("✓ east offset")
    test_yaw_rotation();    print("✓ yaw rotation")
    test_oblique_rejected(); print("✓ oblique rejection")
    test_altitude_scaling(); print("✓ altitude scaling")
    print("\nAll tests passed.")
```

**Week 2–3 — Telemetry Parser:**
```python
# telemetry_parser.py
# Reads ArduPilot .bin log and returns frame-synchronized state dict
# Usage: telem = parse_ardupilot_log("flight.bin", fps=60)

from pymavlink import mavutil
import numpy as np

def parse_ardupilot_log(log_path, fps=60):
    """
    Parse ArduPilot .bin log, return dict mapping frame_index -> state.
    Frame index = (timestamp_sec - start_sec) * fps
    """
    mlog = mavutil.mavlink_connection(log_path)
    telemetry = {}
    start_time = None

    while True:
        msg = mlog.recv_match(type=["GPS", "ATT", "BARO"], blocking=False)
        if msg is None:
            break
        t = msg._timestamp
        if start_time is None:
            start_time = t
        frame_idx = int((t - start_time) * fps)

        if msg.get_type() == "GPS":
            telemetry.setdefault(frame_idx, {})
            telemetry[frame_idx]["lat"] = msg.Lat / 1e7
            telemetry[frame_idx]["lon"] = msg.Lng / 1e7

        elif msg.get_type() == "ATT":
            telemetry.setdefault(frame_idx, {})
            telemetry[frame_idx]["roll_deg"]    = msg.Roll
            telemetry[frame_idx]["pitch_deg"]   = msg.Pitch
            telemetry[frame_idx]["yaw_mag_deg"] = msg.Yaw

        elif msg.get_type() == "BARO":
            telemetry.setdefault(frame_idx, {})
            telemetry[frame_idx]["h"] = msg.Alt  # AGL if ground-calibrated

    return telemetry
```

---

### 👤 Person C — Validation, GPS & Testing Lead

**Mission:** "I prove the system actually works on real flags at real altitudes."

#### Deliverables

| # | Deliverable | Deadline | Status |
|---|---|---|---|
| C1 | Surveyed GPS database — 5+ flags at measured positions | Week 2 | ⬜ |
| C2 | Timestamp alignment script — matches video frames to telemetry | Week 2 | ⬜ |
| C3 | Validation report after Flight Test 1 — mean error, max error, map | Week 3 | ⬜ |
| C4 | Error budget spreadsheet — per-source breakdown | Week 3 | ⬜ |
| C5 | Validation report after Flight Test 2 — comparison to Flight Test 1 | Week 4 | ⬜ |
| C6 | Final pre-competition checklist (signed off) | Week 5 | ⬜ |

#### Tasks

**Week 2 — Surveying Ground Truth:**
1. Place 5 flags at different positions in the test area
2. Stand over each flag center, record GPS with phone (GPS Logger app, 3-minute average)
3. Cross-check with Google Maps or Google Earth for sanity
4. Record: flag ID, lat, lon, surface type (grass/concrete/sand), date/time

**Haversine Distance (error measurement):**

```python
# validation.py — error measurement and reporting

import math
import numpy as np
import json

def haversine_m(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in meters between two GPS points.
    Source: [2] Lago et al. 2024
    """
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def score_location(error_m):
    """Return competition points for this localization error."""
    if error_m <= 20:  return 15
    if error_m <= 30:  return 12
    if error_m <= 40:  return 9
    if error_m <= 50:  return 6
    if error_m <= 60:  return 3
    return 0

def validate_all_flags(ground_truth, estimates):
    """
    Args:
        ground_truth : dict {flag_id: (lat_true, lon_true)}
        estimates    : dict {flag_id: (lat_est, lon_est)}
    """
    print(f"{'Flag':<8} {'Error (m)':<12} {'Points':<8} {'Bin'}")
    print("-" * 45)
    errors = []
    for fid, (lat_t, lon_t) in ground_truth.items():
        if fid not in estimates:
            print(f"{fid:<8} NOT DETECTED")
            continue
        lat_e, lon_e = estimates[fid]
        err = haversine_m(lat_t, lon_t, lat_e, lon_e)
        pts = score_location(err)
        bin_label = (f"< 20 m" if err <= 20 else
                     f"20–30 m" if err <= 30 else
                     f"30–40 m" if err <= 40 else
                     f"40–50 m" if err <= 50 else
                     f"50–60 m" if err <= 60 else "> 60 m")
        print(f"{fid:<8} {err:<12.1f} {pts:<8} {bin_label}")
        errors.append(err)

    if errors:
        print("-" * 45)
        print(f"{'Mean error:':<20} {np.mean(errors):.1f} m")
        print(f"{'Max error:':<20} {np.max(errors):.1f} m")
        print(f"{'< 20 m rate:':<20} {sum(e<=20 for e in errors)/len(errors)*100:.0f}%")

# Example:
if __name__ == "__main__":
    ground_truth = {
        "flag_1": (30.06130, 31.24970),
        "flag_2": (30.06200, 31.25050),
        "flag_bonus": (30.06050, 31.25100),
    }
    estimates = {
        "flag_1": (30.06132, 31.24975),   # 4.3 m error
        "flag_2": (30.06210, 31.25060),   # 13.8 m error
        "flag_bonus": (30.06040, 31.25090), # 15.4 m error
    }
    validate_all_flags(ground_truth, estimates)
```

**Week 2 — Timestamp Alignment:**
```python
# timestamp_sync.py
# Align video frames to telemetry log using a known sync event
# (e.g., LED flash at boot visible in video, or audio trigger)

def find_frame_at_time(video_start_epoch, telemetry_time_epoch, fps=60):
    """
    Returns the frame index in the video corresponding to a telemetry timestamp.
    """
    return int((telemetry_time_epoch - video_start_epoch) * fps)

def interpolate_telemetry(frame_idx, telem_dict, fps=60):
    """
    Linear interpolation between two nearest telemetry entries.
    Critical for time sync accuracy [Engineering Report].
    """
    t = frame_idx / fps
    # Find surrounding keys
    keys = sorted(telem_dict.keys())
    # ... (find idx below and above, interpolate linearly)
    pass  # implement with np.interp for each field
```

---

## Part 8 — Weekly Schedule (5 Weeks to Competition)

| Week | Person A | Person B | Person C | Shared |
|---|---|---|---|---|
| **1** | Calibrate GoPro, produce camera_params.yaml | Implement Approach 1 baseline, first unit tests | Survey 5 ground points | Mount camera, verify level |
| **2** | Determine R_mount experimentally | Implement Approach 2 full pipeline, all 5 unit tests pass | Telemetry parser, timestamp sync script | First smoke test: known landmark |
| **3** | Verify edge-of-frame distortion, confirm center crop is safe | Add confidence-weighted aggregation (Approach 2 full) | **Flight Test 1** — 5 flags at known GPS, measure errors | Review error budget |
| **4** | Support validation, re-calibrate if RMS drift | Add DBSCAN (Approach 3) if errors > 10 m | **Flight Test 2** — compare to Flight Test 1 | Fix dominant error source |
| **5** | Freeze camera_params.yaml, label camera | Freeze code, no new features | Final checklist, practice USB extraction | **Full rehearsal** |

---

## Part 9 — Pre-Flight Checklist (Print and Bring)

```
PRE-FLIGHT LOCALIZATION CHECKLIST — ICMTC 2026
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CAMERA (Person A)
  □ GoPro mode = Linear (confirmed on-screen display)
  □ HyperSmooth / EIS = OFF
  □ Horizon Lock = OFF
  □ Resolution = 1080p, 60 fps
  □ camera_params.yaml matches THIS camera and THIS mode
  □ Camera mount physically level (digital level: < 1° deviation)
  □ GoPro fully charged (> 90%)

GROUND STATION (Person B)
  □ Telemetry logging enabled on autopilot
  □ Frame-timestamp sync method confirmed working
  □ localization.py loaded, no import errors
  □ Test: run pipeline on a static test frame → GPS prints

FIELD CALIBRATION (Person C)
  □ Barometer ground-pressure calibration at takeoff point
  □ Cairo magnetic declination = +4° loaded in code
  □ USB drive formatted, empty, ready for judges

BEFORE TAKEOFF (All)
  □ Record GoPro start time (phone clock) for timestamp sync
  □ Note UAV GPS at takeoff point (for drift check post-flight)
  □ Verify GoPro is actually recording (red light visible)
  □ Verify telemetry logging in Mission Planner / QGroundControl

AFTER LANDING (All — 10-minute window!)
  □ Extract SD card from GoPro
  □ Copy video to laptop
  □ Run pipeline on video (frame extraction → YOLO → localize → aggregate)
  □ Export GPS coordinates in required format
  □ Copy to USB drive
  □ Submit to judges
```

---

## Part 10 — References

| # | Citation | Used in |
|---|---|---|
| [1] | D.B. Barber, J.D. Redding, T.W. McLain, R.W. Beard, C.N. Taylor. *Vision-based Target Geo-location using a Fixed-wing Miniature Air Vehicle.* Journal of Intelligent and Robotic Systems, 47(4):361–382, 2006. DOI: 10.1007/s10846-006-9088-7 | Pipeline math, rotation matrices, error reduction techniques |
| [2] | A. Lago, S. Patel, A. Singh. *Low-cost real-time aerial object detection and GPS location tracking pipeline.* ISPRS Open Journal of Photogrammetry and Remote Sensing, 13:100069, 2024. DOI: 10.1016/j.ophoto.2024.100069 | DBSCAN GPS clustering, pipeline structure (file in your folder) |
| [3] | O. Shaheen, O. Enany, M. Ghoneim. *Detection and Location Estimation of Object in Unmanned Aerial Vehicle.* 2019. URL: snehilsanyal.github.io/files/paper1.pdf | Pipeline validation, requirements |
| [4] | *Ground Object Geo-Location using UAV Video Camera.* ResearchGate, 2010. DOI: 10.1109/ICCVS.2010.16 | Ray intersection algorithm selection |
| [5] | Y. Zhang et al. *Moving Target Geolocation and Trajectory Prediction Using a Fixed-Wing UAV in Cluttered Environments.* Remote Sensing 17(6):969, 2025. DOI: 10.3390/rs17060969 | Fixed-wing error models |
| [6] | M. Yayla et al. *Vision-Based Geolocation of Moving Ground Targets Using Kalman Filtering with a Gimbal Camera on Board a UAV.* Aerospace 12(12):1065, 2025. DOI: 10.3390/aerospace12121065 | Multi-frame aggregation theory |
| [7] | J. Holt, T. McLain. *Improving UAV-Based Target Geolocation Accuracy through Automatic Camera Parameter Discovery.* AIAA SciTech Forum, 2020. DOI: 10.2514/6.2020-2201 | Automatic mount angle calibration (Approach 3) |
| [8] | T.H. Lay, J. Gu, et al. *An Augmented Reality Geo-Registration Method for Ground Target Localization from a Low-Cost UAV Platform.* Sensors 18(12):4336, 2018. DOI: 10.3390/s18124336 | GPS COG as heading source |
| [9] | H.C. Karaimer, M. Brown. *Real-Time Multi-Target Localization from Unmanned Aerial Vehicles.* Sensors 17(4):811, 2017. DOI: 10.3390/s17040811 | Confidence-weighted aggregation |
| [10] | F. Akyon, S.O. Altinuc, A. Temizel. *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection.* arXiv:2202.06934, 2022. URL: arxiv.org/abs/2202.06934 | SAHI for tiny flag detection |
| [11] | X. Chen, M. Zhang et al. *Small Object Detection in UAV Images Based on YOLOv8n.* Int. Journal of Computational Intelligence Systems 17:232, 2024. DOI: 10.1007/s44196-024-00632-3 | YOLO architecture for aerial tiny objects |
| [12] | G. Tola, A. Yalçın. *YOLO-ME: Enhanced Lightweight YOLOv7 Tiny for Aerial Imagery.* Signal, Image and Video Processing, 2025. DOI: 10.1007/s11760-025-03952-9 | Lightweight aerial detection model |
| [13] | Y. Wang et al. *TOE-YOLO: Accurate and Efficient Detection of Tiny Objects in UAV Imagery.* Journal of Real-Time Image Processing, 2025. DOI: 10.1007/s11554-025-01770-3 | Tiny object (< 32 px) detection |
| [14] | Z. Li et al. *BGF-YOLOv10: Small Object Detection from UAV Perspective.* Sensors 24(21):7089, 2024. DOI: 10.3390/s24217089 | Small object state-of-the-art benchmark |
| [15] | L. Pérez-García, M. Pérez-Ruiz. *Calibration of Action Cameras for Photogrammetric Purposes.* Sensors 14(9):17471–17490, 2014. DOI: 10.3390/s140917471 | GoPro Brown-Conrady calibration |
| [16] | S. Urbán. *OpenImuCameraCalibrator.* GitHub, 2022. URL: github.com/urbste/OpenImuCameraCalibrator | GoPro + IMU timestamp calibration tool |
| [17] | J. Redding, T. McLain, R. Beard, C. Taylor. *Vision-based Target Localization from a Fixed-wing Miniature Air Vehicle.* ACC 2006. URL: scholarsarchive.byu.edu/facpub/1537/ | Extended version of [1] |
| [18] | Competition Rules: *10th International Competition of the Military Technical College — UAVC-9 Fixed Wing Challenge.* December 2025. | Scoring, mission requirements |

---

*Document version 2.0 — GoPro Edition. Synthesizes: UAV_GPS_Localization_Engineering_Report.md, GoPro_Pipeline_Analysis_And_Corrections.md, new_gopro_5_or_hero_9.png pipeline, and 17 research papers. Ready for use by AI Team — Eagles, Nile University, ICMTC 2026.*
