# UAV GPS Localization Engineering Report
## ICMTC 2026 — Fixed Wing Challenge (UAVC-9)
### Aerial Flag Detection → Pixel → Ground → GPS Pipeline

**Team Size:** 3 engineers
**Mission Target:** Mission 1 (Rapid Assist) — Flag location & identification scoring (50 pts)
**Operational Envelope:** 50–100 m AGL, downward-looking camera, flat airport terrain
**Scoring Bins:** 20 / 30 / 40 / 50 / 60 m from target center

---

## 1. Problem Statement

### 1.1 The Core Problem

The mission requires the UAV to convert a **detected pixel coordinate** of a flag in a video frame into a **GPS coordinate (latitude, longitude)** that will be evaluated against the judges' ground truth. The full chain:

```
Video frame  →  YOLO bbox  →  bbox center (px, py)
            →  undistortion
            →  normalized camera ray
            →  ray–ground intersection (using altitude)
            →  ground offset (X, Y) in meters
            →  rotation into world frame (using UAV heading)
            →  GPS offset (Δlat, Δlon)
            →  absolute GPS (lat_flag, lon_flag)
```

### 1.2 Inputs Available at Inference Time

| Input | Source | Quality |
|---|---|---|
| Pixel coordinates (px, py) | YOLO detector on aerial frame | Sub-pixel from bbox center; jitter ±2–10 px |
| UAV GPS (lat, lon) | Onboard GPS / autopilot telemetry | ~2–5 m CEP (consumer GPS) |
| UAV altitude AGL (h) | Barometer / GPS / rangefinder | ±1–5 m depending on source |
| UAV attitude (roll, pitch, yaw) | IMU / autopilot | ±0.5–2° typical |
| Camera intrinsics (K, distortion) | Pre-flight calibration | Static, recomputed per camera |
| Frame timestamp | Recording / telemetry | Sync error 50–500 ms possible |

### 1.3 Required Output

For each detected flag:
- **GPS coordinates** (lat, lon) — submitted via USB to judges
- Accuracy bins: **< 20 m** (full points), 20–30, 30–40, 40–50, 50–60, >60 (zero)

### 1.4 Why This Problem Is Hard

**Aerial GPS localization is geometry amplified by altitude.** Errors that look tiny in the image become large on the ground:

| Error source | Image-space size | Ground error @ 80 m |
|---|---|---|
| 1 pixel bbox jitter | 1 px | ~8–15 cm (fine) |
| 0.5° camera tilt | (cosine effect) | **~0.7 m** |
| 2° unmodeled tilt | — | **~2.8 m** |
| 5 m altitude error | — | scales offset by ±6% (≈0.6 m at 10 m offset) |
| 1° heading error | — | ~1.7 cm/m of ground offset → 0.5 m at 30 m |
| GPS noise (UAV) | — | **2–5 m baseline** (additive to everything) |

The brutal arithmetic: **at 80 m altitude with a 60° FOV, 1 pixel ≈ 8–10 cm on the ground** for a 1080p frame, but **a 1° tilt error displaces the target by ~1.4 m**. Calibration and attitude dominate; pixel accuracy is rarely the bottleneck once detection works.

### 1.5 Why Altitude and Calibration Matter

- The forward model is `X_ground = (x_pixel − cx) × h / fx`. Every meter of ground offset is **linearly proportional to altitude**. If altitude is wrong by 10%, every localization is wrong by 10%.
- The intrinsic parameters (`fx, fy, cx, cy`) and distortion coefficients determine whether the pixel-to-ray mapping is correct. **An uncalibrated GoPro at the wide setting can have 5–10% radial distortion at the image edge — enough by itself to move a target by 5+ meters.**

---

## 2. Current Localization Pipeline

### 2.1 Existing Baseline (per `plan.md`)

```
image → undistortion → YOLO → bbox center → pixel normalization
      → altitude scaling → GPS conversion
```

Current equations (reproduced):

| Step | Equation |
|---|---|
| Bbox center | `cx_box = (x1+x2)/2`, `cy_box = (y1+y2)/2` |
| Pixel offset | `dx = cx_box − cx`, `dy = cy_box − cy` |
| Normalize | `x_n = dx/fx`, `y_n = dy/fy` |
| Ground offset | `X = x_n · h`, `Y = −y_n · h` |
| GPS delta | `Δlat = Y/111000`, `Δlon = X/(111000·cos(lat))` |
| Final | `lat_obj = lat_uav + Δlat`, `lon_obj = lon_uav + Δlon` |

### 2.2 What This Model Assumes (Implicitly)

1. The optical axis is **exactly vertical** (nadir-pointing).
2. The camera principal point is correctly known (`cx, cy`).
3. Focal lengths `fx, fy` are accurate.
4. The image is **fully undistorted** before pixel measurement.
5. UAV heading is aligned with image axes (camera +X = world East, camera +Y = world North) — **this is almost never true in flight**.
6. The ground is a flat plane at `z = 0`.
7. Altitude `h` is the true height above the ground at the flag's location.

### 2.3 What Is Naive

| Issue | Impact |
|---|---|
| **No heading rotation** | Image axes ≠ world axes. A flag detected at the +X side of the image is NOT necessarily East of the UAV. Error: up to 100% of the offset magnitude. **Critical bug.** |
| **No pitch/roll compensation** | A 3° pitch from cruise = ~4 m ground shift at 80 m even before any detection. |
| **`111000` constant** | OK to ~0.5% in Egypt (~111,320 m/° lat actually). Negligible. |
| **Flat-earth in longitude** | OK for distances < few km. Negligible. |
| **`X = x_n·h` only valid for nadir camera** | This is the "pinhole + flat ground + zero tilt" shortcut. Acceptable as a baseline IF tilt is genuinely small. |

### 2.4 What Is Practically Acceptable

- The flat-earth GPS conversion (`/111000`) is fine for competition-scale distances.
- Pinhole + nadir is a reasonable **baseline** if the autopilot can hold pitch within ±2° and the camera is rigidly mounted level.
- Bounding-box center is acceptable for a 1×2 m flag at ~80 m (flag occupies ~40–60 px in width on a 1080p GoPro at 90° FOV — usable).

### 2.5 What Will Fail in Real Flight

1. **Heading rotation missing** → flags will be reported in the wrong direction. This alone can push errors past 60 m.
2. **Camera mount not truly nadir** → systematic bias. A 5° fixed tilt = constant ~7 m offset error.
3. **GoPro wide-angle uncorrected** → edge-of-frame flags localize 5–15 m off.
4. **Altitude from GPS instead of barometer/AGL** → GPS altitude noise is 2–3× horizontal noise; can be 10+ m off.
5. **Detection on a motion-blurred frame** → bbox center jitter increases.

---

## 3. Localization Mathematics & Geometry

The recommended pipeline replaces the naive shortcut with a proper ray–plane intersection. Each stage below is a separable module that one team member can own.

### Stage 1 — Camera Calibration (Intrinsics + Distortion)

**Purpose:** Recover `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]` and distortion coefficients (`k1, k2, p1, p2, k3` for pinhole-radial, or fisheye `k1..k4`).

**Method:** OpenCV checkerboard calibration (`cv2.calibrateCamera` or `cv2.fisheye.calibrate`).

**Failure modes:**
- Too few images / poor coverage → unstable `fx`, drifts 5–10%.
- All images at similar distance → `fx`/`fy` couple with depth → bad reprojection.
- GoPro "Linear" mode vs "Wide" mode have **completely different K** — recalibrate each one.

**Implementation cost:** Low (1 afternoon). **Highest leverage step in the entire pipeline.**

---

### Stage 2 — Distortion Correction (Undistortion)

**Purpose:** Convert raw pixel `(u_raw, v_raw)` into the equivalent pinhole pixel `(u, v)`.

For pinhole-radial (GoPro Linear): `cv2.undistortPoints(pts, K, dist, P=K_new)`.
For fisheye: `cv2.fisheye.undistortPoints(...)`.

**Important practical note:** You do **not** need to undistort the full image for localization — undistort **only the bbox center pixel**. This is ~1000× cheaper and avoids resampling artifacts.

**Failure modes:**
- Applying pinhole undistortion to fisheye data → large residual at edges.
- Forgetting that GoPro often applies in-camera Hyperlapse / Hypersmooth stabilization → effective intrinsics change frame-to-frame.

---

### Stage 3 — Pixel → Normalized Camera Coordinates

After undistortion:

```
x_n = (u − cx) / fx
y_n = (v − cy) / fy
```

This defines the **ray direction in the camera frame**:

```
r_cam = [x_n, y_n, 1]^T   (unnormalized, pointing along camera +Z which is "out of the lens")
```

**Intuition:** `(x_n, y_n)` is "how many focal-lengths off-axis is this pixel." Multiplied by depth, you get metric offset.

---

### Stage 4 — Rotate Ray Into World Frame

This is the step **missing from the current baseline**. Build a rotation matrix from UAV attitude (roll φ, pitch θ, yaw ψ) and the camera mount rotation `R_mount` (usually identity for nadir mount, or a 90° flip depending on how you define "camera-down").

```
R_world_from_cam = R_yaw(ψ) · R_pitch(θ) · R_roll(φ) · R_mount
r_world = R_world_from_cam · r_cam
```

For a strict downward camera with image +X = aircraft forward, image +Y = aircraft right:

```
R_mount = [[1,0,0],[0,1,0],[0,0,1]]   (with z pointing down to ground)
```

**Failure modes:**
- Sign errors on yaw (East-of-North vs. North-of-East). **Test with a known landmark before the competition.**
- Confusing intrinsic vs extrinsic Euler conventions — pick ZYX (yaw–pitch–roll) and stick with it.
- Using magnetic heading vs true heading without correction (~4° magnetic declination in Cairo).

---

### Stage 5 — Ray–Ground Intersection

The world ray from the UAV position `P_uav = (0, 0, h)` (camera-centered local NED) with direction `r_world = (rx, ry, rz)` hits the ground `z = 0` at parameter `t = h / rz`:

```
X = t · rx        (East offset, meters)
Y = t · ry        (North offset, meters)
```

For a **truly nadir** camera with **zero attitude**, this collapses to `X = x_n·h, Y = y_n·h`, recovering the baseline equation — but now the formulation is correct for **any** attitude.

**Failure cases:**
- `rz` close to zero → ray nearly parallel to ground → numerically explodes. Reject any detection where the ray pitch from vertical exceeds ~45° (fast sanity guard).
- Ground not flat → use a DEM or accept residual error (negligible at airport).

---

### Stage 6 — Ground Offset → GPS

Convert local NED (North, East) offsets to GPS:

```
Δlat = N / R_earth        where R_earth ≈ 6 378 137 m → 1° ≈ 111 320 m
Δlon = E / (R_earth · cos(lat_uav))
lat_flag = lat_uav + Δlat · (180/π)
lon_flag = lon_uav + Δlon · (180/π)
```

Equivalent simplified form (sufficient for <1 km):

```
lat_flag = lat_uav + (N / 111320)
lon_flag = lon_uav + (E / (111320 · cos(lat_uav · π/180)))
```

**Cairo latitude ~30°N → cos(lat) ≈ 0.866 → 1° lon ≈ 96 km.**

For higher fidelity (negligible benefit at this scale): use `pyproj` to transform local ENU → WGS84.

---

## 4. Coordinate Systems Explained

There are **four** systems and confusing two of them is the #1 source of "we're 40 m off and don't know why" bugs.

### 4.1 Image Coordinate System (pixels)
- Origin: **top-left** corner of the image.
- +u = right, +v = **down** (note the Y inversion — this is why the baseline has `Y = −y_n·h`).
- Units: pixels.

### 4.2 Camera Coordinate System (meters, scaled)
- Origin: optical center.
- +X = right (image +u), +Y = down (image +v), +Z = out of lens (forward into scene).
- This is the OpenCV convention. Right-handed.

### 4.3 Body / World Local Frame (meters, NED or ENU)
- Origin: UAV current position.
- NED: +X = North, +Y = East, +Z = **down**.
- ENU: +X = East, +Y = North, +Z = up.
- Pick **one** and document it. **Mixing NED and ENU is the most common bug.**

### 4.4 GPS / Geodetic Frame (WGS84)
- Latitude, longitude (degrees), altitude (m above ellipsoid or above MSL — also a source of confusion).

### 4.5 Y-Axis Inversion (The Pixel Trap)

Image +v points **down** but world North points **up** on a map. That's why the naive baseline has the negative sign:

```
Y_ground = −y_n · h    (image-down → world-South sign flip)
```

Once you do the full attitude rotation in Stage 4, this sign flip is handled automatically by `R_mount`. **Don't apply it twice.**

### 4.6 Why `f / h` Is a Useful Number

The ratio **focal length / altitude** is the "image scale":
- At 80 m altitude, GoPro Linear ~ 1500 px focal length → image scale = 1500/80 = **18.75 px/m**.
- A 1×2 m flag covers ~19 × 38 px. **This sets a hard floor on detection: below ~5 px the flag becomes invisible to YOLO.**

This means the practically usable altitude band is the one where flags are ≥10 px on the long side: at GoPro Linear with f≈1500 px, that's < 200 m. Comfortable margin.

### 4.7 Meaning of X/Y Ground Offsets

`(X, Y) = (East, North)` displacement of the flag relative to the UAV at the instant the frame was captured. **Always tied to the UAV's GPS at that frame's timestamp.** Frame-to-UAV-position synchronization is therefore critical (see error analysis).

---

## 5. Localization Error Analysis

### 5.1 Error Budget Table (at h = 80 m, flag at 20 m horizontal offset)

| Source | Typical magnitude | Propagated ground error | Severity | Mitigation |
|---|---|---|---|---|
| Camera calibration error (fx) | 2% | ~0.4 m | Low | Good checkerboard pass |
| Residual fisheye distortion (edge) | 1–3% radial | 0.2–0.6 m | Low–Med | Use fisheye model or crop center 60% |
| Bbox center jitter | ±3 px | ±0.15 m | Negligible | Median over 3 frames |
| **UAV GPS noise** | ±3 m CEP | **±3 m** | **HIGH** | Multi-frame averaging; RTK if available |
| **Altitude error** | ±5 m | ±1.25 m (6% × 20 m) | Med | Use barometer + ground-pressure cal |
| **Camera tilt (roll/pitch)** | 2° | ~2.8 m | **HIGH** | IMU compensation OR fly straight-and-level passes |
| **Yaw / heading error** | 2° | ~0.7 m at 20 m offset | Med | Use GPS course-over-ground when moving |
| Time sync (GPS↔frame) | 200 ms @ 20 m/s | **4 m** | **HIGH** | Log frame timestamps; align via cross-correlation |
| Motion blur | — | bbox jitter +5–10 px | Low–Med | Higher shutter speed; reject blurry frames |
| GoPro EIS (Hypersmooth) | unknown crop & K shift | up to 5 m | **HIGH** | **Disable EIS** for localization runs |

**Total RSS (root-sum-square) typical:** ~5–7 m if everything is reasonable. ~15–25 m if heading/tilt/EIS unaddressed.

### 5.2 Error Propagation Intuition

The pipeline is locally linear, so errors RSS-combine. The two dominant terms in practice are:
1. **UAV GPS noise** (you can't reduce below ~2 m without RTK).
2. **Attitude × altitude product** — a 2° tilt at 80 m is bigger than any other geometric error.

**Implication:** Closing the gap from "good calibration" to "good localization" is overwhelmingly about **attitude logging and time sync**, not better detection or better fx.

### 5.3 The Multi-Frame Averaging Win

If the same flag is detected in N independent frames and the per-frame error has σ ≈ 5 m (mostly GPS noise), averaging gives σ/√N. Ten frames → ~1.6 m. **This is the single highest-leverage mitigation in the whole system.**

---

## 6. Camera Transition: Raspberry Pi Fisheye → GoPro

### 6.1 What Changes Mathematically

| Property | RPi Camera (OV5647 + fisheye) | GoPro (e.g. Hero 10/11/12) |
|---|---|---|
| Distortion model | Fisheye (`cv2.fisheye`, equidistant) | Pinhole-radial (in Linear mode) or fisheye (in Wide) |
| Focal length (px @ 1080p) | ~500–800 (very wide) | ~900 (Wide) / ~1500 (Linear) / ~2200 (Narrow) |
| Principal point | Near center | Near center (GoPro is reasonably symmetric) |
| FOV | ~200° advertised, usable ~150° | 122° (Wide), 92° (Linear), 73° (Narrow) |
| Rolling shutter | Severe (slow readout) | Moderate (fast but present) |
| In-camera processing | None | **Hypersmooth, Horizon Lock, lens correction** — all change effective intrinsics |

### 6.2 What Must Be Recalibrated

**Everything.** Specifically:
- New `K` matrix (recompute fx, fy, cx, cy).
- New distortion model — likely pinhole-radial if using GoPro "Linear" output.
- New FOV → new ground footprint at altitude → new pixel-per-meter scale.
- The flat-ground, nadir-camera assumption can stay if you re-verify the mount.

### 6.3 What Assumptions Remain Valid

- Flat terrain (airport).
- Single ground plane.
- Downward-facing intent.
- Pinhole approximation (GoPro Linear mode is well-behaved within ~30° from center).

### 6.4 Operational Risks (GoPro-Specific)

| Risk | Why it matters | Mitigation |
|---|---|---|
| **HyperSmooth / EIS** enabled | Crops the frame and warps it dynamically → intrinsics not constant | **Disable EIS for localization flights.** This is critical and free. |
| **Horizon Lock** | Rotates frame to keep horizon level → image axes no longer match UAV body axes | **Disable.** |
| **Lens setting** changed between calibration and flight | Linear vs Wide vs SuperView are different cameras | Mark the setting; verify with a printed reference before takeoff |
| **Rolling shutter** at airspeed | Each row read at different time → distortion when flying fast | Use highest available shutter / lowest exposure; localize from frames during slow legs |
| **Wide-angle distortion residual** | Edge pixels still have 1–2% residual after pinhole correction | **Localize only flags in the central 60–70% of the frame** if possible (re-fly if needed) |
| Battery hot-swap mid-mission | GoPro might re-apply default settings | Lock settings; verify post-swap |

### 6.5 Stabilization Effects (subtle but important)

GoPro stabilization works by digitally rotating/cropping the image. The **mapping from sensor pixel → output pixel changes per frame**. You cannot calibrate this. **Disable EIS** for the localization mission.

---

## 7. Practical Localization Strategy

### 7.1 Recommended Approach (Ranked)

**Baseline (must-have, 1 week of work):**
1. Pinhole + flat-ground + IMU-compensated attitude rotation.
2. Multi-frame averaging (≥5 frames per flag).
3. Disable GoPro EIS / Horizon Lock.
4. GoPro Linear mode (not Wide), localize from central 70% of frame.

**Advanced (only if baseline error > 15 m):**
- Ray–plane intersection with full rotation (we already include this above).
- IMU bias estimation.
- Rolling-shutter compensation per row (complex; rarely worth it at our altitudes).

### 7.2 Acceptable Assumptions

| Assumption | Acceptable? | Rationale |
|---|---|---|
| Flat ground | YES | Airport is flat. |
| Constant altitude during a single detection burst | YES | UAV holds altitude ±2 m typically. |
| `cos(lat)` constant within mission | YES | 1 km extent at lat 30° |
| Nadir camera | **NO without IMU rotation** — but acceptable IF mechanical mount is verified vertical and flight is straight-and-level over the search zone |
| No wind-induced tilt | NO | The autopilot will pitch up to maintain altitude against wind. Compensate via IMU. |

### 7.3 What Should Be Implemented First

```
Week 1: GoPro calibration + simple pinhole undistortion + naive baseline
Week 2: Add multi-frame averaging + sanity tests against known ground points
Week 3: Add IMU attitude rotation
Week 4: Validation flights, error budget measurement, tuning
```

### 7.4 What Is Likely Overengineering

- **RTK GPS** unless you already own one — 2–3 m CEP from consumer GPS is fine for ≤20 m scoring bin.
- **Kalman filtering** of GPS — not enough independent observations per flag to justify it.
- **Bundle adjustment / SfM** — vastly more complex than scoring requires.
- **Per-frame rolling-shutter correction** — savings < 1 m at our airspeeds.
- **Online learning of distortion** — calibrate offline; it doesn't change in-flight.

### 7.5 Minimum Viable vs Advanced

| Capability | MVP | Advanced |
|---|---|---|
| Undistortion | Pinhole + radial | Fisheye explicit |
| Attitude | Assumed zero / measured roll-pitch | Full IMU at frame timestamp |
| Altitude | Barometer | Barometer fused with GPS |
| Aggregation | Mean of N frames | Median + outlier reject + per-frame weight |
| Heading | Course-over-ground from GPS | IMU yaw with declination correction |

---

## 8. Localization Experiments

### 8.1 Experiment List Overview

| # | Experiment | Owner | Duration | Output |
|---|---|---|---|---|
| 1 | Checkerboard calibration | Calibration owner | 1 day | `K`, dist coeffs, reprojection RMS |
| 2 | GPS noise benchmark | GPS/validation owner | 1 day | UAV-static CEP @ field site |
| 3 | Altitude sensitivity | Geometry owner | 0.5 day | Error vs altitude curve |
| 4 | Tilt sensitivity | Geometry owner | 0.5 day | Error vs tilt curve |
| 5 | Edge-of-image distortion | Calibration owner | 0.5 day | Residual error vs radial distance |
| 6 | End-to-end repeatability | Validation owner | 1 day | σ of GPS estimate over N passes |

### Experiment 1 — Checkerboard Calibration

- **Objective:** Recover GoPro intrinsics + distortion for the chosen lens mode.
- **Setup:** 9×6 (or 10×7) checkerboard, 30 mm squares, A2 mounted on rigid foamboard. GoPro in chosen mode (Linear, EIS off).
- **Procedure:** Capture 30–50 still images covering full frame area (corners, edges, center), multiple distances (0.5 m, 1 m, 2 m), multiple orientations.
- **Measurement:** OpenCV `calibrateCamera` reprojection RMS error.
- **Expected outcome:** RMS < 0.5 px is excellent, < 1.0 px acceptable. If > 1.5 px, recapture.

### Experiment 2 — GPS Error Benchmarking

- **Objective:** Establish baseline UAV GPS noise floor at the actual flight field.
- **Setup:** UAV stationary on a surveyed point for 5 minutes, logging at 10 Hz.
- **Measurement:** Standard deviation and 95th percentile error of reported lat/lon vs truth.
- **Expected outcome:** 2–5 m CEP for consumer GPS. Use as floor in error budget.

### Experiment 3 — Altitude Sensitivity Test

- **Objective:** Quantify how altitude error propagates to ground error.
- **Setup:** Place flag at known GPS. Fly at 60, 70, 80, 90, 100 m. Detect, localize.
- **Measurement:** Localization error vs altitude.
- **Expected outcome:** Linear scaling with altitude. Catches barometer drift.

### Experiment 4 — Tilt Sensitivity Test

- **Objective:** Quantify tilt-error propagation.
- **Setup:** With UAV held by hand at known GPS+altitude, deliberately tilt camera at 0°, 2°, 5°, 10°. Log IMU and detection.
- **Measurement:** Localization error vs tilt with/without IMU rotation applied.
- **Expected outcome:** Without compensation: ~h·sin(tilt). With compensation: error stays flat.

### Experiment 5 — Edge-of-Image Distortion Analysis

- **Objective:** Determine the safe central region of the frame.
- **Setup:** Lay a precisely measured grid on the ground; fly nadir over it.
- **Measurement:** Reprojection error of each grid intersection vs radial distance from `(cx, cy)`.
- **Expected outcome:** Error grows toward edges. Decide whether to crop or accept.

### Experiment 6 — Localization Repeatability

- **Objective:** End-to-end σ of estimated GPS over multiple passes.
- **Setup:** Flag at surveyed GPS. Fly 5 independent passes at 80 m. Record all per-frame estimates and the averaged result.
- **Measurement:** σ across passes. Bias vs truth.
- **Expected outcome:** With multi-frame averaging, σ < 3 m. Bias indicates systematic error (calibration / mount).

---

## 9. Team Task Distribution (3 People)

Three roles, each with clear deliverables and dependencies. Keep weekly sync to handle the interface between Geometry and Validation.

### Role A — Calibration & Camera Lead

**Responsibilities:**
- Camera calibration (RPi initially, GoPro after transition).
- Maintains `K` and distortion coefficients in a versioned file.
- Tests undistortion correctness.
- Owns the camera mount geometry document.

**Deliverables:**
- `camera_params.yaml` (intrinsics + distortion).
- Undistortion module (`undistort_pixel(u,v,K,dist) → (u',v')`).
- Reprojection-RMS report.

**Dependencies:**
- Hardware: GoPro in hand, fixed lens mode.
- Provides intrinsics to Role B.

### Role B — Geometry & Pipeline Lead

**Responsibilities:**
- Implements the full pixel → GPS pipeline.
- Implements IMU rotation.
- Implements multi-frame aggregation.
- Owns the localization codebase.

**Deliverables:**
- `localize(detection, telemetry) → (lat, lon)` function.
- Module-level unit tests with synthetic ground truth.
- Pipeline runs end-to-end on a sample video.

**Dependencies:**
- Receives `K, dist` from Role A.
- Provides estimated GPS to Role C for validation.

### Role C — Validation, GPS & Experiments Lead

**Responsibilities:**
- Survey ground truth GPS points.
- Runs validation flights.
- Owns error-budget spreadsheet.
- Visualizes results (estimated vs truth on a map).

**Deliverables:**
- Surveyed GT GPS database for test flags.
- Validation report after each flight.
- Per-flag error metric and final scoring estimate.

**Dependencies:**
- Needs working pipeline from Role B.
- Provides telemetry sync / time-alignment scripts.

### Shared:
- **Detection (YOLO)** stays a shared concern but is mostly upstream. The geometry team only needs `(u, v)` per flag per frame.
- **Camera mount design** is a hardware action item owned jointly by A and B.

---

## 10. Localization Roadmap (5 Phases)

### Phase 1 — Simplified Baseline (Week 1–2)
- **Goals:** Pixel → GPS using current naive equations + at least an attempted yaw rotation.
- **Tasks:** Implement Step 1–7 from `plan.md`. Add a yaw rotation step. Smoke-test against a known ground point.
- **Validation:** Single static target, hand-held UAV at known altitude.
- **Risks:** Sign/axis confusion. **Mitigation:** unit test against synthetic ground truth.
- **Exit criteria:** Pipeline returns a GPS within 30 m of truth in static test.

### Phase 2 — Camera Calibration (Week 2–3)
- **Goals:** Replace placeholder K with a real, measured GoPro calibration.
- **Tasks:** Checkerboard capture (Exp 1). Undistort module. Verify reprojection.
- **Validation:** Reprojection RMS < 1 px.
- **Risks:** Calibrated in wrong mode (e.g., Wide instead of Linear). **Mitigation:** physical sticker label on GoPro indicating mode.
- **Exit criteria:** Undistorted grid lines are visibly straight; numerical RMS verified.

### Phase 3 — GPS Validation (Week 3–4)
- **Goals:** Quantify real-world end-to-end error.
- **Tasks:** Surveyed ground points + flight test + comparison.
- **Validation:** ≥3 flags at known GPS, ≥5 passes each.
- **Risks:** Telemetry timestamp drift. **Mitigation:** log frame index + GPS time and check alignment.
- **Exit criteria:** Mean error < 15 m, max error < 30 m without further tuning.

### Phase 4 — Error Reduction (Week 4–5)
- **Goals:** Push mean error to < 8 m.
- **Tasks:** IMU rotation, multi-frame averaging, EIS-off verification, central-crop policy.
- **Validation:** Re-run Phase 3 experiments and compare error distributions.
- **Risks:** IMU convention bug introduces new error. **Mitigation:** unit tests with synthetic attitude.
- **Exit criteria:** ≥80% of detections land in < 20 m bin (full points).

### Phase 5 — Robust Localization (Week 5+)
- **Goals:** Handle adversarial cases — gusts, edge-of-frame detections, partial occlusion.
- **Tasks:** Outlier rejection in multi-frame aggregation; per-frame confidence weighting.
- **Validation:** Test with deliberately tilted flight, edge-of-frame placements.
- **Risks:** Over-engineering. **Mitigation:** stop here if Phase 4 already achieves 20 m bin reliably.
- **Exit criteria:** Robustness validated under expected competition conditions.

---

## 11. Final Engineering Recommendations

### 11.1 What Should Be Validated First

1. **GoPro calibration** with the exact mode used in flight. Print intrinsics on a sticker on the camera.
2. **Camera mount verticality** — measure with a digital level on the bench.
3. **GPS time / frame time alignment** — without this, all downstream work is built on sand.
4. **EIS / Horizon Lock are OFF** — visible in the GoPro settings menu, verified by recording a static scene.

### 11.2 Dangerous Assumptions

| Assumption | Why dangerous |
|---|---|
| "The autopilot keeps the camera nadir" | Wind = constant pitch up. Causes systematic bias. |
| "Heading from autopilot matches image" | Camera-to-body yaw offset is real and must be measured. |
| "GoPro intrinsics are the same across modes" | They differ by ~50%. |
| "Altitude AGL = GPS altitude" | GPS altitude is noisy and references ellipsoid, not ground. Use barometer with ground-pressure calibration. |
| "1 detection = 1 GPS" | Single detections are 5–10 m noisy due to GPS alone. Always aggregate. |

### 11.3 Likely Failure Points

- **Day-of competition surprise**: GoPro firmware update silently re-enables EIS → all calibration invalid. **Mitigation: settings checklist, verify visually before takeoff.**
- **Telemetry log gap during the search zone**: pipeline has no UAV GPS at frame time. **Mitigation: interpolation with strict gap limit (200 ms).**
- **Flag detection at frame edge**: large distortion residual + tilt amplification. **Mitigation: prefer central detections; re-fly the search pattern if needed.**

### 11.4 Acceptable Simplifications

- Pinhole instead of fisheye for GoPro Linear mode → < 1% additional error.
- Constant `cos(lat)` over the mission → negligible.
- Ignore Earth curvature → negligible at 1 km.
- Mean instead of full Kalman → loss of < 10% on σ_final.

### 11.5 Likely Overengineering

- RTK GPS upgrade.
- Real-time SfM.
- Per-frame rolling-shutter row-by-row correction.
- Active gimbal stabilization software in the pipeline.
- Neural network "learned" calibration.

### 11.6 Minimum Viable Localization Pipeline

```python
def localize(u, v, K, dist, h, lat_uav, lon_uav, roll, pitch, yaw):
    # 1. Undistort the single pixel
    u_p, v_p = undistort_pixel(u, v, K, dist)
    # 2. Normalize
    x_n = (u_p - K[0,2]) / K[0,0]
    y_n = (v_p - K[1,2]) / K[1,1]
    # 3. Camera ray
    r_cam = np.array([x_n, y_n, 1.0])
    # 4. Rotate to world
    R = R_yaw(yaw) @ R_pitch(pitch) @ R_roll(roll) @ R_mount
    r_world = R @ r_cam
    # 5. Ray-ground intersection
    t = h / r_world[2]
    E, N = t * r_world[0], t * r_world[1]   # ENU
    # 6. To GPS
    dlat = N / 111320.0
    dlon = E / (111320.0 * math.cos(math.radians(lat_uav)))
    return lat_uav + dlat, lon_uav + dlon

# Aggregate across N frames:
def localize_flag(detections):
    estimates = [localize(*d) for d in detections]
    return np.median(estimates, axis=0)   # outlier-robust
```

That's the whole system. Everything else is either calibration data, telemetry sync, or validation.

### 11.7 Most Practical Competition-Ready Solution

**Hardware:** GoPro in Linear mode, EIS off, rigid nadir mount, verified level.

**Software:** The function above + a GoPro calibration file + IMU/GPS log alignment.

**Operational:** Fly the search zone in straight-and-level passes at 80 m. Detect each flag in 5+ frames. Aggregate. Submit median.

**Expected performance:** Mean error 5–8 m. ≥80% of detections in the 20 m bin = **full flag points (50/50 for inside-area flags, +45 bonus available)**.

This is achievable in 4–5 weeks with the 3-person team and the equipment described in `plan.md`.

---

## Appendix A — Quick Reference Equations

```
Undistorted normalized:    x_n = (u - cx) / fx,   y_n = (v - cy) / fy
Camera ray:                r_cam = [x_n, y_n, 1]
World ray:                 r_world = R_yaw · R_pitch · R_roll · R_mount · r_cam
Ground hit:                t = h / r_world.z;   E = t·r_world.x;   N = t·r_world.y
GPS:                       lat += N / 111320
                           lon += E / (111320 · cos(lat))
```

## Appendix B — Pre-Flight Checklist (Localization-Specific)

- [ ] GoPro mode = Linear (verify on screen)
- [ ] HyperSmooth / EIS = OFF
- [ ] Horizon Lock = OFF
- [ ] Resolution & FPS match calibration
- [ ] Camera mount checked level (digital level on bench)
- [ ] Telemetry recording enabled with frame-sync mark (LED flash visible to camera at boot)
- [ ] Barometer ground-pressure calibration set at takeoff point
- [ ] Magnetic declination loaded for Cairo (~4° E)
- [ ] `K` and distortion file matches the camera mode actually selected
