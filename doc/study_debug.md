# study_debug.md — Build & Debug the Localization Pipeline Yourself

This is the companion to `documentation.md`. The design doc says *what* and *why*; this file is the *how* — written so **you** write the code, not an autocomplete. It gives, per module: the contract (I/O), the math, **pseudocode (not finished code)**, the bugs that will actually bite you, and a test you can run before moving on.

> Working rule: **never integrate a module you haven't unit-tested in isolation.** Geolocation bugs are silent — a wrong sign gives you a plausible-looking coordinate 200 m away, and you won't notice until the leaderboard does. Test each stage against a synthetic case with a known answer.

---

## 0. Mental model & the #1 source of bugs: coordinate frames

Before any code, write these four frames on paper and pin it to your monitor:

| Frame | Axes | Used by |
|---|---|---|
| World ENU | X=East, Y=North, Z=Up | everything downstream |
| Body FRD | X=Forward, Y=Right, Z=Down | Pixhawk attitude |
| Camera optical | X=right, Y=down, Z=forward (out of lens) | OpenCV, `K` |
| Image | u→right, v→down (px) | detector |

**90% of "my coordinate is in the sea" bugs are a frame/sign error here.** Pick ENU (not NED) and never mix. Write the three rotation matrices (`R_world_body`, `R_body_cam`, `R_cam_image`) explicitly and unit-test each with a hand-computed vector.

**Self-test before continuing:** camera level, pointing straight down, pixel at image centre. The world ray must be `[0,0,-1]` (straight down in ENU). If your code says anything else, stop and fix the frames now.

---

## Module [0] — Ingest / Seek

**Contract:** in = `video.mp4, t0, t1, stride`; out = iterator of `(frame_idx, t_video_seconds, image)`.

**Pseudocode**
```
open video
seek to t0                      # decoder-level seek, not read-and-discard
while frame.pts <= t1:
    if (frame_idx - first_idx) % stride == 0:
        yield (frame_idx, pts_to_seconds(frame.pts), frame)
```

**Gotchas**
- `cap.set(CAP_PROP_POS_FRAMES, n)` is unreliable with variable-GOP H.265 — prefer seeking by **time** (`CAP_PROP_POS_MSEC`) or use PyAV/ffmpeg and read the real PTS.
- Don't assume constant fps. Read each frame's actual timestamp; GoPro can drop frames.
- Off-by-one: is `t1` inclusive? Decide and document.

**Test:** seek to a known timestamp where you've noted what's on screen; assert the returned frame shows it. Assert `t_video` is monotonic and within `[t0,t1]`.

---

## Module [1] — Detection

**Contract:** in = image; out = list of `{class, conf, bbox=(x1,y1,x2,y2), anchor=(u,v)}`.

> **Competition reality:** the flags lie **flat on the ground** (no pole), so the anchor is just the **bbox centroid**. No pole-base case.

**Anchor rule:**
```
anchor = ( (x1+x2)/2 , (y1+y2)/2 )      # centroid — flag is flat on the ground
```

**Two-stage detect → classify (recommended).** Keep the YOLO detector coarse (`flag / not-flag`), then **crop** the bbox and run a separate **fine-grained classifier** on the close-up to get the flag's identity (the ~100-class national-flag model). The same crop is what you save for the USB submission — one image, two jobs.

**Gotchas**
- Letterbox/resize: if you pad to a square for YOLO, you must map boxes back to original pixel coordinates **before** the anchor is used by `K`. A very common silent error.
- Confidence threshold trades detections (good for clustering) vs false positives (bad). Keep raw conf; filter later.

**Test:** draw the anchor (centroid) on 10 frames and eyeball it — the dot must sit on the flag. Crop the bbox and confirm the close-up is clean enough for the classifier.

---

## Module [2] — Tracking

**Contract:** in = per-frame detections; out = same + `track_id`.

**Gotchas**
- Tracks fragment on occlusion/re-entry and on a second fly-over — **expected**. Don't try to make the tracker perfect; module [8] (spatial clustering) is what guarantees one-flag-one-result. The tracker just helps and provides nice per-tracklet stats.
- Don't trust `track_id` as the final flag identity.

**Test:** count distinct track_ids on a clip with 1 flag in continuous view — should be ~1. If it's 10, your motion model / IoU threshold is off, but it won't break the pipeline.

---

## Module [3] — Telemetry normalize

**Contract:** in = `.ulog`/`.bin`; out = table indexed by `t_utc` with `lat, lon, alt_hae, alt_msl, agl, roll, pitch, yaw (or quat), v_ned, fix, nsat`.

**Pseudocode**
```
msgs = parse_log(file)                     # pyulog (PX4) or pymavlink (ArduPilot .bin)
pos  = extract(GLOBAL_POSITION_INT / VEHICLE_GPS_POSITION)
att  = extract(ATTITUDE / quaternion)
resample/merge onto a common time grid     # keep native rate where possible
store BOTH alt_hae and alt_msl
write parquet
```

**Gotchas**
- **Altitude fields lie about their datum.** Know exactly which message gives HAE vs AMSL/MSL. PX4 `vehicle_global_position.alt` is AMSL; raw GPS is often HAE. Record both and label them — this feeds the geoid fix in [7].
- Quaternion convention (Hamilton vs JPL, and frame order) varies. Verify by checking a known straight-and-level segment: roll≈pitch≈0.
- Units: cm vs m, deg×1e7 vs deg. MAVLink loves scaled integers.

**Test:** plot alt vs time and roll/pitch/yaw vs time. Straight-and-level cruise must show flat ~0 roll/pitch. A bank you remember must appear with the right sign.

---

## Module [4] — Time sync (the sneaky one)

> **Phase note.** In the **current GoPro-only build this module is essentially free**: the GPMF telemetry and the video frames come from the same device and clock, so `frame_time = frame_idx / fps` already lines up with `time_s` in the telemetry. `sync.py` just interpolates telemetry onto each detection's timestamp — no cross-correlation needed. Everything below applies to **Phase B**, when a Pixhawk adds a second independent clock and you must recover the offset. Build the rest of the pipeline first; come back here when the Pixhawk arrives.

**Contract (Phase B):** in = GoPro GPMF gyro series + Pixhawk gyro/attitude-rate series; out = `(a, b)` for `t_utc = a·t_video + b`.

**Math:** cross-correlate the two angular-rate magnitude signals; the lag at peak correlation = offset `b`. Resample both to a common high rate first.
```
g_cam = |gyro_gpmf|     resampled to 200 Hz
g_pix = |gyro_pixhawk|  resampled to 200 Hz
lag*  = argmax_lag  crosscorr(g_cam, g_pix)
b     = lag* / rate
# fit 'a' (drift) by repeating on early and late windows and fitting a line
```

**Gotchas**
- Do the sync over a **high-motion** window (wing-rocks at launch). Cross-correlation on smooth cruise is ambiguous.
- Sign of the lag — which signal leads? Test with a known shift (below).
- Don't forget EKF/processing latency hides inside `b`; that's fine, the sync absorbs it.

**Test (synthetic):** take one IMU signal, shift it by exactly +120 ms, add noise, recover. Your code must return b ≈ −0.120 (or +0.120 — pin the sign with this test).

**Why it matters (do this arithmetic once):** at 25 m/s, 100 ms = 2.5 m ground error, pure bias on every flag. 10 ms = 0.25 m. This is why GPS-UTC matching alone (≈0.1 s) isn't enough and IMU cross-corr (≈10 ms) is preferred.

---

## Module [5] — Pose interpolation

**Contract:** in = telemetry table + a query `t_utc`; out = `(lat, lon, alt, q_world_body)`.

**Pseudocode**
```
i = searchsorted(times, t)
frac = (t - times[i-1]) / (times[i] - times[i-1])
pos = lerp(pos[i-1], pos[i], frac)                 # linear
q   = slerp(q[i-1], q[i], frac)                    # SLERP, normalized
```

**Gotchas**
- **Never lerp Euler angles.** Convert to quaternion, SLERP, convert back. Lerping across the ±180° yaw wrap gives a wild spin.
- Extrapolation past the table edge is dangerous — clamp or refuse, and that's why [3] keeps a ±1 s margin around the window.

**Test:** query exactly at a table timestamp → must return that row unchanged. Query at the midpoint of a constant-rate yaw sweep → must be the geometric midpoint, no wrap glitch.

---

## Module [6] — Ray build

**Contract:** in = anchor `(u,v)`, pose, calib `K, R_body_cam`; out = world unit ray `d_world` + camera centre `C`.

**Math**
```
d_cam   = normalize( K⁻¹ · [u, v, 1]ᵀ )            # undistort first if not Linear lens
R_wc    = R_world_body(q) · R_body_cam
d_world = R_wc · d_cam
C       = enu(lat,lon,alt) − R_world_body · lever_arm
```

**Gotchas**
- Undistortion: if you fly **Linear**, residual distortion is small but nonzero — still apply your calibrated coeffs for best accuracy. If you (wrongly) fly Wide, you *must* undistort with the fisheye model or everything is garbage.
- `K` must match the **exact resolution** you flew. A `K` calibrated at 4K used on 5.3K footage is silently wrong by the scale ratio.
- Apply the **lever arm** in world frame (rotate the body-frame offset by `R_world_body`).

**Test (the golden one):** synthetic camera at 100 m, pointing straight down, level. Centre pixel ray must be `[0,0,-1]`. Tilt pitch 10° forward → ray must tilt forward (north if yaw=0) by ~10°. Hand-check the angle.

---

## Module [7] — Ground intersection (+ AGL/DEM/geoid)

**Contract:** in = `C, d_world`, terrain source; out = target `(lat, lon, alt)` + 2×2 ENU covariance.

**Flat-plane math**
```
s* = (h_ground − C_up) / d_world_up        # d_world_up < 0 when looking down
P  = C + s* · d_world
```
**DEM iterative**
```
h = mean_field_elev
repeat 3–5×:
    P = intersect_plane(C, d, h)
    h = DEM(P.lat, P.lon)            # in the SAME datum as C's altitude!
return P
```
**Geoid fix (do not skip):**
```
alt_msl = alt_hae − N(lat, lon)     # EGM2008; reconcile GPS(HAE) with DEM(MSL) ONCE
```

**Gotchas**
- **Datum mismatch (HAE vs MSL)** is the headline altitude bug — see documentation.md §3.4. Pick one datum, convert everything into it, unit-test with a known point.
- If `d_world_up ≥ 0` the ray points at/above the horizon → no intersection. Guard it; flag the detection invalid.
- Covariance: propagate attitude σ, altitude σ, pixel σ through the projection (numeric Jacobian is fine). You need this for fusion weighting and CEP.

**Test:** camera at (0,0,100 m) straight down → target = (0,0,0), slant=100, off-nadir=0. Then pitch 30°, flat ground → target should be `100·tan30 ≈ 57.7 m` north of nadir. Hand-verify. Then raise ground 10 m and confirm the point moves the right direction.

---

## Module [8] — Data association / clustering

**Contract:** in = all per-detection world points (+cov, class, conf, frame, t); out = cluster label per point.

**Pseudocode**
```
pts_enu = to_local_enu(all lat/lon)
labels  = DBSCAN(eps=4.0, min_samples=3, metric='euclidean').fit(pts_enu)   # per class
# label == -1 → noise (drop or low-confidence singleton)
```

> **Why DBSCAN and not KMeans?** You don't know how many flags there are (2, 3, or noisy) — KMeans needs that count `k` up front. KMeans also forces every point into a cluster, so one false-positive detection drags a centroid off the real flag; DBSCAN labels stray points as noise (−1) and discards them. For our few, well-separated flags a plain "group everything within X metres" threshold (the current greedy clustering) works just as well — DBSCAN is only the robust off-the-shelf version of that same idea. Just **don't use KMeans.**

**Gotchas**
- `eps` is in **metres** — convert lat/lon to a local ENU plane first; don't DBSCAN on raw degrees (1° lat ≠ 1° lon ≠ metres).
- Tune `eps` to roughly your single-detection spread. Too small → one flag splits into several; too big → two close flags merge. With a 20 m scoring tolerance and flags tens of metres apart, this is forgiving.
- Cluster **within class** so flag/not-flag never merge.

**Test:** synthesize 5 ground-truth points, scatter 38 noisy detections around them (σ≈2 m), run DBSCAN → must recover 5 clusters with the right membership. This is also your end-to-end "38→5" proof.

---

## Module [9] — Fusion + confidence

**Contract:** in = one cluster of points (+cov, conf, off-nadir, frame, t); out = one result row.

**Pseudocode**
```
inliers = ransac_or_MAD_reject(cluster)             # drop wild points
W = Σ inv(cov_i)                                     # inverse-variance
mu = inv(W) · Σ (inv(cov_i) · p_i)                   # weighted mean (ENU) → lat/lon
cep50 = 1.177 · sqrt(mean eigenvalue of inv(W))      # approx circular error
rep = argmin_i (trace(cov_i))                        # most-nadir/cleanest detection
confidence = blend(mean_conf, N, tightness, offnadir)  # see documentation.md §5.3
```

**Gotchas**
- Biases (boresight, GPS) don't shrink with N — don't report an absurdly tiny CEP just because N is large. Floor the CEP at your systematic error estimate.
- Weight down high-obliquity detections (their cov is large — inverse-variance does this automatically if your [7] covariance is honest).

**Test:** feed a tight cluster → mean near centroid, small CEP. Inject one 50 m outlier → RANSAC must drop it and the result must barely move.

---

## End-to-end synthetic harness (build this early — it's your safety net)

Before touching real footage, simulate the whole chain with known truth:

```
1. Place 5 known flags at known lat/lon/elev.
2. Fly a synthetic trajectory (positions + attitudes over time) past them.
3. For each frame, PROJECT each visible flag into the image (forward camera model)
   → gives you synthetic detections WITH known ground truth.
4. Run your pipeline [5]→[9] on these.
5. Assert recovered lat/lon are within, say, 1 m of truth (no sensor noise),
   and that you get exactly 5 clusters.
6. Then add noise (attitude σ, GPS σ, pixel σ) and check the error matches
   your §3.5 budget and the CEP you report.
```

If the noise-free case isn't sub-metre, you have a frame/sign/datum bug — **do not** debug on real data until this passes. The forward projector in step 3 is the inverse of module [6]/[7]; writing both forces you to get the geometry right.

---

## Debugging playbook (symptom → likely cause)

| Symptom | Most likely cause | Where to look |
|---|---|---|
| Coordinates land in a consistent wrong **direction** | Frame/sign error (ENU/NED, yaw sign, camera-axis) | Module 0 self-test, [6] golden test |
| Coordinates off by a roughly **constant distance along-track** | Time-sync offset `b` wrong | [4] synthetic shift test |
| Altitude/scale off by a **constant ~tens of metres** | HAE vs MSL geoid mismatch | [7] datum, [3] which alt field |
| Error **grows with obliquity** | Attitude or boresight calibration | calib, §3.5; fly more nadir |
| Right place but **scale wrong** (too near/far) | `K` resolution mismatch or wrong focal length | [6], recalibrate intrinsics at flight resolution |
| **One flag → several results** | DBSCAN `eps` too small / track fragmentation untreated | [8] eps tuning |
| **Two flags → one result** | `eps` too big, or flags closer than your CEP | [8], improve per-detection accuracy first |
| Random large outliers in a cluster | Bad detections / bad pose interp at edges | [5] extrapolation guard, [9] RANSAC |
| "In the sea" / NaN | Ray points above horizon, or interp past table edge | [7] up-component guard, [5] clamp |

---

## Suggested build order (and what "done" means for each)

1. **Frames + golden ray test** ([0]-paper, [6]) — done when the straight-down test passes.
2. **Telemetry normalize** ([3]) — done when plots look physically right and both alt datums are labelled.
3. **Intersection + geoid** ([7]) — done when the pitch-30° hand-check matches.
4. **Synthetic end-to-end harness** — done when noise-free recovery is sub-metre.
5. **Time sync** ([4]) — done when the synthetic shift test recovers the lag and sign.
6. **Detection + tracking + anchor** ([1][2]) — done when centroid anchors sit on the flags and crops are clean enough for the classifier.
7. **Clustering + fusion** ([8][9]) — done when 38 synthetic detections → 5 clusters.
8. **Seek/windowing + speedups** ([0] full) — done when a 2.5-min window runs in seconds.
9. **Validate on real flight with ≥2 surveyed ground-control flags** — done when measured CEP matches your reported CEP.

Each "done" is a test you can re-run. Keep them; they're your regression suite when you tweak settings the night before the competition.

---

## A note on "vibe coding"

You were right to ask for this. The trap isn't using help — it's shipping geometry you can't derive. For this project the non-negotiable things to understand *in your own head*: the four coordinate frames and their rotations, why HyperSmooth/Horizon Lock break the model, the HAE-vs-MSL datum, and the §3.5 error scaling (why obliquity hurts). If you can re-derive the straight-down ray test and the pitch-30° intersection on paper, you own this pipeline. Everything else is plumbing you can write and test one module at a time.
