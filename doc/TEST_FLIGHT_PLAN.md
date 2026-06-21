# Test Flight Plan — GoPro Hero 13, Offline Pipeline

**ICMTC 2026 — UAVC-9 Flag Localization — AI Team, Eagles (Nile University)**
**Written:** 2026-06-21 · **Flight:** tomorrow, 2026-06-22 · **Team size:** 3
**Camera decision: GoPro Hero 13 Black, Linear mode, telemetry from GoPro GPMF only (no Pixhawk for v1).**

---

## 0. Read this first — the one-day reality

You have ~1 working day and a flight tomorrow. The flag detector exists (initial version). The **integration layer does not** — `pipeline/` is an empty folder. So the goal for tomorrow is **not** the 8-stage Approach 2/3 in the docs. The goal is a **working end-to-end MVP that turns a recorded GoPro clip into GPS coordinates offline on the laptop**, validated against at least one known ground point. Everything fancy (DBSCAN, SAHI, auto-mount calibration) is explicitly out of scope for tomorrow.

The single most important decision you've already made — **GoPro-only telemetry** — is the right one for a one-day timeline, and it changes the architecture in your favour. See §2.

> **Build it incrementally, not all at once.** Don't write the whole pipeline and discover at the end you don't know what each stage produced. Follow the companion doc **`STEP_BY_STEP_BUILD.md`** — every step has a test and an expected output, so you validate each tool (e.g. the GPMF extractor) on a real video *before* moving on.

---

## 1. Camera decision: what the Hero 13 changes vs. the existing docs

`FINAL_Pipeline_Document.md` is written for the **Hero 5 / Hero 9**. The Hero 13 is a different camera, so three things must change:

| Item | Doc says (Hero 5/9) | Hero 13 reality | Action |
|---|---|---|---|
| **Intrinsics fx/fy/cx/cy** | fx≈1432 (H9) / 1060 (H5) | Different sensor + HB lens — **unknown until calibrated** | Re-calibrate. Do **not** ship the doc's numbers. |
| **Distortion model** | Brown-Conrady (correct) | Still Brown-Conrady in Linear mode | Keep model, get **new** k1,k2,p1,p2 from calibration. |
| **GPS** | Hero 9 had GPS; doc leans on Pixhawk | **GPS is back on Hero 13** — 10 Hz, U-blox MAX-M10S (same chip as Hero 11) | Use GoPro GPS embedded in video. |
| **Attitude / tilt** | Aircraft IMU roll/pitch/yaw + R_mount | GoPro GPMF carries **GRAV** (gravity vector) + **GYRO** + **CORI** (orientation) | Use **GRAV for tilt**, GPS course-over-ground for heading. R_mount tilt calibration becomes unnecessary (see §2). |

**Must-set GoPro settings before every flight (unchanged from doc, still critical):**
Linear mode · HyperSmooth/EIS **OFF** · Horizon Lock **OFF** · 1080p or higher, 30–60 fps · GPS **ON** · battery >90%. EIS on warps frames and silently destroys localization — verify on-screen.

**Detectability sanity check (do not skip):** at 75–100 m the 1×2 m flag is only ~20–40 px long. Your detector was trained on the initial set — confirm it fires on flags this small in real Hero 13 footage **before** trusting the localization numbers. If it misses at altitude, fly lower (75 m) for tomorrow.

---

## 2. Why GoPro-only simplifies the integration (the key insight)

In the original architecture the two hardest, riskiest problems are **(a) synchronising the video stream to a separate telemetry log**, and **(b) calibrating the camera mount tilt + aircraft attitude**. The senior's answers and the GPT test doc both rank synchronisation as the #1 risk.

With Hero 13 GPMF, **GPS + accelerometer + gyro + gravity vector are all embedded inside the same video file, sharing one clock.** That means:

- **Synchronisation is largely solved.** No cross-device clock alignment. You only interpolate GPMF samples (GPS 10 Hz, IMU higher) to each frame's timestamp — a within-file operation, not a cross-stream guess. Risk drops from "highest" to "low."
- **Mount tilt calibration is unnecessary.** GRAV gives the camera's true tilt off-nadir **per frame**, directly. You no longer measure mount roll/pitch or aircraft attitude. The only remaining mount unknown is a **single yaw-offset constant** (the angle between the image's "up" and the flight direction), which you set once.
- **Heading without a magnetometer.** Use GPS **course-over-ground (COG)** from successive positions for azimuth — immune to motor magnetic interference, exactly as reference [8] recommends.

So the revised per-frame chain for tomorrow is:

```
GoPro .mp4
  ├─ video frames ──► YOLO ──► bbox center (u,v), conf
  └─ GPMF metadata ─► GPS(lat,lon), GRAV(tilt), COG(heading)   ← same file, same clock
                          │
            undistort (u,v) ─► camera ray
                          │
   rotate ray to world using GRAV (tilt) + COG (heading) + yaw-offset const
                          │
            ray–ground intersection at altitude h  ─► (ΔE, ΔN)
                          │
            GPS conversion ─► per-frame (lat,lon)
                          │
      median / confidence-weighted aggregate over frames ─► final flag GPS
```

You already have ~80% of the geometry: `src/approach2.py` does undistortion → ray → rotation → ray-ground → GPS → weighted aggregation. **Don't rewrite it — wrap it.** The only genuinely new code is (1) the GPMF extractor and (2) a `build_R_from_grav_cog()` that replaces `build_R_total()`.

> ⚠️ **Altitude is now your #1 error source, not sync.** GoPro GPS *altitude* is noisy (can be ±10–20 m), and altitude error maps almost 1:1 into ground error. **Do not use GoPro GPS altitude as h.** Instead use **AGL = (known field elevation baseline) → fixed flight altitude**, or read the planned flight altitude, or calibrate at takeoff. Decide your altitude source today (§3, Person A).

---

## 2.5 Getting telemetry locally + connecting timestamp → location → YOLO (no website)

You currently get the CSV by uploading the video to a website. **Drop that** — no internet at the competition, and you can't depend on a third-party site under a 10-minute clock. The website just runs a GPMF parser you can run yourself, offline. The whole join works because **the GPMF telemetry samples and the video frames are timestamped from the same `t=0` (start of the video)** — so connecting them is just matching by time, no cross-device sync.

The five-step data flow (detailed, with per-step tests, in `STEP_BY_STEP_BUILD.md`):

1. **Extract telemetry locally.** Use `telemetrik` (pure Python, zero deps) or `py-gpmf-parser`. Output per sample: `time_s, lat, lon, GRAV(tilt), GYRO`. GPS arrives ~10 Hz. `exiftool -ee` / `ffprobe` confirm the GPMF track exists. *Validate it by comparing the local CSV to the website CSV you already have — they should match.*
2. **YOLO with timestamps.** Read fps once (`ffprobe`). For each detection store `(frame_time = frame_index / fps, u, v, conf)`. `frame_time` is on the same clock as the telemetry.
3. **Join by time (the "sync").** For each detection at `t`, interpolate telemetry to `t`: `lat_uav, lon_uav = np.interp(t, gps_t, gps_lat/lon)`, tilt from GRAV at `t`, heading from GPS course-over-ground around `t`. Result per detection: `(u, v, conf, lat_uav, lon_uav, h, tilt, heading)`.
4. **Localize** each row through `src/approach2.py` geometry (needs calibrated K/dist + altitude `h`).
5. **Aggregate** per flag (median / weighted mean) → one `(lat, lon)` → write to USB in the required format.

**Buildable today without the camera:** steps 1, 2, 3, 5 on your 2 existing videos. Step 4 needs calibration (camera) + altitude — wire it with placeholder Hero 13 intrinsics now, swap in real numbers at the flight.

---

## 3. Today's plan (3 people, parallel) — get to a working MVP

Target by end of day: **a recorded test clip runs end-to-end on the laptop and prints a flag GPS that lands within ~20 m of a surveyed point.** Even if it's only the simplified nadir+GRAV path, that is a successful test-flight outcome.

### 👤 Person A — Camera + GPMF telemetry
1. **Calibrate the Hero 13** in the exact flight mode (Linear, EIS off, chosen resolution). Print 9×6 checkerboard, capture 40–50 shots covering frame edges/corners at several tilts, run `cv2.calibrateCamera`, require RMS < 1.0 px. Save `configs/camera_params_hero13.yaml` with K + dist. *(Use the calibration script in `FINAL_Pipeline_Document.md` Part 7 verbatim.)*
2. **Stand up GPMF extraction.** Install a parser — `telemetrik` (pure Python, zero deps) or `py-gpmf-parser`. Write `pipeline/gpmf_extract.py` that takes a `.mp4` and returns time series for **GPS5/GPS9, GRAV, GYRO, CORI** plus their timestamps. Dump one short clip to CSV/JSON and eyeball it: GPS moves, GRAV ≈ (0,0,1)-ish when level.
3. **Lock the altitude source.** Decide today: fixed planned AGL, or barometer/field-elevation baseline. Write it into the config. This is the highest-leverage decision of the day.

### 👤 Person B — Geometry integration (critical path)
1. Add `build_R_from_grav_cog(grav_cam, cog_rad, yaw_offset_rad)` in `src/` that builds camera→world rotation from the gravity vector (tilt) and COG (heading). Keep `localize_full()`'s undistort → ray → ray-ground → GPS stages unchanged.
2. **Safe fallback first:** if GRAV shows near-nadir tilt (<3°), fall back to the Approach 1 nadir formula. A correct nadir MVP beats a buggy full-rotation path for tomorrow.
3. Write `pipeline/run_localization.py`: load video → for sampled frames (every Nth, ~5–10 fps is plenty) get YOLO detections + interpolated GPMF → call `localize_full` → collect per-frame (lat,lon,conf) → aggregate (median for v1; confidence-weighted if time). Output `results.json` with one GPS per flag.
4. **Run the existing unit-test math first** (the 5 tests in `FINAL_Pipeline_Document.md` Part 7 — nadir center, east offset, yaw rotation, oblique rejection, altitude scaling). These already pass on the core math (verified). Put them in `tests/`. Green tests = trust the geometry, debug only the integration.

### 👤 Person C — Ground truth + validation harness
1. **Survey 2–3 known points today.** Place targets, stand over each centre, log GPS with a phone app (3-min average), cross-check on Google Earth. Record id/lat/lon/surface. These are tomorrow's scoring proxies.
2. Wire up `validation.py` (haversine error, points-per-bin, mean/median/max, % within 20 m) — the code is ready in `FINAL_Pipeline_Document.md` Part 7. Add **CEP** and **% < 20 m** since that maps directly to competition points.
3. **Dry run before the field:** the moment Person A has one GoPro clip + Person B has the runner, process a *handheld* clip walked over a surveyed target and measure error. This is the phone-style validation the senior endorsed — do it on the ground today so tomorrow only adds altitude.
4. Prepare the **USB submission rehearsal**: required coordinate format, empty FAT32 USB, and time the "land → extract SD → copy → run → export → submit" loop. The 10-minute window is part of the test.

### Shared, end of day
- One **integration smoke test**: a real Hero 13 clip → `run_localization.py` → a GPS that's within ~20 m of a surveyed point on the ground. If this passes today, tomorrow is low-risk.
- Freeze the config. Print the pre-flight checklist (`FINAL_Pipeline_Document.md` Part 9), adapted: tick GPS ON, EIS OFF, altitude source set.

---

## 4. Tomorrow — flight day

1. **Pre-flight:** run the Part 9 checklist. Confirm GoPro: Linear, EIS off, Horizon Lock off, GPS on, recording (red light). Note takeoff GPS + clock.
2. **Fly 75–80 m AGL** (not 100 m) so the flag is ≥28 px and detection is reliable. Straight, level passes over each target; a few seconds per flag at 5–10 usable fps gives 20+ frames for averaging.
3. **Land → offline run** on the laptop: extract SD → `run_localization.py video.mp4 → results.json` → `validation.py` against surveyed truth. Everything runs locally; no internet, as designed.
4. **Record metrics:** mean, median, max error, % < 20 m, and per-flag points. That's your go/no-go evidence for the competition.

**MVP definition of success for tomorrow:** at least one flag localised within 20 m using the offline GoPro-only pipeline. Anything beyond (multiple flags, GRAV tilt correction working, confidence weighting) is upside.

---

## 5. Validation & review of `gptrecommended_test.md`

**Overall verdict: the GPT test plan is sound, correctly sequenced, and worth following.** Its phase order (detect → calibrate → telemetry → sync → ray → mount → rotate → ground → GPS → ground-truth) matches the real data dependencies, and it correctly elevates **synchronisation** and **altitude** as the top practical risks — consistent with both the senior's answers and a clean reading of the math. Its closing claim *"the localization mathematics is correct"* checks out: I re-ran the core geometry (nadir centre → 0 error, 10 m-east pixel → 10.00 m, altitude doubling → exactly 2.0× offset). So **adopt it as the validation backbone.**

However, it was written for the **generic separate-streams architecture**. For your **GoPro-Hero-13-only, offline** setup, four corrections/additions are needed:

**Corrections**

1. **Phase 4 (Synchronisation) is no longer "highest priority."** With GPMF, video and telemetry share one clock in one file. Downgrade this from cross-stream alignment (hard) to within-file interpolation of 10 Hz GPS / IMU to frame time (easy). Keep the interpolation step; drop the cross-device delay hunt.
2. **Phase 6 (Mount calibration) shrinks.** GRAV measures camera tilt per frame, so you do **not** calibrate mount roll/pitch or aircraft attitude. The only mount unknown is a **single yaw-offset constant** (image-up vs. flight direction). Rewrite Phase 6 to "determine one yaw-offset angle," not full mount alignment.
3. **Altitude is the real #1 risk — make it explicit.** The doc lists "altitude error 2–20 m" in a table but doesn't act on it. Promote it: **do not use GoPro GPS altitude**; fix h from planned AGL or a barometric/field-elevation baseline. This single choice dominates your error budget now that sync is handled.
4. **Phase 1 (YOLO) needs an altitude-vs-detectability check.** Add: confirm the detector fires on a ~20–40 px flag in *real Hero 13* footage at flight altitude. Detection size, not just count, is the gate.

**Missing phases to add**

5. **Phase 0 — GPMF extraction & decode.** Parse and sanity-check GPS/GRAV/GYRO/CORI before anything else; it's the data source for half the pipeline.
6. **Frame sampling / extraction phase.** Decide fps to process (5–10 fps is enough); avoids the 7-min-video-takes-15-min throughput problem the senior raised, exactly via frame sampling.
7. **End-to-end dry run on a recorded clip before flight** (handheld over a surveyed point). Catches integration bugs on the ground, where they're cheap.
8. **USB export-format + 10-minute submission rehearsal.** The submission loop is part of mission success and is untested in the current doc.

**Metrics:** keep mean/median/max; **add CEP and % within 20 m**, since those map directly to the scoring table. The senior's real-world distance-measurement approach is the right ground-truth method.

---

## 6. Scope discipline — what to cut for tomorrow

| Keep (MVP) | Defer (post-flight) |
|---|---|
| GPMF extraction (GPS, GRAV, COG) | DBSCAN outlier clustering (Approach 3) |
| Undistortion + ray-ground (reuse approach2) | SAHI sliced detection |
| GRAV-tilt rotation **or** nadir fallback | Auto-mount-angle optimisation |
| Median / confidence-weighted aggregation | Pixhawk integration / dual-GPS fusion |
| Validation vs. 2–3 surveyed points | CORI-based full attitude (use as cross-check only) |

---

## 7. Top risks for tomorrow (ranked)

1. **Altitude source wrong/noisy** → biggest ground error. Fix h today; don't trust GoPro GPS altitude.
2. **EIS/Horizon Lock left on** → warped frames, silent failure. Verify on-screen every flight.
3. **Detector misses small flags at altitude** → no detections to localise. Test on real footage today; fly lower if needed.
4. **Integration not finished** → mitigate by shipping the nadir+GRAV fallback (simple, robust) rather than the full rotation path.
5. **Yaw-offset constant guessed wrong** → flag lands in the right ring but wrong direction. Pin it down in today's ground dry run over a known point.

---

### Sources
- GoPro Hero 13 GPS is back (10 Hz, U-blox MAX-M10S): [GoPro Telemetry Extractor blog](https://goprotelemetryextractor.com/blog/gps-is-back-go-pro-hero-13), [GoPro HERO13 GPS support](https://community.gopro.com/s/article/HERO13-Black-GPS-Information?language=en_US)
- GPMF streams (GPS5/9, GRAV, CORI, GYRO) & Python extraction: [gopro/gpmf-parser](https://github.com/gopro/gpmf-parser), [telemetrik](https://github.com/kmatzen/telemetrik), [py-gpmf-parser](https://github.com/urbste/py-gpmf-parser)
- Internal: `doc/FINAL_Pipeline_Document.md`, `doc/Egyptian Team Answers.md`, `doc/gptrecommended_test.md`, `doc/pipeline_workflow.png`, `src/approach1–3.py`
