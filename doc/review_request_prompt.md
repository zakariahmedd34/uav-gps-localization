# Project Review Request — UAV Flag Localization

You are a **senior computer-vision / robotics / UAV-navigation engineer**. I'm about to give
you the full codebase and docs for an autonomous-UAV flag-localization project for a competition.
**Review it critically** — do not just validate it. Find problems, risks, and failure modes, and
give me **prioritized decisions**. Be specific and technical.

## The competition (constraints that shape everything)
- **ICMTC 2026 UAVC-9 Fixed-Wing Challenge** — autonomous fixed-wing UAV (MTOW ≤ 10 kg, VLOS).
- Relevant task (Mission 1): after a payload drop, **search a zone and locate/identify flags on the
  ground and report each flag's GPS coordinates**. There are **2 flags in the search area + 1 bonus
  flag** elsewhere in the geofence.
- **Targets:** national flags, **2 m × 1 m, lying FLAT on the ground**. Flat desert airfield (6th
  October, Cairo). Aircraft cruise ~20–25 m/s.
- **Operating altitude: 50–100 m AGL.**
- **Scoring (accuracy):** 15 pts if within **20 m**, scaling down to 3 pts at 50–60 m. So **~20 m =
  full marks** — this is a forgiving tolerance. Plus 10 pts for correct flag "identification."
- **Submission:** GPS coordinates **+ an image of each flag**, on a **USB drive**, handed in shortly
  after the ~10-minute mission (tight time budget, done on-site on one laptop).

## Hardware
- **Camera:** GoPro Hero 13, **calibrated** (Linear lens, real intrinsics, RMS ~2.1 px @ 4K).
  Flown with **HyperSmooth OFF, Horizon Lock OFF, Linear lens** to keep a rigid pinhole model.
- **Flight controller:** Pixhawk / ArduPilot — produces `.bin` DataFlash logs (GPS lat/lon,
  ATT roll/pitch/yaw, POS.RelHomeAlt = AGL; **no rangefinder**).

## Current pipeline (offline Python) — TWO paths

**PRIMARY — GoPro-only** (the video's GPMF metadata carries its own GPS + gravity, on the *same
clock* as the frames, so there is **no video↔telemetry time-sync problem**):
1. `detect.py` — YOLO (flag / not-flag), saves a crop per detection; **anchor = bbox centroid**
   (flags lie flat, no pole).
2. `gpmf_extract.py` — GPS9 + GRAV from GPMF; **AGL = GPS_alt − first_sample_alt** (flat-ground
   approximation, **no DEM**).
3. `sync.py` — joins telemetry to detections by shared timestamp; **heading from GPS
   course-over-ground**, tilt from the gravity vector.
4. `localization.py` — pixel → undistort → camera ray → rotate to world (gravity + heading) →
   intersect ground plane at AGL → lat/lon → **greedy distance clustering** into flags →
   representative (highest-conf) crop per flag → writes submission (`targets.csv` + `flags/` images).

**SECONDARY — Pixhawk path (built but NOT validated; no paired video+log flight exists yet):**
`pixhawk_extract.py` (.bin/.log → telemetry) and `sync_pixhawk.py` (affine time model
`t_log = scale·t_video + offset`, anchored by a wing-rock or GPS-UTC, circular yaw interpolation).
The localizer that uses roll/pitch/yaw + AGL + boresight is designed but **not implemented**.

## Key design decisions & rationale (please challenge these)
- **GoPro-only is the primary plan** — removes the two-clock time-sync problem and fits the
  10-minute window. Pixhawk is the accuracy upgrade for windy/oblique cases.
- **No DEM** — flat airfield + 20 m tolerance → flat-ground with AGL = GPS_alt − ground.
- **Fly near-nadir over the targets** to neutralize the GoPro's heading weakness (heading comes
  from GPS course-over-ground, which is wrong in wind/crab; a flag directly below is yaw-insensitive).
- **Clustering:** simple greedy distance threshold (not KMeans — unknown flag count + outliers).
- **Anchor = bbox centroid** (flat flags).

## Validated vs NOT validated
- **Validated:** camera calibration (RMS 2.1 px), geometry unit tests, time-windowing, crop saving,
  sync interpolation (incl. circular-yaw wrap), submission format, Pixhawk `.bin` parsing.
- **NOT validated:** end-to-end **absolute accuracy** — no paired flight with surveyed flag
  coordinates yet. The only test clip (G1) is handheld (no real climb → auto-AGL untestable on it).

## Concerns I already know about
- Heading accuracy (GPS course-over-ground vs true yaw) in wind — the main GoPro-only weakness.
- GoPro GPS altitude noise (±3–5 m) → AGL noise.
- Detecting **small flags at 50–100 m** (a 2 m target is only tens of pixels) — recall, false positives.
- Two-stage detect→classify (flag identity / which country) not built — pending judge clarification.
- Rolling shutter; boresight calibration (Pixhawk path); geoid/datum if mixing MSL.

## Problems observed on the FIRST real-mission run (please diagnose and optimize)
Priority framing: **optimize the GoPro-only (first) approach as much as possible; treat the
Pixhawk approach as the emergency/backup.**

1. **Calibration mode mismatch (I suspect this is the root cause).** The mission video was
   recorded with the GoPro in **Wide** lens mode, but the camera calibration was done in **Linear**
   mode. So the intrinsics + distortion don't match the footage → every pixel→ray→ground point is
   wrong. How much of the bad output does this explain, and what's the cleanest fix (recalibrate in
   Wide vs re-record in Linear vs undistort Wide→Linear in software)?
2. **AGL looks physically wrong.** AGL = GoPro_GPS_alt − first_sample_alt yields values >100 m,
   some **negative**, and **jumps to ~200 m** — impossible at 50–100 m operating altitude. GoPro GPS
   *vertical* accuracy is poor. How should I validate and clean this (physics/rate check,
   median filter + outlier rejection, compare to Pixhawk RelHomeAlt, or just use a fixed nominal
   altitude)? Noisy AGL also scatters the geolocations.
3. **Submission has >20 rows for ~5 physical flags,** with disagreeing locations. I believe this is a
   *symptom* of (1)+(2) scattering the per-detection points so greedy clustering makes 20+ clusters.
   Is that right, or is the clustering itself the problem? I considered dropping clustering for a
   per-second / tracking-based association — is that better, or does it not help if the inputs are bad?

**Please tell me the true root-cause ordering and give the minimal fixes to get the GoPro-only path
to ~5 flags within 20 m.**

## What I want from you (be critical, prioritized, specific)
1. **Review the overall approach/architecture for THIS competition.** Is GoPro-only the right
   primary? Where and how will it fail?
2. **Find problems / risks / failure modes I may have missed** — especially the DL/CV parts
   (small-flag detection at altitude, false positives, the classifier, dataset/training concerns)
   and the localization/geometry.
3. **Give concrete DECISIONS and a prioritized action list** before the competition — what to
   build/validate first, what to drop.
4. **Sanity-check the accuracy story:** is ~20 m realistic with GoPro-only near-nadir? What's the
   dominant error term, and how would you reduce it?
5. **Critique the validation plan:** I intend a synthetic harness that forward-projects flags at
   known coordinates through a real flight trajectory to measure localization error in metres. Is
   that sound? What would you add?

The full code and documentation are attached. Please respond with: **(a) critical findings/problems,
(b) prioritized decisions/recommendations, (c) any red flags for the competition** — and push back
on anything you disagree with.
