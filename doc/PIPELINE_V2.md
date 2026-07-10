# Pipeline V2 — Fixes, Test-Day Plan, Decision Rules

Post-mortem of the first mission run produced five code fixes (all implemented) and one
procedural checklist. Verified: unit tests pass; a synthetic ground-truth flight through
the full localizer recovers 3/3 flags with ≤0.5 m error, rejects oblique detections, and
drops under-supported clutter.

---

## 1. What changed (before → after)

| # | Fix | Before | After | Why (1–2 lines) | Where |
|---|-----|--------|-------|-----------------|-------|
| 1 | **GPS quality gate** | All GPS9 samples used; `fix`/`dop` decoded then discarded | Samples with `fix < 3` or `dop > 10` dropped; hard error if nothing survives | Mission video carried a stale pre-lock fix 21 km away for ~100 s; it poisoned position AND the AGL anchor | `gpmf_extract.py` → `gate_gps_quality()` |
| 2 | **Teleport rejection** | Position spikes passed through | Sample dropped if implied speed to BOTH neighbours > 80 m/s | Post-lock single-sample glitches survive the fix gate and scatter geolocations | `gpmf_extract.py` → `drop_teleports()` |
| 3 | **AGL anchor** | `agl = alt − alt[first sample]` | `agl = alt − median(alt)` over a stationary (<1.5 m/s), locked window ≥10 s at the START or END of the video; loud fallback + plausibility warning (median airborne AGL must be 30–130 m) | First sample was the garbage fix → all AGL inflated ~55 m (used 112–177 m vs true ~60–90 m), a ~2× range error on every offset. Same-receiver differencing also cancels the geoid/datum question | `gpmf_extract.py` → `find_ground_alt()` |
| 4 | **GPS-UTC kept** | UTC decoded, discarded | `utc_s = days_since_2000·86400 + secs_since_midnight` written to `telemetry.csv` and passed into `synced.csv` | Pixhawk `.bin` logs GPS time too → a future Pixhawk join (AGL / position / attitude) is a table merge by UTC, no wing-rock sync needed | `gpmf_extract.py`, `sync.py` |
| 5 | **Off-nadir gate** | Rays accepted up to 84° off vertical | Total ray angle (tilt + pixel offset) > 30° rejected (`--max-offnadir`) | An oblique ray has a `h·tan θ` lever arm that heading error rotates; near-nadir rays are insensitive to both heading and AGL error | `localization.py` → `off_nadir_deg()` |
| 6 | **AGL sanity per detection** | `h ≤ 0` only | `h` outside `[--min-agl, --max-agl]` (20–150 m) rejected; warning if >50 % rejected | Implausible heights must be refused, not silently projected into scattered points. `--alt` (manual) is still trusted as-is | `localization.py` main loop |
| 7 | **DBSCAN + geometric median** | Greedy first-point clustering (order-dependent, fragments scatter into 25 m balls) + confidence-weighted mean | DBSCAN (`eps = --cluster-m`, default 12 m), noise points discarded; cluster centre = Weiszfeld geometric median (outlier-robust) | Greedy clustering turned scattered inputs into 60 "flags"; DBSCAN is order-independent and labels stragglers as noise instead of new flags | `localization.py` → `dbscan()`, `geometric_median()` |
| 8 | **Top-K cap** | Unlimited rows in submission | `--top-k` keeps the K best-supported clusters (run_pipeline default: 3 = 2 flags + 1 bonus) | Mission 1 expects ~3 reported flags; a 60-row submission risks scoring zero | `localization.py`, wired in `run_pipeline.py` |
| 9 | **Wiring** | `detection.csv` vs `detections.csv` footgun; detect ran at `imgsz=640` | One canonical `detections.csv`; `--imgsz` exposed, default 1920 | At 640, a 2 m flag at 60–100 m AGL is 6–10 px — below reliable YOLO size. At 1920 it is 3× larger | `run_pipeline.py` |

**New CLI flags:** `gpmf_extract.py --min-fix --max-dop --max-speed` · `localization.py --max-offnadir --min-agl --max-agl --top-k` · `run_pipeline.py --top-k --max-offnadir --imgsz`.

**Operational consequence of the 30° gate:** accepted swath ≈ `2·h·tan(30°)` ≈ **0.9 m of
ground per metre of altitude to each side** (±40 m at 70 m AGL). Plan lawnmower pass
spacing ≤ that swath so every flag is overflown near-nadir. Widening the gate
(`--max-offnadir 40`) trades accuracy for coverage — verified working in simulation.

---

## 2. Test-day execution plan

### Pre-flight (ground)
1. GoPro settings — verified by a SECOND person: **Linear lens, 4K, HyperSmooth OFF,
   Horizon Lock OFF, GPS ON** (any change later ⇒ recalibrate).
2. Power on GoPro → **wait for GPS lock icon** → only then press record.
3. Record **60 s stationary on the ground** before launch (this is the AGL anchor).
4. Place **3–5 targets at surveyed points** (phone GPS averaged ≥30 s per point; write
   the coordinates down before flying).
5. Pixhawk logging armed; note home position. GoPro mounted, lens clean.

### Flight
1. Wing-rock once after takeoff and once before landing (enables the Pixhawk attitude
   path; costs nothing).
2. **Straight, level lawnmower passes directly over the targets** at 60–80 m AGL;
   pass spacing ≤ `2·h·tan(30°)` (≈80 m at 70 m). No orbiting over targets.
3. At least one repeat pass from a different heading (exposes azimuth/heading bias).
4. Keep recording through landing + 30 s stationary (backup AGL anchor).

### Post-flight (same day)
1. Copy MP4 + Pixhawk `.bin`; archive both untouched.
2. `python pipeline/run_pipeline.py --video <MP4>` — check the log for:
   GPS-gate keep-rate (>90 % expected), AGL anchor source + ground alt, airborne AGL
   inside 30–130 m, off-nadir rejection counts.
3. Compare reported flag coordinates to the surveyed points → per-flag error (m),
   % within 20 m.
4. `pixhawk_extract.py` on the `.bin`; compare GoPro AGL vs `RelHomeAlt`, and GoPro
   GRAV-tilt vs `ATT` roll/pitch on straight segments (validates GRAV).
5. Record detection recall: of all frames where a target is visibly in frame, how many
   were detected (spot-check the overlay frames).

---

## 3. Decision rules after the flight

| Observation | Decision |
|---|---|
| Median per-flag error **≤ 20 m** (and ≤ 20 m for every target overflown near-nadir) | **GoPro-only pipeline is PRIMARY.** Freeze code; competition config = this flight's flags. Pixhawk stays as emergency backup. |
| Errors **> 20 m and dominated by tilt/heading** (signature: error grows with `tilt_deg`; different-heading passes disagree on the same target; GRAV-tilt vs Pixhawk ATT disagree in turns) | **Switch to the Pixhawk hybrid:** join `RelHomeAlt` + `ATT` yaw/roll/pitch by GPS-UTC (`utc_s` is already in `synced.csv`); wing-rock affine only if UTC residual > 0.2 s. GoPro keeps pixels + clock. |
| Errors dominated by a **constant radial scale** (all flags pushed outward/inward proportionally to distance-from-nadir) | AGL still biased → fix anchor window or take AGL from Pixhawk; as emergency, `--alt <planned AGL>`. |
| **Detection recall fails** at altitude (targets visibly in frame but not detected) | Raise `--imgsz` to 2560/3840 or switch to tiled inference; lower `--thresh` to 0.3 and rely on `--min-frames` for precision; if still failing, add this flight's frames to training data and retrain before competition. |
| A flag is seen but **gated away** (off-nadir rejections eat a whole target) | Flight-plan problem, not code: tighten pass spacing, or fly a dedicated pass over it; only then consider `--max-offnadir 40`. |
| GPS-gate keep-rate **< 50 %** | Procedure problem: GPS was not locked before recording. Fix the pre-flight step; do not tune thresholds around it. |

---

## 4. Verification performed (no flight data needed)

- Unit tests (`tests/test_geometry.py`): off-nadir angle, DBSCAN (two blobs + noise;
  60 m chain does not fragment), geometric-median outlier robustness, rotation sanity —
  all pass.
- Synthetic ground-truth flight (straight pass, 70 m AGL, 3 flags + 45°-tilt oblique
  detections + 3-frame clutter): localizer reports exactly the real flags,
  **errors 0.02–0.53 m**, obliques rejected by the gate, clutter dropped as DBSCAN noise,
  top-K cap exercised.
- Caveat: synthetic data shares the projection model with the localizer, so it validates
  plumbing/gates/clustering — NOT the heading-sign convention or sensor errors. Those are
  exactly what the test flight measures.
