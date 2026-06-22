# Step-by-Step Build Guide (with a test after every step)

**Companion to `TEST_FLIGHT_PLAN.md`.** Build the pipeline **one piece at a time**. After each step there's a **🔎 Test** and the **✅ Expected output** you should see. **Do not move to the next step until the test passes.** This way you always know exactly what each stage produces, and when something breaks you know which step did it.

> What you have today: 2 GoPro videos on the laptop, a CSV you got from a website, no physical camera yet. That's enough for Steps 0–5 and 8–10. Steps 6–7 need calibration + altitude (camera at the flight) — you'll wire them with placeholders now and swap real numbers in later.

Put a video in `data/` and call it `test.mp4` for the commands below.

---

## Step 0 — Environment

**Do:** install the tools.
```bash
pip install numpy opencv-python pandas pyyaml ultralytics
pip install telemetrik          # pure-Python GPMF parser (try this first)
# fallback if telemetrik gives trouble:
# pip install py-gpmf-parser
sudo apt-get install -y exiftool   # optional, for quick metadata checks
```
**🔎 Test:**
```bash
python -c "import numpy, cv2, pandas, ultralytics, telemetrik; print('all imports OK')"
```
**✅ Expected:** `all imports OK` and no traceback.
**Gate:** if an import fails, fix it before anything else.

---

## Step 1 — Inspect the video, recover the recording parameters

You don't need to remember the record settings — the file stores them.

**Do:**
```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames test.mp4
ffprobe -v error -show_entries stream=codec_tag_string -of default=nw=1 test.mp4 | grep -i gpmd
```
**🔎 Test:** read the numbers.
**✅ Expected:**
- width/height (e.g. `1920 x 1080`), and `r_frame_rate` like `60/1` or `30/1` → **write the fps down, you need it in Step 3**.
- The second command prints a line containing `gpmd` → **the GPMF telemetry track exists** (this is the GPS/IMU data). If it prints nothing, that video has no telemetry — try the other video.

**Gate:** you know fps + resolution, and you've confirmed a `gpmd` track exists.

---

## Step 2 — Extract telemetry locally (replace the website)

**Do:** write `pipeline/gpmf_extract.py` that reads `test.mp4` and writes `telemetry.csv` with columns
`time_s, lat, lon, alt, grav_x, grav_y, grav_z`. (telemetrik / py-gpmf-parser both expose GPS5/GPS9, GRAV, GYRO, CORI streams with per-sample timestamps.)
```bash
python pipeline/gpmf_extract.py test.mp4 telemetry.csv
head -5 telemetry.csv
```
**🔎 Test (two checks):**
1. Print the first rows and the GPS range:
   ```bash
   python -c "import pandas as pd; d=pd.read_csv('telemetry.csv'); print(d.head()); \
   print('lat',d.lat.min(),d.lat.max(),'| lon',d.lon.min(),d.lon.max(),'| n=',len(d))"
   ```
2. **Cross-check against the website CSV you already have** — same video, the lat/lon tracks should match.

**✅ Expected:**
- `lat`/`lon` are real coordinates for your test site (e.g. ~30.0x, 31.2x for Cairo), not zeros or NaN.
- ~10 GPS samples per second of video (a 60 s clip → ~600 rows).
- `grav_*` ≈ `(0, 0, 1)` (or whichever axis is "down") when the camera was held level → confirms GRAV decodes correctly.
- Local track ≈ website track.

**Gate:** local extraction matches the website. **Now you never need the website again.**

---

## Step 3 — Run YOLO and timestamp every detection

**Do:** write `pipeline/detect.py` that runs your trained model on sampled frames (every Nth frame; N = fps/10 gives ~10 fps) and writes `detections.csv`:
`frame_idx, frame_time, u, v, conf` where `frame_time = frame_idx / fps` (use the fps from Step 1) and `(u,v)` is the bbox center.
```bash
python pipeline/detect.py test.mp4 --fps 60 --out detections.csv
```
**🔎 Test:** draw the boxes on ~5 frames and look at them.
```bash
python pipeline/detect.py test.mp4 --fps 60 --save-overlay overlay/   # writes annotated frames
```
**✅ Expected:**
- Boxes land **on the flags**, not on background. False positives are visible here, not in a number at the end.
- `detections.csv` has rows with plausible `u` in `[0,width]`, `v` in `[0,height]`, `conf` in `[0,1]`.
- Note the **flag size in pixels** in the overlay — at altitude it'll be ~20–40 px. If the model misses small flags, that's your detection problem, found early.

**Gate:** you can see correct boxes and `frame_time` is filled in.

---

## Step 4 — Join detection time → telemetry (the sync), test it in isolation

**Do:** write `pipeline/sync.py` with a function that, given `detections.csv` + `telemetry.csv`, interpolates telemetry to each detection's `frame_time`:
`lat_uav, lon_uav = np.interp(t, tel.time_s, tel.lat/lon)`, plus GRAV at `t`. Output `synced.csv`:
`frame_time, u, v, conf, lat_uav, lon_uav, grav_x, grav_y, grav_z`.

**🔎 Test:** pick ONE detection and print it next to the two GPS samples that bracket it in time.
```bash
python pipeline/sync.py detections.csv telemetry.csv synced.csv --debug-row 0
```
**✅ Expected:** the interpolated `lat_uav, lon_uav` falls **between** the GPS sample just before and just after that timestamp. (If it's outside that range, your interpolation or time units are wrong — fix here, not later.)

**Gate:** one row verified by hand. This is the step the senior and the test doc call the #1 risk — prove it on a single row before trusting thousands.

---

## Step 5 — Heading (course-over-ground) and tilt from GRAV

**Do:** add to `sync.py`: `heading` from successive GPS positions (`atan2(dEast, dNorth)`), and `tilt_deg` = angle between GRAV and straight-down.

**🔎 Test:**
```bash
python -c "import pandas as pd,math; d=pd.read_csv('synced.csv'); \
print('heading deg sample:', d.heading.head().round(1).tolist()); \
print('tilt deg sample:', d.tilt_deg.head().round(1).tolist())"
```
**✅ Expected:**
- `heading` (0=North, 90=East) matches the direction the drone was actually flying in that clip.
- `tilt_deg` is small (a few degrees) for level flight, larger in turns. Not NaN, not 90.

**Gate:** heading points the right way; tilt is sane.

---

## Step 6 — Localize ONE detection → GPS (needs K + altitude)

You don't have calibration yet, so use **placeholder Hero 13 intrinsics** to test the *plumbing*; real K comes from the checkerboard at the flight.

**Do:** call `localize_full()` from `src/approach2.py` for a single synced row, with placeholder `K`, `dist=0`, a fixed `h` (your planned altitude, e.g. 80 m), and the row's `lat_uav, lon_uav, tilt(from GRAV), heading`.

**🔎 Test (two checks):**
1. Run the 5 existing unit tests (nadir center, east offset, yaw rotation, oblique rejection, altitude scaling) — code is in `FINAL_Pipeline_Document.md` Part 7. *(Already verified passing on the core math.)*
   ```bash
   pytest tests/ -v
   ```
2. Localize one real row and measure how far the result is from the drone itself.
```python
# the flag must be on the ground near the drone — not kilometres away
```
**✅ Expected:**
- All 5 unit tests **pass**.
- The single-frame flag GPS is **within ~100 m of `lat_uav/lon_uav`** (at 80 m altitude the flag can't be far). If it's wildly off, the rotation/sign is wrong — debug with the unit tests, not the video.

**Gate:** unit tests green + one real frame gives a plausible nearby coordinate.

> Note: results here are only as good as the **placeholder** K and the guessed `h`. Real accuracy waits for calibration + a real altitude source — but the *logic* is proven now.

---

## Step 7 — Aggregate many frames → one flag GPS

**Do:** use the aggregation in `src/approach2.py` (median for v1, confidence-weighted if time). Input: all per-frame estimates for a flag. Output: one `(lat, lon)`.

**🔎 Test:** feed it synthetic input you control.
```python
# 20 estimates all at (30.0010, 31.0010) + tiny noise, plus ONE outlier at (30.05, 31.05)
# median/weighted-mean should return ~ (30.0010, 31.0010) and ignore the outlier
```
**✅ Expected:** result ≈ the cluster, outlier rejected. Confirms aggregation is robust before real data.

**Gate:** outlier doesn't move the answer.

---

## Step 8 — End-to-end run on the real videos

**Do:** write `pipeline/localization.py` chaining Steps 2→3→4→5→6→7 → `results.json` (one GPS per flag).
```bash
python pipeline/localization.py test.mp4 --camera configs/camera_params_hero13.yaml --alt 80 --out results.json
cat results.json
```
**🔎 Test:** run on **both** existing videos.
**✅ Expected:** `results.json` lists a coordinate per detected flag, each near the flight path. No crashes, no NaN. Times to run should be seconds-to-minutes (use frame sampling — don't process every frame; this is how you avoid the "7-min video took 15 min" problem).

**Gate:** full chain runs unattended and prints coordinates.

---

## Step 9 — Validate against ground truth

**Do:** use `validation.py` (haversine error, mean/median/max, % within 20 m) from `FINAL_Pipeline_Document.md` Part 7. Person C surveys 2–3 known points.

**🔎 Test:** run a handheld clip walked over a surveyed target (do this **today**, on the ground).
**✅ Expected:** a printed error per flag in metres and the % under 20 m. This is your real score proxy. (Handheld error will be larger than flight because altitude/tilt differ — you're testing the software, not the flight yet.)

**Gate:** you can produce an error number against a known point.

---

## Step 10 — Submission rehearsal

**Do:** export `results.json` → the competition's required coordinate format onto an empty FAT32 USB. Time the loop: land → pull SD → copy → run → export → submit.
**🔎 Test:** run the whole loop on a recorded clip with a stopwatch.
**✅ Expected:** under 10 minutes, file on USB in the right format. If it's slow, increase frame sampling (Step 3) or reduce resolution processed.

**Gate:** you can deliver a correctly-formatted file inside the time window.

---

## Order of attack for today (parallel, 3 people)

| Person | Steps | Needs camera? |
|---|---|---|
| **A** | 0, 1, 2 (telemetry extraction), then calibration at flight | calibration only |
| **B** | 3, 4, 5, 6, 7, 8 (the geometry + chain) | no — use placeholders |
| **C** | 9, 10 (ground truth, validation, submission) | no |

**Stop-and-check rule:** nobody moves to their next step until the current step's ✅ Expected output is met. If you're ever unsure what a stage produced, you skipped a test — go back one step.
