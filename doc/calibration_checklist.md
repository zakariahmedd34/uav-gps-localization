# Camera Calibration Checklist (Hero 13)

**Golden rule:** a calibration is valid only for the EXACT camera config it was shot in
(resolution + lens mode + stabilization). Change any of those later → recalibrate.

## Step 0 — Lock the flight camera config FIRST
Decide and write down the settings you will actually fly, because calibration must match them:
- [ ] Resolution (e.g. 4K 3840×2160) — same as detection runs.
- [ ] Lens mode = **Linear** (recommended; near-pinhole, easy to calibrate). Avoid Wide/SuperView/HyperView.
- [ ] **HyperSmooth stabilization = OFF**, **Horizon Lock = OFF** (they warp the image per-frame and break the model).
- [ ] Frame rate, shutter, ISO, etc. (don't affect intrinsics, but lock them for consistency).
→ Confirm with the control team that this config is acceptable on the aircraft (mounting, power, recording).

## Step 1 — Get a checkerboard
- [ ] Generate a proper pattern (use **calib.io/camera-calibrator** "checkerboard", or any clean 9×6 / 10×7 board). Printing a random Google image is fine ONLY if it's a true checkerboard with even squares.
- [ ] Print **without "fit to page" scaling** (that distorts the aspect ratio). Print as large as you can — A3 is better than A4 for filming at distance.
- [ ] Mount it **dead flat** on something rigid: foam board, clipboard, glass, or a hard table. A bent board ruins calibration.
- [ ] Use **matte** paper (glossy causes glare the corner finder hates).
- [ ] After printing, **measure one square with a ruler** and use that real number as `--square-size` (in metres). Don't trust the nominal size.
- [ ] Count **internal corners** (squares − 1 each way): a 10×7-square board = 9×6 internal corners → `--rows 6 --cols 9`.

## Step 2 — Shoot a VIDEO (recommended over photos)
Why video: GoPro Photo mode ≠ Video mode (different FOV/resolution). You process video, so calibrate from video.
- [ ] Record a 1–2 min clip **in the Step-0 flight config**.
- [ ] Move the board so it visits: center, **all four corners**, the **edges**, and several **tilts** (20–45°) and a few **distances**. Edges/corners matter most — that's where distortion lives.
- [ ] Move **slowly / pause** (no motion blur), keep it **well-lit**, keep the **whole board in frame** every time.

## Step 3 — Extract frames
```
python pipeline\extract_calib_frames.py --video YOUR_CALIB.MP4 --out data\checkerboard --rows 6 --cols 9 --target 25
```
- [ ] ~20–25 sharp, well-spread frames land in `data\checkerboard\`.

## Step 4 — Calibrate
```
python pipeline\calibrate_camera.py --images "data\checkerboard\*.jpg" --rows 6 --cols 9 --square-size 0.025 --out configs\camera_params_hero13.yaml
```
- [ ] Replace `0.025` with your measured square size (m), and `--rows/--cols` with your board.
- [ ] **RMS reprojection error < 1.0 px.** If higher → add more/better-spread frames and re-run before trusting it.

## Step 5 — Verify
```
python pipeline\localization.py artifacts\synced.csv --alt 1.5 --camera configs\camera_params_hero13.yaml --out artifacts\results.json
```
- [ ] Prints `Camera: loaded ... fx=... fy=... @ WxH rms=...px` (no crash, no PLACEHOLDER warning).
- [ ] Optional eyeball test: undistort one frame and check straight edges look straight.

---

## To confirm with the CONTROL TEAM today
Telemetry / autopilot (drives the whole localization):
- [ ] Does the autopilot log **GPS lat/lon, altitude, and full attitude (roll, pitch, and YAW/heading)** with timestamps? At what rate?  (We need true heading, NOT GPS course-over-ground.)
- [ ] **AGL:** direct from a downward **rangefinder/lidar**, or only **barometric relative to home/takeoff**? What's the reference point?
- [ ] Altitude **datum**: relative-to-home / AMSL / ellipsoidal (HAE)?
- [ ] **Time sync:** is there a **GPS/UTC timestamp** in the log? Can we mark a sync event (note takeoff instant, or do a visible wing-rock / LED) to align log ↔ GoPro video?
- [ ] Autopilot firmware + log format: **ArduPilot `.bin`** or **PX4 `.ulog`**? Can we get the file **immediately after each flight** (within the submission window)?
- [ ] **GPS quality:** standard GPS or **RTK/DGPS**? Expected horizontal accuracy? (sets our accuracy floor)

Hardware / integration (decide together):
- [ ] Is the GoPro **rigidly mounted** to the airframe, and at what **angle** (nadir or tilted)? Will it move/gimbal? (Fixed + known angle = better; lock it before calibrating.)
- [ ] Final **camera config** (Step 0) OK on the aircraft?

Mission data:
- [ ] Will the GCS give us **geofence / search-area / waypoint coordinates digitally** (so we can label in-area vs the bonus flag)?
- [ ] Is the search area **flat relative to takeoff** (any metres of elevation difference)? (confirms flat-ground assumption)

For the JUDGES (not control team), when possible:
- [ ] Does "identify the flag" mean **confirm it's a flag**, or **name the country**? (decides whether the 100-class classifier is essential)
