# UAV Flag Localization

Find the GPS location of flags seen in GoPro aerial video, to within ~20 m, fully
offline. Built for the ICMTC 2026 UAVC-9 Fixed-Wing Challenge.

Only the GoPro Hero 13 video is needed. Its embedded GPMF metadata already holds GPS,
a gravity vector (camera tilt), and gyro, all on the same clock as the video frames,
so no separate flight-controller log or time syncing is required.

## How it works

For each frame with a detected flag: turn the pixel into a viewing ray, rotate it to
the world using gravity (tilt) and heading, intersect the ground at the known height,
then convert to GPS. A flag is seen in many frames, so the estimates are clustered and
averaged into one coordinate.

## Pipeline

```
GoPro .MP4
   ├─ detect.py        → detections.csv   (frame_time, u, v, conf)
   └─ gpmf_extract.py  → telemetry.csv    (time_s, lat, lon, alt, grav_x/y/z)
                              │
                          sync.py          → synced.csv   (join by time + heading + tilt)
                              │
                       localization.py     → results.json (one GPS per flag)

run_pipeline.py runs all four in order; outputs go to artifacts/.
```

## Files

| File | Input | Output |
|------|-------|--------|
| `gpmf_extract.py` | GoPro `.MP4` | `telemetry.csv` (GPS gated by fix/dop, AGL anchored to a stationary window, GPS-UTC kept); uses `gps9_parser.py` for Hero 13 |
| `detect.py` | video + YOLO weights | `detections.csv` (bbox centre + timestamp per detection) |
| `sync.py` | detections + telemetry | `synced.csv` (drone position, heading, tilt, utc_s per detection) |
| `localization.py` | `synced.csv` | `results.json` (off-nadir gate → DBSCAN → geometric median → top-K) |
| `run_pipeline.py` | video | runs stages 1–4, writes everything to `artifacts/` |
| `calibrate_camera.py` | checkerboard photos | camera intrinsics YAML |
| `tests/test_geometry.py` | — | geometry + clustering unit tests |

**See `doc/PIPELINE_V2.md`** for what changed after the first mission run, the
test-day checklist, and the post-flight decision rules. Read it before flying.

## Install & run

```bash
pip install -r requirements.txt
# GPU (strongly recommended — CPU runs take 30+ min): install the CUDA torch build
# per machine, then verify torch.cuda.is_available() is True:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# competition run (top-3 flags, GPU):
python pipeline/run_pipeline.py --video MISSION.MP4 --device 0 --top-k 3

# testing (no cap; add --alt if the AGL anchor warning appears in the log):
python pipeline/run_pipeline.py --video data/ground_truth/G2.MP4 --device 0 --top-k 0

python tests/test_geometry.py   # or pytest tests/ -v
```

Main options: `--alt` = constant camera height above target (overrides the AGL
column); `--top-k` = max flags reported (0 = no cap); `--imgsz` = YOLO input size
(default 1920 — don't reduce if recall matters); `--device 0` = GPU;
`--start/--end` = process only a time window (big runtime saver);
`--camera` = calibration YAML; `--heading` = fixed heading for stationary clips.

**Iterating on localization only** (detect/sync already done — takes seconds):

```bash
python pipeline/localization.py artifacts/synced.csv \
  --camera configs/camera_params_hero13.yaml \
  --max-offnadir 30 --cluster-m 12 --top-k 0 --out artifacts/results.csv
```

## Competition context (ICMTC 2026 UAVC-9)

- Targets are **2 m × 1 m flags lying flat on the ground** (no pole) → anchor = bbox centroid.
- **Accuracy tolerance is 20 m** for full points (down to 60 m for partial) → aim for robust ≤20 m, not cm-precision.
- **Flat desert airfield, 50–100 m AGL** → flat-ground assumption is valid; **no DEM needed**.
- The aircraft has an **autopilot + GCS**; AGL/GPS/attitude come from the **flight log (control team)**. GoPro GPMF is backup.
- Mission 1: locate **2 flags in the search area + 1 bonus flag** in the geofence.
- **Submission = GPS coordinates + a cropped image per flag on a USB drive.**
- See `documentation.md` §0.4 for the full addendum and the questions to ask the control team.

## Remaining work

- ~~Calibrate the Hero 13~~ **DONE** — `configs/camera_params_hero13.yaml`
  (**Linear lens** — the flight MUST use Linear or this yaml is invalid).
- ~~USB submission export~~ **DONE** — `artifacts/submission/targets.csv` + `flags/` crops.
- ~~GPS gating / AGL anchor / DBSCAN / top-K~~ **DONE** — see `doc/PIPELINE_V2.md`.
- **Validate end-to-end against surveyed points** (error in metres, % within 20 m) —
  still the biggest open item; no flight with ground truth exists yet.
- Pixhawk hybrid (join `RelHomeAlt`/attitude by GPS-UTC — `utc_s` is already in
  `synced.csv`) if the flight shows tilt/heading-dominated errors.
- Two-stage detect→classify for flag identity (10 pts): current plan is a HUMAN reads
  the 3 crops and types the country names — don't automate under time pressure.

## Known weaknesses

- **Heading** is the weakest input (GPS course-over-ground ≠ true yaw in wind/crab).
  Mitigated by the 30° off-nadir gate + straight overflight passes; NOT validated
  against ground truth yet.
- **Lens mode is a hard dependency:** the calibration is Linear-only. Wide footage
  through this yaml gives radius-dependent errors up to tens of metres (the first
  mission run proved it). Verify the camera setting before EVERY recording.
- **Needs a GPS fix before recording starts:** the gate drops pre-lock samples, but
  nothing can recover position for detections made while GPS was bad. Power on early,
  wait for the lock icon, record 60 s stationary (that window is also the AGL anchor).
- **Altitude:** `--alt` error maps ~1:1 into ground error at off-nadir angles (near
  nadir it barely matters — another reason for the gate).
- **Small targets:** a 2 m flag is ~6–10 px at YOLO's old 640 input — keep
  `--imgsz 1920`; verify recall on the overlay frames after every run.
- **Flat-ground assumption:** fine for the 6th October airfield (confirmed ~flat).
```
