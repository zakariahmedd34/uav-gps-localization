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
| `gpmf_extract.py` | GoPro `.MP4` | `telemetry.csv` (GPS + gravity); uses `gps9_parser.py` for Hero 13 |
| `detect.py` | video + YOLO weights | `detections.csv` (bbox centre + timestamp per detection) |
| `sync.py` | detections + telemetry | `synced.csv` (drone position, heading, tilt per detection) |
| `localization.py` | `synced.csv` | `results.json` (flag GPS coordinates) |
| `run_pipeline.py` | video | runs stages 1–4, writes everything to `artifacts/` |
| `calibrate_camera.py` | checkerboard photos | camera intrinsics YAML |
| `tests/test_geometry.py` | — | geometry unit tests |

## Install & run

```bash
pip install -r requirements.txt

python pipeline/run_pipeline.py --video data/ground_truth/G1.MP4 --alt 80
pytest tests/ -v
```

Main options: `--alt` = camera height above the target (AGL on a flight, not MSL);
`--camera` = calibration YAML; `--heading` = fixed heading for stationary clips.

## Remaining work

- Run `calibrate_camera.py` on the Hero 13; until then a placeholder lens is used and
  distances are not accurate.
- Provide a real above-ground altitude (the GoPro GPS altitude is MSL, not AGL).
- Add validation against surveyed points (error, % within 20 m).
- Add the USB submission export format.

## Known weaknesses

- **Calibration YAML mismatch:** `calibrate_camera.py` writes `camera_matrix`/
  `dist_coeffs`, but `localization.py` expects `K`/`dist` — reconcile before `--camera`
  works.
- **Heading** is the weakest input: needs a known azimuth (or course-over-ground on a
  moving flight); errors rotate the result.
- **Needs a GPS fix:** no satellite lock means invalid coordinates (use `--fake-gps`
  to test geometry).
- **Altitude-sensitive:** error in `--alt` maps almost directly into ground error.
- **Small targets:** a flag is ~20–40 px at altitude; detection must be verified there.
- **Flat-ground assumption:** terrain relief is not modelled.
```
