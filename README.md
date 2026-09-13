# UAV Flag Localization

Finds the GPS coordinates of flags lying on the ground in GoPro Hero 13 aerial video,
fully offline. Built for the ICMTC 2026 UAVC-9 Fixed-Wing Challenge (2 m x 1 m flags,
20 m accuracy tolerance, 50-100 m AGL over flat ground).

Everything comes from the video file: the GoPro stores GPS and a gravity vector
(camera tilt) in its GPMF metadata track on the same clock as the frames, so no
flight-controller log or time sync is needed.

## Pipeline

```
GoPro .MP4
  1. telemetry  read GPS9 + gravity from the video, drop bad GPS fixes and spikes,
                AGL = altitude minus the ground altitude from a stationary window
  2. detect     YOLO on 1 frame per second inside --start/--end, save a crop per detection
  3. sync       interpolate position, AGL, gravity and heading (GPS course) to each detection
  4. localize   pixel -> undistorted camera ray -> rotate by gravity + heading
                -> reject rays > 30 deg off vertical -> intersect ground -> lat/lon
                -> group detections into flags -> keep the top 3
```

## Requirements

- Python 3.10+
- An NVIDIA GPU is optional but strongly recommended (a full video on CPU takes 30+ min)
- `models/best.pt`: the trained YOLO weights (not in git, copy it in)

## Installation

```bash
pip install -r requirements.txt

# for --gpu: install the CUDA build of PyTorch, then check torch.cuda.is_available() is True
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Usage

Copy the video from the GoPro to a normal folder first. Python cannot open the
`This PC\HERO13 Black\...` (MTP) path directly.

```bash
# CPU
python main.py --video "path/to/GX014771.MP4"

# GPU
python main.py --video "path/to/GX014771.MP4" --gpu

# only a section of the video (MM:SS, HH:MM:SS or seconds)
python main.py --video "path/to/GX014771.MP4" --start 02:00 --end 05:30 --gpu
```

**Optional tuning.** The defaults are the tested mission settings, so you can leave these out:

| Flag | Default | What it changes |
|------|---------|-----------------|
| `--sample-fps` | 1 | Frames analysed per second of video. More gives more detections per flag but is slower. |
| `--min-frames` | 3 | Detections a flag needs to be submitted. Raise it with `--sample-fps` (e.g. 3 fps -> 5). |
| `--top-k` | 3 | Max flags submitted (mission: 2 + 1 bonus). `0` = no limit. |
| `--imgsz` | 1280 | YOLO input size (the model's training size). Lower is faster but misses small flags. |
| `--alt` | off | Fixed camera height above the ground (m). Only if the log warns the AGL anchor failed. |

```bash
# e.g. denser sampling on a short section
python main.py --video "path/to/GX014771.MP4" --start 02:00 --end 03:00 --gpu --sample-fps 3 --min-frames 5
```

**Outputs** (`artifacts/`):

- `submission/targets.csv` + `submission/flags/`: the official result, one GPS point
  and one crop per flag (what goes on the USB drive).
- `submission/candidates.csv` + `submission/candidates_flags/`: every candidate,
  marked OFFICIAL or filtered_out. Review it before submitting: a real flag may have
  been filtered out.
- `telemetry.csv`, `detections.csv`, `synced.csv`, `crops/`: intermediate data for debugging.
- The run log is written to `logs/`.

Tests: `python tests/test_geometry.py`

## Project structure

```
main.py                         command-line entry point (runs the 4 steps)
pipeline/
  telemetry.py                  GPS9 + gravity extraction, GPS cleaning, AGL, sync
  detect.py                     YOLO detection + crops
  localize.py                   geometry, clustering, submission files
calibration/hero13_linear_4k.yaml   camera intrinsics + distortion
models/best.pt                  YOLO weights (not in git)
tests/test_geometry.py          geometry and clustering tests
data/                           ground-truth material and test videos (videos not in git)
```

Other fixed settings (confidence threshold, 30 deg off-nadir limit, 12 m clustering radius,
GPS quality gates) are named constants at the top of each file in `pipeline/`.

## Important assumptions

- **Camera: GoPro Hero 13 Black, Linear lens, 4K.** The calibration is only valid for
  Linear. Other resolutions are handled by scaling the calibration, but 4K Linear is
  the most accurate. Wide/SuperView footage gives errors of tens of metres.
- **GPS must be locked before recording starts.** Samples without a 3D fix are dropped.
  Record about 60 s stationary on the ground at the start or end: that window is the
  AGL reference.
- **Flat ground** under the flags (no terrain model).
- **Heading comes from GPS course-over-ground**, which is not the true yaw in crosswind.
  The 30 deg off-nadir gate limits the effect. Fly straight passes over the area.
- YOLO runs at 1280 px, the size the model was trained at. A 2 m flag is only a few
  pixels at lower sizes.
