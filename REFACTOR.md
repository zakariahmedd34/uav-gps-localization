# Refactor notes (September 2026)

What changed in the project, why, and what now covers each removed file or folder.
The detection and localization algorithm was **not** changed.

## 1. Results are unchanged (verified)

The old code (commit `da9ba94` plus the uncommitted GPU work) and the new code were run
side by side on the same videos with the same settings. Output files were compared byte by byte.

| Video / settings | Detections old = new | `targets.csv`, `candidates.csv`, all crops |
|---|---|---|
| G2 full video, GPU, 1 fps, min 3, top 3, imgsz 1280 | 418 = 418 | identical |
| G2 03:20-04:20, GPU, same settings | 192 = 192 | identical |
| G2 03:20-03:50, CPU, same settings | 88 = 88 | identical |
| G2 03:20-03:50, GPU, `--sample-fps 3 --min-frames 5 --imgsz 960 --top-k 0` | 277 = 277 | identical |
| G3 02:00-05:30 (5.3K, calibration auto-scaled): old `localization.py` vs new on the same data | - | identical |
| Telemetry extraction (G2) | `telemetry.csv` bit-identical | - |

Speed is unchanged (same run: 81 s old, 69-81 s new). The geometry tests pass.

## 2. Structure: old -> new

| Old | New | Notes |
|---|---|---|
| `pipeline/run_pipeline.py` (ran 4 scripts as subprocesses) | `main.py` | One process, one CLI. Steps are plain function calls. |
| `pipeline/gpmf_extract.py` | `pipeline/telemetry.py` -> `extract()` | Same GPS gate, spike removal, AGL anchor. |
| `pipeline/gps9_parser.py` | `pipeline/telemetry.py` -> `_read_gps9()` + helpers | Same GPS9 decoder. |
| `pipeline/sync.py` | `pipeline/telemetry.py` -> `sync()` | Same interpolation and GPS course heading. |
| `pipeline/detect.py` | `pipeline/detect.py` -> `detect()` | Same sampling, seek/decode paths, FP16, crops. |
| `pipeline/localization.py` | `pipeline/localize.py` -> `localize()` | Same geometry, gates, tracklets, DBSCAN, median, top-k. |
| `pipeline/logger.py` | Standard `logging`, set up in `main.py` | Same log file in `logs/`; now also printed to the console. |
| `pipeline/exception.py` (`CustomException`) | Python traceback, logged by `main.py` | The traceback already shows file and line. |
| `configs/camera_params_hero13.yaml` | `calibration/hero13_linear_4k.yaml` | Same numbers; header comment added. |
| `pipeline/weights/best.pt` | `models/best.pt` | Same file (not in git). |

Python code went from 2,245 lines in 11 files to 864 lines in 4 files.

## 3. Removed, and what covers it

| Removed | Why it was safe | What covers it now / how to get it back |
|---|---|---|
| `approachs/` (approach1-3.py) | Nothing imported them (approach3 only imported approach2). Early experiments. | The working method is in `pipeline/localize.py`. Old files: `git show da9ba94:approachs/approach2.py`. |
| `notebook/` (2 notebooks) | Not used by the pipeline. | Old files in git history. |
| `doc/` (12 files) | Replaced by one README. | `README.md` (usage, pipeline, assumptions) and this file. Old docs in git history. The 2 untracked post drafts were moved to `..\uav-gps-localization_doc_drafts_backup\`, not deleted. |
| `configs/camera_params_hero13.SAMPLE.yaml` | Fake numbers, never used for real runs. | `calibration/hero13_linear_4k.yaml` (real calibration). |
| `configs/camera_params_hero13_5p3k.yaml` | Only 4K values times 1.383. | `localize.load_camera()` scales K automatically when the video is not 4K (the log warns). |
| `pipeline/calibrate_camera.py`, `pipeline/extract_calib_frames.py` | Only used once to create the calibration file, not part of a run. | Calibration result is kept. To recalibrate: `git show da9ba94:pipeline/calibrate_camera.py > calibrate_camera.py`. |
| Placeholder camera intrinsics (fx=1500) | Gave silently wrong coordinates if the yaml was missing. | Missing calibration now stops with a clear error. |
| GPS5 reading | Only for older GoPros. Every Hero 13 log said "No GPS5 found -- falling back to GPS9". | GPS9 decoder (always used on Hero 13). |
| ffprobe sanity check | Only printed info; ffprobe is not installed, so it was always skipped. | Missing telemetry track still raises a clear error. |
| `utc_s` column | Kept only for a future Pixhawk log join; nothing read it. | Can be re-added from GPS9 fields `days_since_2000`, `secs_since_midnight` (still decoded). |
| `tilt_deg` column in `synced.csv` | Diagnostic only; localization uses the gravity vector directly. | Log reports how many rays the off-nadir gate rejected. |
| `results.csv` (it contained JSON) | Duplicate of `submission/targets.csv`. | `submission/targets.csv`. |
| `--no-tracklets` (raw-point clustering) | Old alternative path, never used by the pipeline. | Tracklet clustering (the path that was always used). |
| `--heading`, `--fake-gps`, `--debug-row` | Only for stationary handheld test clips / debugging. | Real flights use GPS course heading (unchanged). |
| `--hwdec` | Opt-in, off by default, performance only; can silently give zero detections on some drivers. | Software decode (what every real run used). |
| `--overlay` | Debug images only, off by default. | `artifacts/crops/` and `submission/candidates_flags/` show every detection. |
| `--keep-artifacts`, `--artifacts`, `--model`, `--camera` | Paths/cleanup options. | Fixed paths: `artifacts/`, `models/best.pt`, `calibration/`. Old crops are always cleared. |
| `--no-half` | Performance only. | FP16 is on automatically with `--gpu`. |
| `--cluster-m`, `--max-offnadir`, min/max AGL, confidence | Not in the real command. | Named constants at the top of `pipeline/localize.py` / `detect.py` (same values: 12 m, 30 deg, 20-150 m, 0.4). |
| `pytest` dependency | Tests run with plain Python. | `python tests/test_geometry.py`. |

## 4. Command line

Old:
```bash
python pipeline\run_pipeline.py --video X.MP4 --device 0 --top-k 3 --sample-fps 1 --min-frames 3 --imgsz 1280
```
New (same result):
```bash
python main.py --video X.MP4 --gpu --top-k 3 --sample-fps 1 --min-frames 3 --imgsz 1280
python main.py --video X.MP4 --gpu            # identical: those are the defaults
```

| Old argument | New | Reason |
|---|---|---|
| `--video` | `--video` | Required input. |
| `--start`, `--end` | `--start`, `--end` | Section of the video (MM:SS, HH:MM:SS or seconds). |
| `--device 0` | `--gpu` | Simpler. FP16 still automatic. Error if CUDA is missing. |
| `--sample-fps` | `--sample-fps` (default 1) | Kept: changes how many detections each flag gets. |
| `--min-frames` | `--min-frames` (default 3) | Kept: decides which candidates become flags. |
| `--top-k` | `--top-k` (default 3, 0 = no limit) | Kept: max flags submitted. |
| `--imgsz` | `--imgsz` (default 1280) | Kept: detection quality. 1280 is the model's training size (read from `best.pt`). |
| `--alt` | `--alt` | Kept: the only fallback when the AGL anchor fails. |

## 5. Behaviour differences you may notice

- Without `--gpu` the run is on CPU. The old code used the GPU automatically when available.
- Log messages are also printed in the terminal (before: only in `logs/`).
- `artifacts/results.csv` is no longer written; read `artifacts/submission/targets.csv`.
- `synced.csv` has no `utc_s` / `tilt_deg` columns; `telemetry.csv` has no `utc_s`.
- A missing calibration file stops the run instead of using fake intrinsics.

## 6. Getting old files back

Everything removed from git is still in history:
```bash
git show da9ba94:pipeline/run_pipeline.py      # print an old file
git checkout da9ba94 -- doc/                     # restore a whole folder
```
