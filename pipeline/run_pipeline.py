"""
run_pipeline.py — end-to-end orchestrator.

Runs the whole chain on one GoPro video and writes every intermediate to the
artifacts/ directory, then prints the final flag coordinates:

    video.mp4
      ├─[1] gpmf_extract.py  ->  artifacts/telemetry.csv
      ├─[2] detect.py        ->  artifacts/detections.csv
      ├─[3] sync.py          ->  artifacts/synced.csv
      └─[4] localization.py  ->  artifacts/results.json

Each stage is still runnable on its own; this just connects them.

Usage:
    python pipeline/run_pipeline.py --video data/ground_truth/G1.MP4 --alt 80
    python pipeline/run_pipeline.py --video data/ground_truth/G1.MP4 --alt 1.5 \
        --heading 0 --sample-fps 5
"""

import os
import sys
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))      # .../pipeline
ROOT = os.path.dirname(HERE)                            # repo root
PY = sys.executable


def run(cmd, cwd=None, label=""):
    print(f"\n{'='*70}\n[{label}] {' '.join(str(c) for c in cmd)}\n{'='*70}")
    r = subprocess.run([str(c) for c in cmd], cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"\nPIPELINE STOPPED: stage '{label}' failed "
                 f"(exit {r.returncode}). Fix it and re-run.")


def main():
    ap = argparse.ArgumentParser(description="End-to-end flag localization pipeline")
    ap.add_argument("--video", default=os.path.join(ROOT, "data/ground_truth/G1.MP4"),
                    help="GoPro .mp4/.MP4 to process")
    ap.add_argument("--alt", type=float, required=True,
                    help="Camera height ABOVE the flag (m). Flight = AGL; handheld test ~1.5")
    ap.add_argument("--model", default=os.path.join(HERE, "weights/best.pt"))
    ap.add_argument("--camera", default=os.path.join(ROOT, "configs/camera_params_hero13.yaml"),
                    help="Calibration yaml. Ignored (placeholder used) if empty/missing.")
    ap.add_argument("--heading", type=float, default=None,
                    help="Fixed heading (deg) for stationary clips; omit on a real flight (uses COG)")
    ap.add_argument("--sample-fps", type=float, default=5.0)
    ap.add_argument("--artifacts", default=os.path.join(ROOT, "artifacts"),
                    help="Where all intermediate + output files go")
    args = ap.parse_args()

    video = os.path.abspath(args.video)
    if not os.path.isfile(video):
        sys.exit(f"Video not found: {video}")
    art = os.path.abspath(args.artifacts)
    os.makedirs(art, exist_ok=True)

    telemetry = os.path.join(art, "telemetry.csv")
    detections = os.path.join(art, "detections.csv")
    synced = os.path.join(art, "synced.csv")
    results = os.path.join(art, "results.json")
    overlay = os.path.join(art, "overlay")

    print(f"Video:     {video}\nArtifacts: {art}")

    # [1] telemetry — gpmf_extract.py writes telemetry.csv into its cwd
    run([PY, os.path.join(HERE, "gpmf_extract.py"), video],
        cwd=art, label="1/4 GPMF extract")

    # [2] detections
    run([PY, os.path.join(HERE, "detect.py"),
         "--model", args.model, "--source", video,
         "--out", detections, "--sample-fps", args.sample_fps,
         "--save-overlay", overlay],
        label="2/4 YOLO detect")

    # [3] sync
    sync_cmd = [PY, os.path.join(HERE, "sync.py"), detections, telemetry, synced]
    if args.heading is not None:
        sync_cmd += ["--heading", args.heading]
    run(sync_cmd, label="3/4 sync")

    # [4] localize  (only pass --camera if the yaml has real content)
    loc_cmd = [PY, os.path.join(HERE, "localization.py"), synced,
               "--alt", args.alt, "--out", results]
    if os.path.isfile(args.camera) and os.path.getsize(args.camera) > 0:
        loc_cmd += ["--camera", args.camera]
    else:
        print("NOTE: camera yaml empty/missing — localization uses PLACEHOLDER "
              "intrinsics. Calibrate the Hero 13 for accurate metres.")
    run(loc_cmd, label="4/4 localize")

    print(f"\n{'='*70}\nDONE. Outputs in {art}/")
    print(f"  telemetry.csv  detections.csv  synced.csv  results.json  overlay/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
