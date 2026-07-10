"""
author: @Zakaria_34

run_pipeline.py - run the whole flag-localization pipeline end to end

video --> (1) gpmf_extract -output-> telemetry.csv
          (2) detect       -output-> detection.csv
          (3) sync         -output-> synced.csv
          (4) localization -output-> results.csv    

all stages log into ONE shared file under log/.
"""
# setup tools
import os
import sys
import subprocess
import argparse
import datetime

HERE = os.path.dirname(os.path.abspath(__file__)) # ./uav-gps-localization/pipeline
ROOT = os.path.dirname(HERE) # project root folder ./uav-gps-localization
PY = sys.executable

#shared pipeline log
LOGS_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
os.environ["PIPELINE_LOG_FILE"] = os.path.join(
    LOGS_DIR, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log"
)

from logger import logging
from exception import CustomException

def run(cmd, cwd=ROOT, label=""):
    cmd = [str(c) for c in cmd]
    logging.info(f"[{label}] START: {" ".join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd)
    except Exception as e:
        logging.error(f"[{label}] could not launch")
        raise CustomException(e, sys) from e
    if result.returncode != 0:
        logging.error(f"[{label}] FAILED with returncode {result.returncode}")
        sys.exit(f"Pipeline stopped:stage '{label}' failed.see thelig")  
    logging.info(f"[{label}] DONE")

def parse_args():
    """
            Terminal input
                ↓
            argparse
                ↓
            args object
                ↓
            pipeline uses args.*
    """
    ap = argparse.ArgumentParser(description="End-to-end flag localization pipeline")
    ap.add_argument("--video", default=os.path.join(ROOT, "data/ground_truth/G1.MP4"),
                    help="GoPro .mp4 to process")
    ap.add_argument("--alt", type=float, default=None,
                    help="Constant height above the flag (m). Omit to auto-use per-detection "
                         "AGL from telemetry; pass a value (e.g. 1.5) for handheld clips.")
    ap.add_argument("--model", default=os.path.join(HERE, "weights/best.pt"),
                    help="YOLO weights")
    ap.add_argument("--camera", default=os.path.join(ROOT, "configs/camera_params_hero13.yaml"),
                    help="Calibration yaml (K + dist)")
    ap.add_argument("--heading", type=float, default=None,
                    help="Fixed heading (deg) for stationary clips; omit on a real flight")
    ap.add_argument("--sample-fps", type=float, default=3.0, help="Detections per second")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Max flags to report (mission 1: 2 + 1 bonus = 3). "
                         "Pass 0 to disable the cap.")
    ap.add_argument("--max-offnadir", type=float, default=30.0,
                    help="Reject rays more than this off vertical (deg)")
    ap.add_argument("--imgsz", type=int, default=1920,
                    help="YOLO inference size — small flags at 50-100 m AGL need "
                         ">=1920 (640 shrinks a 2 m flag below detectability)")
    ap.add_argument("--device", default=None,
                    help="YOLO device: 0 for GPU, cpu for CPU (default: auto)")
    ap.add_argument("--start", default=None, help="Process FROM this time (HH:MM:SS or seconds)")
    ap.add_argument("--end", default=None, help="Process UP TO this time (HH:MM:SS or seconds)")
    ap.add_argument("--artifacts", default=os.path.join(ROOT, "artifacts"),
                    help="Where all intermediate + output files go")
    return ap.parse_args()


def main():
    args = parse_args()

    video = os.path.abspath(args.video)
    if not os.path.isfile(video):
        sys.exit(f"Video not found: {video}")
    art = os.path.abspath(args.artifacts)
    os.makedirs(art, exist_ok=True)

    telemetry = os.path.join(art, "telemetry.csv")
    detections = os.path.join(art, "detections.csv")   # ONE canonical name
    synced = os.path.join(art, "synced.csv")
    results = os.path.join(art, "results.csv")
    overlay = os.path.join(art, "overlay.mp4")
    crops = os.path.join(art, "crops")
    submit = os.path.join(art, "submission")

    logging.info(f"Pipeline start | video={video} | artifacts={art}")

    # video 
    # --> (1) gpmf_extract -output-> telemetry.csv
    run([PY,os.path.join(HERE, "gpmf_extract.py"), video],
    cwd = art, label= "1/4 gpmf_extract")
    
    #(2) detect       -output->  detection.csv

    detect_cmd = [PY, os.path.join(HERE, "detect.py"),
                  "--model", args.model, "--source", video,
                  "--out", detections, "--sample-fps", args.sample_fps,
                  "--imgsz", args.imgsz,
                  "--save-overlay", overlay, "--save-crops", crops]
    if args.device is not None:
        detect_cmd += ["--device", args.device]
    
    # optional time window
    if args.start is not None:
        detect_cmd += ["--start", args.start]
    if args.end is not None:
        detect_cmd += ["--end", args.end] 
       
    run(detect_cmd, label="2/4 detect")

    
    #(3) sync         -output-> synced.csv
    sync_cmd = [PY, os.path.join(HERE, "sync.py"), detections, telemetry, synced]
    if args.heading is not None:
        sync_cmd += ["--heading", args.heading]
    run(sync_cmd, label="3/4 sync")

    #(4) localization -output-> results.csv
    loc_cmd = [PY, os.path.join(HERE, "localization.py"), synced, "--out", results,
               "--crops-dir", crops, "--submit", submit,
               "--max-offnadir", args.max_offnadir]
    if args.top_k and args.top_k > 0:
        loc_cmd += ["--top-k", args.top_k]
    if args.alt is not None:
        loc_cmd += ["--alt", args.alt]

    if os.path.isfile(args.camera) and os.path.getsize(args.camera) > 0:
        loc_cmd += ["--camera", args.camera]
    else:
        logging.warning("camera yaml missing/empty — localization uses PLACEHOLDER intrinsics")
    run(loc_cmd, label="4/4 localize")

if __name__ == "__main__":
    main()
