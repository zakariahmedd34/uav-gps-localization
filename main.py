"""
UAV flag localization: GoPro Hero 13 video -> GPS coordinates of the flags on the ground.

    python main.py --video GX014771.MP4 [--start 02:00] [--end 05:30] [--gpu]

Steps: telemetry (GPS + gravity from the video) -> YOLO detection -> sync by time
-> localization. Outputs go to artifacts/, the run log to logs/.
"""

import os
import sys
import shutil
import logging
import argparse
import datetime

from pipeline import telemetry, detect, localize

ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(ROOT, "artifacts")


def parse_time(s):
    """'HH:MM:SS', 'MM:SS' or plain seconds -> seconds."""
    sec = 0.0
    for part in s.strip().split(":"):
        sec = sec * 60 + float(part)
    return sec


def main():
    ap = argparse.ArgumentParser(description="Locate flags in GoPro Hero 13 (Linear lens) UAV video.")
    ap.add_argument("--video", required=True, help="GoPro .MP4 file")
    ap.add_argument("--start", type=parse_time,
                    help="start of the section to process: MM:SS, HH:MM:SS or seconds (default: video start)")
    ap.add_argument("--end", type=parse_time, help="end of the section to process (default: video end)")
    ap.add_argument("--gpu", action="store_true", help="run YOLO on the NVIDIA GPU (default: CPU)")

    tuning = ap.add_argument_group("optional tuning (defaults are the tested mission settings)")
    tuning.add_argument("--sample-fps", type=float, default=detect.SAMPLE_FPS,
                        help="frames analysed per second of video (default: %(default)s). More = more "
                             "detections per flag but slower; raise --min-frames with it (e.g. 3 fps -> 5)")
    tuning.add_argument("--min-frames", type=int, default=localize.MIN_FRAMES,
                        help="detections a flag needs to be submitted (default: %(default)s)")
    tuning.add_argument("--top-k", type=int, default=localize.TOP_K,
                        help="max flags submitted, 0 = no limit (default: %(default)s; "
                             "all candidates are always in candidates.csv)")
    tuning.add_argument("--imgsz", type=int, default=detect.IMGSZ,
                        help="YOLO input size in pixels (default: %(default)s, the training size). "
                             "Lower is faster but misses small flags")
    tuning.add_argument("--alt", type=float,
                        help="fixed camera height above the ground in metres. Only use it if the log "
                             "warns that the AGL anchor failed")
    args = ap.parse_args()
    if not os.path.isfile(args.video):
        ap.error(f"video not found: {args.video}")
    if args.start is not None and args.end is not None and args.end <= args.start:
        ap.error("--end must be after --start")
    if args.sample_fps <= 0 or args.imgsz <= 0 or args.min_frames < 1 or args.top_k < 0:
        ap.error("--sample-fps and --imgsz must be > 0, --min-frames >= 1, --top-k >= 0")

    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    log_file = os.path.join(ROOT, "logs", datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log")
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d - %(message)s",
                        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()])

    crops = os.path.join(ARTIFACTS, "crops")
    submission = os.path.join(ARTIFACTS, "submission")
    for path in (crops, submission):     # stale crops must never leak into a new submission
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(ARTIFACTS, exist_ok=True)

    logging.info("Video %s | section %s -> %s | %s | sample-fps=%s min-frames=%d top-k=%d imgsz=%d",
                 os.path.abspath(args.video),
                 args.start if args.start is not None else "start",
                 args.end if args.end is not None else "end", "GPU" if args.gpu else "CPU",
                 args.sample_fps, args.min_frames, args.top_k, args.imgsz)
    try:
        logging.info("[1/4] Reading telemetry")
        tel = telemetry.extract(args.video)       # whole video: the AGL anchor needs the ground segments
        tel.to_csv(os.path.join(ARTIFACTS, "telemetry.csv"), index=False)

        logging.info("[2/4] Detecting flags")
        det = detect.detect(args.video, crops, args.start, args.end, args.gpu, args.sample_fps, args.imgsz)
        det.to_csv(os.path.join(ARTIFACTS, "detections.csv"), index=False)

        logging.info("[3/4] Syncing detections with telemetry")
        synced = telemetry.sync(det, tel)
        synced.to_csv(os.path.join(ARTIFACTS, "synced.csv"), index=False)

        logging.info("[4/4] Localizing flags")
        flags = localize.localize(synced, crops, submission, args.alt, args.min_frames, args.top_k)
    except Exception:
        logging.exception("Pipeline failed (log: %s)", log_file)
        sys.exit(1)

    logging.info("Done: %d flag(s) in %s (review candidates.csv before submitting). Log: %s",
                 len(flags), os.path.join(submission, "targets.csv"), log_file)


if __name__ == "__main__":
    main()
