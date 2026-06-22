"""
video -> telemetry.csv  (time_s, lat, lon, alt, grav_x, grav_y, grav_z)

Uses `telemetrik` to parse GPMF directly from the MP4's moov/trak boxes.
No ffmpeg gpmd-track extraction step is needed: telemetrik reads the
GoPro metadata track straight out of the original .MP4.

    pip install telemetrik --break-system-packages
"""

import subprocess
import json
import pandas as pd
from telemetrik import extract_all_telemetry
from telemetrik.parser import get_boxes, get_samples, _from_bytes
from os.path import getsize

from gps9_parser import get_gps9_stream


# ----------------------------------------------------------------------
# Step 0: ffprobe sanity check (fps, resolution, confirm gpmd track exists)
# ----------------------------------------------------------------------
def ffprobe_check(video_path):
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", video_path,
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    info = json.loads(out)

    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    gpmd_stream = next(
        (s for s in info["streams"] if s.get("codec_tag_string") == "gpmd"), None
    )

    fps = eval(video_stream["r_frame_rate"])  # e.g. "29970/1000" -> 29.97
    resolution = (video_stream["width"], video_stream["height"])

    print(f"[ffprobe] fps={fps:.3f} resolution={resolution} "
          f"gpmd_track={'FOUND (stream ' + str(gpmd_stream['index']) + ')' if gpmd_stream else 'NOT FOUND'}")

    if gpmd_stream is None:
        raise ValueError(
            f"No GPMF metadata ('gpmd') track found in {video_path}. "
            "Make sure GPS was enabled when recording."
        )

    return {"fps": fps, "resolution": resolution, "gpmd_stream_index": gpmd_stream["index"]}


# ----------------------------------------------------------------------
# Helper: locate the gpmd track's sample table + timebase (same logic
# telemetrik uses internally, exposed here so we can run our own GPS9
# decoder against it when GPS5 isn't present).
# ----------------------------------------------------------------------
def _get_gpmd_samples_and_timebase(f, size):
    stbl = None
    gpmf_mdia = None

    minf_boxes = get_boxes(f, 0, size, ["moov", "trak", "mdia", "minf"])
    for box in minf_boxes:
        if get_boxes(f, box.offset, box.size, ["minf", "gmhd", "gpmd"]):
            stbl = get_boxes(f, box.offset, box.size, ["minf", "stbl"])[0]
            mdia_boxes = get_boxes(f, 0, size, ["moov", "trak", "mdia"])
            for mdia in mdia_boxes:
                if box.offset >= mdia.offset and box.offset < mdia.offset + mdia.size:
                    gpmf_mdia = mdia
                    break
            break
    if stbl is None:
        raise ValueError("No GPMF ('gpmd') track found in this video.")

    time_base = None
    if gpmf_mdia:
        mdhd_boxes = get_boxes(f, gpmf_mdia.offset, gpmf_mdia.size, ["mdia", "mdhd"])
        if mdhd_boxes:
            mdhd = mdhd_boxes[0]
            f.seek(mdhd.offset + 12)
            f.read(8)
            timescale = _from_bytes(f.read(4))
            time_base = (1, timescale)

    samples = get_samples(f, stbl)
    return samples, time_base


# ----------------------------------------------------------------------
# Step 1: extract + align GPS5/GPS9 and GRAV into one dataframe
# ----------------------------------------------------------------------
def extract_video_to_csv(video_path, output_csv="telemetry.csv", check=True):
    if check:
        # ffprobe is only an optional sanity print; telemetrik reads the GPMF
        # straight from the MP4. Don't let a missing ffprobe kill the run.
        try:
            ffprobe_check(video_path)
        except FileNotFoundError:
            print("[ffprobe] not found on PATH — skipping sanity check "
                  "(install ffmpeg for it). Continuing with telemetrik.")

    print("[1] Parsing GPMF via telemetrik...")

    # Try GPS5 first (older cameras). Hero 11/12/13 don't record GPS5 at
    # all -- they switched to GPS9 -- so fall back to our own GPS9 decoder
    # if GPS5 isn't present.
    streams = extract_all_telemetry(video_path, streams=["GPS5", "GRAV"])

    if "GRAV" not in streams:
        raise ValueError("No GRAV stream found in this file's GPMF data.")
    grav_stream = streams["GRAV"]

    if "GPS5" in streams:
        print("[1a] Using GPS5 stream.")
        gps_stream = streams["GPS5"]
        gps_rows = [
            {"time_s": t, "lat": v[0], "lon": v[1], "alt": v[2]}
            for t, v in gps_stream.pts_data
        ]
    else:
        print("[1a] No GPS5 found -- falling back to GPS9 (Hero 11/12/13).")
        with open(video_path, "rb") as f:
            samples, time_base = _get_gpmd_samples_and_timebase(f, getsize(video_path))
            gps9_data = get_gps9_stream(f, samples, time_base=time_base)

        if not gps9_data:
            raise ValueError(
                "No GPS5 or GPS9 stream found -- was GPS enabled/locked during recording?"
            )

        gps_rows = [
            {"time_s": t, "lat": v["lat"], "lon": v["lon"], "alt": v["alt"]}
            for t, v in gps9_data
        ]

    grav_rows = [
        {"time_s": t, "grav_x": v[0], "grav_y": v[1], "grav_z": v[2]}
        for t, v in grav_stream.pts_data
    ]

    gps_df = pd.DataFrame(gps_rows).sort_values("time_s")
    grav_df = pd.DataFrame(grav_rows).sort_values("time_s")

    print(f"[2] GPS samples: {len(gps_df)} | GRAV samples: {len(grav_df)}")

    # GRAV runs at a much higher rate than GPS (~200 Hz vs ~10-18 Hz), so we
    # snap each GPS row to its nearest GRAV reading in time.
    df = pd.merge_asof(
        gps_df, grav_df, on="time_s", direction="nearest", tolerance=0.5
    )

    df = df[["time_s", "lat", "lon", "alt", "grav_x", "grav_y", "grav_z"]]

    df.to_csv(output_csv, index=False)
    print("[DONE] Saved:", output_csv)

    print("\n--- First rows ---")
    print(df.head())
    print("\n--- Lat/Lon range ---")
    print(f"lat: [{df['lat'].min()}, {df['lat'].max()}]")
    print(f"lon: [{df['lon'].min()}, {df['lon'].max()}]")
    print(f"alt (GoPro GPS, NOT AGL): [{df['alt'].min()}, {df['alt'].max()}]")

    return df


if __name__ == "__main__":
    import sys
    extract_video_to_csv(sys.argv[1])