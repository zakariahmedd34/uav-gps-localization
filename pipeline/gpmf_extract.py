"""
video -> telemetry.csv
  (time_s, utc_s, lat, lon, alt, agl, speed_2d, fix, dop, grav_x, grav_y, grav_z)

Uses `telemetrik` to parse GPMF directly from the MP4's moov/trak boxes.
No ffmpeg gpmd-track extraction step is needed: telemetrik reads the
GoPro metadata track straight out of the original .MP4.

    pip install telemetrik --break-system-packages

V2 fixes (post first-mission debrief):
  * GPS QUALITY GATE — drop samples with fix < 3 or dop > threshold. The first
    mission video carried a stale pre-lock fix 21 km away for ~100 s; it must
    never reach sync/localization.
  * TELEPORT REJECTION — drop single-sample position spikes (> max_speed vs
    both neighbours) that survive the fix gate.
  * AGL ANCHOR — ground altitude = median alt over a STATIONARY window at the
    start or end of the (gated) telemetry, NOT the first raw sample.
  * GPS-UTC KEPT — utc_s (days_since_2000*86400 + secs_since_midnight) is
    written per sample so a Pixhawk .bin can later be joined by UTC with no
    wing-rock sync.
"""

import sys
import math
import subprocess
import json
import numpy as np
import pandas as pd
from telemetrik import extract_all_telemetry
from telemetrik.parser import get_boxes, get_samples, _from_bytes
from os.path import getsize

from gps9_parser import get_gps9_stream

from logger import logging
from exception import CustomException


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

    logging.info("[ffprobe] fps=%.3f resolution=%s gpmd_track=%s", fps, resolution,
                 f"FOUND (stream {gpmd_stream['index']})" if gpmd_stream else "NOT FOUND")

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


EARTH_M_PER_DEG = 111320.0


# ----------------------------------------------------------------------
# V2 helpers: GPS quality gate, teleport rejection, AGL anchor
# ----------------------------------------------------------------------
def gate_gps_quality(gps_df, min_fix=3, max_dop=10.0):
    """Keep only samples with a 3D fix and acceptable DOP.

    GPS5 files have no per-sample fix/dop -> gate is skipped with a warning.
    """
    if "fix" not in gps_df.columns or gps_df["fix"].isna().all():
        logging.warning("[GPS gate] no fix/dop fields (GPS5 file?) — gate SKIPPED. "
                        "Verify the track manually before trusting positions.")
        return gps_df
    good = (gps_df["fix"] >= min_fix) & (gps_df["dop"] <= max_dop)
    n0, n1 = len(gps_df), int(good.sum())
    logging.info("[GPS gate] fix>=%d & dop<=%.1f keeps %d/%d samples (dropped %d)",
                 min_fix, max_dop, n1, n0, n0 - n1)
    if n1 == 0:
        raise ValueError(
            f"All {n0} GPS samples fail the quality gate (fix>= {min_fix}, "
            f"dop<={max_dop}). GPS never locked — this video cannot be localized."
        )
    if n1 < 0.5 * n0:
        logging.warning("[GPS gate] more than half the GPS samples were rejected — "
                        "check that the camera had a lock BEFORE recording started.")
    return gps_df[good].reset_index(drop=True)


def drop_teleports(gps_df, max_speed=80.0):
    """Drop single-sample position spikes: a sample is rejected when the implied
    speed to BOTH its neighbours exceeds max_speed (m/s)."""
    if len(gps_df) < 3:
        return gps_df
    lat = gps_df["lat"].to_numpy()
    lon = gps_df["lon"].to_numpy()
    t = gps_df["time_s"].to_numpy()

    def speed(i, j):
        dt = max(abs(t[j] - t[i]), 1e-3)
        dn = (lat[j] - lat[i]) * EARTH_M_PER_DEG
        de = (lon[j] - lon[i]) * EARTH_M_PER_DEG * math.cos(math.radians(lat[i]))
        return math.hypot(dn, de) / dt

    keep = np.ones(len(gps_df), dtype=bool)
    for i in range(1, len(gps_df) - 1):
        if speed(i - 1, i) > max_speed and speed(i, i + 1) > max_speed:
            keep[i] = False
    dropped = int((~keep).sum())
    if dropped:
        logging.info("[teleport] dropped %d spike sample(s) (> %.0f m/s vs both neighbours)",
                     dropped, max_speed)
    return gps_df[keep].reset_index(drop=True)


def find_ground_alt(gps_df, stationary_speed=1.5, min_window_s=10.0, search_s=120.0):
    """Ground altitude = median alt over a stationary window at the START or END
    of the quality-gated telemetry (GoPro on the ground before takeoff / after
    landing). Returns (ground_alt, source) or (None, reason).

    Differencing altitudes from the SAME receiver cancels the geoid/datum
    question, so agl = alt - ground_alt is datum-safe.
    """
    t = gps_df["time_s"].to_numpy()
    alt = gps_df["alt"].to_numpy()

    if "speed_2d" in gps_df.columns and gps_df["speed_2d"].notna().any():
        spd = gps_df["speed_2d"].to_numpy()
    else:  # derive from positions
        dn = np.diff(gps_df["lat"].to_numpy()) * EARTH_M_PER_DEG
        de = (np.diff(gps_df["lon"].to_numpy()) * EARTH_M_PER_DEG
              * np.cos(np.radians(gps_df["lat"].to_numpy()[:-1])))
        dt = np.maximum(np.diff(t), 1e-3)
        spd = np.concatenate([[0.0], np.hypot(dn, de) / dt])

    for label, region in (("start", t <= t[0] + search_s),
                          ("end", t >= t[-1] - search_s)):
        idx = np.where(region & (spd < stationary_speed))[0]
        if len(idx) < 2:
            continue
        # longest contiguous run (gap <= 1 s between consecutive kept samples)
        runs, cur = [], [idx[0]]
        for a, b in zip(idx[:-1], idx[1:]):
            if t[b] - t[a] <= 1.0:
                cur.append(b)
            else:
                runs.append(cur)
                cur = [b]
        runs.append(cur)
        best = max(runs, key=lambda r: t[r[-1]] - t[r[0]])
        dur = t[best[-1]] - t[best[0]]
        if dur >= min_window_s:
            ground = float(np.median(alt[best]))
            logging.info("[AGL anchor] stationary window at %s: t=%.1f-%.1fs "
                         "(%.1fs, %d samples) -> ground_alt=%.1f m",
                         label, t[best[0]], t[best[-1]], dur, len(best), ground)
            return ground, label
    return None, (f"no stationary window >= {min_window_s}s (speed < "
                  f"{stationary_speed} m/s) in the first/last {search_s}s")


# ----------------------------------------------------------------------
# Step 1: extract + align GPS5/GPS9 and GRAV into one dataframe
# ----------------------------------------------------------------------
def extract_video_to_csv(video_path, output_csv="telemetry.csv", check=True,
                         min_fix=3, max_dop=10.0, max_speed=80.0):
    logging.info("gpmf_extract: START video=%s -> %s", video_path, output_csv)
    if check:
        # ffprobe is only an optional sanity print; telemetrik reads the GPMF
        # straight from the MP4. Don't let a missing ffprobe kill the run.
        try:
            ffprobe_check(video_path)
        except FileNotFoundError:
            logging.warning("[ffprobe] not found on PATH — skipping sanity check "
                            "(install ffmpeg for it). Continuing with telemetrik.")

    logging.info("[1] Parsing GPMF via telemetrik...")

    # Try GPS5 first (older cameras). Hero 11/12/13 don't record GPS5 at
    # all -- they switched to GPS9 -- so fall back to our own GPS9 decoder
    # if GPS5 isn't present.
    streams = extract_all_telemetry(video_path, streams=["GPS5", "GRAV"])

    if "GRAV" not in streams:
        raise ValueError("No GRAV stream found in this file's GPMF data.")
    grav_stream = streams["GRAV"]

    if "GPS5" in streams:
        logging.info("[1a] Using GPS5 stream (no per-sample fix/dop/UTC available).")
        gps_stream = streams["GPS5"]
        gps_rows = [
            {"time_s": t, "lat": v[0], "lon": v[1], "alt": v[2],
             "speed_2d": None, "fix": None, "dop": None, "utc_s": None}
            for t, v in gps_stream.pts_data
        ]
    else:
        logging.info("[1a] No GPS5 found -- falling back to GPS9 (Hero 11/12/13).")
        with open(video_path, "rb") as f:
            samples, time_base = _get_gpmd_samples_and_timebase(f, getsize(video_path))
            gps9_data = get_gps9_stream(f, samples, time_base=time_base)

        if not gps9_data:
            raise ValueError(
                "No GPS5 or GPS9 stream found -- was GPS enabled/locked during recording?"
            )

        # keep fix/dop (quality gate) + speed_2d (AGL anchor) + GPS-UTC
        # (Pixhawk .bin join later: both devices timestamp in GPS time).
        gps_rows = [
            {"time_s": t, "lat": v["lat"], "lon": v["lon"], "alt": v["alt"],
             "speed_2d": v.get("speed_2d"),
             "fix": v.get("fix"), "dop": v.get("dop"),
             "utc_s": (v["days_since_2000"] * 86400.0 + v["secs_since_midnight"])
                      if v.get("days_since_2000") is not None else None}
            for t, v in gps9_data
        ]

    grav_rows = [
        {"time_s": t, "grav_x": v[0], "grav_y": v[1], "grav_z": v[2]}
        for t, v in grav_stream.pts_data
    ]

    gps_df = pd.DataFrame(gps_rows).sort_values("time_s").reset_index(drop=True)
    grav_df = pd.DataFrame(grav_rows).sort_values("time_s")

    logging.info("[2] GPS samples: %d | GRAV samples: %d", len(gps_df), len(grav_df))

    # --- V2: quality gate + teleport rejection BEFORE anything uses positions ---
    gps_df = gate_gps_quality(gps_df, min_fix=min_fix, max_dop=max_dop)
    gps_df = drop_teleports(gps_df, max_speed=max_speed)

    # GRAV runs at a much higher rate than GPS (~200 Hz vs ~10-18 Hz), so we
    # snap each GPS row to its nearest GRAV reading in time.
    df = pd.merge_asof(
        gps_df, grav_df, on="time_s", direction="nearest", tolerance=0.5
    )

    # --- V2: AGL anchor = median alt of a stationary locked window (start or
    # end of the video), NOT the first raw sample. Same-receiver differencing
    # cancels the geoid/datum question.
    ground_alt, src = find_ground_alt(df)
    if ground_alt is None:
        logging.warning("[AGL anchor] %s — FALLING BACK to first gated sample "
                        "(alt=%.1f). AGL may be biased: prefer --alt override or a "
                        "Pixhawk RelHomeAlt join in localization.", src, df["alt"].iloc[0])
        ground_alt = float(df["alt"].iloc[0])
    df["agl"] = df["alt"] - ground_alt

    # plausibility: a fixed-wing mission flies 30-130 m AGL; warn loudly outside it
    agl_air = df.loc[df["agl"].abs() > 5, "agl"]
    if len(agl_air) and (agl_air.median() < 30 or agl_air.median() > 130):
        logging.warning("[AGL anchor] median airborne AGL = %.1f m is OUTSIDE the "
                        "plausible 30-130 m mission band — anchor or GPS alt is "
                        "suspect. Do NOT trust localization without --alt override.",
                        agl_air.median())

    df = df[["time_s", "utc_s", "lat", "lon", "alt", "agl", "speed_2d",
             "fix", "dop", "grav_x", "grav_y", "grav_z"]]

    df.to_csv(output_csv, index=False)
    logging.info("[DONE] Saved: %s", output_csv)
    logging.info("First rows:\n%s", df.head())
    logging.info("lat range: [%s, %s]", df["lat"].min(), df["lat"].max())
    logging.info("lon range: [%s, %s]", df["lon"].min(), df["lon"].max())
    logging.info("alt (GoPro GPS, NOT AGL) range: [%s, %s]", df["alt"].min(), df["alt"].max())
    logging.info("agl range: [%.1f, %.1f] m (ground_alt=%.1f)",
                 df["agl"].min(), df["agl"].max(), ground_alt)

    return df


if __name__ == "__main__":
    try:
        import argparse
        ap = argparse.ArgumentParser(description="GoPro .MP4 -> telemetry.csv")
        ap.add_argument("video")
        ap.add_argument("--out", default="telemetry.csv")
        ap.add_argument("--min-fix", type=int, default=3,
                        help="Minimum GPS fix type (3 = 3D lock)")
        ap.add_argument("--max-dop", type=float, default=10.0,
                        help="Maximum dilution of precision")
        ap.add_argument("--max-speed", type=float, default=80.0,
                        help="Teleport rejection threshold (m/s)")
        a = ap.parse_args()
        extract_video_to_csv(a.video, a.out, min_fix=a.min_fix,
                             max_dop=a.max_dop, max_speed=a.max_speed)
    except Exception as e:
        raise CustomException(e, sys) from e
