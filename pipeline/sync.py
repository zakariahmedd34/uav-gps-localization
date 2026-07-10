"""
sync.py — Join GPMF telemetry to YOLO detections by timestamp.

Both files share the same clock (t=0 = start of the GoPro video):
    detections.csv : frame_idx, frame_time, u, v, conf, class
    telemetry.csv  : time_s, lat, lon, alt, grav_x, grav_y, grav_z

For each detection at frame_time t, we interpolate the drone state to t and add
heading (GPS course-over-ground) and tilt (from the gravity vector).

Output: synced.csv
    frame_time, u, v, conf, class,
    lat_uav, lon_uav, alt, grav_x, grav_y, grav_z, heading_deg, tilt_deg

Usage:
    python sync.py detections.csv telemetry.csv synced.csv
    python sync.py detections.csv telemetry.csv synced.csv --debug-row 0
"""

import sys
import math
import argparse
import numpy as np
import pandas as pd

from logger import logging
from exception import CustomException

EARTH_M_PER_DEG = 111320.0


def compute_heading_series(t, lat, lon, min_dt=0.7):
    """Heading (deg, 0=N, 90=E) at each telemetry sample from course-over-ground.

    Uses a forward point at least `min_dt` seconds later so noisy adjacent GPS
    fixes don't dominate. Returns nan where no forward point exists.
    """
    n = len(t)
    heading = np.full(n, np.nan)
    for i in range(n):
        j = i
        while j < n - 1 and (t[j] - t[i]) < min_dt:
            j += 1
        if j == i:
            continue
        dN = (lat[j] - lat[i]) * EARTH_M_PER_DEG
        dE = (lon[j] - lon[i]) * EARTH_M_PER_DEG * math.cos(math.radians(lat[i]))
        if dN == 0 and dE == 0:
            continue
        heading[i] = math.degrees(math.atan2(dE, dN)) % 360.0
    # fill leading/trailing gaps so interpolation has values everywhere
    s = pd.Series(heading).ffill().bfill()
    return s.to_numpy()


def interp_circular(x, xp, deg):
    """Interpolate an angular (degrees) series safely across the 0/360 wrap."""
    rad = np.radians(deg)
    c = np.interp(x, xp, np.cos(rad))
    s = np.interp(x, xp, np.sin(rad))
    return (np.degrees(np.arctan2(s, c))) % 360.0


def main():
    ap = argparse.ArgumentParser(description="Join telemetry to detections by time")
    ap.add_argument("detections")
    ap.add_argument("telemetry")
    ap.add_argument("out", nargs="?", default="synced.csv")
    ap.add_argument("--min-dt", type=float, default=0.7,
                    help="GPS spacing (s) for course-over-ground heading")
    ap.add_argument("--heading", type=float, default=None,
                    help="Fixed heading (deg, 0=N) — keeps REAL GPS but overrides "
                         "course-over-ground. Use for stationary/handheld clips where "
                         "the drone doesn't move and COG is just GPS noise.")
    ap.add_argument("--fake-gps", default=None,
                    help="'lat,lon,alt[,heading]' — override GPS with a fixed point "
                         "when the recording has no satellite fix. Lets you test the "
                         "localize/aggregate chain. GRAV/tilt stay real.")
    ap.add_argument("--debug-row", type=int, default=None,
                    help="Log bracketing telemetry for this detection row (test)")
    args = ap.parse_args()
    logging.info("sync: START detections=%s telemetry=%s", args.detections, args.telemetry)

    det = pd.read_csv(args.detections)
    tel = pd.read_csv(args.telemetry).sort_values("time_s").reset_index(drop=True)

    t = tel["time_s"].to_numpy()
    if not np.all(np.diff(t) >= 0):
        logging.error("telemetry time_s is not sorted/monotonic")
        sys.exit("ERROR: telemetry time_s is not sorted/monotonic.")

    # --- sanity: do the two clocks overlap? ---
    logging.info("telemetry time_s: %.2f -> %.2f (%d samples)", t.min(), t.max(), len(t))
    logging.info("detection frame_time: %.2f -> %.2f (%d detections)",
                 det.frame_time.min(), det.frame_time.max(), len(det))

    ft = det["frame_time"].to_numpy()
    out_of_range = (ft < t.min()) | (ft > t.max())
    if out_of_range.any():
        logging.warning("dropping %d detections outside the telemetry time range",
                        int(out_of_range.sum()))
    det = det[~out_of_range].reset_index(drop=True)
    ft = det["frame_time"].to_numpy()

    # --- interpolate drone state to each detection time ---
    lat_uav = np.interp(ft, t, tel["lat"])
    lon_uav = np.interp(ft, t, tel["lon"])
    alt = np.interp(ft, t, tel["alt"])
    # GPS-UTC passthrough (if gpmf_extract wrote it) — lets a Pixhawk .bin be
    # joined later by UTC instead of a wing-rock time sync.
    utc_s = (np.interp(ft, t, tel["utc_s"])
             if "utc_s" in tel.columns and tel["utc_s"].notna().any() else None)
    # AGL = alt above the ground (first telemetry sample). Prefer the column
    # gpmf_extract wrote; else derive it here from alt minus the first alt.
    agl_src = tel["agl"] if "agl" in tel.columns else (tel["alt"] - tel["alt"].iloc[0])
    agl = np.interp(ft, t, agl_src)
    gx = np.interp(ft, t, tel["grav_x"])
    gy = np.interp(ft, t, tel["grav_y"])
    gz = np.interp(ft, t, tel["grav_z"])

    # --- optional: override GPS with a fixed point (no-fix test recordings) ---
    fake_heading = None
    if args.fake_gps:
        vals = [float(x) for x in args.fake_gps.split(",")]
        lat_uav[:] = vals[0]
        lon_uav[:] = vals[1]
        alt[:] = vals[2]
        if len(vals) > 3:
            fake_heading = vals[3]
        logging.info("GPS overridden with fixed point lat=%s lon=%s alt=%s "
                     "(testing the chain; GRAV/tilt still real)", vals[0], vals[1], vals[2])

    # --- heading from GPS course-over-ground ---
    if args.heading is not None:
        heading = np.full(len(ft), args.heading)
        logging.info("heading fixed at %s deg (COG ignored)", args.heading)
    elif fake_heading is not None:
        heading = np.full(len(ft), fake_heading)
    elif args.fake_gps:
        heading = np.zeros(len(ft))  # no movement -> heading undefined; use 0 (North)
    else:
        heading_tel = compute_heading_series(t, tel["lat"].to_numpy(),
                                             tel["lon"].to_numpy(), args.min_dt)
        heading = interp_circular(ft, t, heading_tel)

    # --- tilt from gravity vector (deviation from vertical) ---
    # Auto-detect which axis is "down" = component with the largest mean magnitude.
    means = np.abs([tel["grav_x"].mean(), tel["grav_y"].mean(), tel["grav_z"].mean()])
    down_axis = int(np.argmax(means))
    g = np.vstack([gx, gy, gz]).T
    g_down = np.abs(g[:, down_axis])
    g_norm = np.linalg.norm(g, axis=1)
    tilt_deg = np.degrees(np.arccos(np.clip(g_down / np.where(g_norm == 0, 1, g_norm),
                                            -1, 1)))
    logging.info("gravity 'down' axis = %s (tilt %.1f-%.1f deg, median %.1f)",
                 "xyz"[down_axis], tilt_deg.min(), tilt_deg.max(), np.median(tilt_deg))
    logging.info("heading %.0f-%.0f deg", heading.min(), heading.max())

    synced = pd.DataFrame({
        "frame_time": det["frame_time"], "u": det["u"], "v": det["v"],
        "conf": det["conf"], "class": det["class"],
        "crop": det["crop"] if "crop" in det.columns else "",
        "lat_uav": lat_uav, "lon_uav": lon_uav, "alt": alt, "agl": agl,
        "grav_x": gx, "grav_y": gy, "grav_z": gz,
        "heading_deg": heading, "tilt_deg": tilt_deg,
    })
    if utc_s is not None:
        synced["utc_s"] = utc_s

    # --- one-row bracket test ---
    if args.debug_row is not None:
        i = args.debug_row
        ti = synced["frame_time"].iloc[i]
        before = tel[tel["time_s"] <= ti].tail(1)
        after = tel[tel["time_s"] >= ti].head(1)
        logging.info("--- DEBUG ROW %d (interp must fall BETWEEN before/after) ---", i)
        logging.info("frame_time = %.3f", ti)
        logging.info("before: %s", before[["time_s", "lat", "lon"]].to_dict("records"))
        logging.info("interp: lat=%.7f lon=%.7f",
                     synced["lat_uav"].iloc[i], synced["lon_uav"].iloc[i])
        logging.info("after:  %s", after[["time_s", "lat", "lon"]].to_dict("records"))

    synced.to_csv(args.out, index=False)
    logging.info("sync: DONE wrote %s (%d rows)", args.out, len(synced))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise CustomException(e, sys) from e
