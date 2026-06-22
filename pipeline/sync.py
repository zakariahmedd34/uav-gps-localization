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
                    help="Print bracketing telemetry for this detection row (test)")
    args = ap.parse_args()

    det = pd.read_csv(args.detections)
    tel = pd.read_csv(args.telemetry).sort_values("time_s").reset_index(drop=True)

    t = tel["time_s"].to_numpy()
    if not np.all(np.diff(t) >= 0):
        sys.exit("ERROR: telemetry time_s is not sorted/monotonic.")

    # --- sanity: do the two clocks overlap? ---
    print(f"telemetry time_s: {t.min():.2f} -> {t.max():.2f}  ({len(t)} samples)")
    print(f"detection frame_time: {det.frame_time.min():.2f} -> "
          f"{det.frame_time.max():.2f}  ({len(det)} detections)")

    ft = det["frame_time"].to_numpy()
    out_of_range = (ft < t.min()) | (ft > t.max())
    if out_of_range.any():
        print(f"WARNING: dropping {int(out_of_range.sum())} detections outside "
              f"the telemetry time range.")
    det = det[~out_of_range].reset_index(drop=True)
    ft = det["frame_time"].to_numpy()

    # --- interpolate drone state to each detection time ---
    lat_uav = np.interp(ft, t, tel["lat"])
    lon_uav = np.interp(ft, t, tel["lon"])
    alt = np.interp(ft, t, tel["alt"])
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
        print(f"NOTE: GPS overridden with fixed point lat={vals[0]}, lon={vals[1]}, "
              f"alt={vals[2]}  (testing the chain; GRAV/tilt are still real).")

    # --- heading from GPS course-over-ground ---
    if args.heading is not None:
        heading = np.full(len(ft), args.heading)
        print(f"NOTE: heading fixed at {args.heading} deg (COG ignored).")
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
    print(f"gravity 'down' axis detected: {'xyz'[down_axis]}  "
          f"(tilt {tilt_deg.min():.1f}-{tilt_deg.max():.1f} deg, "
          f"median {np.median(tilt_deg):.1f})")
    print(f"heading {heading.min():.0f}-{heading.max():.0f} deg")

    synced = pd.DataFrame({
        "frame_time": det["frame_time"], "u": det["u"], "v": det["v"],
        "conf": det["conf"], "class": det["class"],
        "lat_uav": lat_uav, "lon_uav": lon_uav, "alt": alt,
        "grav_x": gx, "grav_y": gy, "grav_z": gz,
        "heading_deg": heading, "tilt_deg": tilt_deg,
    })

    # --- one-row bracket test ---
    if args.debug_row is not None:
        i = args.debug_row
        ti = synced["frame_time"].iloc[i]
        before = tel[tel["time_s"] <= ti].tail(1)
        after = tel[tel["time_s"] >= ti].head(1)
        print("\n--- DEBUG ROW", i, "(interp must fall BETWEEN before/after) ---")
        print(f"frame_time = {ti:.3f}")
        print("before:", before[["time_s", "lat", "lon"]].to_dict("records"))
        print("interp:  lat={:.7f} lon={:.7f}".format(
            synced['lat_uav'].iloc[i], synced['lon_uav'].iloc[i]))
        print("after: ", after[["time_s", "lat", "lon"]].to_dict("records"))

    synced.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(synced)} rows)")


if __name__ == "__main__":
    main()
