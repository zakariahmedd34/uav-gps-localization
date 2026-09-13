"""
telemetry.py - GPS + gravity telemetry from the GoPro Hero 13 video itself.

extract(video)       -> one row per GPS sample: position, AGL and gravity vector,
                        with bad GPS fixes and position spikes removed.
sync(detections, tel) -> drone state (position, AGL, gravity, heading) interpolated
                        to every detection time.

The GPMF telemetry and the video frames share one clock (t=0 = start of the MP4),
so no time-offset estimation is needed.
"""

import math
import struct
import logging
from os.path import getsize

import numpy as np
import pandas as pd
from telemetrik import extract_all_telemetry
from telemetrik.parser import get_boxes, get_samples, get_gpmf_boxes, _from_bytes

EARTH = 111320.0        # metres per degree of latitude
MIN_FIX = 3             # GPS fix type required (3 = 3D lock)
MAX_DOP = 10.0          # max GPS dilution of precision
MAX_SPEED = 80.0        # m/s; single-sample position jumps faster than this are dropped
HEADING_MIN_DT = 0.7    # s between GPS points used for course-over-ground heading


# ----------------------------------------------------------------------------
# GPS9 decoding. The Hero 13 records GPS9 (not the older GPS5), which telemetrik
# does not decode. GPS9 is a complex GPMF type: per-field byte widths come from a
# sibling TYPE key, and SCAL holds one divisor per field.
# Reference: https://github.com/gopro/gpmf-parser
# ----------------------------------------------------------------------------
GPS9_FIELDS = ["lat", "lon", "alt", "speed_2d", "speed_3d",
               "days_since_2000", "secs_since_midnight", "dop", "fix"]
N_FIELDS = len(GPS9_FIELDS)
TYPE_WIDTHS = {"b": 1, "B": 1, "c": 1, "s": 2, "S": 2,
               "l": 4, "L": 4, "f": 4, "F": 4, "q": 4,
               "d": 8, "j": 8, "J": 8, "Q": 8, "U": 16, "G": 16}


def _expand_type_string(s):
    """Expand bracket notation like 'f[8]L' -> 'ffffffffL'."""
    out, i = [], 0
    while i < len(s):
        if i + 1 < len(s) and s[i + 1] == "[":
            j = s.index("]", i + 1)
            out.append(s[i] * int(s[i + 2:j]))
            i = j + 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _read_typed_value(f, type_char):
    raw = f.read(TYPE_WIDTHS.get(type_char, 4))
    if type_char == "f":
        return struct.unpack(">f", raw)[0]
    if type_char == "d":
        return struct.unpack(">d", raw)[0]
    return int.from_bytes(raw, "big", signed=type_char.islower())


def _read_scale_array(f, scal_box):
    """SCAL is either one shared divisor or one divisor per field."""
    f.seek(scal_box.offset + 8)
    divisors = []
    for _ in range(scal_box.repeat):
        v = _from_bytes(f.read(scal_box.struct_size), signed=True)
        divisors.append(v if v != 0 else 1)
    if len(divisors) == 1:
        return divisors * N_FIELDS
    return (divisors + [1] * N_FIELDS)[:N_FIELDS]


def _gpmd_samples_and_timebase(f, size):
    """Locate the GPMF ('gpmd') track's sample table and its timescale."""
    stbl = gpmf_mdia = None
    for box in get_boxes(f, 0, size, ["moov", "trak", "mdia", "minf"]):
        if get_boxes(f, box.offset, box.size, ["minf", "gmhd", "gpmd"]):
            stbl = get_boxes(f, box.offset, box.size, ["minf", "stbl"])[0]
            for mdia in get_boxes(f, 0, size, ["moov", "trak", "mdia"]):
                if mdia.offset <= box.offset < mdia.offset + mdia.size:
                    gpmf_mdia = mdia
                    break
            break
    if stbl is None:
        raise ValueError("No GPMF ('gpmd') telemetry track in this video.")

    time_base = None
    if gpmf_mdia:
        mdhd_boxes = get_boxes(f, gpmf_mdia.offset, gpmf_mdia.size, ["mdia", "mdhd"])
        if mdhd_boxes:
            f.seek(mdhd_boxes[0].offset + 12)
            f.read(8)
            time_base = (1, _from_bytes(f.read(4)))
    return get_samples(f, stbl), time_base


def _read_gps9(f, samples, time_base):
    """Return [(pts_seconds, {field: value}), ...] on the video's PTS timeline."""
    payloads = []           # (stmp_ms, sample_pts, gps9_box, divisors)
    field_types = None      # TYPE is sticky: read it once
    for sample in samples:
        strm = None
        for box in get_gpmf_boxes(f, sample.offset, sample.size, ["DEVC", "STRM"]):
            if get_gpmf_boxes(f, box.offset, box.size, ["STRM", "GPS9"]):
                strm = box
                break
        if strm is None:
            continue
        stmp_boxes = get_gpmf_boxes(f, strm.offset, strm.size, ["STRM", "STMP"])
        if not stmp_boxes:
            continue
        stmp = stmp_boxes[0]
        f.seek(stmp.offset + 8)
        stmp_ms = _from_bytes(f.read(stmp.struct_size * stmp.repeat)) / 1000
        gps9_box = get_gpmf_boxes(f, strm.offset, strm.size, ["STRM", "GPS9"])[0]

        if field_types is None:
            field_types = ["l"] * N_FIELDS          # fallback if there is no TYPE key
            type_boxes = get_gpmf_boxes(f, strm.offset, strm.size, ["STRM", "TYPE"])
            if type_boxes:
                tbox = type_boxes[0]
                f.seek(tbox.offset + 8)
                raw = f.read(tbox.struct_size * tbox.repeat)
                expanded = _expand_type_string(raw.decode("ascii", errors="ignore").rstrip("\x00"))
                if len(expanded) == N_FIELDS:
                    field_types = list(expanded)
            logging.info("GPS9 field types: %s", "".join(field_types))

        scal_boxes = get_gpmf_boxes(f, strm.offset, strm.size, ["STRM", "SCAL"])
        divisors = _read_scale_array(f, scal_boxes[0]) if scal_boxes else [1] * N_FIELDS
        payloads.append((stmp_ms, sample.pts if sample.pts is not None else 0, gps9_box, divisors))

    out = []
    dur_ms = dur_pts = None     # per-reading duration; the last payload reuses the previous one
    for idx, (base_ms, base_pts, box, divisors) in enumerate(payloads):
        f.seek(box.offset + 8)
        if idx + 1 < len(payloads):
            dur_ms = (payloads[idx + 1][0] - base_ms) / box.repeat
            if time_base is not None:
                dur_pts = (payloads[idx + 1][1] - base_pts) / box.repeat
        for i in range(box.repeat):
            raw = [_read_typed_value(f, t) for t in field_types]
            record = dict(zip(GPS9_FIELDS, [r / d for r, d in zip(raw, divisors)]))
            if time_base is not None and dur_pts is not None:
                t = (base_pts + i * dur_pts) * time_base[0] / time_base[1]
            else:
                t = (base_ms + i * (dur_ms or 0)) / 1000
            out.append((t, record))
    return out


# ----------------------------------------------------------------------------
# GPS cleaning + AGL
# ----------------------------------------------------------------------------
def _gate_gps_quality(gps):
    """Keep only samples with a 3D fix and acceptable DOP. A stale pre-lock fix
    (seen 21 km away in the first mission video) must never reach localization."""
    good = (gps["fix"] >= MIN_FIX) & (gps["dop"] <= MAX_DOP)
    n0, n1 = len(gps), int(good.sum())
    logging.info("GPS gate: fix>=%d & dop<=%.1f keeps %d/%d samples", MIN_FIX, MAX_DOP, n1, n0)
    if n1 == 0:
        raise ValueError("No GPS sample has a 3D fix - GPS never locked, this video "
                         "cannot be localized.")
    if n1 < 0.5 * n0:
        logging.warning("More than half the GPS samples were rejected - make sure the "
                        "camera has a GPS lock BEFORE recording starts.")
    return gps[good].reset_index(drop=True)


def _drop_teleports(gps):
    """Drop single-sample position spikes: the implied speed to BOTH neighbours
    exceeds MAX_SPEED."""
    if len(gps) < 3:
        return gps
    lat, lon, t = gps["lat"].to_numpy(), gps["lon"].to_numpy(), gps["time_s"].to_numpy()

    def speed(i, j):
        dn = (lat[j] - lat[i]) * EARTH
        de = (lon[j] - lon[i]) * EARTH * math.cos(math.radians(lat[i]))
        return math.hypot(dn, de) / max(abs(t[j] - t[i]), 1e-3)

    keep = np.ones(len(gps), dtype=bool)
    for i in range(1, len(gps) - 1):
        if speed(i - 1, i) > MAX_SPEED and speed(i, i + 1) > MAX_SPEED:
            keep[i] = False
    if (~keep).any():
        logging.info("Dropped %d GPS spike sample(s)", int((~keep).sum()))
    return gps[keep].reset_index(drop=True)


def _find_ground_alt(df, stationary_speed=1.5, min_window_s=10.0, search_s=120.0):
    """Ground altitude = median GPS altitude over the longest stationary window
    (camera on the ground) in the first or last `search_s` seconds. Differencing
    altitudes from the same receiver cancels the geoid/datum question.
    Returns the altitude, or None if no such window exists."""
    t, alt, spd = df["time_s"].to_numpy(), df["alt"].to_numpy(), df["speed_2d"].to_numpy()
    for label, region in (("start", t <= t[0] + search_s), ("end", t >= t[-1] - search_s)):
        idx = np.where(region & (spd < stationary_speed))[0]
        if len(idx) < 2:
            continue
        runs, cur = [], [idx[0]]            # contiguous runs (gap <= 1 s)
        for a, b in zip(idx[:-1], idx[1:]):
            if t[b] - t[a] <= 1.0:
                cur.append(b)
            else:
                runs.append(cur)
                cur = [b]
        runs.append(cur)
        best = max(runs, key=lambda r: t[r[-1]] - t[r[0]])
        if t[best[-1]] - t[best[0]] >= min_window_s:
            ground = float(np.median(alt[best]))
            logging.info("AGL anchor: stationary window at %s, t=%.1f-%.1fs -> ground alt %.1f m",
                         label, t[best[0]], t[best[-1]], ground)
            return ground
    return None


def extract(video):
    """GoPro MP4 -> DataFrame(time_s, lat, lon, alt, agl, speed_2d, fix, dop, grav_x, grav_y, grav_z)."""
    streams = extract_all_telemetry(video, streams=["GRAV"])
    if "GRAV" not in streams:
        raise ValueError("No GRAV (gravity) stream in this video's telemetry.")
    with open(video, "rb") as f:
        samples, time_base = _gpmd_samples_and_timebase(f, getsize(video))
        gps9 = _read_gps9(f, samples, time_base)
    if not gps9:
        raise ValueError("No GPS9 stream in this video - was GPS enabled on the GoPro?")

    gps = pd.DataFrame([{"time_s": t, "lat": v["lat"], "lon": v["lon"], "alt": v["alt"],
                         "speed_2d": v["speed_2d"], "fix": v["fix"], "dop": v["dop"]}
                        for t, v in gps9]).sort_values("time_s").reset_index(drop=True)
    grav = pd.DataFrame([{"time_s": t, "grav_x": v[0], "grav_y": v[1], "grav_z": v[2]}
                         for t, v in streams["GRAV"].pts_data]).sort_values("time_s")
    logging.info("GPS samples: %d | GRAV samples: %d", len(gps), len(grav))

    gps = _drop_teleports(_gate_gps_quality(gps))
    # GRAV runs much faster than GPS: snap each GPS row to its nearest GRAV reading
    df = pd.merge_asof(gps, grav, on="time_s", direction="nearest", tolerance=0.5)

    ground_alt = _find_ground_alt(df)
    if ground_alt is None:
        ground_alt = float(df["alt"].iloc[0])
        logging.warning("AGL anchor: no stationary window >= 10 s at the start/end of the "
                        "video - using the first GPS sample (alt=%.1f m). AGL may be biased: "
                        "consider --alt.", ground_alt)
    df["agl"] = df["alt"] - ground_alt

    airborne = df.loc[df["agl"].abs() > 5, "agl"]
    if len(airborne) and not 30 <= airborne.median() <= 130:
        logging.warning("Median airborne AGL = %.1f m is outside the plausible 30-130 m "
                        "mission band - the AGL anchor or GPS altitude is suspect. Do not "
                        "trust the result without --alt.", airborne.median())
    logging.info("AGL range: %.1f to %.1f m", df["agl"].min(), df["agl"].max())
    return df[["time_s", "lat", "lon", "alt", "agl", "speed_2d", "fix", "dop",
               "grav_x", "grav_y", "grav_z"]]


# ----------------------------------------------------------------------------
# Sync detections to telemetry
# ----------------------------------------------------------------------------
def _heading_series(t, lat, lon):
    """Heading (deg, 0=N, 90=E) from GPS course-over-ground, using a forward point
    at least HEADING_MIN_DT later so adjacent-fix noise doesn't dominate."""
    n = len(t)
    heading = np.full(n, np.nan)
    for i in range(n):
        j = i
        while j < n - 1 and (t[j] - t[i]) < HEADING_MIN_DT:
            j += 1
        if j == i:
            continue
        dN = (lat[j] - lat[i]) * EARTH
        dE = (lon[j] - lon[i]) * EARTH * math.cos(math.radians(lat[i]))
        if dN == 0 and dE == 0:
            continue
        heading[i] = math.degrees(math.atan2(dE, dN)) % 360.0
    return pd.Series(heading).ffill().bfill().to_numpy()


def sync(det, tel):
    """Interpolate the drone state from `tel` to each detection's frame_time."""
    t = tel["time_s"].to_numpy()
    inside = (det["frame_time"] >= t.min()) & (det["frame_time"] <= t.max())
    if (~inside).any():
        logging.warning("Dropping %d detection(s) outside the GPS time range", int((~inside).sum()))
    out = det[inside].reset_index(drop=True)
    ft = out["frame_time"].to_numpy()

    for col, src in (("lat_uav", "lat"), ("lon_uav", "lon"), ("alt", "alt"), ("agl", "agl"),
                     ("grav_x", "grav_x"), ("grav_y", "grav_y"), ("grav_z", "grav_z")):
        out[col] = np.interp(ft, t, tel[src])

    # interpolate heading on the unit circle so 359 -> 1 deg doesn't pass through 180
    rad = np.radians(_heading_series(t, tel["lat"].to_numpy(), tel["lon"].to_numpy()))
    out["heading_deg"] = np.degrees(np.arctan2(np.interp(ft, t, np.sin(rad)),
                                               np.interp(ft, t, np.cos(rad)))) % 360.0
    logging.info("Synced %d detection(s) to telemetry", len(out))
    return out
