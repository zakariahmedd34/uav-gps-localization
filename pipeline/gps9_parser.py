"""
Custom GPS9 decoder.

telemetrik (the library) only has special handling for the older GPS5
stream. Hero 11/12/13 cameras record GPS9 instead, which packs 9 fields
per sample:

    lat, lon, alt, speed_2d, speed_3d,
    days_since_2000, secs_since_midnight, dop, fix

Two things make GPS9 trickier than GPS5:

1. It's a *complex* GPMF type (its own header type char is '?' /
   GPMF_TYPE_COMPLEX) -- the fields are NOT uniform 4-byte values. The
   real per-field layout (e.g. "llllllSSL") is given by a sibling TYPE
   key inside the same STRM, one character per field, each meaning a
   different byte width. Assuming uniform 4-byte fields silently reads
   the wrong bytes for everything past the first couple of fields.
2. Its SCAL (scale/divisor) metadata is a multi-element array -- one
   divisor per field -- not a single shared divisor.

Reference: https://github.com/gopro/gpmf-parser (GPS9 / TYPE / SCAL)
"""

import struct

from telemetrik.parser import get_gpmf_boxes, _from_bytes

GPS9_FIELDS = [
    "lat", "lon", "alt", "speed_2d", "speed_3d",
    "days_since_2000", "secs_since_midnight", "dop", "fix",
]
N_FIELDS = len(GPS9_FIELDS)

# GPMF type-char -> byte width (per gpmf-parser's GPMF_SampleType table)
_TYPE_WIDTHS = {
    "b": 1, "B": 1, "c": 1,
    "s": 2, "S": 2,
    "l": 4, "L": 4, "f": 4, "F": 4, "q": 4,
    "d": 8, "j": 8, "J": 8, "Q": 8,
    "U": 16, "G": 16,
}


def _expand_type_string(s):
    """Expand bracket-notation like 'f[8]L' -> 'ffffffffL'."""
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if i + 1 < len(s) and s[i + 1] == "[":
            j = s.index("]", i + 1)
            n = int(s[i + 2:j])
            out.append(c * n)
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _read_typed_value(f, type_char):
    width = _TYPE_WIDTHS.get(type_char, 4)
    raw = f.read(width)
    if type_char == "f":
        return struct.unpack(">f", raw)[0]
    if type_char == "d":
        return struct.unpack(">d", raw)[0]
    signed = type_char.islower()
    return int.from_bytes(raw, "big", signed=signed)


def _read_scale_array(f, scal_box, n_fields):
    """Read SCAL as either a single shared divisor or a per-field array."""
    f.seek(scal_box.offset + 8)
    width = scal_box.struct_size
    n = scal_box.repeat
    divisors = []
    for _ in range(n):
        v = _from_bytes(f.read(width), signed=True)
        divisors.append(v if v != 0 else 1)
    if n == 1:
        divisors = divisors * n_fields
    elif n != n_fields:
        divisors = (divisors + [1] * n_fields)[:n_fields]
    return divisors


def get_gps9_stream(f, samples, time_base=None, debug=True):
    """
    Returns a list of (pts_seconds, {field_name: value}) for GPS9,
    using the video's actual PTS timeline (same convention as
    telemetrik's GPS5/GRAV pts_data).
    """
    out = []

    stmp_timestamps = []
    sample_pts_list = []
    payload_info = []
    field_types = None  # discovered once, reused (TYPE is sticky)

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
        stmp_us = _from_bytes(f.read(stmp.struct_size * stmp.repeat))
        stmp_ms = stmp_us / 1000

        gps9_box = get_gpmf_boxes(f, strm.offset, strm.size, ["STRM", "GPS9"])[0]

        if field_types is None:
            type_boxes = get_gpmf_boxes(f, strm.offset, strm.size, ["STRM", "TYPE"])
            if type_boxes:
                tbox = type_boxes[0]
                f.seek(tbox.offset + 8)
                raw_type_str = f.read(tbox.struct_size * tbox.repeat).decode(
                    "ascii", errors="ignore"
                ).rstrip("\x00")
                expanded = _expand_type_string(raw_type_str)
                if len(expanded) == N_FIELDS:
                    field_types = list(expanded)

            if field_types is None:
                # Fallback: assume uniform signed-long fields if no TYPE
                # box was found (shouldn't normally happen for GPS9).
                field_types = ["l"] * N_FIELDS

            if debug:
                print(f"[gps9] field_types={''.join(field_types)} "
                      f"(expected layout: lat,lon,alt,spd2d,spd3d,days,secs,dop,fix)")

        scal_boxes = get_gpmf_boxes(f, strm.offset, strm.size, ["STRM", "SCAL"])
        divisors = (
            _read_scale_array(f, scal_boxes[0], N_FIELDS) if scal_boxes else [1] * N_FIELDS
        )
        if debug and len(payload_info) == 0:
            print(f"[gps9] first-payload SCAL divisors={divisors}")

        stmp_timestamps.append(stmp_ms)
        sample_pts_list.append(sample.pts if sample.pts is not None else 0)
        payload_info.append((gps9_box, divisors))

    reading_duration_ms = None
    reading_duration_pts = None

    for idx, (gps9_box, divisors) in enumerate(payload_info):
        f.seek(gps9_box.offset + 8)
        base_ms = stmp_timestamps[idx]
        base_pts = sample_pts_list[idx]

        if idx + 1 < len(stmp_timestamps):
            delta_ms = stmp_timestamps[idx + 1] - base_ms
            reading_duration_ms = delta_ms / gps9_box.repeat
            if time_base is not None:
                delta_pts = sample_pts_list[idx + 1] - base_pts
                reading_duration_pts = delta_pts / gps9_box.repeat

        for i in range(gps9_box.repeat):
            raw = [_read_typed_value(f, t) for t in field_types]
            values = [r / d for r, d in zip(raw, divisors)]
            record = dict(zip(GPS9_FIELDS, values))

            if time_base is not None and reading_duration_pts is not None:
                pts_value = base_pts + (i * reading_duration_pts)
                pts_seconds = pts_value * time_base[0] / time_base[1]
            else:
                pts_seconds = (base_ms + i * (reading_duration_ms or 0)) / 1000

            out.append((pts_seconds, record))

    return out