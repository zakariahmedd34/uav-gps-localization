"""
APPROACH 1 — SIMPLIFIED BASELINE
Source: Simplified version from UAV_GPS_Localization_Engineering_Report.md
Assumption: Camera is perfectly nadir, no IMU, no distortion correction.
Use as sanity check only.
"""
import numpy as np
import math

def localize_baseline(u, v, K, h, lat_uav, lon_uav):
    """
    Simplified nadir localization — no IMU, no undistortion.
    Args:
        u, v       : raw pixel coordinates of YOLO bbox center
        K          : 3x3 camera intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
        h          : altitude AGL in meters
        lat_uav    : UAV latitude (degrees)
        lon_uav    : UAV longitude (degrees)

    Returns:
        (lat_flag, lon_flag) or None
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Normalize pixel offset (no undistortion)
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy

    # Nadir ground offset (image +Y is down, so North = -y_n * h)
    dE = x_n * h       # East  (meters)
    dN = -y_n * h      # North (meters) — negative sign: image-down = world-south

    # GPS conversion
    dlat = dN / 111320.0
    dlon = dE / (111320.0 * math.cos(math.radians(lat_uav)))

    return lat_uav + dlat, lon_uav + dlon


def aggregate_approach1(estimates):
    """
    Simple median aggregation.

    Args:
        estimates : list of (lat, lon) tuples

    Returns:
        (lat_final, lon_final) or None
    """
    if len(estimates) < 3:
        return None
    coords = np.array(estimates)
    return tuple(np.median(coords, axis=0))


# ─── USAGE EXAMPLE ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example camera (Hero 9 Linear 1080p — replace with calibrated values)
    K = np.array([[1432, 0, 960],
                  [0, 1432, 540],
                  [0,    0,   1]], dtype=float)

    # UAV state (from telemetry log)
    h       = 80.0          # meters AGL
    lat_uav = 30.0613       # degrees (example Cairo-area lat)
    lon_uav = 31.2497       # degrees (example Cairo-area lon)

    # Detection from YOLO (example)
    detections = [(950, 560), (952, 558), (948, 562)]  # (u, v) per frame

    estimates = []
    for u, v in detections:
        est = localize_baseline(u, v, K, h, lat_uav, lon_uav)
        estimates.append(est)

    result = aggregate_approach1(estimates)
    print(f"Approach 1 → lat: {result[0]:.7f}, lon: {result[1]:.7f}")