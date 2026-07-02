"""
localization.py — synced.csv -> flag GPS coordinates (results.json)

Self-contained: all geometry lives in this file (no imports from src/).

For each synced detection:
  1. undistort the pixel  (Brown-Conrady, via cv2.undistortPoints)
  2. build a camera ray
  3. rotate the ray to the world frame using the GRAVITY vector (tilt) + heading
  4. intersect the ground plane at height h  ->  East/North offset
  5. convert offset to GPS  ->  per-detection (lat, lon)
Then cluster the per-detection GPS points into flags and aggregate each cluster.

WORLD FRAME: x=East, y=North, z=Down. UAV at origin, ground at z=h.
CAMERA FRAME: x=image-right, y=image-down, z=optical-axis(into scene).
  GRAV is expressed in this camera frame (z~1 when the camera looks straight down).

IMPORTANT inputs you must set correctly:
  --alt   = height of the camera ABOVE THE FLAG (metres). For the handheld test
            clip this is ~1.5, NOT the 173 m MSL in the telemetry. On a real
            flight use AGL = GPS_alt - takeoff_ground_elevation.
  --camera= calibration yaml (K + dist). Without it, PLACEHOLDER intrinsics are
            used and the metric result is only a plumbing check, not accurate.

Usage:
  python localization.py synced.csv --alt 1.5 --out results.json
  python localization.py synced.csv --alt 80 --camera ../configs/camera_params_hero13.yaml
"""

import sys
import json
import math
import argparse
import numpy as np
import pandas as pd

from logger import logging
from exception import CustomException

try:
    import cv2
except ImportError:
    logging.error("opencv-python is required (pip install opencv-python)")
    sys.exit("ERROR: opencv-python is required (pip install opencv-python).")

EARTH = 111320.0


def load_camera(path, default_cx, default_cy):
    """Return (K, dist, image_size).

    Accepts BOTH calibration-yaml schemas so the file calibrate_camera.py
    writes works directly:
      (a) calibrate_camera.py schema:
            camera_matrix: {fx, fy, cx, cy}
            dist_coeffs:   [k1, k2, p1, p2, k3, ...]
            image_width / image_height
      (b) legacy/manual schema:
            K:    [[fx,0,cx],[0,fy,cy],[0,0,1]]
            dist: [k1, k2, p1, p2, k3, ...]
    Falls back to a loud placeholder when no yaml is given.

    image_size is (width, height) or None — used only to warn if the
    detections were run at a different resolution than the calibration.
    """
    if path:
        import yaml
        with open(path) as f:
            c = yaml.safe_load(f) or {}

        image_size = None
        if "image_width" in c and "image_height" in c:
            image_size = (int(c["image_width"]), int(c["image_height"]))

        if "camera_matrix" in c:                      # schema (a)
            cm = c["camera_matrix"]
            fx, fy = float(cm["fx"]), float(cm["fy"])
            cx, cy = float(cm["cx"]), float(cm["cy"])
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            dist = np.array(c.get("dist_coeffs", []), dtype=np.float64).ravel()
        elif "K" in c:                                # schema (b)
            K = np.array(c["K"], dtype=np.float64)
            dist = np.array(c.get("dist", c.get("dist_coeffs", [])),
                            dtype=np.float64).ravel()
        else:
            logging.error("calibration yaml '%s' has neither 'camera_matrix' nor 'K'", path)
            sys.exit(
                f"ERROR: calibration yaml '{path}' has neither 'camera_matrix' "
                "(fx/fy/cx/cy) nor 'K'. Re-run calibrate_camera.py or fix the file."
            )

        if dist.size == 0:
            logging.warning("no distortion coeffs in yaml; assuming zero "
                            "(fine only if you flew the Linear lens)")
            dist = np.zeros(5)

        rms = c.get("rms_reprojection_error_px")
        rms_str = f" rms={rms:.3f}px" if isinstance(rms, (int, float)) else ""
        size_str = f" @ {image_size[0]}x{image_size[1]}" if image_size else ""
        logging.info("Camera: loaded %s fx=%.1f fy=%.1f cx=%.1f cy=%.1f%s%s",
                     path, K[0, 0], K[1, 1], K[0, 2], K[1, 2], size_str, rms_str)
        return K, dist, image_size

    # PLACEHOLDER — replace with real calibration before trusting metres
    fx = fy = 1500.0
    K = np.array([[fx, 0, default_cx], [0, fy, default_cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)
    logging.warning("*** PLACEHOLDER intrinsics (fx=1500, dist=0) — plumbing only, "
                    "NOT accurate. Calibrate the Hero 13 and pass --camera. ***")
    return K, dist, None


def R_grav_heading(grav, heading_deg):
    """Camera->world rotation: align gravity to world-down, then yaw by heading.

    grav        : (3,) gravity vector in camera frame (need not be unit)
    heading_deg : rotation about world-down (0=North), sets azimuth
    """
    g = np.asarray(grav, float)
    n = np.linalg.norm(g)
    if n == 0:
        return None
    g = g / n
    target = np.array([0.0, 0.0, 1.0])           # world down in camera-aligned frame
    axis = np.cross(g, target)
    s = np.linalg.norm(axis)
    c = float(np.dot(g, target))
    if s < 1e-9:                                  # already aligned (or flipped)
        R_level = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        axis /= s
        Kx = np.array([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]])
        R_level = np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)   # Rodrigues
    psi = math.radians(heading_deg)
    Rz = np.array([[math.cos(psi), -math.sin(psi), 0],
                   [math.sin(psi),  math.cos(psi), 0],
                   [0, 0, 1]])
    return Rz @ R_level


def cluster_to_flags(points_m, threshold_m):
    """Greedy clustering of (E,N) metre points. Returns list of index lists."""
    clusters, assigned = [], [False] * len(points_m)
    for i in range(len(points_m)):
        if assigned[i]:
            continue
        members = [i]
        assigned[i] = True
        ci = points_m[i]
        for j in range(i + 1, len(points_m)):
            if not assigned[j] and np.hypot(*(points_m[j] - ci)) < threshold_m:
                members.append(j)
                assigned[j] = True
        clusters.append(members)
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("synced")
    ap.add_argument("--alt", type=float, required=True,
                    help="Camera height ABOVE the flag (m). Handheld test ~1.5; flight = AGL.")
    ap.add_argument("--camera", default=None, help="Calibration yaml (K + dist)")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--cluster-m", type=float, default=25.0,
                    help="Max distance (m) to group detections into the same flag")
    ap.add_argument("--min-frames", type=int, default=5,
                    help="Drop flags seen in fewer detections than this")
    ap.add_argument("--cx", type=float, default=1352.0, help="placeholder cx if no yaml")
    ap.add_argument("--cy", type=float, default=760.0, help="placeholder cy if no yaml")
    args = ap.parse_args()
    logging.info("localize: START synced=%s alt=%s camera=%s", args.synced, args.alt, args.camera)

    d = pd.read_csv(args.synced)
    K, dist, image_size = load_camera(args.camera, args.cx, args.cy)
    h = args.alt
    logging.info("Height above flag h = %s m | %d detections", h, len(d))

    # Sanity: intrinsics are tied to the pixel resolution. If detections clearly
    # exceed the calibration image size, K is being used at the wrong scale.
    if image_size and len(d):
        if d.u.max() > image_size[0] * 1.02 or d.v.max() > image_size[1] * 1.02:
            logging.warning("detection pixels reach (%.0f,%.0f) but calibration is %dx%d. "
                            "Re-calibrate at the SAME resolution/lens, or metres will be off.",
                            d.u.max(), d.v.max(), image_size[0], image_size[1])

    lats, lons, confs, ENs = [], [], [], []
    rejected = 0
    for _, r in d.iterrows():
        # 1-2: undistort -> normalized camera ray
        pt = np.array([[[float(r.u), float(r.v)]]], dtype=np.float64)
        xn, yn = cv2.undistortPoints(pt, K, dist)[0, 0]
        r_cam = np.array([xn, yn, 1.0])

        # 3: rotate to world using gravity + heading
        R = R_grav_heading([r.grav_x, r.grav_y, r.grav_z], r.heading_deg)
        if R is None:
            rejected += 1
            continue
        rw = R @ r_cam

        # 4: ground intersection (ray must point downward, +z)
        if rw[2] < 0.1:
            rejected += 1
            continue
        t = h / rw[2]
        dE, dN = t * rw[0], t * rw[1]

        # 5: offset -> GPS
        lat = r.lat_uav + dN / EARTH
        lon = r.lon_uav + dE / (EARTH * math.cos(math.radians(r.lat_uav)))
        lats.append(lat)
        lons.append(lon)
        confs.append(r.conf)
        ENs.append([dE, dN])

    if not lats:
        logging.error("no valid localizations (all rays rejected). Check tilt/heading/h")
        sys.exit("No valid localizations (all rays rejected). Check tilt/heading/h.")
    logging.info("Localized %d detections (%d rejected as oblique)", len(lats), rejected)

    # cluster per-detection GPS into flags (in local metres)
    lat0 = np.mean(lats)
    pts_m = np.column_stack([
        (np.array(lons) - np.mean(lons)) * EARTH * math.cos(math.radians(lat0)),
        (np.array(lats) - lat0) * EARTH])
    clusters = cluster_to_flags(pts_m, args.cluster_m)

    results = []
    confs = np.array(confs)
    lats, lons = np.array(lats), np.array(lons)
    for k, members in enumerate(sorted(clusters, key=len, reverse=True)):
        if len(members) < args.min_frames:
            continue
        idx = np.array(members)
        w = confs[idx] / confs[idx].sum()           # confidence-weighted mean
        results.append({
            "flag_id": k + 1,
            "lat": round(float(np.dot(w, lats[idx])), 7),
            "lon": round(float(np.dot(w, lons[idx])), 7),
            "n_detections": int(len(members)),
            "mean_conf": round(float(confs[idx].mean()), 3),
        })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    logging.info("Found %d flag(s):", len(results))
    for res in results:
        logging.info("  flag %d: lat=%s lon=%s (%d det, conf %s)",
                     res["flag_id"], res["lat"], res["lon"],
                     res["n_detections"], res["mean_conf"])
    logging.info("localize: DONE wrote %s", args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise CustomException(e, sys) from e
