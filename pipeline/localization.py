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

V2 fixes (post first-mission debrief):
  * OFF-NADIR GATE (--max-offnadir, default 30 deg): oblique rays have
    h*tan(theta) lever arms that heading error rotates -- reject them.
  * AGL SANITY (--min-agl/--max-agl): per-detection heights outside the
    plausible mission band are rejected instead of silently scattering points.
  * DBSCAN + GEOMETRIC MEDIAN instead of greedy clustering + weighted mean.
  * TOP-K CAP (--top-k): never submit more flags than the mission expects.

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

import os
import sys
import csv
import json
import math
import shutil
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


def off_nadir_deg(rw):
    """Angle (deg) between a world ray and straight-down. 180 if degenerate."""
    n = np.linalg.norm(rw)
    if n == 0:
        return 180.0
    return math.degrees(math.acos(np.clip(rw[2] / n, -1.0, 1.0)))


def dbscan(P, eps, min_samples):
    """Minimal DBSCAN on an (n,2) metre array. Returns labels (-1 = noise).

    Replaces greedy first-point clustering: density-based, order-independent,
    no fragmentation of elongated scatter into arbitrary fixed-radius balls.
    """
    n = len(P)
    if n == 0:
        return np.empty(0, dtype=int)
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    neighbors = [np.flatnonzero(d2[i] <= eps * eps) for i in range(n)]
    core = np.array([len(nb) >= min_samples for nb in neighbors])
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cid = 0
    for i in range(n):
        if visited[i] or not core[i]:
            continue
        stack = [i]
        visited[i] = True
        labels[i] = cid
        while stack:
            j = stack.pop()
            for k in neighbors[j]:
                if labels[k] == -1:
                    labels[k] = cid          # border point
                if core[k] and not visited[k]:
                    visited[k] = True
                    stack.append(k)
        cid += 1
    return labels


def geometric_median(P, iters=200, tol=1e-6):
    """Weiszfeld geometric median of an (n,2) array — outlier-robust cluster
    centre (a weighted mean drags toward stray mis-localized detections)."""
    P = np.asarray(P, float)
    x = P.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(P - x, axis=1)
        d = np.where(d < 1e-9, 1e-9, d)
        w = 1.0 / d
        x_new = (P * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x


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


def write_submission(results, submit_dir, crops_dir):
    """Write the competition submission: targets.csv + a flags/ folder holding one
    representative crop per flag (what goes on the USB drive)."""
    flags_dir = os.path.join(submit_dir, "flags")
    os.makedirs(flags_dir, exist_ok=True)
    rows = []
    for res in results:
        fid = f"FLAG_{res['flag_id']:02d}"
        crop_out = ""
        rep_crop = res.get("rep_crop", "")
        if crops_dir and rep_crop:
            src = os.path.join(crops_dir, rep_crop)
            if os.path.isfile(src):
                crop_out = fid + ".jpg"
                shutil.copyfile(src, os.path.join(flags_dir, crop_out))
        rows.append([fid, res.get("class", ""), res["lat"], res["lon"],
                     res["n_detections"], res.get("rep_frame_time", ""),
                     res.get("mean_conf", ""), crop_out])
    with open(os.path.join(submit_dir, "targets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Flag_ID", "Flag_Class", "Latitude", "Longitude", "Detection_Count",
                    "Representative_Frame_Time", "Mean_Conf", "Representative_Crop"])
        w.writerows(rows)
    logging.info("submission: wrote %s/targets.csv + %d flag crop(s) in flags/",
                 submit_dir, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("synced")
    ap.add_argument("--alt", type=float, default=None,
                    help="Constant camera height above the flag (m). If omitted, uses the "
                         "per-detection AGL from the 'agl' column of synced.csv (handheld "
                         "clips with no real climb should still pass --alt, e.g. 1.5).")
    ap.add_argument("--camera", default=None, help="Calibration yaml (K + dist)")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--cluster-m", type=float, default=12.0,
                    help="DBSCAN eps (m): neighbourhood radius for grouping detections")
    ap.add_argument("--min-frames", type=int, default=5,
                    help="Drop flags seen in fewer detections than this")
    ap.add_argument("--max-offnadir", type=float, default=30.0,
                    help="Reject rays more than this many degrees from straight-down "
                         "(oblique rays have huge, heading-sensitive lever arms)")
    ap.add_argument("--min-agl", type=float, default=20.0,
                    help="Reject detections whose per-detection AGL is below this (m)")
    ap.add_argument("--max-agl", type=float, default=150.0,
                    help="Reject detections whose per-detection AGL is above this (m)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="Report at most this many flags, ranked by detection count "
                         "(mission 1: 2 flags + 1 bonus -> 3)")
    ap.add_argument("--cx", type=float, default=1352.0, help="placeholder cx if no yaml")
    ap.add_argument("--cy", type=float, default=760.0, help="placeholder cy if no yaml")
    ap.add_argument("--crops-dir", default=None,
                    help="Folder where detect.py saved crops (to copy the representative one)")
    ap.add_argument("--submit", default=None,
                    help="Output folder for the competition submission (targets.csv + flags/)")
    args = ap.parse_args()
    logging.info("localize: START synced=%s alt=%s camera=%s", args.synced, args.alt, args.camera)

    d = pd.read_csv(args.synced)
    K, dist, image_size = load_camera(args.camera, args.cx, args.cy)

    # Height source: a constant --alt overrides; otherwise per-detection AGL column.
    use_agl_col = args.alt is None
    if use_agl_col and "agl" not in d.columns:
        logging.error("no --alt given and no 'agl' column in %s", args.synced)
        sys.exit("Provide --alt, or re-run sync so synced.csv has an 'agl' column.")
    logging.info("Height source: %s | %d detections",
                 "per-detection AGL column" if use_agl_col else f"constant --alt={args.alt} m",
                 len(d))

    # Sanity: intrinsics are tied to the pixel resolution. If detections clearly
    # exceed the calibration image size, K is being used at the wrong scale.
    if image_size and len(d):
        if d.u.max() > image_size[0] * 1.02 or d.v.max() > image_size[1] * 1.02:
            logging.warning("detection pixels reach (%.0f,%.0f) but calibration is %dx%d. "
                            "Re-calibrate at the SAME resolution/lens, or metres will be off.",
                            d.u.max(), d.v.max(), image_size[0], image_size[1])

    lats, lons, confs, ENs = [], [], [], []
    crops, ftimes, classes = [], [], []
    rej_oblique = rej_agl = rej_rot = 0
    for _, r in d.iterrows():
        # 1-2: undistort -> normalized camera ray
        pt = np.array([[[float(r.u), float(r.v)]]], dtype=np.float64)
        xn, yn = cv2.undistortPoints(pt, K, dist)[0, 0]
        r_cam = np.array([xn, yn, 1.0])

        # 3: rotate to world using gravity + heading
        R = R_grav_heading([r.grav_x, r.grav_y, r.grav_z], r.heading_deg)
        if R is None:
            rej_rot += 1
            continue
        rw = R @ r_cam

        # 4a: OFF-NADIR GATE — total ray angle (camera tilt + pixel offset) vs
        # vertical. Oblique rays have lever arms of h*tan(theta) that heading
        # error rotates; near-nadir rays are insensitive to heading AND AGL.
        if off_nadir_deg(rw) > args.max_offnadir:
            rej_oblique += 1
            continue

        # 4b: height sanity. --alt (manual) is trusted as-is; the per-detection
        # AGL column must sit inside the plausible mission band.
        h = float(r.agl) if use_agl_col else args.alt
        if h <= 0 or (use_agl_col and not (args.min_agl <= h <= args.max_agl)):
            rej_agl += 1
            continue

        # 4c: ground intersection
        t = h / rw[2]
        dE, dN = t * rw[0], t * rw[1]

        # 5: offset -> GPS
        lat = r.lat_uav + dN / EARTH
        lon = r.lon_uav + dE / (EARTH * math.cos(math.radians(r.lat_uav)))
        lats.append(lat)
        lons.append(lon)
        confs.append(r.conf)
        ENs.append([dE, dN])
        crops.append(r.crop if "crop" in d.columns else "")
        ftimes.append(float(r.frame_time))
        classes.append(r["class"])

    if not lats:
        logging.error("no valid localizations: %d oblique, %d bad-AGL, %d bad-rotation "
                      "of %d detections. Check AGL anchor / tilt / --alt.",
                      rej_oblique, rej_agl, rej_rot, len(d))
        sys.exit("No valid localizations (all rays rejected). Check tilt/heading/h.")
    logging.info("Localized %d/%d detections (rejected: %d off-nadir>%.0fdeg, "
                 "%d AGL outside [%.0f,%.0f]m, %d bad rotation)",
                 len(lats), len(d), rej_oblique, args.max_offnadir,
                 rej_agl, args.min_agl, args.max_agl, rej_rot)
    if rej_agl > 0.5 * len(d):
        logging.warning("More than half the detections were rejected for implausible "
                        "AGL — the AGL anchor is probably wrong. Consider --alt.")

    # cluster per-detection GPS into flags (in local metres)
    lat0 = np.mean(lats)
    lon0 = np.mean(lons)
    pts_m = np.column_stack([
        (np.array(lons) - lon0) * EARTH * math.cos(math.radians(lat0)),
        (np.array(lats) - lat0) * EARTH])

    # DBSCAN (density-based, order-independent) instead of greedy first-point
    # clustering; min_samples ties cluster support to --min-frames.
    labels = dbscan(pts_m, eps=args.cluster_m,
                    min_samples=max(3, min(args.min_frames, 5)))
    n_noise = int((labels == -1).sum())
    cluster_ids = [c for c in np.unique(labels) if c != -1]
    logging.info("DBSCAN(eps=%.1f m): %d cluster(s), %d noise point(s) of %d",
                 args.cluster_m, len(cluster_ids), n_noise, len(pts_m))

    results = []
    confs = np.array(confs)
    lats, lons = np.array(lats), np.array(lons)
    crops = np.array(crops, dtype=object)
    ftimes = np.array(ftimes)
    classes = np.array(classes, dtype=object)
    clusters = [np.flatnonzero(labels == c) for c in cluster_ids]
    clusters = [idx for idx in clusters if len(idx) >= args.min_frames]
    clusters.sort(key=len, reverse=True)

    if args.top_k is not None and len(clusters) > args.top_k:
        logging.warning("capping output at top %d of %d clusters (by detection count) "
                        "— the dropped ones are likely scatter/duplicates",
                        args.top_k, len(clusters))
        clusters = clusters[:args.top_k]

    for k, idx in enumerate(clusters):
        # geometric median in local metres -> back to lat/lon (outlier-robust)
        mE, mN = geometric_median(pts_m[idx])
        rep = int(idx[int(np.argmax(confs[idx]))])  # representative = highest-conf detection
        results.append({
            "flag_id": k + 1,
            "class": str(classes[rep]),
            "lat": round(float(lat0 + mN / EARTH), 7),
            "lon": round(float(lon0 + mE / (EARTH * math.cos(math.radians(lat0)))), 7),
            "n_detections": int(len(idx)),
            "mean_conf": round(float(confs[idx].mean()), 3),
            "rep_frame_time": round(float(ftimes[rep]), 3),
            "rep_crop": str(crops[rep]),
        })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    logging.info("Found %d flag(s):", len(results))
    for res in results:
        logging.info("  flag %d: lat=%s lon=%s (%d det, conf %s)",
                     res["flag_id"], res["lat"], res["lon"],
                     res["n_detections"], res["mean_conf"])

    # --- competition submission: targets.csv + flags/<crop> for the USB drive ---
    if args.submit:
        write_submission(results, args.submit, args.crops_dir)

    logging.info("localize: DONE wrote %s", args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise CustomException(e, sys) from e
