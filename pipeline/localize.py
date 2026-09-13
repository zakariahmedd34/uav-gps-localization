"""
localize.py - synced detections -> GPS coordinates of each flag.

Per detection:
  1. undistort the pixel with the Hero 13 Linear calibration
  2. rotate the camera ray to the world frame using gravity (tilt) + GPS heading
  3. reject rays more than MAX_OFFNADIR from vertical (heading error dominates there)
  4. intersect the flat ground at the drone's AGL -> East/North offset -> lat/lon
Then group detections into flags: tracklets (one flag across consecutive frames)
-> DBSCAN on tracklet centres -> geometric median per flag -> top TOP_K by count.

World frame: x=East, y=North, z=Down, drone at the origin, ground at z=h.
Camera frame: x=image right, y=image down, z=optical axis.
"""

import os
import csv
import math
import shutil
import logging

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION = os.path.join(ROOT, "calibration", "hero13_linear_4k.yaml")
EARTH = 111320.0        # metres per degree of latitude
MAX_OFFNADIR = 30.0     # deg; oblique rays have h*tan(angle) lever arms that heading error rotates
MIN_AGL, MAX_AGL = 20.0, 150.0   # m; telemetry AGL outside this band is rejected
CLUSTER_M = 12.0        # m; DBSCAN radius for merging tracklet centres into one flag
MIN_FRAMES = 3          # default detections needed for an official flag (suits 1 sample/s)
TOP_K = 3               # default max flags submitted (mission 1: 2 flags + 1 bonus)


def load_camera(vid_w, vid_h):
    """Return (K, dist) for the calibration, with K scaled if the video resolution
    differs from the calibrated 4K (distortion is normalized, so it is kept)."""
    with open(CALIBRATION) as f:
        c = yaml.safe_load(f)
    cm = c["camera_matrix"]
    K = np.array([[cm["fx"], 0, cm["cx"]], [0, cm["fy"], cm["cy"]], [0, 0, 1]], dtype=np.float64)
    dist = np.array(c["dist_coeffs"], dtype=np.float64)
    sx, sy = vid_w / c["image_width"], vid_h / c["image_height"]
    if abs(sx - 1) > 0.02 or abs(sy - 1) > 0.02:
        K[0] *= sx      # fx, cx
        K[1] *= sy      # fy, cy
        logging.warning("Video is %dx%d but calibration is %dx%d: scaled K by (%.4f, %.4f). "
                        "Record in 4K Linear for full accuracy.",
                        vid_w, vid_h, c["image_width"], c["image_height"], sx, sy)
    return K, dist


def off_nadir_deg(rw):
    """Angle (deg) between a world ray and straight down. 180 if degenerate."""
    n = np.linalg.norm(rw)
    if n == 0:
        return 180.0
    return math.degrees(math.acos(np.clip(rw[2] / n, -1.0, 1.0)))


def dbscan(P, eps, min_samples):
    """Minimal DBSCAN on an (n,2) metre array. Returns labels (-1 = noise)."""
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
    """Weiszfeld geometric median of an (n,2) array - an outlier-robust centre."""
    P = np.asarray(P, float)
    x = P.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(P - x, axis=1)
        w = 1.0 / np.where(d < 1e-9, 1e-9, d)
        x_new = (P * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def build_tracklets(frame_time, u, v, max_gap_s=1.5, px_per_s=800.0):
    """Tracklet id per detection by temporal + pixel continuity: one flag seen
    during one pass = one tracklet. Two detections in the same frame never share one."""
    order = np.argsort(frame_time, kind="stable")
    tids = np.full(len(frame_time), -1, dtype=int)
    active = []          # [tid, last_t, last_u, last_v]
    next_id = 0
    for i in order:
        t, uu, vv = float(frame_time[i]), float(u[i]), float(v[i])
        best, best_d = None, None
        for tr in active:
            dt = t - tr[1]
            if dt <= 1e-9 or dt > max_gap_s:
                continue                      # same frame or too old
            dist = math.hypot(uu - tr[2], vv - tr[3])
            if dist <= px_per_s * dt + 50.0 and (best_d is None or dist < best_d):
                best, best_d = tr, dist
        if best is None:
            best = [next_id, t, uu, vv]
            next_id += 1
            active.append(best)
        else:
            best[1], best[2], best[3] = t, uu, vv
        tids[i] = best[0]
        active = [tr for tr in active if t - tr[1] <= max_gap_s]
    return tids


def R_grav_heading(grav, heading_deg):
    """Camera->world rotation: align gravity with world-down, then yaw by heading (0=North)."""
    g = np.asarray(grav, float)
    n = np.linalg.norm(g)
    if n == 0:
        return None
    g = g / n
    target = np.array([0.0, 0.0, 1.0])
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
                   [math.sin(psi), math.cos(psi), 0],
                   [0, 0, 1]])
    return Rz @ R_level


def localize(d, crops_dir, submit_dir, alt=None, min_frames=MIN_FRAMES, top_k=TOP_K):
    """Localize flags from synced detections `d` and write the submission.
    `alt` = fixed camera height above the ground (m); None = per-detection telemetry AGL.
    A flag needs `min_frames` detections; at most `top_k` flags are submitted (0 = no limit)."""
    if d.empty:
        raise RuntimeError("No flag detections in the processed video section.")
    K, dist = load_camera(int(d["vid_w"].iloc[0]), int(d["vid_h"].iloc[0]))
    logging.info("Height source: %s", "telemetry AGL" if alt is None else f"--alt {alt} m")
    all_tids = build_tracklets(d["frame_time"].to_numpy(), d["u"].to_numpy(), d["v"].to_numpy())

    lats, lons, kept = [], [], []
    rej_oblique = rej_agl = rej_rot = 0
    for i, (_, r) in enumerate(d.iterrows()):
        pt = np.array([[[float(r.u), float(r.v)]]], dtype=np.float64)
        xn, yn = cv2.undistortPoints(pt, K, dist)[0, 0]
        R = R_grav_heading([r.grav_x, r.grav_y, r.grav_z], r.heading_deg)
        if R is None:
            rej_rot += 1
            continue
        rw = R @ np.array([xn, yn, 1.0])
        if off_nadir_deg(rw) > MAX_OFFNADIR:
            rej_oblique += 1
            continue
        h = float(r.agl) if alt is None else alt     # a manual --alt is trusted as-is
        if h <= 0 or (alt is None and not MIN_AGL <= h <= MAX_AGL):
            rej_agl += 1
            continue
        t = h / rw[2]
        dE, dN = t * rw[0], t * rw[1]
        lats.append(r.lat_uav + dN / EARTH)
        lons.append(r.lon_uav + dE / (EARTH * math.cos(math.radians(r.lat_uav))))
        kept.append(i)

    logging.info("Localized %d/%d detections (rejected: %d off-nadir > %.0f deg, %d AGL outside "
                 "%.0f-%.0f m, %d bad gravity)", len(kept), len(d), rej_oblique, MAX_OFFNADIR,
                 rej_agl, MIN_AGL, MAX_AGL, rej_rot)
    if not kept:
        raise RuntimeError("No valid localizations (all rays rejected). Check camera tilt, "
                           "GPS lock and AGL (try --alt).")
    if rej_agl > 0.5 * len(d):
        logging.warning("More than half the detections had implausible AGL - the AGL anchor "
                        "is probably wrong. Consider --alt.")

    kept = np.array(kept)
    confs = d["conf"].to_numpy()[kept]
    ftimes = d["frame_time"].to_numpy(dtype=float)[kept]
    crops = d["crop"].to_numpy()[kept]
    classes = d["class"].to_numpy()[kept]
    tks = all_tids[kept]

    # cluster in local metres around the mean position
    lat0, lon0 = np.mean(lats), np.mean(lons)
    m_per_deg_lon = EARTH * math.cos(math.radians(lat0))
    pts_m = np.column_stack([(np.array(lons) - lon0) * m_per_deg_lon,
                             (np.array(lats) - lat0) * EARTH])

    # one geometric-median centre per tracklet, then merge centres within CLUSTER_M.
    # This separates flags whose scattered ground points overlap.
    uniq = np.unique(tks)
    centres = np.array([geometric_median(pts_m[tks == tid]) for tid in uniq])
    c_labels = dbscan(centres, eps=CLUSTER_M, min_samples=1)      # min_samples=1: no noise
    tid_to_label = dict(zip(uniq, c_labels))
    labels = np.array([tid_to_label[tid] for tid in tks])
    clusters = [np.flatnonzero(labels == c) for c in np.unique(c_labels)]
    clusters.sort(key=len, reverse=True)
    logging.info("%d tracklet(s) -> %d candidate flag(s)", len(uniq), len(clusters))

    candidates = []
    for idx in clusters:
        mE, mN = geometric_median(pts_m[idx])
        rep = int(idx[int(np.argmax(confs[idx]))])     # highest-confidence detection
        candidates.append({
            "lat": round(float(lat0 + mN / EARTH), 7),
            "lon": round(float(lon0 + mE / m_per_deg_lon), 7),
            "n": int(len(idx)),
            "conf": round(float(confs[idx].mean()), 3),
            "time": round(float(ftimes[rep]), 3),
            "crop": str(crops[rep]),
            "class": str(classes[rep]),
        })
    official = [c for c in candidates if c["n"] >= min_frames]
    if top_k > 0 and len(official) > top_k:
        logging.warning("Keeping the top %d of %d flags (by detection count); the rest are "
                        "in candidates.csv for review", top_k, len(official))
        official = official[:top_k]
    _write_submission(official, candidates, crops_dir, submit_dir)

    for k, c in enumerate(official, 1):
        logging.info("FLAG_%02d: lat=%s lon=%s (%d detections, conf %s)",
                     k, c["lat"], c["lon"], c["n"], c["conf"])
    return official


def _write_submission(official, candidates, crops_dir, submit_dir):
    """targets.csv + flags/ = the official submission (what goes on the USB drive).
    candidates.csv + candidates_flags/ = every candidate, for a human to review."""
    os.makedirs(os.path.join(submit_dir, "flags"), exist_ok=True)
    os.makedirs(os.path.join(submit_dir, "candidates_flags"), exist_ok=True)

    def copy_crop(c, folder, name):
        src = os.path.join(crops_dir, c["crop"])
        if not c["crop"] or not os.path.isfile(src):
            return ""
        shutil.copyfile(src, os.path.join(submit_dir, folder, name))
        return name

    with open(os.path.join(submit_dir, "targets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Flag_ID", "Flag_Class", "Latitude", "Longitude", "Detection_Count",
                    "Representative_Frame_Time", "Mean_Conf", "Representative_Crop"])
        for k, c in enumerate(official, 1):
            fid = f"FLAG_{k:02d}"
            w.writerow([fid, c["class"], c["lat"], c["lon"], c["n"], c["time"], c["conf"],
                        copy_crop(c, "flags", fid + ".jpg")])

    official_ids = {id(c) for c in official}
    with open(os.path.join(submit_dir, "candidates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Candidate_ID", "Status", "Latitude", "Longitude", "Detection_Count",
                    "Mean_Conf", "Rep_Frame_Time", "Crop"])
        for k, c in enumerate(candidates, 1):
            cid = f"CAND_{k:02d}"
            w.writerow([cid, "OFFICIAL" if id(c) in official_ids else "filtered_out",
                        c["lat"], c["lon"], c["n"], c["conf"], c["time"],
                        copy_crop(c, "candidates_flags", cid + ".jpg")])
    logging.info("Submission written to %s (%d flag(s), %d candidate(s))",
                 submit_dir, len(official), len(candidates))
