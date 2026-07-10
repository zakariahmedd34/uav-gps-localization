"""
APPROACH 3 — ADVANCED PIPELINE
Sources:
  [2]  Lago et al. 2024 — DBSCAN GPS clustering
  [10] SAHI (2022) — Sliced inference for tiny flags at 100 m
  [7]  AIAA 2020 — Automatic mount angle discovery
  [9]  PMC Multi-target — Confidence-weighted aggregation
  [8]  Sensors AR — GPS COG as heading source

Added over Approach 2:
  - DBSCAN spatial clustering to reject isolated false-positive GPS estimates
  - Optional SAHI detection wrapper for high-altitude frames
  - GPS course-over-ground (COG) heading option
"""

import numpy as np
import math
import cv2

# Reuse rotation functions from Approach 2:
from approach2 import Rx, Ry, Rz, build_R_total, localize_full


# ─── DETECTION WITH OPTIONAL SAHI ────────────────────────────────────────────

def detect_flags_yolo(frame, yolo_model, use_sahi=False,
                      slice_size=512, overlap=0.2, conf_threshold=0.5):
    """
    Detect flags in a frame. Optionally uses SAHI for high-altitude frames.

    Args:
        frame           : BGR image (numpy array)
        yolo_model      : loaded Ultralytics YOLO model
        use_sahi        : True at altitudes >= 90 m (flag < 25 px)
        slice_size      : SAHI tile size in pixels
        overlap         : SAHI tile overlap ratio
        conf_threshold  : minimum confidence to accept detection

    Returns:
        list of (u, v, confidence) tuples
    """
    detections = []

    if use_sahi:
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            # Wrap model for SAHI (run once, reuse)
            result = get_sliced_prediction(
                frame,
                yolo_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
            )
            for pred in result.object_prediction_list:
                if pred.score.value >= conf_threshold:
                    bbox = pred.bbox
                    u = (bbox.minx + bbox.maxx) / 2
                    v = (bbox.miny + bbox.maxy) / 2
                    detections.append((u, v, pred.score.value))
        except ImportError:
            print("SAHI not installed — falling back to standard YOLO")
            use_sahi = False

    if not use_sahi:
        results = yolo_model(frame, conf=conf_threshold, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            u = (x1 + x2) / 2
            v = (y1 + y2) / 2
            c = float(box.conf[0])
            detections.append((u, v, c))

    return detections


# ─── GPS COG HEADING ─────────────────────────────────────────────────────────

def heading_from_gps_cog(lat1, lon1, lat2, lon2):
    """
    Compute true heading from GPS course-over-ground.
    More accurate than magnetometer for moving fixed-wing UAV [8].

    Args:
        (lat1, lon1) : previous GPS position
        (lat2, lon2) : current GPS position (both in degrees)

    Returns:
        True heading in radians (0 = North, π/2 = East)
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Convert to meters
    dlat_m = dlat * 111320.0
    dlon_m = dlon * 111320.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.atan2(dlon_m, dlat_m)  # atan2(East, North) = bearing from North


# ─── DBSCAN AGGREGATION ───────────────────────────────────────────────────────

def aggregate_approach3(estimates_with_confidence,
                        min_frames=10,
                        dbscan_eps_m=15.0,
                        dbscan_min_samples=3):
    """
    DBSCAN clustering + confidence-weighted mean.
    Source: [2] Lago et al. 2024 — removes isolated false-positive GPS estimates.

    Args:
        estimates_with_confidence : list of ((lat, lon), confidence) tuples
        min_frames                : minimum valid detections required
        dbscan_eps_m              : cluster radius in meters (15 m default)
        dbscan_min_samples        : minimum points to form a cluster

    Returns:
        (lat_final, lon_final) or None
    """
    from sklearn.cluster import DBSCAN

    valid = [(est, c) for est, c in estimates_with_confidence
             if est is not None and c > 0.4]

    if len(valid) < min_frames:
        return None

    coords   = np.array([est for est, _ in valid])
    conf_arr = np.array([c   for _, c   in valid])

    # Convert to local meters for clustering
    lat0 = np.mean(coords[:, 0])
    lon0 = np.mean(coords[:, 1])
    coords_m = np.column_stack([
        (coords[:, 0] - lat0) * 111320.0,
        (coords[:, 1] - lon0) * 111320.0 * math.cos(math.radians(lat0))
    ])

    # DBSCAN: find tight spatial cluster
    labels = DBSCAN(eps=dbscan_eps_m,
                    min_samples=dbscan_min_samples).fit_predict(coords_m)

    valid_mask = labels >= 0
    if not valid_mask.any():
        # No cluster found — fall back to confidence-weighted mean of all
        w = conf_arr / conf_arr.sum()
        return float(np.dot(w, coords[:, 0])), float(np.dot(w, coords[:, 1]))

    # Largest cluster
    cluster_labels = labels[valid_mask]
    best_label     = np.bincount(cluster_labels).argmax()
    cluster_mask   = labels == best_label

    cluster_coords = coords[cluster_mask]
    cluster_conf   = conf_arr[cluster_mask]

    # Confidence-weighted mean within cluster
    w       = cluster_conf / cluster_conf.sum()
    lat_fin = float(np.dot(w, cluster_coords[:, 0]))
    lon_fin = float(np.dot(w, cluster_coords[:, 1]))

    cluster_size = cluster_mask.sum()
    rejected     = len(valid) - cluster_size
    print(f"  DBSCAN: {cluster_size} frames in cluster, {rejected} outliers rejected")

    return lat_fin, lon_fin


# ─── FULL VIDEO PROCESSING LOOP ──────────────────────────────────────────────

def process_video_for_flags(video_path, telemetry_log,
                             K, dist, R_mount,
                             yolo_model,
                             frame_sample_rate=5,
                             altitude_sahi_threshold=90.0):
    """
    Process a full mission video and return GPS estimates for all detected flags.

    Args:
        video_path            : path to .mp4 video file
        telemetry_log         : dict mapping frame_index -> {lat, lon, h, roll, pitch, yaw_mag}
        K, dist, R_mount      : camera parameters
        yolo_model            : loaded Ultralytics YOLO model
        frame_sample_rate     : process every Nth frame (5 = 12 fps from 60 fps)
        altitude_sahi_threshold : use SAHI when h > this value (meters)

    Returns:
        dict: {flag_id: (lat_final, lon_final)}
    """
    CAIRO_DECLINATION_RAD = math.radians(4.0)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    all_estimates = []   # (estimate, confidence) per frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_sample_rate != 0:
            frame_idx += 1
            continue

        # Get synchronized telemetry for this frame
        telem = telemetry_log.get(frame_idx)
        if telem is None:
            frame_idx += 1
            continue

        lat_uav   = telem["lat"]
        lon_uav   = telem["lon"]
        h         = telem["h"]
        roll      = math.radians(telem["roll_deg"])
        pitch     = math.radians(telem["pitch_deg"])
        yaw_true  = math.radians(telem["yaw_mag_deg"]) + CAIRO_DECLINATION_RAD

        # Use SAHI at high altitude where flags are < 25 px
        use_sahi = h > altitude_sahi_threshold

        detections = detect_flags_yolo(frame, yolo_model, use_sahi=use_sahi)

        for u, v, conf in detections:
            est = localize_full(u, v, K, dist, h,
                                lat_uav, lon_uav,
                                roll, pitch, yaw_true,
                                R_mount)
            all_estimates.append((est, conf))

        frame_idx += 1

    cap.release()

    # Aggregate all estimates → final flag GPS
    final_gps = aggregate_approach3(all_estimates, min_frames=10)
    return final_gps


# ─── AUTOMATIC MOUNT CALIBRATION ─────────────────────────────────────────────

def auto_calibrate_mount(known_gps_target, observations, K, dist, h_mean,
                         lat_uav_mean, lon_uav_mean):
    """
    Automatically discover the camera mount offset angles.
    Source: [7] AIAA 2020 — Improving UAV-Based Target Geolocation Accuracy.

    Flies over a known GPS point, collects multiple detections, then
    optimizes mount roll/pitch/yaw offsets to minimize localization error.

    Args:
        known_gps_target : (lat_true, lon_true) of the ground marker
        observations     : list of (u, v, roll_rad, pitch_rad, yaw_rad) per frame
        K, dist          : camera intrinsics
        h_mean           : approximate flight altitude
        lat_uav_mean     : approximate UAV latitude during pass
        lon_uav_mean     : approximate UAV longitude during pass

    Returns:
        R_mount_optimized : 3x3 optimized camera-to-body rotation
    """
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    lat_true, lon_true = known_gps_target

    def error_fn(euler_xyz):
        """Total squared error across all observations."""
        droll, dpitch, dyaw = euler_xyz
        R_adj = Rotation.from_euler('xyz', [droll, dpitch, dyaw]).as_matrix()

        total_err = 0.0
        for u, v, roll, pitch, yaw in observations:
            r_world = build_R_total(roll + droll, pitch + dpitch,
                                    yaw + dyaw, R_adj) @ \
                      np.array([(u - K[0,2]) / K[0,0],
                                (v - K[1,2]) / K[1,1],
                                1.0])
            if abs(r_world[2]) < 0.05:
                continue
            t  = h_mean / r_world[2]
            dE = t * r_world[0]
            dN = t * r_world[1]
            lat_est = lat_uav_mean + dN / 111320.0
            lon_est = lon_uav_mean + dE / (111320.0 * math.cos(math.radians(lat_uav_mean)))
            # Error in meters
            err_m = math.sqrt(((lat_est - lat_true) * 111320) ** 2 +
                              ((lon_est - lon_true) * 111320 * math.cos(math.radians(lat_true))) ** 2)
            total_err += err_m ** 2

        return total_err

    result = minimize(error_fn, [0.0, 0.0, 0.0],
                      method='Nelder-Mead',
                      options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 5000})

    droll, dpitch, dyaw = result.x
    R_mount_opt = Rotation.from_euler('xyz', [droll, dpitch, dyaw]).as_matrix()
    print(f"Mount offset: roll={math.degrees(droll):.2f}°, "
          f"pitch={math.degrees(dpitch):.2f}°, yaw={math.degrees(dyaw):.2f}°")
    print(f"Residual RMS error: {math.sqrt(result.fun / max(len(observations), 1)):.2f} m")
    return R_mount_opt