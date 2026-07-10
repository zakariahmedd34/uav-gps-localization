"""
APPROACH 2 — FULL PIPELINE (RECOMMENDED)
Sources:
  [1] Barber & Redding (2006) — ZYX rotation, ray-ground intersection
  [15] MDPI Sensors (2014) — Brown-Conrady for GoPro
  UAV_GPS_Localization_Engineering_Report.md — error analysis
  GoPro_Pipeline_Analysis_And_Corrections.md — undistortion fix

Implements:
  - Brown-Conrady undistortion (cv2.undistortPoints, NOT fisheye)
  - Full ZYX attitude rotation with mount compensation
  - Ray-ground intersection
  - Confidence-weighted multi-frame aggregation (N >= 20)
"""

import numpy as np
import math
import cv2


# ─── ROTATION UTILITIES ──────────────────────────────────────────────────────

def Rx(a):
    """Rotation about X (roll)."""
    return np.array([[1, 0, 0],
                     [0, math.cos(a), -math.sin(a)],
                     [0, math.sin(a),  math.cos(a)]])

def Ry(a):
    """Rotation about Y (pitch)."""
    return np.array([[ math.cos(a), 0, math.sin(a)],
                     [0,            1, 0           ],
                     [-math.sin(a), 0, math.cos(a)]])

def Rz(a):
    """Rotation about Z (yaw)."""
    return np.array([[math.cos(a), -math.sin(a), 0],
                     [math.sin(a),  math.cos(a), 0],
                     [0,            0,            1]])

def build_R_total(roll_rad, pitch_rad, yaw_true_rad, R_mount):
    """
    Full rotation: camera frame -> body frame -> NED world frame.
    ZYX Euler convention (Barber & Redding [1]).

    Args:
        roll_rad      : roll in radians (from IMU)
        pitch_rad     : pitch in radians (from IMU)
        yaw_true_rad  : TRUE heading in radians (magnetic + declination)
        R_mount       : 3x3 camera-to-body rotation (determined by mount geometry)

    Returns:
        3x3 combined rotation matrix
    """
    R_body_to_ned = Rz(yaw_true_rad) @ Ry(pitch_rad) @ Rx(roll_rad)
    return R_body_to_ned @ R_mount


# ─── SINGLE-FRAME LOCALIZATION ───────────────────────────────────────────────

def localize_full(u, v, K, dist, h,
                  lat_uav, lon_uav,
                  roll_rad, pitch_rad, yaw_true_rad,
                  R_mount,
                  oblique_threshold=0.1):
    """
    Full single-frame GPS localization for GoPro Hero 5/9 in Linear mode.

    Args:
        u, v              : raw YOLO bbox center pixel (float)
        K                 : 3x3 intrinsic matrix (from cv2.calibrateCamera)
        dist              : distortion coefficients [k1,k2,p1,p2,k3]
        h                 : altitude AGL in meters (use barometer, not GPS altitude)
        lat_uav           : UAV latitude degrees
        lon_uav           : UAV longitude degrees
        roll_rad          : roll  (from IMU, radians)
        pitch_rad         : pitch (from IMU, radians)
        yaw_true_rad      : TRUE heading (magnetic heading + 0.0698 rad for Cairo +4°)
        R_mount           : camera-to-body 3x3 rotation
        oblique_threshold : reject frames where camera ray z-component < this value

    Returns:
        (lat_flag, lon_flag) in degrees, or None if rejected
    """
    # Stage 3: Undistort detection point (Brown-Conrady, NOT fisheye API)
    pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
    pt_corr = cv2.undistortPoints(pt, K, dist, P=K)
    u_c = float(pt_corr[0, 0, 0])
    v_c = float(pt_corr[0, 0, 1])

    # Stage 4: Pixel → normalized camera ray
    x_n = (u_c - K[0, 2]) / K[0, 0]
    y_n = (v_c - K[1, 2]) / K[1, 1]
    r_cam = np.array([x_n, y_n, 1.0])

    # Stage 5: Rotate to world (NED)
    R = build_R_total(roll_rad, pitch_rad, yaw_true_rad, R_mount)
    r_world = R @ r_cam

    # Stage 6: Ray–ground intersection (reject oblique rays)
    if abs(r_world[2]) < oblique_threshold:
        return None
    t  = h / r_world[2]
    dE = t * r_world[0]   # East offset  (meters)
    dN = t * r_world[1]   # North offset (meters)

    # Stage 7: GPS conversion (flat-earth, valid < 10 km)
    dlat = dN / 111320.0
    dlon = dE / (111320.0 * math.cos(math.radians(lat_uav)))

    return lat_uav + dlat, lon_uav + dlon


# ─── CONFIDENCE-WEIGHTED AGGREGATION ─────────────────────────────────────────

def aggregate_approach2(estimates_with_confidence, min_frames=10):
    """
    Confidence-weighted mean aggregation across N frames.
    Source: [9] Real-Time Multi-Target Localization (PMC)

    Args:
        estimates_with_confidence : list of ((lat, lon), confidence) tuples
        min_frames                : reject flag if fewer valid frames

    Returns:
        (lat_final, lon_final) or None
    """
    valid = [(est, c) for est, c in estimates_with_confidence
             if est is not None and c > 0.4]

    if len(valid) < min_frames:
        return None

    total_weight = sum(c for _, c in valid)
    lat_w = sum(est[0] * c for est, c in valid) / total_weight
    lon_w = sum(est[1] * c for est, c in valid) / total_weight

    return lat_w, lon_w


# ─── USAGE EXAMPLE ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    # Load calibrated camera parameters (produced by Person A)
    # Uncomment after calibration is done:
    # with open("camera_params.yaml") as f:
    #     cam = yaml.safe_load(f)
    # K    = np.array(cam["K"])
    # dist = np.array(cam["dist"])

    # For now — pre-calibration Hero 9 estimates:
    K    = np.array([[1432, 0, 960],
                     [0, 1432, 540],
                     [0,    0,   1]], dtype=float)
    dist = np.array([-0.09, 0.06, 0.0, 0.0, 0.0])  # [k1, k2, p1, p2, k3]

    # Camera mount (nadir, image-top = aircraft forward)
    R_mount = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]], dtype=float)

    CAIRO_DECLINATION_RAD = math.radians(4.0)   # +4° East

    # Simulated detections from one flag pass
    # Each entry: (u_px, v_px, lat_uav, lon_uav, h_m,
    #              roll_rad, pitch_rad, mag_heading_rad, confidence)
    pass_data = [
        (960, 545, 30.0613, 31.2497, 79.5, 0.01, -0.02, 0.52, 0.91),
        (962, 543, 30.0612, 31.2498, 80.1, 0.02, -0.01, 0.53, 0.88),
        (958, 547, 30.0614, 31.2496, 79.8, 0.00, -0.03, 0.51, 0.93),
    ]

    results = []
    for u, v, lat, lon, h, roll, pitch, mag_hdg, conf in pass_data:
        true_hdg = mag_hdg + CAIRO_DECLINATION_RAD
        est = localize_full(u, v, K, dist, h,
                            lat, lon,
                            roll, pitch, true_hdg,
                            R_mount)
        results.append((est, conf))
        if est:
            print(f"  Frame → lat: {est[0]:.7f}, lon: {est[1]:.7f} (conf: {conf:.2f})")

    final = aggregate_approach2(results, min_frames=2)  # use 2 for demo; set 10 in production
    if final:
        print(f"\nApproach 2 Final → lat: {final[0]:.7f}, lon: {final[1]:.7f}")