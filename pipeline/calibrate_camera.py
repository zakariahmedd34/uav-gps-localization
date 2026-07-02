"""
Checkerboard camera calibration for the Hero 13.

    python calibrate_camera.py --images "path/to/checkerboard_photos/*.jpg" \
                                --rows 5 --cols 8 --square-size 0.02 \
                                --out configs/camera_params_hero13.yaml

--rows/--cols are INTERNAL corners (not squares). E.g. a 9x6-square board
has 8x5 internal corners.
--square-size is the real-world size of one checkerboard square, in
meters (use whatever unit you like, just be consistent -- it only
affects the scale of translation vectors, not the intrinsics/distortion
you actually care about here).

Usage notes:
- Shoot 15-25 photos of the checkerboard, filling different parts of the
  frame (corners and edges matter most -- that's where distortion is
  strongest), at a few different distances/angles. Keep the board flat
  and well-lit, avoid motion blur.
- Use the SAME camera settings (resolution, FOV/lens mode, e.g.
  Wide/Linear) you'll actually fly with -- intrinsics are tied to that
  exact mode.
- Target: RMS reprojection error < 1.0 px. If it's higher, this script
  will tell you and you should add more/better-distributed images and
  re-run rather than trust the result.
"""

import argparse
import glob
import sys

import cv2
import numpy as np
import yaml

from logger import logging
from exception import CustomException


def calibrate(image_glob, rows, cols, square_size):
    image_paths = sorted(glob.glob(image_glob))
    if not image_paths:
        raise FileNotFoundError(f"No images matched: {image_glob}")

    logging.info("[1] Found %d candidate images.", len(image_paths))

    # 3D object points for one checkerboard view, in board coordinates
    # (z=0 plane), scaled by square_size.
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []  # 3D points, one array per accepted image
    imgpoints = []  # corresponding 2D corner points
    image_size = None
    used, skipped = 0, 0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            logging.warning("    [skip] could not read: %s", path)
            skipped += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)
        elif gray.shape[::-1] != image_size:
            logging.warning("    [skip] inconsistent resolution: %s", path)
            skipped += 1
            continue

        found, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found:
            logging.warning("    [skip] no checkerboard found: %s", path)
            skipped += 1
            continue

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        used += 1

    logging.info("[2] Used %d images, skipped %d.", used, skipped)

    if used < 10:
        logging.warning("Only %d usable images -- calibration may be unstable. "
                        "Aim for 15-25 well-distributed shots.", used)

    if used == 0:
        raise RuntimeError("No usable checkerboard detections -- cannot calibrate.")

    logging.info("[3] Running cv2.calibrateCamera...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    return {
        "rms": rms,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "image_size": image_size,
        "n_images_used": used,
        "n_images_skipped": skipped,
    }


def save_yaml(result, out_path, rows, cols, square_size, camera_label="hero13"):
    cm = result["camera_matrix"]
    dc = result["dist_coeffs"].flatten()

    data = {
        "camera": camera_label,
        "image_width": int(result["image_size"][0]),
        "image_height": int(result["image_size"][1]),
        "rms_reprojection_error_px": float(result["rms"]),
        "n_images_used": result["n_images_used"],
        "n_images_skipped": result["n_images_skipped"],
        "checkerboard_internal_corners": {"rows": rows, "cols": cols},
        "square_size_m": square_size,
        "camera_matrix": {
            "fx": float(cm[0, 0]),
            "fy": float(cm[1, 1]),
            "cx": float(cm[0, 2]),
            "cy": float(cm[1, 2]),
        },
        # OpenCV's standard plumb-bob model: k1,k2,p1,p2,k3[,k4,k5,k6]
        "dist_coeffs": [float(v) for v in dc],
    }

    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    logging.info("[DONE] Saved: %s", out_path)
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True, help="Glob pattern, e.g. 'calib_photos/*.jpg'")
    ap.add_argument("--rows", type=int, required=True, help="Internal corner rows")
    ap.add_argument("--cols", type=int, required=True, help="Internal corner cols")
    ap.add_argument("--square-size", type=float, required=True, help="Square size (meters)")
    ap.add_argument("--out", default="configs/camera_params_hero13.yaml")
    args = ap.parse_args()
    logging.info("calibrate_camera: START images=%s rows=%d cols=%d",
                 args.images, args.rows, args.cols)

    result = calibrate(args.images, args.rows, args.cols, args.square_size)

    logging.info("--- RMS reprojection error: %.4f px ---", result["rms"])
    if result["rms"] >= 1.0:
        logging.warning("RMS >= 1.0 px target NOT met. Recommend retaking images "
                        "(more views, better corner/edge coverage, sharper focus, "
                        "less motion blur) and re-running before trusting these intrinsics.")
    else:
        logging.info("[OK] RMS target met (< 1.0 px).")

    save_yaml(result, args.out, args.rows, args.cols, args.square_size)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise CustomException(e, sys) from e
