"""
Smoke tests for the localization geometry (no video/telemetry needed).

Run:  pytest tests/ -v       (from the repo root)

These pin down the parts that are easy to get sign-flipped: the gravity->world
rotation and the ground projection. If these pass, the math in localization.py
is trustworthy and any bad output is an INPUT problem (calibration, h, heading).
"""

import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from localization import (R_grav_heading, EARTH, off_nadir_deg,   # noqa: E402
                          dbscan, geometric_median)


def project(grav, heading_deg, u, v, K, h, lat0=30.0, lon0=31.0):
    """Mini end-to-end: pixel -> world ray -> ground -> GPS (no distortion)."""
    xn = (u - K[0, 2]) / K[0, 0]
    yn = (v - K[1, 2]) / K[1, 1]
    R = R_grav_heading(grav, heading_deg)
    rw = R @ np.array([xn, yn, 1.0])
    t = h / rw[2]
    dE, dN = t * rw[0], t * rw[1]
    lat = lat0 + dN / EARTH
    lon = lon0 + dE / (EARTH * math.cos(math.radians(lat0)))
    return lat, lon, dE, dN


K = np.array([[1500.0, 0, 960.0], [0, 1500.0, 540.0], [0, 0, 1.0]])


def test_nadir_gravity_is_identity_tilt():
    """Gravity straight down (0,0,1) + heading 0 -> level rotation = identity."""
    R = R_grav_heading([0, 0, 1], 0.0)
    assert np.allclose(R, np.eye(3), atol=1e-9)


def test_center_pixel_is_below_drone():
    """Flag at image center, nadir -> directly under the drone (zero offset)."""
    lat, lon, dE, dN = project([0, 0, 1], 0.0, 960, 540, K, h=80)
    assert abs(dE) < 1e-6 and abs(dN) < 1e-6


def test_altitude_scales_offset_linearly():
    """Doubling height doubles the ground offset for the same pixel."""
    _, _, dE80, _ = project([0, 0, 1], 0.0, 1100, 540, K, h=80)
    _, _, dE160, _ = project([0, 0, 1], 0.0, 1100, 540, K, h=160)
    assert abs(dE160 / dE80 - 2.0) < 1e-6


def test_offset_is_bounded_at_known_altitude():
    """A pixel near center at 80 m can't put the flag kilometres away."""
    _, _, dE, dN = project([0, 0, 1], 0.0, 1200, 700, K, h=80)
    assert math.hypot(dE, dN) < 200


def test_gravity_normalization_is_robust():
    """Non-unit gravity vector still yields a proper rotation (det=1)."""
    R = R_grav_heading([0.0, 0.2, 5.0], 30.0)   # not unit length
    assert abs(np.linalg.det(R) - 1.0) < 1e-9


# ---------------------------------------------------------------- V2 fixes ---

def test_off_nadir_angle():
    """Straight-down ray = 0 deg; 45-deg ray = 45 deg; upward ray > 90."""
    assert abs(off_nadir_deg(np.array([0, 0, 1.0]))) < 1e-9
    assert abs(off_nadir_deg(np.array([1.0, 0, 1.0])) - 45.0) < 1e-6
    assert off_nadir_deg(np.array([0, 0, -1.0])) > 90


def test_dbscan_two_blobs_plus_noise():
    """Two 30m-apart blobs + one far outlier -> 2 clusters, outlier = noise."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 2, (20, 2))
    b = np.column_stack([rng.normal(30, 2, 20), rng.normal(0, 2, 20)])
    noise = np.array([[500.0, 500.0]])
    P = np.vstack([a, b, noise])
    labels = dbscan(P, eps=8.0, min_samples=4)
    assert len({l for l in labels if l != -1}) == 2
    assert labels[-1] == -1                      # the outlier is noise
    assert (labels[:20] == labels[0]).all()      # blob A holds together
    assert (labels[20:40] == labels[20]).all()   # blob B holds together


def test_dbscan_does_not_fragment_elongated_scatter():
    """A 60m-long chain of points (the greedy-clustering failure mode) must
    come out as ONE cluster, not a string of eps-sized balls."""
    chain = np.column_stack([np.linspace(0, 60, 40), np.zeros(40)])
    labels = dbscan(chain, eps=8.0, min_samples=3)
    assert len({l for l in labels if l != -1}) == 1


def test_geometric_median_robust_to_outlier():
    """Median of a tight blob + one 500m outlier stays inside the blob;
    the (old) mean would be dragged ~10m away."""
    P = np.vstack([np.zeros((50, 2)), [[500.0, 0.0]]])
    m = geometric_median(P)
    assert np.linalg.norm(m) < 1.0
    assert np.linalg.norm(P.mean(axis=0)) > 5.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("\nAll geometry tests passed.")
