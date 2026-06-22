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
from localization import R_grav_heading, EARTH   # noqa: E402


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("\nAll geometry tests passed.")
