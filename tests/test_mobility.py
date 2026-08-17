"""Slice 6 validation: Waypoint + CircularOrbit mobility models."""

import math

import numpy as np
import pytest

from owdesmp.mobility.circular_orbit import CircularOrbit
from owdesmp.mobility.waypoint import Waypoint


# ------------------------------------------------------------------ waypoint

def test_waypoint_lerps_between_anchors():
    wp = Waypoint([
        {"t_s": 0.0, "pos_m": [0.0, 0.0, 0.0]},
        {"t_s": 1.0, "pos_m": [10.0, 0.0, 0.0]},
    ])
    assert np.allclose(wp.position_at(0.0), [0.0, 0.0, 0.0])
    assert np.allclose(wp.position_at(0.5), [5.0, 0.0, 0.0])
    assert np.allclose(wp.position_at(1.0), [10.0, 0.0, 0.0])


def test_waypoint_clamps_before_first_and_after_last():
    wp = Waypoint([
        {"t_s": 1.0, "pos_m": [0.0, 0.0, 0.0]},
        {"t_s": 2.0, "pos_m": [10.0, 0.0, 0.0]},
    ])
    assert np.allclose(wp.position_at(0.0), [0.0, 0.0, 0.0])
    assert np.allclose(wp.position_at(5.0), [10.0, 0.0, 0.0])
    # Velocity zero outside the trajectory.
    assert np.allclose(wp.velocity_at(0.0), np.zeros(3))
    assert np.allclose(wp.velocity_at(5.0), np.zeros(3))


def test_waypoint_velocity_matches_segment_secant():
    wp = Waypoint([
        {"t_s": 0.0, "pos_m": [0.0, 0.0, 0.0]},
        {"t_s": 2.0, "pos_m": [10.0, 4.0, 2.0]},
        {"t_s": 5.0, "pos_m": [16.0, 4.0, 2.0]},
    ])
    # First segment: (10,4,2)/2s = (5,2,1) m/s
    assert np.allclose(wp.velocity_at(1.0), [5.0, 2.0, 1.0])
    # Second segment: (6,0,0)/3s = (2,0,0) m/s
    assert np.allclose(wp.velocity_at(3.0), [2.0, 0.0, 0.0])


def test_waypoint_rejects_unsorted_anchors():
    with pytest.raises(ValueError):
        Waypoint([
            {"t_s": 2.0, "pos_m": [0.0, 0.0, 0.0]},
            {"t_s": 1.0, "pos_m": [0.0, 0.0, 0.0]},
        ])


# --------------------------------------------------------------- circular orbit

def test_circular_orbit_period_returns_to_start():
    orb = CircularOrbit(center_m=[0, 0, 0], radius_m=1000.0, period_s=10.0)
    p0 = orb.position_at(0.0)
    pT = orb.position_at(10.0)
    assert np.allclose(p0, pT, atol=1e-6)


def test_circular_orbit_half_period_diametrically_opposite():
    orb = CircularOrbit(center_m=[0, 0, 0], radius_m=1000.0, period_s=10.0)
    p0 = orb.position_at(0.0)
    p_half = orb.position_at(5.0)
    assert np.allclose(p0 + p_half, np.zeros(3), atol=1e-6), (
        f"after T/2 the orbiter should be diametrically opposite: p0={p0}, "
        f"p_half={p_half}"
    )


def test_circular_orbit_distance_constant():
    orb = CircularOrbit(center_m=[100, 200, 300], radius_m=50.0, period_s=2.0)
    for t in [0.0, 0.3, 0.7, 1.1, 1.9]:
        d = float(np.linalg.norm(orb.position_at(t) - np.array([100, 200, 300])))
        assert math.isclose(d, 50.0, abs_tol=1e-6)


def test_circular_orbit_speed_matches_r_omega():
    R = 1000.0
    T = 10.0
    orb = CircularOrbit(center_m=[0, 0, 0], radius_m=R, period_s=T)
    omega = 2 * math.pi / T
    for t in [0.0, 1.7, 4.3, 9.9]:
        v = orb.velocity_at(t)
        assert math.isclose(float(np.linalg.norm(v)), R * omega, rel_tol=1e-6)


def test_circular_orbit_velocity_orthogonal_to_radius_vector():
    orb = CircularOrbit(center_m=[10, 20, 30], radius_m=5.0, period_s=4.0)
    for t in [0.0, 0.5, 1.3, 3.7]:
        r_vec = orb.position_at(t) - np.array([10, 20, 30])
        v_vec = orb.velocity_at(t)
        # v . r = 0 for circular motion
        assert abs(float(np.dot(r_vec, v_vec))) < 1e-6


def test_circular_orbit_in_plane_normal_to_axis():
    """All positions lie in the plane perpendicular to plane_normal through center."""
    normal = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
    center = np.array([5.0, -3.0, 2.0])
    orb = CircularOrbit(
        center_m=center.tolist(),
        radius_m=10.0,
        period_s=4.0,
        plane_normal=normal.tolist(),
    )
    for t in [0.0, 0.5, 1.0, 2.5, 3.8]:
        offset = orb.position_at(t) - center
        # offset . normal should be ~0 (in-plane).
        assert abs(float(np.dot(offset, normal))) < 1e-6
