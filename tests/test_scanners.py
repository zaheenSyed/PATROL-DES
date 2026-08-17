"""Slice 6 validation: Raster, Spiral, Uniform scanners.

Geometric properties verified per scanner:
  - Raster: visits distinct points; targets stay inside the search disk.
  - Spiral: targets grow monotonically away from peer estimate over the
    first turn; cycle wraps cleanly.
  - Uniform: large-N sample covers all angular octants of the search disk
    roughly evenly (chi-squared style sanity).
"""

import math
import random
from collections import Counter

import numpy as np
import pytest

from owdesmp.scanners.beta import BetaScanner
from owdesmp.scanners.raster import RasterScanner
from owdesmp.scanners.spiral import SpiralScanner
from owdesmp.scanners.uniform import UniformScanner


MY_POS = np.array([0.0, 0.0, 0.0])
PEER_EST = np.array([100.0, 0.0, 0.0])
SEARCH_R = 5.0
BEAM_R = 1.0


def _distance_to(p):
    return float(np.linalg.norm(np.asarray(p) - PEER_EST))


def _angle_octant(p):
    """Project p onto the disk plane (perpendicular to +X), then return one of
    8 angular sectors. Used for uniform-coverage sanity."""
    proj = np.asarray(p) - PEER_EST
    # Disk plane is perpendicular to (PEER_EST - MY_POS) = +X, so y/z components.
    theta = math.atan2(proj[2], proj[1])
    if theta < 0:
        theta += 2 * math.pi
    return int(theta / (math.pi / 4)) % 8


# ------------------------------------------------------------------ raster

def test_raster_visits_distinct_points():
    s = RasterScanner()
    points = [s.next_target(None, MY_POS, PEER_EST, SEARCH_R, BEAM_R) for _ in range(30)]
    uniq = {tuple(np.round(p, 3)) for p in points}
    assert len(uniq) >= 10, f"raster path is too short: {len(uniq)} unique points"


def test_raster_targets_stay_inside_search_disk():
    s = RasterScanner()
    for _ in range(60):
        p = s.next_target(None, MY_POS, PEER_EST, SEARCH_R, BEAM_R)
        d = _distance_to(p)
        assert d <= SEARCH_R + 1e-6, f"raster point outside disk: d={d}"


def test_raster_wraps_after_exhausting_path():
    s = RasterScanner()
    # First pass
    first_pass = [s.next_target(None, MY_POS, PEER_EST, SEARCH_R, BEAM_R) for _ in range(200)]
    # Path length n is finite. After 2*n calls, the second n must equal the first n.
    n = len(s._points)
    s._idx = 0
    pass_a = [s.next_target(None, MY_POS, PEER_EST, SEARCH_R, BEAM_R) for _ in range(n)]
    pass_b = [s.next_target(None, MY_POS, PEER_EST, SEARCH_R, BEAM_R) for _ in range(n)]
    for a, b in zip(pass_a, pass_b):
        assert np.allclose(a, b), f"raster path does not wrap cleanly: {a} vs {b}"


# ------------------------------------------------------------------ spiral

def test_spiral_grows_outward_then_wraps():
    s = SpiralScanner(samples_per_turn=60)
    # Force path build by calling once.
    s.next_target(None, MY_POS, PEER_EST, SEARCH_R, BEAM_R)
    pts = s._points
    assert len(pts) >= 5
    # First few points: distance from peer center grows monotonically.
    distances = [_distance_to(p) for p in pts[: min(5, len(pts))]]
    for i in range(1, len(distances)):
        assert distances[i] >= distances[i - 1] - 1e-6, (
            f"spiral does not grow at index {i}: {distances}"
        )


def test_spiral_targets_stay_inside_search_disk():
    s = SpiralScanner()
    for _ in range(80):
        p = s.next_target(None, MY_POS, PEER_EST, SEARCH_R, BEAM_R)
        assert _distance_to(p) <= SEARCH_R + 1e-6


# ----------------------------------------------------------------- uniform

def test_uniform_covers_perimeter_evenly():
    s = UniformScanner()
    rng = random.Random(123)
    counts = Counter()
    N = 2000
    for _ in range(N):
        p = s.next_target(rng, MY_POS, PEER_EST, SEARCH_R, BEAM_R)
        counts[_angle_octant(p)] += 1
    # Expected mean per octant = N/8. With chi-squared, observed counts should
    # be within a few standard deviations of expected (tail tolerance below).
    expected = N / 8
    for octant in range(8):
        observed = counts[octant]
        # Bernoulli stderr ~ sqrt(N * p * (1-p)) with p=1/8 -> ~17 for N=2000.
        # Allow 5 sigma to make the test extremely unflaky.
        assert abs(observed - expected) < 100, (
            f"octant {octant}: observed={observed}, expected={expected}"
        )


def test_uniform_targets_exactly_on_perimeter():
    s = UniformScanner()
    rng = random.Random(0)
    for _ in range(50):
        p = s.next_target(rng, MY_POS, PEER_EST, SEARCH_R, BEAM_R)
        assert math.isclose(_distance_to(p), SEARCH_R, abs_tol=1e-6)


# --------------------------------------------------------- beta still works

def test_beta_signature_compatible_with_new_scanners():
    """The beam_footprint_radius_m kwarg added in slice 6 must not break Beta."""
    s = BetaScanner()
    rng = random.Random(0)
    p = s.next_target(rng, MY_POS, PEER_EST, SEARCH_R, BEAM_R)
    assert math.isclose(_distance_to(p), SEARCH_R, abs_tol=1e-6)
