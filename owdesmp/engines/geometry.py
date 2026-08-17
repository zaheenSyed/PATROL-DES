"""Geometry engine.

Provides:
  - vector ops (unit, angle_between, direction_to)
  - slew_time(from_dir, to_dir, max_rate)
  - mutual_alignment(node_a, node_b, t): True iff both nodes' beams point at
    each other within their beam half-divergence AND distance is within range.
  - propagation_delay(node_a, node_b, t): one-way speed-of-light delay.

The simulator maintains an `alignment_active` table keyed by frozenset({a, b}).
After any node's pointing or position changes (BEAM_MOVE_END,
NODE_MOBILITY_UPDATE), the simulator recomputes mutual_alignment for affected
pairs and fires ALIGNMENT_START / ALIGNMENT_END on flips.
"""

import math

import numpy as np


# Speed of light in vacuum (m/s). Used by the simulator to compute one-way
# propagation delay between TX and RX times. Negligible at lab/UAV ranges
# (3.3 ns/m) but ~3.3 ms one-way at 1000 km satellite ISL distances.
SPEED_OF_LIGHT_M_S = 299_792_458.0


def lerp_unit(d_0: np.ndarray, d_1: np.ndarray, frac: float) -> np.ndarray:
    """Linear interpolation between two unit vectors, renormalized.

    Matches the legacy fso_sim_modules.py pointing-update rule. For the small
    inter-target angular steps a steered FSO terminal actually traverses this
    is indistinguishable from spherical-linear interpolation but cheaper. The
    renormalize keeps the result on the unit sphere even when |d_0 - d_1| is
    not tiny.
    """
    v = (1.0 - frac) * d_0 + frac * d_1
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return d_0.copy()
    return v / n


def propagation_delay_s(node_a, node_b, t: float) -> float:
    """One-way light-time between two geometry-enabled nodes at time t.

    Returns 0.0 if either node has no mobility model (i.e. geometry is
    disabled for the pair - matches slice-1 protocol-only semantics).
    """
    if node_a.mobility_model is None or node_b.mobility_model is None:
        return 0.0
    pa = node_a.position_at(t)
    pb = node_b.position_at(t)
    dist = float(np.linalg.norm(pb - pa))
    return dist / SPEED_OF_LIGHT_M_S


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns angle in radians between v1 and v2, in [0, pi]."""
    u1 = unit(v1)
    u2 = unit(v2)
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return math.acos(dot)


def direction_to(from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
    return unit(to_pos - from_pos)


def slew_time(from_dir: np.ndarray, to_dir: np.ndarray, max_rate_rad_s: float) -> float:
    """Seconds to slew between two beam directions at max angular rate."""
    if max_rate_rad_s <= 0.0:
        return 0.0
    return angle_between(from_dir, to_dir) / max_rate_rad_s


def mutual_alignment(node_a, node_b, t: float) -> bool:
    """True iff A's beam points at B AND B's beam points at A AND both within range."""
    if not (node_a.has_geometry and node_b.has_geometry):
        return True  # fallback: any geometry-disabled pair is always "aligned"
    if node_a.current_pointing is None or node_b.current_pointing is None:
        return False

    pa = node_a.position_at(t)
    pb = node_b.position_at(t)
    dist = float(np.linalg.norm(pb - pa))
    if dist == 0.0:
        return True

    range_a = float(node_a.optical_terminal.get("range_m", float("inf")))
    range_b = float(node_b.optical_terminal.get("range_m", float("inf")))
    if dist > min(range_a, range_b):
        return False

    half_div_a = 0.5 * float(node_a.optical_terminal["beam_divergence_rad"])
    half_div_b = 0.5 * float(node_b.optical_terminal["beam_divergence_rad"])

    # A's pointing vs (A -> B)
    a_to_b = (pb - pa) / dist
    err_a = angle_between(node_a.current_pointing, a_to_b)
    if err_a > half_div_a:
        return False

    # B's pointing vs (B -> A)
    b_to_a = -a_to_b
    err_b = angle_between(node_b.current_pointing, b_to_a)
    if err_b > half_div_b:
        return False

    return True
