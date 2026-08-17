"""Slice 2 validation: geometry layer correctness.

  1. Static-aligned matches the protocol-only distribution.
     With beams locked at each other, the geometry gate is permanently True
     and discovery rates should match slice-1 within Monte Carlo tolerance.

  2. Mispointed beams never discover.
     If A's beam points orthogonal to (A->B) and there's no scanner to move
     it, alignment is permanently False and discovery never happens.

  3. Out-of-range pairs never discover.
     Distance > range_m blocks alignment regardless of pointing.

  4. Beta scanner can discover an uncertain-position peer.
     With a generous beam and small uncertainty region, the Beta walk will
     find alignment within a long-enough duration.

  5. Vector-op invariants (mutual_alignment, slew_time).
"""

import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "legacy" / "handshake_des"))

from owdesmp.core.scenario_loader import load_scenario  # noqa: E402
from owdesmp.core.simulator import Simulator  # noqa: E402
from owdesmp.engines.geometry import (  # noqa: E402
    angle_between,
    mutual_alignment,
    slew_time,
    unit,
)


def _success_rate(scenario: dict, trials: int, base_seed: int, duration: float) -> float:
    sc = deepcopy(scenario)
    sc["simulation"]["duration_s"] = float(duration)
    succ = 0
    for k in range(trials):
        sim = Simulator(scenario=sc, run_id=k, seed=base_seed + k)
        if sim.run()["success"]:
            succ += 1
    return succ / trials


# --------------------------------------------------------------- 1. static aligned

def test_static_aligned_matches_protocol_only_distribution():
    """Geometry layer must be transparent when beams are permanently aligned."""
    proto_only = load_scenario(REPO_ROOT / "scenarios" / "two_node_validation.json")
    aligned = load_scenario(REPO_ROOT / "scenarios" / "two_node_static_aligned.json")

    # Pick a mid-curve duration where the rate is sensitive to bugs (not 0, not 1).
    duration = 0.004
    p_proto = _success_rate(proto_only, trials=600, base_seed=1000, duration=duration)
    p_geom = _success_rate(aligned, trials=600, base_seed=2000, duration=duration)
    # Independent MC variance combined (loose 3-sigma + small fixed slack).
    se = math.sqrt(p_proto * (1 - p_proto) / 600 + p_geom * (1 - p_geom) / 600)
    assert abs(p_proto - p_geom) <= 3 * se + 0.02, (
        f"static-aligned diverges: protocol-only={p_proto:.3f}, geometry={p_geom:.3f}"
    )


# ----------------------------------------------------------------- 2. mispointed

def test_mispointed_pair_never_discovers():
    sc = load_scenario(REPO_ROOT / "scenarios" / "two_node_static_aligned.json")
    # Point both beams along +Y (orthogonal to the A->B vector along +X).
    sc["nodes"][0]["initial_pointing"] = [0.0, 1.0, 0.0]
    sc["nodes"][1]["initial_pointing"] = [0.0, 1.0, 0.0]
    rate = _success_rate(sc, trials=50, base_seed=3000, duration=0.02)
    assert rate == 0.0, f"expected 0%, got {rate}"


# -------------------------------------------------------------- 3. out-of-range

def test_out_of_range_pair_never_discovers():
    sc = load_scenario(REPO_ROOT / "scenarios" / "two_node_static_aligned.json")
    # Move B far beyond A's range_m.
    sc["nodes"][1]["position"]["initial_m"] = [10000.0, 0.0, 0.0]
    rate = _success_rate(sc, trials=50, base_seed=4000, duration=0.02)
    assert rate == 0.0, f"expected 0%, got {rate}"


# ------------------------------------------------------------- 4. beta discovery

# Geometry of the BetaScanner perimeter sampling: scanner targets lie on the
# circle of radius `uncertainty_radius_m` centered at the peer estimate, in the
# plane perpendicular to (searcher -> peer). The peer (if it sits at the
# estimate) is reached by the beam iff the beam footprint radius at distance d
# (= d * tan(beam_half_divergence)) is greater than the perimeter radius.
# So:
#   footprint_r > search_r  =>  beam covers peer for every scanner target
#   footprint_r < search_r  =>  beam never covers peer (perimeter is a wall)


def test_beta_scanner_finds_peer_when_beam_covers_perimeter():
    """5deg divergence at 100m -> footprint ~4.37m > 3m search radius. Discovery succeeds."""
    sc = load_scenario(REPO_ROOT / "scenarios" / "two_node_beta_search.json")
    rate = _success_rate(sc, trials=30, base_seed=5000, duration=0.05)
    assert rate >= 0.7, f"beam wide enough to cover peer but discovery rate={rate}"


def test_beta_scanner_cannot_align_when_beam_narrower_than_perimeter():
    """1deg divergence at 100m -> footprint ~0.87m < 3m search radius.
    After the initial-pointing window (~1ms slew), beam never points at peer."""
    sc = load_scenario(REPO_ROOT / "scenarios" / "two_node_beta_search.json")
    for n in sc["nodes"]:
        n["optical_terminal"]["beam_divergence_deg"] = 1.0
    rate = _success_rate(sc, trials=30, base_seed=6000, duration=0.5)
    assert rate == 0.0, (
        f"narrow beam should not be able to maintain alignment from scanner perimeter; rate={rate}"
    )


# ------------------------------------------------------------- 5. vector invariants

def test_angle_between_basics():
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    assert math.isclose(angle_between(x, x), 0.0, abs_tol=1e-9)
    assert math.isclose(angle_between(x, y), math.pi / 2, abs_tol=1e-9)
    assert math.isclose(angle_between(x, -x), math.pi, abs_tol=1e-9)


def test_slew_time_proportional_to_angle():
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    t_90 = slew_time(x, y, max_rate_rad_s=1.0)
    assert math.isclose(t_90, math.pi / 2, rel_tol=1e-9)
    t_180 = slew_time(x, -x, max_rate_rad_s=1.0)
    assert math.isclose(t_180, math.pi, rel_tol=1e-9)


def test_mutual_alignment_no_geometry_default_true():
    """Nodes without optical_terminal/mobility are treated as always aligned."""
    sc = load_scenario(REPO_ROOT / "scenarios" / "two_node_validation.json")
    sim = Simulator(scenario=sc, run_id=0, seed=0)
    a = sim.nodes["A"]
    b = sim.nodes["B"]
    assert not a.has_geometry
    assert not b.has_geometry
    assert mutual_alignment(a, b, 0.0) is True
