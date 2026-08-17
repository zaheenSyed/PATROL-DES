"""Slice 1 validation: OWDES-MP must reproduce hand_des_4.py's success curve.

For each T_align in a grid, we estimate the discovery success probability by
Monte Carlo using two independent simulators:
  - OWDES-MP (the new event-driven engine under test)
  - legacy/handshake_des/hand_des_4.periodic_trial (the validated oracle)

Both are seeded distinctly per trial. Curves are compared point-by-point with a
2-sigma tolerance computed from binomial sampling noise. Disagreement larger
than that means a timing/state-machine bug, not Monte Carlo variance.
"""

import math
import random
import sys
from copy import deepcopy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "legacy" / "handshake_des"))
import hand_des_4  # noqa: E402

from owdesmp.core.scenario_loader import load_scenario  # noqa: E402
from owdesmp.core.simulator import Simulator  # noqa: E402

# Parameters matched to hand_des_4.py:190-196.
TA = TB = 0.001512
TX = 304e-6
PROC = 100e-6
PACKET_DURATION = TX + PROC  # 404 us
PROCESSING_DELAY = PROC      # 100 us

LEGACY_PARAMS = hand_des_4.PeriodicParams(
    d=100.0, c=3e8, Rb=1e9,
    TA=TA, TB=TB, t_proc=PROC,
    L_SYN=4 * 8, L_SYNACK=4 * 8, L_ACK=4 * 8,
    q_SYN=1.0, q_SYNACK=1.0, q_ACK=1.0,
)

T_GRID = [0.0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020]
TRIALS = 1000


def _base_scenario() -> dict:
    return load_scenario(REPO_ROOT / "scenarios" / "two_node_validation.json")


def _owdes_success_rate(t_align: float, trials: int, base_seed: int) -> float:
    scenario = _base_scenario()
    scenario["simulation"]["duration_s"] = float(t_align)
    successes = 0
    for k in range(trials):
        sim = Simulator(scenario=scenario, run_id=k, seed=base_seed + k)
        if sim.run()["success"]:
            successes += 1
    return successes / trials


def _legacy_success_rate(t_align: float, trials: int, base_seed: int) -> float:
    successes = 0
    for k in range(trials):
        # hand_des_4.periodic_trial reseeds inside when seed is set;
        # we want independent draws per trial, so let it use the global RNG
        # which we reseed deterministically.
        random.seed(base_seed + k)
        params = hand_des_4.PeriodicParams(
            **{**LEGACY_PARAMS.__dict__, "seed": None}
        )
        if hand_des_4.periodic_trial(float(t_align), params):
            successes += 1
    return successes / trials


def _two_sigma(p: float, n: int) -> float:
    """Binomial standard error * 2, with a floor to handle p in {0, 1}."""
    p = min(max(p, 0.0), 1.0)
    se = math.sqrt(max(p * (1 - p), 1.0 / n) / n)
    return 2.0 * se


@pytest.mark.parametrize("t_align", T_GRID)
def test_two_node_success_curve_matches_legacy(t_align: float) -> None:
    p_owdes = _owdes_success_rate(t_align, TRIALS, base_seed=1000)
    p_legacy = _legacy_success_rate(t_align, TRIALS, base_seed=2000)
    # Combined 2-sigma envelope: roughly sqrt(2)*2*se since variances add.
    tol = math.hypot(_two_sigma(p_owdes, TRIALS), _two_sigma(p_legacy, TRIALS))
    assert abs(p_owdes - p_legacy) <= tol + 0.01, (
        f"T_align={t_align:.4f}: OWDES-MP={p_owdes:.3f}, legacy={p_legacy:.3f}, "
        f"diff={abs(p_owdes - p_legacy):.3f}, tol={tol + 0.01:.3f}"
    )


def test_curves_monotonic_and_bounded() -> None:
    """Sanity checks independent of the legacy reference."""
    rates = [_owdes_success_rate(t, 500, base_seed=3000) for t in T_GRID]
    assert all(0.0 <= r <= 1.0 for r in rates)
    # Rough monotonicity: each step should be >= previous - small MC slack
    for prev, cur in zip(rates, rates[1:]):
        assert cur >= prev - 0.05, f"non-monotonic: {prev} -> {cur} (rates={rates})"
    assert rates[0] == 0.0, "T_align=0 must always fail"
    assert rates[-1] > 0.5, f"T_align=0.02 should be mostly successful, got {rates[-1]}"
