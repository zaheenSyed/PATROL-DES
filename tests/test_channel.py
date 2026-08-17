"""Slice 3 validation: channel engine correctness.

  1. channel.mode = "disabled" leaves slice-2 behavior unchanged.
  2. Threshold mode with comfortable link margin still discovers reliably.
  3. Threshold mode with a hard sensitivity threshold blocks discovery
     (verifies the channel gate is actually applied, not bypassed).
  4. Pure 1/r^2 geometric falloff: received_power_dBm at 2d is
     ~6 dB below received_power_dBm at d (when other losses are negligible).
  5. Stochastic mode introduces run-to-run variance in discovery time
     even at parameters where threshold mode is deterministic-pass.
  6. packets_dropped_channel counter increments on weak-link runs.
"""

import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from owdesmp.core.scenario_loader import load_scenario  # noqa: E402
from owdesmp.core.simulator import Simulator  # noqa: E402
from owdesmp.engines.channel import ChannelModel  # noqa: E402


def _success_rate(scenario, trials, base_seed):
    succ = 0
    for k in range(trials):
        sim = Simulator(scenario=scenario, run_id=k, seed=base_seed + k)
        if sim.run()["success"]:
            succ += 1
    return succ / trials


def _channel_strong():
    return load_scenario(REPO_ROOT / "scenarios" / "two_node_channel.json")


# --------------------------------------------------------------- 1. disabled mode

def test_disabled_mode_matches_slice2_static_aligned():
    """When channel.mode = disabled (or section absent), slice-2 static-aligned
    success rate is preserved."""
    sc_no_channel = load_scenario(REPO_ROOT / "scenarios" / "two_node_static_aligned.json")
    sc_disabled = deepcopy(sc_no_channel)
    sc_disabled["channel"] = {"mode": "disabled"}

    p_no = _success_rate(sc_no_channel, trials=200, base_seed=1000)
    p_dis = _success_rate(sc_disabled, trials=200, base_seed=1000)
    assert p_no == p_dis, f"disabled mode changed rate: {p_no} vs {p_dis}"


# --------------------------------------------------------------- 2. strong link

def test_threshold_mode_strong_link_discovers():
    """At 100m with 20 dBm tx and -35 dBm sensitivity, link margin is ~20 dB."""
    sc = _channel_strong()
    rate = _success_rate(sc, trials=200, base_seed=2000)
    assert rate >= 0.95, f"strong link should discover reliably; got {rate}"


# --------------------------------------------------------------- 3. weak link

def test_threshold_mode_weak_link_blocks_discovery():
    """Move B to 5000m so geometric loss outpaces tx power, putting P_rx below sensitivity."""
    sc = _channel_strong()
    sc["nodes"][1]["position"]["initial_m"] = [5000.0, 0.0, 0.0]
    # Range must be raised so the geometry gate doesn't pre-empt the channel gate.
    for n in sc["nodes"]:
        n["optical_terminal"]["range_m"] = 1e7
    rate = _success_rate(sc, trials=50, base_seed=3000)
    assert rate == 0.0, f"weak link must not discover; got {rate}"


# --------------------------------------------------------------- 4. 1/r^2 falloff

def test_received_power_inverse_square_falloff():
    """With atmospheric loss ~0 and aligned beams, doubling distance loses 6 dB."""
    sc = _channel_strong()
    # Zero atmospheric attenuation so geometric spreading dominates.
    sc["channel"]["sigma_dBkm"] = 0.0
    sim = Simulator(scenario=sc, run_id=0, seed=0)
    a = sim.nodes["A"]
    b = sim.nodes["B"]
    # B is at 100m. Measure power at 100m, then move B to 200m.
    p_100 = sim.channel.received_power_dBm(a, b, 0.0)

    sc2 = deepcopy(sc)
    sc2["nodes"][1]["position"]["initial_m"] = [200.0, 0.0, 0.0]
    sim2 = Simulator(scenario=sc2, run_id=0, seed=0)
    p_200 = sim2.channel.received_power_dBm(sim2.nodes["A"], sim2.nodes["B"], 0.0)
    diff = p_100 - p_200
    assert math.isclose(diff, 6.0206, abs_tol=0.05), (
        f"expected ~6 dB falloff doubling distance, got {diff:.3f} dB"
    )


# --------------------------------------------------------------- 5. stochastic mode

def test_stochastic_mode_introduces_variance():
    """Same starting state but different turbulence draws should give different
    discovery times. We use a Cn2 large enough that the Gamma-Gamma sample
    has a visible standard deviation in dB."""
    sc = _channel_strong()
    sc["channel"]["mode"] = "stochastic"
    sc["channel"]["Cn2"] = 1e-13  # strong turbulence so fades are noticeable
    # Tighten the margin so fades have a chance to bring P_rx below threshold.
    for n in sc["nodes"]:
        n["optical_terminal"]["receiver_sensitivity_dBm"] = -16.0  # link margin ~1.5 dB

    drops_by_run = []
    for k in range(30):
        sim = Simulator(scenario=sc, run_id=k, seed=5000 + k)
        r = sim.run()
        drops_by_run.append(r["packets_dropped_channel"])
    # Some runs must see channel drops (turbulence-induced fades).
    assert max(drops_by_run) > 0, (
        f"stochastic mode should produce at least one channel drop across 30 runs; got {drops_by_run}"
    )


# --------------------------------------------------------------- 6. drop counter

def test_drop_counter_increments_on_weak_link():
    sc = _channel_strong()
    sc["nodes"][1]["position"]["initial_m"] = [5000.0, 0.0, 0.0]
    for n in sc["nodes"]:
        n["optical_terminal"]["range_m"] = 1e7
    sim = Simulator(scenario=sc, run_id=0, seed=42)
    r = sim.run()
    assert r["packets_dropped_channel"] > 0
    assert r["packets_dropped_alignment"] == 0  # alignment was OK, channel rejected
    assert r["success"] is False


# --------------------------------------------------------------- 7. ChannelModel unit

def test_kim_attenuation_reasonable_range():
    """At 20 km visibility / 1550 nm, expect a small but non-zero dB/km."""
    from owdesmp.engines.channel import kim_attenuation_dBkm
    s = kim_attenuation_dBkm(20.0, 1550.0)
    assert 0.0 < s < 5.0


def test_channel_disabled_returns_true_without_optical_terminal():
    """packet_succeeds in disabled mode should return True even if terminals lack channel fields."""
    ch = ChannelModel(mode="disabled")
    class _N:
        optical_terminal = None
        current_pointing = None
        def position_at(self, t):
            return np.zeros(3)
    assert ch.packet_succeeds(_N(), _N(), 0.0) is True
