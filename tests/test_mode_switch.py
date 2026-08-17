"""ATP dual-beam mode-switch (Phase R0) tests."""

from copy import deepcopy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from owdesmp.core.scenario_loader import load_scenario  # noqa: E402
from owdesmp.core.simulator import Simulator  # noqa: E402


def _base():
    """Static-aligned two-node baseline; opt in to maintenance by editing."""
    return load_scenario(REPO_ROOT / "scenarios" / "two_node_static_aligned.json")


def _add_maintenance(sc, narrow_div_deg=0.5, settling_s=0.02):
    """Add a maintenance terminal that's narrower than the acquisition beam
    to every node, plus the simulator-level settling time. Also extends the
    sim duration AND switches to fixed_duration termination so the sim runs
    long enough for the BEAM_MODE_SWITCH event (scheduled settling_s after
    LINK_ESTABLISHED) to actually fire."""
    sc["simulation"]["mode_switch_settling_s"] = settling_s
    sc["simulation"]["duration_s"] = max(
        float(sc["simulation"].get("duration_s", 0.0)),
        0.01 + 2 * settling_s,
    )
    sc["termination_condition"] = {"type": "fixed_duration"}
    for n in sc["nodes"]:
        wide = dict(n["optical_terminal"])
        narrow = dict(wide)
        narrow["beam_divergence_deg"] = narrow_div_deg
        n["optical_terminal_maintenance"] = narrow
    return sc


def test_no_maintenance_config_is_a_no_op():
    """Scenarios without optical_terminal_maintenance behave identically."""
    sc = _base()
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    r = sim.run()
    assert r["mode_switches"] == 0
    assert r["packets_dropped_mode_switch"] == 0
    # Both nodes still in acquisition mode (== False because never switched).
    for n in sim.nodes.values():
        assert n.in_maintenance_mode is False


def test_mode_switches_fire_after_link_established():
    sc = _add_maintenance(_base())
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    r = sim.run()
    # Both endpoints should have switched (LINK_ESTABLISHED fires once but
    # the scheduler hands each endpoint its own BEAM_MODE_SWITCH).
    assert r["mode_switches"] == 2
    for n in sim.nodes.values():
        assert n.in_maintenance_mode is True


def test_switch_time_respects_settling():
    """The active optical_terminal should still be the acquisition config
    inside the settling window, and the maintenance config after."""
    settling = 0.01
    sc = _add_maintenance(_base(), settling_s=settling)
    # Force the sim to stop well before settling completes by reading
    # mode_switch_active_until after first_link_time.
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    r = sim.run()
    first_link = r["first_link_time"]
    assert first_link is not None
    # Both endpoints had switched-active windows that ran from first_link to
    # first_link + settling. After the sim ends (long after), in_maintenance_mode
    # is True. The window itself just blocked TX during settling.
    for n in sim.nodes.values():
        assert n.mode_switch_active_until == 0.0  # cleared on switch
        assert n.in_maintenance_mode is True


def test_packets_during_settling_are_dropped():
    """With a 100 ms settling window and keepalives every 1 ms, ~100 keepalive
    TXs fall inside the window and must drop while the optics are switching."""
    sc = _add_maintenance(_base(), settling_s=0.1)
    sc["simulation"]["duration_s"] = 0.2
    # Add keepalives so there's actually TX activity during the settling
    # window (the bare static-aligned scenario has linked_pairs silencing
    # beacons and no keepalive_period_s, so nothing tries to TX).
    for n in sc["nodes"]:
        n["protocol"]["keepalive_period_s"] = 0.001
        n["protocol"]["link_timeout_s"] = 0.5  # don't let the test trigger LINK_LOST
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    r = sim.run()
    assert r["packets_dropped_mode_switch"] > 0, (
        f"expected drops during the 100 ms settling window; got {r}"
    )


def test_narrow_beam_after_switch_is_observable():
    """Post-switch, the node's effective half-divergence must be the
    maintenance config's value, not the original wide value."""
    sc = _add_maintenance(_base(), narrow_div_deg=0.5)
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    sim.run()
    for n in sim.nodes.values():
        assert n.in_maintenance_mode is True
        assert n.optical_terminal["beam_divergence_deg"] == 0.5


def test_revert_to_acquisition_on_link_loss_off_by_default():
    """Without the opt-in flag, LINK_LOST does NOT swap the mode back."""
    sc = _add_maintenance(_base(), settling_s=0.005)
    sc["simulation"]["duration_s"] = 0.5
    # Drive a LINK_LOST by adding a drifting peer + keepalive timeout.
    sc["nodes"][1]["position"] = {
        "model": "constant_velocity",
        "initial_m": [100.0, 0.0, 0.0],
        "velocity_mps": [0.0, 150.0, 0.0],
    }
    sc["simulation"]["mobility_update_period_s"] = 0.001
    sc["nodes"][0]["protocol"]["keepalive_period_s"] = 0.001
    sc["nodes"][1]["protocol"]["keepalive_period_s"] = 0.001
    sc["nodes"][0]["protocol"]["link_timeout_s"] = 0.005
    sc["nodes"][1]["protocol"]["link_timeout_s"] = 0.005
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    r = sim.run()
    assert r["links_lost"] >= 1, "scenario didn't produce a LINK_LOST"
    # No flag means: once in maintenance, stays in maintenance even after loss.
    for n in sim.nodes.values():
        assert n.in_maintenance_mode is True


def test_revert_to_acquisition_on_link_loss_swaps_back():
    """With the opt-in flag set, LINK_LOST schedules a swap back to acquisition.
    After settling, both nodes are out of maintenance mode."""
    sc = _add_maintenance(_base(), settling_s=0.005)
    sc["simulation"]["duration_s"] = 0.5
    sc["simulation"]["revert_to_acquisition_on_link_loss"] = True
    sc["nodes"][1]["position"] = {
        "model": "constant_velocity",
        "initial_m": [100.0, 0.0, 0.0],
        "velocity_mps": [0.0, 150.0, 0.0],
    }
    sc["simulation"]["mobility_update_period_s"] = 0.001
    sc["nodes"][0]["protocol"]["keepalive_period_s"] = 0.001
    sc["nodes"][1]["protocol"]["keepalive_period_s"] = 0.001
    sc["nodes"][0]["protocol"]["link_timeout_s"] = 0.005
    sc["nodes"][1]["protocol"]["link_timeout_s"] = 0.005
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    r = sim.run()
    assert r["links_lost"] >= 1
    # First switch was forward (LINK_ESTABLISHED), then revert (LINK_LOST).
    # 4 switches expected at minimum: 2 forward + 2 revert.
    assert r["mode_switches"] >= 4, (
        f"expected >=4 mode switches (2 forward + 2 revert); got {r['mode_switches']}"
    )
    # Both should be back in acquisition by sim end.
    for n in sim.nodes.values():
        assert n.in_maintenance_mode is False, (
            f"{n.node_id} did not revert; in_maintenance_mode={n.in_maintenance_mode}"
        )
        assert n.optical_terminal["beam_divergence_deg"] == 5.0  # the wide setting from _base()
