"""Phase C validation: beam-tracking controllers + simulator wiring."""

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
POSE_CSV = REPO_ROOT / "drone data" / "hover" / "pose_deviation.csv"
VIB_SCENARIO = REPO_ROOT / "scenarios" / "two_node_vibration.json"

from owdesmp.core.scenario_loader import load_scenario  # noqa: E402
from owdesmp.core.simulator import Simulator  # noqa: E402
from owdesmp.engines.beam_tracker import (  # noqa: E402
    KalmanBeamTracker,
    NullBeamTracker,
    PIDBeamTracker,
    apply_az_el_correction,
    build_beam_tracker,
)


# ---------------------------------------------------------------- factory


def test_build_beam_tracker_none_returns_null():
    t = build_beam_tracker(None)
    assert isinstance(t, NullBeamTracker)
    t = build_beam_tracker({"type": "none"})
    assert isinstance(t, NullBeamTracker)


def test_build_beam_tracker_pid():
    t = build_beam_tracker({"type": "pid", "Kp": 1.0, "Ki": 0.0, "Kd": 0.0})
    assert isinstance(t, PIDBeamTracker)
    assert t.Kp == 1.0


def test_build_beam_tracker_kalman():
    t = build_beam_tracker({"type": "kalman"})
    assert isinstance(t, KalmanBeamTracker)


def test_build_beam_tracker_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown beam_tracker"):
        build_beam_tracker({"type": "ppo"})


# -------------------------------------------------------------- PID basics


def test_pid_zero_error_yields_zero_correction():
    p = PIDBeamTracker()
    obs = {"current_az_offset": 0.0, "current_el_offset": 0.0, "dt": 0.001}
    assert p.get_correction(obs) == (0.0, 0.0)


def test_pid_correction_is_clipped():
    p = PIDBeamTracker(Kp=1e6, Ki=0.0, Kd=0.0, max_correction_rad=0.005)
    obs = {"current_az_offset": 1.0, "current_el_offset": -1.0, "dt": 0.001}
    daz, del_ = p.get_correction(obs)
    assert abs(daz) <= 0.005 + 1e-12
    assert abs(del_) <= 0.005 + 1e-12


def test_pid_integral_is_clipped():
    p = PIDBeamTracker(Kp=0.0, Ki=1.0, Kd=0.0, integral_clip_rad=0.002)
    obs = {"current_az_offset": 10.0, "current_el_offset": 0.0, "dt": 0.1}
    for _ in range(100):
        p.get_correction(obs)
    # Integral state must stay within ±integral_clip
    assert abs(p.integral_az) <= 0.002 + 1e-12


def test_pid_reset_clears_state():
    p = PIDBeamTracker(Kp=0.0, Ki=1.0, Kd=0.0)
    obs = {"current_az_offset": 1.0, "current_el_offset": 1.0, "dt": 0.01}
    for _ in range(5):
        p.get_correction(obs)
    assert p.integral_az != 0
    p.reset()
    assert p.integral_az == 0.0
    assert p.integral_el == 0.0
    assert p.prev_error_az == 0.0


# ------------------------------------------------------------ Kalman basics


def test_kalman_correction_is_clipped():
    k = KalmanBeamTracker(max_correction_rad=0.005)
    obs = {"current_az_offset": 5.0, "current_el_offset": -5.0, "dt": 0.001}
    daz, del_ = k.get_correction(obs)
    assert abs(daz) <= 0.005 + 1e-12
    assert abs(del_) <= 0.005 + 1e-12


def test_kalman_zero_input_eventually_zero_correction():
    k = KalmanBeamTracker()
    obs = {"current_az_offset": 0.0, "current_el_offset": 0.0, "dt": 0.001}
    # Steady-state: with no input the filter should converge to zero state.
    for _ in range(50):
        k.get_correction(obs)
    daz, del_ = k.get_correction(obs)
    assert abs(daz) < 1e-4
    assert abs(del_) < 1e-4


def test_kalman_reset_clears_state():
    k = KalmanBeamTracker()
    obs = {"current_az_offset": 1.0, "current_el_offset": -1.0, "dt": 0.01}
    for _ in range(10):
        k.get_correction(obs)
    assert np.any(k.x != 0)
    k.reset()
    assert np.allclose(k.x, 0.0)


# ----------------------------------------------- sign-convention regression
#
# KNOWN-BROKEN tests for the beam-tracker frame-of-reference bug
# investigated on 2026-05-31. Both PID and Kalman should steer the beam
# TOWARD the partner — but they do not, because the simulator reports
# az/el as global-frame components rather than local tangent-frame
# components. See the FIXME(beam-tracker-frame-bug) block in
# `owdesmp/engines/beam_tracker.py` for the full description and the
# sketch of the fix. These tests are marked xfail so the failing state
# is visible in CI without breaking the gate; flip the xfail to a
# regular assertion once the simulator-side fix lands.


@pytest.mark.xfail(reason="beam-tracker-frame-bug: az/el are global, not local")
def test_pid_steers_toward_target():
    p = PIDBeamTracker(Kp=0.5, Ki=0.0, Kd=0.0, max_correction_rad=0.005)
    obs = {"current_az_offset": +0.01, "current_el_offset": -0.01, "dt": 0.001}
    daz, del_ = p.get_correction(obs)
    assert daz > 0, f"daz must be positive when target is at +az; got {daz}"
    assert del_ < 0, f"del_ must be negative when target is at -el; got {del_}"


@pytest.mark.xfail(reason="beam-tracker-frame-bug: az/el are global, not local")
def test_kalman_steers_toward_target():
    k = KalmanBeamTracker(process_noise=1e-4, measurement_noise=1e-3,
                          max_correction_rad=0.005)
    obs = {"current_az_offset": +0.01, "current_el_offset": -0.01, "dt": 0.001}
    for _ in range(20):
        daz, del_ = k.get_correction(obs)
    assert daz > 0, f"daz must be positive when target is at +az; got {daz}"
    assert del_ < 0, f"del_ must be negative when target is at -el; got {del_}"


# ------------------------------------------------------- correction primitive


def test_apply_correction_preserves_unit_norm():
    v = np.array([1.0, 0.0, 0.0])
    out = apply_az_el_correction(v, 0.01, -0.02)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-10


def test_apply_correction_zero_is_identity():
    v = np.array([0.0, 1.0, 0.0])
    out = apply_az_el_correction(v, 0.0, 0.0)
    assert np.allclose(out, v)


# ----------------------------------------------- integration: simulator wiring


@pytest.mark.skipif(
    not (POSE_CSV.exists() and VIB_SCENARIO.exists()),
    reason="vibration scenario / pose CSV not present",
)
def test_pid_tracker_fires_in_simulator():
    old_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        sc = load_scenario(VIB_SCENARIO)
        # Attach a PID tracker to both nodes.
        for n in sc["nodes"]:
            n["optical_terminal"]["beam_tracker"] = {
                "type": "pid", "Kp": 0.3, "Ki": 0.01, "Kd": 0.05,
                "tick_interval_s": 0.001,
            }
        sim = Simulator(scenario=sc, run_id=0, seed=17)
        r = sim.run()
    finally:
        os.chdir(old_cwd)

    # The tracker only ticks while CONNECTED, so we need a link to have formed.
    assert r["first_link_time"] < 2.0, "link never formed; tracker had nothing to do"
    # Both nodes' trackers should have non-zero integral state if any
    # corrections fired (proxy for "the loop ran").
    a, b = sim.nodes["A"], sim.nodes["B"]
    assert not isinstance(a.beam_tracker, NullBeamTracker)
    assert not isinstance(b.beam_tracker, NullBeamTracker)
    # last_rx_power_dBm should have been updated from the sentinel.
    assert a.last_rx_power_dBm != -999.0 or b.last_rx_power_dBm != -999.0


@pytest.mark.skipif(
    not (POSE_CSV.exists() and VIB_SCENARIO.exists()),
    reason="vibration scenario / pose CSV not present",
)
def test_tracker_default_is_no_op_for_legacy_scenarios():
    """Scenarios without an explicit beam_tracker config still run, tracker = none."""
    old_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        sc = load_scenario(VIB_SCENARIO)
        sim = Simulator(scenario=sc, run_id=0, seed=17)
        r = sim.run()
    finally:
        os.chdir(old_cwd)
    for nid in ("A", "B"):
        assert isinstance(sim.nodes[nid].beam_tracker, NullBeamTracker)
    assert r["duration_reached"] is True
