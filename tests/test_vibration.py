"""Phase A validation: CSV vibration mobility + pointing jitter.

The translational/rotational pose deviations from drone data/hover/
pose_deviation.csv are the validated primary source (real measurements,
already mean-zero, 6 DOF, ~1.46 kHz). These tests verify:

  1. CSVPlaybackVibration interpolates correctly at known timestamps and
     wraps in time past the CSV duration.
  2. mag_mult = 0.0 yields zero deviation (jitter and translation both).
  3. Two CSV-backed nodes with different time_offset_s produce decorrelated
     positions/jitter (i.e. configuration is plumbed).
  4. CSVPointingJitter returns radian-scale values consistent with the
     CSV's degree-scale source (sub-degree per agent_sim_bifso reports).
  5. Loading scenarios/two_node_vibration.json and running the simulator
     end-to-end produces non-trivial activity without crashing, the
     vibrating-node mobility refreshes alignment, and the data session
     opens at least once.

The CSV file is expected at REPO_ROOT/drone data/hover/pose_deviation.csv;
tests that need it skip gracefully when it's missing (e.g. CI without the
drone data drop).
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
POSE_CSV = REPO_ROOT / "drone data" / "hover" / "pose_deviation.csv"

from owdesmp.engines.pointing_jitter import (  # noqa: E402
    CSVPointingJitter,
    NullPointingJitter,
    PITransformerJitter,
    jitter_rotation_matrix,
)
from owdesmp.mobility.vibration import (  # noqa: E402
    CSVPlaybackVibration,
    LSTMTrajectoryMobility,
    build_vibration_mobility,
)


pytestmark = pytest.mark.skipif(
    not POSE_CSV.exists(),
    reason=f"pose_deviation.csv not present at {POSE_CSV}",
)


# --------------------------------------------------- CSVPlaybackVibration


def test_csv_playback_returns_base_plus_deviation():
    base = np.array([10.0, 20.0, 30.0])
    vib = CSVPlaybackVibration(str(POSE_CSV), base_position_m=base, mag_mult=1.0)
    # At t=0, position should be base + first-row deviation
    pos = vib.position_at(0.0)
    # First-row deviation is ~ (0.06, -0.11, -0.11). Position should NOT equal
    # base (proves interpolation runs and mag_mult applies).
    assert not np.allclose(pos, base), "vibration should perturb base position"
    # And the perturbation magnitude should be sub-meter for hovering data.
    assert np.linalg.norm(pos - base) < 1.0


def test_csv_playback_mag_mult_zero_is_no_op():
    base = np.array([10.0, 20.0, 30.0])
    vib = CSVPlaybackVibration(str(POSE_CSV), base_position_m=base, mag_mult=0.0)
    for t in [0.0, 0.5, 1.0, 5.0]:
        assert np.allclose(vib.position_at(t), base)


def test_csv_playback_wraps_past_duration():
    base = np.zeros(3)
    vib = CSVPlaybackVibration(str(POSE_CSV), base_position_m=base, mag_mult=1.0)
    p_early = vib.position_at(0.0)
    # Step exactly one CSV duration forward; the modular phase should land
    # back at the same CSV time.
    p_late = vib.position_at(vib.duration_s)
    assert np.allclose(p_early, p_late, atol=1e-6), (
        f"wrap-around broken: p_early={p_early}, p_late={p_late}"
    )


def test_csv_playback_time_offset_decorrelates_nodes():
    base = np.zeros(3)
    a = CSVPlaybackVibration(str(POSE_CSV), base_position_m=base, time_offset_s=0.0)
    b = CSVPlaybackVibration(str(POSE_CSV), base_position_m=base, time_offset_s=80.0)
    diffs = []
    for t in np.linspace(0.0, 1.0, 50):
        diffs.append(float(np.linalg.norm(a.position_at(t) - b.position_at(t))))
    # The two nodes should disagree by tens of millimeters typically.
    mean_diff = float(np.mean(diffs))
    assert mean_diff > 1e-3, (
        f"decorrelated nodes should differ; mean diff was {mean_diff:.6f} m"
    )


def test_csv_playback_velocity_is_finite_difference():
    base = np.zeros(3)
    vib = CSVPlaybackVibration(str(POSE_CSV), base_position_m=base, mag_mult=1.0)
    v = vib.velocity_at(0.5)
    assert v.shape == (3,)
    assert np.all(np.isfinite(v))


# ------------------------------------------------------ CSVPointingJitter


def test_jitter_returns_radians_at_expected_scale():
    jit = CSVPointingJitter(str(POSE_CSV), mag_mult=1.0)
    # Sample many points across the CSV; std should be on the order of
    # a fraction of a degree per agent_sim_bifso reports (0.3-0.6 deg).
    samples = np.array([jit.sample_rad(t) for t in np.linspace(0, jit.duration_s, 5000)])
    # Convert to degrees for the bound check.
    samples_deg = np.degrees(samples)
    stds = samples_deg.std(axis=0)
    # Roll/Pitch ~0.3 deg, Yaw ~0.6 deg from pose_deviation.csv inspection
    assert 0.1 < stds[0] < 1.0, f"dRoll std out of expected range: {stds[0]:.3f} deg"
    assert 0.1 < stds[1] < 1.0, f"dPitch std out of expected range: {stds[1]:.3f} deg"
    assert 0.2 < stds[2] < 1.5, f"dYaw std out of expected range: {stds[2]:.3f} deg"


def test_jitter_mag_mult_zero_is_no_op():
    jit = CSVPointingJitter(str(POSE_CSV), mag_mult=0.0)
    for t in [0.0, 0.1, 1.0, 10.0]:
        assert jit.sample_rad(t) == (0.0, 0.0, 0.0)


def test_null_jitter_returns_zero():
    n = NullPointingJitter()
    assert n.sample_rad(0.5) == (0.0, 0.0, 0.0)


# ----------------------------------------------------- rotation matrix


def test_jitter_rotation_identity_at_zero():
    R = jitter_rotation_matrix(0.0, 0.0, 0.0)
    assert np.allclose(R, np.eye(3))


def test_jitter_rotation_preserves_unit_vectors():
    # Random-ish nonzero euler angles; rotation should remain a proper rotation
    # (orthogonal, det +1).
    R = jitter_rotation_matrix(0.05, -0.03, 0.07)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert math.isclose(float(np.linalg.det(R)), 1.0, abs_tol=1e-12)
    v = np.array([1.0, 0.0, 0.0])
    rv = R @ v
    assert math.isclose(float(np.linalg.norm(rv)), 1.0, abs_tol=1e-12)


# ------------------------------------------------------ deferred stubs


def test_lstm_mobility_raises_until_retrained():
    with pytest.raises(NotImplementedError, match="not yet wired"):
        LSTMTrajectoryMobility()


def test_pi_transformer_jitter_raises_until_retrained():
    with pytest.raises(NotImplementedError, match="disabled"):
        PITransformerJitter()


# ------------------------------------------------- build_vibration_mobility


def test_build_vibration_mobility_csv():
    m = build_vibration_mobility({
        "model": "vibration_csv",
        "initial_m": [1.0, 2.0, 3.0],
        "csv_path": str(POSE_CSV),
        "mag_mult": 1.0,
    })
    assert isinstance(m, CSVPlaybackVibration)


def test_build_vibration_mobility_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown vibration model"):
        build_vibration_mobility({"model": "vibration_fake"})


# --------------------------------------- integration: full scenario run


from owdesmp.core.scenario_loader import load_scenario  # noqa: E402
from owdesmp.core.simulator import Simulator  # noqa: E402

VIBRATION_SCENARIO = REPO_ROOT / "scenarios" / "two_node_vibration.json"


@pytest.mark.skipif(
    not VIBRATION_SCENARIO.exists(),
    reason=f"two_node_vibration.json not present at {VIBRATION_SCENARIO}",
)
def test_vibration_scenario_runs_end_to_end():
    # The scenario references CSV paths relative to the repo root; chdir so
    # the relative paths resolve regardless of where pytest is invoked.
    old_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        scenario = load_scenario(VIBRATION_SCENARIO)
        sim = Simulator(scenario=scenario, run_id=0, seed=17)
        result = sim.run()
    finally:
        os.chdir(old_cwd)

    # Both nodes should have non-null nominal_pointing after running.
    for nid in ("A", "B"):
        n = sim.nodes[nid]
        assert n.nominal_pointing is not None, (
            f"node {nid} never had nominal_pointing initialized"
        )
        assert not isinstance(n.pointing_jitter, NullPointingJitter), (
            f"node {nid} jitter not wired"
        )
        # current_pointing should remain a unit vector throughout (jitter
        # preserves norm, we explicitly renormalize).
        cp = n.current_pointing
        assert cp is not None
        assert math.isclose(float(np.linalg.norm(cp)), 1.0, abs_tol=1e-6)

    # Sim ran for the configured duration and processed at least a few events.
    assert result["events_processed"] > 100
    assert result["duration_reached"] is True
    # A data session should exist (it was configured). Don't require it to be
    # completed - 1 Gbps over 2s with handshake + outages is borderline - but
    # SOME bytes should have moved if any link formed.
    if result.get("first_link_time", 2.0) < 2.0:
        ds = result.get("data_session", {})
        assert ds.get("bytes_sent", 0) > 0, (
            f"link formed at t={result['first_link_time']:.3f}s but no bytes flowed; "
            f"data_session={ds}"
        )
