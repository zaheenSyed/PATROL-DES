"""R3 smoke tests for TrackerATPEnv.

These don't train anything; they just verify the env hits the Gymnasium
contract (reset returns valid obs, step returns 5-tuple, random rollouts
complete without errors).
"""

import math
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from owdesmp.rl.tracker_env import PolicyHolderTracker, TrackerATPEnv  # noqa: E402


def _scenario_factory(seed: int) -> dict:
    """Two-node ATP scenario with switch_with_fallback config: wide acquisition
    beam, narrow maintenance beam, revert on link loss, drone-CSV vibration.
    Matches the structure exp_atp_baseline builds so the env trains on the
    same distribution the R0 baselines were evaluated on."""
    from experiments.exp_atp_baseline import (  # noqa: E402
        build_scenario, sample_node_b_positions,
    )
    from random import Random
    pos_rng = Random(seed)
    pos = sample_node_b_positions(1, 100.0, 2.0, pos_rng)[0]
    sc = build_scenario(
        policy="switch_with_fallback",
        mag_mult=1.0,
        dist_m=100.0,
        pos_b=pos,
        duration_s=1.0,  # short so smoke tests are fast
        seed=seed,
    )
    return sc


def test_env_constructs_and_resets():
    env = TrackerATPEnv(scenario_factory=_scenario_factory, seed=0)
    obs, info = env.reset(seed=42)
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    for key in env.observation_space.spaces:
        assert key in obs, f"missing obs key: {key}"
        assert obs[key].shape == env.observation_space.spaces[key].shape
    env.close()


def test_env_action_space_shape():
    env = TrackerATPEnv(scenario_factory=_scenario_factory, seed=0)
    a = env.action_space.sample()
    assert a.shape == (3,)
    env.close()


def test_zero_action_rollout_terminates():
    """All-zero actions must not raise; the episode terminates when the sim ends."""
    env = TrackerATPEnv(scenario_factory=_scenario_factory, max_steps=2000, seed=0)
    obs, _ = env.reset(seed=11)
    zero_action = np.zeros(3, dtype=np.float32)
    steps = 0
    done = False
    total_reward = 0.0
    while not done and steps < 5000:
        obs, reward, terminated, truncated, info = env.step(zero_action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
    assert done, "zero-action rollout did not terminate within 5000 steps"
    assert steps > 0
    env.close()


def test_random_action_rollout_terminates():
    env = TrackerATPEnv(scenario_factory=_scenario_factory, max_steps=2000, seed=0)
    obs, _ = env.reset(seed=22)
    rng = np.random.default_rng(7)
    steps = 0
    done = False
    while not done and steps < 5000:
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        # Sanity on every step.
        assert math.isfinite(reward)
        steps += 1
    assert done
    env.close()


def test_mode_switch_action_is_observable_in_info():
    """When the agent requests a switch with action[2] > 0, the info dict
    should reflect that."""
    env = TrackerATPEnv(scenario_factory=_scenario_factory, max_steps=200, seed=0)
    env.reset(seed=33)
    # Try repeatedly requesting a switch.
    forced_switch = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    saw_switch_request = False
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(forced_switch)
        if info.get("mode_swap_requested", False):
            saw_switch_request = True
            break
        if terminated or truncated:
            break
    assert saw_switch_request, "force-switch action never got accepted"
    env.close()


def test_reward_increases_with_throughput():
    """A trivially aligned scenario (zero vibration) should yield non-negative
    rewards on most steps. Sanity check that reward is wired to byte deltas."""
    def _aligned_factory(seed):
        return _scenario_factory_with_mag(seed, 0.0)
    env = TrackerATPEnv(scenario_factory=_aligned_factory, max_steps=200, seed=0)
    env.reset(seed=44)
    zero_action = np.zeros(3, dtype=np.float32)
    positive_steps = 0
    total_steps = 0
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        total_steps += 1
        if reward > 0:
            positive_steps += 1
        if terminated or truncated:
            break
    # We expect most steps to yield positive throughput reward in a clean,
    # vibration-free run.
    assert positive_steps > total_steps * 0.3, (
        f"only {positive_steps}/{total_steps} steps had positive reward; "
        f"reward wiring may be broken"
    )
    env.close()


def _scenario_factory_with_mag(seed: int, mag: float) -> dict:
    from experiments.exp_atp_baseline import (
        build_scenario, sample_node_b_positions,
    )
    from random import Random
    pos_rng = Random(seed)
    pos = sample_node_b_positions(1, 100.0, 2.0, pos_rng)[0]
    return build_scenario(
        policy="always_wide",
        mag_mult=mag,
        dist_m=100.0,
        pos_b=pos,
        duration_s=1.0,
        seed=seed,
    )


def test_policy_holder_tracker_returns_set_action():
    """Direct unit test on the helper class - no env required."""
    t = PolicyHolderTracker()
    assert t.get_correction({}) == (0.0, 0.0)
    t.set_action(0.001, -0.002)
    assert t.get_correction({}) == (0.001, -0.002)
    t.reset()
    assert t.get_correction({}) == (0.0, 0.0)
