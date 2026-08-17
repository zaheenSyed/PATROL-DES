"""Gymnasium environment wrapper for ATP (acquisition-tracking-pointing) RL.

The agent observes the tracker's view of the link (rx power, angular offsets,
mode state, time-since events, recent throughput) at each TRACKER_TICK and
chooses:

  action[0]: azimuth correction in rad   (continuous, clipped to +/- max_corr)
  action[1]: elevation correction in rad (continuous, clipped to +/- max_corr)
  action[2]: mode-switch logit           (> 0 requests an ATP swap)

We deliberately put the discrete mode-switch into a Box element rather than
using gymnasium.spaces.Tuple/Dict, so stable-baselines3 PPO with the default
MlpPolicy / MultiInputPolicy works without a custom hybrid head. The agent
learns the threshold implicitly.

Reward (the user's locked Option-2 choice from the previous session):

    throughput_term  = bytes_acked_this_tick / max_possible_bytes_this_tick
    switch_penalty   = 0.1   if a mode swap was actually scheduled this tick
    outage_penalty   = 5.0   if a LINK_LOST event fired during this tick
    steering_cost    = 0.001 * (daz**2 + del**2)
    reward           = throughput_term - switch_penalty - outage_penalty - steering_cost

The env drives the simulator via Simulator.advance_until_event(TRACKER_TICK)
and Simulator._dispatch(ev). It replaces both nodes' beam_tracker with a
PolicyHolderTracker that returns the agent's pointing correction when the
TRACKER_TICK handler asks for it — so all the existing tracker bookkeeping
(rx_power gradient, alignment refresh after correction) is reused.

Episode termination:
  - sim reaches duration_s (SIM_END)
  - sim reaches a true network-level termination_condition

Truncation:
  - max_steps reached (optional, default disabled)
"""

from typing import Callable, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "gymnasium is required for owdesmp.rl. Install with: pip install gymnasium"
    ) from e

from owdesmp.core.event import (
    BEAM_MODE_SWITCH,
    SIM_END,
    TRACKER_TICK,
)
from owdesmp.core.simulator import Simulator
from owdesmp.engines.beam_tracker import BeamTracker


class PolicyHolderTracker(BeamTracker):
    """BeamTracker that returns externally-set (daz, del) on demand.

    The env writes the agent's pointing correction here just before
    dispatching the pending TRACKER_TICK, so the existing
    Simulator._on_tracker_tick handler can use it transparently.
    """

    name = "rl_policy_holder"

    # The RL agent commands an absolute fine-pointing OFFSET from the tracked
    # target direction, NOT a cumulative increment. This is essential: a
    # stochastic policy can never output exactly zero-mean actions, and under
    # cumulative semantics any residual bias integrates into multi-degree beam
    # drift over an episode's hundreds of ticks (the bug that made the first
    # two training runs collapse to <1 MB). With offset semantics, zero action
    # tracks the target perfectly and the agent's job is to add a small offset
    # that cancels the INSTANTANEOUS jitter.
    correction_is_absolute_offset = True

    def __init__(self):
        self._pending_action = (0.0, 0.0)

    def set_action(self, daz: float, del_: float) -> None:
        self._pending_action = (float(daz), float(del_))

    def get_correction(self, obs):
        return self._pending_action

    def reset(self) -> None:
        self._pending_action = (0.0, 0.0)


class TrackerATPEnv(gym.Env):
    """Gymnasium environment for ATP beam-tracker RL."""

    metadata = {"render_modes": []}

    # Normalization constants (paper baselines stay in these ranges).
    _RX_DBM_REF = -50.0
    _RX_DBM_SCALE = 50.0
    _MAX_RX_NORMALIZED_BPS = 1.0e9  # 1 Gbps reference for throughput term

    def __init__(
        self,
        scenario_factory: Callable[[int], dict],
        max_corr_rad: float = 0.005,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            scenario_factory: callable(seed) -> scenario dict. Called once
                              per reset() with a fresh integer seed drawn
                              from the env's RNG.
            max_corr_rad: hard clip on each-axis pointing correction.
            max_steps: optional truncation budget per episode.
            seed: env-level RNG seed for the scenario_factory seed stream.
        """
        super().__init__()
        self.scenario_factory = scenario_factory
        self.max_corr = float(max_corr_rad)
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        # Action: [norm_daz, norm_del, mode_switch_logit] all in [-1, +1].
        # Pointing axes are scaled internally by max_corr in step(). Using
        # a normalized action space lets PPO's default unbounded Gaussian
        # (log_std_init~0, std~1) sit comfortably inside the action range,
        # so most stochastic samples DON'T saturate against the clip - the
        # policy actually gets gradient signal from small corrections.
        # The third axis is the mode-switch logit (> 0 = request swap).
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: dict (works with sb3 MultiInputPolicy).
        self.observation_space = spaces.Dict({
            "rx_power_dBm_norm":       _box1(-1.0, 1.0),
            "rx_grad_az":              _box1(-10.0, 10.0),
            "rx_grad_el":              _box1(-10.0, 10.0),
            "az_offset":               _box1(-np.pi, np.pi),
            "el_offset":               _box1(-np.pi, np.pi),
            "is_maintenance":          _box1(0.0, 1.0),
            "is_settling":             _box1(0.0, 1.0),
            "time_since_mode_switch":  _box1(0.0, 1e6),
            "time_since_outage":       _box1(0.0, 1e6),
            "recent_throughput_norm":  _box1(0.0, 1.0),
            "dt":                      _box1(0.0, 1.0),
        })

        # Runtime state (set on reset).
        self.sim: Optional[Simulator] = None
        self._pending_tick = None
        self._terminated = False
        self._steps = 0
        self._last_mode_switch_time = -1.0
        self._last_outage_time = -1.0
        self._recent_throughput_norm = 0.0
        self._last_bytes_acked = 0.0
        self._last_outage_count = 0
        self._last_mode_switch_count = 0

    # ----------------------------------------------------------------- reset

    # Up to this many fresh scenarios will be drawn per reset() if a draw
    # fails to ever produce a TRACKER_TICK (common at extreme vibration mag).
    _RESET_MAX_RETRIES = 8

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        last_info = {}
        for attempt in range(self._RESET_MAX_RETRIES):
            scenario_seed = int(self._rng.integers(0, 2 ** 31))
            scenario = self.scenario_factory(scenario_seed)

            self.sim = Simulator(scenario=scenario, run_id=0, seed=scenario_seed)
            # Swap in the policy-holder tracker on every node so we control all
            # the pointing corrections from this env.
            for node in self.sim.nodes.values():
                node.beam_tracker = PolicyHolderTracker()

            self._terminated = False
            self._steps = 0
            self._last_mode_switch_time = 0.0
            self._last_outage_time = 0.0
            self._recent_throughput_norm = 0.0
            self._last_bytes_acked = 0.0
            self._last_outage_count = 0
            self._last_mode_switch_count = 0

            # Spin the queue to the first TRACKER_TICK. If the link never
            # establishes (common at high vibration mag), the sim will end
            # before we ever get one - retry with a fresh seed.
            ev = self.sim.advance_until_event({TRACKER_TICK, SIM_END})
            if ev is not None and ev.type == TRACKER_TICK:
                self._pending_tick = ev
                return self._build_obs(ev), {"reset_attempts": attempt + 1}

            # This draw failed to ever yield a tracker tick. Finalize the
            # sim so its bookkeeping is clean, then try again.
            try:
                self.sim._finalize()
            except Exception:  # noqa: BLE001
                pass
            last_info = {"reason": "no_tracker_tick_before_end",
                         "reset_attempts": attempt + 1}

        # All retries exhausted; return a degenerate but valid obs. step()
        # will detect _pending_tick is None and return a terminal 5-tuple,
        # letting SB3's VecEnv auto-reset gracefully.
        self._pending_tick = None
        self._terminated = True
        return self._zero_obs(), last_info

    # ------------------------------------------------------------------ step

    def step(self, action):
        if self._terminated or self._pending_tick is None:
            # Defensive: if SB3 calls step() on a terminal env (e.g. a reset
            # that exhausted retries and returned a degenerate obs), return a
            # zero-reward terminal 5-tuple instead of raising. VecEnvs will
            # then auto-reset on the next iteration. Raising here crashes the
            # whole training run, which is much worse than dropping one step.
            info = {"reason": "step_after_terminal",
                    "data_session": (self.sim.result.get("data_session", {})
                                     if self.sim is not None else {})}
            return self._zero_obs(), 0.0, True, False, info

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # action[0:2] are normalized to [-1, +1]; rescale to physical rad and
        # then clip defensively. action[2] is the mode-switch logit (raw).
        norm_az = float(np.clip(action[0], -1.0, 1.0))
        norm_el = float(np.clip(action[1], -1.0, 1.0))
        daz = norm_az * self.max_corr
        del_ = norm_el * self.max_corr
        switch_logit = float(action[2]) if action.size > 2 else 0.0

        ev = self._pending_tick
        nid = ev.payload["node"]
        node = self.sim.nodes[nid]
        if isinstance(node.beam_tracker, PolicyHolderTracker):
            node.beam_tracker.set_action(daz, del_)

        # Schedule a mode swap if requested AND legal: not currently settling,
        # the node has a maintenance terminal configured, and the node is
        # involved in at least one linked pair.
        mode_swap_requested = False
        if (
            switch_logit > 0.0
            and node.optical_terminal_maintenance is not None
            and self.sim.clock >= node.mode_switch_active_until
            and any(nid in p for p in self.sim.linked_pairs)
        ):
            node.mode_switch_active_until = (
                self.sim.clock + self.sim.mode_switch_settling_s
            )
            self.sim.queue.push(
                self.sim.clock + self.sim.mode_switch_settling_s,
                BEAM_MODE_SWITCH,
                {"node": nid},
            )
            mode_swap_requested = True
            self._last_mode_switch_time = self.sim.clock

        # Capture pre-state for reward.
        prev_bytes = self._current_bytes_acked()
        prev_outages = self._current_outage_count()
        prev_mode_switches = self.sim.result.get("mode_switches", 0)

        # Dispatch the held tracker tick (applies action + reschedules next).
        self.sim._dispatch(ev)

        # Advance to next decision point.
        next_ev = self.sim.advance_until_event({TRACKER_TICK, SIM_END})

        # Reward components.
        bytes_acked_now = self._current_bytes_acked()
        bytes_delta = max(0.0, bytes_acked_now - prev_bytes)
        new_outages = self._current_outage_count() - prev_outages
        new_mode_switches = self.sim.result.get("mode_switches", 0) - prev_mode_switches
        if new_outages > 0:
            self._last_outage_time = self.sim.clock

        tick_interval = node.beam_tracker_tick_s
        max_bytes_this_tick = self._MAX_RX_NORMALIZED_BPS * tick_interval / 8.0
        throughput_term = (bytes_delta / max_bytes_this_tick) if max_bytes_this_tick > 0 else 0.0
        switch_penalty = 0.5 if (mode_swap_requested or new_mode_switches > 0) else 0.0
        outage_penalty = 5.0 if new_outages > 0 else 0.0
        # Steering cost on the NORMALIZED action so doing-nothing is a real
        # competitor at the reward level. Max correction (norm=1) costs 0.1
        # per axis, i.e. up to 0.2 - comparable to typical throughput_term.
        # Earlier value (0.001 * raw_rad^2) was effectively zero and the
        # policy learned to spam saturating corrections.
        steering_cost = 0.1 * (norm_az * norm_az + norm_el * norm_el)
        reward = float(throughput_term - switch_penalty - outage_penalty - steering_cost)

        self._recent_throughput_norm = float(np.clip(throughput_term, 0.0, 1.0))
        self._steps += 1

        # Termination / truncation. advance_until_event(...) returns None when
        # the queue empties or a non-matching event triggers a true termination
        # condition; it returns SIM_END directly if SIM_END was popped as a
        # match. Both are terminal states.
        terminated = False
        truncated = False
        if next_ev is None or next_ev.type == SIM_END:
            terminated = True
            self._terminated = True
            self.sim._finalize()
        elif self.max_steps is not None and self._steps >= self.max_steps:
            truncated = True
            self._terminated = True
            self.sim._finalize()

        info = {
            "clock": self.sim.clock,
            "bytes_delta": bytes_delta,
            "mode_swap_requested": mode_swap_requested,
            "outage_this_step": bool(new_outages > 0),
            "is_settling": self.sim.clock < node.mode_switch_active_until,
            "is_maintenance": node.in_maintenance_mode,
        }
        if terminated or truncated:
            obs = self._zero_obs()
            info["data_session"] = self.sim.result.get("data_session", {})
        else:
            self._pending_tick = next_ev
            obs = self._build_obs(next_ev)
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.sim is not None and not getattr(self.sim, "_finalized", False):
            self.sim._finalize()

    # ------------------------------------------------------------- obs/state

    def _build_obs(self, ev) -> dict:
        nid = ev.payload["node"]
        node = self.sim.nodes[nid]
        rx_dBm = float(node.last_rx_power_dBm)
        rx_norm = float(np.clip((rx_dBm - self._RX_DBM_REF) / self._RX_DBM_SCALE, -1.0, 1.0))

        # Reasonable defaults for fields we don't have a direct simulator
        # readout for yet (the tracker handler computes gradients/offsets
        # internally; we'd duplicate that work here for the agent's first
        # tick).
        return {
            "rx_power_dBm_norm":      np.array([rx_norm], dtype=np.float32),
            "rx_grad_az":             np.array([0.0], dtype=np.float32),
            "rx_grad_el":             np.array([0.0], dtype=np.float32),
            "az_offset":              np.array([0.0], dtype=np.float32),
            "el_offset":              np.array([0.0], dtype=np.float32),
            "is_maintenance":         np.array([1.0 if node.in_maintenance_mode else 0.0], dtype=np.float32),
            "is_settling":            np.array([1.0 if self.sim.clock < node.mode_switch_active_until else 0.0], dtype=np.float32),
            "time_since_mode_switch": np.array([self.sim.clock - self._last_mode_switch_time], dtype=np.float32),
            "time_since_outage":      np.array([self.sim.clock - self._last_outage_time], dtype=np.float32),
            "recent_throughput_norm": np.array([self._recent_throughput_norm], dtype=np.float32),
            "dt":                     np.array([node.beam_tracker_tick_s], dtype=np.float32),
        }

    def _zero_obs(self) -> dict:
        return {k: np.zeros(s.shape, dtype=np.float32)
                for k, s in self.observation_space.spaces.items()}

    def _current_bytes_acked(self) -> float:
        ds = self.sim.data_session if self.sim is not None else None
        if ds is None:
            return 0.0
        return float(ds.tcp.total_bytes_acked)

    def _current_outage_count(self) -> int:
        ds = self.sim.data_session if self.sim is not None else None
        if ds is None:
            return 0
        return int(ds.outage_count)


def _box1(low: float, high: float) -> spaces.Box:
    return spaces.Box(low=np.float32(low), high=np.float32(high),
                      shape=(1,), dtype=np.float32)
