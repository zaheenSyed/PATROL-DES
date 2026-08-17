"""Beam-tracking controllers for sustained alignment under jitter.

Trackers run at a high cadence (~1 kHz) while a link is CONNECTED. Each tick
the simulator hands the tracker an observation dict and gets back a small
azimuth/elevation correction (in RADIANS) which is applied to the node's
nominal_pointing - i.e. the tracker is trying to ANTICIPATE the next jitter
sample so the actual current_pointing (= R(jitter) @ nominal_pointing) lands
on the peer.

Three concrete trackers:

  NullBeamTracker     no-op; matches legacy "tracker disabled" behavior
  PIDBeamTracker      proportional + integral + derivative on the rx_power
                      gradient (proxy for angular error). Tunable Kp/Ki/Kd.
  KalmanBeamTracker   constant-velocity Kalman over [az, el, daz/dt, del/dt]
                      with the same gradient measurement. Predicts future
                      pointing offset, applies it inverted.

The observation dict the simulator builds (see Simulator._build_tracker_obs):

  rx_power_dBm           current min(rx_a, rx_b)
  rx_power_gradient_az   approx d(rx)/d(az)   (rad^-1, sign-only on first tick)
  rx_power_gradient_el   approx d(rx)/d(el)
  current_az_offset      |angle_between(pointing_a, vec(a->b))|
  current_el_offset      |angle_between(pointing_b, vec(b->a))|
  dt                     time since last tracker tick

The corrections come back already CLIPPED to ±max_correction_rad so the
simulator can apply them blindly.
"""

from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------- base + null


class BeamTracker:
    """Abstract base. Subclasses override get_correction().

    correction_is_absolute_offset controls how the simulator applies the
    returned (daz, del):
      - False (default, legacy): correction is ACCUMULATED onto the node's
        current nominal_pointing each tick. A persistent bias integrates into
        drift, so feedback controllers (PID/Kalman) that drive error toward
        zero must be used. Required for the gradient-following trackers.
      - True: correction is an OFFSET applied to the ideal target direction
        (a_to_b) recomputed every tick, so nominal = rotate(a_to_b, daz, del).
        Zero correction => perfect tracking; bias/noise does NOT accumulate.
        This is what makes the RL fine-pointing problem learnable for a
        stochastic policy (see PolicyHolderTracker).
    """

    name = "base"
    correction_is_absolute_offset = False

    def get_correction(self, obs: Dict[str, float]) -> Tuple[float, float]:
        return (0.0, 0.0)

    def reset(self) -> None:
        pass


class NullBeamTracker(BeamTracker):
    name = "none"

    def get_correction(self, obs):
        return (0.0, 0.0)


# --------------------------------------------------------------------- PID


class PIDBeamTracker(BeamTracker):
    """Proportional-Integral-Derivative on the rx-power gradient.

    The gradient is a proxy for angular error: when the beam drifts AWAY
    from the peer, rx_power drops and the gradient becomes negative on the
    axis that drifted, so we steer toward the gradient direction.

    Args:
        Kp/Ki/Kd: gains
        max_correction_rad: hard clip on each-axis correction per tick
        integral_clip_rad: hard clip on integral state (anti-windup)
    """

    name = "PID"

    def __init__(
        self,
        Kp: float = 0.5,
        Ki: float = 0.01,
        Kd: float = 0.1,
        max_correction_rad: float = 0.005,
        integral_clip_rad: float = 0.01,
    ):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.max_correction = float(max_correction_rad)
        self.integral_clip = float(integral_clip_rad)

        self.integral_az = 0.0
        self.integral_el = 0.0
        self.prev_error_az = 0.0
        self.prev_error_el = 0.0

    def reset(self) -> None:
        self.integral_az = 0.0
        self.integral_el = 0.0
        self.prev_error_az = 0.0
        self.prev_error_el = 0.0

    def get_correction(self, obs):
        dt = float(obs.get("dt", 0.001))
        if dt <= 0.0:
            return (0.0, 0.0)
        # FIXME(beam-tracker-frame-bug): this controller does NOT correctly
        # steer the beam for arbitrary node orientations. Investigation on
        # 2026-05-31 traced two coupled issues:
        #
        #   (1) simulator._on_tracker_tick reports
        #         az_err = (a_to_b - ref)[1]   # global Y
        #         el_err = (a_to_b - ref)[2]   # global Z
        #       which is only the local azimuth/elevation tangent when the
        #       beam happens to point along world +x. For a node pointing
        #       -x the local azimuth axis is REVERSED in global Y, so the
        #       same correction has opposite effect on the two nodes.
        #
        #   (2) apply_az_el_correction rotates around global Z (yaw) and Y
        #       (pitch). For direction (+1,0,0) and daz>0 the beam moves
        #       toward +y; for direction (-1,0,0) the SAME daz moves the
        #       beam toward -y. The two effects compound so that in a
        #       symmetric A<->B link one node converges and the other
        #       diverges, and the mutual link drops within a few hundred
        #       ms regardless of Kp or sign convention.
        #
        # Proper fix is to (a) compute a LOCAL tangent basis from the
        # current pointing direction, project (a_to_b - ref) onto it for
        # az_err / el_err, and (b) rewrite apply_az_el_correction to
        # rotate around the local up / right axes. ~30 LoC change in
        # simulator.py + this file.
        #
        # Until that lands, the closed-loop PID/Kalman/RL trackers
        # CANNOT be trusted in any paper-quality experiment that has
        # two nodes pointing at each other. Use `tracker_type="none"`.
        # Sign of e_az / e_el is irrelevant until the frame is fixed.
        e_az = -float(obs.get("current_az_offset", 0.0))
        e_el = -float(obs.get("current_el_offset", 0.0))

        self.integral_az = float(np.clip(self.integral_az + e_az * dt,
                                         -self.integral_clip, self.integral_clip))
        self.integral_el = float(np.clip(self.integral_el + e_el * dt,
                                         -self.integral_clip, self.integral_clip))
        deriv_az = (e_az - self.prev_error_az) / dt
        deriv_el = (e_el - self.prev_error_el) / dt

        daz = self.Kp * e_az + self.Ki * self.integral_az + self.Kd * deriv_az
        del_ = self.Kp * e_el + self.Ki * self.integral_el + self.Kd * deriv_el

        self.prev_error_az = e_az
        self.prev_error_el = e_el

        daz = float(np.clip(daz, -self.max_correction, self.max_correction))
        del_ = float(np.clip(del_, -self.max_correction, self.max_correction))
        return (daz, del_)


# -------------------------------------------------------------------- Kalman


class KalmanBeamTracker(BeamTracker):
    """Constant-velocity Kalman filter over [az, el, az_dot, el_dot].

    Measurement is the gradient-implied angular error (same proxy as PID,
    but the filter smooths it). Correction = predicted future offset (one
    step ahead) inverted, clipped.
    """

    name = "Kalman"

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-3,
        max_correction_rad: float = 0.005,
    ):
        self.max_correction = float(max_correction_rad)
        # State: [az, el, az_dot, el_dot]
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1e-2
        # Measurement matrix: we observe [az, el].
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])
        # Process noise: bigger variance on the velocity components.
        self.Q = np.eye(4) * process_noise
        self.Q[2, 2] = process_noise * 10.0
        self.Q[3, 3] = process_noise * 10.0
        self.R = np.eye(2) * float(measurement_noise)

    def reset(self) -> None:
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1e-2

    def get_correction(self, obs):
        dt = float(obs.get("dt", 0.001))
        if dt <= 0.0:
            return (0.0, 0.0)
        # State transition (constant velocity).
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt

        # Predict.
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + self.Q

        # Measurement: direct angular offsets (quadrant-detector model). State
        # x[0:2] = (az_offset, el_offset); x[2:4] = their velocities. Was
        # previously fed gradient proxies, which produced an inconsistent
        # observation model — the gradient is not an offset.
        z = np.array([float(obs.get("current_az_offset", 0.0)),
                      float(obs.get("current_el_offset", 0.0))])
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        # FIXME(beam-tracker-frame-bug): see the long comment in
        # PIDBeamTracker.get_correction. Until the simulator computes
        # az/el in a LOCAL tangent frame, this Kalman filter cannot
        # produce a useful correction for nodes that don't point along
        # world +x. Use tracker_type="none" until the simulator-side fix
        # lands.
        daz = -(self.x[0] + self.x[2] * dt)
        del_ = -(self.x[1] + self.x[3] * dt)
        daz = float(np.clip(daz, -self.max_correction, self.max_correction))
        del_ = float(np.clip(del_, -self.max_correction, self.max_correction))
        return (daz, del_)


# ------------------------------------------------------------------- factory


def build_beam_tracker(cfg):
    """Factory used by scenario_loader / _build_node.

    cfg schema (under optical_terminal.beam_tracker):
      {
        "type": "none" | "pid" | "kalman",
        "Kp": ..., "Ki": ..., "Kd": ...,          # pid
        "process_noise": ..., "measurement_noise": ...,  # kalman
        "max_correction_rad": ...,
        "tick_interval_s": 0.001                  # consumed by the simulator,
                                                    # not the tracker itself
      }
    """
    if cfg is None:
        return NullBeamTracker()
    t = cfg.get("type", "none")
    if t == "none":
        return NullBeamTracker()
    if t == "pid":
        return PIDBeamTracker(
            Kp=cfg.get("Kp", 0.5),
            Ki=cfg.get("Ki", 0.01),
            Kd=cfg.get("Kd", 0.1),
            max_correction_rad=cfg.get("max_correction_rad", 0.005),
            integral_clip_rad=cfg.get("integral_clip_rad", 0.01),
        )
    if t == "kalman":
        return KalmanBeamTracker(
            process_noise=cfg.get("process_noise", 1e-4),
            measurement_noise=cfg.get("measurement_noise", 1e-3),
            max_correction_rad=cfg.get("max_correction_rad", 0.005),
        )
    raise ValueError(f"Unknown beam_tracker type: {t!r}")


def apply_az_el_correction(direction: np.ndarray, daz: float, del_: float) -> np.ndarray:
    """Apply a small azimuth/elevation correction to a unit pointing vector.

    daz rotates around Z (yaw), del_ rotates around Y (pitch). For small
    angles this is equivalent to translating the unit vector along the
    azimuth/elevation tangents and renormalizing.
    """
    cy, sy = np.cos(daz), np.sin(daz)
    cp, sp = np.cos(del_), np.sin(del_)
    R = np.array([
        [cy * cp, -sy, cy * sp],
        [sy * cp, cy, sy * sp],
        [-sp, 0.0, cp],
    ])
    out = R @ direction
    n = float(np.linalg.norm(out))
    if n > 0:
        out = out / n
    return out
