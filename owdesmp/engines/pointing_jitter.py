"""Pointing-jitter engines.

A pointing jitter source maps sim time -> (dRoll, dPitch, dYaw) in RADIANS.
The simulator applies the resulting rotation to a node's nominal beam
direction after each NODE_MOBILITY_UPDATE, before refreshing alignment.

Two backends:

  CSVPointingJitter
    Plays back rotational deviations from pose_deviation.csv (same file as
    CSVPlaybackVibration). Source data is in DEGREES; this class converts to
    radians at the boundary. Supports time_offset_s and mag_mult so two
    nodes can use the same CSV while staying decorrelated.

  PITransformerJitter
    Stub. The trained PI-Transformer fails validation on Yaw because it was
    trained per-file-normalized but the saved global stats over-amplify any
    per-file Yaw variation by ~125x at inference. Construction raises with
    the diagnosis until the model is retrained on deviation-form targets.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class NullPointingJitter:
    """No-op jitter used when no pointing_jitter config is supplied."""

    def sample_rad(self, t: float) -> Tuple[float, float, float]:
        return (0.0, 0.0, 0.0)


class CSVPointingJitter:
    """Rotational deviation playback from pose_deviation.csv.

    Args:
        csv_path: path to pose_deviation.csv with columns
            t, dRoll, dPitch, dYaw  (degrees)
        time_offset_s: phase offset; lets two nodes decorrelate
        mag_mult: scalar multiplier on all jitter (0 disables)
    """

    def __init__(
        self,
        csv_path: str,
        time_offset_s: float = 0.0,
        mag_mult: float = 1.0,
    ):
        self._csv_path = str(csv_path)
        self._time_offset = float(time_offset_s)
        self._mag_mult = float(mag_mult)

        df = pd.read_csv(self._csv_path)
        required = {"t", "dRoll", "dPitch", "dYaw"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{self._csv_path} missing columns: {missing}")

        self._t = df["t"].to_numpy(dtype=float)
        self._duration = float(self._t[-1] - self._t[0])
        if self._duration <= 0:
            raise ValueError(f"{self._csv_path} has zero/negative duration")

        # Store in radians up front (CSV is in degrees per inspection).
        dr = np.radians(df["dRoll"].to_numpy(dtype=float))
        dp = np.radians(df["dPitch"].to_numpy(dtype=float))
        dy = np.radians(df["dYaw"].to_numpy(dtype=float))
        self._fr = interp1d(self._t, dr, kind="linear", bounds_error=False, fill_value=0.0)
        self._fp = interp1d(self._t, dp, kind="linear", bounds_error=False, fill_value=0.0)
        self._fy = interp1d(self._t, dy, kind="linear", bounds_error=False, fill_value=0.0)

    @property
    def duration_s(self) -> float:
        return self._duration

    def _phase(self, t: float) -> float:
        rel = (float(t) + self._time_offset) - self._t[0]
        wrapped = rel % self._duration
        return self._t[0] + wrapped

    def sample_rad(self, t: float) -> Tuple[float, float, float]:
        if self._mag_mult == 0.0:
            return (0.0, 0.0, 0.0)
        tc = self._phase(t)
        dr = float(self._fr(tc)) * self._mag_mult
        dp = float(self._fp(tc)) * self._mag_mult
        dy = float(self._fy(tc)) * self._mag_mult
        return (dr, dp, dy)


class PITransformerJitter:
    """Stub — disabled until the model is retrained on deviation-form data."""

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "PITransformerJitter is disabled. validate_models.py showed the "
            "model's autoregressive Yaw output drifts by ~135 deg/step when "
            "denormalized with global stats, because training normalized "
            "per-file (each file's Yaw std ~1 deg) but inference must use one "
            "stats set (global Yaw std ~124 deg). Retrain the model on "
            "(Pitch - running_mean, Yaw - running_mean, vN, vE, vD) so it "
            "predicts JITTER directly, then re-enable this class. "
            "Use pointing_jitter.type: csv_playback in the meantime."
        )

    def sample_rad(self, t: float) -> Tuple[float, float, float]:  # pragma: no cover
        raise NotImplementedError


def build_pointing_jitter(cfg: dict):
    """Factory used by scenario_loader / simulator._build_optical_terminal.

    cfg schema (under optical_terminal.pointing_jitter):
      {
        "type": "csv_playback" | "pi_transformer",
        "csv_path": "...",            # csv_playback only
        "time_offset_s": 0.0,         # csv_playback only
        "mag_mult": 1.0,              # csv_playback only
        "model_path": "...",          # pi_transformer only (when wired)
        "stats_path": "...",          # pi_transformer only (when wired)
      }
    """
    if cfg is None:
        return NullPointingJitter()
    t = cfg.get("type", "csv_playback")
    if t == "csv_playback":
        return CSVPointingJitter(
            csv_path=cfg["csv_path"],
            time_offset_s=float(cfg.get("time_offset_s", 0.0)),
            mag_mult=float(cfg.get("mag_mult", 1.0)),
        )
    if t == "pi_transformer":
        return PITransformerJitter(
            model_path=cfg.get("model_path"),
            stats_path=cfg.get("stats_path"),
            mag_mult=float(cfg.get("mag_mult", 1.0)),
        )
    raise ValueError(f"Unknown pointing_jitter type: {t!r}")


def jitter_rotation_matrix(droll: float, dpitch: float, dyaw: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from intrinsic ZYX Euler angles (rad).

    Order: yaw (Z) -> pitch (Y) -> roll (X), matching the convention used by
    agent_sim_bifso/VibratingNode so cross-validation against that simulator
    stays meaningful.
    """
    cr, sr = np.cos(droll), np.sin(droll)
    cp, sp = np.cos(dpitch), np.sin(dpitch)
    cy, sy = np.cos(dyaw), np.sin(dyaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )
