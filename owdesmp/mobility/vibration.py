"""Vibration mobility models.

Two backends share the MobilityModel interface (position_at, velocity_at):

  CSVPlaybackVibration
    Plays back real measured 6-DOF pose deviations from pose_deviation.csv
    (drone data/hover/pose_deviation.csv). Only the translational
    components (dX, dY, dZ) drive position; rotational components feed the
    pointing_jitter engine instead. Wraps in time so simulations longer
    than the CSV duration loop seamlessly.

  LSTMTrajectoryMobility
    Optional secondary backend that wraps the trained LSTM trajectory model
    from drone data/rl_simulation_models/lstm_trajectory_model.pth. Predicts
    absolute (PN, PE, PD) and reports `predicted - hover_center` as the
    deviation. Requires torch. Currently STUB — the LSTM was trained on
    sample-rate-mismatched data; needs a rate-corrected retrain before it
    can drive vibration. Raises a clear error at construction time.

Validation: CSVPlaybackVibration matches agent_sim_bifso's VibratingNode
playback behavior, so the two simulators produce comparable vibration
trajectories under the same CSV and time offset.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class CSVPlaybackVibration:
    """Position deviation from a high-rate measured 6-DOF CSV.

    Args:
        csv_path: path to pose_deviation.csv with columns
            t, dX, dY, dZ, dRoll, dPitch, dYaw
        base_position_m: nominal (3D) position; deviations add on top
        time_offset_s: phase offset into the CSV; lets two nodes use the
            same CSV but be decorrelated
        mag_mult: scalar multiplier on all deviations (0 disables vibration)
    """

    def __init__(
        self,
        csv_path: str,
        base_position_m,
        time_offset_s: float = 0.0,
        mag_mult: float = 1.0,
    ):
        self._csv_path = str(csv_path)
        self._base = np.asarray(base_position_m, dtype=float)
        self._time_offset = float(time_offset_s)
        self._mag_mult = float(mag_mult)

        df = pd.read_csv(self._csv_path)
        required = {"t", "dX", "dY", "dZ"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{self._csv_path} missing columns: {missing}")

        self._t = df["t"].to_numpy(dtype=float)
        self._duration = float(self._t[-1] - self._t[0])
        if self._duration <= 0:
            raise ValueError(f"{self._csv_path} has zero/negative duration")

        # Build one interpolator per translational axis. interp1d with
        # bounds_error=False + fill_value=0 means out-of-range queries (which
        # shouldn't happen because we wrap below) return 0 instead of NaN.
        dx = df["dX"].to_numpy(dtype=float)
        dy = df["dY"].to_numpy(dtype=float)
        dz = df["dZ"].to_numpy(dtype=float)
        self._fx = interp1d(self._t, dx, kind="linear", bounds_error=False, fill_value=0.0)
        self._fy = interp1d(self._t, dy, kind="linear", bounds_error=False, fill_value=0.0)
        self._fz = interp1d(self._t, dz, kind="linear", bounds_error=False, fill_value=0.0)

    @property
    def duration_s(self) -> float:
        return self._duration

    def _phase(self, t: float) -> float:
        """Map sim time to a CSV time within [t_min, t_max] via wrap-around."""
        rel = (float(t) + self._time_offset) - self._t[0]
        wrapped = rel % self._duration
        return self._t[0] + wrapped

    def position_at(self, t: float) -> np.ndarray:
        if self._mag_mult == 0.0:
            return self._base.copy()
        tc = self._phase(t)
        dx = float(self._fx(tc)) * self._mag_mult
        dy = float(self._fy(tc)) * self._mag_mult
        dz = float(self._fz(tc)) * self._mag_mult
        return self._base + np.array([dx, dy, dz], dtype=float)

    def velocity_at(self, t: float) -> np.ndarray:
        # Finite-difference around t (forward/backward step sized to the CSV
        # median dt to stay within one sample where possible).
        h = 1e-3
        p0 = self.position_at(t - h)
        p1 = self.position_at(t + h)
        return (p1 - p0) / (2.0 * h)


class LSTMTrajectoryMobility:
    """Stub — see module docstring for retrain rationale."""

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "LSTMTrajectoryMobility is not yet wired. The trained LSTM "
            "(drone data/rl_simulation_models/lstm_trajectory_model.pth) "
            "operates on what appears to be sample-rate-mismatched data "
            "(groupby(40) of a 10 Hz source = 0.25 Hz effective). It also "
            "predicts ABSOLUTE positions, so an inference wrapper must "
            "subtract hover_center to recover deviation. Use "
            "vibration_source: csv_playback until the model is retrained."
        )

    def position_at(self, t: float) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def velocity_at(self, t: float) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def build_vibration_mobility(cfg: dict):
    """Factory used by scenario_loader / simulator._build_mobility.

    cfg schema:
      {
        "model": "vibration_csv" | "vibration_lstm_nn",
        "initial_m": [x, y, z],
        "csv_path": "...",            # csv_playback only
        "time_offset_s": 0.0,         # csv_playback only
        "mag_mult": 1.0,              # csv_playback only
        "model_path": "...",          # lstm_nn only (when wired)
        "stats_path": "...",          # lstm_nn only (when wired)
      }
    """
    model = cfg.get("model")
    if model == "vibration_csv":
        return CSVPlaybackVibration(
            csv_path=cfg["csv_path"],
            base_position_m=cfg["initial_m"],
            time_offset_s=float(cfg.get("time_offset_s", 0.0)),
            mag_mult=float(cfg.get("mag_mult", 1.0)),
        )
    if model == "vibration_lstm_nn":
        return LSTMTrajectoryMobility(
            model_path=cfg.get("model_path"),
            stats_path=cfg.get("stats_path"),
            hover_center_m=cfg.get("initial_m"),
            mag_mult=float(cfg.get("mag_mult", 1.0)),
        )
    raise ValueError(f"Unknown vibration model: {model!r}")
