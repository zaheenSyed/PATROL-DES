"""Constant-velocity mobility: linear extrapolation from an initial position."""

import numpy as np


class ConstantVelocity:
    def __init__(self, initial_m, velocity_mps):
        self._p0 = np.asarray(initial_m, dtype=float)
        self._v = np.asarray(velocity_mps, dtype=float)

    def position_at(self, t: float) -> np.ndarray:
        return self._p0 + self._v * t

    def velocity_at(self, t: float) -> np.ndarray:
        return self._v.copy()
