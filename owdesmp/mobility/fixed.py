"""Fixed (stationary) mobility model."""

import numpy as np


class FixedPosition:
    def __init__(self, position_m):
        self._p = np.asarray(position_m, dtype=float)

    def position_at(self, t: float) -> np.ndarray:
        return self._p.copy()

    def velocity_at(self, t: float) -> np.ndarray:
        return np.zeros(3, dtype=float)
