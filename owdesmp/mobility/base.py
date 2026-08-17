"""Mobility model interface.

All mobility models map sim time -> 3D position. Velocity is optional and used
by mobility-aware metrics or by the channel engine (Doppler) in later slices.
"""

from typing import Protocol

import numpy as np


class MobilityModel(Protocol):
    def position_at(self, t: float) -> np.ndarray: ...
    def velocity_at(self, t: float) -> np.ndarray: ...
