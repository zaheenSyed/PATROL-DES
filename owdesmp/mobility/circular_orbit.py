"""Simplified circular orbit.

A node orbits `center_m` at fixed `radius_m` and `period_s` in the plane
normal to `plane_normal` (default: +Z). `phase_rad` shifts the starting
angle so multiple orbiters can be co-planar but offset.

For LEO-style sanity checks: a 500 km satellite has T ~ 5670 s and v ~ 7600
m/s; with this model you'd write radius_m = R_earth + 500 km and
period_s = 2 pi sqrt(r^3 / mu) externally and feed those in.
"""

import math

import numpy as np

from owdesmp.scanners.beta import _find_orthogonal_vectors


class CircularOrbit:
    def __init__(
        self,
        center_m,
        radius_m: float,
        period_s: float,
        plane_normal=(0.0, 0.0, 1.0),
        phase_rad: float = 0.0,
    ):
        self._c = np.asarray(center_m, dtype=float)
        self._r = float(radius_m)
        self._T = float(period_s)
        if self._T <= 0:
            raise ValueError("period_s must be positive")
        n = np.asarray(plane_normal, dtype=float)
        n_norm = float(np.linalg.norm(n))
        if n_norm == 0:
            raise ValueError("plane_normal cannot be zero")
        self._n = n / n_norm
        self._phase = float(phase_rad)
        # Two orthonormal basis vectors in the orbital plane.
        self._u, self._v = _find_orthogonal_vectors(self._n)

    def _theta(self, t: float) -> float:
        return self._phase + (2.0 * math.pi / self._T) * t

    def position_at(self, t: float) -> np.ndarray:
        theta = self._theta(t)
        return self._c + self._r * (math.cos(theta) * self._u + math.sin(theta) * self._v)

    def velocity_at(self, t: float) -> np.ndarray:
        omega = 2.0 * math.pi / self._T
        theta = self._theta(t)
        return self._r * omega * (-math.sin(theta) * self._u + math.cos(theta) * self._v)
