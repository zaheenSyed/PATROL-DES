"""Waypoint mobility: linear interpolation between (t, position) anchors.

Before the first waypoint time, position is clamped to the first anchor.
After the last waypoint time, position is clamped to the last anchor (no
extrapolation - waypoints define a closed trajectory). velocity_at returns
the secant velocity of the segment containing t, or zero outside the
trajectory.
"""

import numpy as np


class Waypoint:
    def __init__(self, waypoints):
        """waypoints: list of {'t_s': float, 'pos_m': [x, y, z]} (or equivalent
        list-of-pairs). Must be sorted in ascending t."""
        if not waypoints:
            raise ValueError("Waypoint mobility requires at least one waypoint")
        ts, ps = [], []
        for wp in waypoints:
            if isinstance(wp, dict):
                ts.append(float(wp["t_s"]))
                ps.append([float(v) for v in wp["pos_m"]])
            else:
                t, pos = wp
                ts.append(float(t))
                ps.append([float(v) for v in pos])
        self._t = np.asarray(ts, dtype=float)
        self._p = np.asarray(ps, dtype=float)
        if not np.all(np.diff(self._t) >= 0):
            raise ValueError("Waypoints must be sorted by t_s in non-decreasing order")

    def position_at(self, t: float) -> np.ndarray:
        if t <= self._t[0]:
            return self._p[0].copy()
        if t >= self._t[-1]:
            return self._p[-1].copy()
        idx = int(np.searchsorted(self._t, t, side="right")) - 1
        t0, t1 = self._t[idx], self._t[idx + 1]
        p0, p1 = self._p[idx], self._p[idx + 1]
        if t1 == t0:
            return p0.copy()
        frac = (t - t0) / (t1 - t0)
        return p0 + frac * (p1 - p0)

    def velocity_at(self, t: float) -> np.ndarray:
        if t < self._t[0] or t > self._t[-1]:
            return np.zeros(3, dtype=float)
        idx = int(np.searchsorted(self._t, t, side="right")) - 1
        idx = max(0, min(idx, len(self._t) - 2))
        t0, t1 = self._t[idx], self._t[idx + 1]
        if t1 == t0:
            return np.zeros(3, dtype=float)
        return (self._p[idx + 1] - self._p[idx]) / (t1 - t0)
