"""Beta-distribution random-walk scanner.

Ported from legacy/fso_sim_modules.py:316 (`BetaScanner`). The legacy version
generates a 1000-point path up-front; this DES version returns one target per
call and keeps the walk state across calls so serial correlation is preserved.

Walk update: theta_new = (sample * 2*pi) - (pi - theta_prev), where
sample ~ Beta(alpha, beta). The target point lies on the perimeter of the
search disk (the intersection of the search sphere of radius `search_radius_m`
centered at the peer's estimated position with the plane perpendicular to
the searcher->peer vector).
"""

import math

import numpy as np


def _find_orthogonal_vectors(normal: np.ndarray):
    """Two orthonormal vectors spanning the plane perpendicular to `normal`."""
    norm_val = float(np.linalg.norm(normal))
    if norm_val == 0.0:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    n = normal / norm_val
    min_idx = int(np.argmin(np.abs(n)))
    ref = np.zeros(3, dtype=float)
    ref[min_idx] = 1.0
    v1 = np.cross(n, ref)
    v1n = float(np.linalg.norm(v1))
    if v1n == 0.0:
        ref = np.zeros(3, dtype=float)
        ref[(min_idx + 1) % 3] = 1.0
        v1 = np.cross(n, ref)
        v1n = float(np.linalg.norm(v1))
    v1 /= v1n
    v2 = np.cross(n, v1)
    v2 /= float(np.linalg.norm(v2))
    return v1, v2


class BetaScanner:
    """Beta-walk perimeter / disk scanner.

    Parameters
    ----------
    alpha, beta
        Shape parameters of the Beta(alpha, beta) draw that picks the
        angular step. U-shaped (alpha = beta < 1) drives consecutive
        targets near-antipodal so the chord between them sweeps the disk
        interior.
    radius_mode
        ``"perimeter"`` (default) — every target lies on the search-disk
        perimeter. This is the legacy/published behaviour.
        ``"uniform_disk"`` — radius is drawn ``r = R * sqrt(U)`` (uniform
        by area). Some targets land near the centre, which shortens the
        chord traversal and lets the beam dwell briefly inside the
        peer-alignment cone before the next slew. Recommended at close
        range with narrow beam, where perimeter-only chords sweep through
        the alignment window faster than the handshake can complete.
    """

    def __init__(
        self,
        alpha: float = 0.41485,
        beta: float = 0.41485,
        radius_mode: str = "perimeter",
        center_start: bool = False,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        if radius_mode not in ("perimeter", "uniform_disk"):
            raise ValueError(
                f"radius_mode must be 'perimeter' or 'uniform_disk', "
                f"got {radius_mode!r}"
            )
        self.radius_mode = radius_mode
        # When True, the FIRST target returned is the disk centre (the peer
        # estimate), so the beta walk dwells at the centre before slewing to its
        # first random walk point -- matching the stock spiral/raster, which
        # begin at the centre. Makes all scanners start from a common centre
        # point. Off by default so legacy beta behaviour is unchanged.
        self.center_start = bool(center_start)
        self._first = True
        self._theta = 0.0

    def next_target(
        self,
        rng,
        my_pos: np.ndarray,
        peer_estimated_pos: np.ndarray,
        search_radius_m: float,
        beam_footprint_radius_m: float = 0.0,
    ) -> np.ndarray:
        # Beta walk picks the next theta on the search-disk plane; the
        # beam_footprint_radius does not influence target selection.
        del beam_footprint_radius_m
        if self.center_start and self._first:
            # First dwell at the disk centre (peer estimate), then walk.
            self._first = False
            return peer_estimated_pos.copy()
        self._first = False
        sample = rng.betavariate(self.alpha, self.beta)
        i_theta = sample * 2.0 * math.pi
        self._theta = i_theta - (math.pi - self._theta)

        if self.radius_mode == "uniform_disk":
            # Uniform-by-area radius: r = R * sqrt(U). Targets are scattered
            # over the disk rather than confined to the perimeter, so chord
            # lengths shrink on average and the beam gets a dwell-like pause
            # near small-r targets before the next slew.
            r = search_radius_m * math.sqrt(rng.uniform(0.0, 1.0))
        else:
            r = search_radius_m

        k = peer_estimated_pos - my_pos
        i_vec, j_vec = _find_orthogonal_vectors(k)
        return (
            peer_estimated_pos
            + r * math.cos(self._theta) * i_vec
            + r * math.sin(self._theta) * j_vec
        )
