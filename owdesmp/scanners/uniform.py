"""Uniform random perimeter scanner.

Memoryless: each call samples a fresh uniform angle in [0, 2*pi) and returns a
point on the search-disk perimeter. Unlike the Beta walk, consecutive targets
are independent. Useful as a baseline against which structured scans (Beta,
Raster, Spiral) are compared.
"""

import math

import numpy as np

from owdesmp.scanners.beta import _find_orthogonal_vectors


class UniformScanner:
    def next_target(
        self,
        rng,
        my_pos: np.ndarray,
        peer_estimated_pos: np.ndarray,
        search_radius_m: float,
        beam_footprint_radius_m: float = 0.0,
    ) -> np.ndarray:
        del beam_footprint_radius_m
        theta = rng.uniform(0.0, 2.0 * math.pi)
        i_vec, j_vec = _find_orthogonal_vectors(peer_estimated_pos - my_pos)
        return (
            peer_estimated_pos
            + search_radius_m * math.cos(theta) * i_vec
            + search_radius_m * math.sin(theta) * j_vec
        )
