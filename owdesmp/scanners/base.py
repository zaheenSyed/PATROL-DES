"""Scanner interface.

A scanner picks the next 3D beam target point given the searcher's position,
the peer's estimated position, the search-disk radius (uncertainty extent),
and the beam footprint radius at peer range (which raster/spiral need for
stripe spacing and pitch).

DES-style on-demand interface: each call returns ONE next target. Stateful
scanners (Beta walk, raster cursor, spiral angle) maintain their state on
the instance. This differs from legacy/fso_sim_modules.py where the full
path is generated up-front via `generate_path(...)` and indexed by
`advance_point()`.
"""

from typing import Protocol

import numpy as np


class Scanner(Protocol):
    def next_target(
        self,
        rng,
        my_pos: np.ndarray,
        peer_estimated_pos: np.ndarray,
        search_radius_m: float,
        beam_footprint_radius_m: float = 0.0,
    ) -> np.ndarray: ...
