"""Raster (snake-pattern) scanner.

Ported from legacy/fso_sim_modules.py:370. The legacy code generates the full
snake path up-front in `generate_path(...)`; this DES version generates on
the first `next_target` call and indexes into the path on subsequent calls,
wrapping to the start once the path is exhausted.

Snake pattern: rows of horizontal sweeps inside the search disk, with each
row stepping vertically by 2 * beam_footprint_radius. Direction alternates
per row, and an extra link point is inserted between rows so the path is
continuous.
"""

import math
from typing import List

import numpy as np

from owdesmp.scanners.beta import _find_orthogonal_vectors


class RasterScanner:
    def __init__(self, step_along: float = 0.9, full_coverage: bool = False,
                 perimeter_start: bool = False,
                 center_lead_in: bool = False,
                 dense_rows: bool = False) -> None:
        # Fraction of beam_footprint_radius to step along the horizontal sweep.
        # 0.9 gives ~10% overlap between consecutive beam footprints.
        self.step_along = float(step_along)
        # When True, rotate the scan basis by a random angle so the snake starts
        # at a random point near the disk boundary (a random circumference
        # entry) and the centre is NOT prepended - removing the fixed centre
        # head-start the deterministic patterns otherwise get over the Beta walk.
        # The rotation angle is drawn per trial from the scanner rng. Off by
        # default so legacy results stand.
        self.perimeter_start = bool(perimeter_start)
        # When True, the scan begins from a RANDOM point on the search-disk
        # circumference (the same entry the Beta walk gets), dead-heads to the
        # disk centre, and only THEN runs the normal full-coverage snake. All
        # three scanners then pay an equal perimeter->centre lead-in, so the
        # deterministic patterns no longer get a free centred-target dwell over
        # the Beta walk. Off by default (legacy safe). Implies full_coverage so
        # the snake tiles the whole disk (>= 1 full pass).
        self.center_lead_in = bool(center_lead_in)
        # When True (full_coverage only), rows are spaced by the beam footprint
        # RADIUS (~50% overlap) and at least 3 rows are always placed, instead of
        # the default 2*beam_r spacing. Once beam_r >= search_r/2 the default
        # produces only two edge rows, which leaves the far-edge annulus of the
        # disk thinly covered and fails ~10-20% of near-edge targets; the denser
        # rows tile the disk reliably. Off by default so legacy/wide-offset
        # results are unchanged.
        self.dense_rows = bool(dense_rows)
        # When True, guarantee the rows cover the whole disk. The default starts
        # the first row at the disk edge (y = search_r, where x_lim = 0, a single
        # useless point) and steps by 2*beam_r; for wide beams this leaves the
        # lower disk unscanned and causes hard failures. With full_coverage the
        # rows are inset by beam_r so the beam covers the top/bottom edges, and a
        # beam wider than the disk collapses to a single centre dwell. Off by
        # default so legacy results are unchanged.
        self.full_coverage = bool(full_coverage)
        self._points: List[np.ndarray] = []
        self._idx: int = 0
        self._key = None  # cache key so we can rebuild if geometry changes

    def _build_path(self, rng, my_pos, peer_est, search_r, beam_r) -> None:
        k = peer_est - my_pos
        i_vec, j_vec = _find_orthogonal_vectors(k)

        if self.perimeter_start and rng is not None:
            # Rotate the (i, j) scan basis by a random angle so the snake's
            # first row (a near-boundary point) lands at a random circumference
            # angle rather than a fixed one.
            phi0 = rng.uniform(0.0, 2.0 * math.pi)
            c, s = math.cos(phi0), math.sin(phi0)
            i_vec, j_vec = (c * i_vec + s * j_vec, -s * i_vec + c * j_vec)

        stripe = 2.0 * beam_r
        delta = self.step_along * beam_r
        if stripe <= 0 or delta <= 0:
            # Degenerate: just one target at peer estimate.
            self._points = [peer_est.copy()]
            return

        if self.full_coverage or self.center_lead_in:
            if beam_r >= search_r:
                # The beam already covers the whole disk from its centre, so a
                # single dwell suffices for coverage. Return TWO distinct dwell
                # points (centre and a small in-disk offset) rather than one: a
                # single repeated target gives dt_slew == 0, and BEAM_MOVE_END
                # reschedules SCAN_TARGET_SELECT at the same clock, so the sim
                # spins at one timestamp without ever advancing time or firing
                # beacons (the legacy single-point 42-hour hang). The wide beam
                # covers the disk from the centre point regardless, so discovery
                # is unaffected.
                self._points = [peer_est.copy(),
                                peer_est + (0.25 * search_r) * i_vec]
                return
            # Inset the first/last rows by beam_r so the beam reaches the top
            # and bottom edges, and place enough rows (spacing <= 2*beam_r) that
            # the footprints tile the full disk. linspace keeps BOTH edge rows.
            top = search_r - beam_r
            if self.dense_rows:
                # Row spacing = beam_r (~50% overlap); >= 3 rows so a centre-ish
                # row always exists and the far-edge annulus is covered.
                n_rows = max(3, int(math.ceil((2.0 * top) / beam_r)) + 1)
            else:
                n_rows = max(2, int(math.ceil((2.0 * top) / stripe)) + 1)
            y_vals = np.linspace(top, -top, n_rows)
        else:
            y_vals = np.arange(search_r, -search_r - 1e-9, -stripe)
        pts = []
        direction = 1
        for idx, y in enumerate(y_vals):
            x_lim = math.sqrt(max(0.0, search_r * search_r - y * y))
            if x_lim == 0.0:
                # Degenerate edge row - a single point at (0, y_edge). Including
                # it is harmless and avoids "missing" the disk-edge sample.
                pts.append(peer_est + 0.0 * i_vec + y * j_vec)
                direction *= -1
                continue
            xs = np.arange(-x_lim, x_lim + 1e-9, delta)
            if direction == -1:
                xs = xs[::-1]
            last_x = 0.0
            for x in xs:
                pts.append(peer_est + x * i_vec + y * j_vec)
                last_x = x
            direction *= -1
            # Continuity link to next row, clamped so it lies inside the disk.
            if idx < len(y_vals) - 1:
                next_y = y_vals[idx + 1]
                next_x_lim = math.sqrt(max(0.0, search_r * search_r - next_y * next_y))
                if next_x_lim > 0:
                    link_x = max(-next_x_lim, min(last_x, next_x_lim))
                    pts.append(peer_est + link_x * i_vec + next_y * j_vec)
        if not pts:
            pts = [peer_est.copy()]
        # Guarantee path has >= 2 DISTINCT targets and includes peer_est.
        # If beam_footprint >= search_radius the loop above can produce a path
        # of one disk-edge point; the scanner would then return the same point
        # forever, current_pointing locks onto it, dt_slew collapses to 0, and
        # the simulator hits a zero-time-advance loop in SCAN_TARGET_SELECT /
        # BEAM_MOVE_END (observed as a 42-hour CPU hang). Prepending peer_est
        # gives the scanner two distinct targets to alternate between and the
        # wide beam covers the disk on the first dwell anyway.
        if not any(np.allclose(p, peer_est) for p in pts):
            # Legacy: prepend the centre so the path includes peer_est and has
            # >= 2 distinct targets. With perimeter_start we deliberately do NOT
            # start at the centre; only fall back to the centre prepend if the
            # path is degenerate (< 2 points), to keep the zero-slew hang guard.
            if not self.perimeter_start or len(pts) < 2:
                pts.insert(0, peer_est.copy())

        if self.center_lead_in:
            # Random circumference entry -> dead-head to centre -> then the
            # full-coverage snake built above. Prepend [entry, centre]; drop a
            # duplicate leading centre so we never emit two identical
            # consecutive targets (zero-slew hang guard).
            phi0 = rng.uniform(0.0, 2.0 * math.pi) if rng is not None else 0.0
            entry = peer_est + search_r * (math.cos(phi0) * i_vec
                                           + math.sin(phi0) * j_vec)
            if pts and np.allclose(pts[0], peer_est):
                pts = [entry] + pts
            else:
                pts = [entry, peer_est.copy()] + pts
        self._points = pts

    def next_target(
        self,
        rng,
        my_pos: np.ndarray,
        peer_estimated_pos: np.ndarray,
        search_radius_m: float,
        beam_footprint_radius_m: float = 0.0,
    ) -> np.ndarray:
        key = (
            tuple(np.round(my_pos, 6)),
            tuple(np.round(peer_estimated_pos, 6)),
            round(search_radius_m, 6),
            round(beam_footprint_radius_m, 6),
        )
        if key != self._key:
            self._build_path(rng, my_pos, peer_estimated_pos, search_radius_m, beam_footprint_radius_m)
            self._idx = 0
            self._key = key
        target = self._points[self._idx]
        self._idx = (self._idx + 1) % len(self._points)
        return target
