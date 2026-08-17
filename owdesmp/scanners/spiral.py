"""Archimedean spiral scanner.

Ported from legacy/fso_sim_modules.py:434. Path is r(theta) = b * theta with
b = beam_footprint_radius / pi so consecutive turns of the spiral are spaced
by ~beam_footprint_radius (no coverage gaps and minimal overlap).

Number of sampled points is chosen from beam coverage: ~theta_max steps where
theta_max = search_radius / b.
"""

import math
from typing import List

import numpy as np

from owdesmp.scanners.beta import _find_orthogonal_vectors


class SpiralScanner:
    def __init__(self, samples_per_turn: int = 60, full_coverage: bool = False,
                 perimeter_start: bool = False,
                 center_lead_in: bool = False,
                 dense_turns: bool = False) -> None:
        # How densely to sample along the spiral (per 2*pi of theta).
        self.samples_per_turn = int(samples_per_turn)
        # When True, the spiral starts at a RANDOM point on the search-disk
        # circumference and winds INWARD to the centre (>= 1 full revolution),
        # instead of starting at the centre and winding out. This removes the
        # fixed centre head-start the deterministic patterns otherwise get over
        # the Beta walk (which starts on the perimeter). The entry angle is drawn
        # per trial from the scanner rng. Off by default so legacy results stand.
        self.perimeter_start = bool(perimeter_start)
        # When True, the scan begins from a RANDOM point on the search-disk
        # circumference (the same kind of entry the Beta walk gets), transits
        # ("dead-heads") to the disk centre, and only THEN runs the normal
        # centre-out spiral coverage (>= 1 full revolution). This removes the
        # free centred-target dwell the deterministic patterns otherwise enjoy
        # over the Beta walk: all three now pay a perimeter->centre lead-in, and
        # the spiral/raster ordering is then set by outward-coverage efficiency
        # rather than by who happens to sit on the centre first. The entry angle
        # is drawn per trial from the scanner rng. Off by default (legacy safe).
        # Forces the centre-out (not inward) spiral so the lead-in is a clean
        # dead-head; implies full-coverage pitch so >= 1 revolution is scanned.
        self.center_lead_in = bool(center_lead_in)
        # When True, guarantee at least one FULL revolution within the search
        # disk even for wide beams. The default pitch b = beam_r/pi makes
        # theta_max = pi*search_r/beam_r, which drops below 2*pi (less than one
        # turn) once beam_r > search_r/2 - leaving an angular wedge of the disk
        # unscanned and causing hard discovery failures at wide divergence. With
        # full_coverage the pitch is capped at search_r/(2*pi) so theta_max >=
        # 2*pi, restoring complete angular coverage. Off by default so legacy
        # results are unchanged.
        self.full_coverage = bool(full_coverage)
        # When True, use a coverage pitch b = beam_r/(2*pi) so consecutive turns
        # are spaced by the beam footprint RADIUS (~50% overlap) rather than the
        # default 2*beam_r. The default leaves ~10-15% of targets in the angular
        # gaps between turns once beam_r ~ search_r (only ~1 turn fits); the dense
        # pitch tiles the disk reliably. Doubles the path length (spiral stays the
        # slowest scanner) but removes the coverage failures. Off by default so
        # legacy/wide-offset results are unchanged.
        self.dense_turns = bool(dense_turns)
        self._points: List[np.ndarray] = []
        self._idx: int = 0
        self._key = None

    def _build_path(self, rng, my_pos, peer_est, search_r, beam_r) -> None:
        i_vec, j_vec = _find_orthogonal_vectors(peer_est - my_pos)

        if beam_r <= 0:
            self._points = [peer_est.copy()]
            return
        b = beam_r / math.pi
        if self.dense_turns:
            # Overlapping turns (spacing = beam_r) for reliable tiling.
            b = beam_r / (2.0 * math.pi)
        if (self.full_coverage or self.perimeter_start
                or self.center_lead_in or self.dense_turns) and search_r > 0:
            # Cap pitch so the spiral completes >= 1 full turn inside search_r.
            b = min(b, search_r / (2.0 * math.pi))
        theta_max = search_r / b
        n = max(2, int(theta_max / (2 * math.pi) * self.samples_per_turn) + 1)
        thetas = np.linspace(0.0, theta_max, n)
        if self.center_lead_in:
            # Random circumference entry, dead-head to centre, THEN centre-out
            # spiral coverage. The centre-out spiral already starts at r=0
            # (theta=0), so only the perimeter entry point is prepended (the
            # dead-head perimeter->centre is the slew from entry to that first
            # centre point). This overrides perimeter_start's inward winding.
            #
            # Use a robust coverage pitch: radial spacing between consecutive
            # turns (2*pi*b) equals the beam footprint RADIUS, i.e. b =
            # beam_r/(2*pi). Consecutive footprints then overlap by ~50%, so the
            # spiral genuinely tiles the disk instead of leaving angular wedges
            # (the default b = beam_r/pi spaces turns by 2*beam_r and, once the
            # full-coverage cap forces a single turn for beam_r ~ search_r,
            # leaves gaps that fail ~10% of narrow-beam trials). For a beam wider
            # than the disk the cap keeps >= 1 full revolution.
            b_cov = beam_r / (2.0 * math.pi)
            if search_r > 0:
                b_cov = min(b_cov, search_r / (2.0 * math.pi))
            theta_max_cov = search_r / b_cov
            n_cov = max(2, int(theta_max_cov / (2 * math.pi)
                               * self.samples_per_turn) + 1)
            thetas_cov = np.linspace(0.0, theta_max_cov, n_cov)
            phi0 = rng.uniform(0.0, 2.0 * math.pi) if rng is not None else 0.0
            entry = peer_est + search_r * (math.cos(phi0) * i_vec
                                           + math.sin(phi0) * j_vec)
            pts = [entry]
            pts += [
                peer_est + (b_cov * t) * (math.cos(t) * i_vec
                                          + math.sin(t) * j_vec)
                for t in thetas_cov
            ]
            self._points = pts
            return
        if self.perimeter_start:
            # Enter at a random circumference angle and wind inward to centre:
            # reverse theta (theta_max -> 0) and offset the angle by phi0.
            phi0 = rng.uniform(0.0, 2.0 * math.pi) if rng is not None else 0.0
            pts = [
                peer_est + (b * t) * (math.cos(t + phi0) * i_vec
                                      + math.sin(t + phi0) * j_vec)
                for t in thetas[::-1]
            ]
        else:
            pts = [
                peer_est + (b * t) * (math.cos(t) * i_vec + math.sin(t) * j_vec)
                for t in thetas
            ]
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
