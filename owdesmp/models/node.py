"""Node model.

Slice 1 introduced the protocol-only fields. Slice 2 adds optional geometry
fields (mobility, scanner, optical_terminal, current_pointing). When
mobility_model and optical_terminal are absent, the simulator runs in
"always aligned" mode and slice-1 parity is preserved.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Node:
    # Protocol (slice 1) ------------------------------------------------------
    node_id: str
    start_time: float
    beacon_period: float
    beacon_phase: float       # initial phase offset in [0, beacon_period)
    packet_duration: float    # transmission_time + decode time (sender->receiver delay)
    processing_delay: float   # post-RX delay before this node can transmit next
    handshake_timeout: float

    # Link maintenance (slice 5). Both default to 0 -> maintenance disabled,
    # link persists forever once established. Scenarios that opt in get
    # periodic KeepAlive transmissions and a link_timeout-based health gate.
    keepalive_period: float = 0.0
    link_timeout: float = 0.0

    # Geometry (slice 2, optional) -------------------------------------------
    # Position is derived from mobility_model.position_at(t); never cached here.
    mobility_model: Optional[Any] = None
    current_pointing: Optional[np.ndarray] = None  # 3D unit vector; what alignment uses
    scanner: Optional[Any] = None
    optical_terminal: Optional[Dict[str, float]] = None
    # Per-peer estimated positions and uncertainty radius for scanner targeting.
    # Shape: {peer_id: {"estimated_position": np.ndarray, "uncertainty_radius_m": float}}
    neighbor_estimates: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Vibration / jitter (Phase A vibration port, optional) -------------------
    # When pointing_jitter is set (anything other than NullPointingJitter), the
    # simulator treats `nominal_pointing` as the scanner/slew-set "intended"
    # direction and computes `current_pointing = R(jitter(t)) @ nominal_pointing`
    # at every mobility tick. When pointing_jitter is None / Null, the legacy
    # behavior holds and `nominal_pointing` is unused.
    pointing_jitter: Optional[Any] = None
    nominal_pointing: Optional[np.ndarray] = None

    # Beam tracker (Phase C, optional) ---------------------------------------
    # Closed-loop pointing correction during CONNECTED state. The simulator
    # schedules TRACKER_TICK events at beam_tracker_tick_s, computes an
    # observation, and applies the returned (daz, del) correction to
    # nominal_pointing. NullBeamTracker is a no-op.
    beam_tracker: Optional[Any] = None
    beam_tracker_tick_s: float = 0.001
    # Last rx_power sample (per direction). Initialized lazily on first
    # TRACKER_TICK; used to compute the gradient proxy fed to PID/Kalman.
    last_rx_power_dBm: float = -999.0

    # ATP dual-beam (Phase R0, optional) -------------------------------------
    # When optical_terminal_maintenance is set, the simulator schedules a
    # BEAM_MODE_SWITCH event at (LINK_ESTABLISHED + mode_switch_settling_s)
    # that swaps optical_terminal -> optical_terminal_maintenance. The
    # current `optical_terminal` field always points at the *active* mode's
    # config; switching is a destructive swap (slot pattern), with the
    # original wide-beam config preserved in `optical_terminal_acquisition`
    # so the simulator can revert on LINK_LOST if asked to.
    optical_terminal_acquisition: Optional[Dict[str, float]] = None
    optical_terminal_maintenance: Optional[Dict[str, float]] = None
    in_maintenance_mode: bool = False
    # During (clock < mode_switch_active_until) the optics are settling and
    # any PACKET_TX on this node drops. 0 = not switching.
    mode_switch_active_until: float = 0.0

    # Runtime state ----------------------------------------------------------
    ready_time: float = 0.0
    active: bool = True

    @property
    def has_geometry(self) -> bool:
        return self.mobility_model is not None and self.optical_terminal is not None

    def position_at(self, t: float) -> np.ndarray:
        return self.mobility_model.position_at(t)
