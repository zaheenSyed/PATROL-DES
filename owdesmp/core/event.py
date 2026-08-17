"""Event types and Event dataclass for the OWDES-MP discrete-event simulator."""

from dataclasses import dataclass
from typing import Any, Optional

# Event type constants. Slice 1 introduced the protocol-only subset; slice 2 adds
# the geometry/mobility/scanner subset. Remaining types arrive with later slices.
SIM_START = "SIM_START"
SIM_END = "SIM_END"
NODE_START = "NODE_START"
BEACON_TICK = "BEACON_TICK"
PACKET_TX = "PACKET_TX"
PACKET_RX = "PACKET_RX"
LINK_ESTABLISHED = "LINK_ESTABLISHED"

# Slice 2: geometry layer
NODE_MOBILITY_UPDATE = "NODE_MOBILITY_UPDATE"
SCAN_TARGET_SELECT = "SCAN_TARGET_SELECT"
BEAM_MOVE_SAMPLE = "BEAM_MOVE_SAMPLE"  # mid-slew sample for smooth pointing
BEAM_MOVE_END = "BEAM_MOVE_END"
ALIGNMENT_START = "ALIGNMENT_START"
ALIGNMENT_END = "ALIGNMENT_END"

# Slice 5: link maintenance
LINK_KEEPALIVE_TX = "LINK_KEEPALIVE_TX"
LINK_HEALTH_CHECK = "LINK_HEALTH_CHECK"
LINK_LOST = "LINK_LOST"
LINK_REACQUIRE_START = "LINK_REACQUIRE_START"

# Phase A: data plane (TCP flow + transfer session)
DATA_SESSION_START = "DATA_SESSION_START"
DATA_TICK = "DATA_TICK"
DATA_SESSION_END = "DATA_SESSION_END"

# Phase C: beam-tracking control loop (PID / Kalman / RL). Scheduled at the
# tracker's configured tick interval while a link is CONNECTED; cancelled
# implicitly when LINK_LOST resets the pair-FSM.
TRACKER_TICK = "TRACKER_TICK"

# Phase R0/R1: ATP (acquisition-tracking-pointing) mode switch. Real dual-beam
# FSO terminals carry a wide-FOV acquisition beam + a narrow-FOV fine-tracking
# beam. A node with optical_terminal_maintenance configured swaps to it at
# (LINK_ESTABLISHED + mode_switch_settling_s); during the settling window the
# optics are unstable and the simulator drops any PACKET_TX on the affected node.
BEAM_MODE_SWITCH = "BEAM_MODE_SWITCH"


@dataclass(frozen=True)
class Event:
    time: float
    seq: int
    type: str
    payload: Optional[Any] = None
