"""OWDES-MP discrete-event simulator main loop.

Slices implemented:
  - Slice 1: DES core, periodic BEACON_TICK, PACKET_TX/RX, pairwise FSM,
             first_link_discovered termination, per-node ready gate.
  - Slice 2: mobility, scanners, geometry engine. Pair alignment is tracked
             in `alignment_active[frozenset({a,b})]`. PACKET_TX is gated on
             current alignment; if a pair is not aligned at TX time the packet
             is dropped (no PACKET_RX scheduled). Alignment is recomputed at
             NODE_START and after every BEAM_MOVE_END.

Default behavior: scenarios with no mobility/optical_terminal config keep the
"always aligned" semantics from slice 1 and the slice-1 parity test continues
to pass.

Known limitation (slice 2): the alignment gate is applied at TX time only.
Mid-flight alignment loss does not currently fail an in-flight packet. The
channel engine in slice 3 will model mid-flight failures explicitly.
"""

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from owdesmp.core.event import (
    ALIGNMENT_END,
    ALIGNMENT_START,
    BEACON_TICK,
    BEAM_MODE_SWITCH,
    BEAM_MOVE_END,
    BEAM_MOVE_SAMPLE,
    DATA_SESSION_END,
    DATA_SESSION_START,
    DATA_TICK,
    LINK_ESTABLISHED,
    LINK_HEALTH_CHECK,
    LINK_KEEPALIVE_TX,
    LINK_LOST,
    NODE_MOBILITY_UPDATE,
    NODE_START,
    PACKET_RX,
    PACKET_TX,
    SCAN_TARGET_SELECT,
    SIM_END,
    SIM_START,
    TRACKER_TICK,
)
from owdesmp.core.event_queue import EventQueue
from owdesmp.core.random_manager import RandomManager
from owdesmp.core.termination import make_termination
from owdesmp.engines.channel import ChannelModel
from owdesmp.engines.data_transfer import DataTransferSession
from owdesmp.engines.tcp_flow import TCPFlowReno, per_from_received_power, shannon_capacity_bps
from owdesmp.engines.geometry import (
    angle_between,
    direction_to,
    lerp_unit,
    mutual_alignment,
    propagation_delay_s,
    slew_time,
    unit,
)
from owdesmp.engines.protocol import INIT, PairwiseProtocol
from owdesmp.engines.beam_tracker import (
    NullBeamTracker,
    apply_az_el_correction,
    build_beam_tracker,
)
from owdesmp.engines.pointing_jitter import (
    NullPointingJitter,
    build_pointing_jitter,
    jitter_rotation_matrix,
)
from owdesmp.mobility.circular_orbit import CircularOrbit
from owdesmp.mobility.constant_velocity import ConstantVelocity
from owdesmp.mobility.fixed import FixedPosition
from owdesmp.mobility.vibration import build_vibration_mobility
from owdesmp.mobility.waypoint import Waypoint
from owdesmp.models.network_graph import NetworkState
from owdesmp.models.node import Node
from owdesmp.models.packet import KEEPALIVE
from owdesmp.scanners.beta import BetaScanner
from owdesmp.scanners.raster import RasterScanner
from owdesmp.scanners.spiral import SpiralScanner
from owdesmp.scanners.uniform import UniformScanner


def _deg_to_rad(deg: float) -> float:
    return float(deg) * np.pi / 180.0


def _build_mobility(cfg: dict):
    if cfg is None:
        return None
    model = cfg.get("model", "fixed")
    if model == "fixed":
        return FixedPosition(cfg["initial_m"])
    if model == "constant_velocity":
        return ConstantVelocity(cfg["initial_m"], cfg["velocity_mps"])
    if model == "waypoint":
        return Waypoint(cfg["waypoints"])
    if model == "circular_orbit":
        return CircularOrbit(
            center_m=cfg.get("center_m", [0.0, 0.0, 0.0]),
            radius_m=cfg["radius_m"],
            period_s=cfg["period_s"],
            plane_normal=cfg.get("plane_normal", [0.0, 0.0, 1.0]),
            phase_rad=cfg.get("phase_rad", 0.0),
        )
    if model in ("vibration_csv", "vibration_lstm_nn"):
        return build_vibration_mobility(cfg)
    raise ValueError(f"Unknown mobility model: {model!r}")


def _build_scanner(cfg: dict):
    if cfg is None:
        return None
    type_ = cfg.get("type", "beta")
    if type_ == "beta":
        return BetaScanner(
            alpha=cfg.get("alpha", 0.41485),
            beta=cfg.get("beta", 0.41485),
            radius_mode=cfg.get("radius_mode", "perimeter"),
            center_start=cfg.get("center_start", False),
        )
    if type_ == "raster":
        return RasterScanner(
            step_along=cfg.get("step_along", 0.9),
            full_coverage=cfg.get("full_coverage", False),
            perimeter_start=cfg.get("perimeter_start", False),
            center_lead_in=cfg.get("center_lead_in", False),
            dense_rows=cfg.get("dense_rows", False),
        )
    if type_ == "spiral":
        return SpiralScanner(
            samples_per_turn=cfg.get("samples_per_turn", 60),
            full_coverage=cfg.get("full_coverage", False),
            perimeter_start=cfg.get("perimeter_start", False),
            center_lead_in=cfg.get("center_lead_in", False),
            dense_turns=cfg.get("dense_turns", False),
        )
    if type_ == "uniform":
        return UniformScanner()
    raise ValueError(
        f"Unknown scanner type: {type_!r} (helical and trajectory_from_csv "
        f"are not yet implemented)"
    )


def _build_optical_terminal(cfg: dict):
    if cfg is None:
        return None
    out = dict(cfg)
    if "beam_divergence_deg" in cfg and "beam_divergence_rad" not in cfg:
        out["beam_divergence_rad"] = _deg_to_rad(cfg["beam_divergence_deg"])
    if "max_angular_speed_deg_s" in cfg and "max_angular_speed_rad_s" not in cfg:
        out["max_angular_speed_rad_s"] = _deg_to_rad(cfg["max_angular_speed_deg_s"])
    # pointing_jitter is consumed by _build_node, not stored on the terminal dict
    out.pop("pointing_jitter", None)
    out.pop("beam_tracker", None)
    return out


class Simulator:
    def __init__(self, scenario: dict, run_id: int = 0, seed: Optional[int] = None) -> None:
        self.scenario = scenario
        self.run_id = run_id
        base_seed = seed if seed is not None else scenario.get("random_seed", 0)
        self.rng = RandomManager(base_seed)

        self.queue = EventQueue()
        self.clock: float = 0.0
        self.duration: float = float(scenario["simulation"]["duration_s"])

        # Build nodes
        self.nodes: Dict[str, Node] = {}
        for n_cfg in scenario["nodes"]:
            self.nodes[n_cfg["id"]] = self._build_node(n_cfg)

        # Wire neighbor estimates from scenario["neighbor_knowledge"] (slice 2 schema).
        self._wire_neighbor_estimates(scenario.get("neighbor_knowledge", {}))

        ids = list(self.nodes.keys())
        self.protocols: Dict[Tuple[str, str], PairwiseProtocol] = {
            (i, j): PairwiseProtocol() for i in ids for j in ids if i != j
        }
        self.peers: Dict[str, List[str]] = {i: [j for j in ids if j != i] for i in ids}
        self.linked_pairs: Set[FrozenSet[str]] = set()

        # Round-robin beam-scheduling cursor per node. Each BEACON_TICK that
        # successfully transmits advances this so multi-peer nodes can't
        # starve their lower-index neighbors.
        self._next_peer_idx: Dict[str, int] = {i: 0 for i in ids}

        # Pairwise alignment state. A missing key means "not yet started" - treated
        # as not-aligned. Once both nodes of a pair have started (NODE_START fired),
        # the key is present with a boolean.
        self.alignment_active: Dict[FrozenSet[str], bool] = {}
        # Node IDs whose NODE_START has fired (used for incremental alignment init).
        self._started_nodes: Set[str] = set()

        # Slice 5 link-maintenance state. last_keepalive_rx[pair] records the
        # most recent moment either endpoint of `pair` received any packet
        # from the other (initial value: LINK_ESTABLISHED time). The
        # LINK_HEALTH_CHECK event compares (clock - last_keepalive_rx) to the
        # max link_timeout across endpoints and fires LINK_LOST if exceeded.
        self.last_keepalive_rx: Dict[FrozenSet[str], float] = {}

        self.mobility_update_period: float = float(
            scenario.get("simulation", {}).get("mobility_update_period_s", 0.001)
        )
        # Smooth-beam slew sampling. When > 0, slews are subdivided into
        # BEAM_MOVE_SAMPLE events at this cadence and current_pointing is
        # linearly interpolated, mirroring legacy fso_sim_modules.py:Node.update.
        # When 0, the beam JUMPS from old to new direction at slew completion
        # (cheaper but misses alignment crossings during the slew - the
        # legacy/OWDES-MP shape mismatch). Recommended: 0.0001 (100 us) for
        # paper-quality scanner-comparison results.
        self.beam_sample_interval: float = float(
            scenario.get("simulation", {}).get("beam_sample_interval_s", 0.0001)
        )
        # Per-node slew context (when smooth sampling is active).
        # {node_id: {"start_dir", "target_dir", "start_time", "duration"}}
        self._slew_state: Dict[str, dict] = {}
        # When True (and the pair is not yet linked), losing alignment resets
        # the pair-FSM back to INIT. Matches the legacy fso_sim_modules.py
        # `current_dwell_time = 0` reset and is necessary to reproduce the
        # right-rising side of the omega U-shape (where high omega causes
        # alignment crossings shorter than the 3-way handshake duration).
        self.reset_fsm_on_alignment_loss: bool = bool(
            scenario.get("simulation", {}).get("reset_fsm_on_alignment_loss", True)
        )
        # Phase R0 ATP dual-beam: how long the optics take to settle after a
        # BEAM_MODE_SWITCH. Real Mynaric/Tesat ATP terminals quote 10-50 ms;
        # we default to 20 ms. During settling, the affected node's PACKET_TX
        # events drop and increment packets_dropped_mode_switch.
        self.mode_switch_settling_s: float = float(
            scenario.get("simulation", {}).get("mode_switch_settling_s", 0.02)
        )
        # Phase R0 (switch_with_fallback): if True, LINK_LOST on a node that
        # is currently in maintenance mode schedules a BEAM_MODE_SWITCH back
        # to acquisition (the wide beam) so re-discovery is possible. Off by
        # default - the naive switch_on_link baseline doesn't get this.
        self.revert_to_acquisition_on_link_loss: bool = bool(
            scenario.get("simulation", {}).get("revert_to_acquisition_on_link_loss", False)
        )

        term_cfg = scenario.get("termination_condition", {"type": "first_link_discovered"})
        self.termination = make_termination(term_cfg)
        self.network_state = NetworkState(ids)

        # Channel engine. Default mode is "disabled" so scenarios without a
        # `channel` section preserve slice-1/slice-2 always-succeeds behavior.
        ch_cfg = dict(scenario.get("channel", {"mode": "disabled"}))
        self.channel = ChannelModel(**ch_cfg)

        # Phase A: optional data-transfer session. Configured via top-level
        # `data_session`. When absent, no DATA_TICK events fire and behavior
        # is identical to slice 6.
        ds_cfg = scenario.get("data_session")
        self.data_session: Optional[DataTransferSession] = None
        self.data_session_cfg: Optional[dict] = None
        if ds_cfg is not None:
            self.data_session_cfg = dict(ds_cfg)
            tcp_kwargs = dict(ds_cfg.get("tcp", {}))
            self.data_session = DataTransferSession(
                src=ds_cfg["src"],
                dst=ds_cfg["dst"],
                bytes_to_transfer=float(ds_cfg["bytes_to_transfer"]),
                tick_interval_s=float(ds_cfg.get("tick_interval_s", 0.01)),
                tcp=TCPFlowReno(**tcp_kwargs),
            )

        self.result: dict = {
            "run_id": run_id,
            "success": False,
            "first_link_time": None,
            "events_processed": 0,
            "duration_reached": False,
            "geometry_enabled": any(n.has_geometry for n in self.nodes.values()),
            "channel_mode": self.channel.mode,
            "packets_dropped_alignment": 0,
            "packets_dropped_channel": 0,
            "packets_dropped_mode_switch": 0,
            "mode_switches": 0,
            # Slice 4 multi-node metrics
            "termination_type": self.termination.name,
            "pairwise_discovery_matrix": {},  # frozenset({a,b}) -> time
            "n_links_at_end": 0,
            "n_components_at_end": len(ids),
            # Slice 5 maintenance metrics
            "links_lost": 0,
            "keepalives_sent": 0,
            "keepalives_received": 0,
        }

    # ---------------------------------------------------------------- node build

    def _build_node(self, cfg: dict) -> Node:
        proto = cfg["protocol"]
        period = float(proto["beacon_period_s"])
        phase_cfg = proto.get("beacon_phase_s", "random")
        if phase_cfg == "random":
            phase = self.rng.stream(f"phase:{cfg['id']}").uniform(0.0, period)
        else:
            phase = float(phase_cfg)

        mobility = _build_mobility(cfg.get("position"))
        scanner = _build_scanner(cfg.get("scanner"))
        terminal_cfg = cfg.get("optical_terminal") or {}
        terminal = _build_optical_terminal(terminal_cfg)
        # Phase R0: optional dual-beam ATP. If `optical_terminal_maintenance`
        # is present, the simulator schedules a BEAM_MODE_SWITCH at
        # (LINK_ESTABLISHED + simulation.mode_switch_settling_s) that swaps
        # this terminal in. The original acquisition terminal is preserved on
        # the Node so the test/inspection code can read both configs.
        maintenance_cfg = cfg.get("optical_terminal_maintenance")
        maintenance_terminal = (
            _build_optical_terminal(maintenance_cfg) if maintenance_cfg is not None else None
        )

        # pointing_jitter lives under optical_terminal in scenario JSON but
        # becomes a Node-level field. Missing/None config yields a no-op jitter
        # so the simulator can call sample_rad unconditionally.
        jitter_cfg = terminal_cfg.get("pointing_jitter") if terminal_cfg else None
        pointing_jitter = build_pointing_jitter(jitter_cfg) if jitter_cfg else NullPointingJitter()

        # beam_tracker also lives under optical_terminal in scenario JSON.
        tracker_cfg = terminal_cfg.get("beam_tracker") if terminal_cfg else None
        beam_tracker = build_beam_tracker(tracker_cfg) if tracker_cfg else NullBeamTracker()
        tracker_tick_s = float(tracker_cfg.get("tick_interval_s", 0.001)) if tracker_cfg else 0.001

        initial_pointing = None
        if cfg.get("initial_pointing") is not None:
            initial_pointing = unit(np.asarray(cfg["initial_pointing"], dtype=float))

        return Node(
            node_id=cfg["id"],
            start_time=float(cfg.get("start_time_s", 0.0)),
            beacon_period=period,
            beacon_phase=phase,
            packet_duration=float(proto["packet_duration_s"]),
            processing_delay=float(proto["processing_delay_s"]),
            handshake_timeout=float(proto.get("handshake_timeout_s", 0.02)),
            keepalive_period=float(proto.get("keepalive_period_s", 0.0)),
            link_timeout=float(proto.get("link_timeout_s", 0.0)),
            mobility_model=mobility,
            scanner=scanner,
            optical_terminal=terminal,
            current_pointing=initial_pointing,
            pointing_jitter=pointing_jitter,
            nominal_pointing=initial_pointing.copy() if initial_pointing is not None else None,
            beam_tracker=beam_tracker,
            beam_tracker_tick_s=tracker_tick_s,
            optical_terminal_acquisition=terminal if maintenance_terminal is not None else None,
            optical_terminal_maintenance=maintenance_terminal,
        )

    def _wire_neighbor_estimates(self, nk_cfg: dict) -> None:
        for entry in nk_cfg.get("pairs", []):
            src = entry["src"]
            dst = entry["dst"]
            if src not in self.nodes:
                continue
            self.nodes[src].neighbor_estimates[dst] = {
                "estimated_position": np.asarray(entry["estimated_position_m"], dtype=float),
                "uncertainty_radius_m": float(entry.get("uncertainty_radius_m", 0.0)),
            }

    # ------------------------------------------------------------------ run loop

    def _bootstrap(self) -> None:
        """Push the bootstrap events (SIM_START, per-node NODE_START, SIM_END)
        into the queue. Idempotent: only fires once per simulator instance.
        Both run() and the RL step API call this before consuming events."""
        if getattr(self, "_bootstrapped", False):
            return
        self.queue.push(0.0, SIM_START, None)
        for node in self.nodes.values():
            self.queue.push(node.start_time, NODE_START, {"node": node.node_id})
        self.queue.push(self.duration, SIM_END, None)
        self._bootstrapped = True

    def _dispatch(self, ev) -> bool:
        """Process one event. Returns True if the run loop should stop.

        Shared by run() and advance_until_event(). Returns True when:
          - SIM_END is reached
          - LINK_ESTABLISHED triggers a true termination_condition
        Otherwise the caller continues popping events.
        """
        self.clock = ev.time
        self.result["events_processed"] += 1
        t = ev.type
        if t == SIM_START:
            return False
        if t == SIM_END:
            self.result["duration_reached"] = True
            return True
        if t == NODE_START:
            self._on_node_start(ev)
        elif t == BEACON_TICK:
            self._on_beacon_tick(ev)
        elif t == PACKET_TX:
            self._on_packet_tx(ev)
        elif t == PACKET_RX:
            self._on_packet_rx(ev)
        elif t == SCAN_TARGET_SELECT:
            self._on_scan_target_select(ev)
        elif t == BEAM_MOVE_SAMPLE:
            self._on_beam_move_sample(ev)
        elif t == BEAM_MOVE_END:
            self._on_beam_move_end(ev)
        elif t == NODE_MOBILITY_UPDATE:
            self._on_node_mobility_update(ev)
        elif t == ALIGNMENT_START or t == ALIGNMENT_END:
            return False
        elif t == LINK_KEEPALIVE_TX:
            self._on_link_keepalive_tx(ev)
        elif t == LINK_HEALTH_CHECK:
            self._on_link_health_check(ev)
        elif t == LINK_LOST:
            self._on_link_lost(ev)
        elif t == TRACKER_TICK:
            self._on_tracker_tick(ev)
        elif t == DATA_TICK:
            self._on_data_tick(ev)
        elif t == DATA_SESSION_START or t == DATA_SESSION_END:
            return False
        elif t == BEAM_MODE_SWITCH:
            self._on_beam_mode_switch(ev)
        elif t == LINK_ESTABLISHED:
            self._on_link_established(ev)
            if self.termination.check(self.network_state, self.clock):
                self.result["success"] = True
                self.result["termination_time"] = self.clock
                return True
        return False

    def _finalize(self) -> dict:
        """Post-loop cleanup: pin discovery / termination times, summarize the
        data session (calling its outstanding-outage finalize first), populate
        graph statistics. Shared by run() and the Gym env's close()."""
        if getattr(self, "_finalized", False):
            return self.result
        if self.result["first_link_time"] is None:
            self.result["first_link_time"] = self.duration
        if "termination_time" not in self.result:
            self.result["termination_time"] = self.duration
        self.result["n_links_at_end"] = self.network_state.graph.number_of_edges()
        self.result["n_components_at_end"] = self.network_state.number_of_components()
        if self.data_session is not None:
            self.data_session.finalize(self.clock)
            self.result["data_session"] = self.data_session.summary()
        self._finalized = True
        return self.result

    def run(self) -> dict:
        self._bootstrap()
        while self.queue:
            ev = self.queue.pop()
            if ev.time > self.duration:
                self.result["duration_reached"] = True
                break
            if self._dispatch(ev):
                break
        return self._finalize()

    # ---------------------------------------------------------- RL step API

    def advance_until_event(self, event_types) -> Optional["Event"]:
        """Run the dispatch loop until an event matching `event_types` is
        popped. Returns the matching event WITHOUT dispatching it - the
        caller is expected to either dispatch it via _dispatch(ev) (after
        optionally mutating state, e.g., setting an action on a tracker)
        or to skip it.

        Returns None if:
          - the queue empties
          - an event past self.duration is popped (sim ends)
          - a non-matching event triggers a true termination_condition

        `event_types` may be a single string or any iterable of strings.

        On entry, _bootstrap() is invoked if it hasn't already been. This
        method does NOT call _finalize() - the caller is responsible for
        finalize() once it's done stepping.
        """
        if isinstance(event_types, str):
            target = {event_types}
        else:
            target = set(event_types)
        self._bootstrap()
        while self.queue:
            ev = self.queue.pop()
            if ev.time > self.duration:
                self.result["duration_reached"] = True
                return None
            if ev.type in target:
                return ev
            if self._dispatch(ev):
                return None
        return None

    # ----------------------------------------------------------------- handlers

    def _on_node_start(self, ev) -> None:
        node = self.nodes[ev.payload["node"]]
        self._started_nodes.add(node.node_id)

        # Initial pointing for geometry-enabled nodes.
        if node.has_geometry and node.current_pointing is None:
            # If a scanner exists, point at the peer estimate as the initial
            # "boresight"; the scanner will then move the beam off boresight.
            # If no scanner and no explicit pointing, default to peer estimate
            # (this is the static-aligned scenario case).
            peer_id = self._first_peer_with_estimate(node.node_id)
            if peer_id is not None:
                est = node.neighbor_estimates[peer_id]["estimated_position"]
                pos = node.position_at(self.clock)
                self._set_pointing(node, direction_to(pos, est))

        # Schedule first BEACON_TICK.
        first_tick = node.start_time + node.beacon_phase
        while first_tick < self.clock:
            first_tick += node.beacon_period
        self.queue.push(first_tick, BEACON_TICK, {"node": node.node_id})

        # Schedule first SCAN_TARGET_SELECT (only after pointing is initialized).
        if node.has_geometry and node.scanner is not None:
            self.queue.push(self.clock, SCAN_TARGET_SELECT, {"node": node.node_id})

        # Slice 5: nodes with non-fixed mobility OR with pointing jitter get
        # periodic NODE_MOBILITY_UPDATE ticks. A FixedPosition node with jitter
        # still needs the tick to re-sample jitter; without ticks the beam
        # rotation would freeze at sim_start jitter forever.
        has_jitter = node.pointing_jitter is not None and not isinstance(
            node.pointing_jitter, NullPointingJitter
        )
        needs_mobility_ticks = (
            node.has_geometry
            and node.mobility_model is not None
            and (not isinstance(node.mobility_model, FixedPosition) or has_jitter)
            and self.mobility_update_period > 0
        )
        if needs_mobility_ticks:
            self.queue.push(
                self.clock + self.mobility_update_period,
                NODE_MOBILITY_UPDATE,
                {"node": node.node_id},
            )

        # Recompute alignment vs already-started peers.
        for peer_id in self.peers[node.node_id]:
            if peer_id in self._started_nodes:
                self._refresh_alignment(node.node_id, peer_id)

    def _on_beacon_tick(self, ev) -> None:
        node = self.nodes[ev.payload["node"]]
        # Always reschedule next beacon, even if this one can't transmit.
        self.queue.push(self.clock + node.beacon_period, BEACON_TICK, {"node": node.node_id})

        if self.clock < node.ready_time:
            return

        # Round-robin over peers starting from the next-served index, picking
        # the first peer that (a) is not yet linked, (b) is currently aligned,
        # and (c) has a pending packet to send per its pair-FSM state.
        peers = self.peers[node.node_id]
        n_peers = len(peers)
        if n_peers == 0:
            return
        start = self._next_peer_idx[node.node_id]
        for offset in range(n_peers):
            idx = (start + offset) % n_peers
            peer_id = peers[idx]
            if frozenset({node.node_id, peer_id}) in self.linked_pairs:
                continue
            if not self._pair_aligned(node.node_id, peer_id):
                continue
            protocol = self.protocols[(node.node_id, peer_id)]
            ptype = protocol.next_tx_type()
            if ptype is None:
                continue
            packet = {
                "src": node.node_id,
                "dst": peer_id,
                "type": ptype,
                "tx_time": self.clock,
            }
            self.queue.push(self.clock, PACKET_TX, packet)
            self._next_peer_idx[node.node_id] = (idx + 1) % n_peers
            return

    def _on_packet_tx(self, ev) -> None:
        packet = ev.payload
        src = packet["src"]
        dst = packet["dst"]
        # Phase R0 ATP mode-switch gate. If either endpoint's optics are
        # settling between WIDE and NARROW the packet drops. This includes
        # KEEPALIVEs and data-plane segments, which is the realistic
        # behaviour - the optical path is literally unstable during the swap.
        src_node = self.nodes[src]
        dst_node = self.nodes[dst]
        if (
            self.clock < src_node.mode_switch_active_until
            or self.clock < dst_node.mode_switch_active_until
        ):
            self.result["packets_dropped_mode_switch"] += 1
            return
        # Alignment gate: if both nodes have geometry, the pair must be aligned at TX time.
        # If either has no geometry, fall back to "always aligned" (slice 1 semantics).
        if not self._pair_aligned(src, dst):
            self.result["packets_dropped_alignment"] += 1
            return
        # Channel gate: in disabled mode this is a no-op; in threshold/stochastic
        # the packet fails if received power is below sensitivity.
        rng = self.rng.stream("channel") if self.channel.mode == "stochastic" else None
        if not self.channel.packet_succeeds(
            self.nodes[src], self.nodes[dst], self.clock, rng
        ):
            self.result["packets_dropped_channel"] += 1
            return
        src_node = self.nodes[src]
        dst_node = self.nodes[dst]
        # Speed-of-light propagation delay. Negligible at lab/UAV ranges,
        # ~3.3 ms one-way at 1000 km satellite ISL distances.
        prop_delay = propagation_delay_s(src_node, dst_node, self.clock)
        rx_time = self.clock + src_node.packet_duration + prop_delay
        self.queue.push(rx_time, PACKET_RX, packet)

    def _on_packet_rx(self, ev) -> None:
        packet = ev.payload
        src = packet["src"]
        dst = packet["dst"]
        ptype = packet["type"]
        pair = frozenset({src, dst})

        # KEEPALIVE arriving on a still-linked pair refreshes the health timer
        # without touching the handshake FSM.
        if ptype == KEEPALIVE:
            if pair in self.linked_pairs:
                self.last_keepalive_rx[pair] = self.clock
                self.result["keepalives_received"] += 1
            return

        if pair in self.linked_pairs:
            return

        protocol = self.protocols[(dst, src)]
        prev_state = protocol.state
        linked = protocol.on_rx(ptype)
        transitioned = linked or (protocol.state != prev_state)

        if transitioned:
            dst_node = self.nodes[dst]
            dst_node.ready_time = self.clock + dst_node.processing_delay

        if linked:
            self.queue.push(self.clock, LINK_ESTABLISHED, {"a": src, "b": dst})

    def _on_link_established(self, ev) -> None:
        a, b = ev.payload["a"], ev.payload["b"]
        pair = frozenset({a, b})
        self.linked_pairs.add(pair)
        added = self.network_state.add_link(a, b, self.clock)
        if added:
            # Only record first-discovery time; reacquired links keep the original.
            self.result["pairwise_discovery_matrix"].setdefault(pair, self.clock)
            if self.result["first_link_time"] is None:
                self.result["first_link_time"] = self.clock

        # Phase R0 ATP: if either endpoint has a configured maintenance terminal
        # and hasn't switched yet, schedule the swap. Settling time starts now
        # so the mode_switch_active_until window blocks PACKET_TX from this
        # node until the optics stabilize. The actual config swap happens at
        # the BEAM_MODE_SWITCH event.
        for nid in (a, b):
            node = self.nodes[nid]
            if (
                node.optical_terminal_maintenance is not None
                and not node.in_maintenance_mode
            ):
                node.mode_switch_active_until = self.clock + self.mode_switch_settling_s
                self.queue.push(
                    self.clock + self.mode_switch_settling_s,
                    BEAM_MODE_SWITCH,
                    {"node": nid},
                )

        # Slice 5: arm the maintenance machinery if either endpoint opted in.
        self.last_keepalive_rx[pair] = self.clock
        for src, dst in [(a, b), (b, a)]:
            period = self.nodes[src].keepalive_period
            if period > 0:
                self.queue.push(
                    self.clock + period,
                    LINK_KEEPALIVE_TX,
                    {"src": src, "dst": dst},
                )
        timeout = max(self.nodes[a].link_timeout, self.nodes[b].link_timeout)
        if timeout > 0:
            self.queue.push(
                self.clock + timeout,
                LINK_HEALTH_CHECK,
                {"pair": pair},
            )

        # Phase A: if this pair carries the data session, (re)activate it.
        if self.data_session is not None and self._is_session_pair(a, b):
            was_pending = self.data_session.state == "PENDING"
            self.data_session.on_link_established(self.clock)
            if was_pending:
                self.queue.push(self.clock, DATA_SESSION_START, {"src": a, "dst": b})
            # Schedule the first DATA_TICK (or the next if we're resuming).
            self.queue.push(
                self.clock + self.data_session.tick_interval_s,
                DATA_TICK, {"pair": pair},
            )

        # Phase C: spin up TRACKER_TICK for each endpoint that has a non-null
        # beam tracker. Each tracker runs independently at its own cadence.
        for nid in (a, b):
            node = self.nodes[nid]
            if node.beam_tracker is None or isinstance(node.beam_tracker, NullBeamTracker):
                continue
            node.beam_tracker.reset()
            node.last_rx_power_dBm = -999.0
            self.queue.push(
                self.clock + node.beam_tracker_tick_s,
                TRACKER_TICK,
                {"node": nid, "pair": pair},
            )

    # ---------------------------------------------------------- geometry events

    def _on_scan_target_select(self, ev) -> None:
        node = self.nodes[ev.payload["node"]]
        if node.scanner is None:
            return

        peer_id = self._first_peer_with_estimate(node.node_id)
        if peer_id is None:
            return
        est = node.neighbor_estimates[peer_id]
        search_radius = est["uncertainty_radius_m"]

        rng = self.rng.stream(f"scanner:{node.node_id}")
        my_pos = node.position_at(self.clock)
        # Beam footprint radius at peer-estimate range. Raster/Spiral need it
        # to set their stripe/pitch; Beta/Uniform ignore it.
        peer_est_pos = est["estimated_position"]
        d_est = float(np.linalg.norm(peer_est_pos - my_pos))
        beam_div = float(node.optical_terminal["beam_divergence_rad"])
        beam_fp_radius = d_est * float(np.tan(beam_div / 2.0))
        target_pos = node.scanner.next_target(
            rng, my_pos, peer_est_pos, search_radius, beam_fp_radius
        )

        new_dir = direction_to(my_pos, target_pos)
        max_rate = float(node.optical_terminal["max_angular_speed_rad_s"])
        dt_slew = slew_time(node.current_pointing, new_dir, max_rate)

        # Smooth-slew path: subdivide so alignment crossings during the slew
        # are visible to the alignment gate. Required to reproduce the legacy
        # fso_sim_modules.py speed-curve shape.
        if (
            self.beam_sample_interval > 0.0
            and dt_slew > self.beam_sample_interval
            and node.current_pointing is not None
        ):
            self._slew_state[node.node_id] = {
                "start_dir": node.current_pointing.copy(),
                "target_dir": new_dir,
                "start_time": self.clock,
                "duration": dt_slew,
            }
            self.queue.push(
                self.clock + self.beam_sample_interval,
                BEAM_MOVE_SAMPLE,
                {"node": node.node_id},
            )
            return

        # Jump path (back-compat / fast mode): single BEAM_MOVE_END at slew end.
        self.queue.push(
            self.clock + dt_slew,
            BEAM_MOVE_END,
            {"node": node.node_id, "new_pointing": new_dir},
        )

    def _on_beam_move_sample(self, ev) -> None:
        """Mid-slew sample. Linearly interpolates the beam pointing toward
        the slew target, refreshes alignment with every peer, and either
        schedules the next sample or rolls into BEAM_MOVE_END if the slew is
        done."""
        node = self.nodes[ev.payload["node"]]
        state = self._slew_state.get(node.node_id)
        if state is None:
            return  # slew was cancelled; nothing to do

        elapsed = self.clock - state["start_time"]
        if elapsed >= state["duration"] - 1e-12:
            # Slew complete - finalize inline (avoid building a synthetic Event).
            self._slew_state.pop(node.node_id, None)
            self._set_pointing(node, state["target_dir"])
            for peer_id in self.peers[node.node_id]:
                if peer_id in self._started_nodes:
                    self._refresh_alignment(node.node_id, peer_id)
            if node.scanner is not None:
                self.queue.push(self.clock, SCAN_TARGET_SELECT, {"node": node.node_id})
            return

        # Mid-slew: interpolate pointing and refresh alignment.
        frac = elapsed / state["duration"]
        self._set_pointing(
            node,
            lerp_unit(state["start_dir"], state["target_dir"], frac),
        )
        for peer_id in self.peers[node.node_id]:
            if peer_id in self._started_nodes:
                self._refresh_alignment(node.node_id, peer_id)

        remaining = state["duration"] - elapsed
        dt_next = min(self.beam_sample_interval, remaining)
        # Guard the slice-5 FP-precision trap: never reschedule at the same
        # clock value.
        if dt_next <= 0.0:
            dt_next = max(remaining, 1e-12)
        self.queue.push(
            self.clock + dt_next,
            BEAM_MOVE_SAMPLE,
            {"node": node.node_id},
        )

    def _on_beam_move_end(self, ev) -> None:
        node = self.nodes[ev.payload["node"]]
        self._set_pointing(node, ev.payload["new_pointing"])
        # Clear any lingering slew context.
        self._slew_state.pop(node.node_id, None)

        # Recompute alignment with all peers.
        for peer_id in self.peers[node.node_id]:
            if peer_id in self._started_nodes:
                self._refresh_alignment(node.node_id, peer_id)

        # Beta scanner is continuous: immediately pick the next target.
        if node.scanner is not None:
            self.queue.push(self.clock, SCAN_TARGET_SELECT, {"node": node.node_id})

    # ------------------------------------------------------------ ATP mode

    def _on_beam_mode_switch(self, ev) -> None:
        """ATP dual-beam mode swap.

        Bidirectional: toggles between acquisition (wide) and maintenance
        (narrow) based on the node's current `in_maintenance_mode` flag.
        Scheduled at (event_trigger + settling_s) by:

          - _on_link_established (acquisition -> maintenance), once per endpoint
          - _on_link_lost when simulation.revert_to_acquisition_on_link_loss is
            True (maintenance -> acquisition)

        Two defensive checks - both are important when link_timeout is short
        relative to mode_switch_settling_s and the keepalive-during-settling
        drops can trigger LINK_LOST inside the settling window:

          1. A forward swap (acquisition -> maintenance) is committed ONLY if
             the node still participates in at least one linked pair. If the
             link died inside the settling window, we stay in acquisition so
             the next BEACON_TICK / scanner cycle can re-discover with the
             wide beam.
          2. mode_switch_active_until is always cleared so PACKET_TX flows
             again whether we committed the swap or not.
        """
        nid = ev.payload["node"]
        node = self.nodes[nid]
        # Settling window ends regardless of what we do next.
        node.mode_switch_active_until = 0.0

        if node.in_maintenance_mode:
            # Revert to acquisition (fallback path). Always safe to commit.
            if node.optical_terminal_acquisition is None:
                return
            node.optical_terminal = node.optical_terminal_acquisition
            node.in_maintenance_mode = False
        else:
            # Forward path requires the link to still be alive. Without this,
            # a LINK_LOST inside the settling window leaves a forward swap
            # queued that would land on a dead link and wedge the node in
            # maintenance mode with no peer to revert from.
            still_linked = any(
                nid in pair for pair in self.linked_pairs
            )
            if not still_linked:
                return
            if node.optical_terminal_maintenance is None:
                return
            node.optical_terminal = node.optical_terminal_maintenance
            node.in_maintenance_mode = True
        self.result["mode_switches"] += 1
        for peer_id in self.peers[node.node_id]:
            if peer_id in self._started_nodes:
                self._refresh_alignment(node.node_id, peer_id)

    # ------------------------------------------------------------ data plane

    def _is_session_pair(self, a: str, b: str) -> bool:
        if self.data_session is None:
            return False
        return frozenset({a, b}) == frozenset({self.data_session.src, self.data_session.dst})

    def _on_data_tick(self, ev) -> None:
        if self.data_session is None:
            return
        pair = ev.payload["pair"]
        # Only advance if the link is still up. If it dropped we'll be re-armed
        # by the next LINK_ESTABLISHED.
        if pair not in self.linked_pairs:
            return
        sess = self.data_session
        if sess.state not in ("ACTIVE",):
            return

        a, b = sess.src, sess.dst
        src_node = self.nodes[a]
        dst_node = self.nodes[b]

        # Compute capacity + PER + RTT at the current moment. Channel is
        # evaluated in the data direction (src -> dst).
        if self.channel.mode == "disabled":
            # When channel modeling is off, treat capacity as the configured
            # cap from the data_session config (default 1 Gbps).
            capacity_bps = float(self.data_session_cfg.get("capacity_bps", 1e9))
            per = 0.0
        else:
            rng = self.rng.stream("channel") if self.channel.mode == "stochastic" else None
            p_rx = self.channel.received_power_dBm(src_node, dst_node, self.clock, rng)
            sensitivity = float(dst_node.optical_terminal.get("receiver_sensitivity_dBm", -45.0))
            per = per_from_received_power(p_rx, sensitivity)
            # Capacity model: Shannon limit relative to a configured noise
            # floor (default -65 dBm). Capped at modulation max from config.
            noise_floor = float(self.data_session_cfg.get("noise_floor_dBm", -65.0))
            bandwidth = float(self.data_session_cfg.get("bandwidth_hz", 1e9))
            max_rate = float(self.data_session_cfg.get("max_data_rate_bps", 10e9))
            capacity_bps = shannon_capacity_bps(p_rx, noise_floor, bandwidth, max_rate)

        # RTT = 2 * one-way propagation + 2 * processing delay
        from owdesmp.engines.geometry import propagation_delay_s
        prop = propagation_delay_s(src_node, dst_node, self.clock)
        rtt = 2.0 * prop + 2.0 * src_node.processing_delay

        sess.tick(self.clock, capacity_bps, per, rtt)

        if sess.state == "COMPLETED":
            self.queue.push(self.clock, DATA_SESSION_END, {"src": a, "dst": b})
            return
        # Schedule next tick. If the link drops before then, _on_link_lost
        # will pause the session and a future LINK_ESTABLISHED will resume.
        self.queue.push(
            self.clock + sess.tick_interval_s,
            DATA_TICK,
            {"pair": pair},
        )

    # ---------------------------------------------------------------- helpers

    def _pair_aligned(self, a: str, b: str) -> bool:
        na, nb = self.nodes[a], self.nodes[b]
        if not (na.has_geometry and nb.has_geometry):
            return True  # always-aligned default
        return self.alignment_active.get(frozenset({a, b}), False)

    def _refresh_alignment(self, a: str, b: str) -> None:
        na, nb = self.nodes[a], self.nodes[b]
        if not (na.has_geometry and nb.has_geometry):
            return
        key = frozenset({a, b})
        now = mutual_alignment(na, nb, self.clock)
        was = self.alignment_active.get(key, False)
        if now and not was:
            self.alignment_active[key] = True
            self.queue.push(self.clock, ALIGNMENT_START, {"a": a, "b": b})
        elif was and not now:
            self.alignment_active[key] = False
            self.queue.push(self.clock, ALIGNMENT_END, {"a": a, "b": b})
            # Legacy parity: pre-link pairs lose handshake progress when
            # alignment breaks. Each alignment window must complete the
            # three-way handshake on its own. Already-linked pairs are NOT
            # reset here (the slice-5 maintenance machinery owns LINK_LOST).
            if self.reset_fsm_on_alignment_loss and key not in self.linked_pairs:
                self.protocols[(a, b)].state = INIT
                self.protocols[(b, a)].state = INIT

    # -------------------------------------------------------- link maintenance

    def _on_link_keepalive_tx(self, ev) -> None:
        src = ev.payload["src"]
        dst = ev.payload["dst"]
        pair = frozenset({src, dst})
        if pair not in self.linked_pairs:
            return  # link already lost; stop the keepalive train
        # Reschedule the next keepalive even if this one drops.
        period = self.nodes[src].keepalive_period
        if period > 0:
            self.queue.push(
                self.clock + period,
                LINK_KEEPALIVE_TX,
                ev.payload,
            )
        # Send the keepalive through the normal PACKET_TX path so alignment
        # and channel gates apply (this is how mobility-induced loss bites).
        self.queue.push(
            self.clock,
            PACKET_TX,
            {"src": src, "dst": dst, "type": KEEPALIVE, "tx_time": self.clock},
        )
        self.result["keepalives_sent"] += 1

    def _on_link_health_check(self, ev) -> None:
        pair = ev.payload["pair"]
        if pair not in self.linked_pairs:
            return
        a, b = tuple(pair)
        timeout = max(self.nodes[a].link_timeout, self.nodes[b].link_timeout)
        if timeout <= 0:
            return  # maintenance disabled
        last = self.last_keepalive_rx.get(pair, self.clock)
        next_deadline = last + timeout
        # `next_deadline > self.clock` is the only safe condition for rescheduling.
        # Equality (or FP-slack equality) means the deadline already expired, in
        # which case rescheduling at `next_deadline == self.clock` would loop
        # forever at the same timestamp.
        if next_deadline <= self.clock:
            self.queue.push(self.clock, LINK_LOST, {"a": a, "b": b})
        else:
            self.queue.push(next_deadline, LINK_HEALTH_CHECK, {"pair": pair})

    def _on_link_lost(self, ev) -> None:
        a, b = ev.payload["a"], ev.payload["b"]
        pair = frozenset({a, b})
        if pair not in self.linked_pairs:
            return  # already cleaned up
        self.linked_pairs.discard(pair)
        if self.network_state.graph.has_edge(a, b):
            self.network_state.graph.remove_edge(a, b)
        # Reset both pair-FSMs so the next BEACON_TICK re-initiates discovery.
        self.protocols[(a, b)].state = INIT
        self.protocols[(b, a)].state = INIT
        self.last_keepalive_rx.pop(pair, None)
        self.result["links_lost"] += 1
        # Phase A: pause the data session and collapse cwnd to 1 MSS.
        if self.data_session is not None and self._is_session_pair(a, b):
            self.data_session.on_link_lost(self.clock)
        # Phase R0 (switch_with_fallback): if either endpoint is in maintenance
        # mode, schedule a swap BACK to acquisition so re-discovery can use the
        # wide beam. Settling delay applies to the revert too.
        if self.revert_to_acquisition_on_link_loss:
            for nid in (a, b):
                node = self.nodes[nid]
                if node.in_maintenance_mode and node.optical_terminal_acquisition is not None:
                    node.mode_switch_active_until = self.clock + self.mode_switch_settling_s
                    self.queue.push(
                        self.clock + self.mode_switch_settling_s,
                        BEAM_MODE_SWITCH,
                        {"node": nid},
                    )

    # ------------------------------------------------------------- mobility

    def _on_tracker_tick(self, ev) -> None:
        """Closed-loop beam-pointing correction during CONNECTED state.

        Builds an observation (rx_power + gradient proxy + current offsets),
        asks the tracker for (daz, del) in radians, applies that to the node's
        nominal_pointing. The next NODE_MOBILITY_UPDATE re-applies jitter on
        top of the now-corrected nominal so the actual current_pointing
        benefits from the correction.
        """
        nid = ev.payload["node"]
        pair = ev.payload["pair"]
        # Stop ticking if the link is no longer up (LINK_LOST already fired).
        if pair not in self.linked_pairs:
            return
        node = self.nodes[nid]
        if node.beam_tracker is None or isinstance(node.beam_tracker, NullBeamTracker):
            return

        a, b = tuple(pair)
        partner_id = b if a == nid else a
        partner = self.nodes[partner_id]
        if not (node.has_geometry and partner.has_geometry):
            return

        # Current rx_power and angular offset (proxy for control error).
        pa = node.position_at(self.clock)
        pb = partner.position_at(self.clock)
        vec = pb - pa
        dist = float(np.linalg.norm(vec))
        if dist == 0.0:
            return
        a_to_b = vec / dist
        # angle_between is magnitude-only; for the tracker we need signed
        # az/el offsets. We observe the JITTERED current_pointing (= what a
        # quadrant detector on the partner would actually see), not the
        # un-jittered nominal — that's what makes the controller able to
        # cancel jitter. Corrections are applied to nominal so the next
        # mobility tick reapplies jitter on top of the corrected nominal.
        ref = node.current_pointing if node.current_pointing is not None else node.nominal_pointing
        err_vec = a_to_b - ref
        # Small-angle linearization onto two tangent axes (Y=azimuth, Z=elevation).
        az_err = float(err_vec[1])
        el_err = float(err_vec[2])

        # Received power (for the rx-power-delta gradient).
        if self.channel.mode == "disabled":
            rx_dBm = 0.0
        else:
            rng = self.rng.stream("channel") if self.channel.mode == "stochastic" else None
            rx_dBm = self.channel.received_power_dBm(partner, node, self.clock, rng)

        if node.last_rx_power_dBm > -900:
            drx = rx_dBm - node.last_rx_power_dBm
            # Gradient proxy: rx_power change attributed to current axis error.
            # Sign convention: when power drops on positive-az drift, gradient
            # is negative on az -> correction is negative on az.
            grad_az = drx * (-1.0 if az_err > 0 else 1.0) * 0.1
            grad_el = drx * (-1.0 if el_err > 0 else 1.0) * 0.1
        else:
            grad_az = -az_err  # bootstrap: steer toward partner direction
            grad_el = -el_err
        node.last_rx_power_dBm = rx_dBm

        obs = {
            "rx_power_dBm": rx_dBm,
            "rx_power_gradient_az": grad_az,
            "rx_power_gradient_el": grad_el,
            "current_az_offset": az_err,
            "current_el_offset": el_err,
            "dt": node.beam_tracker_tick_s,
        }
        daz, del_ = node.beam_tracker.get_correction(obs)

        # Apply correction. Two semantics (see BeamTracker docstring):
        #   - absolute offset (RL fine-pointing): nominal = rotate(a_to_b, .)
        #     recomputed from the ideal direction each tick, so a zero or
        #     slightly-biased action does NOT accumulate into drift. Always
        #     re-applied so a zero action snaps the beam onto the target.
        #   - cumulative (legacy PID/Kalman): correction added onto the
        #     existing nominal; only applied when nonzero.
        if getattr(node.beam_tracker, "correction_is_absolute_offset", False):
            if node.nominal_pointing is not None:
                new_nominal = apply_az_el_correction(a_to_b, daz, del_)
                self._set_pointing(node, new_nominal)
                for peer_id in self.peers[node.node_id]:
                    if peer_id in self._started_nodes:
                        self._refresh_alignment(node.node_id, peer_id)
        elif (daz != 0.0 or del_ != 0.0) and node.nominal_pointing is not None:
            new_nominal = apply_az_el_correction(node.nominal_pointing, daz, del_)
            self._set_pointing(node, new_nominal)
            for peer_id in self.peers[node.node_id]:
                if peer_id in self._started_nodes:
                    self._refresh_alignment(node.node_id, peer_id)

        # Reschedule.
        self.queue.push(
            self.clock + node.beam_tracker_tick_s,
            TRACKER_TICK,
            {"node": nid, "pair": pair},
        )

    def _on_node_mobility_update(self, ev) -> None:
        node = self.nodes[ev.payload["node"]]
        # Re-sample pointing jitter against the current nominal pointing so the
        # beam rotates with the vibration even when the scanner is idle. No-op
        # for NullPointingJitter, so legacy scenarios are unaffected.
        if (
            node.nominal_pointing is not None
            and node.pointing_jitter is not None
            and not isinstance(node.pointing_jitter, NullPointingJitter)
        ):
            dr, dp, dy = node.pointing_jitter.sample_rad(self.clock)
            R = jitter_rotation_matrix(dr, dp, dy)
            rotated = R @ node.nominal_pointing
            n = float(np.linalg.norm(rotated))
            if n > 0:
                rotated = rotated / n
            node.current_pointing = rotated
        # Refresh alignment with every started peer; positions and pointing
        # may both have drifted since the last update.
        for peer_id in self.peers[node.node_id]:
            if peer_id in self._started_nodes:
                self._refresh_alignment(node.node_id, peer_id)
        # Self-reschedule.
        if self.mobility_update_period > 0:
            self.queue.push(
                self.clock + self.mobility_update_period,
                NODE_MOBILITY_UPDATE,
                {"node": node.node_id},
            )

    def _set_pointing(self, node: Node, direction: np.ndarray) -> None:
        """Set both nominal_pointing (scanner-intended) and current_pointing
        (jittered). Used everywhere current_pointing would otherwise be
        assigned directly. Idempotent under NullPointingJitter."""
        direction = np.asarray(direction, dtype=float)
        node.nominal_pointing = direction.copy()
        if node.pointing_jitter is None or isinstance(node.pointing_jitter, NullPointingJitter):
            node.current_pointing = direction
            return
        dr, dp, dy = node.pointing_jitter.sample_rad(self.clock)
        R = jitter_rotation_matrix(dr, dp, dy)
        rotated = R @ direction
        n = float(np.linalg.norm(rotated))
        if n > 0:
            rotated = rotated / n
        node.current_pointing = rotated

    # ---------------------------------------------------------------- helpers

    def _first_peer_with_estimate(self, node_id: str) -> Optional[str]:
        """Next peer the scanner should target: first un-discovered peer with a
        known position estimate. Once a pair is linked, the scanner advances
        to the next un-discovered peer (sequential discovery)."""
        for p in self.peers[node_id]:
            if frozenset({node_id, p}) in self.linked_pairs:
                continue
            if p in self.nodes[node_id].neighbor_estimates:
                return p
        return None
