"""Data-transfer session coordinator.

Couples the TCP flow model to the simulator's link-state lifecycle:

  LINK_ESTABLISHED   -> start (or resume) the session, schedule DATA_TICK
  DATA_TICK          -> advance the TCP flow by one tick (cwnd, throughput,
                        bytes transferred). If the link is alive and there's
                        still bytes to send, schedule the next DATA_TICK.
  LINK_LOST          -> pause the session, tcp.on_outage() (cwnd -> 1 MSS).
                        The next LINK_ESTABLISHED resumes.
  DATA_SESSION_END   -> total_bytes >= bytes_to_transfer (or sim ends).

Configuration lives in scenario["data_session"] (top-level, opt-in). If absent
the simulator behaves as before (no data plane). Slice 1-6 tests therefore
keep passing unchanged.

The simulator owns one session keyed on the source pair (sorted tuple). For
a multi-node deployment with multiple flows you'd hold a dict of these and
look up by (src, dst) - kept single-flow for slice-7's two-node validation.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from owdesmp.engines.tcp_flow import TCPFlowReno


@dataclass
class DataTransferSession:
    src: str
    dst: str
    bytes_to_transfer: float           # 1e15 = 1 PB
    tick_interval_s: float = 0.01      # 10 ms default fluid step
    tcp: TCPFlowReno = field(default_factory=TCPFlowReno)

    # Lifecycle state
    state: str = "PENDING"             # PENDING / ACTIVE / PAUSED / COMPLETED
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    last_tick_at: float = 0.0

    # Metrics
    outage_count: int = 0
    total_outage_duration_s: float = 0.0
    outage_started_at: Optional[float] = None
    reconnection_times_s: List[float] = field(default_factory=list)

    # Throughput history (clock_s, instantaneous_bps) for plotting/analysis
    throughput_trace: List = field(default_factory=list)

    def __post_init__(self):
        # If caller passed a dict for tcp via scenario loading, build the dataclass.
        if isinstance(self.tcp, dict):
            self.tcp = TCPFlowReno(**self.tcp)

    # ------------------------------------------------------------ lifecycle

    def on_link_established(self, clock: float) -> None:
        if self.state == "PENDING":
            self.state = "ACTIVE"
            self.started_at = clock
            self.last_tick_at = clock
        elif self.state == "PAUSED":
            # Reconnection: record how long we were down.
            self.state = "ACTIVE"
            self.last_tick_at = clock
            if self.outage_started_at is not None:
                gap = clock - self.outage_started_at
                self.total_outage_duration_s += gap
                self.reconnection_times_s.append(gap)
                self.outage_started_at = None

    def on_link_lost(self, clock: float) -> None:
        if self.state != "ACTIVE":
            return
        self.state = "PAUSED"
        self.outage_count += 1
        self.outage_started_at = clock
        self.tcp.on_outage()

    def tick(self, clock: float, capacity_bps: float, per: float, rtt_s: float) -> float:
        if self.state != "ACTIVE":
            return 0.0
        dt = clock - self.last_tick_at
        self.last_tick_at = clock
        if dt <= 0.0:
            return 0.0
        # Trim final tick so we don't overshoot the target byte count.
        remaining = self.bytes_to_transfer - self.tcp.total_bytes_acked
        if remaining <= 0.0:
            self._complete(clock)
            return 0.0
        bytes_acked = self.tcp.tick(dt, capacity_bps, per, rtt_s, clock_s=clock)
        instantaneous_bps = (bytes_acked * 8.0) / dt if dt > 0.0 else 0.0
        self.throughput_trace.append((clock, instantaneous_bps))
        if self.tcp.total_bytes_acked >= self.bytes_to_transfer:
            self._complete(clock)
        return bytes_acked

    def _complete(self, clock: float) -> None:
        self.state = "COMPLETED"
        self.completed_at = clock

    # --------------------------------------------------------- finalization

    def finalize(self, sim_end_clock: float) -> None:
        """Close any outstanding outage against sim_end_clock.

        Without this, `total_outage_duration_s` only accumulates outages that
        ENDED with a re-establish (the only place it was incremented before).
        When a tracker breaks the link near the start of a trial and the
        simulator never reacquires before the trial ends, the entire dead-link
        interval would silently drop out of the metric — making the reported
        availability look 100% even when no bytes flowed for the bulk of the
        run. The simulator calls this once, at the end of `run()`, before
        invoking `summary()`. Calling it on a session that's not currently
        PAUSED is a no-op so it's safe to invoke unconditionally.
        """
        if self.state == "PAUSED" and self.outage_started_at is not None:
            gap = max(0.0, sim_end_clock - self.outage_started_at)
            self.total_outage_duration_s += gap
            # Deliberately NOT appended to reconnection_times_s — there was no
            # reconnection. mean_reconnection_time_s stays representative of
            # actual reconnect events.
            self.outage_started_at = None

    # ----------------------------------------------------------- summary

    def summary(self) -> dict:
        duration = None
        if self.completed_at is not None and self.started_at is not None:
            duration = self.completed_at - self.started_at
        eff_bps = None
        if duration and duration > 0.0:
            eff_bps = (self.tcp.total_bytes_acked * 8.0) / duration
        # Throughput-trace-derived "data-plane availability": the number of
        # DATA_TICKs that actually ran (one trace entry per active tick),
        # multiplied by the tick interval, is a byte-accurate proxy for how
        # long the link was ACTIVE. Reported separately so consumers can
        # cross-check against the outage-bookkeeping availability.
        return {
            "src": self.src,
            "dst": self.dst,
            "state": self.state,
            "bytes_to_transfer": self.bytes_to_transfer,
            "bytes_acked": self.tcp.total_bytes_acked,
            "bytes_sent": self.tcp.total_bytes_sent,
            "duration_s": duration,
            "effective_bps": eff_bps,
            "outage_count": self.outage_count,
            "total_outage_duration_s": self.total_outage_duration_s,
            "reconnection_times_s": list(self.reconnection_times_s),
            "mean_reconnection_time_s": (
                sum(self.reconnection_times_s) / len(self.reconnection_times_s)
                if self.reconnection_times_s else None
            ),
            "loss_events": self.tcp.loss_events,
            "timeout_events": self.tcp.timeout_events,
            "final_cwnd_bytes": self.tcp.cwnd_bytes,
            # Data-plane availability metric — see docstring on finalize().
            "n_ticks_active": len(self.throughput_trace),
            "tick_interval_s": self.tick_interval_s,
            "active_time_s": len(self.throughput_trace) * self.tick_interval_s,
        }
