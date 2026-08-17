"""TCP Reno fluid-flow model.

Per-packet TCP simulation would be intractable for the 1-PB transfers OWDES-MP
needs to study (10^14+ segments at 1500 B). This module uses the standard
fluid approximation:

  - cwnd is treated as a continuous variable in bytes
  - per DATA_TICK of length dt:
      effective_bps = min(cwnd * 8 / rtt, capacity_bps)
      bytes_attempted = effective_bps * dt / 8
      bytes_acked = bytes_attempted * (1 - PER)
  - cwnd grows by 1 MSS per RTT in congestion avoidance, doubles per RTT in
    slow start (so over dt: cwnd *= 2^(dt/rtt) and cwnd += mss * dt/rtt)
  - a loss event in the tick triggers fast retransmit: ssthresh = cwnd/2,
    cwnd = ssthresh
  - an outage (LINK_LOST in the simulator) triggers timeout: ssthresh = cwnd/2,
    cwnd = 1 MSS, slow start re-engages

The model is deliberately small (Reno only, no SACK/dup-ack details) so the
slice-1 parity-style validation strategy applies: a unit test pins the cwnd
trace under known inputs.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TCPFlowReno:
    mss_bytes: int = 1500
    init_cwnd_mss: int = 1
    init_ssthresh_bytes: int = 65535  # high so slow-start runs until first loss
    min_rtt_s: float = 1e-4  # floor to avoid div-by-zero when geometry not configured

    # Mutable state (set in __post_init__)
    cwnd_bytes: float = field(init=False)
    ssthresh_bytes: float = field(init=False)
    in_slow_start: bool = field(init=False, default=True)
    rtt_s: float = field(init=False, default=0.01)

    # Counters
    total_bytes_sent: float = field(init=False, default=0.0)
    total_bytes_acked: float = field(init=False, default=0.0)
    loss_events: int = field(init=False, default=0)
    timeout_events: int = field(init=False, default=0)

    # Optional trace for plotting (list of (clock_s, cwnd_bytes))
    cwnd_trace: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        self.cwnd_bytes = float(self.init_cwnd_mss * self.mss_bytes)
        self.ssthresh_bytes = float(self.init_ssthresh_bytes)

    # ------------------------------------------------------------------ tick

    def tick(
        self,
        dt_s: float,
        capacity_bps: float,
        per: float,
        rtt_s: float,
        clock_s: float = 0.0,
    ) -> float:
        """Advance one tick of length dt_s. Returns bytes ACKed this tick.

        capacity_bps: instantaneous link capacity in bits/second (0 if outage).
        per: packet error rate in [0, 1]. The model uses this as an expected
             loss-fraction over the tick rather than rolling per-packet.
        rtt_s: current RTT estimate (2*prop + processing). Stored for next tick.
        """
        self.rtt_s = max(float(rtt_s), self.min_rtt_s)

        # If link is dead this tick, no progress.
        if capacity_bps <= 0.0 or dt_s <= 0.0:
            self.cwnd_trace.append((clock_s, self.cwnd_bytes))
            return 0.0

        # Effective throughput is min of window-limited and capacity-limited.
        cwnd_bits = self.cwnd_bytes * 8.0
        window_limited_bps = cwnd_bits / self.rtt_s
        effective_bps = min(window_limited_bps, capacity_bps)

        bytes_attempted = effective_bps * dt_s / 8.0
        clipped_per = max(0.0, min(1.0, float(per)))
        bytes_acked = bytes_attempted * (1.0 - clipped_per)

        self.total_bytes_sent += bytes_attempted
        self.total_bytes_acked += bytes_acked

        # cwnd evolution. If any loss this tick, treat as a fast-retransmit
        # event regardless of magnitude (fluid simplification). Otherwise
        # grow cwnd according to phase.
        if clipped_per > 0.0:
            self._on_loss()
        else:
            self._grow(dt_s)

        self.cwnd_trace.append((clock_s, self.cwnd_bytes))
        return bytes_acked

    # ------------------------------------------------------- cwnd evolution

    def _grow(self, dt_s: float) -> None:
        if self.rtt_s <= 0.0:
            return
        if self.in_slow_start:
            # cwnd *= 2 per RTT  ->  per dt: multiply by 2^(dt/RTT)
            growth = 2.0 ** (dt_s / self.rtt_s)
            self.cwnd_bytes *= growth
            if self.cwnd_bytes >= self.ssthresh_bytes:
                self.in_slow_start = False
        else:
            # +1 MSS per RTT (additive increase)
            self.cwnd_bytes += self.mss_bytes * (dt_s / self.rtt_s)

    def _on_loss(self) -> None:
        # Reno fast-retransmit: ssthresh = max(2 MSS, cwnd / 2); cwnd = ssthresh
        self.ssthresh_bytes = max(2.0 * self.mss_bytes, self.cwnd_bytes / 2.0)
        self.cwnd_bytes = self.ssthresh_bytes
        self.in_slow_start = False
        self.loss_events += 1

    def on_outage(self) -> None:
        """Called by the simulator when the link is lost. Mimics a TCP
        retransmission timeout: cwnd collapses to 1 MSS and slow start re-engages."""
        self.ssthresh_bytes = max(2.0 * self.mss_bytes, self.cwnd_bytes / 2.0)
        self.cwnd_bytes = float(self.mss_bytes)
        self.in_slow_start = True
        self.timeout_events += 1


# ------------------------------------------------------------------ helpers

def per_from_received_power(
    p_rx_dbm: float,
    sensitivity_dbm: float,
    margin_for_residual_per_dB: float = 6.0,
    residual_per: float = 1e-6,
) -> float:
    """Simple PER mapping from received power. Used by the simulator to feed
    the TCP flow's `per` argument.

      P_rx <= sensitivity:               PER = 1.0  (total loss; link is dead)
      sensitivity < P_rx < sens+margin:  PER linearly interpolates to residual
      P_rx >= sensitivity + margin:      PER = residual_per

    A real receiver would have a Q-function-based BER -> PER curve; this is
    deliberately the simplest model that produces the correct qualitative
    behavior (sharp falloff near sensitivity, near-zero loss with margin).
    """
    if p_rx_dbm < sensitivity_dbm:
        return 1.0
    margin = p_rx_dbm - sensitivity_dbm
    if margin >= margin_for_residual_per_dB:
        return residual_per
    # Linear interpolation between 1.0 at margin=0 and residual_per at margin=margin_for_residual.
    frac = margin / margin_for_residual_per_dB
    return (1.0 - frac) * 1.0 + frac * residual_per


def shannon_capacity_bps(
    p_rx_dbm: float,
    noise_floor_dbm: float,
    bandwidth_hz: float,
    max_data_rate_bps: float = float("inf"),
) -> float:
    """Shannon limit C = B * log2(1 + SNR), capped at max_data_rate_bps.

    Returns 0 if SNR <= 0 (i.e., P_rx below noise floor).
    """
    import math
    snr_db = p_rx_dbm - noise_floor_dbm
    if snr_db <= 0:
        return 0.0
    snr_linear = 10.0 ** (snr_db / 10.0)
    cap = bandwidth_hz * math.log2(1.0 + snr_linear)
    return min(cap, max_data_rate_bps)
