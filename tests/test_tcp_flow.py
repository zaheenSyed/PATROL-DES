"""Unit tests for the TCP Reno fluid-flow model."""

import math

import pytest

from owdesmp.engines.tcp_flow import (
    TCPFlowReno,
    per_from_received_power,
    shannon_capacity_bps,
)


# ------------------------------------------------------------ cwnd evolution

def test_slow_start_doubles_cwnd_per_rtt():
    """In slow start with no losses, cwnd doubles every RTT (within tick rounding)."""
    flow = TCPFlowReno(mss_bytes=1500, init_cwnd_mss=1, init_ssthresh_bytes=10**9)
    rtt = 0.01
    # Run 2 RTTs worth of ticks
    n_ticks = 200
    dt = rtt / 100
    for _ in range(n_ticks):
        flow.tick(dt_s=dt, capacity_bps=1e12, per=0.0, rtt_s=rtt)
    # After 2 RTTs cwnd should be ~4 MSS (initial 1, doubled twice).
    assert math.isclose(flow.cwnd_bytes, 4 * 1500, rel_tol=0.05)


def test_congestion_avoidance_linear_growth():
    """In CA, cwnd grows by ~1 MSS per RTT."""
    flow = TCPFlowReno(mss_bytes=1500, init_cwnd_mss=10, init_ssthresh_bytes=10 * 1500)
    flow.in_slow_start = False
    rtt = 0.01
    dt = rtt / 100
    initial = flow.cwnd_bytes
    for _ in range(100):
        flow.tick(dt_s=dt, capacity_bps=1e12, per=0.0, rtt_s=rtt)
    # After 1 RTT in CA, cwnd should be initial + 1 MSS.
    assert math.isclose(flow.cwnd_bytes, initial + 1500, rel_tol=0.01)


def test_loss_halves_cwnd_and_exits_slow_start():
    flow = TCPFlowReno(mss_bytes=1500, init_cwnd_mss=64)
    pre_cwnd = flow.cwnd_bytes
    flow.tick(dt_s=0.001, capacity_bps=1e10, per=0.01, rtt_s=0.01)
    # Fast-retransmit: cwnd = ssthresh = pre/2.
    assert math.isclose(flow.cwnd_bytes, pre_cwnd / 2, rel_tol=1e-9)
    assert flow.in_slow_start is False
    assert flow.loss_events == 1


def test_outage_collapses_cwnd_to_one_mss():
    flow = TCPFlowReno(mss_bytes=1500, init_cwnd_mss=100)
    flow.on_outage()
    assert flow.cwnd_bytes == 1500
    assert flow.in_slow_start is True
    assert flow.timeout_events == 1


# ------------------------------------------------------- throughput limits

def test_capacity_limited_throughput():
    """When cwnd is plenty, throughput is capped by link capacity."""
    flow = TCPFlowReno(mss_bytes=1500, init_cwnd_mss=10**4)  # huge cwnd
    bytes_acked = flow.tick(dt_s=0.001, capacity_bps=1e9, per=0.0, rtt_s=0.01)
    # 1 Gbps * 1 ms = 1 Mbit = 125 000 bytes.
    assert math.isclose(bytes_acked, 125000, rel_tol=1e-9)


def test_window_limited_throughput():
    """Small cwnd caps throughput below capacity."""
    flow = TCPFlowReno(mss_bytes=1500, init_cwnd_mss=10)  # 10 * 1500 = 15 kB cwnd
    bytes_acked = flow.tick(dt_s=0.001, capacity_bps=1e12, per=0.0, rtt_s=0.01)
    # window-limited: cwnd_bits/RTT = 15000*8/0.01 = 1.2e7 bps; over 1 ms: 12 000 bits = 1500 B.
    assert math.isclose(bytes_acked, 1500, rel_tol=0.05)


# -------------------------------------------------------------- per mapping

def test_per_zero_below_sensitivity():
    """Below sensitivity, PER = 1 (no progress)."""
    p = per_from_received_power(p_rx_dbm=-50, sensitivity_dbm=-35)
    assert p == 1.0


def test_per_residual_with_high_margin():
    p = per_from_received_power(p_rx_dbm=0, sensitivity_dbm=-35, margin_for_residual_per_dB=6.0)
    assert p < 1e-3


def test_per_monotonic_decreasing_in_margin():
    sens = -35
    margins = [0.0, 1.0, 3.0, 6.0, 12.0]
    pers = [per_from_received_power(sens + m, sens) for m in margins]
    for i in range(len(pers) - 1):
        assert pers[i + 1] <= pers[i]


# ----------------------------------------------------------- shannon helper

def test_shannon_zero_below_noise_floor():
    assert shannon_capacity_bps(-80, -60, 1e9) == 0.0


def test_shannon_capacity_grows_with_snr():
    low = shannon_capacity_bps(-40, -60, 1e9)
    high = shannon_capacity_bps(-30, -60, 1e9)
    assert high > low


def test_shannon_capped_at_max_rate():
    cap = shannon_capacity_bps(0, -60, 1e12, max_data_rate_bps=1e10)
    assert cap == 1e10
