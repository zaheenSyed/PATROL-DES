"""End-to-end data session test.

Validates the discover -> transfer -> outage -> pause lifecycle by running
the two_node_data_session scenario:

    A static at origin, B drifts perpendicular to A's beam at 100 m/s.
    Link forms quickly, data flows, link breaks ~44 ms later when B exits
    A's 2.5 deg half-divergence, session is paused with cwnd collapsed to
    1 MSS. (Re-discovery is not expected here - B keeps moving away.)
"""

from copy import deepcopy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from owdesmp.core.scenario_loader import load_scenario  # noqa: E402
from owdesmp.core.simulator import Simulator  # noqa: E402


def _base():
    return load_scenario(REPO_ROOT / "scenarios" / "two_node_data_session.json")


def test_session_starts_after_discovery():
    sc = _base()
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    r = sim.run()
    assert r["first_link_time"] is not None and r["first_link_time"] < 0.01
    ds = r["data_session"]
    assert ds["state"] in ("ACTIVE", "PAUSED", "COMPLETED")
    assert ds["bytes_acked"] > 0, "session should have transferred bytes after discovery"


def test_link_loss_pauses_session_and_collapses_cwnd():
    sc = _base()
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    r = sim.run()
    assert r["links_lost"] >= 1
    ds = r["data_session"]
    # Outage was recorded.
    assert ds["outage_count"] >= 1
    # cwnd was collapsed to 1 MSS on outage. With B never returning into the
    # beam, session stays in PAUSED and the final cwnd is the post-outage
    # value (~1 MSS for Reno timeout).
    assert ds["state"] == "PAUSED"
    assert ds["final_cwnd_bytes"] <= 1500 * 4  # 1 MSS, with some growth tolerance


def test_session_does_not_run_without_data_session_config():
    """Slice 1 scenarios with no data_session block must keep working unchanged."""
    sc = load_scenario(REPO_ROOT / "scenarios" / "two_node_validation.json")
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    r = sim.run()
    assert "data_session" not in r


def test_throughput_trace_recorded_while_active():
    sc = _base()
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    sim.run()
    # The trace lives on the live session instance.
    trace = sim.data_session.throughput_trace
    assert len(trace) >= 5, f"expected several DATA_TICK trace entries; got {len(trace)}"
    # Each entry is (clock_s, bps). All bps non-negative.
    assert all(bps >= 0.0 for _, bps in trace)


def test_session_summary_fields_present():
    sc = _base()
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    r = sim.run()
    ds = r["data_session"]
    for key in [
        "bytes_acked", "bytes_sent", "outage_count", "timeout_events",
        "final_cwnd_bytes", "state", "effective_bps",
        # finalize() patch: throughput-derived availability cross-check fields.
        "n_ticks_active", "tick_interval_s", "active_time_s",
    ]:
        assert key in ds


def test_finalize_closes_outstanding_outage_at_sim_end():
    """Regression for the bookkeeping bug: when the link is lost partway
    through the trial and the simulator never reacquires before sim end,
    total_outage_duration_s must reflect the dead-link time.

    The two_node_data_session scenario has B drifting out of A's beam, so
    after LINK_LOST the link cannot recover. Before the finalize() patch,
    total_outage_duration_s was 0 because it was only incremented inside
    on_link_established(), which never fired.
    """
    sc = _base()
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    r = sim.run()
    assert r["links_lost"] >= 1, "scenario must produce a LINK_LOST for this test"
    ds = r["data_session"]
    assert ds["state"] == "PAUSED", "link did not stay down through end of sim"
    # total_outage_duration_s must be nonzero and consistent with the time
    # between LINK_LOST and sim end.
    assert ds["total_outage_duration_s"] > 0.0, (
        "outage time not recorded — finalize() is not closing the outstanding "
        "outage at sim end"
    )
    # Sanity: active_time_s + total_outage_duration_s should be close to the
    # post-discovery window (modulo TCP tick alignment slack).
    sim_end = sc["simulation"]["duration_s"]
    post = sim_end - r["first_link_time"]
    accounted = ds["active_time_s"] + ds["total_outage_duration_s"]
    assert abs(accounted - post) / post < 0.05, (
        f"active+outage time ({accounted:.3f}s) should account for the "
        f"post-discovery window ({post:.3f}s) to within 5%"
    )


def test_link_availability_consistent_with_throughput_availability():
    """After the finalize() patch, the outage-bookkeeping availability and
    the throughput-trace-derived availability must agree to within a tick
    or two."""
    sc = _base()
    sim_end = float(sc["simulation"]["duration_s"])
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    r = sim.run()
    ds = r["data_session"]

    post = sim_end - r["first_link_time"]
    if post <= 0:
        return  # nothing to assert
    bookkeeping_avail = max(0.0, post - ds["total_outage_duration_s"]) / post
    tput_avail = min(1.0, ds["active_time_s"] / post)
    # Disagreement under 5% of post-discovery window (= one or two ticks at
    # the default 10 ms tick interval).
    assert abs(bookkeeping_avail - tput_avail) < 0.05, (
        f"availability metrics disagree: "
        f"bookkeeping={bookkeeping_avail:.3f}, throughput={tput_avail:.3f}; "
        f"diff={(bookkeeping_avail - tput_avail):.3f}"
    )
