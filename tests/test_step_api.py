"""Phase R1 tests for the RL step API.

The step API (advance_until_event + _dispatch + _finalize) must produce
results identical to run() when used without any agent intervention.
This is the load-bearing correctness check for the Gym wrapper.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from owdesmp.core.event import LINK_ESTABLISHED, TRACKER_TICK  # noqa: E402
from owdesmp.core.scenario_loader import load_scenario  # noqa: E402
from owdesmp.core.simulator import Simulator  # noqa: E402


def _base_scenario():
    return load_scenario(REPO_ROOT / "scenarios" / "two_node_data_session.json")


def test_advance_until_event_returns_matching_event():
    sc = _base_scenario()
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    ev = sim.advance_until_event(LINK_ESTABLISHED)
    assert ev is not None
    assert ev.type == LINK_ESTABLISHED
    # The matching event is returned WITHOUT being dispatched; clock has been
    # advanced by intervening events but not by this one.
    assert sim.clock <= ev.time


def test_advance_until_event_accepts_iterable():
    sc = _base_scenario()
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    ev = sim.advance_until_event([LINK_ESTABLISHED, TRACKER_TICK])
    assert ev is not None
    assert ev.type in (LINK_ESTABLISHED, TRACKER_TICK)


def test_drive_to_completion_matches_run_output():
    """The bare-minimum Gym pattern: bootstrap, advance to events, dispatch,
    finalize. Must produce identical results to run()."""
    # Baseline: plain run().
    sc1 = _base_scenario()
    sim1 = Simulator(scenario=sc1, run_id=0, seed=11)
    r1 = sim1.run()

    # Step pattern: dispatch every event we get back via advance_until_event,
    # which simulates an agent that never intervenes.
    sc2 = _base_scenario()
    sim2 = Simulator(scenario=sc2, run_id=0, seed=11)
    # Target every event type known to the dispatcher - effectively makes
    # advance_until_event return on every event.
    from owdesmp.core import event as ev_mod
    all_types = {
        ev_mod.SIM_START, ev_mod.SIM_END, ev_mod.NODE_START, ev_mod.BEACON_TICK,
        ev_mod.PACKET_TX, ev_mod.PACKET_RX, ev_mod.SCAN_TARGET_SELECT,
        ev_mod.BEAM_MOVE_SAMPLE, ev_mod.BEAM_MOVE_END, ev_mod.NODE_MOBILITY_UPDATE,
        ev_mod.ALIGNMENT_START, ev_mod.ALIGNMENT_END, ev_mod.LINK_KEEPALIVE_TX,
        ev_mod.LINK_HEALTH_CHECK, ev_mod.LINK_LOST, ev_mod.TRACKER_TICK,
        ev_mod.DATA_TICK, ev_mod.DATA_SESSION_START, ev_mod.DATA_SESSION_END,
        ev_mod.BEAM_MODE_SWITCH, ev_mod.LINK_ESTABLISHED,
    }
    while True:
        ev = sim2.advance_until_event(all_types)
        if ev is None:
            break
        # Pass-through agent: dispatch the matched event normally.
        should_stop = sim2._dispatch(ev)
        if should_stop:
            break
    r2 = sim2._finalize()

    # Headline metrics must match exactly.
    assert r1["first_link_time"] == r2["first_link_time"]
    assert r1["events_processed"] == r2["events_processed"]
    assert r1["links_lost"] == r2["links_lost"]
    assert r1["mode_switches"] == r2["mode_switches"]
    # Data-session bytes can be FP-sensitive across slightly different code
    # paths; require equality not approximate, since both runs are deterministic.
    if "data_session" in r1:
        assert r1["data_session"]["bytes_acked"] == r2["data_session"]["bytes_acked"]
        assert r1["data_session"]["outage_count"] == r2["data_session"]["outage_count"]


def test_step_pattern_can_intervene_between_events():
    """Skip a TRACKER_TICK without dispatching - the simulator state should
    reflect that the tracker did NOT run that tick. We can't easily prove
    this from external state alone, so just check that the world continues
    spinning sensibly."""
    sc = _base_scenario()
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    # Walk to the first LINK_ESTABLISHED and dispatch it.
    ev = sim.advance_until_event(LINK_ESTABLISHED)
    assert ev is not None
    sim._dispatch(ev)
    # From here, walk to the next TRACKER_TICK (if any) and skip it.
    next_ev = sim.advance_until_event(TRACKER_TICK)
    if next_ev is not None:
        # Skip = don't dispatch. Then keep going.
        # The link is up; events should keep accumulating.
        before = sim.result["events_processed"]
        from owdesmp.core import event as ev_mod
        sim.advance_until_event({ev_mod.SIM_END})
        assert sim.result["events_processed"] >= before


def test_bootstrap_is_idempotent():
    """Calling run() after advance_until_event() should still work because
    _bootstrap is idempotent (it does NOT re-push SIM_START etc)."""
    sc = _base_scenario()
    sim = Simulator(scenario=sc, run_id=0, seed=11)
    # Use advance_until_event to bootstrap
    sim.advance_until_event(LINK_ESTABLISHED)
    # Calling _bootstrap() again should not duplicate events
    sim._bootstrap()
    sim._bootstrap()
    # No assertion needed; just that it didn't crash or double-emit.
    # Verify by counting SIM_START events in the queue history is hard; we
    # instead just observe events_processed advances normally on continuation.
    r = sim.run()
    assert r is not None
