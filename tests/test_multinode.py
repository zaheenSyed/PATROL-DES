"""Slice 4 validation: multi-node graph metrics + network-level terminations.

  1. first_link_discovered triggers at the earliest pair (fastest of 3 pairs).
  2. every_node_has_neighbor triggers strictly later than first_link
     (need a second link to cover the third node).
  3. connected_graph for 3 nodes triggers on the second link (any 2 of 3
     pairs makes the graph connected).
  4. all_required_links_discovered with the full 3-pair set fires only after
     every pair has linked.
  5. all_required_links_discovered with a subset fires when that subset is
     covered, even if other pairs aren't yet linked.
  6. pairwise_discovery_matrix records every pair the simulator forms.
  7. Round-robin beam scheduling: each peer of a multi-peer node is served;
     no pair is starved.
  8. Sanity: result.n_components_at_end and n_links_at_end are filled.
"""

from copy import deepcopy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from owdesmp.core.scenario_loader import load_scenario  # noqa: E402
from owdesmp.core.simulator import Simulator  # noqa: E402


def _base():
    return load_scenario(REPO_ROOT / "scenarios" / "three_node_validation.json")


def _run(scenario, seed=1):
    sim = Simulator(scenario=scenario, run_id=0, seed=seed)
    return sim.run()


# ----------------------------------------------------------- 1. first_link

def test_first_link_discovered_triggers_early():
    sc = _base()
    sc["termination_condition"] = {"type": "first_link_discovered"}
    r = _run(sc)
    assert r["success"] is True
    assert r["n_links_at_end"] == 1, (
        f"first_link should stop after the first link forms; got {r['n_links_at_end']}"
    )
    assert r["termination_time"] == r["first_link_time"]


# ----------------------------------------------------------- 2. every_node_neighbor

def test_every_node_has_neighbor_needs_two_links():
    """In a 3-node mesh, the first link covers 2 of 3 nodes. The third node
    needs a second link to be a neighbor of someone."""
    sc = _base()
    sc["termination_condition"] = {"type": "every_node_has_neighbor"}
    r = _run(sc)
    assert r["success"] is True
    assert r["n_links_at_end"] >= 2
    assert r["termination_time"] > r["first_link_time"]


# ----------------------------------------------------------- 3. connected_graph

def test_connected_graph_triggers_on_second_link_in_triangle():
    """Any 2 of 3 pairs in a 3-node graph make it connected (path through the
    common node). connected_graph and every_node_has_neighbor coincide here."""
    sc = _base()
    sc["termination_condition"] = {"type": "connected_graph"}
    r = _run(sc)
    assert r["success"] is True
    assert r["n_components_at_end"] == 1
    assert r["n_links_at_end"] == 2


# ----------------------------------------------------------- 4. all_required full

def test_all_required_links_full_set_waits_for_all_pairs():
    sc = _base()
    sc["termination_condition"] = {
        "type": "all_required_links_discovered",
        "required_links": [["A", "B"], ["B", "C"], ["A", "C"]],
    }
    r = _run(sc)
    assert r["success"] is True
    assert r["n_links_at_end"] == 3, (
        f"all 3 pairs should be linked; got {r['n_links_at_end']}"
    )
    # All 3 pairs in the pairwise matrix.
    keys = set(r["pairwise_discovery_matrix"].keys())
    assert keys == {
        frozenset({"A", "B"}),
        frozenset({"B", "C"}),
        frozenset({"A", "C"}),
    }


# ----------------------------------------------------------- 5. all_required subset

def test_all_required_links_subset_triggers_when_subset_complete():
    sc = _base()
    # Require only A-B and B-C. A-C may or may not have formed by then.
    sc["termination_condition"] = {
        "type": "all_required_links_discovered",
        "required_links": [["A", "B"], ["B", "C"]],
    }
    r = _run(sc)
    assert r["success"] is True
    keys = set(r["pairwise_discovery_matrix"].keys())
    assert frozenset({"A", "B"}) in keys
    assert frozenset({"B", "C"}) in keys


# ----------------------------------------------------------- 6. pairwise matrix

def test_pairwise_matrix_records_link_times():
    sc = _base()
    sc["termination_condition"] = {
        "type": "all_required_links_discovered",
        "required_links": [["A", "B"], ["B", "C"], ["A", "C"]],
    }
    r = _run(sc)
    m = r["pairwise_discovery_matrix"]
    # All times are positive and ordered before termination_time.
    for pair, t in m.items():
        assert 0.0 < t <= r["termination_time"], f"pair {pair} t={t}"


# ----------------------------------------------------------- 7. no starvation

def test_round_robin_no_pair_starves():
    """If round-robin worked, every pair has been seen across many trials.
    We aggregate over 50 runs and check every pair forms in at least one run."""
    sc = _base()
    sc["termination_condition"] = {
        "type": "all_required_links_discovered",
        "required_links": [["A", "B"], ["B", "C"], ["A", "C"]],
    }
    pairs_ever_formed = set()
    for k in range(50):
        sim = Simulator(scenario=sc, run_id=k, seed=100 + k)
        r = sim.run()
        if r["success"]:
            pairs_ever_formed.update(r["pairwise_discovery_matrix"].keys())
    assert pairs_ever_formed == {
        frozenset({"A", "B"}),
        frozenset({"B", "C"}),
        frozenset({"A", "C"}),
    }


# ----------------------------------------------------------- 8. result fields

def test_result_fields_populated():
    sc = _base()
    r = _run(sc)
    assert "termination_type" in r
    assert r["termination_type"] == "connected_graph"
    assert "n_links_at_end" in r
    assert "n_components_at_end" in r
    assert "pairwise_discovery_matrix" in r
