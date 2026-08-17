"""Termination conditions.

Each strategy's `check(network_state, t)` is called after every LINK_ESTABLISHED
event. Returning True ends the simulation.

Slice 1: first_link_discovered, fixed_duration.
Slice 4: every_node_has_neighbor, connected_graph, all_required_links_discovered.
"""

from typing import FrozenSet, Iterable, List, Protocol


class TerminationCondition(Protocol):
    name: str

    def check(self, network_state, sim_time: float) -> bool: ...


class FirstLinkDiscovered:
    name = "first_link_discovered"

    def check(self, network_state, sim_time: float) -> bool:
        return network_state.graph.number_of_edges() >= 1


class EveryNodeHasNeighbor:
    name = "every_node_has_neighbor"

    def check(self, network_state, sim_time: float) -> bool:
        return network_state.every_node_has_neighbor()


class ConnectedGraph:
    name = "connected_graph"

    def check(self, network_state, sim_time: float) -> bool:
        return network_state.is_connected()


class AllRequiredLinksDiscovered:
    name = "all_required_links_discovered"

    def __init__(self, required_links: Iterable[Iterable[str]]) -> None:
        self._required: List[FrozenSet[str]] = [
            frozenset(pair) for pair in required_links
        ]

    def check(self, network_state, sim_time: float) -> bool:
        if not self._required:
            return False
        edges = {frozenset({u, v}) for u, v in network_state.graph.edges}
        return all(r in edges for r in self._required)


class FixedDuration:
    """Never terminates early; simulator stops at duration_s."""
    name = "fixed_duration"

    def check(self, network_state, sim_time: float) -> bool:
        return False


def make_termination(config: dict) -> TerminationCondition:
    type_ = config.get("type", "first_link_discovered")
    if type_ == "first_link_discovered":
        return FirstLinkDiscovered()
    if type_ == "every_node_has_neighbor":
        return EveryNodeHasNeighbor()
    if type_ == "connected_graph":
        return ConnectedGraph()
    if type_ == "all_required_links_discovered":
        return AllRequiredLinksDiscovered(config.get("required_links", []))
    if type_ == "fixed_duration":
        return FixedDuration()
    raise ValueError(f"Unknown termination type: {type_!r}")
