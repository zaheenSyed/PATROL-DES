"""Discovered-link graph.

Wraps a networkx.Graph so the simulator can ask network-level questions
(is_connected, every node has at least one neighbor, count components) used
by the slice-4 termination conditions and the slice-5 metrics.

The `links` property exposes an {(sorted-pair): first-seen-time} dict so the
slice-1 / slice-2 / slice-3 termination code that referenced `network_state.links`
keeps working without changes.
"""

from typing import Dict, Iterable, Tuple

import networkx as nx


class NetworkState:
    def __init__(self, node_ids: Iterable[str]) -> None:
        self.graph = nx.Graph()
        for nid in node_ids:
            self.graph.add_node(nid)

    def add_link(self, a: str, b: str, t: float) -> bool:
        if self.graph.has_edge(a, b):
            return False
        self.graph.add_edge(a, b, time=t)
        return True

    @property
    def links(self) -> Dict[Tuple[str, str], float]:
        return {
            tuple(sorted((u, v))): self.graph.edges[u, v]["time"]
            for u, v in self.graph.edges
        }

    def is_connected(self) -> bool:
        if self.graph.number_of_nodes() == 0:
            return False
        return nx.is_connected(self.graph)

    def every_node_has_neighbor(self) -> bool:
        if self.graph.number_of_nodes() == 0:
            return False
        return all(deg >= 1 for _, deg in self.graph.degree)

    def number_of_components(self) -> int:
        return nx.number_connected_components(self.graph)
