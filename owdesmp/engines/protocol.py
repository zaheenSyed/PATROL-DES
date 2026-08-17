"""Pairwise 3-way handshake state machine.

Important design decision (matches hand_des_4.py:54-113 semantics exactly):
the FSM state tracks RX history only. TX does NOT advance state. Whether to
re-transmit a packet on the next beacon is decided by reading current state
and applying next_tx_type(). Whether a pair has finished the handshake is
tracked externally by the simulator (as a set of linked pairs) so that both
ends stop transmitting on the same network-level event.

State of node N's protocol toward peer P (one FSM per ordered pair (N, P)):
  INIT          - no contact yet; next TX will be BEACON
  HEARD_BEACON  - received BEACON from P; next TX will be BEACON_ACK
  HEARD_BACK    - received BEACON_ACK from P; next TX will be ACK
"""

from owdesmp.models.packet import ACK, BEACON, BEACON_ACK

INIT = "INIT"
HEARD_BEACON = "HEARD_BEACON"
HEARD_BACK = "HEARD_BACK"


class PairwiseProtocol:
    def __init__(self) -> None:
        self.state: str = INIT

    def next_tx_type(self):
        """Packet type this side should transmit next based on RX history."""
        if self.state == INIT:
            return BEACON
        if self.state == HEARD_BEACON:
            return BEACON_ACK
        if self.state == HEARD_BACK:
            return ACK
        return None

    def on_rx(self, packet_type: str) -> bool:
        """Advance state on reception. Returns True iff this RX is the closing ACK.

        ACK is accepted regardless of current state: in the asymmetric handshake
        where only the peer initiated, this side may still be in HEARD_BEACON
        when the peer's closing ACK arrives. Matches hand_des_4.py:95
        (`if msg == "ACK": return True` with no state guard).
        """
        if packet_type == BEACON and self.state == INIT:
            self.state = HEARD_BEACON
            return False
        if packet_type == BEACON_ACK and self.state in (INIT, HEARD_BEACON):
            self.state = HEARD_BACK
            return False
        if packet_type == ACK:
            return True
        return False
