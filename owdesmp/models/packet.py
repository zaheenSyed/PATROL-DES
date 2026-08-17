"""Packet types and Packet dataclass for the 3-way handshake.

Spec rename from hand_des_4.py's SYN/SYNACK/ACK to spec terminology:
  BEACON     ~ SYN     (opening discovery packet)
  BEACON_ACK ~ SYNACK  (response acknowledging the beacon)
  ACK        ~ ACK     (closes the handshake)
"""

from dataclasses import dataclass

BEACON = "BEACON"
BEACON_ACK = "BEACON_ACK"
ACK = "ACK"
KEEPALIVE = "KEEPALIVE"  # reserved for slice 5


@dataclass
class Packet:
    src: str
    dst: str
    type: str
    tx_time: float
    duration: float
