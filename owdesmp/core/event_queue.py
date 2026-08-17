"""Priority queue for discrete events, keyed by (time, seq).

The sequence number is a strict tie-breaker so events scheduled at the same time
fire in insertion order. This is the same pattern as hand_des_4.py:57.
"""

import heapq
from typing import Any, Optional

from owdesmp.core.event import Event


class EventQueue:
    def __init__(self) -> None:
        self._pq: list = []
        self._seq: int = 0

    def push(self, time: float, type_: str, payload: Any = None) -> None:
        self._seq += 1
        heapq.heappush(self._pq, (time, self._seq, type_, payload))

    def pop(self) -> Optional[Event]:
        if not self._pq:
            return None
        time, seq, type_, payload = heapq.heappop(self._pq)
        return Event(time=time, seq=seq, type=type_, payload=payload)

    def peek_time(self) -> Optional[float]:
        if not self._pq:
            return None
        return self._pq[0][0]

    def __len__(self) -> int:
        return len(self._pq)

    def __bool__(self) -> bool:
        return bool(self._pq)
