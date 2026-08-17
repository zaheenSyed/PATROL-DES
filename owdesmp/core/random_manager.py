"""Per-stream seeded RNGs.

Each subsystem (per-node beacon phase, channel turbulence, scanner choice, ...)
draws from its own random.Random instance so that adding or removing a feature
does not perturb unrelated random sequences. Stream seed = stable hash of
(base_seed, stream_name).
"""

import hashlib
import random
from typing import Dict


class RandomManager:
    def __init__(self, base_seed: int) -> None:
        self.base_seed = int(base_seed)
        self._streams: Dict[str, random.Random] = {}

    def stream(self, name: str) -> random.Random:
        if name not in self._streams:
            # hashlib gives a stable hash across Python invocations (unlike builtin hash()).
            h = hashlib.sha256(f"{self.base_seed}:{name}".encode()).digest()
            seed = int.from_bytes(h[:8], "big")
            self._streams[name] = random.Random(seed)
        return self._streams[name]
