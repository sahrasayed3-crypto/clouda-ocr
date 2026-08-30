from __future__ import annotations

import hashlib
import random


def derive_seed(base_seed: int, *parts: str) -> int:
    material = ":".join([str(base_seed), *parts])
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def seeded_rng(base_seed: int, *parts: str) -> random.Random:
    return random.Random(derive_seed(base_seed, *parts))
