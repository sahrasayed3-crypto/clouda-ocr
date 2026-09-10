"""Deterministic ordering utilities for the training data loader.

All ordering/seed derivation uses SHA-256/BLAKE2b over canonical key
material — never Python's built-in ``hash()``. Seeds are derived with the
same field-separated convention as ``clouda_data.factory.seed.derive``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterator, Sequence

_SEED_MASK = (1 << 63) - 1
_UNIT_SEPARATOR = "\x1f"


def derive_loader_seed(
    global_seed: int,
    *,
    epoch: int,
    world_size: int,
    rank: int,
    num_workers: int,
    worker_id: int,
    config_hash: str,
) -> int:
    """Derive a stable 63-bit seed for one (epoch, topology, config)."""

    key = _UNIT_SEPARATOR.join(
        (
            "clouda.training_data.shuffle.v1",
            str(int(global_seed)),
            str(int(epoch)),
            str(int(world_size)),
            str(int(rank)),
            str(int(num_workers)),
            str(int(worker_id)),
            str(config_hash),
        )
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _SEED_MASK


def stable_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def permute_indices(count: int, seed: int) -> list[int]:
    """Deterministic permutation of range(count) seeded by an integer.

    Fisher-Yates with a stable-hash PRNG. Requires O(count) ints only
    (used for shard-count-sized lists, never for full sample payloads).
    """

    if count < 0:
        raise ValueError("count must be >= 0")
    order = list(range(count))
    state = seed & _SEED_MASK
    for i in range(count - 1, 0, -1):
        state = _next_state(state)
        j = state % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def _next_state(state: int) -> int:
    digest = hashlib.blake2b(
        state.to_bytes(8, "big"), digest_size=8, person=b"c1d"
    ).digest()
    return int.from_bytes(digest, "big") & _SEED_MASK


class StableRandom:
    """Small deterministic PRNG for bounded shuffle buffers."""

    def __init__(self, seed: int) -> None:
        self._state = seed & _SEED_MASK

    def random(self) -> float:
        self._state = _next_state(self._state)
        return self._state / (1 << 63)

    def randbelow(self, n: int) -> int:
        if n <= 0:
            raise ValueError("n must be positive")
        return int(self.random() * n) % n


def bounded_shuffle_stream(
    stream: Iterator[Any],
    buffer_size: int,
    seed: int,
) -> Iterator[Any]:
    """Deterministic bounded-shuffle: stream through a fixed-size buffer.

    Order is reproducible for the same input stream + seed; distribution is
    approximately uniform within the buffer window (documented tradeoff vs.
    exact global shuffle, which would need O(N) sample metadata in RAM).
    """

    if buffer_size < 1:
        raise ValueError("buffer_size must be >= 1")
    rng = StableRandom(seed)
    buffer: list[Any] = []
    for item in stream:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = rng.randbelow(len(buffer))
        yield buffer[index]
        buffer[index] = item
    while buffer:
        yield buffer.pop(rng.randbelow(len(buffer)))


def shard_order(
    shard_ids: Sequence[str],
    seed: int,
) -> list[str]:
    """Deterministic shard visit order for one epoch."""

    return [
        shard_ids[i]
        for i in permute_indices(len(shard_ids), seed ^ stable_hash("shard_order"))
    ]
