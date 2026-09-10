"""Runtime sampling/balancing and curriculum hooks (generic mechanisms).

These hooks do NOT decide what is easy/hard and do not duplicate Dataset
Selection. They consume explicit weights/strata metadata that upstream
systems (dataset selection, hard-example analysis) attach to manifest rows.
Absent metadata means "no balancing" — a plain deterministic pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from clouda_data.training_data.models import SampleReference
from clouda_data.training_data.ordering import StableRandom, stable_hash

# Recognized strata keys (only used when present in row metadata).
STRATA_KEYS = (
    "document_type",
    "profile",
    "distortion",
    "source_class",
    "language",
    "difficulty_bucket",
    "difficulty",
)


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(key)
    provenance = row.get("provenance")
    if isinstance(provenance, Mapping):
        return provenance.get(key)
    return None


def stratum_of(sample: SampleReference) -> str | None:
    """Return the first recognized stratum value present in the row."""

    for key in STRATA_KEYS:
        value = _row_value(sample.row, key)
        if value not in (None, ""):
            return f"{key}={value}"
    return None


@dataclass(frozen=True)
class SamplingSchedule:
    """Per-epoch stratum weights supplied by upstream systems.

    ``weights`` maps stratum values (e.g. ``difficulty_bucket=hard``) to a
    relative keep-probability in [0, 1]. Strata without an entry keep
    probability 1.0. This is runtime reweighting only — the full dataset is
    never expanded in memory.
    """

    weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", dict(self.weights))
        for key, value in self.weights.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"Sampling weight must be within [0, 1]: {key}={value}"
                )


def weighted_stream(
    samples: Iterator[SampleReference],
    schedule: SamplingSchedule,
    *,
    global_seed: int,
    epoch: int,
) -> Iterator[SampleReference]:
    """Deterministically downsample strata per the schedule."""

    if not schedule.weights:
        yield from samples
        return
    seed = stable_hash(
        f"clouda.training_data.weighted|{global_seed}|{epoch}"
    )
    rng = StableRandom(seed)
    for sample in samples:
        stratum = stratum_of(sample)
        probability = (
            float(schedule.weights[stratum])
            if stratum is not None and stratum in schedule.weights
            else 1.0
        )
        if probability >= 1.0 or rng.random() < probability:
            yield sample


CurriculumFn = Callable[[int], SamplingSchedule]


class CurriculumHook:
    """Generic epoch->schedule hook driven by explicit upstream weights."""

    def __init__(self, schedule_for_epoch: CurriculumFn) -> None:
        self._schedule_for_epoch = schedule_for_epoch

    def schedule(self, epoch: int) -> SamplingSchedule:
        return self._schedule_for_epoch(epoch)

    def stream(
        self,
        samples: Iterator[SampleReference],
        *,
        global_seed: int,
        epoch: int,
    ) -> Iterator[SampleReference]:
        yield from weighted_stream(
            samples, self.schedule(epoch), global_seed=global_seed, epoch=epoch
        )


def static_schedule(weights: Mapping[str, float]) -> CurriculumHook:
    """Constant schedule for all epochs."""

    return CurriculumHook(lambda _epoch: SamplingSchedule(dict(weights)))


def stratified_counts(samples: Sequence[SampleReference]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        stratum = stratum_of(sample)
        if stratum is None:
            continue
        counts[stratum] = counts.get(stratum, 0) + 1
    return counts
