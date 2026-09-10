"""Collation and transform hooks for future trainer adapters.

The loader never hard-codes a model's preprocessing. Trainer adapters
provide a :class:`SampleTransform` (per-sample preparation — image loading,
resize, processor/tokenizer, prompt/label construction) and a
:class:`BatchCollator` (per-batch tensorization / formatting). Both are
plain protocols with a no-op reference implementation used in tests and
dry-runs.

Future HunyuanOCR adapters implement these two hooks; nothing else in this
subsystem changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from clouda_data.training_data.artifacts import ArtifactLoader
from clouda_data.training_data.models import SampleReference


@runtime_checkable
class SampleTransform(Protocol):
    """Prepare one sample reference (may load artifacts lazily)."""

    def __call__(self, sample: SampleReference) -> Any: ...


@runtime_checkable
class BatchCollator(Protocol):
    """Combine transformed samples into one model-ready batch."""

    def __call__(self, transformed: list[Any]) -> Any: ...


@dataclass
class TransformedSample:
    """Default transform output: metadata + lazily-resolved payload refs."""

    sample: SampleReference
    image_path: str | None = None
    image: Any = None  # PIL image, only if the transform loaded it
    text: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class NoopTransform:
    """Reference transform: resolves paths, loads nothing."""

    def __init__(self, dataset_root: str | Any) -> None:
        self.artifacts = ArtifactLoader(dataset_root)

    def __call__(self, sample: SampleReference) -> TransformedSample:
        resolved = self.artifacts.resolve(sample.image_path)
        return TransformedSample(
            sample=sample,
            image_path=str(resolved) if resolved else sample.image_path,
            text=sample.text,
        )


class NoopCollator:
    """Reference collator: passes the list through unchanged."""

    def __call__(self, transformed: list[Any]) -> list[Any]:
        return list(transformed)


class LoadingTransform:
    """Test/mock transform: eagerly decodes images when present."""

    def __init__(self, dataset_root: str | Any) -> None:
        self.artifacts = ArtifactLoader(dataset_root)

    def __call__(self, sample: SampleReference) -> TransformedSample:
        image = self.artifacts.image(sample.image_path)
        text = self.artifacts.text(sample.text)
        return TransformedSample(
            sample=sample, image=image, text=text, image_path=sample.image_path
        )


def transform_batch(
    batch: dict[str, Any],
    transform: SampleTransform,
    collator: BatchCollator,
) -> Any:
    """Apply transform + collator to a loader batch."""

    return collator([transform(sample) for sample in batch["samples"]])


TransformFactory = Callable[[str], SampleTransform]
