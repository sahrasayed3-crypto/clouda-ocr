"""Optional PyTorch adapter for the training data loader.

This module imports torch LAZILY: importing :mod:`clouda_data.training_data`
never requires torch, and the base package works without it. Use
:func:`torch_available` / :func:`require_torch` to probe.

The adapter exposes the loader as a ``torch.utils.data.IterableDataset``
with the standard PyTorch distributed/worker contract:

- ``world_size``/``rank`` come from ``torch.distributed`` when initialized
  (overridable explicitly);
- ``num_workers``/``worker_id`` come from ``torch.utils.data.get_worker_info``;
- assignment is deterministic and disjoint across (rank, worker) pairs, so
  ``DataLoader(dataset, num_workers=N)`` yields every sample exactly once.

This module is intentionally NOT imported from the package ``__init__``.
"""

from __future__ import annotations

from typing import Any, Iterator

_TORCH_HINT = (
    "PyTorch is not installed. Install the 'training' extra "
    "(pip install -e .[training]) or add torch to use the torch adapter."
)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(_TORCH_HINT) from exc
    return torch


def make_iterable_dataset(loader: Any) -> Any:
    """Build a torch IterableDataset class bound to a loader factory.

    ``loader`` may be a :class:`StreamingTrainingDataLoader` instance or a
    zero-arg callable returning one (a factory is required for multi-worker
    use, because DataLoader workers each need their own loader instance).
    """

    torch = require_torch()

    def _resolve_loader():
        return (
            loader()
            if callable(loader) and not hasattr(loader, "iter_samples")
            else loader
        )

    iterable_dataset_base: Any = torch.utils.data.IterableDataset

    class CloudaIterableDataset(iterable_dataset_base):
        def __init__(self, *, epoch: int = 0, drop_last: bool = False) -> None:
            self.epoch = epoch
            self.drop_last = drop_last

        def _topology(self) -> tuple[int, int, int, int]:
            resolved = _resolve_loader()
            workers = resolved.config.workers
            world_size, rank = workers.world_size, workers.rank
            info = torch.utils.data.get_worker_info()
            num_workers = info.num_workers if info else 1
            worker_id = info.id if info else 0
            if (
                world_size > 1
                and torch.distributed.is_available()
                and (torch.distributed.is_initialized())
            ):
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            return world_size, rank, num_workers, worker_id

        def __iter__(self) -> Iterator[dict[str, Any]]:
            resolved = _resolve_loader()
            world_size, rank, num_workers, worker_id = self._topology()
            from clouda_data.training_data.models import WorkerConfig
            from dataclasses import replace

            scoped = replace(
                resolved.config,
                workers=WorkerConfig(
                    world_size=world_size,
                    rank=rank,
                    num_workers=num_workers,
                    worker_id=worker_id,
                ),
            )
            per_worker = _scoped_loader(resolved, scoped)
            for sample in per_worker.iter_samples(epoch=self.epoch):
                yield {
                    "sample_id": sample.sample_id,
                    "shard_id": sample.shard_id,
                    "image_path": sample.image_path,
                    "text": sample.text,
                    "row": sample.row,
                }

    return CloudaIterableDataset


def _scoped_loader(base: Any, scoped_config: Any) -> Any:
    """Rebuild a loader with a worker-scoped config (cheap: index is shared)."""

    from pathlib import Path

    from clouda_data.training_data.loader import StreamingTrainingDataLoader

    return StreamingTrainingDataLoader(
        shard_index_path=base.index_path,
        loader_config=scoped_config,
        manifest_path=base._manifest_path,
        dataset_root=Path(base.root),
    )


def make_dataloader(
    loader: Any,
    *,
    batch_size: int | None = None,
    num_workers: int = 0,
    epoch: int = 0,
    **dataloader_kwargs: Any,
) -> Any:
    """Convenience wrapper returning a configured torch DataLoader."""

    torch = require_torch()
    dataset = make_iterable_dataset(loader)(epoch=epoch)
    kwargs: dict[str, Any] = {"batch_size": None}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    kwargs.update(dataloader_kwargs)
    return torch.utils.data.DataLoader(dataset, num_workers=num_workers, **kwargs)
