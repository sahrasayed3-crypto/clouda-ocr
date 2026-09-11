"""Model adapter contract for the real training runtime.

Adapters decouple the trainer backend from any specific OCR model. A future
HunyuanOCR-1.5 adapter implements the same protocol; the synthetic adapter
here exists to prove real optimization end-to-end on CPU without downloading
any model.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from clouda_training.experiments.config import ExperimentConfig


@runtime_checkable
class ModelAdapter(Protocol):
    """Contract every trainable-model adapter must satisfy."""

    def build_model(self, config: ExperimentConfig) -> Any:
        """Construct a fresh model instance (deterministic given seed)."""
        ...

    def forward_loss(self, model: Any, batch: Any) -> Any:
        """Run forward pass and return a scalar loss tensor."""
        ...

    def trainable_parameters(self, model: Any) -> list[Any]:
        """Return the parameter list to optimize."""
        ...

    def model_state(self, model: Any) -> Any:
        """Extract serializable model state."""
        ...

    def load_model_state(self, model: Any, state: Any) -> None:
        """Restore model state in place."""
        ...

    def make_batch(self, step: int, batch_size: int, seed: int) -> Any:
        """Build the deterministic training batch for a given step."""
        ...


def build_synthetic_dataset_pairs(
    count: int, seed: int
) -> list[tuple[list[float], float]]:
    """Deterministic tiny regression dataset: y = w·x + b + tiny noise.

    Fixed ground-truth weights so learning is verifiable.
    """
    import random

    rng = random.Random(seed)
    true_w = [0.5, -1.25, 2.0]
    true_b = 0.75
    pairs: list[tuple[list[float], float]] = []
    for _ in range(count):
        x = [rng.uniform(-1.0, 1.0) for _ in range(3)]
        y = sum(w * xi for w, xi in zip(true_w, x)) + true_b + rng.uniform(-0.01, 0.01)
        pairs.append((x, y))
    return pairs


class SyntheticLinearAdapter:
    """Tiny deterministic linear regression adapter for CPU proof runs.

    3 features -> 1 output. Deliberately trivial so full-batch gradient
    descent converges visibly within a handful of steps.
    """

    input_features = 3

    def build_model(self, config: ExperimentConfig) -> Any:
        from clouda_training.runtime.backend import require_torch

        require_torch()
        import torch

        generator = torch.Generator().manual_seed(config.training.seed)
        model = torch.nn.Linear(self.input_features, 1, dtype=torch.float64)
        with torch.no_grad():
            for param in model.parameters():
                param.copy_(
                    torch.randn(param.shape, generator=generator, dtype=torch.float64)
                )
        return model

    def forward_loss(self, model: Any, batch: Any) -> Any:
        import torch

        inputs, targets = batch
        predictions = model(inputs)
        return torch.nn.functional.mse_loss(predictions, targets)

    def trainable_parameters(self, model: Any) -> list[Any]:
        return list(model.parameters())

    def model_state(self, model: Any) -> Any:
        return {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    def load_model_state(self, model: Any, state: Any) -> None:
        model.load_state_dict(state)

    def make_batch(self, step: int, batch_size: int, seed: int) -> Any:
        import torch

        pairs = build_synthetic_dataset_pairs(count=max(batch_size * 8, 32), seed=seed)
        # Deterministic per-step slice (step-indexed rotation = epoch-free streaming)
        start = (step * batch_size) % len(pairs)
        selected = [pairs[(start + i) % len(pairs)] for i in range(batch_size)]
        inputs = torch.tensor([row[0] for row in selected], dtype=torch.float64)
        targets = torch.tensor([[row[1]] for row in selected], dtype=torch.float64)
        return inputs, targets
