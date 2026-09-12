"""Synthetic Qwen3-VL-like fixtures (offline/CPU/deterministic).

Mimics ONLY the verified interfaces Clouda consumes — proves the Qwen adapter
performs REAL backward/optimization through TorchTrainerBackend. Does NOT
prove real Qwen3-VL training (no weights, no GPU).
Label: synthetic compatibility fixture.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch


class MockQwenProcessor:
    """Builds deterministic batches (input_ids, attention, labels)."""

    def __init__(self, vocab_size: int = 512, seq_len: int = 32, seed: int = 29):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.seed = seed

    def build_training_batch(self, *, step: int, batch_size: int, seed: int):
        generator = torch.Generator().manual_seed(seed * 1_000_003 + step)
        input_ids = torch.randint(
            1, self.vocab_size, (batch_size, self.seq_len), generator=generator
        )
        attention = torch.ones_like(input_ids)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
        }


class MockQwenVisionMerger(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 8, dtype=torch.float64)


class MockQwenVisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, dtype=torch.float64)
        self.merger = MockQwenVisionMerger()


class MockQwenLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 512):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 8, dtype=torch.float64)
        self.head = torch.nn.Linear(8, vocab_size, dtype=torch.float64)


class MockQwen3VLForConditionalGeneration(torch.nn.Module):
    """Mock with the verified Qwen3-VL attribute layout + HF-style loss output.

    Layout (transformers >= 4.57 native):
      model.visual (with .merger), model.language_model, model.lm_head.
    """

    def __init__(self, vocab_size: int = 512):
        super().__init__()
        self.visual = MockQwenVisionModel()
        self.language_model = MockQwenLanguageModel(vocab_size)
        self.lm_head = torch.nn.Linear(vocab_size, vocab_size, dtype=torch.float64)
        self.config = {"mock": True}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: object,
    ):
        hidden = self.language_model.embed(input_ids)
        vision = self.visual.proj(
            torch.ones(hidden.shape[:2] + (8,), dtype=hidden.dtype)
        )
        hidden = hidden + self.visual.merger.fc(vision)
        logits = self.language_model.head(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )

        # HF-style output object (attribute access .loss) with pre-declared
        # attributes so static type checks see the shape.
        out: Any = SimpleNamespace(loss=loss, logits=logits)
        # keep lm_head in the gradient graph (verified layout parity) via a
        # zero-weighted vocab-space projection so the optimizer sees a grad.
        pooled = torch.zeros(1, self.lm_head.in_features, dtype=hidden.dtype)
        out.pooled_logits = self.lm_head(pooled)
        if loss is not None:
            out.loss = loss + 0.0 * out.pooled_logits.sum()
        return out
