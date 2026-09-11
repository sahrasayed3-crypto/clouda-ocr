"""Synthetic Hunyuan-like fixtures (Phase 27).

Mimics ONLY the verified interfaces Clouda consumes — proves the adapter
bridge performs REAL backward/optimization through TorchTrainerBackend.
Does NOT prove real Hunyuan training (no weights, no GPU).
Label: synthetic compatibility fixture.
"""

from __future__ import annotations

import torch


class MockHunyuanProcessor:
    """Builds deterministic packed-ish batches (input_ids, attention, labels)."""

    def __init__(self, vocab_size: int = 512, seq_len: int = 32, seed: int = 17):
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


class MockHunyuanVisionTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, dtype=torch.float64)


class MockHunyuanProjector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 8, dtype=torch.float64)


class MockHunyuanLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 512):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 8, dtype=torch.float64)
        self.head = torch.nn.Linear(8, vocab_size, dtype=torch.float64)


class MockHunyuanVLForConditionalGeneration(torch.nn.Module):
    """Mock with upstream-like component attributes + HF-style loss output."""

    def __init__(self, vocab_size: int = 512):
        super().__init__()
        self.vision_tower = MockHunyuanVisionTower()
        self.mm_projector = MockHunyuanProjector()
        self.language_model = MockHunyuanLanguageModel(vocab_size)
        self.config = {"mock": True}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: object,
    ):
        hidden = self.language_model.embed(input_ids)
        # pretend vision context contributes
        vision = self.vision_tower.proj(
            torch.ones(hidden.shape[:2] + (8,), dtype=hidden.dtype)
        )
        hidden = hidden + self.mm_projector.fc(vision)
        logits = self.language_model.head(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )

        # HF-style output object (attribute access .loss)
        class Output:
            pass

        out = Output()
        out.loss = loss
        out.logits = logits
        return out
