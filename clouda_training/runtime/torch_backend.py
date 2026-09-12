"""Real PyTorch training backend for Clouda OCR.

Executes genuine optimization steps: zero_grad -> forward -> loss.backward()
-> gradient accumulation -> clipping -> optimizer.step() -> scheduler.step()
-> checkpoint. Deterministic given the experiment seed; full state is
captured for exact resume.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from clouda_training.experiments.config import ExperimentConfig
from clouda_training.experiments.metrics import MetricLogger
from clouda_training.runtime.adapter import ModelAdapter, SyntheticLinearAdapter
from clouda_training.runtime.backend import (
    TrainingState,
    require_torch,
)
from clouda_training.runtime.checkpoint_torch import (
    load_torch_state,
    save_torch_state,
    state_file_path,
)
from clouda_training.runtime.rng import capture_rng_state, restore_rng_state


@dataclass(frozen=True)
class BackendStepResult:
    step: int
    loss: float
    learning_rate: float
    grad_norm: float | None


class TorchTrainerBackend:
    """Genuine PyTorch optimization loop (CPU-first, device-configurable)."""

    def __init__(
        self,
        config: ExperimentConfig,
        metrics: MetricLogger,
        checkpoints: Any,
        *,
        adapter: ModelAdapter | None = None,
        fail_at_step: int | None = None,
        interrupt_at_step: int | None = None,
    ) -> None:
        require_torch()
        import torch

        self.torch = torch
        self.config = config
        self.metrics = metrics
        self.checkpoints = checkpoints
        self.adapter = adapter or SyntheticLinearAdapter()
        self.fail_at_step = fail_at_step
        self.interrupt_at_step = interrupt_at_step
        self.device = torch.device(config.runtime.device)
        self.model = self.adapter.build_model(config).to(self.device)
        self._build_optimizer_and_scheduler()
        self._scheduler_configured = config.training.scheduler != "none"
        self._rng_state_at_init = capture_rng_state()

    # ------------------------------------------------------------------ setup
    def _build_optimizer_and_scheduler(self) -> None:
        import torch

        params = self.adapter.trainable_parameters(self.model)
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        scheduler_name = self.config.training.scheduler
        warmup = self.config.training.warmup_steps
        if scheduler_name == "linear":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda step: min(1.0, (step + 1) / max(warmup, 1)) if warmup else 1.0,
            )
        elif scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(self.config.training.max_steps or 100, 1)
            )
        elif scheduler_name != "none":
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    # ------------------------------------------------------------------ batch
    def _step_batch(self, step: int, accumulation_index: int):
        seed = self.config.training.seed * 1_000_003 + step * 31 + accumulation_index
        return self.adapter.make_batch(
            step=step,
            batch_size=self.config.training.batch_size,
            seed=seed,
        )

    # ------------------------------------------------------------- train loop
    def train(self, *, start_step: int = 0) -> Any:
        from clouda_training.experiments.trainer import TrainerResult

        if start_step > 0:
            self._restore_from_latest_checkpoint(start_step)
        started = time.monotonic()
        maximum = self.config.training.max_steps or self.config.training.epochs * 5
        accumulation = max(self.config.training.gradient_accumulation_steps, 1)
        torch = self.torch

        for step in range(start_step + 1, maximum + 1):
            if step == self.fail_at_step:
                raise RuntimeError(f"Torch trainer injected failure at step {step}")
            if step == self.interrupt_at_step:
                raise KeyboardInterrupt()
            self.optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            micro_losses: list[float] = []
            for micro in range(accumulation):
                batch = self._step_batch(step, micro)
                # Adapter-agnostic: batches may be (inputs, targets) tuples or
                # model-specific dicts (e.g. Hunyuan multimodal fields).
                if (
                    isinstance(batch, tuple)
                    and len(batch) == 2
                    and hasattr(batch[0], "to")
                ):
                    batch = (batch[0].to(self.device), batch[1].to(self.device))
                elif isinstance(batch, dict):
                    batch = {
                        k: (v.to(self.device) if hasattr(v, "to") else v)
                        for k, v in batch.items()
                    }
                loss = self.adapter.forward_loss(self.model, batch)
                (loss / accumulation).backward()
                running_loss += float(loss.detach()) / accumulation
                micro_losses.append(float(loss.detach()))

            grad_norm: float | None = None
            if self.config.training.max_grad_norm > 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.adapter.trainable_parameters(self.model),
                        self.config.training.max_grad_norm,
                    )
                )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            loss_value = running_loss
            current_metrics = {
                "loss": round(loss_value, 8),
                "learning_rate": learning_rate,
            }
            if grad_norm is not None:
                current_metrics["grad_norm"] = round(grad_norm, 8)

            if step % self.config.tracking.log_steps == 0:
                for name, value in current_metrics.items():
                    self.metrics.append(
                        step=step,
                        epoch=step / maximum * self.config.training.epochs,
                        split="train",
                        metric_name=name,
                        value=value,
                    )

            if (
                self.config.checkpoint.save_strategy == "steps"
                and step % self.config.checkpoint.save_steps == 0
            ):
                self._save_checkpoint(step=step, metrics=current_metrics)

        return TrainerResult(maximum, start_step, time.monotonic() - started)

    # ------------------------------------------------------------- checkpoints
    def _save_checkpoint(self, *, step: int, metrics: dict[str, float]) -> None:
        payload = {
            "model": self.adapter.model_state(self.model),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "rng": capture_rng_state(),
            "step": step,
        }
        # Framework CheckpointManager (when present) records metadata, integrity,
        # retention and best tracking. Our state.bin is written next to it and
        # registered in its metadata.json for resume validation.
        info = self.checkpoints.save(step=step, epoch=step, metrics=metrics)
        directory = info.path if info is not None else self._fallback_ckpt_dir(step)
        digest = save_torch_state(directory, payload)

        metadata_path = directory / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        metadata["torch_state_file"] = state_file_path(directory).name
        metadata["torch_state_sha256"] = digest
        # Adapter identity (id/version/upstream revision/trainable flags) rides
        # in checkpoint metadata; no machine-specific paths are recorded.
        identity = getattr(self.adapter, "identity", None)
        if callable(identity):
            metadata["adapter_identity"] = identity()
        tmp = metadata_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(metadata), encoding="utf-8", newline="\n")
        os.replace(tmp, metadata_path)

    def _fallback_ckpt_dir(self, step: int):
        """Directory for direct-backend usage (framework-less probes/tests)."""
        base = self.config.runtime.output_root / "_backend_probe" / "checkpoints"
        directory = base / f"step-{step:08d}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _latest_torch_checkpoint(self) -> tuple[int, dict[str, Any]] | None:

        items = self.checkpoints
        latest = items.latest()
        if latest is None:
            return None
        metadata_path = latest.path / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "torch_state_sha256" not in metadata:
            return None
        payload = load_torch_state(
            latest.path, expected_sha256=metadata["torch_state_sha256"]
        )
        return int(payload["step"]), payload

    def _restore_from_latest_checkpoint(self, start_step: int) -> None:
        """Restore model/optimizer/scheduler/RNG from the newest checkpoint.

        Pick the checkpoint with the greatest step <= start_step so an
        interrupted run resumes from exactly where it stopped.
        """

        latest = self.checkpoints.latest()
        if latest is None:
            return
        metadata_path = latest.path / "metadata.json"
        if not metadata_path.is_file():
            return
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = metadata.get("torch_state_sha256")
        if digest is None:
            return  # framework-only checkpoint (no torch state recorded)
        payload = load_torch_state(latest.path, expected_sha256=digest)
        # Reject resumes whose adapter identity changed since the checkpoint.
        recorded_identity = metadata.get("adapter_identity")
        if recorded_identity is not None:
            identity = getattr(self.adapter, "identity", None)
            current_identity = identity() if callable(identity) else None
            if current_identity is not None and current_identity != recorded_identity:
                raise RuntimeError(
                    "checkpoint adapter identity mismatch — refusing to resume "
                    f"with a different adapter/config: {recorded_identity} != {current_identity}"
                )
        self.adapter.load_model_state(self.model, payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and payload.get("scheduler") is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        if payload.get("rng"):
            restore_rng_state(payload["rng"])

    # ------------------------------------------------------------------ state
    def capture_state(self, step: int = 0) -> TrainingState:
        payload = {
            "model": self.adapter.model_state(self.model),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "rng": capture_rng_state(),
            "step": int(step),
        }
        step_value = int(step)
        return TrainingState(step=step_value, epoch=float(step_value), payload=payload)

    def restore_state(self, state: TrainingState) -> None:
        payload = state.payload
        self.adapter.load_model_state(self.model, payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and payload.get("scheduler") is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        if payload.get("rng"):
            restore_rng_state(payload["rng"])
