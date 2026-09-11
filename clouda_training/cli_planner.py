"""Planner CLI: `clouda-training plan <config>` (design only, no training).

The existing `plan` subcommand (no-training execution plan) is legacy and
stays untouched; the planner commands live under `experiment-plan`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _planner_parser(subparsers: argparse._SubParsersAction) -> None:
    planner = subparsers.add_parser(
        "experiment-plan",
        help="Design a training experiment plan + resource estimate (no training).",
    )
    planner.add_argument("config", type=Path)
    planner.add_argument("--json", action="store_true")
    planner.add_argument(
        "--matrix", action="store_true", help="Generate the small candidate matrix."
    )
    planner.add_argument(
        "--vram-gb",
        type=float,
        default=None,
        help="Target per-GPU VRAM for the hardware envelope.",
    )
    planner.add_argument("--gpu-count", type=int, default=1)
    planner.add_argument(
        "--parameter-count",
        type=float,
        default=None,
        help="DECLARED parameter count (weights are not downloaded).",
    )
    planner.add_argument("--cost-per-gpu-hour", type=float, default=None)


def _load_experiment_config(path: Path) -> Any:
    from clouda_training.experiments.config import load_experiment_config

    return load_experiment_config(path)


def _planner_command(args: argparse.Namespace) -> int:
    try:
        config = _load_experiment_config(args.config)
    except Exception as exc:  # noqa: BLE001 — invalid invocation
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2

    from clouda_training.planner.models import (
        HardwareEnvelope,
        ParameterMetadata,
        StorageKind,
        TrainingMode,
        TrainingScale,
        default_profile,
    )
    from clouda_training.planner.planner import (
        build_experiment_matrix,
        build_experiment_plan,
        render_plan_report,
    )

    hardware = HardwareEnvelope(
        gpu_count=args.gpu_count or 1,
        per_gpu_vram_gb=args.vram_gb,
        storage_kind=StorageKind.UNKNOWN,
    )
    # Only parameter_count is DECLARED by the operator; trainable count is
    # deliberately NOT fabricated (trainable==total is false for selective/
    # LoRA modes — memory.py handles the full-finetune fallback explicitly).
    parameter_metadata = ParameterMetadata(
        parameter_count=int(args.parameter_count) if args.parameter_count else None,
    )
    profile = default_profile(
        TrainingScale.PILOT, training_mode=TrainingMode.SELECTIVE_FINETUNE
    )

    if args.matrix:
        plans = build_experiment_matrix(
            config,
            hardware=hardware,
            parameter_metadata=parameter_metadata,
        )
        if args.json:
            print(
                json.dumps([p.to_dict() for p in plans], ensure_ascii=False, indent=2)
            )
        else:
            print(f"EXPERIMENT MATRIX ({len(plans)} candidates)")
            for index, plan in enumerate(plans):
                print(
                    f"{chr(65 + index)} — {plan.profile.scale.value}: "
                    f"steps={plan.step_plan.planned_optimizer_steps} "
                    f"eff_batch={plan.step_plan.effective_batch_size} "
                    f"fit={plan.resources.fit.value if plan.resources else 'UNKNOWN'}"
                )
        return 0

    plan = build_experiment_plan(
        config, profile, hardware=hardware, parameter_metadata=parameter_metadata
    )
    if args.json:
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_plan_report(plan, model_label=config.model.model_id))
    return 0


__all__ = ["_planner_command", "_planner_parser"]
