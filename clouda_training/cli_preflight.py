"""Preflight CLI: `clouda-training preflight <config>` (no training starts).

Exit codes: 0 = READY / READY_WITH_WARNINGS, 1 = NOT_READY,
2 = invalid invocation/config (unreadable config file, bad path).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_REPORT_WIDTH = 26


def _preflight_parser(subparsers: argparse._SubParsersAction) -> None:
    pre = subparsers.add_parser(
        "preflight",
        help="Validate one training experiment config is safe/ready (no training).",
    )
    pre.add_argument("config", type=Path)
    pre.add_argument("--json", action="store_true", help="Machine-readable output.")
    pre.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as blockers (READY_WITH_WARNINGS -> exit 1).",
    )
    pre.add_argument(
        "--no-write-probe",
        action="store_true",
        help="Skip the output-directory writability probe.",
    )


def _load_experiment_config(path: Path) -> Any:
    from clouda_training.experiments.config import load_experiment_config

    return load_experiment_config(path)


def _render_human(payload: dict[str, Any]) -> None:
    print("TRAINING PREFLIGHT")
    print()
    for section in payload.get("sections", []):
        worst = section.get("worst_status")
        print(f"{section['name'] + ' ' + str(worst):.<{_REPORT_WIDTH}}")
        for check in section.get("checks", []):
            marker = {
                "PASS": "[PASS]",
                "WARN": "[WARN]",
                "FAIL": "[FAIL]",
                "SKIP": "[SKIP]",
                "UNAVAILABLE": "[UNAVAIL]",
            }.get(check["status"], "[?????]")
            print(f"  {marker} {check['name']}: {check.get('detail', '')}")
        print()
    plan = payload.get("training_plan")
    if plan:
        print("TRAINING PLAN")
        for key in (
            "effective_batch_size",
            "micro_batches_per_epoch",
            "optimizer_steps_per_epoch",
            "planned_optimizer_steps",
            "estimated_checkpoint_count",
            "world_size",
        ):
            print(f"  {key}: {plan.get(key)}")
        for note in plan.get("notes", []):
            print(f"  note: {note}")
        print()
    print(f"STATUS: {payload['final_status']}")
    if payload.get("blockers"):
        print()
        print("BLOCKERS:")
        for blocker in payload["blockers"]:
            print(f"- {blocker['check_name']}: {blocker['reason']}")
    if payload.get("warnings"):
        print()
        print("WARNINGS:")
        for warning in payload["warnings"]:
            print(f"- {warning['check_name']}: {warning['message']}")


def _preflight_command(args: argparse.Namespace) -> int:
    config_path: Path = args.config
    try:
        config = _load_experiment_config(config_path)
    except Exception as exc:  # noqa: BLE001 — report any config failure as exit 2
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"cannot load experiment config {config_path}: {exc}",
                },
                indent=2,
            )
        )
        return 2

    from clouda_training.preflight.orchestrator import run_preflight

    report = run_preflight(config, write_probe=not args.no_write_probe)
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _render_human(payload)

    if payload["final_status"] == "NOT_READY":
        return 1
    if args.strict and payload["final_status"] == "READY_WITH_WARNINGS":
        return 1
    return 0


__all__ = ["_preflight_command", "_preflight_parser"]
