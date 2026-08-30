from __future__ import annotations

import argparse
import json
from pathlib import Path

from clouda_contracts.storage import StorageRoots
from clouda_data.locations import default_catalog_path
from clouda_training.config.models import load_training_config
from clouda_training.planner import plan_training
from clouda_training.exporter import (
    SUPPORTED_FORMATS,
    export_training_data,
    training_statistics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clouda-training")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Create a no-training execution plan.")
    plan.add_argument("--config", required=True, type=Path)
    plan.add_argument("--catalog", type=Path)
    plan.add_argument("--output", type=Path)
    export = subparsers.add_parser("export")
    export.add_argument("manifest", type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument(
        "--format", choices=sorted(SUPPORTED_FORMATS), default="generic_jsonl"
    )
    export.add_argument("--seed", type=int, default=20260723)
    export.add_argument(
        "--purpose",
        choices=["commercial_training", "evaluation"],
        default="commercial_training",
    )
    export.add_argument("--benchmark-exclusion", action="append", default=[])
    export.add_argument("--max-contribution-per-source", type=int, default=100)
    export.add_argument("--no-balance", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    split = subparsers.add_parser("split")
    split.add_argument("manifest", type=Path)
    split.add_argument("--output", required=True, type=Path)
    split.add_argument("--seed", type=int, default=20260723)
    split.add_argument(
        "--purpose",
        choices=["commercial_training", "evaluation"],
        default="commercial_training",
    )
    statistics = subparsers.add_parser("statistics")
    statistics.add_argument("manifest", type=Path)
    estimate = subparsers.add_parser("estimate-storage")
    estimate.add_argument("manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"export", "split"}:
        result = export_training_data(
            args.manifest,
            args.output,
            export_format=getattr(args, "format", "generic_jsonl"),
            seed=args.seed,
            purpose=args.purpose,
            benchmark_exclusions=set(getattr(args, "benchmark_exclusion", [])),
            max_contribution_per_source=getattr(
                args, "max_contribution_per_source", 100
            ),
            balance=not getattr(args, "no_balance", False),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if not result["document_leakage"] else 1
    if args.command in {"statistics", "estimate-storage"}:
        result = training_statistics(args.manifest)
        if args.command == "estimate-storage":
            result = {
                "records": result["records"],
                "source_bytes": result["bytes"],
                "estimated_jsonl_bytes": max(1024, result["records"] * 2048),
                "training_started": False,
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate":
        result = training_statistics(args.manifest)
        print(
            json.dumps(
                {"valid": result["records"] > 0, **result}, ensure_ascii=False, indent=2
            )
        )
        return 0 if result["records"] > 0 else 1
    config = load_training_config(args.config)
    plan_result = plan_training(
        config,
        roots=StorageRoots.from_env(),
        catalog_path=args.catalog or default_catalog_path(),
    )
    payload = json.dumps(plan_result.to_dict(), ensure_ascii=False, indent=2)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
