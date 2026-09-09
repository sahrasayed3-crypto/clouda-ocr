from __future__ import annotations

import argparse
import csv
import io
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
from clouda_training.experiments import (
    ConfigError,
    RunStatus,
    compare_runs,
    list_checkpoints,
    list_runs,
    load_experiment_config,
    load_run,
    resume_run,
    run_experiment,
)


def _machine_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )


def _runs_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))


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

    validate_config = subparsers.add_parser(
        "validate-config", help="Validate and hash an experiment configuration."
    )
    validate_config.add_argument("config", type=Path)
    validate_config.add_argument("--override", action="append", default=[])
    _machine_flag(validate_config)
    for command, help_text in (
        ("run", "Run a configured adapter (mock adapters only in this build)."),
        ("dry-run", "Run the complete offline mock experiment lifecycle."),
    ):
        run_parser = subparsers.add_parser(command, help=help_text)
        run_parser.add_argument("config", type=Path)
        run_parser.add_argument("--override", action="append", default=[])
        _machine_flag(run_parser)
    listing = subparsers.add_parser("list", help="List filesystem-registered runs.")
    _runs_root(listing)
    listing.add_argument("--status", choices=[status.value for status in RunStatus])
    listing.add_argument("--tag", action="append", default=[])
    _machine_flag(listing)
    show = subparsers.add_parser("show", help="Show one run and its summary.")
    show.add_argument("run_id")
    _runs_root(show)
    _machine_flag(show)
    compare = subparsers.add_parser(
        "compare", help="Compare configs and metrics for two runs."
    )
    compare.add_argument("run_a")
    compare.add_argument("run_b")
    _runs_root(compare)
    compare.add_argument("--format", choices=["human", "json", "csv"], default="human")
    resume = subparsers.add_parser(
        "resume", help="Resume an interrupted or failed mock run."
    )
    resume.add_argument("run_id")
    _runs_root(resume)
    _machine_flag(resume)
    checkpoints = subparsers.add_parser(
        "checkpoints", help="List validated checkpoints for a run."
    )
    checkpoints.add_argument("run_id")
    _runs_root(checkpoints)
    _machine_flag(checkpoints)
    return parser


def _emit(payload: object, *, machine: bool, heading: str = "") -> None:
    if machine:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if heading:
        print(heading)
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                print(f"{item.get('run_id', '-')}: {item.get('status', '')}")
            else:
                print(item)
    elif isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
    else:
        print(payload)


def _comparison_csv(payload: dict) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["kind", "field", "run_a", "run_b"])
    for kind in ("config_differences", "metric_differences"):
        for field, values in payload[kind].items():
            writer.writerow([kind, field, values[0], values[1]])
    return output.getvalue()


def _experiment_command(args: argparse.Namespace) -> int:
    try:
        if args.command == "validate-config":
            config = load_experiment_config(args.config, overrides=args.override)
            _emit(
                {
                    "valid": True,
                    "config_hash": config.hash,
                    "experiment": config.experiment.name,
                },
                machine=args.json,
                heading="Experiment configuration is valid.",
            )
            return 0
        if args.command in {"run", "dry-run"}:
            overrides = list(args.override)
            if args.command == "dry-run":
                overrides.extend(["runtime.dry_run=true", "runtime.offline=true"])
            run = run_experiment(
                load_experiment_config(args.config, overrides=overrides)
            )
            _emit(run.to_dict(), machine=args.json, heading="Experiment run completed.")
            return 0
        if args.command == "list":
            status = RunStatus(args.status) if args.status else None
            run_payload = [
                run.to_dict()
                for run in list_runs(args.runs_root, status=status, tags=set(args.tag))
            ]
            _emit(run_payload, machine=args.json, heading="Experiment runs")
            return 0
        if args.command == "show":
            _emit(load_run(args.run_id, args.runs_root).to_dict(), machine=args.json)
            return 0
        if args.command == "compare":
            comparison_payload = compare_runs(
                load_run(args.run_a, args.runs_root),
                load_run(args.run_b, args.runs_root),
            )
            if args.format == "csv":
                print(_comparison_csv(comparison_payload), end="")
            else:
                _emit(
                    comparison_payload,
                    machine=args.format == "json",
                    heading="Run comparison",
                )
            return 0
        if args.command == "resume":
            run = resume_run(args.run_id, args.runs_root)
            _emit(run.to_dict(), machine=args.json, heading="Experiment run resumed.")
            return 0
        if args.command == "checkpoints":
            run = load_run(args.run_id, args.runs_root)
            _emit(
                [item.to_dict() for item in list_checkpoints(run.path)],
                machine=args.json,
                heading="Checkpoints",
            )
            return 0
    except (
        ConfigError,
        FileNotFoundError,
        PermissionError,
        ValueError,
        RuntimeError,
    ) as exc:
        _emit(
            {"valid": False, "error": str(exc), "error_type": type(exc).__name__},
            machine=getattr(args, "json", False)
            or getattr(args, "format", "") == "json",
        )
        return 2
    raise AssertionError(f"Unhandled experiment command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {
        "validate-config",
        "run",
        "dry-run",
        "list",
        "show",
        "compare",
        "resume",
        "checkpoints",
    }:
        return _experiment_command(args)
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
