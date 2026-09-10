"""Clouda Results Store command surface.

Follows the Data Factory pattern: implementations live here, and the commands
are registered on the unified ``clouda-data`` CLI (see
``clouda_data.pipeline.cli``) as ``results-ingest``, ``results-verify``,
``results-list-runs``, ``results-list-models``, ``results-show-page``,
``results-worst-pages``, and ``results-export``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .identity import RESULTS_SCHEMA_VERSION
from .ingest import (
    benchmark_manifest_to_pages,
    ocr_arabic_results_to_runs,
)
from .service import ResultsService
from .store import UnknownRecordError


def _emit(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_ingest(args: argparse.Namespace) -> int:
    """Ingest the canonical Arabic OCR benchmark metadata into a store."""

    svc = ResultsService(args.store)
    issues: list[str] = []
    registered_pages = 0

    dataset_id = args.dataset_id
    dataset_version = args.dataset_version

    pages = benchmark_manifest_to_pages(
        args.manifest,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        private_root_note=args.private_root_note,
    )
    from .models import BenchmarkDataset, Provenance

    dataset = BenchmarkDataset(
        dataset_id=dataset_id,
        version=dataset_version,
        name="Final Arabic OCR benchmark (metadata-only)",
        description="177 distorted pages derived from 100 clean source records.",
        page_count=len(pages),
        splits=tuple(sorted({page.split for page in pages})),
        provenance=Provenance(
            source_format="benchmarks.ocr_arabic.benchmark_manifest.v1",
            source_uri=Path(args.manifest).as_posix(),
            adapter_version="clouda.results.ingest.v1",
        ),
    )
    svc.store.save_dataset(dataset.to_dict())

    # A placeholder run groups the ingested pages (metadata-only benchmark:
    # predictions may be attached later from private raw evidence).
    run = svc.create_run(
        model_id=args.model_id or "benchmark-metadata",
        model_revision=args.model_revision or "unresolved",
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        split="unassigned",
        metadata={"ingest_source": str(args.manifest)},
        created_at="ingest",
    )
    run_id = run.run_id
    registered_pages = svc.add_pages(run_id, pages)
    _emit(
        {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "dataset": dataset.identity,
            "run_id": run_id,
            "pages_registered": registered_pages,
            "pages_total": len(pages),
            "issues": issues,
            "ok": not issues,
        }
    )
    return 0 if not issues else 1


def command_ingest_runs(args: argparse.Namespace) -> int:
    """Ingest benchmark leaderboard rows as runs + model records."""

    svc = ResultsService(args.store)
    models: list[Any] = []
    runs = ocr_arabic_results_to_runs(
        args.results_csv,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        model_records_out=models,
    )
    registered = 0
    for run in runs:
        svc.store.save_run_metadata(run.to_dict())
        registered += 1
    for model in models:
        svc.store.save_model(model.to_dict())
    _emit(
        {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "runs_registered": registered,
            "models_registered": len(models),
        }
    )
    return 0


def command_verify(args: argparse.Namespace) -> int:
    svc = ResultsService(args.store, read_only=True)
    try:
        report = svc.verify(args.run_id)
    except UnknownRecordError as exc:
        _emit({"ok": False, "error": str(exc)})
        return 2
    _emit(report)
    return 0 if report["ok"] else 1


def command_list_runs(args: argparse.Namespace) -> int:
    svc = ResultsService(args.store, read_only=True)
    runs = svc.list_runs(model_id=args.model, status=args.status)
    if args.dataset:
        runs = [run for run in runs if run.get("dataset_id") == args.dataset]
    _emit(runs)
    return 0


def command_list_models(args: argparse.Namespace) -> int:
    svc = ResultsService(args.store, read_only=True)
    _emit(svc.list_models())
    return 0


def command_show_page(args: argparse.Namespace) -> int:
    svc = ResultsService(args.store, read_only=True)
    try:
        page = svc.get_page(args.run_id, args.page_id)
        payload: dict[str, Any] = {"page": page.to_dict()}
        try:
            gt = svc.get_ground_truth(args.run_id, args.page_id)
            payload["ground_truth"] = gt.to_dict()
        except UnknownRecordError:
            payload["ground_truth"] = None
        predictions = svc.get_predictions(args.run_id, args.page_id)
        payload["predictions"] = [prediction.to_dict() for prediction in predictions]
    except UnknownRecordError as exc:
        _emit({"error": str(exc)})
        return 2
    _emit(payload)
    return 0


def command_worst_pages(args: argparse.Namespace) -> int:
    svc = ResultsService(args.store, read_only=True)
    rows = svc.get_worst_pages(args.run_id, metric=args.metric, limit=args.limit)
    _emit(rows)
    return 0


def command_export(args: argparse.Namespace) -> int:
    svc = ResultsService(args.store, read_only=True)
    try:
        target = svc.store.export_run_copy(args.run_id, args.output)
    except UnknownRecordError as exc:
        _emit({"error": str(exc)})
        return 2
    _emit({"exported": str(target)})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clouda results")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Ingest the Arabic OCR benchmark manifest.")
    p.add_argument("manifest", type=Path)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--dataset-id", default="clouda-ocr-arabic-177")
    p.add_argument("--dataset-version", default="v1")
    p.add_argument("--model-id", default=None)
    p.add_argument("--model-revision", default=None)
    p.add_argument(
        "--private-root-note",
        default=None,
        help="machine-local root kept under source_private (never portable)",
    )
    p.set_defaults(func=command_ingest)

    p = sub.add_parser(
        "ingest-runs", help="Ingest benchmark leaderboard rows as runs/models."
    )
    p.add_argument("results_csv", type=Path)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--dataset-id", default="clouda-ocr-arabic-177")
    p.add_argument("--dataset-version", default="v1")
    p.set_defaults(func=command_ingest_runs)

    p = sub.add_parser("verify", help="Verify bundle integrity for one run.")
    p.add_argument("run_id")
    p.add_argument("--store", type=Path, required=True)
    p.set_defaults(func=command_verify)

    p = sub.add_parser("list-runs", help="List runs.")
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--status", default=None)
    p.set_defaults(func=command_list_runs)

    p = sub.add_parser("list-models", help="List registered models.")
    p.add_argument("--store", type=Path, required=True)
    p.set_defaults(func=command_list_models)

    p = sub.add_parser("show-page", help="Show one page, its GT, and predictions.")
    p.add_argument("run_id")
    p.add_argument("page_id")
    p.add_argument("--store", type=Path, required=True)
    p.set_defaults(func=command_show_page)

    p = sub.add_parser("worst-pages", help="Worst pages by metric (descending).")
    p.add_argument("run_id")
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--metric", default="cer@comparison_arabic_fold_digits")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=command_worst_pages)

    p = sub.add_parser("export", help="Export a run bundle directory.")
    p.add_argument("run_id")
    p.add_argument("output", type=Path)
    p.add_argument("--store", type=Path, required=True)
    p.set_defaults(func=command_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
