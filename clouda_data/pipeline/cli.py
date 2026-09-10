from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from clouda_contracts.storage import StorageRoots
from clouda_data.config.loader import load_config
from clouda_data.datasets.downloader import (
    download_dataset_sample,
    result_to_dict,
    verify_download,
)
from clouda_data.datasets.rasam_batch import (
    download_rasam_first_batch,
    plan_rasam_first_batch,
    plan_to_dict as rasam_plan_to_dict,
    result_to_dict as rasam_result_to_dict,
    verify_rasam_first_batch,
)
from clouda_data.datasets.registry import (
    estimate_source_download,
    get_source,
    list_sources,
    verify_license,
)
from clouda_data.datasets.reports import generate_download_report
from clouda_data.ingestion.file_inspection import inspect_file
from clouda_data.ingestion.registry import read_registry
from clouda_data.ingestion.workflow import (
    ingest_source_manifest,
    plan_to_dict,
    validate_source_manifest_file,
)
from clouda_data.locations import (
    default_data_config_path,
    default_foundation_registry_path,
    default_profile_dir,
    repository_root,
)
from clouda_data.pipeline.profiles import list_profile_paths, load_profile
from clouda_data.pretraining.config import PreparationConfig, load_preparation_config
from clouda_data.pretraining.sources import SourceDefinition
from clouda_data.pretraining import workflow as pretraining_workflow
from clouda_data.rendering import (
    RenderConfig,
    render_document,
    validate_render_manifest,
)
from clouda_data.distortion.workflow import (
    generate_preview,
    read_jsonl,
    run_distortion_batch,
    validate_distortion_manifest,
)
from clouda_data.distortion.checkpoints import RunCheckpointStore
from clouda_data.evaluation.execution import evaluate_manifest
from clouda_data.lifecycle import (
    archive_run,
    cleanup as lifecycle_cleanup,
    verify_archive,
)
from clouda_data.factory.adapters import (
    run_dir_to_dataset_manifest,
)
from clouda_data.training_data.cli import (
    register_training_data_commands,
)
from clouda_data.factory.cli import (
    command_profiles as factory_profiles_command,
    command_seeds as factory_seeds_command,
    command_verify as factory_verify_command,
)
from clouda_data.results.cli import (
    command_export as results_export_command,
    command_ingest as results_ingest_command,
    command_ingest_runs as results_ingest_runs_command,
    command_list_models as results_list_models_command,
    command_list_runs as results_list_runs_command,
    command_show_page as results_show_page_command,
    command_verify as results_verify_command,
    command_worst_pages as results_worst_pages_command,
)


def _project_root() -> Path:
    return repository_root() or Path.cwd().resolve()


def _dataset_workspace() -> Path:
    return StorageRoots.from_env().dataset_root


def _artifact_workspace() -> Path:
    return StorageRoots.from_env().artifact_root


def _cache_workspace() -> Path:
    return StorageRoots.from_env().cache_root


def _catalog_path(value: str | None) -> Path:
    return (
        Path(value).expanduser().resolve()
        if value
        else default_foundation_registry_path()
    )


def _profile_root(value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else default_profile_dir()


def inspect_config(args: argparse.Namespace) -> int:
    config = load_config(
        args.config or default_data_config_path(), _dataset_workspace()
    )
    print(
        json.dumps(
            {
                "render_dpi": config.runtime.render_dpi,
                "workers": config.runtime.workers,
                "aws_enabled": config.aws.enabled,
            },
            indent=2,
        )
    )
    return 0


def validate_project(args: argparse.Namespace) -> int:
    root = _project_root()
    required = ["clouda_data", "tests", "docs"]
    missing = [path for path in required if not (root / path).exists()]
    if missing:
        print("Missing required paths: " + ", ".join(missing), file=sys.stderr)
        return 1
    for profile_path in list_profile_paths():
        load_profile(profile_path)
    print("Project foundation validation passed.")
    return 0


def validate_input(args: argparse.Namespace) -> int:
    from clouda_data.ingestion.manifest import read_page_manifest

    records = read_page_manifest(args.manifest)
    print(f"Input manifest contains {len(records)} page record(s).")
    return 0


def inspect_source(args: argparse.Namespace) -> int:
    payload = [inspect_file(path, args.source_type).__dict__ for path in args.paths]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(item["ok"] for item in payload) else 1


def validate_source(args: argparse.Namespace) -> int:
    plan = validate_source_manifest_file(args.manifest, _dataset_workspace())
    print(json.dumps(plan_to_dict(plan), ensure_ascii=False, indent=2))
    return 0 if plan.ok else 1


def ingest(args: argparse.Namespace) -> int:
    plan = ingest_source_manifest(
        args.manifest, _dataset_workspace(), dry_run=args.dry_run
    )
    print(json.dumps(plan_to_dict(plan), ensure_ascii=False, indent=2))
    return 0 if plan.ok else 1


def list_ingested(args: argparse.Namespace) -> int:
    root = _dataset_workspace()
    page_manifest = root / "data/manifests/page_manifest.json"
    registry = root / "data/manifests/file_registry.json"
    payload = {
        "pages": (
            json.loads(page_manifest.read_text(encoding="utf-8"))
            if page_manifest.exists()
            else []
        ),
        "files": read_registry(registry),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def find_duplicates(args: argparse.Namespace) -> int:
    registry = read_registry(_dataset_workspace() / "data/manifests/file_registry.json")
    by_checksum: dict[str, list[dict]] = {}
    for item in registry:
        by_checksum.setdefault(item["checksum"], []).append(item)
    duplicates = {
        checksum: items
        for checksum, items in by_checksum.items()
        if len(items) > 1 or any(item.get("duplicate_of") for item in items)
    }
    print(json.dumps(duplicates, ensure_ascii=False, indent=2))
    return 0


def generate_ingestion_report(args: argparse.Namespace) -> int:
    root = _dataset_workspace()
    if args.manifest:
        plan = validate_source_manifest_file(args.manifest, root)
        payload = plan_to_dict(plan)
    else:
        payload = {
            "pages": (
                json.loads(
                    (root / "data/manifests/page_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                if (root / "data/manifests/page_manifest.json").exists()
                else []
            ),
            "files": read_registry(root / "data/manifests/file_registry.json"),
        }
    out = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _artifact_workspace() / "reports" / "ingestion_report.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out))
    return 0


def list_dataset_sources(args: argparse.Namespace) -> int:
    sources = list_sources(_catalog_path(args.registry))
    if args.classification:
        sources = [
            source
            for source in sources
            if source["classification"] == args.classification
        ]
    payload = [
        {
            "source_id": source["source_id"],
            "name": source["name"],
            "classification": source["classification"],
            "license": source["license"],
            "commercial_use_status": source["commercial_use_status"],
            "risk_level": source["risk_level"],
        }
        for source in sources
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def describe_dataset_source(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            get_source(args.source_id, _catalog_path(args.registry)),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def verify_dataset_license(args: argparse.Namespace) -> int:
    result = verify_license(get_source(args.source_id, _catalog_path(args.registry)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["sample_download_allowed"] else 1


def estimate_download(args: argparse.Namespace) -> int:
    source = get_source(args.source_id, _catalog_path(args.registry))
    print(
        json.dumps(
            estimate_source_download(source, sample_only=not args.full_dataset),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def download_dataset_sample_cli(args: argparse.Namespace) -> int:
    result = download_dataset_sample(
        args.source_id,
        project_root=_dataset_workspace(),
        registry_path=_catalog_path(args.registry),
        max_bytes=args.max_bytes,
        dry_run=args.dry_run,
    )
    print(json.dumps(result_to_dict(result), ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


def resume_download(args: argparse.Namespace) -> int:
    return download_dataset_sample_cli(args)


def verify_download_cli(args: argparse.Namespace) -> int:
    result = verify_download(args.source_id, project_root=_dataset_workspace())
    print(json.dumps(result_to_dict(result), ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


def generate_download_report_cli(args: argparse.Namespace) -> int:
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _artifact_workspace() / "reports" / "dataset_download_report.json"
    )
    out = generate_download_report(_dataset_workspace(), output)
    print(str(out))
    return 0


def plan_rasam_first_batch_cli(args: argparse.Namespace) -> int:
    plan = plan_rasam_first_batch(
        _dataset_workspace(), batch_size=args.pages, max_bytes=args.max_bytes
    )
    print(json.dumps(rasam_plan_to_dict(plan), ensure_ascii=False, indent=2))
    return 0 if plan.ok else 1


def download_rasam_first_batch_cli(args: argparse.Namespace) -> int:
    result = download_rasam_first_batch(
        _dataset_workspace(), batch_size=args.pages, max_bytes=args.max_bytes
    )
    print(json.dumps(rasam_result_to_dict(result), ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


def verify_rasam_first_batch_cli(args: argparse.Namespace) -> int:
    result = verify_rasam_first_batch(_dataset_workspace())
    print(json.dumps(rasam_result_to_dict(result), ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


def list_profiles(args: argparse.Namespace) -> int:
    for path in list_profile_paths(_profile_root(args.config_dir)):
        profile = load_profile(path)
        print(profile["name"])
    return 0


def describe_profile(args: argparse.Namespace) -> int:
    root = _profile_root(args.config_dir)
    candidates = [
        root / f"{args.name}.yaml",
        root / f"{args.name}.yml",
        root / f"{args.name}.json",
    ]
    path = next(
        (candidate for candidate in candidates if candidate.is_file()), candidates[0]
    )
    profile = load_profile(path)
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    return 0


def preview(args: argparse.Namespace) -> int:
    if not args.synthetic_test:
        print(
            "Only --synthetic-test previews are enabled during foundation preparation.",
            file=sys.stderr,
        )
        return 2
    out = _artifact_workspace() / "previews" / "synthetic_preview.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "Synthetic preview placeholder only. No real distorted page was generated.\n",
        encoding="utf-8",
    )
    print(str(out))
    return 0


def dry_run(args: argparse.Namespace) -> int:
    validate_project(args)
    print(
        "Dry run completed. No ingestion, distortion, training, or AWS action was executed."
    )
    return 0


def status(args: argparse.Namespace) -> int:
    status_path = _project_root() / "docs" / "data_foundation" / "README.md"
    print(
        status_path.read_text(encoding="utf-8")
        if status_path.exists()
        else "Data-foundation overview is missing."
    )
    return 0


def cleanup(args: argparse.Namespace) -> int:
    targets = [_cache_workspace() / "temporary"]
    for target in targets:
        if target.exists():
            for child in target.iterdir():
                if child.is_file():
                    child.unlink()
    print("Cleaned temporary files only.")
    return 0


def run_tests(args: argparse.Namespace) -> int:
    return subprocess.call(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."]
    )


def system_inspection(args: argparse.Namespace) -> int:
    payload = {
        "platform": platform.platform(),
        "python": sys.version,
        "git": shutil.which("git"),
        "docker": shutil.which("docker"),
        "wsl": shutil.which("wsl"),
        "tesseract": shutil.which("tesseract"),
    }
    print(json.dumps(payload, indent=2))
    return 0


def render_cli(args: argparse.Namespace) -> int:
    config = RenderConfig(
        dpi=args.dpi,
        output_format=args.output_format,
        color_mode=args.color_mode,
        max_dimension=args.max_dimension,
        max_pixels=args.max_pixels,
        start_page=args.start_page,
        end_page=args.end_page,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    path = render_document(
        args.source,
        output_root=args.output_root,
        config=config,
        run_id=args.run_id,
    )
    print(str(path))
    return 0


def render_status_cli(args: argparse.Namespace) -> int:
    records = read_jsonl(args.manifest)
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("status", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"records": len(records), "statuses": counts}, indent=2))
    return 0


def render_validate_cli(args: argparse.Namespace) -> int:
    report = validate_render_manifest(args.manifest)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def _resolve_profile(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    root = default_profile_dir()
    choices = [root / f"{value}.yaml", root / f"{value}.yml", root / f"{value}.json"]
    result = next((path for path in choices if path.is_file()), None)
    if result is None:
        raise FileNotFoundError(f"Unknown distortion profile: {value}")
    return result.resolve()


def distort_cli(args: argparse.Namespace) -> int:
    path = run_distortion_batch(
        args.input_manifest,
        _resolve_profile(args.profile),
        output_root=args.output_root,
        seed=args.seed,
        variants=args.variants,
        max_pages=args.maximum_pages,
        allow_large_run=args.allow_large_run,
        dry_run=args.dry_run,
        resume=args.resume,
        fail_fast=args.fail_fast,
        conflict_policy=args.overwrite_policy,
        interrupt_after=args.interrupt_after,
        start_page=args.start_page,
        end_page=args.end_page,
        include_dataset_ids=set(args.include_dataset_ids),
        exclude_dataset_ids=set(args.exclude_dataset_ids),
        maximum_output_bytes=args.maximum_bytes,
        workers=args.workers,
        max_retries=args.max_retries,
        stale_after_seconds=args.stale_after_seconds,
    )
    print(str(path))
    return 0


def distort_status_cli(args: argparse.Namespace) -> int:
    records = read_jsonl(args.manifest)
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("status", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    payload: dict[str, object] = {"records": len(records), "statuses": counts}
    checkpoint_path = Path(args.manifest).resolve().parent / "checkpoints.sqlite3"
    if checkpoint_path.is_file():
        run_manifest = checkpoint_path.parent / "run_manifest.v1.json"
        if run_manifest.is_file():
            run_id = str(
                json.loads(run_manifest.read_text(encoding="utf-8")).get("run_id", "")
            )
            payload["checkpoint"] = RunCheckpointStore(checkpoint_path).summary(run_id)
    print(json.dumps(payload, indent=2))
    return 0


def distort_validate_cli(args: argparse.Namespace) -> int:
    report = validate_distortion_manifest(args.manifest, quarantine=args.quarantine)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def distort_preview_cli(args: argparse.Namespace) -> int:
    path = generate_preview(
        args.manifest,
        limit=args.limit,
        difference=not args.no_difference,
        layout_overlay=args.layout_overlay,
    )
    print(str(path))
    return 0


def validate_distortion_profile_cli(args: argparse.Namespace) -> int:
    profile = load_profile(_resolve_profile(args.profile))
    print(json.dumps({"profile_id": profile["name"], "valid": True}, indent=2))
    return 0


def evaluate_cli(args: argparse.Namespace) -> int:
    path = evaluate_manifest(args.manifest, args.output)
    print(str(path))
    return 0


def lifecycle_cleanup_cli(args: argparse.Namespace) -> int:
    action = args.command.removeprefix("cleanup-")
    report = lifecycle_cleanup(
        action,
        run_id=args.run_id,
        older_than_days=args.older_than_days,
        dry_run=not args.execute,
        confirmation=args.confirm,
    )
    print(json.dumps(report, indent=2))
    return 0


def archive_run_cli(args: argparse.Namespace) -> int:
    report = archive_run(
        args.run_root,
        dry_run=not args.execute,
        confirmation=args.confirm,
    )
    print(json.dumps(report, indent=2))
    return 0


def verify_archive_cli(args: argparse.Namespace) -> int:
    report = verify_archive(args.archive)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


# ---------------------------------------------------------------------------
# Pre-training dataset infrastructure (clouda_data.pretraining)
# ---------------------------------------------------------------------------


def _preparation_config(args: argparse.Namespace) -> PreparationConfig:
    return (
        load_preparation_config(args.config)
        if getattr(args, "config", None)
        else PreparationConfig()
    )


def _apply_seed(
    args: argparse.Namespace, config: PreparationConfig
) -> PreparationConfig:
    seed = getattr(args, "seed", None)
    if seed is not None and seed != config.split_seed:
        return PreparationConfig.from_mapping({**config.to_dict(), "split_seed": seed})
    return config


def _parse_ratios(value: str | None) -> dict[str, float] | None:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("Ratios must be train,validation,test,holdout.")
    names = ("train", "validation", "test", "holdout")
    return dict(zip(names, (float(part) for part in parts), strict=True))


def dataset_register_source_cli(args: argparse.Namespace) -> int:
    source = SourceDefinition(
        source_id=args.source_id,
        name=args.name,
        origin=args.origin or "",
        license=args.license or "unknown",
        languages=tuple(args.languages.split(",")) if args.languages else ("ar",),
        expected_format=args.expected_format,
        local_root=str(Path(args.local_root).resolve()) if args.local_root else "",
        remote_ref=args.remote_ref,
        adapter=args.adapter,
        enabled=not args.disabled,
        redistribution=args.redistribution,
        classification=args.classification,
        notes=args.notes or "",
    )
    path = pretraining_workflow.ensure_source_registered(args.workspace, source)
    print(json.dumps({"registered": source.source_id, "registry": str(path)}, indent=2))
    return 0


def dataset_scan_cli(args: argparse.Namespace) -> int:
    source = pretraining_workflow.resolve_source(args.workspace, args.source)
    report = pretraining_workflow.index_source(
        args.workspace,
        source,
        _preparation_config(args),
        resume=not args.fresh,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def dataset_validate_cli(args: argparse.Namespace) -> int:
    report = pretraining_workflow.validate_workspace(
        args.workspace, _preparation_config(args)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["counts"]["error"] == 0 else 1


def dataset_normalize_cli(args: argparse.Namespace) -> int:
    report = pretraining_workflow.normalize_workspace(
        args.workspace, _preparation_config(args)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def dataset_dedupe_cli(args: argparse.Namespace) -> int:
    report = pretraining_workflow.dedupe_workspace(args.workspace)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def dataset_split_cli(args: argparse.Namespace) -> int:
    report = pretraining_workflow.split_workspace(
        args.workspace,
        seed=args.seed,
        ratios=_parse_ratios(args.ratios),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def dataset_export_cli(args: argparse.Namespace) -> int:
    config = _preparation_config(args)
    if args.include_holdout:
        config = PreparationConfig.from_mapping(
            {**config.to_dict(), "include_holdout_in_export": True}
        )
    result = pretraining_workflow.export_workspace(args.workspace, config)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def dataset_stats_cli(args: argparse.Namespace) -> int:
    stats = pretraining_workflow.stats_workspace(args.workspace)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def dataset_prepare_cli(args: argparse.Namespace) -> int:
    config = _apply_seed(args, _preparation_config(args))
    report = pretraining_workflow.prepare_dataset(
        args.workspace,
        args.source,
        config,
        seed=args.seed,
        dry_run=args.dry_run,
        resume=not args.fresh,
        write_handoff=args.data_factory_handoff,
        handoff_profiles=[
            profile for profile in (args.handoff_profiles or "").split(",") if profile
        ],
        intended_output=args.intended_output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    passed = report["split"]["passed"] and not report["validation"]["counts"]["error"]
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# Clouda Data Factory (clouda_data.factory) — synthetic data generation
# ---------------------------------------------------------------------------


def factory_generate_cli(args: argparse.Namespace) -> int:
    from clouda_data.factory.cli import command_generate

    return command_generate(args)


def factory_run_cli(args: argparse.Namespace) -> int:
    from clouda_data.factory.cli import command_run

    return command_run(args)


def factory_profiles_cli(args: argparse.Namespace) -> int:
    return factory_profiles_command(args)


def factory_verify_cli(args: argparse.Namespace) -> int:
    return factory_verify_command(args)


def factory_seeds_cli(args: argparse.Namespace) -> int:
    return factory_seeds_command(args)


def factory_manifest_cli(args: argparse.Namespace) -> int:
    """Convert a Data Factory run into a canonical pre-training manifest."""
    path, report, manifest_hash = run_dir_to_dataset_manifest(
        args.run_dir,
        args.output,
        source_id=args.source_id,
    )
    print(
        json.dumps(
            {
                "manifest": str(path),
                "manifest_sha256": manifest_hash,
                "report": report.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m clouda_data.pipeline.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect-config")
    p.add_argument("--config")
    p.set_defaults(func=inspect_config)

    p = sub.add_parser("validate-project")
    p.set_defaults(func=validate_project)

    p = sub.add_parser("validate-input")
    p.add_argument("manifest")
    p.set_defaults(func=validate_input)

    p = sub.add_parser("inspect-source")
    p.add_argument("paths", nargs="+")
    p.add_argument(
        "--source-type",
        choices=[
            "pdf",
            "docx",
            "image",
            "text",
            "json_layout",
            "page_xml",
            "alto_xml",
            "ground_truth_json",
        ],
    )
    p.set_defaults(func=inspect_source)

    p = sub.add_parser("validate-source")
    p.add_argument("manifest")
    p.set_defaults(func=validate_source)

    p = sub.add_parser("ingest")
    p.add_argument("manifest")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=ingest)

    p = sub.add_parser("list-ingested")
    p.set_defaults(func=list_ingested)

    p = sub.add_parser("find-duplicates")
    p.set_defaults(func=find_duplicates)

    p = sub.add_parser("generate-ingestion-report")
    p.add_argument("--manifest")
    p.add_argument("--output")
    p.set_defaults(func=generate_ingestion_report)

    p = sub.add_parser("list-dataset-sources")
    p.add_argument("--registry")
    p.add_argument("--classification")
    p.set_defaults(func=list_dataset_sources)

    p = sub.add_parser("describe-dataset-source")
    p.add_argument("source_id")
    p.add_argument("--registry")
    p.set_defaults(func=describe_dataset_source)

    p = sub.add_parser("verify-dataset-license")
    p.add_argument("source_id")
    p.add_argument("--registry")
    p.set_defaults(func=verify_dataset_license)

    p = sub.add_parser("estimate-download")
    p.add_argument("source_id")
    p.add_argument("--registry")
    p.add_argument("--full-dataset", action="store_true")
    p.set_defaults(func=estimate_download)

    p = sub.add_parser("download-dataset-sample")
    p.add_argument("source_id")
    p.add_argument("--registry")
    p.add_argument("--max-bytes", type=int, default=100 * 1024 * 1024)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=download_dataset_sample_cli)

    p = sub.add_parser("resume-download")
    p.add_argument("source_id")
    p.add_argument("--registry")
    p.add_argument("--max-bytes", type=int, default=100 * 1024 * 1024)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=resume_download)

    p = sub.add_parser("verify-download")
    p.add_argument("source_id")
    p.set_defaults(func=verify_download_cli)

    p = sub.add_parser("generate-download-report")
    p.add_argument("--output")
    p.set_defaults(func=generate_download_report_cli)

    p = sub.add_parser("plan-rasam-first-batch")
    p.add_argument("--pages", type=int, default=100)
    p.add_argument("--max-bytes", type=int, default=1024 * 1024 * 1024)
    p.set_defaults(func=plan_rasam_first_batch_cli)

    p = sub.add_parser("download-rasam-first-batch")
    p.add_argument("--pages", type=int, default=100)
    p.add_argument("--max-bytes", type=int, default=1024 * 1024 * 1024)
    p.set_defaults(func=download_rasam_first_batch_cli)

    p = sub.add_parser("verify-rasam-first-batch")
    p.set_defaults(func=verify_rasam_first_batch_cli)

    p = sub.add_parser("list-profiles")
    p.add_argument("--config-dir")
    p.set_defaults(func=list_profiles)

    p = sub.add_parser("describe-profile")
    p.add_argument("name")
    p.add_argument("--config-dir")
    p.set_defaults(func=describe_profile)

    p = sub.add_parser("preview")
    p.add_argument("--synthetic-test", action="store_true")
    p.set_defaults(func=preview)

    p = sub.add_parser("dry-run")
    p.set_defaults(func=dry_run)

    p = sub.add_parser("status")
    p.set_defaults(func=status)

    p = sub.add_parser("cleanup")
    p.set_defaults(func=cleanup)

    p = sub.add_parser("test")
    p.set_defaults(func=run_tests)

    p = sub.add_parser("inspect-system")
    p.set_defaults(func=system_inspection)

    for name in ["render", "render-resume"]:
        p = sub.add_parser(name)
        p.add_argument("source")
        p.add_argument("--output-root")
        p.add_argument("--run-id")
        p.add_argument("--dpi", type=int, default=200)
        p.add_argument(
            "--output-format", choices=["png", "jpeg", "tiff", "webp"], default="png"
        )
        p.add_argument(
            "--color-mode", choices=["color", "grayscale", "binary"], default="color"
        )
        p.add_argument("--max-dimension", type=int, default=8000)
        p.add_argument("--max-pixels", type=int, default=40_000_000)
        p.add_argument("--start-page", type=int, default=1)
        p.add_argument("--end-page", type=int)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--resume", action="store_true", default=name == "render-resume")
        p.set_defaults(func=render_cli)

    p = sub.add_parser("render-status")
    p.add_argument("manifest")
    p.set_defaults(func=render_status_cli)
    p = sub.add_parser("render-validate")
    p.add_argument("manifest")
    p.set_defaults(func=render_validate_cli)

    for name in [
        "distort",
        "distort-page",
        "distort-batch",
        "distort-dry-run",
        "distort-resume",
    ]:
        p = sub.add_parser(name)
        p.add_argument("input_manifest")
        p.add_argument("--profile", required=True)
        p.add_argument("--output-root")
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--variants", type=int, default=1)
        p.add_argument("--start-page", type=int, default=1)
        p.add_argument("--end-page", type=int)
        p.add_argument("--include-dataset-ids", nargs="*", default=[])
        p.add_argument("--exclude-dataset-ids", nargs="*", default=[])
        p.add_argument(
            "--maximum-pages", type=int, default=1 if name == "distort-page" else 100
        )
        p.add_argument("--maximum-bytes", type=int, default=1024 * 1024 * 1024)
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--max-retries", type=int, default=2)
        p.add_argument("--stale-after-seconds", type=int, default=300)
        p.add_argument(
            "--overwrite-policy",
            choices=[
                "reject",
                "skip_identical",
                "version_new",
                "overwrite_only_with_explicit_flag",
            ],
            default="reject",
        )
        p.add_argument(
            "--resume", action="store_true", default=name == "distort-resume"
        )
        p.add_argument(
            "--dry-run", action="store_true", default=name == "distort-dry-run"
        )
        p.add_argument("--fail-fast", action="store_true")
        p.add_argument(
            "--report-format",
            choices=["json", "jsonl", "csv", "markdown"],
            default="jsonl",
        )
        p.add_argument(
            "--logging-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
        )
        p.add_argument("--allow-large-run", action="store_true")
        p.add_argument("--interrupt-after", type=int)
        p.set_defaults(func=distort_cli)

    p = sub.add_parser("distort-status")
    p.add_argument("manifest")
    p.set_defaults(func=distort_status_cli)
    for name in ["distort-validate", "validate"]:
        p = sub.add_parser(name)
        p.add_argument("manifest")
        p.add_argument("--quarantine", action="store_true")
        p.set_defaults(func=distort_validate_cli)
    for name in ["distort-preview", "preview-run"]:
        p = sub.add_parser(name)
        p.add_argument("manifest")
        p.add_argument("--limit", type=int, default=10)
        p.add_argument("--no-difference", action="store_true")
        p.add_argument("--layout-overlay", action="store_true")
        p.set_defaults(func=distort_preview_cli)
    p = sub.add_parser("list-distortion-profiles")
    p.add_argument("--config-dir")
    p.set_defaults(func=list_profiles)
    p = sub.add_parser("validate-distortion-profile")
    p.add_argument("profile")
    p.set_defaults(func=validate_distortion_profile_cli)

    for name in [
        "evaluate",
        "evaluate-manifest",
        "evaluate-run",
        "evaluation-report",
        "compare-models",
    ]:
        p = sub.add_parser(name)
        p.add_argument("manifest")
        p.add_argument("--output")
        p.set_defaults(func=evaluate_cli)

    for name in ["cleanup-preview", "cleanup-failed", "cleanup-temp"]:
        p = sub.add_parser(name)
        p.add_argument("--run-id")
        p.add_argument("--older-than-days", type=int, default=0)
        p.add_argument("--execute", action="store_true")
        p.add_argument("--confirm")
        p.set_defaults(func=lifecycle_cleanup_cli)
    p = sub.add_parser("archive-run")
    p.add_argument("run_root")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm")
    p.set_defaults(func=archive_run_cli)
    p = sub.add_parser("verify-archive")
    p.add_argument("archive")
    p.set_defaults(func=verify_archive_cli)

    # Pre-training dataset infrastructure ---------------------------------
    p = sub.add_parser("dataset-register-source")
    p.add_argument("workspace")
    p.add_argument("source_id")
    p.add_argument("--name", required=True)
    p.add_argument("--local-root")
    p.add_argument("--origin")
    p.add_argument("--license")
    p.add_argument("--languages", help="comma separated, e.g. ar,en")
    p.add_argument("--expected-format", default="image+text")
    p.add_argument("--remote-ref")
    p.add_argument("--adapter", default="auto")
    p.add_argument("--disabled", action="store_true")
    p.add_argument(
        "--redistribution",
        choices=["allowed", "attribution_required", "restricted", "forbidden"],
        default="restricted",
    )
    p.add_argument(
        "--classification",
        choices=["training_only", "public", "private", "restricted"],
        default="private",
    )
    p.add_argument("--notes")
    p.set_defaults(func=dataset_register_source_cli)

    p = sub.add_parser("dataset-scan")
    p.add_argument("source", help="registered source id or local directory")
    p.add_argument("workspace")
    p.add_argument("--config")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fresh", action="store_true", help="ignore existing index")
    p.set_defaults(func=dataset_scan_cli)

    p = sub.add_parser("dataset-validate")
    p.add_argument("workspace")
    p.add_argument("--config")
    p.set_defaults(func=dataset_validate_cli)

    p = sub.add_parser("dataset-normalize")
    p.add_argument("workspace")
    p.add_argument("--config")
    p.set_defaults(func=dataset_normalize_cli)

    p = sub.add_parser("dataset-dedupe")
    p.add_argument("workspace")
    p.set_defaults(func=dataset_dedupe_cli)

    p = sub.add_parser("dataset-split")
    p.add_argument("workspace")
    p.add_argument("--seed", type=int)
    p.add_argument("--ratios", help="train,validation,test,holdout floats")
    p.set_defaults(func=dataset_split_cli)

    p = sub.add_parser("dataset-export")
    p.add_argument("workspace")
    p.add_argument("--config")
    p.add_argument("--include-holdout", action="store_true")
    p.set_defaults(func=dataset_export_cli)

    p = sub.add_parser("dataset-stats")
    p.add_argument("workspace")
    p.set_defaults(func=dataset_stats_cli)

    p = sub.add_parser("dataset-prepare")
    p.add_argument("source", help="registered source id or local directory")
    p.add_argument("workspace")
    p.add_argument("--config")
    p.add_argument("--seed", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument(
        "--data-factory-handoff",
        action="store_true",
        help="emit a declarative handoff request for clouda-data-factory",
    )
    p.add_argument("--handoff-profiles", default="")
    p.add_argument("--intended-output")
    p.set_defaults(func=dataset_prepare_cli)

    # Clouda Data Factory ---------------------------------------------------
    p = sub.add_parser(
        "factory-generate",
        help="Generate clean + synthetic scan data from text/image inputs.",
    )
    p.add_argument(
        "inputs", nargs="+", type=Path, help="text/image files or directories"
    )
    p.add_argument("--output", type=Path, required=True, help="runs root directory")
    p.add_argument(
        "--profiles",
        default="05_old_book_medium",
        help="comma-separated profile names cycled across variants",
    )
    p.add_argument("--variants", type=int, default=1, help="scan variants per page")
    p.add_argument("--seed", type=int, default=20260831, help="base seed")
    p.add_argument(
        "--seed-mode",
        choices=["v1", "ocr_benchmark", "arabic_scan_factory"],
        default="v1",
    )
    p.add_argument(
        "--backend", default=None, help="render backend (default: first available)"
    )
    p.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    p.add_argument("--no-pdf", action="store_true", help="skip distorted PDF export")
    p.add_argument("--no-png", action="store_true", help="skip distorted PNG export")
    p.add_argument("--max-pages", type=int, default=8, help="pages per text document")
    p.add_argument(
        "--resume", action="store_true", help="resume an existing run directory"
    )
    p.add_argument("--run-id", default=None, help="explicit run id")
    p.set_defaults(func=factory_generate_cli)

    p = sub.add_parser(
        "factory-run",
        help="Zero-configuration one-command data factory.",
    )
    p.add_argument("input", type=Path, help="text/image/PDF file or mixed directory")
    p.add_argument(
        "output", type=Path, help="output root directory (runs created inside)"
    )
    p.add_argument(
        "--variants", type=int, default=None, help="scan variants per page (default: 2)"
    )
    p.add_argument(
        "--profiles",
        default=None,
        help="comma-separated profile names (default: automatic sensible pair)",
    )
    p.add_argument("--seed", type=int, default=20260831, help="base seed")
    p.add_argument(
        "--seed-mode",
        choices=["v1", "ocr_benchmark", "arabic_scan_factory"],
        default="v1",
    )
    p.add_argument(
        "--backend", default="auto", help="render backend (default: best available)"
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker processes (default: automatic)",
    )
    p.add_argument(
        "--no-pdf", action="store_true", help="skip image-only scan PDF export"
    )
    p.add_argument("--no-png", action="store_true", help="skip distorted PNG export")
    p.add_argument("--max-pages", type=int, default=8, help="pages per text document")
    p.add_argument(
        "--fresh",
        action="store_true",
        help="force a new run instead of auto-resuming the identical run",
    )
    p.set_defaults(func=factory_run_cli)

    p = sub.add_parser(
        "factory-profiles", help="List Data Factory profiles and backends."
    )
    p.set_defaults(func=factory_profiles_cli)

    p = sub.add_parser(
        "factory-verify", help="Verify hashes in a Data Factory run manifest."
    )
    p.add_argument("run_dir", type=Path)
    p.set_defaults(func=factory_verify_cli)

    p = sub.add_parser(
        "factory-seeds", help="Show Data Factory seed derivation vectors."
    )
    p.set_defaults(func=factory_seeds_cli)

    p = sub.add_parser(
        "factory-manifest",
        help="Convert a Data Factory run into a canonical pre-training manifest.",
    )
    p.add_argument("run_dir", type=Path, help="Data Factory run directory")
    p.add_argument("output", type=Path, help="output canonical manifest path")
    p.add_argument(
        "--source-id",
        default="clouda_data_factory",
        help="source id recorded in the canonical manifest",
    )
    p.set_defaults(func=factory_manifest_cli)

    # Clouda Results Store ---------------------------------------------------
    p = sub.add_parser(
        "results-ingest",
        help="Ingest the Arabic OCR benchmark manifest into a Results Store.",
    )
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
    p.set_defaults(func=results_ingest_command)

    p = sub.add_parser(
        "results-ingest-runs",
        help="Ingest benchmark leaderboard rows as runs and model records.",
    )
    p.add_argument("results_csv", type=Path)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--dataset-id", default="clouda-ocr-arabic-177")
    p.add_argument("--dataset-version", default="v1")
    p.set_defaults(func=results_ingest_runs_command)

    p = sub.add_parser(
        "results-verify", help="Verify Results Store bundle integrity for one run."
    )
    p.add_argument("run_id")
    p.add_argument("--store", type=Path, required=True)
    p.set_defaults(func=results_verify_command)

    p = sub.add_parser("results-list-runs", help="List Results Store runs.")
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--status", default=None)
    p.set_defaults(func=results_list_runs_command)

    p = sub.add_parser("results-list-models", help="List Results Store models.")
    p.add_argument("--store", type=Path, required=True)
    p.set_defaults(func=results_list_models_command)

    p = sub.add_parser(
        "results-show-page", help="Show one page, its ground truth, and predictions."
    )
    p.add_argument("run_id")
    p.add_argument("page_id")
    p.add_argument("--store", type=Path, required=True)
    p.set_defaults(func=results_show_page_command)

    p = sub.add_parser(
        "results-worst-pages", help="List worst pages by metric (descending)."
    )
    p.add_argument("run_id")
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--metric", default="cer@comparison_arabic_fold_digits")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=results_worst_pages_command)

    p = sub.add_parser(
        "results-export", help="Export a run bundle directory (portable copy)."
    )
    p.add_argument("run_id")
    p.add_argument("output", type=Path)
    p.add_argument("--store", type=Path, required=True)
    p.set_defaults(func=results_export_command)
    register_training_data_commands(sub)

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
