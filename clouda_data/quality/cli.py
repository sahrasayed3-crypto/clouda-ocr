"""Operator CLI for the dataset quality gate (``clouda-quality``).

Conventions follow ``clouda_data/pipeline/cli.py``: argparse subcommands,
``_cmd_*(args) -> int`` handlers, JSON as default machine output, UTF-8
stdout reconfiguration for Arabic text on Windows.

Exit codes: 0 gate PASS (or --strict satisfied), 1 gate FAIL, 2 config/usage
error (bad config file, stale resume, missing manifest).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from clouda_data.quality.config import (
    QualityGateConfig,
    load_quality_gate_config,
)
from clouda_data.quality.derived import write_clean_manifest
from clouda_data.quality.gate import ALGORITHM_VERSIONS, run_quality_gate
from clouda_data.quality.manifest_adapter import manifest_sha256
from clouda_data.quality.run_state import (
    QualityRunState,
    checkpoint,
    start_or_resume,
)

PROG = "clouda-quality"


class QualityCliError(Exception):
    """Config/usage error -> exit code 2."""


def _gate_config(args: argparse.Namespace) -> QualityGateConfig:
    try:
        if getattr(args, "config", None):
            return load_quality_gate_config(str(args.config))
        return QualityGateConfig()
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        raise QualityCliError(f"invalid quality gate config: {exc}") from exc


def _print_payload(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_report_payload_text(payload))


def render_report_payload_text(payload: dict[str, Any]) -> str:
    """Human-readable text for an already-built report payload."""

    lines: list[str] = ["Dataset Health Report"]
    lines.append(f"Run: {payload.get('run_id', '')}")
    lines.append(f"Manifest: {payload.get('manifest_sha256', '')}")
    severity = payload.get("severity_counts", {})
    for key in ("critical", "error", "warning", "info"):
        if key in severity:
            lines.append(f"{key.upper()}: {severity[key]}")
    excluded = payload.get("exclusions", [])
    lines.append(f"Excluded samples: {len(excluded)}")
    lines.append(f"Quarantined samples: {payload.get('quarantine_count', 0)}")
    verdict = payload.get("verdict", "PASS")
    lines.append(f"QUALITY GATE: {verdict}")
    return "\n".join(lines)


def _run_state_identity(
    manifest_sha: str, config: QualityGateConfig
) -> QualityRunState:
    return QualityRunState(
        manifest_sha256=manifest_sha,
        row_count=0,
        config_identity=config.identity(),
        algorithm_versions=dict(ALGORITHM_VERSIONS),
        stage="scan",
        stage_cursor={},
        processed_count=0,
        run_id="",
        updated_at="",
    )


def _cmd_scan(args: argparse.Namespace) -> int:
    config = _gate_config(args)
    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise QualityCliError(f"manifest not found: {manifest}")

    manifest_sha = manifest_sha256(manifest)
    if args.resume:
        index_dir = Path(config.paths.index_dir or (manifest.parent / ".quality"))
        try:
            _state, resumed = start_or_resume(
                index_dir, _run_state_identity(manifest_sha, config)
            )
        except Exception as exc:  # noqa: BLE001 - stale/corrupt resume -> exit 2
            raise QualityCliError(f"resume rejected: {exc}") from exc
        if resumed:
            print(f"resuming scan for manifest {manifest_sha[:12]}...", file=sys.stderr)

    scan = run_quality_gate(
        str(manifest),
        config,
        max_samples=int(args.max_samples or 0),
        no_near_duplicates=bool(args.no_near_duplicates),
        cross_split_only=bool(args.cross_split_only),
    )

    if args.resume:
        index_dir = Path(config.paths.index_dir or (manifest.parent / ".quality"))
        checkpoint(
            index_dir,
            stage="scan",
            cursor={"manifest_sha256": scan.manifest_sha256},
            processed_count=len(scan.samples),
        )

    from clouda_data.quality.report import build_report_payload
    from clouda_data.quality.policy import quarantine_sample_ids

    # R2-M2 fix: the CLI always knows the protected rows (it ran the gate)
    # and must redact their finding details by default.
    payload = build_report_payload(
        scan.result,
        protected_ids=frozenset(quarantine_sample_ids(scan.samples, scan.exclusions)),
    )
    payload["verdict"] = scan.result.verdict.value
    payload["manifest_sha256"] = scan.manifest_sha256
    payload["quarantine_count"] = len(scan.quarantine_ids)

    # R4-L1 fix: atomic report write (matches manifest/run_state discipline).
    from clouda_data.pretraining.hashing import atomic_write_text

    if args.output:
        atomic_write_text(
            Path(args.output),
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
    else:
        _print_payload(payload, as_json=True)

    if args.strict and scan.result.verdict.value == "PASS_WITH_WARNINGS":
        return 1
    if scan.result.verdict.value == "FAIL":
        return 1
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    config = _gate_config(args)
    report_path = Path(args.report)
    if not report_path.is_file():
        raise QualityCliError(f"report not found: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        payload.get("config_identity")
        and payload["config_identity"] != config.identity()
    ):
        raise QualityCliError(
            "report was produced by a different gate config "
            f"({payload['config_identity']} != {config.identity()})"
        )
    verdict = payload.get("verdict", "")
    print(f"QUALITY GATE: {verdict}")
    return 1 if verdict == "FAIL" else 0


def _cmd_report(args: argparse.Namespace) -> int:
    report_path = Path(args.input)
    if not report_path.is_file():
        raise QualityCliError(f"report not found: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    fmt = getattr(args, "format", "text")
    if fmt == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_report_payload_text(payload))
    return 0


def _cmd_clean_manifest(args: argparse.Namespace) -> int:
    config = _gate_config(args)
    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise QualityCliError(f"manifest not found: {manifest}")
    if not args.output:
        raise QualityCliError("--output is required for clean-manifest")
    scan = run_quality_gate(str(manifest), config)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "excluded": len(scan.exclusions),
                    "quarantined": len(scan.quarantine_ids),
                    "verdict": scan.result.verdict.value,
                },
                ensure_ascii=False,
            )
        )
        return 0
    result = write_clean_manifest(
        manifest,
        scan.samples,
        scan.exclusions,
        scan.quarantine_ids,
        scan.run,
        Path(args.output),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if scan.result.verdict.value != "FAIL" else 1


def _cmd_inspect_cluster(args: argparse.Namespace) -> int:
    report_path = Path(args.report)
    if not report_path.is_file():
        raise QualityCliError(f"report not found: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    cluster_id = args.cluster_id
    clusters = payload.get("clusters", [])
    match = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if match is None:
        raise QualityCliError(f"cluster not found: {cluster_id}")
    print(json.dumps(match, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Dataset quality gate + deduplication engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Run the full quality gate over a manifest.")
    scan.add_argument("manifest", type=Path)
    scan.add_argument("--config", type=Path)
    scan.add_argument("--output", type=Path)
    scan.add_argument("--strict", action="store_true")
    scan.add_argument("--resume", action="store_true")
    scan.add_argument("--no-near-duplicates", action="store_true")
    scan.add_argument("--cross-split-only", action="store_true")
    scan.add_argument("--max-samples", type=int, default=0)
    scan.set_defaults(func=_cmd_scan)

    verify = sub.add_parser(
        "verify", help="Verify an existing scan report against a config."
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--config", type=Path)
    verify.add_argument("--json", dest="as_json", action="store_true")
    verify.set_defaults(func=_cmd_verify)

    report = sub.add_parser("report", help="Render a prior scan report (no recompute).")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--output", type=Path)
    report.add_argument("--format", choices=["text", "json"], default="text")
    report.set_defaults(func=_cmd_report)

    clean = sub.add_parser(
        "clean-manifest",
        help="Write a cleaned derived manifest (never deletes source data).",
    )
    clean.add_argument("manifest", type=Path)
    clean.add_argument("--output", type=Path, required=True)
    clean.add_argument("--config", type=Path)
    clean.add_argument("--dry-run", action="store_true")
    clean.set_defaults(func=_cmd_clean_manifest)

    cluster = sub.add_parser(
        "inspect-cluster", help="Dump one duplicate cluster with provenance."
    )
    cluster.add_argument("--report", type=Path, required=True)
    cluster.add_argument("--cluster-id", required=True)
    cluster.set_defaults(func=_cmd_inspect_cluster)

    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except QualityCliError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
