from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

MIGRATION_VERSION = 1
ABSOLUTE_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class Verification:
    index: int
    uri: str
    status: str
    size_bytes: int
    checksum: str


def _normalized_parts(value: str) -> tuple[str, ...]:
    normalized = value.replace("\\", "/")
    return tuple(
        part for part in PurePosixPath(normalized).parts if part not in {"/", ""}
    )


def storage_uri(value: str) -> str:
    parts = _normalized_parts(value)
    lowered = [part.lower() for part in parts]
    if "outputs" in lowered:
        index = lowered.index("outputs")
        relative = "/".join(parts[index + 1 :])
        return f"artifact://{relative}"
    if "data" in lowered:
        index = lowered.index("data")
        relative = "/".join(parts[index + 1 :])
        return f"dataset://data/{relative}"
    if value.startswith(("dataset://", "artifact://")):
        return value
    raise ValueError(f"Path cannot be mapped to a storage URI: {value!r}")


def _is_absolute(value: str) -> bool:
    return Path(value).is_absolute() or bool(ABSOLUTE_WINDOWS_PATH.match(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def migrate_records(
    records: Iterable[dict[str, Any]],
    *,
    source_root: Path,
) -> tuple[list[dict[str, Any]], list[Verification]]:
    migrated: list[dict[str, Any]] = []
    verifications: list[Verification] = []
    source_root = source_root.resolve()
    for index, original in enumerate(records):
        record = dict(original)
        old_original = str(record["original_path"])
        old_canonical = str(record["canonical_path"])
        record["legacy_original_path"] = old_original
        record["original_path"] = storage_uri(old_original)
        record["canonical_path"] = storage_uri(old_canonical)
        record["path_schema_version"] = MIGRATION_VERSION

        source_path = Path(old_original)
        if not _is_absolute(old_original):
            source_path = source_root / old_original
        source_path = source_path.resolve(strict=False)
        try:
            source_path.relative_to(source_root)
        except ValueError:
            status = "outside_source_root"
        else:
            if not source_path.is_file():
                status = "missing"
            elif source_path.stat().st_size != int(record["size_bytes"]):
                status = "size_mismatch"
            elif _sha256(source_path) != str(record["checksum"]):
                status = "checksum_mismatch"
            else:
                status = "verified"
        verifications.append(
            Verification(
                index=index,
                uri=str(record["original_path"]),
                status=status,
                size_bytes=int(record["size_bytes"]),
                checksum=str(record["checksum"]),
            )
        )
        migrated.append(record)
    return migrated, verifications


def _summary(verifications: list[Verification]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for result in verifications:
        statuses[result.status] = statuses.get(result.status, 0) + 1
    return {
        "migration_version": MIGRATION_VERSION,
        "record_count": len(verifications),
        "total_declared_bytes": sum(item.size_bytes for item in verifications),
        "verification_statuses": statuses,
        "active_absolute_paths": 0,
    }


def _csv_report(verifications: list[Verification]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=["index", "uri", "status", "size_bytes", "checksum"],
    )
    writer.writeheader()
    for result in verifications:
        writer.writerow(result.__dict__)
    return buffer.getvalue()


def _markdown_report(summary: dict[str, Any]) -> str:
    statuses = "\n".join(
        f"- `{name}`: {count}"
        for name, count in sorted(summary["verification_statuses"].items())
    )
    return (
        "# Manifest path migration\n\n"
        f"- Migration version: {summary['migration_version']}\n"
        f"- Records: {summary['record_count']}\n"
        f"- Declared bytes: {summary['total_declared_bytes']}\n"
        f"- Active absolute paths: {summary['active_absolute_paths']}\n\n"
        "## Checksum verification\n\n"
        f"{statuses}\n"
    )


def _write_outputs(
    *,
    output: Path,
    report_dir: Path,
    migrated: list[dict[str, Any]],
    verifications: list[Verification],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(verifications)
    output.write_text(
        json.dumps(migrated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "manifest_path_migration.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "manifest_path_migration.csv").write_text(
        _csv_report(verifications),
        encoding="utf-8",
    )
    (report_dir / "manifest_path_migration.md").write_text(
        _markdown_report(summary),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the migrated manifest and reports; default is dry-run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Manifest must be a JSON list.")
    migrated, verifications = migrate_records(records, source_root=args.source_root)
    summary = _summary(verifications)
    failed = summary["record_count"] - summary["verification_statuses"].get(
        "verified", 0
    )
    if args.apply:
        if args.output is None or args.report_dir is None:
            raise SystemExit("--apply requires --output and --report-dir")
        if failed:
            print(json.dumps(summary, indent=2))
            return 1
        _write_outputs(
            output=args.output,
            report_dir=args.report_dir,
            migrated=migrated,
            verifications=verifications,
        )
    print(json.dumps(summary, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
