from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from clouda_data.datasets.license_gate import require_dataset_use

RASAM_MANIFESTS = (
    "dataset_registry.json",
    "page_manifest.json",
    "rasam_first_batch_manifest.json",
    "rasam_first_batch_rejections.jsonl",
    "source_document_manifest.json",
)


@dataclass(frozen=True)
class AssetResult:
    source_path: str
    destination_uri: str
    size_bytes: int
    sha256: str
    dataset_id: str
    license_status: str
    copy_result: str
    verification_result: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(source_root: Path) -> Iterable[tuple[Path, str, str]]:
    data = source_root / "data"
    for directory in ("downloads/rasam_dataset", "raw"):
        root = data / directory
        if root.is_dir():
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                yield path, "rasam_dataset", "approved_with_conditions"
    for name in RASAM_MANIFESTS:
        path = data / "manifests" / name
        if path.is_file():
            yield path, "rasam_dataset", "approved_with_conditions"
    synthetic = data / "synthetic_tests"
    if synthetic.is_dir():
        for path in sorted(item for item in synthetic.rglob("*") if item.is_file()):
            yield path, "synthetic_example", "evaluation_only"


def reconcile_assets(
    *,
    source_root: Path,
    destination_root: Path,
    catalog_path: Path,
    apply: bool,
) -> list[AssetResult]:
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    require_dataset_use(
        "rasam_dataset",
        purpose="commercial_training",
        catalog_path=catalog_path,
    )
    results: list[AssetResult] = []
    for source, dataset_id, license_status in _source_files(source_root):
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        checksum = _sha256(source)
        copy_result = "planned"
        verification = "not_run"
        if destination.exists():
            if (
                destination.is_file()
                and destination.stat().st_size == source.stat().st_size
            ):
                verification = (
                    "verified" if _sha256(destination) == checksum else "collision"
                )
                copy_result = "already_present"
            else:
                verification = "collision"
                copy_result = "already_present"
        elif apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                shutil.copy2(source, temporary_path)
                if (
                    temporary_path.stat().st_size != source.stat().st_size
                    or _sha256(temporary_path) != checksum
                ):
                    raise OSError("Copied asset failed size or SHA-256 verification.")
                os.replace(temporary_path, destination)
                copy_result = "copied"
                verification = "verified"
            finally:
                temporary_path.unlink(missing_ok=True)
        results.append(
            AssetResult(
                source_path=str(source),
                destination_uri=(
                    "dataset://" + relative.as_posix().removeprefix("data/")
                ),
                size_bytes=source.stat().st_size,
                sha256=checksum,
                dataset_id=dataset_id,
                license_status=license_status,
                copy_result=copy_result,
                verification_result=verification,
            )
        )
    return results


def summarize(results: list[AssetResult], *, source_root: Path) -> dict:
    checksums: set[str] = set()
    duplicates = 0
    for result in results:
        if result.sha256 in checksums:
            duplicates += 1
        checksums.add(result.sha256)
    rasam_manifest = (
        source_root / "data" / "manifests" / "rasam_first_batch_manifest.json"
    )
    baseline = {}
    if rasam_manifest.is_file():
        payload = json.loads(rasam_manifest.read_text(encoding="utf-8"))
        baseline = {
            "source_pages": payload.get("total_pages"),
            "valid_pages": payload.get("valid_pages"),
            "rejected_pages": payload.get("rejected_pages"),
        }
    return {
        "schema_version": 1,
        "expected_count": len(results),
        "copied_count": sum(item.copy_result == "copied" for item in results),
        "already_present_count": sum(
            item.copy_result == "already_present" for item in results
        ),
        "verified_count": sum(
            item.verification_result == "verified" for item in results
        ),
        "missing_count": sum(item.verification_result == "not_run" for item in results),
        "rejected_count": sum(
            item.verification_result == "collision" for item in results
        ),
        "duplicate_count": duplicates,
        "total_bytes": sum(item.size_bytes for item in results),
        "license_statuses": sorted({item.license_status for item in results}),
        "rasam_baseline": baseline,
    }


def _write_reports(
    report_dir: Path,
    results: list[AssetResult],
    summary: dict,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dataset_reconciliation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "summary": summary,
                "assets": [asdict(item) for item in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(asdict(results[0])) if results else []
    )
    if results:
        writer.writeheader()
        writer.writerows(asdict(item) for item in results)
    (report_dir / "dataset_reconciliation.csv").write_text(
        buffer.getvalue(),
        encoding="utf-8",
    )
    markdown = (
        "# Dataset reconciliation\n\n"
        f"- Expected assets: {summary['expected_count']}\n"
        f"- Verified assets: {summary['verified_count']}\n"
        f"- Total bytes: {summary['total_bytes']}\n"
        f"- Duplicate checksums: {summary['duplicate_count']}\n"
        f"- RASAM pages: {summary['rasam_baseline']}\n"
    )
    (report_dir / "dataset_reconciliation.md").write_text(
        markdown,
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--destination-root", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    results = reconcile_assets(
        source_root=args.source_root,
        destination_root=args.destination_root,
        catalog_path=args.catalog,
        apply=args.apply,
    )
    summary = summarize(results, source_root=args.source_root)
    if args.apply:
        if args.report_dir is None:
            raise SystemExit("--apply requires --report-dir")
        _write_reports(args.report_dir, results, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["rejected_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
