from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabSettings:
    """Server-owned dashboard configuration.

    Browser requests operate on canonical identifiers only. Filesystem roots
    are fixed here at startup and are never accepted from an API payload.
    """

    repo_root: Path
    runs_root: Path
    results_root: Path
    plans_root: Path
    quality_root: Path
    benchmarks_root: Path
    tasks_root: Path
    confirmations_root: Path
    dataset_downloads_root: Path
    download_manifests_root: Path
    dataset_imports_root: Path
    models_root: Path
    model_state_root: Path
    benchmark_plans_root: Path
    trash_root: Path
    host: str = "127.0.0.1"
    port: int = 8000
    local_only: bool = True
    preview_limit: int = 20
    quality_scan_limit: int = 5_000

    @classmethod
    def from_repo(
        cls,
        repo_root: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        local_only: bool = True,
    ) -> "LabSettings":
        root = Path(repo_root).expanduser().resolve()
        return cls(
            repo_root=root,
            runs_root=root / "runs",
            results_root=root / "runs" / "results-store",
            plans_root=root / "runs" / ".lab-plans",
            quality_root=root / "runs" / ".lab-quality",
            benchmarks_root=root / "benchmarks",
            tasks_root=root / "runs" / ".lab-tasks",
            confirmations_root=root / "runs" / ".lab-confirmations",
            dataset_downloads_root=root / "data" / "downloads",
            download_manifests_root=root / "data" / "manifests" / "download_manifests",
            dataset_imports_root=root / "data" / "imports",
            models_root=root / "data" / "models",
            model_state_root=root / "runs" / ".lab-models",
            benchmark_plans_root=root / "runs" / ".lab-benchmark-plans",
            trash_root=root / "runs" / ".lab-trash",
            host=host,
            port=port,
            local_only=local_only,
        )
