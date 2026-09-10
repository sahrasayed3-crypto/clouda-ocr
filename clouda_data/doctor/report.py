"""Environment Doctor orchestrator: assembles the full DoctorReport.

Default mode is lightweight (imports, resources, metadata). ``deep=True``
adds a tiny offline MockTrainer dry-run; both modes are offline and never
mutate the environment outside temp probes that are cleaned up.
"""

from __future__ import annotations

import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clouda_data.locations import repository_root

from .environment import (
    build_dependency_sections,
    check_import_resolution,
    check_runtime,
    _load_pyproject,
)
from .factory import check_factory, check_raqm, check_weasyprint
from .models import DoctorReport, DoctorSection, DoctorStatus
from .security import build_env_report
from .system import check_git, check_storage
from .training import check_gpu, check_training_framework, run_training_dry_run


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _optional_package(name: str) -> dict[str, Any]:
    """Presence marker for optional newer subsystems (Results Store / Lab)."""
    import importlib.util

    spec = importlib.util.find_spec(name)
    return {"present": spec is not None}


def collect_report(
    *,
    deep: bool = False,
    include_factory: bool = True,
    include_render: bool = True,
    include_training: bool = True,
    include_gpu: bool = True,
    include_git: bool = True,
    include_storage: bool = True,
    include_deps: bool = True,
    warn_free_gb: float = 50.0,
    fail_free_gb: float = 10.0,
) -> DoctorReport:
    """Run all requested checks and assemble the report.

    Scoped flags let the CLI expose --factory/--render/--training/--git;
    core runtime and import-resolution checks always run because every other
    section's meaning depends on them.
    """
    repo_root = repository_root()
    pyproject = _load_pyproject(repo_root) if repo_root else {}
    sections: list[DoctorSection] = []

    # Core: runtime + import resolution (always).
    sections.append(
        DoctorSection(
            id="core",
            name="Core",
            checks=check_runtime(pyproject)
            + check_import_resolution(repo_root, pyproject),
        )
    )

    if include_deps:
        sections.extend(build_dependency_sections(pyproject))

    if include_factory:
        sections.append(check_factory(repo_root))

    if include_render:
        sections.append(check_weasyprint())
        sections.append(check_raqm())

    training_section = check_training_framework() if include_training else None
    if training_section is not None:
        sections.append(training_section)

    gpu_section = check_gpu() if include_gpu else None
    if gpu_section is not None:
        sections.append(gpu_section)

    if include_storage:
        sections.append(
            check_storage(
                repo_root, warn_free_gb=warn_free_gb, fail_free_gb=fail_free_gb
            )
        )

    if include_git:
        sections.append(check_git(repo_root))

    # Optional subsystems not present on this branch base: SKIP, not FAIL.
    optional_checks = []
    for pkg_name, label in (
        ("clouda_data.results", "Results Store"),
        ("clouda_lab", "Clouda Lab Backend"),
    ):
        presence = _optional_package(pkg_name)
        from .models import DoctorCheck as _DC

        optional_checks.append(
            _DC(
                id=f"optional.{pkg_name.replace('.', '-')}",
                name=label,
                subsystem="optional",
                status=(
                    DoctorStatus.SKIP if not presence["present"] else DoctorStatus.INFO
                ),
                message=(
                    f"{label} not present on this branch base"
                    if not presence["present"]
                    else f"{label} package detected"
                ),
                required=False,
                details=presence,
            )
        )
    if optional_checks:
        sections.append(
            DoctorSection(
                id="optional-subsystems",
                name="Optional Subsystems",
                checks=optional_checks,
            )
        )

    # Deep mode: tiny offline training dry-run ------------------------------
    deep_checks: list = []
    if deep:
        from .models import DoctorCheck as _DC

        temp_root = Path(tempfile.mkdtemp(prefix="clouda-doctor-deep-"))
        try:
            status, message, detail = run_training_dry_run(temp_root / "runs")
            deep_checks.append(
                _DC(
                    id="deep.training-dry-run",
                    name="Training dry-run (deep)",
                    subsystem="deep",
                    status=status,
                    message=message,
                    required=False,
                    details=detail,
                )
            )
        finally:
            import shutil

            shutil.rmtree(temp_root, ignore_errors=True)
        sections.append(
            DoctorSection(id="deep", name="Deep Checks", checks=deep_checks)
        )

    runtime_summary = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "os": platform.system(),
        "machine": platform.machine(),
        "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
    }
    env_vars = build_env_report()
    report = DoctorReport(
        timestamp=_utc_timestamp(),
        deep=deep,
        runtime=runtime_summary,
        repository={
            "root": str(repo_root) if repo_root else None,
            "worktree_kind": "unknown",
            "env_vars": env_vars,
        },
        sections=sections,
    )
    return report


# Exit codes (docs/doctor.md): 0 = no required FAIL, 1 = required FAIL(s),
# 2 = doctor could not execute. WARN-only normally returns 0.
EXIT_OK = 0
EXIT_CHECKS_FAILED = 1
EXIT_ERROR = 2


def exit_code_for(report: DoctorReport) -> int:
    return EXIT_CHECKS_FAILED if report.failed_required_checks() else EXIT_OK
