"""Storage/filesystem (Phases 12-13) and Git/worktree (Phase 14) checks.

Disk thresholds are **configurable and project-local defaults** — they are
presented as guidance for typical Clouda working directories, never as
universal training requirements. Git checks are read-only: no fetch, no
config changes, no state mutation of any kind.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from clouda_contracts.storage import StorageRoots

from .models import DoctorCheck, DoctorSection, DoctorStatus

# Human-readable defaults; overridable via --min-free-gb on the CLI.
DEFAULT_WARN_FREE_GB = 50.0
DEFAULT_FAIL_FREE_GB = 10.0


def _human_gb(bytes_value: int) -> str:
    value = float(bytes_value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def check_storage(
    repo_root: Path | None,
    *,
    warn_free_gb: float = DEFAULT_WARN_FREE_GB,
    fail_free_gb: float = DEFAULT_FAIL_FREE_GB,
) -> DoctorSection:
    checks: list[DoctorCheck] = []

    # Repo drive free space ------------------------------------------------
    if repo_root is not None:
        try:
            usage = shutil.disk_usage(repo_root)
            free_gb = usage.free / (1024**3)
            if free_gb < fail_free_gb:
                status, note = DoctorStatus.FAIL, "below FAIL threshold"
            elif free_gb < warn_free_gb:
                status, note = DoctorStatus.WARN, "below WARN threshold"
            else:
                status, note = DoctorStatus.PASS, "above warning threshold"
            checks.append(
                DoctorCheck(
                    id="storage.repo-drive",
                    name="Repository drive free space",
                    subsystem="storage",
                    status=status,
                    message=(
                        f"{_human_gb(usage.free)} free on {usage.total and repo_root.anchor or repo_root.anchor}"
                        f" ({note}; thresholds: warn < {warn_free_gb:g} GB, fail < {fail_free_gb:g} GB)"
                    ),
                    required=False,
                    details={
                        "free_bytes": usage.free,
                        "total_bytes": usage.total,
                        "free_gb": round(free_gb, 2),
                        "warn_threshold_gb": warn_free_gb,
                        "fail_threshold_gb": fail_free_gb,
                    },
                    remediation=(
                        None
                        if status is DoctorStatus.PASS
                        else "Free disk space on the repository drive (datasets/artifacts grow quickly)."
                    ),
                )
            )
        except OSError as exc:
            checks.append(
                DoctorCheck(
                    id="storage.repo-drive",
                    name="Repository drive free space",
                    subsystem="storage",
                    status=DoctorStatus.WARN,
                    message=f"Could not query disk usage: {exc}",
                    required=False,
                    details={"error": str(exc)},
                )
            )

    # StorageRoots (state/output roots) ------------------------------------
    try:
        roots = StorageRoots.from_env(read_only=True)
        root_status = []
        for label, path in (
            ("runtime", roots.runtime_root),
            ("datasets", roots.dataset_root),
            ("artifacts", roots.artifact_root),
            ("models", roots.model_root),
            ("cache", roots.cache_root),
        ):
            exists = path.exists()
            root_status.append({"name": label, "path": str(path), "exists": exists})
        checks.append(
            DoctorCheck(
                id="storage.state-roots",
                name="Configured state roots",
                subsystem="storage",
                status=DoctorStatus.INFO,
                message=f"{sum(1 for r in root_status if r['exists'])}/5 configured state roots exist",
                required=False,
                details={"roots": root_status},
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(
            DoctorCheck(
                id="storage.state-roots",
                name="Configured state roots",
                subsystem="storage",
                status=DoctorStatus.WARN,
                message=f"StorageRoots validation failed: {type(exc).__name__}: {exc}",
                required=False,
                details={"error": str(exc)},
                remediation="Check CLOUDA_* storage env vars for invalid or conflicting paths.",
            )
        )

    # Temp dir writability --------------------------------------------------
    try:
        temp_dir = Path(tempfile.gettempdir())
        with tempfile.NamedTemporaryFile(
            prefix="clouda-doctor-tmp-", delete=False
        ) as handle:
            handle.write(b"probe")
            probe = Path(handle.name)
        probe.unlink(missing_ok=True)
        checks.append(
            DoctorCheck(
                id="storage.temp-writable",
                name="Temp directory writable",
                subsystem="storage",
                status=DoctorStatus.PASS,
                message=f"Temp directory writable: {temp_dir}",
                required=False,
                details={"temp": str(temp_dir)},
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(
            DoctorCheck(
                id="storage.temp-writable",
                name="Temp directory writable",
                subsystem="storage",
                status=DoctorStatus.FAIL,
                message=f"Temp directory NOT writable: {exc}",
                required=False,
                details={"error": str(exc)},
                remediation="Fix TEMP/TMP environment or permissions; many pipelines need temp space.",
            )
        )

    # Repo read/write probe -------------------------------------------------
    if repo_root is not None:
        repo_writable = _probe_dir_writable(repo_root)
        checks.append(
            DoctorCheck(
                id="storage.repo-writable",
                name="Repository writable",
                subsystem="storage",
                status=DoctorStatus.PASS if repo_writable else DoctorStatus.WARN,
                message=(
                    "Repository directory writable"
                    if repo_writable
                    else "Repository directory not writable (read-only checkout?)"
                ),
                required=False,
                details={"root": str(repo_root), "writable": repo_writable},
                remediation=(
                    None
                    if repo_writable
                    else "Some workflows (factory runs, exports) need a writable repo root."
                ),
            )
        )

    return DoctorSection(id="storage", name="Storage / Filesystem", checks=checks)


def _probe_dir_writable(directory: Path) -> bool:
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".clouda-doctor-", dir=directory, delete=False
        ) as handle:
            handle.write(b"probe")
            probe = Path(handle.name)
        probe.unlink(missing_ok=True)
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Phase 14 — Git / worktree
# ---------------------------------------------------------------------------


def check_git(repo_root: Path | None) -> DoctorSection:
    """Read-only Git diagnostics (no fetch, no config changes)."""
    checks: list[DoctorCheck] = []
    if repo_root is None or not (repo_root / ".git").exists():
        checks.append(
            DoctorCheck(
                id="git.repository",
                name="Git repository",
                subsystem="git",
                status=DoctorStatus.SKIP,
                message="No Git repository detected at the active root",
                required=False,
                details={"root": str(repo_root) if repo_root else None},
            )
        )
        return DoctorSection(id="git", name="Git / Worktree", checks=checks)

    def _git(*args: str) -> tuple[int, str]:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )
            return completed.returncode, (completed.stdout or "").strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 1, f"git unavailable: {type(exc).__name__}"

    rc, branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    rc_head, head = _git("rev-parse", "HEAD")
    rc_dirty, _ = _git("status", "--porcelain")
    dirty = bool(_git("status", "--porcelain")[1])
    rc_url, origin = _git("remote", "get-url", "origin")
    # Worktree detection: --git-common-dir differs from --git-dir in a linked worktree.
    rc_dir, git_dir = _git("rev-parse", "--git-dir")
    rc_common, common_dir = _git("rev-parse", "--git-common-dir")
    is_linked_worktree = (
        rc_dir == 0
        and rc_common == 0
        and Path(git_dir).resolve() != Path(common_dir).resolve()
    )
    ahead_behind: dict[str, Any] | None = None
    rc_ab, ab_raw = _git("rev-list", "--left-right", "--count", "@{upstream}...HEAD")
    if rc_ab == 0:
        parts = ab_raw.split()
        if len(parts) == 2:
            behind, ahead = parts
            ahead_behind = {"ahead": int(ahead), "behind": int(behind)}

    if rc_head == 0:
        detail = {
            "branch": branch if rc == 0 else None,
            "head": head,
            "dirty": dirty,
            "origin": origin if rc_url == 0 else None,
            "git_dir": git_dir,
            "common_dir": common_dir,
            "is_linked_worktree": is_linked_worktree,
            "tree_kind": (
                "secondary worktree" if is_linked_worktree else "main working tree"
            ),
            "ahead_behind": ahead_behind,
        }
        checks.append(
            DoctorCheck(
                id="git.repository",
                name="Git repository state",
                subsystem="git",
                status=DoctorStatus.INFO,
                message=(
                    f"branch={branch or 'detached'} head={head[:12]} "
                    f"{'dirty' if dirty else 'clean'} ({detail['tree_kind']})"
                ),
                required=False,
                details=detail,
            )
        )
    else:
        checks.append(
            DoctorCheck(
                id="git.repository",
                name="Git repository state",
                subsystem="git",
                status=DoctorStatus.WARN,
                message="Git present but repository state could not be read",
                required=False,
                details={"error": head},
            )
        )
    return DoctorSection(id="git", name="Git / Worktree", checks=checks)
