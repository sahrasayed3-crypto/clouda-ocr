"""Runtime (Phase 3), import-resolution (Phase 4), and dependency (Phase 5) checks.

Import-resolution is the load-bearing safety check for multi-worktree
development: the canonical packages (``clouda_data``, ``clouda_training``,
``clouda_contracts``, ``clouda_models``, ``clouda_lab``) may be imported from this worktree's
sources, from an editable install pointing at another worktree, or from a
stale site-packages copy. The Doctor reports the drift; it never rewrites the
environment.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import io
import platform
import re
import sys
import tomllib
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from .models import DoctorCheck, DoctorSection, DoctorStatus

CANONICAL_PACKAGES = (
    "clouda_data",
    "clouda_training",
    "clouda_contracts",
    "clouda_models",
    "clouda_lab",
)

# distribution name (pyproject/extras) -> import name, only where they differ.
DIST_TO_IMPORT: dict[str, str] = {
    "Pillow": "PIL",
    "opencv-python-headless": "cv2",
    "opencv-python": "cv2",
    "PyYAML": "yaml",
    "python-docx": "docx",
    "pypdfium2": "pypdfium2",
    "PyMuPDF": "fitz",
    "WeasyPrint": "weasyprint",
    "python-multipart": "multipart",
}

#: Extras from pyproject.toml mapped to doctor dependency groups. All names
#: verified against pyproject [project.optional-dependencies] at f868a02.
EXTRA_GROUPS: dict[str, tuple[str, ...]] = {
    "core": ("Pillow", "pypdf", "pypdfium2", "python-docx", "requests"),
    "data": ("PyYAML", "defusedxml", "jsonschema"),
    "factory": ("numpy", "opencv-python-headless", "img2pdf", "pikepdf"),
    "factory-render": ("WeasyPrint",),
    "training": ("PyYAML", "jsonschema"),
    "results": (),
    "lab": ("PyYAML",),
    "training-data": ("Pillow",),
    "server": (
        "fastapi",
        "firebase-admin",
        "starlette",
        "python-multipart",
        "streamlit",
        "uvicorn",
    ),
    "worker": ("redis", "rq"),
}


def _load_pyproject(root: Path) -> dict[str, Any]:
    path = root / "pyproject.toml"
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def installed_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _importable(import_name: str) -> bool:
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            importlib.import_module(import_name)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Phase 3 — Python / runtime
# ---------------------------------------------------------------------------


def python_supported(pyproject: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Evaluate the active interpreter against the project requires-python."""
    data = pyproject or {}
    requires = data.get("project", {}).get("requires-python") or ">=3.11,<3.12"
    version = sys.version_info
    # requires-python is a simple bounded spec like ">=3.11,<3.12"; evaluate
    # the two common bound forms without a full specifier engine dependency.
    ok = True
    for part in (s.strip() for s in requires.split(",")):
        if part.startswith(">="):
            ok = ok and version >= tuple(
                int(x) for x in part[2:].split(".") if x.isdigit()
            )
        elif part.startswith("<"):
            upper = tuple(int(x) for x in part[1:].split(".") if x.isdigit())
            ok = (
                ok and version < upper
                if len(upper) == len(version)
                else ok and version[: len(upper)] < upper
            )
        elif part.startswith(">"):
            ok = ok and version > tuple(
                int(x) for x in part[1:].split(".") if x.isdigit()
            )
    return ok, requires


def check_runtime(pyproject: dict[str, Any] | None = None) -> list[DoctorCheck]:
    data = pyproject or {}
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    ok, requires = python_supported(data)
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    pyver_detail = {
        "python": version_str,
        "requires_python": requires,
        "executable": sys.executable,
        "sys_prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", ""),
        "in_venv": in_venv,
        "architecture": platform.architecture()[0],
        "machine": platform.machine(),
        "os": platform.system(),
        "platform": platform.platform(),
    }
    return [
        DoctorCheck(
            id="core.python-version",
            name="Python version",
            subsystem="core",
            status=DoctorStatus.PASS if ok else DoctorStatus.FAIL,
            message=(
                f"Python {version_str} satisfies project requirement {requires}"
                if ok
                else f"Python {version_str} does not satisfy project requirement {requires}"
            ),
            details=pyver_detail,
            remediation=None if ok else f"Use an interpreter matching {requires}.",
        ),
        DoctorCheck(
            id="core.interpreter",
            name="Interpreter environment",
            subsystem="core",
            status=DoctorStatus.INFO,
            message=(
                f"virtual environment ({sys.prefix})"
                if in_venv
                else "system interpreter (no virtual environment active)"
            ),
            required=False,
            details=pyver_detail,
        ),
    ]


# ---------------------------------------------------------------------------
# Phase 4 — import / worktree resolution
# ---------------------------------------------------------------------------


def _package_origin(module_name: str) -> dict[str, Any] | None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - diagnostic must not crash
        return {"error": f"{type(exc).__name__}: {exc}"}
    file_attr = getattr(module, "__file__", None)
    origin: dict[str, Any] = {"file": str(file_attr) if file_attr else None}
    if file_attr:
        origin["root"] = str(Path(file_attr).resolve().parent.parent)
    return origin


def check_import_resolution(
    repo_root: Path | None,
    pyproject: dict[str, Any] | None = None,
) -> list[DoctorCheck]:
    """Report where canonical packages actually import from.

    EXPECTED source root is the active repository/worktree (or
    ``CLOUDA_PROJECT_ROOT``); ACTUAL is derived from each package's
    ``__file__``. Mixed roots across the canonical packages are flagged.
    """
    checks: list[DoctorCheck] = []
    expected = str(repo_root) if repo_root else None
    origins: dict[str, Any] = {}
    roots: set[str] = set()
    import_errors: list[str] = []

    for package in CANONICAL_PACKAGES:
        origin = _package_origin(package)
        origins[package] = origin
        if origin and "error" in origin:
            import_errors.append(f"{package}: {origin['error']}")
        elif origin and origin.get("root"):
            roots.add(origin["root"])

    consistent = len(roots) == 1
    actual_root = next(iter(roots)) if consistent else None
    matches = bool(expected and actual_root and Path(actual_root) == Path(expected))

    detail: dict[str, Any] = {
        "expected_source_root": expected,
        "actual_import_roots": sorted(roots),
        "per_package": origins,
        "consistent": consistent,
    }
    if import_errors:
        detail["import_errors"] = import_errors
        checks.append(
            DoctorCheck(
                id="core.import-resolution",
                name="Canonical package import resolution",
                subsystem="core",
                status=DoctorStatus.FAIL,
                message="Canonical packages failed to import: "
                + "; ".join(import_errors),
                details=detail,
                remediation=(
                    "Run from the repository root or set CLOUDA_PROJECT_ROOT; "
                    "install with 'pip install -e .' from the intended worktree."
                ),
            )
        )
        return checks

    if not expected:
        status = DoctorStatus.WARN
        message = "Repository root unknown; packages import from " + (
            actual_root or "unknown location"
        )
        remediation = (
            "Set CLOUDA_PROJECT_ROOT or run from the repository/worktree root."
        )
    elif matches and consistent:
        status = DoctorStatus.PASS
        message = f"Canonical packages import from the active worktree ({actual_root})"
        remediation = None
    elif not consistent:
        status = DoctorStatus.FAIL
        message = "Canonical packages import from mixed roots: " + ", ".join(
            sorted(roots)
        )
        remediation = (
            "Reinstall from the intended worktree: pip install -e . "
            "(the current import set is split across checkouts)."
        )
    else:
        status = DoctorStatus.WARN
        message = (
            f"Package imports resolve to another tree: {actual_root} "
            f"(expected {expected})"
        )
        remediation = (
            "Activate/reinstall the intended worktree: "
            f"cd {expected} && pip install -e . (current imports come from {actual_root})."
        )
    checks.append(
        DoctorCheck(
            id="core.import-resolution",
            name="Canonical package import resolution",
            subsystem="core",
            status=status,
            message=message,
            details=detail,
            remediation=remediation,
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Phase 5 — dependency groups
# ---------------------------------------------------------------------------


def check_dependency_group(
    group: str,
    dist_names: tuple[str, ...],
    *,
    required: bool = True,
) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    _dist_name_re = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
    for requirement in dist_names:
        # pyproject extras carry version specs ("PyYAML==6.0.2",
        # "WeasyPrint>=66,<70"); the distribution name is the leading token.
        match = _dist_name_re.match(requirement.strip())
        dist = match.group(0) if match else requirement.strip()
        import_name = DIST_TO_IMPORT.get(dist, dist)
        version = installed_version(dist)
        present = version is not None and _importable(import_name)
        status = (
            DoctorStatus.PASS
            if present
            else (DoctorStatus.FAIL if required else DoctorStatus.WARN)
        )
        message = (
            f"{dist} {version} available"
            if present
            else (
                f"{dist} installed ({version}) but import '{import_name}' failed"
                if version is not None
                else f"{requirement} not installed"
            )
        )
        extra_hint = (
            f'pip install -e ".[{group}]"' if group != "core" else "pip install -e ."
        )
        checks.append(
            DoctorCheck(
                id=f"deps.{group}.{dist}",
                name=f"{dist} ({group})",
                subsystem="dependencies",
                status=status,
                message=message,
                required=required,
                details={
                    "distribution": dist,
                    "import": import_name,
                    "version": version,
                    "group": group,
                    "requirement": requirement,
                },
                remediation=None if present else extra_hint,
            )
        )
    return checks


def build_dependency_sections(
    pyproject: dict[str, Any] | None = None,
) -> list[DoctorSection]:
    data = pyproject or {}
    optional = data.get("project", {}).get("optional-dependencies", {}) or {}
    sections: list[DoctorSection] = []
    for group, fallback_names in EXTRA_GROUPS.items():
        names = tuple(optional.get(group) or fallback_names)
        base = group.split("-")[0]
        required = group in ("core", "data")
        section = DoctorSection(
            id=f"dependencies-{group}",
            name=f"Dependencies: {group}",
            checks=check_dependency_group(group, names, required=required),
        )
        # Keep section ordering stable by group name for deterministic output.
        sections.append(section)
        del base
    return sections
