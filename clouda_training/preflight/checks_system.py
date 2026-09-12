"""System-level preflight checks (Agent D scope).

Checks in this module (all fail-closed, never install anything, never
download, never mutate datasets):

* :func:`check_adapter` — resolve ``config.model.adapter_type`` through the
  canonical registry (both ``register_hunyuan_adapters`` and
  ``register_qwen_adapters`` are invoked first, idempotently). Unknown
  adapter types are a blocker FAIL. ``mock``/``torch`` adapter types skip
  model-specific descriptor checks with an explicit SKIP detail.
* :func:`check_dependencies` — import-probe the descriptor's
  ``required_optional_dependencies`` (report PASS/FAIL; never install).
  ``transformers>=X.Y.Z`` style specifiers additionally probe the installed
  version. ``mock``/``torch`` adapter types pass trivially.
* :func:`check_device` — ``cpu`` always passes; ``cuda`` fails as a blocker
  when torch is absent or ``torch.cuda.is_available()`` is False (torch is
  probed lazily inside the function).
* :func:`check_precision` — compare ``config.model.precision`` against
  ``descriptor.supported_precision``. ``fp16`` on CPU is a blocker FAIL
  (no verified support); ``bf16`` on CPU is a WARN ("unverified without
  GPU" — never a false GPU-ready claim).
* :func:`check_local_model` — for ``hunyuanocr15_sft`` reuse
  ``clouda_training.hunyuan.preflight.run_preflight(model_path=...)``;
  for ``qwen_vl_sft`` check the local path exists + ``config.json``
  (mirrors ``hunyuan`` ``_require_local_path`` logic minimally, without
  duplicating the protection itself). ``mock``/``torch`` SKIP.
* :func:`check_output_storage` — writable probe of ``runtime.output_root``
  via a temp file named ``.preflight_probe_*`` (created + deleted); rejects
  path-traversal (``..`` escaping beyond the intended root); FAILs when
  ``output_root`` equals the dataset manifest's parent directory (protected
  input overlap); reports free disk space via ``shutil.disk_usage`` (UNKNOWN
  is reported, never fabricated).
* :func:`check_capability_hooks` — detects the integrated Dataset Quality and
  Training Data Loader entry points without running them, and delegates broad
  project/environment diagnosis to Environment Doctor instead of duplicating
  it.

No torch/transformers import happens at module import time.
"""

from __future__ import annotations

import importlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from clouda_training.preflight.models import (
    PreflightCheck,
    PreflightStatus,
)

__all__ = [
    "check_adapter",
    "check_capability_hooks",
    "check_dependencies",
    "check_device",
    "check_local_model",
    "check_output_storage",
    "check_precision",
    "ensure_adapters_registered",
    "resolve_descriptor",
]

#: Adapter types that carry no model-specific descriptor semantics.
NON_MODEL_ADAPTER_TYPES = ("mock", "torch")

_PROBE_PREFIX = ".preflight_probe_"

_DEPENDENCY_SPEC_RE = re.compile(
    r"^(?P<package>[A-Za-z_][A-Za-z0-9_.\-]*)"
    r"(?P<op>>=|<=|==|>|<)"
    r"(?P<version>[0-9][A-Za-z0-9.\-]*)$"
)


# ---------------------------------------------------------------------------
# registry helpers
# ---------------------------------------------------------------------------


def ensure_adapters_registered() -> Any:
    """Register both known adapters (idempotent) and return the registry.

    Registration hooks are idempotent by design (guarded by
    ``is_registered``), so calling them on every preflight is safe and keeps
    this module independent of import order.
    """
    from clouda_training.adapters.registry import get_default_registry
    from clouda_training.hunyuan.registration import register_hunyuan_adapters
    from clouda_training.qwen.registration import register_qwen_adapters

    register_hunyuan_adapters()
    register_qwen_adapters()
    return get_default_registry()


def resolve_descriptor(adapter_type: str) -> Any:
    """Return the descriptor for ``adapter_type`` after registering adapters.

    Raises ``UnknownAdapterError`` for unregistered types (fail-closed —
    callers translate that into a blocker FAIL).
    """
    registry = ensure_adapters_registered()
    return registry.get(adapter_type)


# ---------------------------------------------------------------------------
# 1. adapter check
# ---------------------------------------------------------------------------


def check_adapter(config: Any) -> PreflightCheck:
    """Resolve the configured adapter through the canonical registry."""
    adapter_type = config.model.adapter_type
    if adapter_type in NON_MODEL_ADAPTER_TYPES:
        return PreflightCheck(
            name="adapter",
            status=PreflightStatus.SKIP,
            detail=(
                f"adapter_type={adapter_type!r}: model-agnostic backend; "
                "model-specific descriptor checks skipped"
            ),
            blocker=False,
        )
    try:
        descriptor = resolve_descriptor(adapter_type)
    except Exception as exc:  # UnknownAdapterError and any registry failure
        return PreflightCheck(
            name="adapter",
            status=PreflightStatus.FAIL,
            detail=f"unknown adapter_type {adapter_type!r}: {exc}",
            blocker=True,
        )
    problems: list[str] = []
    try:
        descriptor.require_capabilities("supports_full_finetune")
    except Exception as exc:
        problems.append(f"capability: {exc}")
    config_revision = config.model.revision
    upstream_revision = descriptor.upstream_revision
    if upstream_revision and config_revision not in ("unresolved", upstream_revision):
        problems.append(
            f"configured model.revision {config_revision!r} does not match "
            f"verified upstream revision {upstream_revision!r}"
        )
    if problems:
        return PreflightCheck(
            name="adapter",
            status=PreflightStatus.FAIL,
            detail="; ".join(problems),
            blocker=True,
        )
    detail = (
        f"adapter_type={adapter_type!r} resolved; "
        f"model_family={descriptor.model_family!r}; "
        f"upstream={descriptor.upstream_repository}"
        f"@{upstream_revision or 'unpinned'}"
    )
    return PreflightCheck(
        name="adapter", status=PreflightStatus.PASS, detail=detail, blocker=False
    )


# ---------------------------------------------------------------------------
# 2. dependency check (import probe — never install)
# ---------------------------------------------------------------------------


def _distribution_name(package: str) -> str:
    return {"trust_remote_code": "transformers"}.get(package, package)


def _version_satisfied(installed: str, op: str, required: str) -> bool:
    def key(version: str) -> tuple[int, ...]:
        parts: list[int] = []
        for chunk in re.findall(r"\d+", version):
            parts.append(int(chunk))
        return tuple(parts) or (0,)

    left, right = key(installed), key(required)
    if op == ">=":
        return left >= right
    if op == ">":
        return left > right
    if op == "<=":
        return left <= right
    if op == "<":
        return left < right
    return left == right  # ==


def _probe_dependency(spec: str) -> tuple[bool, str]:
    """Import-probe one dependency specifier. Never installs anything."""
    match = _DEPENDENCY_SPEC_RE.match(spec.strip())
    if match is None:
        package = spec.strip()
        op = required = None
    else:
        package = match.group("package")
        op = match.group("op")
        required = match.group("version")
    import_name = _distribution_name(package)
    if import_name == "trust_remote_code":
        # Loaded through transformers' dynamic trust_remote_code path; the
        # transformers import below is the actual probe.
        return True, "trust_remote_code resolved via transformers load path"
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:
        return False, f"import {import_name!r} failed: {exc}"
    if op is not None and required is not None:
        installed = str(getattr(module, "__version__", "") or "0")
        if not _version_satisfied(installed, op, required):
            return False, (f"{import_name} {installed} installed but {spec} required")
        return True, f"{import_name} {installed} satisfies {spec}"
    version = getattr(module, "__version__", None)
    return True, f"{import_name} importable" + (f" {version}" if version else "")


def check_dependencies(config: Any) -> PreflightCheck:
    """Import-probe the descriptor's required optional dependencies."""
    adapter_type = config.model.adapter_type
    if adapter_type in NON_MODEL_ADAPTER_TYPES:
        return PreflightCheck(
            name="dependencies",
            status=PreflightStatus.SKIP,
            detail=(
                f"adapter_type={adapter_type!r}: no optional model "
                "dependencies required"
            ),
            blocker=False,
        )
    try:
        descriptor = resolve_descriptor(adapter_type)
    except Exception as exc:
        return PreflightCheck(
            name="dependencies",
            status=PreflightStatus.FAIL,
            detail=f"cannot probe dependencies: unknown adapter: {exc}",
            blocker=True,
        )
    problems: list[str] = []
    for spec in descriptor.required_optional_dependencies:
        ok, detail = _probe_dependency(spec)
        if not ok:
            problems.append(detail)
    if problems:
        return PreflightCheck(
            name="dependencies",
            status=PreflightStatus.FAIL,
            detail="; ".join(problems) + " (never installed by preflight)",
            blocker=True,
        )
    return PreflightCheck(
        name="dependencies",
        status=PreflightStatus.PASS,
        detail="all required optional dependencies importable: "
        + ", ".join(descriptor.required_optional_dependencies),
        blocker=False,
    )


# ---------------------------------------------------------------------------
# 3. device check (torch probed lazily)
# ---------------------------------------------------------------------------


def _cuda_available() -> tuple[bool, str]:
    """Lazily import torch and probe CUDA. Returns (available, detail)."""
    try:
        import torch  # noqa: PLC0415 — lazy by design
    except Exception as exc:
        return False, f"torch not importable: {exc}"
    try:
        if bool(torch.cuda.is_available()):
            return True, f"torch {torch.__version__}: CUDA available"
        return False, f"torch {torch.__version__}: torch.cuda.is_available() is False"
    except Exception as exc:
        return False, f"CUDA probe failed: {exc}"


def check_device(config: Any) -> PreflightCheck:
    """cpu always passes; cuda requires an available CUDA runtime."""
    device = config.runtime.device
    if device == "cpu":
        return PreflightCheck(
            name="device",
            status=PreflightStatus.PASS,
            detail="device=cpu",
            blocker=False,
        )
    if device == "cuda":
        available, detail = _cuda_available()
        if not available:
            return PreflightCheck(
                name="device",
                status=PreflightStatus.FAIL,
                detail=f"device=cuda requested but unavailable: {detail}",
                blocker=True,
            )
        return PreflightCheck(
            name="device", status=PreflightStatus.PASS, detail=detail, blocker=False
        )
    return PreflightCheck(
        name="device",
        status=PreflightStatus.FAIL,
        detail=f"unsupported device {device!r} (expected 'cpu' or 'cuda')",
        blocker=True,
    )


# ---------------------------------------------------------------------------
# 4. precision check
# ---------------------------------------------------------------------------


def check_precision(config: Any) -> PreflightCheck:
    """Compare configured precision against verified descriptor support."""
    adapter_type = config.model.adapter_type
    precision = config.model.precision
    device = config.runtime.device
    if adapter_type in NON_MODEL_ADAPTER_TYPES:
        return PreflightCheck(
            name="precision",
            status=PreflightStatus.SKIP,
            detail=(
                f"adapter_type={adapter_type!r}: precision {precision!r} not "
                "validated against a model descriptor"
            ),
            blocker=False,
        )
    try:
        descriptor = resolve_descriptor(adapter_type)
    except Exception as exc:
        return PreflightCheck(
            name="precision",
            status=PreflightStatus.FAIL,
            detail=f"cannot check precision: unknown adapter: {exc}",
            blocker=True,
        )
    supported = tuple(descriptor.supported_precision)
    if precision not in supported:
        return PreflightCheck(
            name="precision",
            status=PreflightStatus.FAIL,
            detail=f"precision {precision!r} not in descriptor-supported "
            f"{supported} for {adapter_type!r}",
            blocker=True,
        )
    if device == "cpu" and precision in ("fp16", "float16"):
        # No verified fp16 support on CPU — fail closed, never guess.
        return PreflightCheck(
            name="precision",
            status=PreflightStatus.FAIL,
            detail="fp16 on cpu: no verified CPU fp16 support",
            blocker=True,
        )
    if device == "cpu" and precision in ("bf16", "bfloat16"):
        # Unverified without a GPU — warn only, never claim GPU readiness.
        return PreflightCheck(
            name="precision",
            status=PreflightStatus.WARN,
            detail="bf16 on cpu: unverified without GPU; numerical behaviour "
            "on this device has not been validated",
            blocker=False,
        )
    return PreflightCheck(
        name="precision",
        status=PreflightStatus.PASS,
        detail=f"precision={precision!r} supported by {adapter_type!r} on "
        f"device={device!r}",
        blocker=False,
    )


# ---------------------------------------------------------------------------
# 5. local model check
# ---------------------------------------------------------------------------


def _local_path_problems(model_path: Path) -> list[str]:
    """Minimal mirror of hunyuan ``_require_local_path`` protection.

    The canonical protection lives in the adapters; this only checks the two
    facts preflight can verify offline (dir exists + config.json present) so
    the failure surfaces BEFORE training starts, without reimplementing (or
    weakening) the adapter's own guard.
    """
    problems: list[str] = []
    if not model_path.is_dir():
        problems.append(
            f"local model path does not exist: {model_path} — "
            "this adapter never downloads weights"
        )
        return problems
    if not (model_path / "config.json").is_file():
        problems.append(
            f"missing config.json under {model_path} — not a local model directory"
        )
    return problems


def check_local_model(config: Any) -> PreflightCheck:
    """Verify the configured local model path for real adapters."""
    adapter_type = config.model.adapter_type
    if adapter_type in NON_MODEL_ADAPTER_TYPES:
        return PreflightCheck(
            name="local_model",
            status=PreflightStatus.SKIP,
            detail=f"adapter_type={adapter_type!r}: no local model required",
            blocker=False,
        )
    model_path = Path(str(config.model.model_id)).expanduser()
    if adapter_type == "hunyuanocr15_sft":
        # Reuse the canonical hunyuan preflight (never reimplement it).
        from clouda_training.hunyuan.preflight import run_preflight

        report: dict[str, Any] = run_preflight(model_path=str(model_path))
        failed = [c for c in report["checks"] if not c["passed"]]
        if not report["ok"] or failed:
            names = ", ".join(c["name"] for c in failed) or "unknown"
            return PreflightCheck(
                name="local_model",
                status=PreflightStatus.FAIL,
                detail=f"hunyuan preflight failed checks: {names} "
                f"(model_path={model_path})",
                blocker=True,
            )
        return PreflightCheck(
            name="local_model",
            status=PreflightStatus.PASS,
            detail=f"hunyuan preflight ok for {model_path}",
            blocker=False,
        )
    if adapter_type == "qwen_vl_sft":
        problems = _local_path_problems(model_path)
        if problems:
            return PreflightCheck(
                name="local_model",
                status=PreflightStatus.FAIL,
                detail="; ".join(problems),
                blocker=True,
            )
        return PreflightCheck(
            name="local_model",
            status=PreflightStatus.PASS,
            detail=f"local model directory verified at {model_path}",
            blocker=False,
        )
    return PreflightCheck(
        name="local_model",
        status=PreflightStatus.FAIL,
        detail=f"unknown adapter_type {adapter_type!r}: no local-model rule",
        blocker=True,
    )


# ---------------------------------------------------------------------------
# 6. output storage check
# ---------------------------------------------------------------------------


def _traversal_problems(output_root: Path) -> list[str]:
    """Reject explicit ``..`` traversal components in the configured path.

    Note: an absolute output_root on another drive than the working directory
    is legitimate config (the config loader resolves output_root against the
    config file's base), so only relative ``..`` escapes are rejected here —
    the protected-input overlap check below covers the dangerous collision
    case.
    """
    raw = Path(str(output_root))
    if ".." in raw.parts:
        return [f"output_root contains '..' path-traversal component: {output_root}"]
    return []


def check_output_storage(config: Any) -> PreflightCheck:
    """Writable probe + traversal guard + free-disk-space report."""
    output_root = Path(config.runtime.output_root).expanduser()
    detail_parts: list[str] = []

    # -- protected input overlap: output must not collide with the dataset
    manifest = config.dataset.manifest_path
    if manifest is not None:
        try:
            manifest_parent = Path(manifest).resolve().parent
            if output_root.resolve() == manifest_parent:
                return PreflightCheck(
                    name="output_storage",
                    status=PreflightStatus.FAIL,
                    detail=(
                        f"output_root {output_root} equals the dataset manifest "
                        f"directory {manifest_parent} — training outputs would "
                        "overlap a protected input directory"
                    ),
                    blocker=True,
                )
        except OSError as exc:
            return PreflightCheck(
                name="output_storage",
                status=PreflightStatus.FAIL,
                detail=f"output_root/manifest cannot be resolved: {exc}",
                blocker=True,
            )

    # -- path traversal guard
    traversal = _traversal_problems(output_root)
    if traversal:
        return PreflightCheck(
            name="output_storage",
            status=PreflightStatus.FAIL,
            detail="; ".join(traversal),
            blocker=True,
        )

    # -- writability probe (temp file, always cleaned up)
    if not output_root.exists():
        # Phase 15: "output root exists or can be safely created". Fail only
        # when creation is impossible (unwritable/invalid parent); creating a
        # missing leaf directory is what a real run would do anyway.
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            detail_parts.append(f"created missing output_root: {output_root}")
        except OSError as exc:
            return PreflightCheck(
                name="output_storage",
                status=PreflightStatus.FAIL,
                detail=(
                    f"output_root does not exist and cannot be created "
                    f"({exc}): {output_root}"
                ),
                blocker=True,
            )
    if not output_root.is_dir():
        return PreflightCheck(
            name="output_storage",
            status=PreflightStatus.FAIL,
            detail=f"output_root is not a directory: {output_root}",
            blocker=True,
        )
    probe_path: str | None = None
    try:
        fd, probe_path = tempfile.mkstemp(
            prefix=_PROBE_PREFIX, dir=str(output_root), suffix=".tmp"
        )
        os.close(fd)
        Path(probe_path).unlink()
        detail_parts.append(f"writable (probe {_PROBE_PREFIX}* created+deleted)")
    except OSError as exc:
        return PreflightCheck(
            name="output_storage",
            status=PreflightStatus.FAIL,
            detail=f"output_root not writable ({exc}); "
            f"probe={_PROBE_PREFIX}*.tmp in {output_root}",
            blocker=True,
        )
    finally:
        if probe_path is not None and os.path.exists(probe_path):
            try:
                os.unlink(probe_path)
            except OSError:
                pass  # probe cleanup is best-effort; failure already reported

    # -- free disk space (UNKNOWN is reported, never fabricated)
    try:
        usage = shutil.disk_usage(output_root)
        free_gb = usage.free / (1024**3)
        detail_parts.append(f"disk free: {free_gb:.2f} GiB ({usage.free} bytes)")
    except OSError as exc:
        detail_parts.append(f"disk free: UNKNOWN ({exc})")

    return PreflightCheck(
        name="output_storage",
        status=PreflightStatus.PASS,
        detail=f"output_root={output_root}: " + "; ".join(detail_parts),
        blocker=False,
    )


# ---------------------------------------------------------------------------
# 7. capability hooks
# ---------------------------------------------------------------------------


def check_capability_hooks() -> tuple[PreflightCheck, ...]:
    """Detect integrated capabilities without duplicating their validation."""

    checks: list[PreflightCheck] = []
    for name, module_name, symbol in (
        (
            "DATASET QUALITY CHECK",
            "clouda_data.quality.gate",
            "run_quality_gate",
        ),
        (
            "TRAINING DATA LOADER CHECK",
            "clouda_data.training_data.loader",
            "StreamingTrainingDataLoader",
        ),
    ):
        try:
            module = importlib.import_module(module_name)
            getattr(module, symbol)
        except (ImportError, AttributeError) as exc:
            checks.append(
                PreflightCheck(
                    name=name,
                    status=PreflightStatus.UNAVAILABLE,
                    detail=f"capability unavailable: {exc}",
                    blocker=False,
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    name=name,
                    status=PreflightStatus.PASS,
                    detail=(
                        f"capability available via {module_name}.{symbol}; "
                        "run-specific manifest checks remain in this preflight"
                    ),
                    blocker=False,
                )
            )

    try:
        doctor = importlib.import_module("clouda_data.doctor")
        getattr(doctor, "collect_report")
    except (ImportError, AttributeError) as exc:
        doctor_check = PreflightCheck(
            name="ENVIRONMENT DOCTOR",
            status=PreflightStatus.UNAVAILABLE,
            detail=f"Environment Doctor unavailable: {exc}",
            blocker=False,
        )
    else:
        doctor_check = PreflightCheck(
            name="ENVIRONMENT DOCTOR",
            status=PreflightStatus.SKIP,
            detail=(
                "Environment Doctor is available for project/environment "
                "diagnosis; preflight does not duplicate or invoke it"
            ),
            blocker=False,
        )
    checks.append(doctor_check)
    return tuple(checks)
