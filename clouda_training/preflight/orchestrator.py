"""Preflight orchestrator: assemble the full report for one experiment config.

Aggregates config / system / dataset / plan checks into a single
:class:`PreflightReport`. Read-only: never starts training, never downloads,
never mutates datasets. Allowed side effects: a cleaned temp write probe and
lazy import probes (inside checks_system).
"""

from __future__ import annotations

from typing import Any

from clouda_training.preflight.checks_config import validate_config
from clouda_training.preflight.checks_dataset import (
    check_data_contract,
    check_dataset,
    check_resume,
)
from clouda_training.preflight.checks_system import (
    check_adapter,
    check_capability_hooks,
    check_dependencies,
    check_device,
    check_local_model,
    check_output_storage,
    check_precision,
    ensure_adapters_registered,
)
from clouda_training.preflight.models import (
    PreflightCheck,
    PreflightContext,
    PreflightReport,
    PreflightSection,
    PreflightStatus,
)


def run_preflight(
    config: Any,
    *,
    dataset_row_count: int | None = None,
    world_size: int = 1,
    write_probe: bool = True,
) -> PreflightReport:
    """Build the full preflight report for one concrete experiment config.

    ``dataset_row_count``: optional pre-counted row total (metadata-first —
    the caller may pass the canonical DatasetIdentity.row_count to avoid a
    second manifest read; when None, plan math degrades to max_steps budget).

    ``write_probe=False`` disables the storage write probe (used by --no-write-probe).
    """
    ensure_adapters_registered()
    from clouda_training.preflight.plan import compute_training_plan

    sections: list[PreflightSection] = []

    # 1) Config section
    sections.append(
        PreflightSection(name="Configuration", checks=tuple(validate_config(config)))
    )

    # 2) Model adapter section
    adapter_checks = [
        check_adapter(config),
        check_dependencies(config),
        check_local_model(config),
    ]
    sections.append(
        PreflightSection(name="Model adapter", checks=tuple(adapter_checks))
    )

    # 3) Device / precision section
    sections.append(
        PreflightSection(
            name="Device & precision",
            checks=(check_device(config), check_precision(config)),
        )
    )

    # 4) Dataset section (canonical protection guard)
    dataset_checks = list(check_dataset(config))
    dataset_checks.append(check_data_contract(config))
    sections.append(PreflightSection(name="Dataset", checks=tuple(dataset_checks)))

    # 5) Checkpoint / resume section
    from clouda_training.preflight.plan import validate_checkpoint_cadence

    cadence = validate_checkpoint_cadence(
        save_steps=config.checkpoint.save_steps,
        planned_optimizer_steps=_planned_steps(config, dataset_row_count, world_size),
        save_strategy=config.checkpoint.save_strategy,
    )
    checkpoint_checks = [check_resume(config), *cadence]
    sections.append(
        PreflightSection(name="Checkpoint", checks=tuple(checkpoint_checks))
    )

    # 6) Output storage section
    storage_checks = (
        [check_output_storage(config)]
        if write_probe
        else [PreflightCheck_output_probe_skipped()]
    )
    sections.append(
        PreflightSection(name="Output storage", checks=tuple(storage_checks))
    )

    # 7) Optional capability hooks (UNAVAILABLE warnings, never blockers)
    sections.append(
        PreflightSection(
            name="Optional capabilities",
            checks=check_capability_hooks(),
        )
    )

    # Training plan (pure math; not a pass/fail check — carried on the report).
    # Plan math raises on invalid inputs (batch_size<=0 etc.); those cases are
    # already FAIL blockers from the config section — never let the math crash
    # the whole report.
    try:
        training_plan = compute_training_plan(
            config, dataset_row_count=dataset_row_count, world_size=world_size
        )
    except ValueError as exc:
        training_plan = None
        sections.append(
            PreflightSection(
                name="Training plan",
                checks=(
                    PreflightCheck(
                        name="plan.computable",
                        status=PreflightStatus.FAIL,
                        detail=f"plan math failed: {exc}",
                        blocker=False,  # config blockers already cover this
                    ),
                ),
            )
        )

    adapter_identity = _adapter_identity(config)
    dataset_identity = _dataset_identity(config)
    checkpoint_identity = _checkpoint_identity(config)

    context = PreflightContext(
        model_id=config.model.model_id,
        adapter_type=config.model.adapter_type,
        device=config.runtime.device,
        precision=config.model.precision,
        dataset_id=config.dataset.dataset_id,
        output_root=str(config.runtime.output_root),
    )
    return PreflightReport(
        sections=tuple(sections),
        context=context,
        training_plan=training_plan,
        adapter_identity=adapter_identity,
        dataset_identity=dataset_identity,
        checkpoint_identity=checkpoint_identity,
    )


def _planned_steps(
    config: Any, dataset_row_count: int | None, world_size: int
) -> int | None:
    """Planned optimizer steps (mirrors plan.py semantics) for cadence checks."""
    training = config.training
    if training.max_steps:
        return int(training.max_steps)
    if dataset_row_count and training.batch_size > 0:
        micro = -(-dataset_row_count // training.batch_size)
        steps_per_epoch = -(-micro // max(training.gradient_accumulation_steps, 1))
        return steps_per_epoch * max(training.epochs, 0) or None
    return None  # runtime fallback budget (epochs*5) is not dataset-derived


def PreflightCheck_output_probe_skipped():  # noqa: N802 — deliberate local factory
    from clouda_training.preflight.models import (
        PreflightCheck,
        PreflightStatus,
    )

    return PreflightCheck(
        name="storage.write_probe",
        status=PreflightStatus.SKIP,
        detail="write probe disabled (--no-write-probe)",
    )


def _adapter_identity(config: Any) -> str | None:
    """Path-free adapter identity string for the report (never machine paths)."""
    adapter_type = config.model.adapter_type
    if adapter_type in {"mock", "torch"}:
        return adapter_type
    try:
        from clouda_training.preflight.checks_system import resolve_descriptor

        descriptor = resolve_descriptor(adapter_type)
        identity = descriptor.identity_dict()
        return (
            f"{identity.get('adapter_type')}@{identity.get('upstream_revision')}"
            f" (compat={identity.get('checkpoint_compatibility_id')})"
        )
    except Exception:  # noqa: BLE001 — identity is informational, never blocks
        return None


def _dataset_identity(config: Any) -> str | None:
    from clouda_contracts.checksums import sha256_file

    manifest = config.dataset.manifest_path
    try:
        if manifest.is_file():
            return f"{config.dataset.dataset_id}@{config.dataset.dataset_version} sha256={sha256_file(manifest)}"
    except OSError:
        return None
    return None


def _checkpoint_identity(config: Any) -> str | None:
    resume_from = getattr(config.checkpoint, "resume_from", None)
    if not resume_from:
        return None
    import json
    from pathlib import Path

    metadata_path = Path(resume_from) / "metadata.json"
    if not metadata_path.is_file():
        return f"missing:{resume_from}"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            f"run={payload.get('run_id')} step={payload.get('step')} "
            f"adapter={payload.get('adapter_identity', {}).get('adapter_id') if isinstance(payload.get('adapter_identity'), dict) else None}"
        )
    except (OSError, json.JSONDecodeError):
        return f"unreadable:{resume_from}"


__all__ = ["run_preflight"]
