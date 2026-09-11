"""Dataset + holdout/protection + data-contract checks for the preflight.

Reuses the canonical protection implementation
(``clouda_training.experiments.dataset.validate_training_dataset``) — this
module NEVER reimplements protection parsing. Fail-closed: any protection
violation is a blocker (NOT_READY).

Capability hooks for subsystems NOT merged on this branch (dataset quality
gate, training data loader) are emitted by
``clouda_training.preflight.checks_system.check_capability_hooks`` — not here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clouda_training.preflight.checks_system import resolve_descriptor
from clouda_training.preflight.models import (
    PreflightCheck,
    PreflightStatus,
)


def check_dataset(config: Any) -> tuple[PreflightCheck, ...]:
    """Manifest existence/parsability + canonical protection + identity.

    Uses ``validate_training_dataset`` (the canonical guard) as the single
    source of truth for holdout/protection/identity decisions.
    """
    from clouda_training.experiments.dataset import validate_training_dataset

    dataset = config.dataset
    manifest: Path = dataset.manifest_path

    if not manifest.is_file():
        return (
            PreflightCheck(
                name="dataset.manifest_exists",
                status=PreflightStatus.FAIL,
                detail=f"manifest does not exist: {manifest}",
                blocker=True,
            ),
        )
    try:
        identity = validate_training_dataset(dataset)
    except PermissionError as exc:
        # Holdout/protected split or rows — always a blocker.
        return (
            PreflightCheck(
                name="dataset.protection",
                status=PreflightStatus.FAIL,
                detail=f"protection violation: {exc}",
                blocker=True,
            ),
        )
    except FileNotFoundError as exc:
        return (
            PreflightCheck(
                name="dataset.manifest_exists",
                status=PreflightStatus.FAIL,
                detail=str(exc),
                blocker=True,
            ),
        )
    except OSError as exc:
        # IO errors racing the is_file() check (locked file, unreadable
        # manifest) — fail closed rather than crash the whole preflight.
        return (
            PreflightCheck(
                name="dataset.manifest_readable",
                status=PreflightStatus.FAIL,
                detail=f"manifest could not be read: {exc}",
                blocker=True,
            ),
        )
    except ValueError as exc:
        # Identity mismatch, malformed protection metadata, malformed split
        # metadata, empty selection — all fail closed (canonical behavior).
        return (
            PreflightCheck(
                name="dataset.validity",
                status=PreflightStatus.FAIL,
                detail=f"manifest rejected by canonical validation: {exc}",
                blocker=True,
            ),
        )
    return (
        PreflightCheck(
            name="dataset.identity_and_protection",
            status=PreflightStatus.PASS,
            detail=(
                f"rows={identity.row_count} manifest_sha256={identity.manifest_hash} "
                f"sources={len(identity.source_ids)} (protection: canonical guard passed)"
            ),
        ),
    )


def check_data_contract(config: Any, row_count: int | None = None) -> PreflightCheck:
    """Compare the configured data mode against the adapter's contract.

    Metadata-first: inspects the descriptor's ``supported_data_modes`` and the
    (optional) presence of a packed dataset reference in the config; never
    converts data.
    """
    adapter_type = config.model.adapter_type
    try:
        descriptor = resolve_descriptor(adapter_type)
    except Exception as exc:  # noqa: BLE001 — unknown adapters reported elsewhere
        return PreflightCheck(
            name="data.contract",
            status=PreflightStatus.SKIP,
            detail=f"adapter unknown ({exc}); contract check skipped",
        )
    if descriptor is None or adapter_type in {"mock", "torch"}:
        return PreflightCheck(
            name="data.contract",
            status=PreflightStatus.SKIP,
            detail="generic/mock adapter has no model-specific data contract",
        )
    supported = tuple(getattr(descriptor, "supported_data_modes", ()) or ())
    # The Clouda canonical feed is raw JSONL; packed variants are separate
    # artifacts. Detect a packed-data expectation mismatch: an adapter that
    # lists ONLY packed modes cannot consume the canonical raw manifest.
    if supported and all("packed" in mode for mode in supported):
        return PreflightCheck(
            name="data.contract",
            status=PreflightStatus.FAIL,
            detail=(
                f"adapter {adapter_type!r} supports only packed data modes "
                f"{supported}; run the official packing pipeline first "
                "(preflight never converts data)"
            ),
            blocker=True,
        )
    return PreflightCheck(
        name="data.contract",
        status=PreflightStatus.PASS,
        detail=(
            f"adapter {adapter_type!r} accepts canonical raw feed "
            f"(modes={supported or 'any'})"
        ),
    )


def _read_checkpoint_metadata(path: Path) -> dict[str, Any] | None:
    """Read checkpoint metadata.json without loading state payloads.

    Respects the checkpoint trust boundary: only JSON metadata is read here;
    integrity validation happens via the canonical loader when metadata exists.
    """
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def check_resume(config: Any) -> PreflightCheck:
    """Resume compatibility: identity fields + adapter identity + integrity.

    When no ``resume_from`` is configured this is a SKIP (fresh run). The
    heavy canonical validation is delegated to
    ``CheckpointManager.validate_resume`` semantics, mirrored here read-only
    WITHOUT constructing a manager (no run directory needed).

    Deliberate subset: preflight checks experiment_name/model_id/
    model_revision/dataset_id/dataset_version + adapter identity + state
    integrity. ``config_hash`` and ``run_id`` are NOT checked here — they are
    per-run values (run_id does not exist before a run starts; config_hash
    covers every config field including this preflight's own additions), so
    checking them here would reject every valid resume. The runtime's
    ``validate_resume`` still enforces them at resume time — preflight PASS
    therefore means "identity-compatible", not "guaranteed accepted".
    """
    resume_from = getattr(config.checkpoint, "resume_from", None)
    if not resume_from:
        return PreflightCheck(
            name="checkpoint.resume",
            status=PreflightStatus.SKIP,
            detail="fresh run — no resume checkpoint configured",
        )
    ckpt_dir = Path(resume_from)
    if not ckpt_dir.is_dir():
        return PreflightCheck(
            name="checkpoint.resume",
            status=PreflightStatus.FAIL,
            detail=f"resume checkpoint directory does not exist: {ckpt_dir}",
            blocker=True,
        )
    payload = _read_checkpoint_metadata(ckpt_dir)
    if payload is None:
        return PreflightCheck(
            name="checkpoint.resume",
            status=PreflightStatus.FAIL,
            detail=(f"checkpoint metadata.json missing or unreadable under {ckpt_dir}"),
            blocker=True,
        )
    expected = {
        "experiment_name": config.experiment.name,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "dataset_id": config.dataset.dataset_id,
        "dataset_version": config.dataset.dataset_version,
    }
    differences = {
        key: {"checkpoint": payload.get(key), "config": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if differences:
        return PreflightCheck(
            name="checkpoint.resume",
            status=PreflightStatus.FAIL,
            detail=(
                "checkpoint/config identity mismatch (NOT_READY): "
                + ", ".join(
                    f"{k}: checkpoint={v['checkpoint']!r} config={v['config']!r}"
                    for k, v in differences.items()
                )
            ),
            blocker=True,
        )
    # Adapter-identity gate: checkpoint must carry the same adapter identity
    # the current adapter would produce (cross-family resume rejected).
    # Fail-closed on malformed identity: a checkpoint that records an
    # adapter_identity block without a usable adapter_id is treated as
    # incompatible rather than silently passing the gate.
    recorded_adapter = payload.get("adapter_identity")
    adapter_type = config.model.adapter_type
    if recorded_adapter is not None and adapter_type not in {"mock", "torch"}:
        if not isinstance(recorded_adapter, dict):
            return PreflightCheck(
                name="checkpoint.resume",
                status=PreflightStatus.FAIL,
                detail=(
                    "checkpoint adapter_identity is malformed (not an object) — "
                    "refusing to resume"
                ),
                blocker=True,
            )
        recorded_id = recorded_adapter.get("adapter_id")
        if recorded_id != adapter_type:
            return PreflightCheck(
                name="checkpoint.resume",
                status=PreflightStatus.FAIL,
                detail=(
                    f"cross-adapter resume rejected: checkpoint adapter "
                    f"{recorded_id!r} does not match configured {adapter_type!r}"
                ),
                blocker=True,
            )
    # Integrity: canonical loader semantics — state file must exist and match
    # the recorded torch state digest when present.
    torch_state = payload.get("torch_state_file")
    if torch_state is not None:
        state_path = ckpt_dir / str(torch_state)
        if not state_path.is_file():
            return PreflightCheck(
                name="checkpoint.resume",
                status=PreflightStatus.FAIL,
                detail=f"checkpoint torch state file missing: {state_path}",
                blocker=True,
            )
        import hashlib

        digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
        if payload.get("torch_state_sha256") != digest:
            return PreflightCheck(
                name="checkpoint.resume",
                status=PreflightStatus.FAIL,
                detail=(
                    f"checkpoint integrity mismatch for {state_path} — refusing "
                    "to resume from a corrupt checkpoint"
                ),
                blocker=True,
            )
    return PreflightCheck(
        name="checkpoint.resume",
        status=PreflightStatus.PASS,
        detail=(
            f"resume from {ckpt_dir} (step={payload.get('step')}) — identity, "
            "adapter and integrity checks passed"
        ),
    )


__all__ = [
    "check_data_contract",
    "check_dataset",
    "check_resume",
]
