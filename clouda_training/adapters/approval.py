"""Model training-use approval guard (fail-closed control plane).

A real training run may only start against a model that has been explicitly
approved for the intended training use in a reviewed, versioned approval
catalog. The catalog is operational data, not code: approvals change without
code changes and every change is auditable in git.

Design rules:

* Fail-closed on every ambiguous condition — missing catalog file, malformed
  JSON, unknown schema version, no matching record, or a record without a
  license identifier. Only an explicit ``approved: true`` match permits a run.
* An explicit rejection (``approved: false``) always wins over any broader
  (e.g. wildcard-revision) approval of the same model.
* Approval state is separate from adapter descriptors on purpose: descriptors
  describe verified *capabilities*, the catalog records a human/licensing
  *decision*. Neither may stand in for the other.
* No status is decided here for any real model family — the shipped catalog
  starts empty, so every real adapter is unapproved until an operator edits
  the catalog through review.

Catalog resolution order (mirrors ``clouda_data.locations``):

1. explicit ``catalog_path`` argument,
2. ``CLOUDA_MODEL_APPROVALS`` environment variable (deployment-specific
   catalog),
3. repository-canonical ``configs/models/model_training_approvals.v1.json``
   when running from the repository tree,
4. the packaged default catalog
   ``clouda_training/resources/model_training_approvals.v1.json`` (shipped
   EMPTY — no model is approved by default; included in the wheel via
   package-data so pip installs do not depend on the source-tree layout).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

APPROVALS_SCHEMA_VERSION = 1
APPROVALS_ENV_VAR = "CLOUDA_MODEL_APPROVALS"
DEFAULT_CATALOG_RELPATH = (
    Path("configs") / "models" / "model_training_approvals.v1.json"
)
PACKAGED_CATALOG_NAME = "model_training_approvals.v1.json"

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_RECORD_FIELDS = (
    "adapter_type",
    "model_id",
    "revision",
    "approved",
    "license_id",
    "approved_by",
    "approved_at",
)


class ModelTrainingNotApproved(PermissionError):
    """Raised when a real training run targets an unapproved model.

    Fail-closed on purpose: the message always names the exact gap (missing
    catalog, no record, explicit rejection, malformed entry) so the operator
    can fix the cause instead of debugging a silent fallback.
    """


@dataclass(frozen=True)
class ModelTrainingApproval:
    """One approved (adapter, model, revision) training use."""

    adapter_type: str
    model_id: str
    revision: str
    license_id: str
    approved_by: str
    approved_at: str
    notes: str
    source_catalog: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_type": self.adapter_type,
            "model_id": self.model_id,
            "revision": self.revision,
            "license_id": self.license_id,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "notes": self.notes,
            "source_catalog": self.source_catalog,
        }


def default_catalog_path() -> Path:
    """Resolve the catalog path: env override, repo-canonical, then packaged.

    The packaged default ships EMPTY so a fresh pip install approves no
    model; the repo-canonical file wins in the source tree so operators
    review approvals where the rest of the control plane lives.
    """
    override = os.environ.get(APPROVALS_ENV_VAR)
    if override:
        return Path(override)
    repo_candidate = PACKAGE_ROOT.parent / DEFAULT_CATALOG_RELPATH
    if repo_candidate.is_file():
        return repo_candidate
    return PACKAGE_ROOT / "resources" / PACKAGED_CATALOG_NAME


def load_approval_catalog(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the approval catalog.

    Raises:
        ModelTrainingNotApproved: if the catalog is missing, unreadable, or
            violates the schema (fail-closed — a broken catalog never
            degrades into "unapproved but let it slide").
    """
    catalog_path = Path(path) if path is not None else default_catalog_path()
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelTrainingNotApproved(
            f"Model training-approval catalog not found: {catalog_path}. "
            "Real training requires an explicit approval record. Point "
            f"{APPROVALS_ENV_VAR} at a catalog or create the canonical one."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelTrainingNotApproved(
            f"Model training-approval catalog is unreadable: {catalog_path} ({exc})"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelTrainingNotApproved(
            f"Model training-approval catalog must be an object: {catalog_path}"
        )
    if payload.get("schema_version") != APPROVALS_SCHEMA_VERSION:
        raise ModelTrainingNotApproved(
            f"Model training-approval catalog at {catalog_path} has unsupported "
            f"schema_version {payload.get('schema_version')!r} "
            f"(expected {APPROVALS_SCHEMA_VERSION})"
        )
    approvals = payload.get("approvals", [])
    if not isinstance(approvals, list) or not all(
        isinstance(item, dict) for item in approvals
    ):
        raise ModelTrainingNotApproved(
            f"Model training-approval catalog at {catalog_path} must carry an "
            "'approvals' list of objects"
        )
    return payload


def get_training_approval(
    adapter_type: str,
    model_id: str,
    revision: str,
    *,
    catalog_path: str | Path | None = None,
) -> ModelTrainingApproval | None:
    """Return the approval for this model, or None when none records it."""
    resolved_path = (
        Path(catalog_path) if catalog_path is not None else default_catalog_path()
    )
    payload = load_approval_catalog(resolved_path)
    matches: list[dict[str, Any]] = []
    for record in payload.get("approvals", []):
        missing = [field for field in _REQUIRED_RECORD_FIELDS if field not in record]
        if missing:
            raise ModelTrainingNotApproved(
                f"Approval record in {resolved_path} is missing required "
                f"field(s) {missing} — refusing to interpret a malformed "
                "entry as either approved or rejected"
            )
        if record["adapter_type"] != adapter_type or record["model_id"] != model_id:
            continue
        if record["revision"] not in (revision, "*"):
            continue
        matches.append(record)
    if not matches:
        return None
    rejected = [record for record in matches if not record["approved"]]
    if rejected:
        return None
    approved = matches[0]
    license_id = str(approved.get("license_id", "")).strip()
    if not license_id:
        raise ModelTrainingNotApproved(
            f"Approval record for {adapter_type}/{model_id} in {resolved_path} "
            "has no license_id — an approval without a recorded license is "
            "not a valid approval"
        )
    return ModelTrainingApproval(
        adapter_type=adapter_type,
        model_id=model_id,
        revision=str(approved["revision"]),
        license_id=license_id,
        approved_by=str(approved["approved_by"]),
        approved_at=str(approved["approved_at"]),
        notes=str(approved.get("notes", "")),
        source_catalog=str(resolved_path),
    )


def require_training_approval(
    adapter_type: str,
    model_id: str,
    revision: str,
    *,
    catalog_path: str | Path | None = None,
) -> ModelTrainingApproval:
    """Return the approval or raise :class:`ModelTrainingNotApproved`."""
    approval = get_training_approval(
        adapter_type, model_id, revision, catalog_path=catalog_path
    )
    if approval is None:
        raise ModelTrainingNotApproved(
            f"Model {model_id!r} (adapter {adapter_type!r}, revision "
            f"{revision!r}) is not approved for training use in the approval "
            "catalog. Add an explicit reviewed approval record (with "
            "license_id) before running real training — this guard is "
            "fail-closed by design."
        )
    return approval


__all__ = [
    "APPROVALS_ENV_VAR",
    "APPROVALS_SCHEMA_VERSION",
    "DEFAULT_CATALOG_RELPATH",
    "PACKAGED_CATALOG_NAME",
    "ModelTrainingApproval",
    "ModelTrainingNotApproved",
    "default_catalog_path",
    "get_training_approval",
    "load_approval_catalog",
    "require_training_approval",
]
