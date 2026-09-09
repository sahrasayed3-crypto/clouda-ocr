"""Handoff boundary toward the Data Factory subsystem.

Historically this module described a handoff toward the separate external
``clouda-data-factory`` project. Since the integration, that project is
superseded by ``clouda_data.factory`` inside this repository; the handoff
artifact is still the declarative record of which clean source samples a
factory run should process, and the reverse direction (factory run ->
canonical manifest) lives in ``clouda_data.factory.adapters``.

The handoff artifact contains:

- the candidate manifest (sample ids, source-relative paths, provenance);
- requested distortion profile ids and seed;
- the intended output location;
- a SHA-256 of the dataset manifest for provenance linkage.

The future integration is a pure data contract: a caller with
``clouda-data-factory`` installed would pass ``source_root`` +
``candidate_manifest`` inputs to that project's run command. Keeping this
boundary declarative means the main project's tests never require the
external package or any network access.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .export import ExportConfig, select_exportable
from .hashing import atomic_write_text, sha256_file
from .schema import DatasetSample, sort_key

HANDOFF_BOUNDARY_VERSION = "clouda.data_factory.handoff.v1"


@dataclass(frozen=True)
class DataFactoryHandoff:
    """A declarative handoff request for the external data factory."""

    boundary_version: str
    source_root: str
    candidate_manifest: str
    candidate_manifest_sha256: str
    dataset_manifest: str
    dataset_manifest_sha256: str
    source_ids: list[str]
    requested_profiles: list[str]
    seed: int
    intended_output: str
    sample_count: int
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_data_factory_handoff(
    dataset_root: Path,
    workspace: Path,
    samples: list[DatasetSample],
    *,
    requested_profiles: list[str],
    seed: int,
    intended_output: str,
    dataset_manifest_path: Path,
) -> tuple[DataFactoryHandoff, Path, Path]:
    """Write the candidate manifest and handoff request; return artifacts."""

    candidates = select_exportable(
        sorted(samples, key=sort_key), ExportConfig(include_holdout=False)
    )
    handoff_dir = Path(workspace).resolve() / "handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = [
        {
            "sample_id": sample.sample_id,
            "source_id": sample.source_id,
            "source_path": sample.source_path,
            "image_path": sample.image_path,
            "text": sample.text,
            "language": sample.language,
            "file_sha256": sample.file_sha256,
            "target_split": sample.target_split.value,
        }
        for sample in candidates
    ]
    candidate_path = atomic_write_text(
        handoff_dir / "data_factory_candidates.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in candidate_rows
        ),
    )

    handoff = DataFactoryHandoff(
        boundary_version=HANDOFF_BOUNDARY_VERSION,
        source_root=str(Path(dataset_root).resolve()),
        candidate_manifest=str(candidate_path),
        candidate_manifest_sha256=sha256_file(candidate_path),
        dataset_manifest=str(dataset_manifest_path.resolve()),
        dataset_manifest_sha256=sha256_file(dataset_manifest_path.resolve()),
        source_ids=sorted({sample.source_id for sample in candidates}),
        requested_profiles=sorted(set(requested_profiles)),
        seed=seed,
        intended_output=str(Path(intended_output).resolve()),
        sample_count=len(candidates),
        provenance={
            "note": (
                "Candidate samples only. Rendering/distortion would be performed "
                "by the external clouda-data-factory project; this repository "
                "never calls it."
            )
        },
    )
    request_path = atomic_write_text(
        handoff_dir / "data_factory_handoff.json",
        json.dumps(handoff.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    return handoff, candidate_path, request_path
