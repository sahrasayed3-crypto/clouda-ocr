from __future__ import annotations

import json
from pathlib import Path


def _write_dataset(
    repo: Path,
    *,
    dataset_id: str,
    version: str,
    protected: bool = False,
    rows: int = 3,
) -> None:
    training = repo / "configs" / "training"
    training.mkdir(parents=True, exist_ok=True)
    manifest = training / f"{dataset_id}.jsonl"
    header = {
        "_schema_version": "clouda.pretraining.manifest.v1",
        "_row_count": rows,
        "dataset_id": dataset_id,
        "dataset_version": version,
        "dataset_role": "evaluation_only" if protected else "training",
        "derived_from": "raw-source@1",
    }
    records = [header]
    records.extend(
        {
            "sample_id": f"{dataset_id}-{index}",
            "source_id": "local-fixture",
            "text": f"sample {index}",
            "target_split": "holdout" if protected else "train",
            "provenance": {"origin": "test"},
        }
        for index in range(rows)
    )
    manifest.write_text(
        "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8"
    )
    config = {
        "schema_version": 1,
        "experiment": {"name": f"inspect-{dataset_id}"},
        "model": {"model_id": "mock/clouda", "adapter_type": "mock"},
        "dataset": {
            "dataset_id": dataset_id,
            "dataset_version": version,
            "manifest_path": manifest.name,
            "split": "train",
        },
        "runtime": {
            "device": "cpu",
            "output_root": "../../runs",
            "dry_run": True,
            "offline": True,
            "deterministic": True,
        },
    }
    (training / f"{dataset_id}.json").write_text(json.dumps(config), encoding="utf-8")


def _catalog(tmp_path: Path):
    from clouda_lab.dashboard.catalog import DatasetCatalog
    from clouda_lab.dashboard.settings import LabSettings

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "runs").mkdir()
    (repo / "benchmarks").mkdir()
    _write_dataset(repo, dataset_id="safe-set", version="v1")
    _write_dataset(repo, dataset_id="holdout-set", version="v2", protected=True)
    return DatasetCatalog(LabSettings.from_repo(repo))


def test_catalog_reports_canonical_identity_safety_and_loader_compatibility(
    tmp_path: Path,
):
    catalog = _catalog(tmp_path)

    datasets = catalog.list_datasets()
    assert [item["dataset_id"] for item in datasets] == ["holdout-set", "safe-set"]

    safe = catalog.get_dataset("safe-set")
    assert safe["identity"] == "safe-set@v1"
    assert len(safe["manifest_hash"]) == 64
    assert safe["row_count"] == 3
    assert safe["safety"] == {
        "protected_holdout": False,
        "evaluation_only": False,
        "training_allowed": True,
    }
    assert safe["loader"]["compatible"] is True
    assert safe["lineage"][0] == "raw-source@1"
    assert str(tmp_path) not in repr(safe)

    protected = catalog.get_dataset("holdout-set")
    assert protected["safety"]["protected_holdout"] is True
    assert protected["safety"]["evaluation_only"] is True
    assert protected["safety"]["training_allowed"] is False
    assert protected["loader"]["compatible"] is False


def test_preview_is_bounded_and_quality_state_is_honest(tmp_path: Path):
    catalog = _catalog(tmp_path)

    preview = catalog.preview("safe-set", limit=2)
    assert preview["returned"] == 2
    assert preview["truncated"] is True
    assert [row["sample_id"] for row in preview["records"]] == [
        "safe-set-0",
        "safe-set-1",
    ]
    assert catalog.quality_summary("safe-set")["status"] == "NOT RUN"

    protected_preview = catalog.preview("holdout-set", limit=1)
    protected_row = protected_preview["records"][0]
    assert protected_row["protected"] is True
    assert "text" not in protected_row
    assert "provenance" not in protected_row


def test_unknown_and_unsafe_dataset_identifiers_are_rejected(tmp_path: Path):
    catalog = _catalog(tmp_path)

    for dataset_id in ("missing", "../safe-set"):
        try:
            catalog.get_dataset(dataset_id)
        except (KeyError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid dataset accepted: {dataset_id}")


def test_quality_and_derived_dataset_actions_use_canonical_artifacts_without_path_leaks(
    tmp_path: Path,
):
    catalog = _catalog(tmp_path)

    report = catalog.run_quality("safe-set", max_samples=3)
    assert report["status"] == "AVAILABLE"
    assert report["run_id"]
    assert len(report["report"]["manifest_sha256"]) == 64
    assert report["duplicate_clusters"] == report["report"]["clusters"]["count"]

    derived = catalog.derive("safe-set", "safe-set-clean")
    assert derived["dataset_id"] == "safe-set"
    assert derived["derived_dataset"]["source_manifest_sha256"]
    assert derived["artifact"].startswith("runs/.lab-quality/")
    assert str(tmp_path) not in repr(derived)


def test_catalog_includes_results_store_dataset_metadata_without_enabling_training(
    tmp_path: Path,
):
    from clouda_data.results.service import ResultsService

    catalog = _catalog(tmp_path)
    ResultsService(catalog.settings.results_root).register_dataset(
        dataset_id="results-only",
        version="r1",
        name="Persisted evaluation dataset",
        manifest_sha256="a" * 64,
        splits=["test"],
    )

    result = catalog.get_dataset("results-only")
    assert result["source_type"] == "results_store"
    assert result["identity"] == "results-only@r1"
    assert result["safety"]["training_allowed"] is False
    assert result["loader"]["compatible"] is False


def test_dataset_source_registry_is_read_only_and_does_not_expose_download_targets(
    tmp_path: Path,
):
    catalog = _catalog(tmp_path)

    sources = catalog.list_sources()

    assert sources
    assert all("license" in source for source in sources)
    assert all(source["download_enabled"] is False for source in sources)
    rendered = repr(sources).lower()
    assert "http://" not in rendered
    assert "https://" not in rendered
    assert "sample_assets" not in rendered
