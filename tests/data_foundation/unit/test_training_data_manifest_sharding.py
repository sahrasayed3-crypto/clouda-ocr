"""Manifest input-contract, holdout-safety, sharding and shard-index tests."""

from __future__ import annotations

import json

import pytest

from clouda_data.pretraining.manifest import write_manifest
from clouda_data.training_data.input_contract import (
    ManifestInputError,
    ProtectedManifestError,
    validate_canonical_manifest,
)
from clouda_data.training_data.models import ShardStrategy
from clouda_data.training_data.sharding import (
    ShardIndexError,
    load_shard_index,
    verify_shards,
)

from tests.data_foundation.fixtures.training_data_fixtures import (
    DATASET_ID,
    DATASET_VERSION,
    make_sample_row,
    shard_dataset,
)

# ---------------------------------------------------------------------------
# Canonical manifest input
# ---------------------------------------------------------------------------


class TestCanonicalManifestInput:
    def test_valid_manifest_accepts(self, synthetic_dataset):
        manifest, _root = synthetic_dataset
        identity = validate_canonical_manifest(
            manifest, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION
        )
        assert identity.dataset_id == DATASET_ID
        assert identity.dataset_version == DATASET_VERSION
        assert identity.row_count == 24
        assert len(identity.manifest_sha256) == 64

    def test_missing_manifest_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_canonical_manifest(
                tmp_path / "nope.jsonl",
                dataset_id=DATASET_ID,
                dataset_version=DATASET_VERSION,
            )

    def test_identity_mismatch_rejected(self, synthetic_dataset):
        manifest, _root = synthetic_dataset
        with pytest.raises(ValueError, match="does not match configured"):
            validate_canonical_manifest(
                manifest, dataset_id="other_dataset", dataset_version=DATASET_VERSION
            )

    def test_version_mismatch_rejected(self, synthetic_dataset):
        manifest, _root = synthetic_dataset
        with pytest.raises(ValueError, match="does not match configured"):
            validate_canonical_manifest(
                manifest, dataset_id=DATASET_ID, dataset_version="9.9.9"
            )

    def test_malformed_json_rejected(self, tmp_path):
        path = tmp_path / "broken.jsonl"
        path.write_text('{"sample_id": "x"\n', encoding="utf-8")
        with pytest.raises(ValueError):
            list(
                validate_canonical_manifest(
                    path, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION
                )
            )

    def test_duplicate_sample_ids_rejected(self, tmp_path):
        rows = [make_sample_row(0), make_sample_row(0)]
        manifest = tmp_path / "dupe.jsonl"
        write_manifest(
            manifest,
            rows,
            metadata={"dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION},
        )
        with pytest.raises(ManifestInputError, match="Duplicate sample_id"):
            validate_canonical_manifest(
                manifest, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION
            )

    def test_row_without_payload_rejected(self, tmp_path):
        row = make_sample_row(0)
        del row["image_path"]
        del row["text"]
        manifest = tmp_path / "empty.jsonl"
        write_manifest(
            manifest,
            [row],
            metadata={"dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION},
        )
        with pytest.raises(ManifestInputError, match="neither image_path nor text"):
            validate_canonical_manifest(
                manifest, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION
            )

    def test_excluded_sample_rejected(self, tmp_path):
        row = make_sample_row(0)
        row["exclusion_reason"] = "quality_failure"
        manifest = tmp_path / "excluded.jsonl"
        write_manifest(
            manifest,
            [row],
            metadata={"dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION},
        )
        with pytest.raises(ManifestInputError, match="excluded"):
            validate_canonical_manifest(
                manifest, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION
            )

    def test_error_status_rejected(self, tmp_path):
        row = make_sample_row(0)
        row["validation_status"] = "error"
        manifest = tmp_path / "bad.jsonl"
        write_manifest(
            manifest,
            [row],
            metadata={"dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION},
        )
        with pytest.raises(ManifestInputError, match="validation_status"):
            validate_canonical_manifest(
                manifest, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION
            )


# ---------------------------------------------------------------------------
# Holdout / protected-data safety (fail closed)
# ---------------------------------------------------------------------------


class TestHoldoutSafety:
    def _write(self, tmp_path, rows):
        manifest = tmp_path / "m.jsonl"
        write_manifest(
            manifest,
            rows,
            metadata={"dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION},
        )
        return manifest

    def test_holdout_split_rejected(self, tmp_path):
        rows = [make_sample_row(0), make_sample_row(1, split="holdout")]
        with pytest.raises(ProtectedManifestError):
            validate_canonical_manifest(
                self._write(tmp_path, rows),
                dataset_id=DATASET_ID,
                dataset_version=DATASET_VERSION,
            )

    @pytest.mark.parametrize(
        "alias", ["HOLDOUT", "Holdout", "benchmark_holdout", "private_holdout"]
    )
    def test_holdout_aliases_rejected(self, tmp_path, alias):
        rows = [make_sample_row(0), make_sample_row(1, split=alias)]
        with pytest.raises(ProtectedManifestError):
            validate_canonical_manifest(
                self._write(tmp_path, rows),
                dataset_id=DATASET_ID,
                dataset_version=DATASET_VERSION,
            )

    def test_protected_marker_rejected(self, tmp_path):
        row = make_sample_row(5)
        row["protected"] = True
        with pytest.raises(ProtectedManifestError):
            validate_canonical_manifest(
                self._write(tmp_path, [make_sample_row(0), row]),
                dataset_id=DATASET_ID,
                dataset_version=DATASET_VERSION,
            )

    def test_nested_protected_marker_rejected(self, tmp_path):
        row = make_sample_row(6)
        row["provenance"] = {"protected": "yes"}
        with pytest.raises(ProtectedManifestError):
            validate_canonical_manifest(
                self._write(tmp_path, [make_sample_row(0), row]),
                dataset_id=DATASET_ID,
                dataset_version=DATASET_VERSION,
            )

    def test_role_holdout_rejected(self, tmp_path):
        row = make_sample_row(7)
        row["dataset_role"] = "evaluation_only"
        with pytest.raises(ProtectedManifestError):
            validate_canonical_manifest(
                self._write(tmp_path, [make_sample_row(0), row]),
                dataset_id=DATASET_ID,
                dataset_version=DATASET_VERSION,
            )

    def test_malformed_protection_metadata_rejected(self, tmp_path):
        row = make_sample_row(8)
        row["protected"] = "definitely-maybe"
        manifest = self._write(tmp_path, [make_sample_row(0), row])
        with pytest.raises(ValueError, match="malformed"):
            validate_canonical_manifest(
                manifest, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION
            )

    def test_malformed_split_metadata_rejected(self, tmp_path):
        row = make_sample_row(9)
        row["target_split"] = 123
        manifest = self._write(tmp_path, [row])
        with pytest.raises(ValueError, match="malformed split"):
            validate_canonical_manifest(
                manifest, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION
            )

    def test_clean_manifest_passes_holdout_gate(self, synthetic_dataset):
        manifest, _root = synthetic_dataset
        identity = validate_canonical_manifest(
            manifest, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION
        )
        assert identity.row_count == 24


# ---------------------------------------------------------------------------
# Sharding engine
# ---------------------------------------------------------------------------


class TestSharding:
    def test_no_missing_no_duplicate(self, synthetic_dataset, tmp_path):
        manifest, _root = synthetic_dataset
        index = shard_dataset(manifest, tmp_path / "out", samples_per_shard=7)
        assert index.total_samples == 24
        assert sum(e.sample_count for e in index.shards) == 24
        report = verify_shards(index, tmp_path / "out")
        assert report["ok"] and report["unique_samples"] == 24

    def test_deterministic_shard_ids_and_hashes(self, synthetic_dataset, tmp_path):
        manifest, _root = synthetic_dataset
        out1, out2 = tmp_path / "a", tmp_path / "b"
        i1 = shard_dataset(manifest, out1, samples_per_shard=10)
        i2 = shard_dataset(manifest, out2, samples_per_shard=10)
        assert [e.shard_id for e in i1.shards] == [e.shard_id for e in i2.shards]
        assert [e.sha256 for e in i1.shards] == [e.sha256 for e in i2.shards]

    def test_changed_config_changes_identity(self, synthetic_dataset, tmp_path):
        manifest, _root = synthetic_dataset
        i7 = shard_dataset(manifest, tmp_path / "c7", samples_per_shard=7)
        i10 = shard_dataset(manifest, tmp_path / "c10", samples_per_shard=10)
        assert i7.shard_config_hash != i10.shard_config_hash
        assert [e.shard_id for e in i7.shards] != [e.shard_id for e in i10.shards]

    def test_size_aware_strategy(self, synthetic_dataset, tmp_path):
        manifest, _root = synthetic_dataset
        index = shard_dataset(
            manifest,
            tmp_path / "sz",
            samples_per_shard=5,
            strategy=ShardStrategy.SIZE_AWARE,
        )
        assert index.total_samples == 24
        assert index.total_shards >= 1

    def test_shard_rows_preserve_provenance(self, synthetic_dataset, tmp_path):
        manifest, _root = synthetic_dataset
        index = shard_dataset(manifest, tmp_path / "prov", samples_per_shard=24)
        shard_file = tmp_path / "prov" / "shards" / index.shards[0].path
        record = json.loads(shard_file.read_text(encoding="utf-8").splitlines()[0])
        assert record["row"]["provenance"]["origin"] == "test_fixture"
        assert record["sample_id"].startswith("smp_")

    def test_verify_detects_tampering(self, synthetic_dataset, tmp_path):
        manifest, _root = synthetic_dataset
        index = shard_dataset(manifest, tmp_path / "tamper", samples_per_shard=12)
        shard_file = tmp_path / "tamper" / "shards" / index.shards[0].path
        shard_file.write_text('{"tampered": true}\n', encoding="utf-8")
        with pytest.raises(ShardIndexError, match="hash mismatch"):
            verify_shards(index, tmp_path / "tamper")

    def test_verify_detects_missing_shard(self, synthetic_dataset, tmp_path):
        manifest, _root = synthetic_dataset
        index = shard_dataset(manifest, tmp_path / "gone", samples_per_shard=12)
        (tmp_path / "gone" / "shards" / index.shards[0].path).unlink()
        with pytest.raises(ShardIndexError, match="missing"):
            verify_shards(index, tmp_path / "gone")


# ---------------------------------------------------------------------------
# Shard index
# ---------------------------------------------------------------------------


class TestShardIndex:
    def test_index_records_required_fields(self, shard_index):
        payload = shard_index.to_dict()
        for key in (
            "dataset_id",
            "dataset_version",
            "source_manifest_sha256",
            "shard_config_hash",
            "total_samples",
            "total_shards",
            "shards",
            "schema_version",
        ):
            assert key in payload
        assert payload["total_samples"] == 24
        assert payload["total_shards"] == len(payload["shards"])

    def test_index_roundtrip(self, shard_index):
        clone = type(shard_index).from_dict(shard_index.to_dict())
        assert clone == shard_index

    def test_index_rebuildable(self, synthetic_dataset, tmp_path):
        manifest, _root = synthetic_dataset
        i1 = shard_dataset(manifest, tmp_path / "r1", samples_per_shard=8)
        i2 = shard_dataset(manifest, tmp_path / "r2", samples_per_shard=8)
        assert i1.to_dict() == i2.to_dict()

    def test_loaded_index_matches_built(self, synthetic_dataset, tmp_path):
        manifest, _root = synthetic_dataset
        shard_dataset(manifest, tmp_path / "load", samples_per_shard=8)
        loaded = load_shard_index(tmp_path / "load" / "shard_index.json")
        assert loaded.total_samples == 24
        assert loaded.dataset_id == DATASET_ID

    def test_unsupported_schema_rejected(self):
        from clouda_data.training_data.sharding import ShardIndex

        with pytest.raises(ShardIndexError, match="schema"):
            ShardIndex.from_dict({"schema_version": "bogus.v0", "shards": []})
