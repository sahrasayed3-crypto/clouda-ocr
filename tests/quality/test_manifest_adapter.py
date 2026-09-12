"""Tests for clouda_data.quality.manifest_adapter (Wave2-B contract)."""

from __future__ import annotations

import hashlib

import pytest

from clouda_data.quality.manifest_adapter import (
    UnsafeArtifactPath,
    load_manifest,
    manifest_sha256,
    resolve_artifact_path,
    run_identity,
    training_stream_contract,
)
from conftest import make_manifest, make_row


class TestLoadManifest:
    def test_round_trip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        manifest_path = make_manifest(
            tmp_path,
            [
                make_row("smp_b", text="نص ب"),
                make_row("smp_a", text="نص أ"),
            ],
        )
        header, samples = load_manifest(manifest_path)
        assert header["_schema_version"] == "clouda.pretraining.manifest.v1"
        assert header["_row_count"] == 2
        # Canonical writer sorts rows; loader preserves that order.
        assert [s.sample_id for s in samples] == ["smp_a", "smp_b"]
        assert all(isinstance(s, object) for s in samples)
        assert samples[0].text == "نص أ"

    def test_sha_stability(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        rows = [make_row("smp_a", text="نص")]
        first = make_manifest(tmp_path / "one", rows)
        second = make_manifest(tmp_path / "two", rows)
        assert manifest_sha256(first) == manifest_sha256(second)
        expected = hashlib.sha256(first.read_bytes()).hexdigest()
        assert manifest_sha256(first) == expected


class TestResolveArtifactPath:
    def test_safe_path_resolves_under_root(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        sample = make_row("smp_a", image_path="images/doc/page_001.png")
        resolved = resolve_artifact_path(sample, tmp_path)
        assert resolved == (tmp_path / "images" / "doc" / "page_001.png").resolve()

    def test_traversal_path_refused(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        sample = make_row("smp_a", image_path="../x.png")
        with pytest.raises(UnsafeArtifactPath):
            resolve_artifact_path(sample, tmp_path)

    def test_absolute_path_refused(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        absolute = str(tmp_path / "evil.png")
        sample = make_row("smp_a", image_path=absolute)
        with pytest.raises(UnsafeArtifactPath):
            resolve_artifact_path(sample, tmp_path)

    def test_symlink_refused(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        target = tmp_path / "outside.png"
        target.write_bytes(b"png")
        link = tmp_path / "images"
        link.mkdir()
        try:
            (link / "page.png").symlink_to(target)
        except OSError:
            # Windows without symlink privilege: exercised indirectly.
            pytest.skip("symlink privilege not available on this host")
        sample = make_row("smp_a", image_path="images/page.png")
        with pytest.raises(UnsafeArtifactPath):
            resolve_artifact_path(sample, tmp_path)


class TestRunIdentity:
    def test_deterministic(self) -> None:
        first = run_identity(
            "a" * 64, "clouda.quality.config.v1+deadbeef", {"dhash": "1.0"}
        )
        second = run_identity(
            "a" * 64, "clouda.quality.config.v1+deadbeef", {"dhash": "1.0"}
        )
        assert first == second
        assert first["run_id"].startswith("QRUN_")
        assert len(first["run_id"]) == len("QRUN_") + 16

    def test_differs_on_inputs(self) -> None:
        base = run_identity("a" * 64, "cfg", {"dhash": "1.0"})
        other = run_identity("b" * 64, "cfg", {"dhash": "1.0"})
        assert base["run_id"] != other["run_id"]
        shuffled = run_identity("a" * 64, "cfg", {"dhash": "1.0", "ahash": "1"})
        mixed = run_identity("a" * 64, "cfg", {"ahash": "1", "dhash": "1.0"})
        assert shuffled["run_id"] == mixed["run_id"]


class TestTrainingStreamContract:
    def test_canonical_loader_handoff_documented(self) -> None:
        contract = training_stream_contract()
        assert isinstance(contract, str)
        assert "CANONICAL" in contract
        assert "validate_canonical_manifest" in contract
        assert "StreamingTrainingDataLoader" in contract
        assert "derived" in contract
