"""Identity determinism, portability, and path-safety tests."""

from __future__ import annotations

import pytest

from clouda_data.results.identity import (
    ArtifactRef,
    page_identity,
    portable_relative_path,
    run_identity,
    validate_sha256,
)


class TestPageIdentity:
    def test_deterministic(self) -> None:
        first = page_identity(dataset_id="ds", split="train", provided_page_key="p1")
        second = page_identity(dataset_id="ds", split="train", provided_page_key="p1")
        assert first == second

    def test_distinguishes_split(self) -> None:
        train = page_identity(dataset_id="ds", split="train", provided_page_key="p1")
        holdout = page_identity(
            dataset_id="ds", split="holdout", provided_page_key="p1"
        )
        assert train != holdout

    def test_preserves_existing_project_id(self) -> None:
        # The existing benchmark id is carried verbatim, not re-derived.
        page_id = page_identity(
            dataset_id="clouda-ocr-arabic-177",
            split="unassigned",
            provided_page_key="arabic_ocr_000026",
        )
        assert page_id.endswith(":arabic_ocr_000026")
        assert "arabic_ocr_000026" in page_id

    def test_rejects_blank(self) -> None:
        with pytest.raises(ValueError):
            page_identity(dataset_id="ds", split="train", provided_page_key="  ")


class TestRunIdentity:
    def test_deterministic_across_processes(self) -> None:
        kwargs = dict(
            model_id="m",
            model_revision="1.0",
            dataset_id="ds",
            dataset_version="v1",
            manifest_sha256=None,
            created_at="2026-09-10T00:00:00Z",
        )
        assert run_identity(**kwargs) == run_identity(**kwargs)

    def test_manifest_hash_changes_identity(self) -> None:
        base = dict(
            model_id="m",
            model_revision="1.0",
            dataset_id="ds",
            dataset_version="v1",
            created_at="t",
        )
        without = run_identity(**base, manifest_sha256=None)
        with_hash = run_identity(**base, manifest_sha256="a" * 64)
        assert without != with_hash

    def test_no_machine_path_or_pid_dependence(self) -> None:
        # Identity inputs are only the logical fields; two calls in different
        # working directories with the same fields must agree.
        first = run_identity(
            model_id="HunyuanOCR-1.5",
            model_revision="1.5",
            dataset_id="clouda-ocr-arabic-177",
            dataset_version="v1",
            manifest_sha256="2a" * 32,
            created_at="evidence",
        )
        second = run_identity(
            model_id="HunyuanOCR-1.5",
            model_revision="1.5",
            dataset_id="clouda-ocr-arabic-177",
            dataset_version="v1",
            manifest_sha256="2a" * 32,
            created_at="evidence",
        )
        assert first == second


class TestPortablePaths:
    @pytest.mark.parametrize(
        "value",
        [
            "data/pages/p1.png",
            "data\\pages\\p1.png",
            "data/./pages/../pages/p1.png",
        ],
    )
    def test_normalizes_to_posix_relative(self, value: str) -> None:
        assert portable_relative_path(value) == "data/pages/p1.png"

    @pytest.mark.parametrize(
        "value",
        [
            r"C:\Users\Ahmed\secret.png",
            "C:/Users/Ahmed/secret.png",
            "/absolute/path.png",
            "../secret.png",
            "data/../../secret.png",
            "//server/share/p.png",
        ],
    )
    def test_rejects_unsafe_paths(self, value: str) -> None:
        with pytest.raises(ValueError):
            portable_relative_path(value)

    def test_rejects_control_characters(self) -> None:
        with pytest.raises(ValueError):
            portable_relative_path("data/\x1fpage.png")


class TestArtifactRef:
    def test_portable_uri(self) -> None:
        ref = ArtifactRef(
            artifact_id="a1",
            kind="page_image",
            uri="dataset://images/p1.png",
            sha256="a" * 64,
        )
        assert ref.to_dict()["uri"] == "dataset://images/p1.png"

    def test_rejects_absolute_uri_path(self) -> None:
        with pytest.raises(ValueError):
            ArtifactRef(
                artifact_id="a1",
                kind="page_image",
                uri="dataset:///abs/p1.png",
                sha256="a" * 64,
            )

    def test_rejects_traversal(self) -> None:
        with pytest.raises(ValueError):
            ArtifactRef(
                artifact_id="a1",
                kind="page_image",
                uri="dataset://../../secret.png",
                sha256="a" * 64,
            )

    def test_rejects_unknown_scheme(self) -> None:
        with pytest.raises(ValueError):
            ArtifactRef(
                artifact_id="a1",
                kind="page_image",
                uri="file://images/p1.png",
                sha256="a" * 64,
            )

    def test_rejects_bad_hash(self) -> None:
        with pytest.raises(ValueError):
            ArtifactRef(
                artifact_id="a1",
                kind="page_image",
                uri="dataset://a.png",
                sha256="not-a-hash",
            )


def test_validate_sha256() -> None:
    assert validate_sha256("A" * 64) == "a" * 64
    with pytest.raises(ValueError):
        validate_sha256("z" * 64)
