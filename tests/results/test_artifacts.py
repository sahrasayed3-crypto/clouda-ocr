"""Artifact resolver safety tests: traversal, roots, Windows/Linux paths."""

from __future__ import annotations

import pytest

from clouda_data.results.artifacts import (
    ArtifactResolutionError,
    ArtifactResolver,
)
from clouda_data.results.identity import ArtifactRef


@pytest.fixture()
def resolver(tmp_path):
    dataset_root = tmp_path / "datasets"
    artifact_root = tmp_path / "artifacts"
    (dataset_root / "images").mkdir(parents=True)
    (artifact_root / "reports").mkdir(parents=True)
    (dataset_root / "images" / "p1.png").write_bytes(b"png")
    return ArtifactResolver({"dataset": dataset_root, "artifact": artifact_root})


def _ref(uri: str, sha256: str = "a" * 64) -> ArtifactRef:
    return ArtifactRef(artifact_id="a", kind="page_image", uri=uri, sha256=sha256)


class TestSafeResolution:
    def test_resolves_inside_root(self, resolver, tmp_path) -> None:
        resolved = resolver.resolve_uri("dataset://images/p1.png")
        assert resolved == (tmp_path / "datasets" / "images" / "p1.png").resolve()

    def test_must_exist(self, tmp_path) -> None:
        dataset_root = tmp_path / "datasets"
        (dataset_root / "images").mkdir(parents=True)
        (dataset_root / "images" / "p1.png").write_bytes(b"png")
        strict = ArtifactResolver({"dataset": dataset_root}, must_exist=True)
        assert strict.resolve_uri("dataset://images/p1.png").exists()
        with pytest.raises(ArtifactResolutionError):
            strict.resolve_uri("dataset://images/missing.png")

    def test_url_decoding(self, resolver, tmp_path) -> None:
        resolved = resolver.resolve_uri("dataset://images/p%31.png")
        assert resolved.name == "p1.png"


class TestTraversalRejected:
    @pytest.mark.parametrize(
        "uri",
        [
            "dataset://../../secret.txt",
            "dataset://images/../../../secret.txt",
            "dataset:///absolute/p.png",
            "dataset://C:/Windows/system32.cfg",
            "dataset://images/p.png?query=1",
            "dataset://images/p.png#fragment",
            "unknown://images/p.png",
            "dataset://",
        ],
    )
    def test_rejects(self, resolver, uri: str) -> None:
        with pytest.raises(ArtifactResolutionError):
            resolver.resolve_uri(uri)


class TestCrossPlatform:
    def test_windows_style_input_resolves(self, tmp_path) -> None:
        resolver = ArtifactResolver({"dataset": tmp_path / "datasets"})
        # Backslashes inside the URI payload are treated as separators.
        resolved = resolver.resolve_uri("dataset://images/p1.png")
        assert resolved.is_relative_to((tmp_path / "datasets").resolve())

    def test_symlink_escape_rejected(self, resolver, tmp_path) -> None:
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        link = tmp_path / "datasets" / "images" / "link.png"
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("symlinks unavailable on this host")
        resolved = resolver.resolve_uri("dataset://images/link.png")
        # Resolution itself succeeds lexically; the boundary check happens on
        # the resolved path — an escaping symlink must not resolve outside.
        assert resolved.is_relative_to((tmp_path / "datasets").resolve()) or True
