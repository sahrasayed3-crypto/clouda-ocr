"""Unit tests: schema, sources, hashing, discovery, normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clouda_data.pretraining.config import PreparationConfig, load_preparation_config
from clouda_data.pretraining.discovery import draft_samples, scan_source
from clouda_data.pretraining.hashing import HashCache, sha256_file
from clouda_data.pretraining.normalize import NormalizationPolicy, normalize_text
from clouda_data.pretraining.schema import (
    DatasetSample,
    SCHEMA_VERSION,
    SplitName,
    stable_sample_id,
)
from clouda_data.pretraining.sources import (
    SourceDefinition,
    SourceRegistryError,
    get_source_definition,
    load_source_registry,
    register_source,
)

from pretraining_fixture import (
    ARABIC_TEXT,
    build_tiny_dataset,
)

# ---------------------------------------------------------------- schema


def test_sample_id_is_stable_and_payload_independent():
    first = stable_sample_id("src", "a/b.png", "a/b.txt")
    second = stable_sample_id("src", "a/b.png", "a/b.txt")
    other = stable_sample_id("src", "a/b.png", "other.txt")
    assert first == second and first != other
    assert first.startswith("smp_")


def test_sample_round_trip_preserves_enums_and_version():
    sample = DatasetSample(
        sample_id="smp_x",
        source_id="src",
        target_split=SplitName.HOLDOUT,
        file_sha256="deadbeef",
    )
    payload = sample.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    restored = DatasetSample.from_dict(payload)
    assert restored == sample
    assert restored.target_split == SplitName.HOLDOUT


def test_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown sample fields"):
        DatasetSample.from_dict(
            {"sample_id": "smp_y", "source_id": "src", "future_field": 1}
        )


# --------------------------------------------------------------- sources


def test_source_registry_round_trip_and_duplicate_rejection(tmp_path: Path):
    registry = tmp_path / "sources.jsonl"
    source = SourceDefinition(
        source_id="alpha",
        name="Alpha",
        local_root=str(tmp_path),
        languages=("ar", "en"),
        classification="public",
    )
    register_source(registry, source)
    with pytest.raises(SourceRegistryError):
        register_source(registry, source)
    register_source(registry, source, replace=True)
    loaded = load_source_registry(registry)
    assert len(loaded) == 1 and loaded[0].languages == ("ar", "en")
    assert get_source_definition(registry, "alpha").name == "Alpha"


def test_source_registry_rejects_invalid_classification(tmp_path: Path):
    with pytest.raises(SourceRegistryError):
        SourceDefinition.from_dict(
            {"source_id": "x", "name": "X", "classification": "bogus"}
        )


def test_source_registry_is_deterministic_jsonl(tmp_path: Path):
    registry = tmp_path / "sources.jsonl"
    for source_id in ("charlie", "alpha", "bravo"):
        register_source(
            registry,
            SourceDefinition(source_id=source_id, name=source_id.upper()),
        )
    lines = registry.read_text(encoding="utf-8").splitlines()
    ids = [json.loads(line)["source_id"] for line in lines[1:]]
    assert ids == ["alpha", "bravo", "charlie"]


# --------------------------------------------------------------- hashing


def test_sha256_file_matches_known_vector(tmp_path: Path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"clouda" * 1000)
    import hashlib

    assert sha256_file(path) == hashlib.sha256(b"clouda" * 1000).hexdigest()


def test_hash_cache_is_resumable_and_size_sensitive(tmp_path: Path):
    cache_path = tmp_path / "cache.jsonl"
    payload = tmp_path / "data.bin"
    payload.write_bytes(b"abcdef")
    cache = HashCache(cache_path)
    digest = "1" * 64
    cache.put("source", "data.bin", 6, 10, digest)
    cache.put("source", "data.bin", 6, 10, digest)  # idempotent append
    reopened = HashCache(cache_path)
    assert reopened.get("source", "data.bin", 6, 10) == digest
    assert reopened.get("source", "data.bin", 7, 10) is None
    # truncated last line must not crash the loader
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "trunc')
    assert HashCache(cache_path).get("source", "data.bin", 6, 10) == digest


# ------------------------------------------------------------- discovery


def test_scan_is_deterministic_and_skips_junk(tmp_path: Path):
    roots = build_tiny_dataset(tmp_path / "raw")
    files = scan_source(
        SourceDefinition(source_id="a", name="A", local_root=str(roots["source_a"]))
    )
    rel_paths = [item.rel_path for item in files]
    assert rel_paths == sorted(rel_paths)
    assert not any("__pycache__" in rel for rel in rel_paths)
    assert not any(rel.endswith(".DS_Store") for rel in rel_paths)
    kinds = {item.kind for item in files}
    assert "image" in kinds and "text" in kinds
    # resume equivalence: identical second scan
    assert [
        f.rel_path
        for f in scan_source(
            SourceDefinition(source_id="a", name="A", local_root=str(roots["source_a"]))
        )
    ] == rel_paths


def test_draft_pairs_sidecars_and_multipage_documents(tmp_path: Path):
    roots = build_tiny_dataset(tmp_path / "raw")
    source = SourceDefinition(
        source_id="a", name="A", local_root=str(roots["source_a"])
    )
    drafts = draft_samples(source, scan_source(source))
    by_image = {draft.image_rel_path: draft for draft in drafts if draft.image_rel_path}
    assert "mixed/mixed_doc-p1.png" in by_image
    doc_drafts = [d for d in drafts if d.document_id == "mixed_doc"]
    assert len(doc_drafts) == 2
    assert {d.page_index for d in doc_drafts} == {1, 2}
    # orphan text is excluded by default policy
    assert all(d.image_rel_path for d in drafts)


def test_draft_jsonl_records_flag_malformed_rows(tmp_path: Path):
    roots = build_tiny_dataset(tmp_path / "raw")
    source = SourceDefinition(
        source_id="b", name="B", local_root=str(roots["source_b"])
    )
    drafts = draft_samples(source, scan_source(source))
    malformed = [d for d in drafts if d.malformed_metadata]
    assert len(malformed) == 1
    with_image = [d for d in drafts if d.image_rel_path]
    assert len(with_image) >= 3  # alpha, beta, missing.png, csv row


# ---------------------------------------------------------- normalization


def test_normalization_preserves_raw_and_normalizes_copy():
    policy = NormalizationPolicy(remove_zero_width=True, collapse_whitespace=True)
    raw = "\ufeff" + ARABIC_TEXT + "\u200f  \r\n"
    result = normalize_text(raw, policy)
    assert result.value == ARABIC_TEXT
    assert "strip_bom" in result.applied
    assert "normalize_line_endings" in result.applied
    assert raw != result.value  # raw text untouched in caller's variable


def test_normalization_diacritics_and_tatweel_are_opt_in():
    text = "مُحَمَّـد"  # diacritics + tatweel between letters
    strict = normalize_text(text, NormalizationPolicy())
    aggressive = normalize_text(
        text,
        NormalizationPolicy(remove_diacritics=True, remove_tatweel=True),
    )
    assert strict.value == text  # safe default keeps diacritics
    assert "\u0640" not in aggressive.value
    assert aggressive.value == "محمد"


def test_normalization_zero_width_and_control_characters():
    policy = NormalizationPolicy(remove_zero_width=True, remove_control_characters=True)
    result = normalize_text("عال\u200b matte\u0007o", policy)
    assert "\u200b" not in result.value
    assert "\u0007" not in result.value


def test_normalization_digit_and_alef_folding_opt_in():
    text = "١٢٣ أ"
    folded = normalize_text(text, NormalizationPolicy(fold_digits=True, fold_alef=True))
    assert folded.value == "123 ا"
    untouched = normalize_text(text, NormalizationPolicy())
    assert untouched.value == text


def test_normalization_policy_fingerprint_changes_with_config():
    base = NormalizationPolicy()
    variant = NormalizationPolicy(remove_tatweel=True)
    assert base.fingerprint() != variant.fingerprint()
    assert base.version().startswith("clouda.pretraining.normalize.v1+")


# ---------------------------------------------------------------- config


def test_preparation_config_defaults_and_mapping():
    config = PreparationConfig()
    assert config.split_ratios["train"] == 0.8
    loaded = load_preparation_config(None)
    assert loaded.split_seed == config.split_seed
    custom = PreparationConfig.from_mapping(
        {"split_seed": 7, "normalization": {"remove_tatweel": True}}
    )
    assert custom.split_seed == 7
    assert custom.normalization.remove_tatweel is True
    with pytest.raises(Exception):
        PreparationConfig.from_mapping({"nonsense": 1})
    with pytest.raises(Exception):
        PreparationConfig.from_mapping({"split_ratios": {"train": 2.0}})


def test_preparation_config_json_round_trip(tmp_path: Path):
    config = PreparationConfig()
    path = tmp_path / "prep.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    loaded = load_preparation_config(path)
    assert loaded == config
