"""Shared pytest fixtures for the training data loader subsystem tests.

Offline, CPU-only, no downloads. Synthetic manifests + tiny 4x4 PNGs only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
if str(FIXTURES_DIR) not in sys.path:
    sys.path.insert(0, str(FIXTURES_DIR))

from training_data_fixtures import (  # noqa: E402
    build_synthetic_dataset,
    shard_dataset,
)


@pytest.fixture
def synthetic_dataset(tmp_path: Path):
    manifest, root = build_synthetic_dataset(tmp_path / "dataset", count=24)
    return manifest, root


@pytest.fixture
def shard_index(synthetic_dataset, tmp_path):
    manifest, _root = synthetic_dataset
    return shard_dataset(manifest, tmp_path / "sharded", samples_per_shard=7)


@pytest.fixture
def index_path(shard_index, tmp_path):
    return tmp_path / "sharded" / "shard_index.json"
