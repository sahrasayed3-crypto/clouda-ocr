"""Doctor test fixtures: synthetic report builders and env isolation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the worktree sources importable regardless of how pytest was invoked.
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))


@pytest.fixture()
def clean_worktree_root(tmp_path: Path) -> Path:
    """A temp directory that looks like a repo root (pyproject present)."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "fake"\nrequires-python = ">=3.11,<3.12"\n',
        encoding="utf-8",
    )
    return tmp_path
