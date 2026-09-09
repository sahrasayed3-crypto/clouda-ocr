"""Seed derivation: unified v1 plus byte-compatible legacy modes."""

from .derive import derive_seed, page_seed
from . import legacy

__all__ = ["derive_seed", "page_seed", "legacy"]
