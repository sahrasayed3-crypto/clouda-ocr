"""Lazy, path-safe artifact resolution for training samples.

Artifacts (page images, GT text, auxiliary metadata) are resolved through
the canonical path-safety rules from ``clouda_data.pretraining.schema``
(``canonical_relative_path``) and additionally re-resolved beneath the
dataset root to defend against symlink escape.

Image bytes are only read when the caller explicitly asks for them
(:func:`load_image_bytes` / :func:`load_image`), never during metadata
iteration.
"""

from __future__ import annotations

import io
from pathlib import Path

from clouda_data.pretraining.schema import canonical_relative_path

_TEXT_SUFFIXES = {".txt", ".text", ".gt", ".md"}


def resolve_artifact(root: str | Path, rel_path: str | None) -> Path | None:
    """Resolve a root-relative artifact path safely, or return None.

    Raises ValueError on traversal attempts, returns None when missing.
    """

    if rel_path is None:
        return None
    canonical = canonical_relative_path(str(rel_path))
    root_path = Path(root).resolve()
    candidate = (root_path / canonical).resolve()
    if candidate != root_path and root_path not in candidate.parents:
        raise ValueError(f"Artifact path escapes dataset root: {rel_path!r}")
    if not candidate.is_file():
        return None
    return candidate


def load_image_bytes(root: str | Path, rel_path: str) -> bytes:
    path = resolve_artifact(root, rel_path)
    if path is None:
        raise FileNotFoundError(f"Artifact not found under {root}: {rel_path!r}")
    return path.read_bytes()


def load_image(root: str | Path, rel_path: str):
    """Decode an image with Pillow; raises on corrupt/undecodable bytes."""

    from PIL import Image

    payload = load_image_bytes(root, rel_path)
    image = Image.open(io.BytesIO(payload))
    image.load()
    return image


def verify_image_decodable(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
        return True
    except Exception:
        return False


def verify_text_hash(text: str, expected_sha256: str) -> bool:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest() == expected_sha256


class ArtifactLoader:
    """Typed facade for lazy artifact access with optional size caps."""

    def __init__(self, root: str | Path, *, max_image_bytes: int = 512 << 20) -> None:
        self.root = Path(root)
        self.max_image_bytes = max_image_bytes

    def image_bytes(self, ref_rel_path: str | None) -> bytes | None:
        if ref_rel_path is None:
            return None
        payload = load_image_bytes(self.root, ref_rel_path)
        if len(payload) > self.max_image_bytes:
            raise ValueError(f"Artifact exceeds max_image_bytes: {ref_rel_path!r}")
        return payload

    def image(self, ref_rel_path: str | None):
        if ref_rel_path is None:
            return None
        return load_image(self.root, ref_rel_path)

    def text(self, inline_text: str | None, rel_path: str | None = None) -> str | None:
        if inline_text is not None:
            return inline_text
        if rel_path is None:
            return None
        path = resolve_artifact(self.root, rel_path)
        if path is None:
            return None
        return path.read_text(encoding="utf-8")

    def resolve(self, rel_path: str | None) -> Path | None:
        return resolve_artifact(self.root, rel_path)
