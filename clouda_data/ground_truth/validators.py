from __future__ import annotations

from .checksums import sha256_text


def require_text_preserved(original: str, candidate: str) -> None:
    if sha256_text(original) != sha256_text(candidate):
        raise ValueError(
            "Ground-truth text changed; distortion stages must not alter references."
        )


def validate_normalized_copy(original: str, normalized: str) -> None:
    if not original:
        raise ValueError("Original ground truth is empty.")
    if not normalized:
        raise ValueError("Normalized comparison text is empty.")
