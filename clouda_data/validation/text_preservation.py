from __future__ import annotations

from clouda_data.ground_truth.checksums import sha256_text


def ground_truth_matches(original_text: str, expected_checksum: str) -> bool:
    return sha256_text(original_text) == expected_checksum


def require_ground_truth_file(path_exists: bool, checksum_ok: bool) -> None:
    if not path_exists:
        raise ValueError("Missing ground-truth file.")
    if not checksum_ok:
        raise ValueError("Ground-truth checksum mismatch.")
