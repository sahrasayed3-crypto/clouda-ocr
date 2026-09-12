"""Tiered text duplicate detection for the dataset quality gate.

Three tiers, each strictly more expensive than the last:

* Tier 0 -- ``raw_hash``: domain-separated SHA-256 over the untouched text.
* Tier 1 -- ``normalized_hash``: SHA-256 over the text after the shared
  :data:`DEDUPE_TEXT_POLICY` normalization (which folds diacritics, tatweel,
  alef/ya variants but deliberately keeps Arabic-Indic digits distinct).
  ``TEXT_POLICY_VERSION`` must be persisted alongside every fingerprint so a
  policy change invalidates old fingerprints deterministically.
* Tier 2 -- MinHash over character 4-grams with 128 blake2b-derived
  permutations, banded LSH (16 bands x 8 rows) with a per-band owner cap,
  and exact Jaccard verification.

Everything is fully deterministic: no ``hash()`` and no ``random`` module --
all permutation parameters derive from blake2b over the persisted seed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

from clouda_data.pretraining.normalize import NormalizationPolicy, normalize_text

MINHASH_SEED = "clouda.quality.textdup.v1"
_MINHASH_MERSENNE = 2**61 - 1
_MINHASH_U64_MASK = (1 << 64) - 1
NGRAM_SIZE = 4
MIN_SIGNATURE_LENGTH = 40

DEDUPE_TEXT_POLICY = NormalizationPolicy(
    unicode_form="NFKC",
    presentation_forms="compose",
    strip_bom=True,
    normalize_line_endings=True,
    preserve_line_breaks=True,
    collapse_whitespace=True,
    remove_zero_width=True,
    remove_control_characters=True,
    remove_tatweel=True,
    remove_diacritics=True,
    fold_alef=True,
    fold_ya=True,
    fold_digits=False,
)

TEXT_POLICY_VERSION = DEDUPE_TEXT_POLICY.version()


# ---------------------------------------------------------------------------
# Tier 0 / Tier 1
# ---------------------------------------------------------------------------


def raw_hash(text: str) -> str:
    """Tier 0: domain-separated SHA-256 of the raw, untouched text."""

    digest = hashlib.sha256()
    digest.update(b"clouda.text.raw.v1\x00")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def normalized_hash(text: str, policy: NormalizationPolicy = DEDUPE_TEXT_POLICY) -> str:
    """Tier 1: SHA-256 over the policy-normalized text value."""

    return hashlib.sha256(
        normalize_text(text, policy).value.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Tier 2: MinHash + LSH
# ---------------------------------------------------------------------------


def char_ngrams(text: str, n: int = NGRAM_SIZE) -> tuple[str, ...]:
    """Character n-grams (spaces included) over the given text."""

    if len(text) < n:
        return ()
    return tuple(text[i : i + n] for i in range(len(text) - n + 1))


def _shingle_u64(shingle: str) -> int:
    digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _MINHASH_U64_MASK


def _perm_params(seed: str, index: int) -> tuple[int, int]:
    payload = f"{seed}{index}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    a = int.from_bytes(digest[:8], "big") & _MINHASH_U64_MASK
    b = int.from_bytes(digest[8:], "big") & _MINHASH_U64_MASK
    return a % _MINHASH_MERSENNE or 1, b % _MINHASH_MERSENNE


def minhash_signature(
    text: str,
    perms: int = 128,
    seed: str = MINHASH_SEED,
    policy: NormalizationPolicy = DEDUPE_TEXT_POLICY,
) -> tuple[int, ...] | None:
    """MinHash signature over char 4-grams of the normalized text.

    Returns ``None`` when the normalized text is shorter than
    ``MIN_SIGNATURE_LENGTH`` characters (tier 2 skipped for tiny pages).
    """

    value = normalize_text(text, policy).value
    if len(value) < MIN_SIGNATURE_LENGTH:
        return None
    shingles = [_shingle_u64(g) for g in char_ngrams(value)]
    signature: list[int] = []
    for i in range(perms):
        a, b = _perm_params(seed, i)
        signature.append(min((a * x + b) % _MINHASH_MERSENNE for x in shingles))
    return tuple(signature)


def lsh_candidates(
    sigs: dict[str, tuple[int, ...]],
    bands: int = 16,
    rows: int = 8,
    owner_cap: int = 20,
) -> tuple[set[tuple[str, str]], dict[str, int]]:
    """Band LSH over ``{sample_id: signature}``.

    Returns ``(candidate_pairs, skipped)`` where ``skipped`` counts, per band
    key, the buckets that exceeded ``owner_cap`` and were therefore skipped
    (cannot form reliable candidate pairs).
    """

    if bands * rows != 128:  # 128 perms default; allow other widths
        raise ValueError("bands * rows must equal the signature width (128)")
    candidates: set[tuple[str, str]] = set()
    skipped: dict[str, int] = {}
    for band_idx in range(bands):
        buckets: dict[bytes, list[str]] = {}
        for sample_id, sig in sigs.items():
            if sig is None or len(sig) != bands * rows:
                continue
            key = b"|".join(
                str(v).encode("ascii")
                for v in sig[band_idx * rows : (band_idx + 1) * rows]
            )
            buckets.setdefault(key, []).append(sample_id)
        for key, members in buckets.items():
            if len(members) > owner_cap:
                skipped[
                    f"band_{band_idx}:{hashlib.blake2b(key, digest_size=8).hexdigest()}"
                ] = len(members)
                continue
            for a, b in combinations(sorted(members), 2):
                candidates.add((a, b))
    return candidates, skipped


def jaccard(
    ngrams_a: Iterable[str],
    ngrams_b: Iterable[str],
) -> float:
    """Exact Jaccard similarity between two n-gram sets."""

    set_a, set_b = set(ngrams_a), set(ngrams_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


@dataclass(frozen=True)
class TextPairClassification:
    """Result of tier-2 classification over a batch of texts."""

    near_text: tuple[tuple[str, str], ...]
    review: tuple[tuple[str, str], ...]
    skipped_ids: tuple[str, ...] = field(default_factory=tuple)


def classify_text_pairs(
    texts: dict[str, str],
    bands: int = 16,
    rows: int = 8,
    owner_cap: int = 20,
    near_threshold: float = 0.85,
    review_low: float = 0.70,
) -> TextPairClassification:
    """LSH candidates -> exact Jaccard -> near/review buckets.

    ``near_text`` holds pairs with J >= ``near_threshold`` (near-duplicate
    families); ``review`` holds pairs with ``review_low`` <= J <
    ``near_threshold`` (flagged for human review only).
    """

    sigs: dict[str, tuple[int, ...]] = {}
    skipped_ids: list[str] = []
    for sid, text in sorted(texts.items()):
        sig = minhash_signature(text)
        if sig is None:
            skipped_ids.append(sid)
            continue
        sigs[sid] = sig
    candidates, _skipped = lsh_candidates(
        sigs, bands=bands, rows=rows, owner_cap=owner_cap
    )
    near: list[tuple[str, str]] = []
    review: list[tuple[str, str]] = []
    for sid_a, sid_b in sorted(candidates):
        grams_a = char_ngrams(normalize_text(texts[sid_a], DEDUPE_TEXT_POLICY).value)
        grams_b = char_ngrams(normalize_text(texts[sid_b], DEDUPE_TEXT_POLICY).value)
        score = jaccard(grams_a, grams_b)
        if score >= near_threshold:
            near.append((sid_a, sid_b))
        elif score >= review_low:
            review.append((sid_a, sid_b))
    return TextPairClassification(
        near_text=tuple(sorted(near)),
        review=tuple(sorted(review)),
        skipped_ids=tuple(sorted(skipped_ids)),
    )


__all__ = [
    "MINHASH_SEED",
    "MIN_SIGNATURE_LENGTH",
    "NGRAM_SIZE",
    "DEDUPE_TEXT_POLICY",
    "TEXT_POLICY_VERSION",
    "TextPairClassification",
    "char_ngrams",
    "classify_text_pairs",
    "jaccard",
    "lsh_candidates",
    "minhash_signature",
    "normalized_hash",
    "raw_hash",
]
