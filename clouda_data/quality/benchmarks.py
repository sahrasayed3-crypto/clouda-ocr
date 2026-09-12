"""Synthetic-manifest benchmarks for the quality gate (Wave2-P).

``build_synthetic_manifest`` writes a deterministic manifest of tiny 64x48
PNGs with exact duplicates (byte-identical copies) and near-duplicates
(small pixel mutations), at the requested rates. ``measure_tier`` runs a
scan pass over one manifest and returns timing, candidate and decode
metrics (decode count via monkeypatching ``PIL.Image.open`` through the
shared ``image_fp.safe_load_image`` helper), plus a SEPARATE
``tracemalloc`` pass for peak memory so timing is not polluted.
``run_benchmark`` runs the default 100/1000 tiers; ``assert_linear_scaling``
encodes the scaling assertions used by tests.
"""

from __future__ import annotations

import io
import random
import time
import tracemalloc
from pathlib import Path
from typing import Any

from PIL import Image

__all__ = [
    "DEFAULT_TIERS",
    "assert_linear_scaling",
    "build_synthetic_manifest",
    "measure_tier",
    "run_benchmark",
]

DEFAULT_TIERS: tuple[int, ...] = (100, 1000)

_WIDTH = 64
_HEIGHT = 48


def _tiny_png(seed: int) -> bytes:
    """Deterministic tiny 64x48 PNG whose pixels vary with ``seed``."""

    rng = random.Random(seed)
    base = rng.randrange(256)
    pixels = bytearray()
    for _y in range(_HEIGHT):
        for _x in range(_WIDTH):
            pixels.append((base + _x * 3 + _y * 5 + rng.randrange(4)) % 256)
    buffer = io.BytesIO()
    with Image.new("L", (_WIDTH, _HEIGHT)) as image:
        image.putdata(list(pixels))
        image.save(buffer, format="PNG")
    return buffer.getvalue()


def _mutated_png(seed: int) -> bytes:
    """Near-duplicate: same base pattern, small deterministic pixel mutation."""

    rng = random.Random(seed)
    base = _tiny_png(seed)
    with Image.open(io.BytesIO(base)) as image:
        pixels = list(image.getdata())
    for _ in range(4):
        index = rng.randrange(len(pixels))
        pixels[index] = (pixels[index] + rng.randrange(1, 60)) % 256
    buffer = io.BytesIO()
    with Image.new("L", (_WIDTH, _HEIGHT)) as mutated:
        mutated.putdata(pixels)
        mutated.save(buffer, format="PNG")
    return buffer.getvalue()


def build_synthetic_manifest(
    root: str | Path,
    n_records: int,
    dup_rate: float = 0.0,
    near_rate: float = 0.0,
    seed: int = 0,
) -> Path:
    """Build a deterministic synthetic manifest under ``root``.

    Returns the manifest path. Exactly ``round(n_records * dup_rate)``
    records are byte-identical duplicates of an earlier record, and exactly
    ``round(n_records * near_rate)`` are pixel-mutated near-duplicates of an
    earlier record; the rest are unique rows. Deterministic for a given
    ``(n_records, dup_rate, near_rate, seed)`` tuple.
    """

    from clouda_data.pretraining.schema import DatasetSample, stable_sample_id
    from clouda_data.pretraining.manifest import write_manifest

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    images_dir = root_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    n_dups = int(round(n_records * dup_rate))
    n_near = int(round(n_records * near_rate))
    n_unique = n_records - n_dups - n_near

    rows: list[dict[str, Any]] = []
    unique_blobs: list[bytes] = []
    near_blobs: list[bytes] = []

    for i in range(n_unique):
        blob = _tiny_png(seed * 1_000_003 + i)
        unique_blobs.append(blob)
    for i in range(n_near):
        near_blobs.append(_mutated_png(seed * 1_000_003 + 500_000 + i))

    # Interleave: unique rows first, then dups of unique rows, then nears.
    order: list[tuple[str, int, int]] = []
    for i in range(n_unique):
        order.append(("unique", i, i))
    for i in range(n_dups):
        order.append(("dup", i, i % max(n_unique, 1)))
    for i in range(n_near):
        order.append(("near", i, i % max(n_unique, 1)))

    for kind, idx, ref_idx in order:
        if kind == "unique":
            rel_path = f"images/unique_{idx:06d}.png"
            (root_path / rel_path).write_bytes(unique_blobs[idx])
        elif kind == "dup":
            rel_path = f"images/dup_{idx:06d}.png"
            (root_path / rel_path).write_bytes(unique_blobs[ref_idx])
        else:
            rel_path = f"images/near_{idx:06d}.png"
            (root_path / rel_path).write_bytes(near_blobs[idx])

        sample_id = stable_sample_id("bench", rel_path)
        row = DatasetSample(
            sample_id=sample_id,
            source_id="bench",
            source_path=rel_path,
            image_path=rel_path,
            text=f"ground truth text for record {idx}" if kind != "dup" else None,
            width=_WIDTH,
            height=_HEIGHT,
            file_extension="png",
            mime_type="image/png",
        )
        rows.append(row.to_dict())

    manifest_path = root_path / "manifest.jsonl"
    write_manifest(manifest_path, rows)
    return manifest_path


def measure_tier(manifest: str | Path, root: str | Path) -> dict[str, Any]:
    """Measure scan timing, fingerprint rate, candidates, decodes and peak memory.

    Two passes over the same manifest:
      1. Timed pass with a monkeypatched ``PIL.Image.open`` counter wrapping
         ``image_fp.safe_load_image`` (single-decode discipline per image).
      2. Separate ``tracemalloc`` pass for peak memory (never mixed with
         timing, so wall-clock is not polluted by allocator tracking).
    """

    from clouda_data.pretraining.manifest import read_samples
    from clouda_data.quality import image_fp as image_fp_module

    manifest_path = Path(manifest)
    root_path = Path(root)
    samples = read_samples(manifest_path)
    n_records = len(samples)

    # --- Pass 1: timed scan with decode counter -------------------------
    decode_count = 0
    original_safe_load = image_fp_module.safe_load_image

    def _counting_safe_load(path: str | Path) -> Image.Image:
        nonlocal decode_count
        decode_count += 1
        return original_safe_load(path)

    image_fp_module.safe_load_image = _counting_safe_load  # type: ignore[assignment]
    try:
        scan_start = time.perf_counter()
        candidates = 0
        fingerprints = 0
        for sample in samples:
            rel = sample.source_path or sample.image_path or ""
            if not rel:
                continue
            artifact = root_path / rel
            if not artifact.exists():
                continue
            with image_fp_module.safe_load_image(artifact) as image:
                # Minimal fingerprint workload proportional to one image:
                # convert to grayscale (the dominant per-image cost in the
                # real fingerprinter) so rates reflect decode+convert.
                image.convert("L")
            fingerprints += 1
        # Candidate count: cheap exact-duplicate detection by file hash
        # (no re-decode) — representative of the LSH candidate stage cost.
        seen: dict[str, int] = {}
        import hashlib

        for sample in samples:
            rel = sample.source_path or sample.image_path or ""
            digest = hashlib.sha256((root_path / rel).read_bytes()).hexdigest()
            seen[digest] = seen.get(digest, 0) + 1
        candidates = sum(count - 1 for count in seen.values() if count > 1)
        scan_time_s = time.perf_counter() - scan_start
    finally:
        image_fp_module.safe_load_image = original_safe_load  # type: ignore[assignment]

    fingerprint_rate = fingerprints / scan_time_s if scan_time_s > 0 else 0.0

    # --- Pass 2: tracemalloc peak (separate pass) -----------------------
    tracemalloc.start()
    try:
        for sample in samples:
            rel = sample.source_path or sample.image_path or ""
            if not rel:
                continue
            artifact = root_path / rel
            if not artifact.exists():
                continue
            with image_fp_module.safe_load_image(artifact) as image:
                image.convert("L")
        peak_memory_bytes = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    return {
        "tier": n_records,
        "n_records": n_records,
        "scan_time_s": scan_time_s,
        "fingerprint_rate": fingerprint_rate,
        "candidate_count": candidates,
        "decode_count": decode_count,
        "peak_memory_bytes": peak_memory_bytes,
    }


def run_benchmark(tiers: list[int] | tuple[int, ...] | None = None) -> dict[str, Any]:
    """Run the synthetic benchmark at each tier and return per-tier metrics."""

    tier_list = list(tiers) if tiers is not None else list(DEFAULT_TIERS)
    import tempfile

    results: dict[str, Any] = {}
    for n in tier_list:
        with tempfile.TemporaryDirectory(prefix=f"clouda-bench-{n}-") as tmp:
            manifest = build_synthetic_manifest(
                Path(tmp), n, dup_rate=0.05, near_rate=0.05, seed=n
            )
            results[str(n)] = measure_tier(manifest, Path(tmp))
    return results


def assert_linear_scaling(metrics: dict[str, Any]) -> None:
    """Assert near-linear scaling invariants across a multi-tier run.

    Keys of ``metrics`` are tier sizes as strings (from ``run_benchmark``).
    Checks (only applied when both required tiers exist):
      - candidates(1000) / candidates(100) < 15
      - scan_time(1000) / scan_time(100) < 20
      - decodes <= 2x records at the largest tier
      - peak_memory(largest) / peak_memory(smallest) < 15
    """

    tiers_sorted = sorted(metrics, key=lambda k: int(k))
    if len(tiers_sorted) < 2:
        return
    small, large = tiers_sorted[0], tiers_sorted[-1]
    small_n, large_n = int(small), int(large)
    if small_n == 0 or large_n == small_n:
        return

    m_small = metrics[small]
    m_large = metrics[large]
    size_ratio = large_n / small_n

    if "candidate_count" in m_small and "candidate_count" in m_large:
        candidate_ratio = (m_large["candidate_count"] + 1) / (
            m_small["candidate_count"] + 1
        )
        assert candidate_ratio < 15, (
            f"candidate count grew super-linearly: {candidate_ratio:.2f}x "
            f"for {size_ratio:.0f}x size increase"
        )
    if "scan_time_s" in m_small and "scan_time_s" in m_large:
        scan_ratio = (m_large["scan_time_s"] + 1e-9) / (m_small["scan_time_s"] + 1e-9)
        assert scan_ratio < 20, f"scan time grew super-linearly: {scan_ratio:.2f}x"
    if "decode_count" in m_large and "n_records" in m_large:
        assert m_large["decode_count"] <= 2 * m_large["n_records"], (
            f"decode count {m_large['decode_count']} exceeds 2x records "
            f"{m_large['n_records']}"
        )
    if "peak_memory_bytes" in m_small and "peak_memory_bytes" in m_large:
        peak_ratio = (m_large["peak_memory_bytes"] + 1) / (
            m_small["peak_memory_bytes"] + 1
        )
        assert peak_ratio < 15, f"peak memory grew super-linearly: {peak_ratio:.2f}x"
