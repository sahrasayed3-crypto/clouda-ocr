"""End-to-end factory: image input, all three engines, exports,
determinism across worker counts, resume safety, error isolation.

Migrated from the standalone repository (tests/test_factory.py); imports
retargeted to the integrated package.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from clouda_data.factory.factory import generate_run
from clouda_data.factory.manifest import read_manifest

PIL_features = pytest.importorskip("PIL")

try:
    from PIL import features as _pil_features

    _HAS_RAQM = _pil_features.check("raqm")
except Exception:  # pragma: no cover
    _HAS_RAQM = False

try:
    import weasyprint  # noqa: F401

    _HAS_WEASYPRINT = True
except Exception:  # pragma: no cover
    _HAS_WEASYPRINT = False

PROFILES = ["old_book_medium", "05_old_book_medium", "12_hard_composite"]


def _make_input(tmp_path):
    inbox = tmp_path / "input"
    inbox.mkdir()
    img = Image.new("RGB", (620, 877), (250, 247, 240))
    draw = ImageDraw.Draw(img)
    y = 60
    for _ in range(12):
        draw.text((580, y), "0" * 25, fill=(20, 20, 20))
        y += 30
    img.save(inbox / "sample_page.png")
    (inbox / "sample_page.txt").write_text("sample ground truth\n", encoding="utf-8")
    return inbox, inbox / "sample_page.png", inbox / "sample_page.txt"


def _run(runs_root, inputs, workers=1, seed_mode="v1", run_id=None, resume=False):
    return generate_run(
        inputs=[inputs],
        runs_root=runs_root,
        profile_names=PROFILES,
        variants=3,
        base_seed=12345,
        seed_mode=seed_mode,
        backend="raqm",
        workers=workers,
        export_pdf=True,
        export_png=True,
        max_pages=4,
        resume=resume,
        run_id=run_id,
    )


def _shas(run_dir):
    rows = read_manifest(run_dir / "manifest.jsonl")
    return sorted(
        (r["variant_id"], r["page_index"], r["output_sha256"])
        for r in rows
        if r["status"] == "ok"
    )


def test_image_input_run_complete_outputs(tmp_path):
    inbox, image, _ = _make_input(tmp_path)
    runs = tmp_path / "runs"
    meta = _run(runs, image)
    run_dir = runs / meta["run_id"]
    assert meta["counts"]["ok"] == 3 and meta["counts"]["errors"] == 0
    assert (run_dir / "run_config.json").is_file()
    assert (run_dir / "manifest.jsonl").is_file()
    assert (run_dir / "manifest.csv").is_file()
    assert (run_dir / "metadata.json").is_file()
    assert list((run_dir / "scans").iterdir()), "distorted variant dirs must exist"
    assert list((run_dir / "sources").iterdir()), "pristine source copy must exist"
    assert list((run_dir / "gt").iterdir()), "sibling .txt becomes ground truth"
    for variant_dir in (run_dir / "scans").iterdir():
        assert (variant_dir / "page_000000.png").is_file()
        assert (variant_dir / "page_000000.pdf").is_file()
    original = image.read_bytes()
    assert image.read_bytes() == original


def test_manifest_rows_are_complete_and_hashed(tmp_path):
    inbox, image, _ = _make_input(tmp_path)
    runs = tmp_path / "runs"
    meta = _run(runs, image)
    run_dir = runs / meta["run_id"]
    rows = read_manifest(run_dir / "manifest.jsonl")
    assert len(rows) == 3
    import hashlib

    for row in rows:
        assert row["status"] == "ok"
        assert row["seed_mode"] == "v1"
        assert row["source_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
        assert row["gt_path"] and row["gt_sha256"]
        assert row["transform_steps"]
        out = run_dir / row["output_path"]
        assert out.is_file()
        assert hashlib.sha256(out.read_bytes()).hexdigest() == row["output_sha256"]


def test_worker_count_does_not_change_output(tmp_path):
    inbox, image, _ = _make_input(tmp_path)
    rd1 = tmp_path / "runs1"
    rd2 = tmp_path / "runs2"
    _run(rd1, image, workers=1)
    _run(rd2, image, workers=2)
    assert _shas(next(rd1.iterdir())) == _shas(next(rd2.iterdir()))


def test_seed_mode_changes_output(tmp_path):
    inbox, image, _ = _make_input(tmp_path)
    rd1 = tmp_path / "runsv1"
    rd2 = tmp_path / "runsf"
    _run(rd1, image, seed_mode="v1")
    _run(rd2, image, seed_mode="arabic_scan_factory")
    assert _shas(next(rd1.iterdir())) != _shas(next(rd2.iterdir()))


def test_resume_preserves_manifest_and_refuses_conflict(tmp_path):
    inbox, image, _ = _make_input(tmp_path)
    runs = tmp_path / "runs"
    meta = _run(runs, image)
    run_dir = runs / meta["run_id"]
    before = _shas(run_dir)
    _run(runs, image, run_id=meta["run_id"], resume=True)
    assert _shas(run_dir) == before
    with pytest.raises(FileExistsError):
        _run(runs, image, run_id=meta["run_id"], resume=False)


def test_text_render_failure_is_isolated(tmp_path):
    """On hosts without native render libs the text document yields a clean
    error row instead of crashing the run."""
    if _HAS_RAQM or _HAS_WEASYPRINT:
        pytest.skip("native render stack present; failure path not applicable")
    inbox, _, text_file = _make_input(tmp_path)
    runs = tmp_path / "runs"
    meta = _run(runs, text_file)
    run_dir = runs / meta["run_id"]
    rows = read_manifest(run_dir / "manifest.jsonl")
    assert meta["counts"]["errors"] == 1
    assert rows[0]["status"] == "error" and rows[0]["error"]
