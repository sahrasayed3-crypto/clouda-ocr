"""End-to-end factory: image input, all three engines, exports,
determinism across worker counts, resume safety, error isolation.

Migrated from the standalone repository (tests/test_factory.py); imports
retargeted to the integrated package.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from clouda_data.factory.factory import _expand_inputs, generate_run
from clouda_data.factory.manifest import read_manifest, write_manifest_jsonl

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
        assert not Path(row["gt_path"]).is_absolute()
        assert "\\" not in row["gt_path"]
        assert row["transform_steps"]
        assert row["render_config"]["backend"] == "raqm"
        assert row["distortion_config"]["name"] in PROFILES
        assert row["config_hash"] == meta["config_hash"]
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


def test_resume_refuses_changed_configuration(tmp_path):
    _, image, _ = _make_input(tmp_path)
    runs = tmp_path / "runs"
    meta = _run(runs, image)
    with pytest.raises(ValueError, match="configuration"):
        generate_run(
            inputs=[image],
            runs_root=runs,
            profile_names=PROFILES,
            variants=3,
            base_seed=54321,
            seed_mode="v1",
            backend="raqm",
            workers=1,
            export_pdf=True,
            export_png=True,
            max_pages=4,
            resume=True,
            run_id=meta["run_id"],
        )


def test_resume_repairs_missing_output(tmp_path):
    _, image, _ = _make_input(tmp_path)
    runs = tmp_path / "runs"
    meta = _run(runs, image)
    run_dir = runs / meta["run_id"]
    row = next(
        r for r in read_manifest(run_dir / "manifest.jsonl") if r["status"] == "ok"
    )
    output = run_dir / row["output_path"]
    expected_hash = row["output_sha256"]
    output.unlink()

    _run(runs, image, run_id=meta["run_id"], resume=True)

    repaired = next(
        r
        for r in read_manifest(run_dir / "manifest.jsonl")
        if r["variant_id"] == row["variant_id"] and r["page_index"] == row["page_index"]
    )
    assert output.is_file()
    assert repaired["status"] == "ok"
    assert repaired["output_sha256"] == expected_hash


def test_resume_repairs_missing_secondary_pdf(tmp_path):
    _, image, _ = _make_input(tmp_path)
    runs = tmp_path / "runs"
    meta = _run(runs, image)
    run_dir = runs / meta["run_id"]
    row = next(
        r for r in read_manifest(run_dir / "manifest.jsonl") if r["status"] == "ok"
    )
    pdf = run_dir / row["pdf_path"]
    expected_hash = row["pdf_sha256"]
    pdf.unlink()

    _run(runs, image, run_id=meta["run_id"], resume=True)

    repaired = next(
        r
        for r in read_manifest(run_dir / "manifest.jsonl")
        if r["variant_id"] == row["variant_id"] and r["page_index"] == row["page_index"]
    )
    assert pdf.is_file()
    assert repaired["pdf_sha256"] == expected_hash


def test_resume_replaces_prior_error_row(tmp_path):
    _, image, _ = _make_input(tmp_path)
    runs = tmp_path / "runs"
    meta = _run(runs, image)
    run_dir = runs / meta["run_id"]
    rows = read_manifest(run_dir / "manifest.jsonl")
    rows[0] = {**rows[0], "status": "error", "error": "interrupted"}
    write_manifest_jsonl(rows, run_dir / "manifest.jsonl")

    resumed = _run(runs, image, run_id=meta["run_id"], resume=True)

    repaired = read_manifest(run_dir / "manifest.jsonl")
    assert resumed["counts"]["errors"] == 0
    assert len(repaired) == 3
    assert all(row["status"] == "ok" for row in repaired)


def test_resume_removes_recovered_document_level_error(tmp_path, monkeypatch):
    import clouda_data.factory.factory as factory_module

    _, image, _ = _make_input(tmp_path)
    runs = tmp_path / "runs"
    original_render = factory_module._render_clean

    def fail_render(*args, **kwargs):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(factory_module, "_render_clean", fail_render)
    failed = _run(runs, image)
    monkeypatch.setattr(factory_module, "_render_clean", original_render)

    resumed = _run(runs, image, run_id=failed["run_id"], resume=True)
    rows = read_manifest(runs / failed["run_id"] / "manifest.jsonl")
    assert resumed["counts"]["errors"] == 0
    assert rows and all(row["status"] == "ok" for row in rows)


def test_expand_inputs_deduplicates_resolved_files(tmp_path):
    _, image, _ = _make_input(tmp_path)
    assert _expand_inputs([image, image.parent / "." / image.name]) == [image]


@pytest.mark.skipif(
    os.path.normcase("A.png") == os.path.normcase("a.png"),
    reason="filesystem path normalization is case-insensitive",
)
def test_expand_inputs_preserves_distinct_case_sensitive_files(tmp_path):
    upper = tmp_path / "A.png"
    lower = tmp_path / "a.png"
    upper.write_bytes(b"upper")
    lower.write_bytes(b"lower")
    assert _expand_inputs([upper, lower]) == [upper, lower]


def test_resume_preserves_relative_input_identity(tmp_path, monkeypatch):
    _, image, _ = _make_input(tmp_path)
    monkeypatch.chdir(tmp_path)
    relative_image = image.relative_to(tmp_path)
    runs = Path("runs")
    first = _run(runs, relative_image)
    resumed = _run(runs, relative_image, run_id=first["run_id"], resume=True)
    rows = read_manifest(runs / first["run_id"] / "manifest.jsonl")
    assert resumed["counts"]["errors"] == 0
    assert {row["source_ref"] for row in rows} == {str(relative_image)}


def test_resume_refuses_changed_source_content(tmp_path):
    _, image, _ = _make_input(tmp_path)
    runs = tmp_path / "runs"
    first = _run(runs, image)
    Image.new("RGB", (96, 64), "black").save(image)
    with pytest.raises(ValueError, match="source content"):
        _run(runs, image, run_id=first["run_id"], resume=True)


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


def test_raqm_render_receives_requested_page_limit(tmp_path, monkeypatch):
    from clouda_data.factory.factory import RunConfig, _render_clean
    from clouda_data.factory.ingest.text_file import load_text_source
    from clouda_data.factory.render.base import RenderResult
    import clouda_data.factory.render as render_module

    source_path = tmp_path / "source.txt"
    source_path.write_text("نص عربي", encoding="utf-8")
    seen = {}

    class FakeBackend:
        def render(self, text, image_path, out_dir, style_seed, dpi=150):
            out_dir.mkdir(parents=True, exist_ok=True)
            page = out_dir / "page.png"
            Image.new("RGB", (16, 16), "white").save(page)
            return RenderResult(None, [page], False, "raqm", {})

    def fake_get_backend(name, **kwargs):
        seen.update(kwargs)
        return FakeBackend()

    monkeypatch.setattr(render_module, "get_backend", fake_get_backend)
    rc = RunConfig(
        run_id="r",
        runs_root=tmp_path,
        run_dir=tmp_path / "run",
        profiles=["old_book_medium"],
        variants=1,
        base_seed=1,
        seed_mode="v1",
        backend="raqm",
        workers=1,
        export_pdf=False,
        export_png=True,
        max_pages=2,
        resume=False,
        config_hash="x",
    )

    _render_clean(rc, rc.run_dir / "doc", load_text_source(source_path), 1)

    assert seen == {"max_pages": 2}
