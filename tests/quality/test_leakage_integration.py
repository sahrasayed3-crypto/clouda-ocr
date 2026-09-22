"""Regression tests for the text near-duplicate leakage tier, the FAIL
verdict clean-manifest guard, and the dataset download digest trust model.

Covers the deep-session gap fixes documented in
docs/engineering/RELEASE_READINESS_V0.2.1.md:

A. The gate actually executes the MinHash/LSH text near-duplicate tier
   (previously dead code) and fails on cross-split near-text leakage,
   including Arabic diacritic/tatweel/whitespace variants, while passing
   genuinely distinct texts.
B. A FAIL-verdict gate cannot produce an artifact that looks like an
   approved clean manifest.
C. Sample downloads without a registry-pinned digest fail closed unless
   explicitly opted in; digest trust level is recorded per file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from clouda_data.quality import cli as quality_cli
from clouda_data.quality.derived import (
    FailedVerdictRefusedError,
    write_clean_manifest,
)
from clouda_data.quality.gate import run_quality_gate
from clouda_data.pretraining.hashing import sha256_file
from tests.quality.conftest import (
    make_manifest,
    make_row,
    render_arabic_page,
    save_png,
)

ARABIC_BASE = (
    "كان المخطوط القديم يحتوي على فصول متعددة من التاريخ الإسلامي "
    "وذكر الوقائع التي جرت في الأندلس خلال عصورها المختلفة"
)  # >= 40 normalized chars


def _page(root: Path, name: str, seed: int, text: str) -> dict:
    """A valid artifact-backed row image spec (deterministic Arabic page)."""

    image_path = save_png(
        render_arabic_page(seed, "gate-leakage", text, size=(640, 480)),
        root / "images" / f"{name}.png",
    )
    with Image.open(image_path) as decoded:
        width, height = decoded.size
    return {
        "image_path": "images/" + image_path.name,
        "width": width,
        "height": height,
        "file_sha256": sha256_file(image_path),
    }


def _clean_rows(root: Path, texts: list[str], splits: list[str]) -> list:
    rows = []
    for index, (text, split) in enumerate(zip(texts, splits)):
        page = _page(root, f"p{index}", index, text)
        rows.append(
            make_row(
                f"smp_{index:02d}",
                source_id="src1",
                document_id=f"doc-{index:02d}",
                page_id=f"page-{index:02d}",
                text=text,
                target_split=split,
                **page,
            )
        )
    return rows


def _finding_codes(scan) -> list[str]:
    return sorted({issue.code for issue in scan.result.issues})


class TestTextNearDupGate:
    def test_exact_duplicate_text_across_train_eval_fails(self, tmp_path) -> None:
        manifest = make_manifest(
            tmp_path / "m",
            _clean_rows(
                tmp_path / "m", [ARABIC_BASE, ARABIC_BASE], ["train", "validation"]
            ),
        )
        scan = run_quality_gate(str(manifest))
        assert scan.result.verdict.value == "FAIL"
        assert "LEAK_NEAR_TEXT" in _finding_codes(scan)

    def test_diacritic_variant_across_train_eval_fails(self, tmp_path) -> None:
        diacritic = (
            "كَانَ الْمَخْطُوطُ الْقَدِيمُ يَحْتَوِي عَلَى فُصُولٍ مُتَعَدِّدَةٍ مِنَ التَّارِيخِ"
            + ARABIC_BASE[50:]
        )
        manifest = make_manifest(
            tmp_path / "m",
            _clean_rows(tmp_path / "m", [ARABIC_BASE, diacritic], ["train", "test"]),
        )
        scan = run_quality_gate(str(manifest))
        assert scan.result.verdict.value == "FAIL"
        assert "LEAK_NEAR_TEXT" in _finding_codes(scan)

    def test_tatweel_and_whitespace_variant_across_train_eval_fails(
        self, tmp_path
    ) -> None:
        stretched = (
            ARABIC_BASE[:20] + "ـ" * 6 + ARABIC_BASE[20:40] + "  " + ARABIC_BASE[40:]
        )
        manifest = make_manifest(
            tmp_path / "m",
            _clean_rows(
                tmp_path / "m", [ARABIC_BASE, stretched], ["train", "validation"]
            ),
        )
        scan = run_quality_gate(str(manifest))
        assert scan.result.verdict.value == "FAIL"
        assert "LEAK_NEAR_TEXT" in _finding_codes(scan)

    def test_near_text_across_train_holdout_fails(self, tmp_path) -> None:
        variant = ARABIC_BASE[:-3] + "قديمة"
        manifest = make_manifest(
            tmp_path / "m",
            _clean_rows(tmp_path / "m", [ARABIC_BASE, variant], ["train", "holdout"]),
        )
        scan = run_quality_gate(str(manifest))
        assert scan.result.verdict.value == "FAIL"
        codes = _finding_codes(scan)
        assert "LEAK_NEAR_TEXT" in codes

    def test_distinct_texts_do_not_trigger_near_text(self, tmp_path) -> None:
        others = [
            "ذهب المسافر إلى الشرق لدراسة علوم الفلك والرياضيات في مدارس بغداد الشهيرة",
            "تحكي الحكاية عن بحار فقد طريقه في عاصفة قاسية قبالة سواحل المحيط الهندي",
            "جمع الكاتب حكايات الشعب في ديوان كبير حفظ فيه تراث الأجداد وأمثالهم",
            "زار التجار مدينة حلب القديمة لبيع الحرير والتوابل القادمة من الأراضي الفارسية",
        ]
        manifest = make_manifest(
            tmp_path / "m",
            _clean_rows(
                tmp_path / "m",
                others,
                ["train", "train", "validation", "validation"],
            ),
        )
        scan = run_quality_gate(str(manifest))
        # Unrelated artifact-stage warnings (small/thumbnail-capped fixture
        # images) may appear; what must never happen is a near-text finding.
        assert scan.result.verdict.value in {"PASS", "PASS_WITH_WARNINGS"}
        assert "LEAK_NEAR_TEXT" not in _finding_codes(scan)
        assert "DUP_TEXT_NEAR" not in _finding_codes(scan)

    def test_same_partition_near_text_is_warning_not_failure(self, tmp_path) -> None:
        variant = ARABIC_BASE[:-3] + "عتيق"
        manifest = make_manifest(
            tmp_path / "m",
            _clean_rows(tmp_path / "m", [ARABIC_BASE, variant], ["train", "train"]),
        )
        scan = run_quality_gate(str(manifest))
        assert scan.result.verdict.value == "PASS_WITH_WARNINGS"
        assert "DUP_TEXT_NEAR" in _finding_codes(scan)
        assert "LEAK_NEAR_TEXT" not in _finding_codes(scan)

    def test_text_near_stage_excluded_by_no_near_duplicates_flag(
        self, tmp_path
    ) -> None:
        manifest = make_manifest(
            tmp_path / "m",
            _clean_rows(
                tmp_path / "m", [ARABIC_BASE, ARABIC_BASE], ["train", "validation"]
            ),
        )
        scan = run_quality_gate(str(manifest), no_near_duplicates=True)
        # The L5 exact-hash warning family still exists, but the near-text
        # tier must not run when the operator disabled near-duplicate work.
        assert "LEAK_NEAR_TEXT" not in _finding_codes(scan)

    def test_timings_report_text_near_stage(self, tmp_path) -> None:
        manifest = make_manifest(
            tmp_path / "m",
            _clean_rows(tmp_path / "m", [ARABIC_BASE], ["train"]),
        )
        scan = run_quality_gate(str(manifest))
        assert "text_near_s" in scan.timings.to_dict()


class TestFailVerdictCleanManifestGuard:
    def _failing_scan(self, tmp_path: Path):
        # A row without an image artifact fails the gate (MISSING_IMAGE).
        manifest = make_manifest(
            tmp_path / "m", [make_row("smp_bad", text="نص عربي قصير")]
        )
        return manifest, run_quality_gate(str(manifest))

    def test_write_clean_manifest_refuses_fail_verdict(self, tmp_path) -> None:
        manifest, scan = self._failing_scan(tmp_path)
        assert scan.result.verdict.value == "FAIL"
        with pytest.raises(FailedVerdictRefusedError):
            write_clean_manifest(
                manifest,
                scan.samples,
                list(scan.exclusions),
                list(scan.quarantine_ids),
                scan.run,
                tmp_path / "out" / "clean.jsonl",
            )
        assert not (tmp_path / "out" / "clean.jsonl").exists()

    def test_forensic_opt_in_writes_rejected_manifest(self, tmp_path) -> None:
        manifest, scan = self._failing_scan(tmp_path)
        result = write_clean_manifest(
            manifest,
            scan.samples,
            list(scan.exclusions),
            list(scan.quarantine_ids),
            scan.run,
            tmp_path / "out" / "rejected.jsonl",
            allow_failed_verdict=True,
        )
        assert result["verdict"] == "FAIL"

    def test_cli_clean_manifest_fail_writes_nothing_returns_1(
        self, tmp_path, capsys
    ) -> None:
        manifest, _scan = self._failing_scan(tmp_path)
        exit_code = quality_cli.main(
            [
                "clean-manifest",
                str(manifest),
                "--output",
                str(tmp_path / "clean.jsonl"),
            ]
        )
        assert exit_code == 1
        assert not (tmp_path / "clean.jsonl").exists()
        payload = json.loads(capsys.readouterr().out)
        assert payload["clean_manifest"] == "not_written"
        assert payload["verdict"] == "FAIL"


class TestDownloadDigestTrustModel:
    def _registry(self, root: Path, payload: bytes, sha256: str | None) -> Path:
        import tempfile

        from tests.data_foundation.unit.test_dataset_downloader import (
            LocalServer,
            create_registry,
        )

        server_root = Path(tempfile.mkdtemp(dir=root))
        (server_root / "sample.txt").write_bytes(payload)
        server = LocalServer(server_root)
        server.__enter__()
        self._server = server  # keep alive for the duration of the test
        registry = root / "data/manifests/dataset_registry.json"
        asset = {
            "filename": "sample.txt",
            "url": f"{server.url}/sample.txt",
            "size_bytes": len(payload),
        }
        if sha256 is not None:
            asset["sha256"] = sha256
        registry.parent.mkdir(parents=True, exist_ok=True)
        create_registry(registry, f"{server.url}/sample.txt")
        # Re-write the registry with our explicit asset (create_registry
        # writes its own minimal shape).
        doc = json.loads(registry.read_text(encoding="utf-8"))
        for source in doc["sources"]:
            source["sample_assets"] = [asset]
        registry.write_text(json.dumps(doc), encoding="utf-8")
        return registry

    def _download(self, root: Path, monkeypatch, *, allow_unpinned: bool):
        monkeypatch.setenv("CLOUDA_ALLOW_PRIVATE_DOWNLOADS", "true")
        monkeypatch.setenv("CLOUDA_ALLOW_INSECURE_DOWNLOADS", "true")
        if allow_unpinned:
            monkeypatch.setenv("CLOUDA_ALLOW_UNPINNED_DATASET_DOWNLOADS", "true")
        from clouda_data.datasets.downloader import download_dataset_sample

        return download_dataset_sample(
            "tiny_source", project_root=root, max_bytes=1024 * 1024
        )

    def test_unpinned_asset_fails_closed_by_default(
        self, tmp_path, monkeypatch
    ) -> None:
        from clouda_data.datasets.downloader import (
            UNPINNED_DOWNLOAD_ENV,
            create_ingestion_manifest_for_download,  # noqa: F401
        )

        del UNPINNED_DOWNLOAD_ENV
        payload = "tiny Arabic OCR metadata".encode("utf-8")
        import hashlib

        registry = self._registry(
            tmp_path, payload, hashlib.sha256(payload).hexdigest()
        )
        # Make the asset unpinned again for this test.
        doc = json.loads(registry.read_text(encoding="utf-8"))
        for source in doc["sources"]:
            source["sample_assets"][0].pop("sha256")
        registry.write_text(json.dumps(doc), encoding="utf-8")

        monkeypatch.delenv("CLOUDA_ALLOW_UNPINNED_DATASET_DOWNLOADS", raising=False)
        result = self._download(tmp_path, monkeypatch, allow_unpinned=False)

        assert not result.ok
        assert any("unpinned_digest" in issue for issue in result.issues)
        assert result.files == []

    def test_unpinned_asset_opt_in_records_self_reported(
        self, tmp_path, monkeypatch
    ) -> None:
        payload = "tiny Arabic OCR metadata".encode("utf-8")
        import hashlib

        registry = self._registry(
            tmp_path, payload, hashlib.sha256(payload).hexdigest()
        )
        doc = json.loads(registry.read_text(encoding="utf-8"))
        for source in doc["sources"]:
            source["sample_assets"][0].pop("sha256")
        registry.write_text(json.dumps(doc), encoding="utf-8")

        result = self._download(tmp_path, monkeypatch, allow_unpinned=True)

        assert result.ok, result.issues
        assert result.files[0].digest_source == "self_reported"

    def test_pinned_digest_records_registry_pinned(self, tmp_path, monkeypatch) -> None:
        payload = "tiny Arabic OCR metadata".encode("utf-8")
        import hashlib

        registry = self._registry(
            tmp_path, payload, hashlib.sha256(payload).hexdigest()
        )
        del registry

        result = self._download(tmp_path, monkeypatch, allow_unpinned=False)

        assert result.ok, result.issues
        assert result.files[0].digest_source == "registry_pinned"

    def test_pinned_digest_mismatch_is_rejected(self, tmp_path, monkeypatch) -> None:
        payload = "tiny Arabic OCR metadata".encode("utf-8")
        wrong = "0" * 64
        self._registry(tmp_path, payload, wrong)

        result = self._download(tmp_path, monkeypatch, allow_unpinned=False)

        assert not result.ok
        assert any("checksum mismatch" in issue.lower() for issue in result.issues)
        assert result.files == []
