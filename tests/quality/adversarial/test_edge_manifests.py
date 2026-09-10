"""Adversarial: degenerate and unicode manifests through the full gate.

Gaps: no existing test runs the gate on (a) a header-only manifest
(``_row_count: 0``) or (b) sample_ids containing Arabic/emoji characters
end-to-end (gate -> derived clean manifest -> re-validation).
"""

from __future__ import annotations

from conftest import make_manifest, make_row
from clouda_data.quality.derived import revalidate_derived, write_clean_manifest
from clouda_data.quality.gate import run_quality_gate


def test_header_only_empty_manifest_gate_passes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest = make_manifest(tmp_path / "empty", [])
    header = manifest.read_text(encoding="utf-8").splitlines()[0]
    assert '"_row_count": 0' in header

    scan = run_quality_gate(str(manifest))
    assert scan.samples == []
    assert list(scan.exclusions) == []
    assert list(scan.quarantine_ids) == []
    assert scan.result.verdict.value == "PASS"
    assert scan.result.issues == ()


def test_unicode_sample_ids_end_to_end(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest = make_manifest(
        tmp_path / "uni",
        [
            make_row("smp_نص_🎭", source_id="src1", text="نص عربي"),
            make_row("smp_a", source_id="src1", text="نص آخر"),
        ],
    )
    scan = run_quality_gate(str(manifest))
    ids = {s.sample_id for s in scan.samples}
    assert "smp_نص_🎭" in ids, "unicode sample_id lost during gate load"

    # Derived write + fail-closed re-validation must survive unicode ids.
    result = write_clean_manifest(
        manifest,
        scan.samples,
        list(scan.exclusions),
        list(scan.quarantine_ids),
        scan.run,
        tmp_path / "uni" / "clean.jsonl",
    )
    assert result["clean_row_count"] >= 1
    revalidate_derived(result["clean_manifest_path"], set(scan.excluded_ids))
