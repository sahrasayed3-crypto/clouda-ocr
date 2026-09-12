"""Adversarial: duplicate ``sample_id`` across two sources in one manifest.

Gap: no existing test feeds two rows with the SAME sample_id but different
``source_id`` through the gate. The dedupe engine treats them as
``duplicate_sample_id``; this proves detection AND that downstream cluster
models stay well-formed.
"""

from __future__ import annotations

from conftest import make_manifest, make_row
from clouda_data.quality.gate import run_quality_gate


def test_cross_source_duplicate_sample_id_detected_and_wellformed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest = make_manifest(
        tmp_path / "xsrc",
        [
            make_row("smp_dup", source_id="src1", text="نص أول"),
            make_row("smp_dup", source_id="src2", text="نص ثان"),
        ],
    )
    scan = run_quality_gate(str(manifest))

    states = {s.duplicate_state.value for s in scan.samples}
    assert "duplicate" in states, "cross-source same sample_id not classified"
    dupes = [s for s in scan.samples if s.duplicate_state.value == "duplicate"]
    assert dupes[0].duplicate_of == "smp_dup"

    # Cluster integrity: members must be distinct samples. The cluster id is
    # a hash of sorted member ids, so duplicated ids silently collapse two
    # different rows into one member entry.
    clusters = scan.result.clusters
    assert clusters, "expected a duplicate cluster for cross-source dup ids"
    for cluster in clusters:
        members = list(cluster.member_ids)
        assert len(set(members)) == len(
            members
        ), f"cluster {cluster.cluster_id} has duplicate member ids: {members}"

    # Issue canonical keys must not collide between distinct rows.
    keys = [issue.canonical_key for issue in scan.result.issues]
    assert len(set(keys)) == len(
        keys
    ), f"duplicate issue canonical_key entries: {sorted(keys)}"
