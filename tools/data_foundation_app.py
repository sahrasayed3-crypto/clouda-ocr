from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import streamlit as st

from clouda_contracts.storage import StorageRoots
from clouda_data.datasets.license_gate import load_catalog
from clouda_data.distortion.workflow import (
    generate_preview,
    read_jsonl,
    run_distortion_batch,
    validate_distortion_manifest,
)
from clouda_data.distortion.checkpoints import RunCheckpointStore
from clouda_data.locations import default_profile_dir
from clouda_data.pipeline.profiles import list_profile_paths, load_profile
from clouda_training.exporter import export_training_data

st.set_page_config(page_title="Clouda Data Foundation", page_icon="🧪", layout="wide")
st.title("Clouda Data Foundation — local administration")
st.warning(
    "Local-only administration surface. Bind Streamlit to 127.0.0.1 and do not "
    "publish this page without production authentication."
)

roots = StorageRoots.from_env()
catalog = load_catalog()
datasets = catalog["datasets"]
counts = Counter(str(item["status"]) for item in datasets)

col1, col2, col3 = st.columns(3)
col1.metric("Catalog records", len(datasets))
col2.metric("Approved with conditions", counts["approved_with_conditions"])
col3.metric("Blocked or pending", counts["blocked"] + counts["pending"])

with st.expander("Dataset licenses", expanded=True):
    st.dataframe(
        [
            {
                "dataset": item["dataset_id"],
                "status": item["status"],
                "commercial training": item["commercial_training_allowed"],
                "evaluation": item["evaluation_allowed"],
            }
            for item in datasets
        ],
        use_container_width=True,
        hide_index=True,
    )

profiles = {
    load_profile(path)["name"]: path
    for path in list_profile_paths(default_profile_dir())
}
profile_id = st.selectbox("Distortion profile", sorted(profiles))
profile = load_profile(profiles[profile_id])
st.json(
    {
        "severity": profile.get("severity"),
        "transformations": profile.get("distortions", []),
        "quality_class": profile.get("quality_class_label"),
    }
)

manifest_value = st.text_input(
    "Rendered input manifest",
    value=str(roots.dataset_root / "rendered" / "MANIFEST.jsonl"),
)
seed = st.number_input("Deterministic seed", min_value=0, value=20260723)
pages = st.number_input("Maximum pages", min_value=1, max_value=1000, value=10)
variants = st.number_input("Variants per page", min_value=1, max_value=5, value=1)
large_run = int(pages) > 100
if large_run:
    st.error("This exceeds the safe default of 100 pages.")
confirm = st.checkbox(
    "I confirm this local run"
    + (" and explicitly authorize a large run" if large_run else "")
)

if st.button("Launch limited run", disabled=not confirm):
    manifest = Path(manifest_value).expanduser().resolve()
    progress = st.progress(0, text="Validating configuration")
    try:
        progress.progress(10, text="Starting deterministic distortion run")
        path = run_distortion_batch(
            manifest,
            profiles[profile_id],
            seed=int(seed),
            variants=int(variants),
            max_pages=int(pages),
            allow_large_run=large_run and confirm,
        )
        progress.progress(100, text="Run complete")
        st.success(f"Run completed: {path}")
        st.session_state["distortion_manifest"] = str(path)
    except Exception as exc:
        progress.empty()
        st.error(f"{type(exc).__name__}: {exc}")

distortion_manifest = st.text_input(
    "Distortion manifest",
    value=st.session_state.get("distortion_manifest", ""),
)
if st.button("Generate visual preview", disabled=not distortion_manifest):
    try:
        preview = generate_preview(distortion_manifest, limit=min(int(pages), 20))
        st.success(str(preview))
        st.components.v1.html(
            preview.read_text(encoding="utf-8"), height=700, scrolling=True
        )
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")

if distortion_manifest and Path(distortion_manifest).is_file():
    records = read_jsonl(distortion_manifest)
    statuses = Counter(str(item.get("status")) for item in records)
    st.subheader("Run status")
    st.json({"records": len(records), "statuses": statuses})
    checkpoint_path = Path(distortion_manifest).parent / "checkpoints.sqlite3"
    if records and checkpoint_path.is_file():
        checkpoint = RunCheckpointStore(checkpoint_path)
        st.json({"checkpoint": checkpoint.summary(str(records[0]["run_id"]))})
    if st.button("Validate and report"):
        try:
            validation = validate_distortion_manifest(
                distortion_manifest,
                quarantine=False,
            )
            if validation["passed"]:
                st.success("Validation passed")
            else:
                st.error(f"{len(validation['failures'])} validation failures")
            st.json(validation)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
    if st.button("Export evaluation-only training manifest"):
        output = (
            roots.artifact_root
            / "training"
            / f"{Path(distortion_manifest).parent.name}.jsonl"
        )
        try:
            report = export_training_data(
                distortion_manifest,
                output,
                purpose="evaluation",
            )
            st.success(str(output))
            st.json(report)
            preview_records = read_jsonl(output)[:10]
            st.subheader("Training export preview (first 10 records)")
            st.dataframe(preview_records, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

usage = {
    "datasets_bytes": sum(
        path.stat().st_size for path in roots.dataset_root.rglob("*") if path.is_file()
    ),
    "artifacts_bytes": sum(
        path.stat().st_size for path in roots.artifact_root.rglob("*") if path.is_file()
    ),
    "cache_bytes": sum(
        path.stat().st_size for path in roots.cache_root.rglob("*") if path.is_file()
    ),
}
st.subheader("Storage usage")
st.json(usage)
st.caption(
    json.dumps(
        {
            "state_root": os.getenv("CLOUDA_STATE_HOME", "configured default"),
            "large_runs_enabled": False,
            "public_binding_supported": False,
        }
    )
)
