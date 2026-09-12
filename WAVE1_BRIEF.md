# Wave-1 Discovery Brief — Dataset Quality Gate + Dedup Engine (feature/dataset-quality-dedup)

## Mission (lead)
Build the Training Dataset Quality Gate + Deduplication Engine for Clouda OCR on
feature branch `feature/dataset-quality-dedup`, worktree
`F:\PROJECT\CLOUDA_QUALITY_WT`, based on origin/main @ 12e9200.
Never touch main, never touch other agents' worktrees
(CLOUDA_LAB_WT, CLOUDA_RESULTS_STORE_WT, CLOUDA_TRAINING_DATA_WT, CLOUDA_DOCTOR_WT,
.worktrees/training-foundation-final).

## Verified environment facts (do not re-verify)
- Windows, git-bash shell; run git/python from the worktree dir.
- Python 3.11.16 shared venv; pytest 9.1.1, ruff 0.16.7, black 26.5.1, mypy 2.3.1 installed.
- Worktree is clean; benchmark manifest sha256 verified == 2a499ed0268a5c583a1b174ef241e05affb9c4ba757593eb9ee3fc6f3e4fb893.
- Repo pyproject: name clouda-pdf 0.2.0, requires-python >=3.11,<3.12, deps: Pillow, pypdf, pypdfium2, python-docx, requests.
  Extras: data(PyYAML/defusedxml/jsonschema), factory(numpy/opencv-headless/img2pdf/pikepdf), training, test, dev, security.
  Scripts: clouda-data=clouda_data.pipeline.cli:main, clouda-lab=clouda_lab.cli:main, clouda-training=clouda_training.cli:main.
- Top packages ON MAIN: clouda_data (incl. pretraining/ 14 modules: config,dedupe,discovery,export,handoff,hashing,manifest,normalize,schema,sources,splitting,validation,workflow), clouda_data.results, clouda_lab (19 modules incl. dataset_selection, holdout_guard, results_service, training_orchestrator), clouda_contracts (incl. page_identity, dataset_identity, protection, identity, checksums, storage, storage_uri, security, archive_security), clouda_training, clouda_models, pdfword.
- clouda_data/training_data (Training Data Loader) is NOT on main — only on integration branch. Do not import it; design an adapter boundary + deferred E2E doc.
- Tests on main live in tests/ (incl. tests/lab/, tests/results/, tests/data_foundation/, tests/contracts/, tests/training/, tests/factory/, tests/pretraining_fixture.py, tests/test_pretraining_dedupe_split.py ...).

## Existing dedup/splitting functionality already on main (IMPORTANT — do not duplicate)
clouda_data.pretraining has: dedupe.py (classify_duplicates: file-hash, sample-id,
source-record duplicates; DuplicateState CANONICAL/DUPLICATE/CONFLICTING_DUPLICATE/UNIQUE),
splitting.py (assign_splits: train/validation/test/holdout ratios, group by document_id,
leakage_checks for file-hash and normalized-text-hash shared across splits),
validation.py (validate_sample: missing_image, path_escape, empty/long text,
malformed_metadata; ValidationStatus OK/ERROR; exclusion_reason),
normalize.py + hashing.py (sha256_text), manifest.py (read/write JSONL with
_schema_version clouda.pretraining.manifest.v1 header, atomic tmp writes),
export.py (ExportConfig, select_exportable excludes DUPLICATE+holdout+ERROR),
schema.py (DatasetSample dataclass with evolve(), to_dict(), sort_key).
Your job: map its exact current API and decide what the Quality Gate reuses vs extends.

## Your output format (all Wave-1 agents)
Return a structured markdown report:
1. FINDINGS (verified, with file paths + key symbol names)
2. RECOMMENDATIONS (numbered, concrete)
3. RISKS / UNKNOWN
Do not implement anything. Do not modify files. Read-only inspection.
