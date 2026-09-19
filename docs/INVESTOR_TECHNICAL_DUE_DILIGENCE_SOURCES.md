# Clouda OCR — Due-Diligence Source Map

Companion to [`INVESTOR_TECHNICAL_DUE_DILIGENCE.md`](INVESTOR_TECHNICAL_DUE_DILIGENCE.md). Maps every major claim in that document to file paths, commits/SHAs where applicable, tests, CI evidence, and upstream authoritative sources.

- Audited `origin/main` SHA: `45d0078e551267535f85c79badd432fdc8514bef` (2026-09-19)
- Pack written on: branch `docs/investor-github-hardening` @ `a4c4345` (docs-only delta vs `main`)
- Audit date: 2026-09-20
- Line numbers refer to the audited tree and may drift after future merges.

Legend: **T** = test evidence · **CI** = enforced in `.github/workflows/ci.yml` · **L** = executed live during this audit (2026-09-20) · **GH** = GitHub metadata/API · **UP** = upstream authoritative source.

---

## §1–§2 Identity, main SHA, CI status

| Claim | Primary evidence | Supporting |
|---|---|---|
| Main SHA `45d0078…` (2026-09-19, PR #1 merge) | `git log origin/main -1` **L** | GitHub Actions run `45d0078e` created 2026-09-19T20:57:37Z **GH** |
| CI green on main | GitHub API `actions/runs?per_page=5` **GH L** | Pre-merge failed/cancelled runs on `feature/fix-github-ci-failures` (honest repair trail, PR #1) **GH** |
| Docs-only branch delta | `git diff main..HEAD --stat` (README.md, docs/TESTING.md, docs/investor-github-audit.md, one plan doc) **L** | `docs/investor-github-audit.md` |
| Self-audit exists but is internal only | `docs/investor-github-audit.md` (self-conducted, 2026-09-20) | Absence of any external attestation in SECURITY.md |

## §3 Evidence index — per-row sources

| Topic | Claim evidence (file:line where load-bearing) | Tests |
|---|---|---|
| Page pipeline & states | `docs/ARCHITECTURE.md:5-13,44-51,55-67`; root `ARCHITECTURE.md:25-38` | `tests/test_page_routing.py`, `tests/test_readiness_pipeline.py` **T** |
| Routing evidence-only warnings | `pdfword/page_routing.py:40-45` (`AnalysisWarningCode`), `:474-622` | `tests/test_page_routing.py::test_unavailable_page_evidence_is_explicit_not_fabricated` (L241) **T** |
| Trusted gate 12 checks | `pdfword/page_routing.py:639-862` (rejection checks L713-799; verdict logic L833-856) | gate tests `tests/test_page_routing.py:267-411` **T** |
| Digest binding post-extraction | `pdfword/page_routing.py:953-972`; `pdfword/ocr_pipeline.py:362-386` | `tests/test_page_routing.py::test_post_extraction_digest_mismatch_is_rejected` (L506) **T** |
| Self-review categorical, source-layer discarded | `pdfword/ocr_self_review.py` (547 lines; `del analysis` L239; confidence gate L222-230) | `tests/test_ocr_self_review.py` (14 tests) **T** |
| Selective re-read budget + render identity | `pdfword/ocr_self_review.py:57-64` (ReReadBudget), L151-161, L403-416 | `test_stale_region_is_rejected_without_creating_a_crop` (L195), `test_crop_rejects_same_size_image_with_different_identity` (L235), `test_reread_cannot_replace_clean_text_or_leave_unresolved_corruption` (L349) **T** |
| RTL DOCX | `pdfword/docx_export.py` (bidi/rtl L24-36; footer L43-47,80; tables dropped L140-148) | `tests/test_arabic_fixtures.py` (5 tests) **T** |
| Engine boundary + pinned revision | `pdfword/engines.py:219-331` (esp. L247-253 rejecting `unresolved`/`latest`/`main`) | `tests/test_model_agnostic_engines.py`, `tests/test_engine_boundaries.py` **T** |
| Worker API security + finalization | `pdfword/worker_api.py` (`_authenticate` L426, headers L88, finalization L2065, quality gate L261) | `tests/test_distributed_worker.py` (37 tests; race set L492-1194) **T**; CI `redis-integration` job **CI** |
| Storage roots | `clouda_contracts/storage.py` (`from_env` L88, `validate` L143, `resolve_uri` L186), `clouda_contracts/storage_uri.py` | `tests/contracts/test_storage_roots.py` **T**; property test `tests/security/test_adversarial_hardening.py:411` **T** |
| No external certification | verified by absence: grep across `SECURITY.md`, `docs/security/*`, README | — |
| Static test counts (202 files / 1,895 functions) | `find tests -name "*.py" | wc -l` → 202; `grep -r "def test_" tests --include="*.py" | wc -l` → 1,895 **L** | README quotes 183/~1,900 as static count with rerun instruction |
| Packaging | `pyproject.toml` (name `clouda-pdf`, version `0.2.0`, `requires-python >=3.11,<3.12`, extras L24-85); `constraints/py311.txt` | CI install step + `python -m build` step **CI** |
| Data factory determinism | `clouda_data/factory/seed/derive.py` (BLAKE2b, 63-bit); `clouda_data/factory/provenance/hashing.py`; `clouda_data/factory/factory.py` | `tests/factory/test_seeds.py`, `test_manifest_integrity.py`, `test_e2e.py::test_tiny_e2e_deterministic_across_repeat` **T** |
| 108 distortion operators | `clouda_data/distortion/registry.py` (`DEFAULT_DISTORTIONS`); `clouda_data/distortion/operators.py` | `tests/factory/test_distort.py` **T** |
| Quality gate / leakage | `clouda_data/quality/gate.py`, `leakage.py` (L0–L6), `exact_dup.py`, `image_fp.py`, `text_dup.py`, `near_index.py` | `tests/quality/` (231 tests incl. `test_clustering_determinism.py`) **T** |
| License gate | `clouda_training/exporter.py:80-129`; `clouda_data/datasets/license_gate.py`; `dataset_catalog/registry/datasets_v1.json` (fail-closed flags) | `tests/security/test_adversarial_hardening.py::test_manifest_cannot_forge_commercial_training_license` (L91) **T** |
| Training mock-only / fail-closed | `clouda_training/experiments/trainer.py` (MockTrainer); `runtime/torch_backend.py`; `runtime/adapter.py` (SyntheticLinearAdapter); `docs/training/EXPERIMENT_FRAMEWORK.md` ("fails closed for real training") | `tests/training/`, `tests/runtime/test_torch_e2e.py`, `test_resume_determinism.py` **T** |
| No final model / disabled placeholder | `README.md` ("No final trained OCR model exists"); `configs/models/registry.v1.json` (single `future-vlm-adapter`, `UNRESOLVED`, `checkpoint_uri: null`); `clouda_models/registry.py::validate_no_enabled_placeholder` | `tests/models/test_registry.py` **T** |
| Benchmark framework | `benchmarks/ocr_arabic/` (README, methodology.md, release.json, results.csv, source_manifest.csv [100 rows], source_summary.csv [14 datasets], models.csv, CITATIONS.md) | `tests/benchmarks/test_ocr_arabic_release.py` **T**; validator **L** |
| Adapters mock-verified | `clouda_training/hunyuan/` (adapter/exporter/validators/packing), `clouda_training/qwen/`; `docs/adapters/MULTIMODEL_ADAPTERS.md` (`real_weights_validated=False`, `gpu_validated=False`) | `tests/hunyuan/` (53), `tests/multimodel/` (55) **T** |
| Semantic layer absent | repo-wide grep: no summaries/QA/table-extraction code; `pdfword/intelligence.py` heuristic ratios only; not in `ROADMAP.md` | — |
| Deployment | `README.md` (install/demo), `deploy/linux/` (systemd units), `SECURITY.md` (staging posture) | demo smoke + doctor **CI L** |

## §4 Architecture specifics

- Two consistent architecture docs: root `ARCHITECTURE.md` (43 lines) and `docs/ARCHITECTURE.md` (67 lines); boundary docs `docs/architecture/SYSTEM_OVERVIEW.md`, `DATA_BOUNDARIES.md` ("Runtime workers cannot access dataset or model roots"), `RUNTIME_VS_TRAINING.md` ("The Streamlit and FastAPI entrypoints do not import `clouda_training`").
- Page-state contract: `docs/ARCHITECTURE.md:44-51`; near-blank rule (footer page number ≤12 chars at y ≤0.15, or small image density ≤0.10): `pdfword/page_routing.py:520-546`.
- Route decision map: `pdfword/page_routing.py:876-950` (UNTRUSTED + no OCR → `PENDING_OCR_MODEL` with `review_required = not ocr_available`).
- Demo route verification: `python scripts/demo.py` produced `direct_pdf_text`, `pending_ocr_model`, `blank_page` on the audit machine **L**; script reads `tests/fixtures/digital_text.pdf`, `scanned.pdf`, `blank.pdf`.

## §5 Engineering quality specifics

- CI workflow: `.github/workflows/ci.yml` — matrix `os: [windows-latest, ubuntu-latest]`, `python: ["3.11"]`; `permissions: contents: read`; pinned `actions/checkout@11d5960…`, `actions/setup-python@a26af69…`; `persist-credentials: false` on all four checkouts; pytest step env `CLOUDA_LOCAL_OCR_ENABLED=false`, `CLOUDA_NO_NETWORK_TESTS=true`, `--cov-fail-under=80`. Step list quoted in DD §5.
- Enforcement-of-hygiene tests: `tests/security/test_adversarial_hardening.py::test_deployment_defaults_to_loopback_and_ci_actions_are_pinned` (L392) **T**.
- Lint/type configs: `pyproject.toml [tool.ruff]` L127-145 (select `E4,E7,E9,F`), `[tool.black]` L122-125, `[tool.bandit]` L147-157 (skips B608/B310/B615 — disclosed), `[tool.coverage]` L159-171; `mypy.ini` (py3.11, non-strict, excludes).
- Determinism tests: `tests/quality/test_fixture_determinism.py` (seeded noise byte-identity; canonical manifest ordering, schema `clouda.pretraining.manifest.v1`); `tests/runtime/test_resume_determinism.py` (4 tests incl. `test_interrupted_resume_matches_uninterrupted_run`); `tests/factory/test_seeds.py` (recorded legacy vectors).
- Doctor implementation: `clouda_data/doctor/` (`report.py::collect_report` L40-155; `--deep` L105-160); CLI subcommand `clouda_data/pipeline/cli.py:1324`; docs `docs/operations/ENVIRONMENT_DOCTOR.md`; 81 doctor tests.

## §6 Security specifics

- `SECURITY.md` — quoted protections in DD §6; vuln disclosure is private reporting; historical exercise reports deliberately kept outside public tree.
- `docs/security/`: `THREAT_MODEL.md` (trust-boundary table + controls list), `DATA_PRIVACY.md` (runtime:// vs dataset://; workers cannot read runtime storage), `LOCAL_MODEL_SECURITY.md` ("code-level controls complete; model trust remains a human decision"; `local_files_only=True`, `trust_remote_code=False`), `MULTI_USER_SECURITY_ARCHITECTURE.md` (Firebase/Google identity + SQLite; immutable `owner_user_id`; guest defaults 10 MiB / 5 pages / 1 active job), `TRAINING_DATA_PRIVACY.md` ("User uploads have no route into the training catalog").
- Lab security: `clouda_lab/dashboard/security.py` (`is_loopback` L83, `require_loopback` L94, `browser_safe` L61); `clouda_lab/dashboard/document_intelligence.py` (`MAX_PDF_BYTES = 10*1024*1024` L19, `MAX_PDF_PAGES = 25` L20). Tests: `tests/dashboard/test_security_and_app.py`, `tests/dashboard/test_document_intelligence.py::test_service_performs_no_network_call_or_persistence` (L58) **T**.
- Repository scan: `tools/validation/repository_scan.py` (secret patterns incl. PEM/AKIA/sk-; >5 MiB files; forbidden suffixes `.safetensors/.pt/.pth/.ckpt/.onnx/.gguf/.sqlite3`; forbidden historical paths); CI step "Secret, large-file, and forbidden-path scan" **CI**; tests `tests/security/test_repository_scan.py` **T**.
- Consent two-gate: `clouda_contracts/security.py::may_use_user_document_for_training` (L75-81); `tests/security/test_file_safety.py::test_user_documents_are_not_training_data_without_both_approvals` (L71) **T**.
- Worker race/finalization regression set (14 named tests): `tests/test_distributed_worker.py` L254, 492, 520, 561, 621, 660, 715, 746, 827, 879, 918, 952, 981, 1125, 1159 **T**; executed against real Redis in the CI `redis-integration` job **CI**.
- Archive hardening: `tests/security/test_file_safety.py:29` (zip-slip + decompression ratio); `test_adversarial_hardening.py:254,274,361` (duplicate/case collision, symlinked backup destination, symlinked archive members) **T**.

## §7 Provider/model specifics

- `ExtractionEngine` protocol `pdfword/engines.py:79-94`; `OCRResult` L29; `EngineRegistry` L103-127; `FutureOcrEngine` L184 ("always unavailable"); `FeatureFlaggedLocalModelEngine` L219-331; `OCR_STATUS_PENDING_MODEL` L13.
- Providers: `pdfword/local_ocr_adapters.py` — `MockOCRProvider` L101 (gated `CLOUDA_ALLOW_MOCK_OCR`), `LocalHTTPProvider` L125, `CommandLineOCRProvider` L217, `TransformersVisionLanguageProvider` L326 (`local_files_only=True`, `trust_remote_code=False` L367-407), factory `provider_from_config` L419.
- `clouda_models/`: `metadata.py` (frozen `ModelMetadata` incl. `deployment_status` ∈ {disabled, evaluation, approved}), `registry.py` (`deployable()` requires both approvals), `local_models.py::resolve_local_checkpoint` (rejects executables), `evaluation.py`, `providers.py` (`ModelProvider` protocol). Registry data: `configs/models/registry.v1.json` (one disabled placeholder).
- Confidence policy: `pdfword/ocr_self_review.py:222-230` (confidence only when `native_confidence_validated` is true, threshold < 0.5); README: Document Intelligence "does not expose user-facing accuracy, confidence, or quality percentages"; review-marker scrubbing `pdfword/docx_export.py:50-60`. Test: `tests/test_model_agnostic_engines.py::test_local_model_accepts_success_without_optional_confidence` **T**.
- `docs/MODEL_INTEGRATION.md` — "Benchmarking is complete… not installed or integrated as a final OCR engine"; 6-step activation gate. `docs/MODEL_EVALUATION_TEMPLATE.md` — governance template, unfilled.

## §8 Data/training specifics

- Factory chain: `clouda_data/factory/factory.py` (773 lines; "Ties ingest -> render -> distort -> export -> manifest together… identical for 1 worker or N workers"); render `clouda_data/factory/render/` (WeasyPrint + vendored RAQM byte-parity stack `clouda_data/factory/render/_raqm/`); ingest text/image/HF; profiles `factory/profiles/scan_families.py`.
- Distortion design docs: `docs/data_foundation/DISTORTION_DESIGN.md`, `REAL_DISTORTION_ENGINE.md`, `DISTORTION_PROFILES.md`.
- Pretraining pipeline: `clouda_data/pretraining/` (13 modules; schema `clouda.pretraining.sample.v1`); `docs/pretraining_dataset_infrastructure.md` ("validated on synthetic fixtures"; splitting is union-find group-aware, `splitting.py`).
- Quality gate doc: `docs/data_quality/CLOUDA_DATA_QUALITY.md` (602 lines, includes limitations and FP/FN tradeoffs).
- Dataset catalog: `dataset_catalog/registry/datasets_v1.json`, `foundation_sources_v1.json`, `schemas/dataset-record-v1.schema.json`, `licenses/REVIEW_STATUS.json` (legal_reviewed_at 2026-07-24; 3 approved_with_conditions, 8 pending, 3 research_only, 2 blocked).
- Planner/preflight/loader: `clouda_training/planner/planner.py` ("never runs training"; reuses preflight math), `clouda_training/preflight/` (≥1 blocker ⇒ NOT_READY), `clouda_data/training_data/loader.py` (deterministic shuffle; sealed cursors; holdout fail-closed gate); docs `docs/training/HARDWARE_VALIDATION_TODO.md`, `docs/training_data/HARDWARE_VALIDATION.md` (enumerated unmeasured GPU items).
- Teacher pipeline: `docs/teacher-pipeline/ARCHITECTURE.md`, `KNOWN_LIMITATIONS.md` ("No provider credential or live request has been tested"); `pdfword/teacher_pipeline/`; test `tests/test_teacher_pipeline_phase3.py` **T**; dormant via `TEACHER_PIPELINE_MODE=disabled` (`.env.example` L124).
- Key router (dormant scaffolding): `docs/key-router/ARCHITECTURE.md` ("dormant foundation", `KEY_ROUTER_MODE=disabled`); `pdfword/key_router/`; `tests/test_key_router.py`, `tests/test_key_router_phase2.py` **T**.

## §9 Benchmark specifics

- Identity: `benchmarks/ocr_arabic/release.json` — `benchmark_id: "clouda-ocr-arabic-177-v1"`, `distorted_page_count: 177`, `clean_source_count: 100`, `manifest_sha256: 2a499ed0…fb893`, `asset_publication: "metadata_only"`, `raw_evidence: "retained_privately"`, `complete_run_count: 6`, `partial_run_count: 1`, `failed_run_count: 1`.
- Manifest: `benchmarks/ocr_arabic/benchmark_manifest.jsonl` — 177 lines counted **L**; SHA-256 re-hashed **L** → `2a499ed0268a5c583a1b174ef241e05affb9c4ba757593eb9ee3fc6f3e4fb893` = pinned value in `methodology.md` and `scripts/validate_release.py::EXPECTED_MANIFEST_SHA256`.
- Validator executed: `python benchmarks/ocr_arabic/scripts/validate_release.py` → "OCR Arabic benchmark metadata validation: PASS" **L** (also pins `EXPECTED_RANKING`, `EXPECTED_FAMILY_COUNTS`, forbidden asset suffixes, leaked-path fragments).
- Metrics: `benchmarks/ocr_arabic/methodology.md` ("Normalized Arabic CER is the primary ranking metric and lower is better"; normalization rules quoted in DD §9); implementation `clouda_data/ground_truth/normalization.py` (`normalize_for_comparison`), `clouda_data/evaluation/cer.py|wer.py`; deterministic Levenshtein tie-breaking in `benchmarks/scripts/calculate_metrics.py`.
- Leaderboard: `benchmarks/ocr_arabic/RESULTS.md` (rows quoted verbatim in DD §9); exclusions `dots.mocr` (PARTIAL, 30/177) and `PaddleOCR-VL-1.6` (FAILED_SMOKE, 0/177); hardware caveat quote.
- Model rights: `benchmarks/ocr_arabic/models.csv` — all 8 rows `NOT_CONFIRMED_IN_LOCAL_EVIDENCE` + `NEEDS_REVIEW` ×2 (full table in DD §10); enforced by `tests/benchmarks/test_ocr_arabic_release.py::test_rights_dimensions_and_model_permissions_are_separate` (L121) **T**.
- Provenance: `source_manifest.csv` (100 rows; per-record `evaluation_permission`, `commercial_training_permission`, `asset_redistribution_permission` ∈ {ALLOWED, NOT_ALLOWED, NEEDS_REVIEW}); `source_summary.csv` (14 datasets: 6 KITAB-Bench-derived = 30, CALFA ×5 = 30, Arabic E-Book Corpus = 20, arabic-img2md = 15, craneset = 5).
- Explicit non-findings (searched, not found — do not claim): a documented one-model-per-GPU fairness policy for the 177-page run; the runner's resume/OOM code (private); any 500-page (or larger) benchmark plan; a documented contamination check between the benchmark cohort and future training splits.
- HunyuanOCR-1.5 upstream pin: `UPSTREAM_COMPATIBILITY.md` and `clouda_training/hunyuan/models.py` — `Tencent-Hunyuan/HunyuanOCR` @ `c55965d3da1e`; upstream license `NOASSERTION` (custom Tencent license) **UP** (see https://huggingface.co/tencent/HunyuanOCR).

## §10 Licensing specifics

- `LICENSE` — standard Apache-2.0 text. `NOTICE` — "Copyright 2026 CloudaOCR Team"; third-party carve-outs; bundled fonts Amiri / Scheherazade New / Noto Naskh Arabic / Cairo under SIL OFL 1.1 (attribution lines quoted in NOTICE). **UP** for OFL: https://openfontlicense.org.
- `THIRD_PARTY_NOTICES.md` — repo ships source/docs/fixtures only; runtime deps enumerated; "no weight/dataset redistribution without confirmed license"; Poppler/Tesseract binaries must not be committed.
- `DATA_LICENSES.md` — allowed vs not-allowed repo content lists (quoted in security agent report; synthesized in DD §10).
- `UPSTREAM_COMPATIBILITY.md` — "Clouda does not vendor, redistribute, or auto-download any upstream code, weights, or data"; upstream SFT defaults recorded as baseline only, not tuned for Arabic.
- `SBOM.json` — CycloneDX-shaped, 155 KB, `components[]` with purls + declared licenses.
- Open-source scope exclusions: `README.md` "Open-Source Scope" (lines ~270-276) — enumeration quoted in DD §10.
- Uncertainty flags: catalog dataset review dated 2026-07-24 (`dataset_catalog/licenses/REVIEW_STATUS.json`); **no external legal review is evidenced for upstream model licenses**; HunyuanOCR custom license is the key open question.

## §11 Deployment specifics

- Install: `README.md` install/demo sections; `constraints/py311.txt` ("Direct dependency constraints validated on Python 3.11.9 during the merge"); extras in `pyproject.toml`; CI parity verified (CI installs with the same constraint file and a superset of extras).
- CPU-only base: `requirements-base.txt` (streamlit, fastapi, uvicorn, python-docx, pypdf, pypdfium2, Pillow, redis, rq — no torch); `requirements-rocm.txt` is comment-only ("does not claim tested AMD ROCm support yet").
- Services/scripts: `app.py` (Streamlit), `pdfword/worker_api.py` (FastAPI), `pdfword/worker.py` ("This release intentionally supports WORKER_CONCURRENCY=1 only"), `start_clouda_all.bat/.ps1`, `clouda_lab/cli.py serve` (rejects non-loopback binds).
- Linux: `deploy/linux/` — `install.sh`, `run.sh`, `stop.sh`, `health_check.sh`, `backup.sh`, systemd units (`clouda.service`, `clouda-api.service`, `clouda-backup.*`, `clouda-cleanup.*`), `clouda.env.example`; `docs/operations/DEPLOYMENT.md`. Known defect: `deploy/linux/README.md:11` references `DISTRIBUTED_DEPLOYMENT.md`, which does not exist.
- Offline evidence: CI env flags; `.env.example` `LOCAL_PROCESSING_ENABLED=false`; `docs/lab/CLOUDA_LAB_BACKEND.md` ("All training flows… execute through the deterministic MockTrainer / dry-run path").

## §12–§13 Capability matrix / limitations

Each matrix row's evidence appears in the sections above; the not-yet-available rows trace to: absence greps (semantic layer), `README.md` status table ("Final trained Clouda OCR model — Not built — deliberately; candidates are benchmarked, not adopted"; "Production OCR inference / hosted service — Not built"), and `configs/models/registry.v1.json` placeholder state.

Limitations §13 items trace to: model license status (`models.csv`), self-reported results + private runner (`release.json`), `HARDWARE_VALIDATION_TODO.md`/`HARDWARE_VALIDATION.md` (no GPU validation), absence greps (semantic layer, external audit, market evidence), `pyproject.toml`/`mypy.ini` (gate moderate-ness), `pdfword/worker.py` (concurrency 1), `deploy/linux/README.md:11` (dead reference), and the README's own naming note (`clouda-pdf` package vs Clouda OCR product).

## §14–§16 Roadmap / risks / capital

- Roadmap phases quoted from `docs/ROADMAP.md` ("Next 0–2 months", "Months 2–4", "Months 4–6", closing status paragraph); status headers from `ROADMAP.md` ("Implemented" / "Partially complete / disabled by default" / "External decisions").
- Risk-register mitigations trace to: `docs/adapters/MULTIMODEL_ADAPTERS.md`, `models.csv`, `clouda_data/quality/leakage.py`, `source_summary.csv`, `RESULTS.md` hardware caveat, `clouda_training/planner/`, `deploy/linux/`, `.github/workflows/ci.yml` matrix, `tools/validation/repository_scan.py`.
- Capital: only documented statements quoted verbatim in DD §16 — `README.md` ("limited primarily by access to suitable GPU compute"), `ROADMAP.md`, `docs/ROADMAP.md`, `docs/MODEL_INTEGRATION.md` (same sentence), `docs/data_foundation/AWS_DEPLOYMENT_PLAN.md` (us-east-2 / Spot P / 8 vCPU quota / p3.2xlarge / required AWS work list), `tools/data_foundation/aws/` plans (spot_interruption, automatic_shutdown, checkpoint_upload, result_download, project_upload). **No budget figures exist in the repository** (searched "budget", "cost", "$").

## §17–§18 Reproducibility checklist & live log

Commands executed live on the audit machine (2026-09-20, Windows 10, Python 3.11.9, CPU-only):

| # | Command | Result |
|---|---|---|
| 1 | `git fetch origin` + `git log origin/main -1 --format='%H %ad %s'` | `45d0078e551267535f85c79badd432fdc8514bef 2026-09-19 23:57:34 +0300 Merge pull request #1…` |
| 2 | GitHub API `repos/…/actions/runs?per_page=5` | latest `main` run: `completed / success` (2026-09-19T20:57:37Z) |
| 3 | `python benchmarks/ocr_arabic/scripts/validate_release.py` | `OCR Arabic benchmark metadata validation: PASS` |
| 4 | Manifest line count + SHA-256 re-hash | 177 lines; `2a499ed0268a5c583a1b174ef241e05affb9c4ba757593eb9ee3fc6f3e4fb893` (match) |
| 5 | `python scripts/demo.py` | DOCX + JSON produced; states `direct_pdf_text`, `pending_ocr_model`, `blank_page` |
| 6 | `python -m clouda_data.pipeline.cli doctor --deep` | Deep checks PASS (backend smoke; offline MockTrainer dry-run; canonical PDF self-test); `Overall: PARTIALLY READY`, exit 0 — the only FAIL is machine-specific free-disk space (6.2 GB < 10 GB threshold), which demonstrates the check functions |
| 7 | Static counts | `find tests -name "*.py" | wc -l` → 202; `grep -r "def test_" tests --include="*.py" | wc -l` → 1,895 |
| 8 | `CLOUDA_LOCAL_OCR_ENABLED=false CLOUDA_NO_NETWORK_TESTS=true python -m pytest tests -q` (offline, CPU) | **2,071 passed, 15 skipped, 0 failed, 17 warnings, 648.64 s** (pass count exceeds the 1,895 static `def test_` count due to parametrized tests; skips consistent with env-gated design such as `CLOUDA_CUDA_SMOKE` on a GPU-less machine) |

CI-verified commands (executed by the green `main` run of 2026-09-19, `45d0078e`): full dependency install under `constraints/py311.txt`; import/CLI smoke; `node --check`; `compileall` gate; ruff; black; mypy; offline pytest with coverage ≥ 80; synthetic acceptance; repository scan; Bandit; pip-audit; wheel/sdist build; demo smoke.

## Verification-method notes (read before relying on line numbers)

1. All file evidence was gathered by read-only inspection at branch `docs/investor-github-hardening` @ `a4c4345` (docs-only delta from `main` @ `45d0078e`); code line numbers are identical on `main`.
2. Benchmark **results** (the leaderboard numbers) are self-reported by the project; what this audit independently verified is the *integrity machinery*: manifest hash, validator PASS, manifest row count, exclusion records, and rights-state invariants enforced by tests. The runs themselves ran on private hardware with the runner retained privately (`release.json`), so they cannot be independently re-executed by a reviewer — this is disclosed, not hidden.
3. "No external audit/certification" and "no market evidence" are verified-by-absence claims (searched `SECURITY.md`, `docs/security/`, README, ROADMAP, budget terms). Absence in the public repo does not prove absence elsewhere; it proves the repository provides no such evidence.
4. Live execution in §18 was CPU-only on Windows; CI additionally proves the same suite on Ubuntu. No GPU workloads, training, or networked model downloads were performed or simulated during this audit.
