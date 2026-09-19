# Clouda OCR — Investor Technical Due-Diligence Pack

- Repository: https://github.com/sahrasayed3-crypto/clouda-ocr
- Website: https://cloudaocr.xyz
- Audited revision (`origin/main`): `45d0078e551267535f85c79badd432fdc8514bef` (2026-09-19, merge of PR #1 "fix GitHub CI failures")
- Documentation revision: branch `docs/investor-github-hardening`, HEAD `a4c4345` — contains documentation only; **no runtime code differs from `main`**
- Audit date: 2026-09-20
- Method: read-only inspection of the repository at the SHA above, live GitHub Actions/CI metadata, and local execution of the CPU-only verification commands listed in §18 on the maintainer's machine (Python 3.11.9, Windows, no GPU). No GPU workloads were run, no models were trained, and no results were invented.
- Scope discipline: this pack is a verifiable evidence document, not marketing. Where a claim could not be proven from the repository, it is marked PARTIALLY_VERIFIED, EXPERIMENTAL, PLANNED, or UNKNOWN. The companion file [`INVESTOR_TECHNICAL_DUE_DILIGENCE_SOURCES.md`](INVESTOR_TECHNICAL_DUE_DILIGENCE_SOURCES.md) maps every major claim to its evidence.

---

## 1. Executive technical summary

Clouda OCR is a **document-processing runtime and evaluation platform for Arabic-first PDF → DOCX conversion**, built around a fail-closed pipeline rather than a single OCR model. Today, in its default configuration, it extracts text from born-digital PDFs, routes every page through a trusted digital-text gate, and deliberately leaves scanned pages in a `pending_ocr_model` state because **no final trained OCR model has been selected, licensed, or integrated** — a fact the project states plainly in its README.

What exists and is verified:

- A production PDF→DOCX path for born-digital pages with Arabic/RTL-aware DOCX export, page-level analysis, a trusted-text acceptance gate with 12 rejection checks, post-extraction digest binding, and explicit `blank_page` / `near_blank` / `manual_review` / `pending_ocr_model` states (`pdfword/`, `clouda_contracts/`).
- A model-agnostic OCR provider abstraction (`ExtractionEngine` protocol, engine registry, four provider types) that is disabled by default and requires a pinned model revision before it can ever accept OCR output.
- An implemented, tested OCR self-review and selective re-read subsystem (categorical defect detection, budgeted region re-reads bound to render-identity hashes, conservative reconciliation) that activates only when a local OCR engine is explicitly enabled.
- A deterministic synthetic scan factory (108 registered distortion operators, content-derived BLAKE2b seeds, SHA-256 provenance manifests, resumable runs), a dataset quality gate with L0–L6 cross-split leakage detection, and a fail-closed dataset licensing layer.
- A complete but **mock-only** training experiment framework: planner, preflight validator, streaming data loader, checkpoint/resume machinery, and a real PyTorch backend proven on a synthetic linear adapter — with real GPU training structurally fail-closed.
- A frozen, metadata-only 177-page Arabic OCR benchmark (`clouda-ocr-arabic-177-v1`) whose canonical manifest SHA-256 was independently re-verified during this audit, ranking 6 completed model runs with HunyuanOCR-1.5 first (Normalized Arabic CER 0.391497) and explicitly forbidding cross-GPU speed claims.
- Cross-platform CI (Windows + Ubuntu, Python 3.11) with a 16-step validate job — including lint, type-check, compile gate, offline test suite with an 80% coverage floor, three security scans, dependency audit, wheel/sdist build, and a demo smoke test — currently **green on `main`** (verified live on 2026-09-20).

What does not exist yet (stated, not hidden): a trained Clouda model, production OCR inference, GPU-validated training, any semantic/document-understanding layer, any external security audit, and any documented market or revenue validation. §13 covers limitations; §15 covers risks.

The single-sentence technical assessment: **this is an honest, unusually well-instrumented pre-model platform — the routing, evaluation, licensing, and data infrastructure that would sit around an OCR model are built and tested; the model itself, and the GPU work to train it, are not, and the repository says so everywhere it could.**

## 2. Current main SHA and audit basis

| Item | Value |
|---|---|
| `origin/main` SHA | `45d0078e551267535f85c79badd432fdc8514bef` |
| `main` last commit | 2026-09-19 — "Merge pull request #1 from sahrasayed3-crypto/feature/fix-github-ci-failures" |
| PR count on repo | 1 merged (17 feature branches were merged via PRs into it per `docs/investor-github-audit.md`) |
| CI on `main` | **success** — run created 2026-09-19T20:57:37Z, re-checked via GitHub API on 2026-09-20 |
| This pack written on | branch `docs/investor-github-hardening` @ `a4c4345` (docs-only delta vs `main`) |
| Local verification machine | Windows 10, Python 3.11.9, CPU-only, no GPU |

All file references below resolve at the audited SHA. Static counts in this document were re-measured on 2026-09-20 (§5); live test-suite execution was performed on the audit machine (§18).

## 3. Evidence index

Statuses: VERIFIED (re-producible from the repo/CI at the audited SHA), PARTIALLY_VERIFIED, EXPERIMENTAL (implemented but not exercised on real hardware/weights), PLANNED, UNKNOWN.

| Topic | Claim | Evidence path | Verification | Status |
|---|---|---|---|---|
| Architecture | Page Analyzer → Trusted Text Gate → Decision Engine pipeline | `docs/ARCHITECTURE.md`, `pdfword/page_routing.py` | Code + 32 routing tests | VERIFIED |
| Page routing | Pages classified digital / scanned / hybrid / blank / near-blank with evidence-only warnings | `pdfword/page_routing.py:474-622` | `tests/test_page_routing.py` | VERIFIED |
| Trusted digital text | 12-check acceptance gate, fail-closed to `uncertain` | `pdfword/page_routing.py:639-862` | `tests/test_page_routing.py:267-411` | VERIFIED |
| Trusted text binding | Post-extraction SHA-256 digest re-check; mismatch → `review_required`, text not emitted | `pdfword/page_routing.py:953-972`, `pdfword/ocr_pipeline.py:362-386` | `tests/test_page_routing.py:506` | VERIFIED |
| OCR self-review | Categorical defect review of OCR output only | `pdfword/ocr_self_review.py` | `tests/test_ocr_self_review.py` (14 tests) | VERIFIED (inactive by default) |
| Selective re-read | Budget-bounded (4 regions / 4 attempts), render-identity-locked | `pdfword/ocr_self_review.py:57-64,151-161` | `tests/test_ocr_self_review.py:166-360` | VERIFIED (inactive by default) |
| Structured output | RTL/Arabic DOCX export; text-only; sanitized | `pdfword/docx_export.py` | `tests/test_arabic_fixtures.py` | VERIFIED |
| Model-agnostic boundary | OCR disabled by default; unpinned revisions rejected | `pdfword/engines.py:219-331` | `tests/test_model_agnostic_engines.py` | VERIFIED |
| Security design | Storage-root separation, path containment, worker auth | `SECURITY.md`, `clouda_contracts/storage.py`, `pdfword/worker_api.py` | `tests/contracts/`, `tests/security/`, `tests/test_distributed_worker.py` | VERIFIED |
| Security certification | External audit/certification | none exists | absence verified | NOT_PRESENT (see §6) |
| Tests | 202 test files, 1,895 test functions (static) | `tests/` | re-counted 2026-09-20 | VERIFIED (static) |
| CI | 16-gate validate job, Windows+Ubuntu, green on main | `.github/workflows/ci.yml` | live API check + local reruns | VERIFIED |
| Packaging | `clouda-pdf` 0.2.0, py3.11-pinned, extras match docs | `pyproject.toml`, `constraints/py311.txt` | CI install + wheel/sdist build step | VERIFIED |
| Data pipeline | Deterministic synthetic scan factory, 108 operators | `clouda_data/factory/`, `clouda_data/distortion/` | `tests/factory/` (84 tests) | VERIFIED |
| Data quality | Dedup + L0–L6 leakage gate | `clouda_data/quality/` | `tests/quality/` (231 tests) | VERIFIED |
| Licensing gate | Fail-closed dataset/model rights enforcement | `clouda_training/exporter.py`, `dataset_catalog/` | `tests/security/`, `tests/benchmarks/` | VERIFIED |
| Training framework | Full lifecycle via deterministic MockTrainer; real GPU training fail-closed | `clouda_training/` | `tests/training/`, `tests/runtime/` | VERIFIED (mock) / EXPERIMENTAL (torch) |
| Final trained model | None exists | `README.md`, `clouda_models/registry.v1.json` (single disabled placeholder) | registry validation tests | NOT_PRESENT (deliberate) |
| Benchmark framework | Frozen 177-page Arabic benchmark, metadata-only | `benchmarks/ocr_arabic/` | manifest SHA-256 re-hashed locally; validator PASS | VERIFIED |
| Benchmark runner | The runner that produced the 177-page results | retained privately (`release.json` `raw_evidence`) | not inspectable | UNKNOWN (private) |
| One-model-per-GPU policy | Fairness policy for the 177-page run | not documented in public metadata | searched, not found | UNKNOWN |
| Benchmark results | 6 complete runs; HunyuanOCR-1.5 top at N-CER 0.391497 | `benchmarks/ocr_arabic/RESULTS.md` | self-reported; manifest/validator integrity verified | PARTIALLY_VERIFIED |
| Provider adapters | HunyuanOCR-1.5 / Qwen3-VL SFT adapters, mock-verified | `clouda_training/hunyuan/`, `clouda_training/qwen/` | `tests/hunyuan/`, `tests/multimodel/` | EXPERIMENTAL |
| Semantic understanding | Document summaries / QA / table extraction | absent from code and roadmap | searched | PLANNED (not even specified) |
| Deployment | CPU-only default install; loopback services; Linux systemd deploy | `README.md`, `deploy/linux/` | demo smoke + doctor run locally | VERIFIED (local scale only) |
| License | Apache-2.0, original code only, enumerated carve-outs | `LICENSE`, `NOTICE`, `README.md` | read; no legal review performed | VERIFIED (as text) |

## 4. Architecture (as implemented, not aspired)

The pipeline below is the one actually executed at the audited SHA (`docs/ARCHITECTURE.md` lines 5–13 and 55–60, `pdfword/ocr_pipeline.py::process_pdf`):

```
PDF/image upload (bounded)
  → page analysis (pdfword/page_routing.py)        — geometry, text spans, image evidence,
                                                     content-stream operators, Arabic integrity
  → trusted digital-text gate                      — trusted / untrusted / uncertain + reason codes
  → page decision engine                           — route per page
  → [trusted]   direct PDF text extraction         — digest-bound; mismatch → review_required
  → [untrusted] OCR if an approved engine is enabled,
                else pending_ocr_model             — OCR pass → categorical self-review
                → budgeted selective re-read → conservative reconciliation
  → DOCX export (RTL-aware, text-only, sanitized)  — review placeholders where applicable
```

Page states are a closed contract: `digital_text`, `blank_page`, `near_blank`, `pending_ocr_model`, `failed`, `manual_review` (`docs/ARCHITECTURE.md:44-51`). A short character count alone can never establish `near_blank`; evidence gaps produce explicit warnings, never fabricated data.

Major components:

| Component | Responsibility | Main modules | Key tests | Failure behavior | Maturity |
|---|---|---|---|---|---|
| Page analysis | Per-page evidence: geometry, spans, images, operators, Arabic integrity, layout risk | `pdfword/page_routing.py` (`analyze_pdf_page`) | `tests/test_page_routing.py` (32 tests) | Missing evidence becomes an explicit warning code; page becomes `uncertain` | VERIFIED |
| Trusted text gate | Accept/reject embedded digital text (12 rejection checks: fragmentation, Arabic isolated-character ratio ≥0.60, Unicode corruption, image-dominant hybrid text, reading-order/multi-column risk, …) | `pdfword/page_routing.py::evaluate_digital_text_trust` | `tests/test_page_routing.py:267-411` | Fail-closed: default verdict is `uncertain`; only usable + distributed text is `trusted` | VERIFIED |
| Route decision | Map gate verdict + engine availability to a page path | `pdfword/page_routing.py::decide_page_route` | `tests/test_page_routing.py:413-478` | `uncertain` → manual review; `untrusted` without OCR → `pending_ocr_model` | VERIFIED |
| Direct extraction | Extract trusted digital text, re-verify digest | `pdfword/engines.py::DirectPdfTextEngine`, `page_routing.py:953-972` | `tests/test_readiness_pipeline.py` | Digest mismatch → `review_required`; untrusted text never emitted | VERIFIED |
| OCR pipeline + self-review + selective re-read | First-pass OCR review (categorical defects), budgeted region re-reads locked to render identity, conservative reconciliation | `pdfword/ocr_pipeline.py:444-533`, `pdfword/ocr_self_review.py` | `tests/test_ocr_self_review.py` (14 tests) | Any stale/unverifiable geometry, disagreement, or unresolved corruption → `review_required` | VERIFIED code; **inert by default** (OCR disabled) |
| DOCX export | Editable text-only DOCX; RTL paragraphs, `w:bidi`/`w:rtl`, Arabic page-number footer; text sanitized (bidi overrides stripped) | `pdfword/docx_export.py`, `clouda_contracts/security.py` | `tests/test_arabic_fixtures.py`, `tests/test_quality_acceptance_policy.py` | Review pages get visible placeholders; quality-adjacent wording scrubbed from reasons | VERIFIED |
| Engine boundary | Model-agnostic engine registry; local OCR off by default; pinned-revision requirement | `pdfword/engines.py`, `pdfword/local_ocr_adapters.py` | `tests/test_model_agnostic_engines.py`, `tests/test_engine_boundaries.py` | Unavailable/unconfigured engine → `pending_ocr_model` (neither success nor failure) | VERIFIED |
| Distributed worker | RQ worker + FastAPI internal API; claim tokens, heartbeats, two-phase finalization | `pdfword/worker.py`, `pdfword/worker_api.py` | `tests/test_distributed_worker.py` (37 tests) + Redis CI job | Stale/contested finalization is fenced, recovered, or requeued — never silently overwritten | VERIFIED |

**Semantic / document-level understanding (summaries, table extraction, document QA) is not implemented and is not promised anywhere in the roadmap or docs.** The only related code is crude heuristic ratios in `pdfword/intelligence.py` (e.g. counting lines containing `|`), which is metadata, not understanding. Any investor question about "document intelligence" beyond the bounded Lab upload analyzer (§11) should be treated as future scope.

## 5. Engineering-quality evidence

### Tests

- Static count (re-measured 2026-09-20): **202 Python files under `tests/`, 1,895 `def test_` functions.** The README quotes "183 test files / ~1,900 test functions" as a static count and instructs readers to run the suite for live numbers; the file count in the README is slightly stale relative to the current tree, the function count matches.
- Live verified run (executed 2026-09-20 on the audit machine, CPU-only, Windows, Python 3.11.9, with the CI environment flags `CLOUDA_LOCAL_OCR_ENABLED=false`, `CLOUDA_NO_NETWORK_TESTS=true`): **2,071 passed, 15 skipped, 0 failed**, in 10 min 48 s. The pass count exceeds the static `def test_` count (1,895) because of parametrized tests.
- Coverage categories (per-directory static counts sum to 1,895): core runtime and routing (501), data quality (231), Clouda Lab (156), data foundation (155), preflight (155), planner (100), results store (97), factory (84), doctor (81), dashboard (68), multimodel adapters (55), Hunyuan adapter (53), training (34), security (31), contracts (21), and others.
- Critical regressions covered by named tests, including worker finalization races (`test_duplicate_upload_cannot_recover_an_active_finalization`, `test_worker_claim_token_fences_stale_dispatcher`, `test_worker_start_cannot_steal_job_during_finalization` — `tests/test_distributed_worker.py`), archive traversal/bombs (`tests/security/test_file_safety.py:29`), and storage-URI boundary escapes including a Hypothesis property test (`tests/security/test_adversarial_hardening.py:411`).

### CI (`.github/workflows/ci.yml`, the only workflow)

- Platforms: `windows-latest` + `ubuntu-latest`, Python 3.11. Triggers: push, pull_request, workflow_dispatch. `permissions: contents: read`; concurrency cancel-in-progress; both actions pinned by commit SHA with `persist-credentials: false` on every checkout — and a test enforces the pinning (`tests/security/test_adversarial_hardening.py:392`).
- Job `validate` steps, in order: (1) install tested dependency set `pip install -c constraints/py311.txt -e ".[server,worker,data,training,models,test,dev,factory]"`; (2) import + CLI smoke tests for all five packages; (3) JavaScript syntax check (`node --check` on the Lab dashboard app); (4) `compileall` maintained-source gate; (5) Ruff; (6) Black maintained-source gate (+ broad diagnostic, non-blocking); (7) Mypy maintained-source gate (+ broad diagnostic, non-blocking); (8) offline pytest with `--cov-fail-under=80`; (9) synthetic OCR data-pipeline acceptance; (10) secret/large-file/forbidden-path scan (`tools.validation.repository_scan`); (11) Bandit; (12) `pip_audit` against `constraints/py311.txt`; (13) wheel + sdist build; (14) demo smoke test (`python scripts/demo.py`).
- Optional jobs (workflow_dispatch-gated): Redis integration (runs the distributed-worker suite against `redis:7-alpine`), GPU-adapter import, large fixtures.
- Current state: **green on `main`** (run of 2026-09-19, re-checked 2026-09-20). The pre-merge history on the feature branch shows the CI-repair trail documented in PR #1 — three failed/cancelled runs before the green one, which is consistent with an honest debugging process, not with a hand-green badge.

### Code-quality gates

- Ruff: `E4/E7/E9/F`, line length 88 (narrow rule set — honest limitation, not a strict lint).
- Black: enforced on maintained sources, line length 88.
- Mypy: enforced on maintained sources with `--no-incremental`; **non-strict** (missing imports ignored; untyped defs allowed) — a real but moderate gate.
- Bandit: enforced on `pdfword`, `tools`, `scripts`; deliberately skips `B608` (SQL string building), `B310`, `B615` — disclosed in `pyproject.toml`.
- Coverage: CI fails under 80% (`pdfword` CLI scope plus the five-package `[tool.coverage.run]` source).

### Reproducibility

- Deterministic fixture generation (`tests/fixtures/generate_fixtures.py`, no randomness) with a byte-identity test for seeded noise and canonical manifest ordering (`tests/quality/test_fixture_determinism.py`).
- Content-derived seeds: BLAKE2b over `(global_seed, source_sha256, document_id, page_index, variant_index, profile, distortion_stage, severity)` — "nothing depends on machine state, wall-clock time or iteration order" (`clouda_data/factory/seed/derive.py`), with recorded-vector regression tests (`tests/factory/test_seeds.py`).
- Factory runs are resumable and byte-deterministic across repeats (`tests/factory/test_e2e.py`), with SHA-256 provenance on every manifest row and atomic (`os.replace`) outputs.
- Training-side determinism: interrupted-resume-matches-uninterrupted, same-seed parameter reproduction, exact RNG-state restoration, full checkpoint state capture (`tests/runtime/test_resume_determinism.py`).
- Exact dependency pins: `constraints/py311.txt` validated on Python 3.11.9; the same constraint file is used by CI install, Bandit, and pip-audit; `SBOM.json` (CycloneDX-shaped) at repo root.
- CI is fully offline (`CLOUDA_NO_NETWORK_TESTS=true`, `CLOUDA_LOCAL_OCR_ENABLED=false`): no dataset or model downloads in the test pipeline.

## 6. Security posture

**Design (documented and code-enforced):**

- Five separated storage roots (`runtime://`, `dataset://`, `artifact://`, `model://`, `cache://`) enforced by `clouda_contracts/storage.py`: roots must be distinct, URIs reject `..`, query/fragment, drive syntax, Windows reserved device names, and non-NFC components; storage is read-only by default. A Hypothesis property test asserts URIs never resolve outside their root (`tests/security/test_adversarial_hardening.py:411`).
- Worker API (`pdfword/worker_api.py`): header-key auth via `hmac.compare_digest` (with a rotation hook), TrustedHost middleware defaulting to loopback, docs endpoints disabled, request-size bound, sliding-window rate limiting, and baseline security headers.
- Result finalization integrity: claim-token fencing, two-phase finalization (`prepare_conversion_finalization` → atomic `os.replace` → `complete_conversion_finalization`), bounded streaming to a `.docx.part` temp file, PDF/ZIP magic and archive validation, quality gate forcing sub-90 scores to `manual_review`, and ownership checks on every heartbeat/failure endpoint. Fourteen race/finalization regression tests cover duplicate uploads, stale recovery, and steal attempts (§3, §5).
- Clouda Lab dashboard: loopback-only binding enforced at the dependency level (403 otherwise), document analysis bounded to 10 MiB / 25 pages, in-memory only (no persistence, no network calls — tested), and `browser_safe()` projection that redacts secrets and replaces out-of-root absolute paths with `[PRIVATE PATH]`.
- Local model loading: `local_files_only=True`, `trust_remote_code=False`, model root containment, command providers run absolute executables with argument arrays and no shell.
- Privacy boundary: "User documents are never training data by default" — a code-level two-gate consent function (`clouda_contracts/security.py::may_use_user_document_for_training`) requires both document-specific consent and an approved policy, **both defaulting to false**, and no runtime path calls it (tested).
- Tenant isolation: immutable `owner_user_id`, server-constructed UUID tenant paths, client IDs never used as directory names (`docs/security/MULTI_USER_SECURITY_ARCHITECTURE.md`).

**Automated checks (CI-enforced):** repository scan for secret patterns (PEM, AWS `AKIA…`, `sk-…`, generic `*_API_KEY/SECRET/TOKEN/PASSWORD` assignments), files >5 MiB, forbidden tracked suffixes (`.safetensors`, `.pt`, `.ckpt`, `.onnx`, `.sqlite3`, …) and forbidden historical paths; Bandit; `pip_audit`; plus the adversarial-hardening test suite (manifest license-forgery, symlink escapes, SSRF defenses, dynamic-SQL allowlisting).

**What this is not:** there is **no external security audit, penetration test attestation, or certification (SOC 2 / ISO 27001 or similar)** — verified by absence across `SECURITY.md`, `docs/security/`, and the README. The existing `docs/investor-github-audit.md` is a self-conducted documentation review, not an external attestation. `SECURITY.md` also states historical exercise reports are deliberately kept outside the public tree, so no third-party validation is inspectable today. Security posture should be read as: **strong engineering hygiene and a test-enforced threat model, zero external validation.**

## 7. Model/provider architecture

- **Abstraction.** OCR engines implement the `ExtractionEngine` protocol (`pdfword/engines.py:79`): `available()` + `extract_page() → OCRResult`. A registry holds `DirectPdfTextEngine`, `FutureOcrEngine`, and `FeatureFlaggedLocalModelEngine`; provider backends are configured via env (`pdfword/local_ocr_adapters.py`): `mock`, `local_http`, `openai_compatible`, `command_line`, `transformers`, `qwen_vl`. A parallel `ModelProvider` protocol exists in `clouda_models/providers.py`.
- **Not tied to one model.** Tests explicitly assert engines can be registered and selected from settings without router changes, and that configuration is CPU/GPU-neutral (`tests/test_model_agnostic_engines.py`).
- **Unavailable provider behavior (the key design decision).** If no OCR engine is enabled — the default — untrusted pages route to `pending_ocr_model`: "neither a successful OCR result nor a final processing failure" (`docs/ARCHITECTURE.md:49`). The system never fakes OCR output.
- **Activation bar.** `FeatureFlaggedLocalModelEngine.available()` requires the env flag **and** a provider **and** a pinned model revision — `unresolved`, `latest`, and `main` are rejected (`pdfword/engines.py:247-253`). `clouda_models/registry.py` additionally rejects any non-disabled registry entry with an unpinned revision and requires `approved` on both deployment and commercial-use status to deploy.
- **Model assets.** `clouda_models/` is metadata-only code (frozen `ModelMetadata` dataclass, registry, evaluation records); the shipped registry `configs/models/registry.v1.json` contains exactly one entry — a **disabled placeholder** (`future-vlm-adapter`, revision `UNRESOLVED`, `checkpoint_uri: null`). No weights exist in the repo, and CI's repository scan forbids committing them.
- **Confidence policy.** Engine-reported confidence is consumed internally only when the provider explicitly flags `native_confidence_validated`, and only as one categorical self-review signal (< 0.5 raises a review issue) — `pdfword/ocr_self_review.py:222-230`. **There is no user-facing confidence, accuracy, or quality percentage**: decisions are categorical, and review-page placeholders have percentage-like wording scrubbed. Document Intelligence in the Lab likewise exposes no accuracy numbers (README).
- **Benchmark candidates are not product dependencies.** HunyuanOCR-1.5 and Qwen3-VL exist in the repo only as SFT *adapters* and evaluation candidates; `requirements-models.txt` pulls in `requests` only. The product runtime performs direct extraction only at the audited SHA.

## 8. Data and training infrastructure

**Data side — genuinely built and tested (VERIFIED):**

- **Synthetic scan factory** (`clouda_data/factory/`, `clouda_data/distortion/`): 108 registered deterministic pixel-level distortion operators (blur, noise, compression, illumination, paper aging, ink, geometry, print artifacts, Arabic-specific degradations such as `faint_diacritics`, `merged_strokes`, `baseline_jitter`; synthetic stamps/holes/page numbers), severity-scaled 0.0–1.0, layout-aware region application, WeasyPrint/RAQM render backend.
- **Determinism and provenance**: content-derived BLAKE2b seeds, SHA-256 on source/clean/output/ground-truth per manifest row, atomic writes, resumable runs with a config hash in the run directory name; byte-determinism and seed-vector tests.
- **Quality gate** (`clouda_data/quality/`): exact/near deduplication (image fingerprints, Arabic-normalized text duplicates, LSH), derived-manifest exclusion plans, and **L0–L6 cross-split leakage detection** where any CRITICAL finding fails the gate. Clustering is deterministic under input shuffling (tested).
- **Dataset catalog and licensing** (`dataset_catalog/`): per-dataset license records with global fail-closed defaults (`pending_datasets_blocked: true`, `unknown_datasets_blocked: true`, `production_requires_approved_license: true`); runtime enforcement in `clouda_data/datasets/license_gate.py` and the training exporter, which raises `PermissionError` for blocked/pending/expired datasets and blocks synthetic data from commercial training.

**Training side — real scaffolding, deliberately no real training (EXPERIMENTAL/mock):**

- **Experiment framework** (`clouda_training/`): full run lifecycle with append-only status, SHA-256 run integrity, resume, config hashing, checkpoint manager, and comparison tooling. Execution today goes through `MockTrainer` — "Deterministic CPU-only trainer used to exercise the full lifecycle" — with fault injection.
- A genuine `TorchTrainerBackend` (zero_grad → forward → backward → accumulation → clip → step → checkpoint) exists and is proven end-to-end **on a synthetic linear adapter on CPU** — proving the optimization/resume machinery, not a model.
- **Planner** (step math, VRAM/storage/cost estimates with an assumptions ledger), **preflight validator** (config/dataset/system/model checks; ≥1 blocker ⇒ NOT_READY, fail-closed), and **streaming data loader** (deterministic shuffle, sealed checkpoints, holdout fail-closed gate) are all implemented and tested.
- **Adapter layer**: HunyuanOCR-1.5 (`clouda_training/hunyuan/`) and Qwen3-VL (`clouda_training/qwen/`) SFT adapters — local-only loading, loss-absence fails explicitly, protected/holdout rows can never be exported — verified **at mock level only**: `real_weights_validated=False`, `gpu_validated=False` (`docs/adapters/MULTIMODEL_ADAPTERS.md`).

**Explicit statement required by the evidence: no final trained Clouda OCR model exists.** The README states it verbatim; the model registry contains only a disabled placeholder; no weight files exist anywhere in the repo. What Clouda has built is the deterministic, license-aware, leakage-controlled data and evaluation infrastructure *around* a model it has not yet trained or adopted.

**Teacher pipeline** (`pdfword/teacher_pipeline/`, `docs/teacher-pipeline/`): mock-verified orchestration (rights validation, checksum verification, quota fencing, agreement scoring, provenance persistence); "No provider credential or live request has been tested" (`docs/teacher-pipeline/KNOWN_LIMITATIONS.md`). Dormant by default (`TEACHER_PIPELINE_MODE=disabled`).

## 9. Benchmark methodology and status

The public benchmark is **`clouda-ocr-arabic-177-v1`** (`benchmarks/ocr_arabic/`), a frozen, metadata-only Arabic OCR model evaluation:

- **Frozen manifest**: 177 distorted pages derived from 100 clean source records; canonical manifest SHA-256 `2a499ed0268a5c583a1b174ef241e05affb9c4ba757593eb9ee3fc6f3e4fb893` is pinned in `methodology.md`, in `release.json`, and as `EXPECTED_MANIFEST_SHA256` in `scripts/validate_release.py` — **re-verified by independent re-hash during this audit (2026-09-20: match)**, and the validator returns PASS locally.
- **Page count**: 177 rows in `benchmark_manifest.jsonl` (counted), each with distortion operator/severity, QC block, seed, and full source provenance (dataset, source SHA-256, clean SHA-256, ground-truth SHA-256, repository revision).
- **Dataset provenance** (`source_manifest.csv`, 100 rows; `source_summary.csv`, 14 datasets): KITAB-Bench-derived sets (30 records), CALFA (baybars/iskandar/RASAM-1/RASAM-2/tarima, 30), Arabic E-Book Corpus (20), `arabic-img2md` (15), `craneset/arabic-ocr` (5). Rights decisions are fail-closed (`LOCAL_ONLY` / `NEEDS_REVIEW` states enforced by a release test).
- **Smoke gate**: `PaddleOCR-VL-1.6` failed its smoke test (0/177 pages, not ranked); `dots.mocr` is a partial run (30/177, no completion summary, not ranked). Excluded runs are published, not hidden.
- **Metrics** (`methodology.md`, `clouda_data/evaluation/`): CER, WER, and **Normalized Arabic CER** as the primary ranking metric — deterministic Arabic normalization (NFC, alef-variant folding, `ى→ي`, tatweel/diacritic removal, whitespace collapse) applied to both sides before edit distance; per-page means; values may exceed 1.0; deterministic Levenshtein tie-breaking.
- **Fairness caveats as documented**: runtime is explicitly **not** a controlled cross-GPU comparison (AIN-7B ran on an RTX PRO 6000 95.6 GB; the others on an NVIDIA L4) and the caveat "must not be used to claim that AIN-7B is definitively faster than another model." **A documented one-model-per-GPU fairness policy was not found in the public metadata (UNKNOWN)**, and the runner itself — including its resume/OOM handling — is "retained privately" (`release.json: raw_evidence`), so its behavior cannot be independently inspected. Resume/OOM machinery exists and is tested in adjacent subsystems (data factory, training), but not verifiably in the benchmark runner.
- **Contamination**: provenance and rights controls are strong; an explicit check that the 177-page cohort does not overlap future *training* splits is **not documented in the public metadata** (the quality gate supports such detection when run; the benchmark cohort has not been shown to have passed it). Treat future adaptation runs as needing this gate before results are comparable.
- **Status**: complete **for this cycle** (`complete_run_count: 6`); README/ROADMAP label it "Completed for this cycle". No larger benchmark (e.g., 500 pages) is documented anywhere in the repo. Historical results are self-reported but integrity-pinned by the manifest hash and validator; the leaderboard explicitly states it "does not establish universal model superiority beyond this benchmark."

**Leaderboard (as published in `RESULTS.md`; ranked by Normalized Arabic CER, lower is better):**

| Rank | Model | Pages | CER | WER | Normalized Arabic CER | GPU | Runtime (s) |
|---|---|---|---|---|---|---|---|
| 1 | HunyuanOCR-1.5 | 177 | 0.564666 | 0.719511 | **0.391497** | NVIDIA L4 | 22.004 |
| 2 | MBZUAI/AIN-7B | 177 | 0.752905 | 0.643977 | 0.837028 | RTX PRO 6000 (95.6 GB) | 7.590 |
| 3 | Qari OCR 0.4.0 | 177 | 1.419241 | 1.638313 | 1.076260 | NVIDIA L4 | 48.601 |
| 4 | Qwen3-VL-4B-Instruct | 177 | 2.007940 | 1.529543 | 1.286567 | NVIDIA L4 | 48.088 |
| 5 | DeepSeek-OCR-2 | 177 | 1.727661 | 1.549825 | 1.486473 | NVIDIA L4 | 22.337 |
| 6 | Arabic Nougat Large | 177 | 2.391472 | 1.876031 | 2.226341 | NVIDIA L4 | 3.172 |

All eight models in `models.csv` — including the six ranked — carry `model_license_status = NOT_CONFIRMED_IN_LOCAL_EVIDENCE` and `raw_output_redistribution` / `training_label_permission` = `NEEDS_REVIEW`. Benchmark outputs may not be used as training labels without review.

## 10. Licensing map

No legal guarantees are made or implied by this document; statuses below are as recorded in the repository at the audited SHA.

| Asset | Type | License / recorded status | Commercial note | Source |
|---|---|---|---|---|
| Clouda code (pdfword, clouda_* packages, tests, docs) | Original source | Apache-2.0 | Grant covers original code only; carve-outs below | `LICENSE`, `NOTICE` |
| Bundled Arabic fonts (Amiri, Scheherazade New, Noto Naskh Arabic, Cairo) | Fonts | SIL OFL 1.1 | Redistributable with notices; embedded in outputs as rendered | `NOTICE` |
| Final model weights, LoRA/QLoRA adapters, checkpoints | Not in repo | Not licensed (do not exist) | "May be licensed, hosted, or distributed separately" | `README.md` open-source scope |
| Private reference texts, training data, production service code, deployment secrets, customer data | Not in repo | Excluded | Same separate-licensing track | `README.md`, `THIRD_PARTY_NOTICES.md`, `DATA_LICENSES.md` |
| HunyuanOCR-1.5 (upstream `Tencent-Hunyuan/HunyuanOCR` @ `c55965d3da1e`) | Model | Recorded: `NOASSERTION` / custom Tencent license; local status `NOT_CONFIRMED_IN_LOCAL_EVIDENCE` | Evaluation only; commercial use unconfirmed; upstream code never vendored | `UPSTREAM_COMPATIBILITY.md`, `benchmarks/ocr_arabic/models.csv` |
| AIN-7B, Qari OCR 0.4.0, Qwen3-VL-4B, DeepSeek-OCR-2, Arabic Nougat Large, dots.mocr, PaddleOCR-VL-1.6 | Models | All `NOT_CONFIRMED_IN_LOCAL_EVIDENCE` (Qari: base-model license controls, adapter unconfirmed) | Outputs `NEEDS_REVIEW` for redistribution and training labels | `benchmarks/ocr_arabic/models.csv` |
| Benchmark source datasets (KITAB-Bench-derived, CALFA, Arabic E-Book Corpus, arabic-img2md, craneset) | Datasets | Per-record permission fields; publication is metadata-only, grants no dataset rights | Redistribution decisions fail closed (`LOCAL_ONLY`/`NEEDS_REVIEW`) | `benchmarks/ocr_arabic/source_manifest.csv`, `NOTICE` |
| Catalogued datasets (rasam, SARD, craneset sample: approved_with_conditions; 8 pending; 3 research_only; 2 blocked) | Datasets | `dataset_catalog/licenses/REVIEW_STATUS.json` (legal-reviewed 2026-07-24) | Pending/blocked/unknown datasets cannot enter commercial training (code-enforced) | `dataset_catalog/`, `clouda_training/exporter.py` |

Uncertainty flags: no external legal review is evidenced for the upstream model licenses (the catalog records a 2026-07-24 legal-review date for catalogued datasets only); the HunyuanOCR custom license is the single most consequential open licensing question because it is the leading benchmark candidate and its SFT adapter is the most developed.

## 11. Deployment model (verified scope: local / self-hosted)

- **What runs on CPU today, with no GPU and no network:** born-digital PDF→DOCX conversion, page routing/trust gate, DOCX export, the full data factory, the quality gate, the training planner/preflight/mock-experiment flow, doctor diagnostics, and the Lab dashboard. Verified locally: `scripts/demo.py` and `doctor --deep` both pass on a GPU-less Windows machine.
- **What requires GPU (and does not exist yet):** production OCR inference and real training. `requirements-rocm.txt` is a comment-only placeholder; AMD/ROCm readiness is "architectural and diagnostic only" (`docs/AMD_ROCM_ROADMAP.md`).
- **Install path** (README, mirrored by CI): Python 3.11 venv → `pip install -c constraints/py311.txt -e ".[server,worker,data,training,models,test,dev]"`; dev-lite path via `requirements-dev.txt`. Extras exist for server, worker, data, factory (+ WeasyPrint render extra), training (+ `training-torch`), models, lab, test, dev, security.
- **Services**: Streamlit UI (loopback :8501), FastAPI worker API (loopback :8000, header-key auth), Redis/RQ worker (`WORKER_CONCURRENCY=1` intentionally — the only supported concurrency in this release), Clouda Lab (loopback :8000/lab via `python -m clouda_lab.cli serve`, rejects non-loopback binds).
- **Linux deployment**: `deploy/linux/` ships systemd units (app, API, backup, cleanup timers), an install script that deliberately installs server-only dependencies ("OCR dependencies belong on the worker"), and operations docs requiring an unprivileged `clouda` user and Redis never exposed publicly. (One dead internal reference noted: `deploy/linux/README.md` cites a `DISTRIBUTED_DEPLOYMENT.md` that does not exist in the repo.)
- **Offline behavior**: loopback-only services; Lab navigation never downloads datasets/models/checkpoints; local OCR disabled unless every required setting is explicit; state externalized via `CLOUDA_*_ROOT` env vars.
- **No production scale is claimed anywhere**, and none should be inferred: the audited evidence covers a single-machine, controlled-staging deployment posture ("suitable for local development and controlled staging" — `SECURITY.md`), with multi-user production listed as requiring identity, TLS, and operational capabilities that are documented as prerequisites, not delivered.

## 12. Current capability matrix

Statuses: IMPLEMENTED (code exists) · VALIDATED (implemented + verified execution) · EXPERIMENTAL (implemented, not exercised on real hardware/weights) · PLANNED · NOT_YET_AVAILABLE.

| Capability | Status | Evidence | Notes |
|---|---|---|---|
| PDF ingestion (bounded) | VALIDATED | `pdfword/conversion_service.py`, demo smoke, CI | Byte/page limits; archive/XML checks |
| Image OCR path | IMPLEMENTED (inactive by default) | `pdfword/engines.py`, `local_ocr_adapters.py` | Disabled without pinned model; mock-verified |
| Page analysis | VALIDATED | `pdfword/page_routing.py`, 32 routing tests, demo | Evidence-only warnings |
| Trusted text gate | VALIDATED | `page_routing.py:639-862`, gate tests | 12 rejection checks, fail-closed |
| Routing | VALIDATED | `decide_page_route`, demo output shows all three default routes | digital / OCR-or-pending / review / blank |
| Direct digital-text extraction | VALIDATED | `DirectPdfTextEngine`, digest-bound, demo | Production path |
| OCR self-review | VALIDATED (code+tests; inactive by default) | `pdfword/ocr_self_review.py`, 14 tests | Categorical decisions only |
| Selective re-read | VALIDATED (code+tests; inactive by default) | ReReadBudget + render-identity lock, tests | Bounded: 4 regions / 4 attempts |
| DOCX/RTL output | VALIDATED | `pdfword/docx_export.py`, Arabic fixtures, demo DOCX | Text-only; sanitized |
| Distributed worker path | VALIDATED | `worker.py`, `worker_api.py`, 37 race/finalization tests, Redis CI job | WORKER_CONCURRENCY=1 only |
| Clouda Lab dashboard | VALIDATED | `clouda_lab/`, loopback tests | Loopback-only; bounded upload analyzer |
| Environment doctor | VALIDATED | `clouda_data/doctor/`, 81 tests; run locally | `--deep` = offline release self-test |
| Synthetic scan factory | VALIDATED | `clouda_data/factory/`, 84 factory tests | 108 operators, deterministic, resumable |
| Data quality / leakage gate | VALIDATED | `clouda_data/quality/`, 231 tests | L0–L6 leakage detection |
| Dataset license gate | VALIDATED | `dataset_catalog/`, `clouda_training/exporter.py` | Fail-closed states |
| Training experiment framework | EXPERIMENTAL | `clouda_training/`, planner/preflight/loader tests | MockTrainer/dry-run only |
| Torch training backend | EXPERIMENTAL | `runtime/torch_backend.py`, `tests/runtime/` | Proven on synthetic linear adapter, CPU |
| HunyuanOCR-1.5 / Qwen3-VL SFT adapters | EXPERIMENTAL | `clouda_training/hunyuan/`, `qwen/`, 108 adapter tests | `real_weights_validated=False`, `gpu_validated=False` |
| 177-page Arabic benchmark | VALIDATED (metadata + integrity) | `benchmarks/ocr_arabic/`, validator, re-hashed manifest | Results self-reported; runner private |
| Semantic page understanding | NOT_YET_AVAILABLE | absent from code and roadmap | Not specified, not promised |
| Document-level understanding | NOT_YET_AVAILABLE | absent | Same |
| Final trained Clouda model | NOT_YET_AVAILABLE (deliberate) | README; registry has only a disabled placeholder | The central open milestone |
| GPU benchmark (new cycle) / GPU training | PLANNED | `docs/ROADMAP.md` months 0–6 plan | Gated on GPU access + licensing |
| Production OCR inference / hosted service | PLANNED | README status table: "Not built" | After model selection + validation |

## 13. Known limitations

Stated plainly, per the project's own convention and this audit's findings:

1. **Dependence on upstream OCR quality and licenses.** The product runtime today cannot process scanned pages beyond `pending_ocr_model`; when a model is adopted, output quality and commercial rights depend on an upstream model (likely HunyuanOCR-1.5) whose license is unconfirmed (`NOT_CONFIRMED_IN_LOCAL_EVIDENCE`).
2. **No final trained Clouda model exists** — not a finished-model company with one missing step; the model-adaptation stage has not started because GPU access and rights verification are prerequisites.
3. **Benchmark results are self-reported**, on a 177-page cohort that is small by academic standards; the runner is private; cross-GPU timing is not a controlled comparison; no explicit contamination check against future training splits is documented for this cohort; no larger benchmark exists yet.
4. **No GPU validation of anything**: training, inference, and the adapters are mock-verified only; `HARDWARE_VALIDATION_TODO.md` enumerates what was deliberately never measured (real VRAM, BF16 stability, multi-GPU resume, real CER/WER improvement).
5. **Semantic/document-understanding layer does not exist** — no summaries, table extraction, or document QA, and no specification for them yet.
6. **Security is self-validated**: strong internal hygiene and test-enforced threat model, but no external audit, penetration test, or certification.
7. **Quality gates are moderate in places**: Mypy is non-strict; Ruff uses a narrow rule set; Bandit skips three checks; the 80% coverage floor applies to CI's scoped sources.
8. **Limited market validation evidence**: no customers, users, revenue, or partnerships are claimed or documented anywhere in the repo. None should be inferred.
9. **Single-model-era operational scope**: worker concurrency is intentionally 1; the multi-worker production story is partially documented (one dead doc reference in `deploy/linux/README.md`).
10. **Naming**: the PyPI package is `clouda-pdf` while the product/repo identity is Clouda OCR — explained once in the README but a residual source of confusion.

## 14. Technical roadmap (documented, not invented)

Source: `docs/ROADMAP.md` (time-phased) and `ROADMAP.md` (status-headed). No dates beyond the documented phases are asserted.

| Phase | What it is | Why it matters | Dependencies | Verification criteria (as documented) |
|---|---|---|---|---|
| 0–2 months: GPU compute + candidate planning | Secure suitable GPU compute for model adaptation/training and runtime integration; plan candidate adaptation; define acceptance criteria | Unblocks the only stage the repo cannot do today | Funding/infrastructure; dataset rights (licensing already in progress per `AMD_ROCM_ROADMAP.md`) | Acceptance criteria defined; canonical preflight passes with no blockers |
| 2–4 months: adapt/train + integrate candidate | Adapt/train a candidate behind the model-agnostic `ExtractionEngine` interface; measure accuracy/latency/memory; decide on optional CPU/AMD deployment path | Turns the platform into an end-to-end OCR product | GPU access; confirmed model license; hardware validation list (`HARDWARE_VALIDATION_TODO.md`) | Measured accuracy/latency/memory; fail-closed activation gate (`docs/MODEL_INTEGRATION.md` 6-step gate) |
| 4–6 months: select, validate, publish | Integrate only the selected validated engine; publish reproducible evaluation results; improve DOCX structure | Production credibility | Completed 2–4 month phase | Published reproducible results; improved DOCX structure |

External decisions tracked in `ROADMAP.md`: final model/architecture/GPU platform, commercial permissions for pending datasets, user-document consent policy (currently disabled), production auth provider, and the license for "Project B" code.

## 15. Technical risk register

Likelihood/impact are qualitative (Low/Medium/High), reflecting repository evidence — no probabilities are claimed.

| Risk | Likelihood | Impact | Mitigation (present or planned) | Evidence |
|---|---|---|---|---|
| Upstream model dependency (single leading candidate) | High | High | Model-agnostic engine/adapter layer; fail-closed activation gate; two adapters (Hunyuan, Qwen) already scaffolded | `docs/MODEL_INTEGRATION.md`, `docs/adapters/MULTIMODEL_ADAPTERS.md` |
| Model licensing unconfirmed (HunyuanOCR custom license) | High | High | Fail-closed rights model; `NEEDS_REVIEW` enforced by tests; alternatives benchmarked | `models.csv`, `UPSTREAM_COMPATIBILITY.md`, `tests/benchmarks/` |
| GPU availability (named binding constraint) | High | High | Planned AWS Spot-P plan with checkpoint upload/interruption handling; CPU-first architecture keeps evaluation unblocked | `README.md`, `docs/data_foundation/AWS_DEPLOYMENT_PLAN.md` |
| Benchmark contamination vs future training data | Medium | High | L0–L6 leakage gate exists in the quality tooling; **not yet documented as run on the benchmark cohort** | `clouda_data/quality/leakage.py`, §9 caveat |
| Arabic domain diversity (177-page cohort breadth) | Medium | Medium | Provenance-tracked multi-source cohort (14 datasets); factory can generate domain-targeted scans; larger cycle planned but undocumented | `source_summary.csv`, `docs/ROADMAP.md` |
| Performance variability across hardware | Medium | Medium | Cross-GPU timing caveat forbids speed claims; hardware-validation TODO enumerates unmeasured items | `RESULTS.md` hardware caveat, `HARDWARE_VALIDATION_TODO.md` |
| Document-layout complexity (no layout-aware extraction) | Medium | Medium | Trust gate routes risky layouts to review rather than emitting bad text; DOCX is text-only by design | `page_routing.py` layout-risk checks, `docx_export.py` |
| Training cost overrun | Medium | Medium | Planner produces VRAM/storage/cost estimates with assumptions ledger before any run | `clouda_training/planner/` |
| Deployment complexity at production scale | Medium | Medium | systemd deploy kit, doctor diagnostics, operations docs; worker concurrency=1 limits current scope | `deploy/linux/`, `docs/operations/` |
| Cross-platform behavioral drift | Low | Medium | Windows+Ubuntu CI on the same pinned dependency set; deterministic tests on both platforms | `.github/workflows/ci.yml` matrix |
| Licensing-compliance error in redistribution | Low | High | Repository scan forbids weights/datasets/DBs in-tree; SBOM; fail-closed export gate | `tools/validation/repository_scan.py`, `clouda_training/exporter.py` |

## 16. Capital-to-milestone mapping

**No budget figures, funding ask, or revenue numbers exist anywhere in the repository.** None are invented here. The only documented need is stated qualitatively and repeatedly: *"Model adaptation/training and runtime integration are the next major technical stage. Progress on that stage is currently limited primarily by access to suitable GPU compute."* (`README.md`; repeated in `ROADMAP.md`, `docs/ROADMAP.md`, `docs/MODEL_INTEGRATION.md`.)

| Documented need | Why technically necessary (from repo evidence) | Milestone it unlocks | Evidence |
|---|---|---|---|
| GPU compute access | Real training is structurally fail-closed; adapters are `gpu_validated=False`; every next-stage roadmap item names GPU access | Adaptation/training of a candidate (roadmap 2–4 months); hardware validation list closure | `README.md`, `docs/training/HARDWARE_VALIDATION_TODO.md` |
| Cloud evaluation environment | AWS deployment plan specifies Region us-east-2, Spot family P, current quota 8 vCPUs, expected test instance p3.2xlarge, with required work: checkpoint upload, spot-interruption handling, automatic shutdown, cost estimation, least-privilege IAM | Repeatable, interrupt-tolerant training runs | `docs/data_foundation/AWS_DEPLOYMENT_PLAN.md`, `tools/data_foundation/aws/` |
| Storage / throughput measurement | Hardware-validation docs list NVMe/HDD/network shard-streaming throughput as deliberately unmeasured; pretraining metadata can exceed single-machine memory budgets | Streaming-loader validation at real scale | `docs/training_data/HARDWARE_VALIDATION.md`, `docs/pretraining_dataset_infrastructure.md` |
| Licensing review completion | "Dataset licensing and written-permission verification are still in progress"; model licenses `NOT_CONFIRMED_IN_LOCAL_EVIDENCE` | Commercial-grade model adoption; redistribution decisions | `docs/AMD_ROCM_ROADMAP.md`, `models.csv` |

Expenditure-to-milestone mapping beyond the above would require figures the project has not published; the honest reading is that the platform spend to date is time (the repo itself), and the next tranche is compute plus licensing verification.

## 17. Reproducibility checklist (all commands verified)

Verified on 2026-09-20, Windows 10, Python 3.11.9, CPU-only, on the audit checkout (branch `docs/investor-github-hardening`; commands equally valid on `main`). Every command below was executed during this audit or is executed by the verified-green CI run of 2026-09-19.

1. **Clone**: `git clone https://github.com/sahrasayed3-crypto/clouda-ocr && cd clouda-ocr && git rev-parse HEAD` *(verified: fetch + SHA recorded)*
2. **Install (dev-lite, CPU)**: `python -m venv .venv && pip install -r requirements-dev.txt` — or the full CI-identical set: `pip install -c constraints/py311.txt -e ".[server,worker,data,training,models,test,dev]"` *(verified in CI step 1; local machine had the dependency set installed and functional)*
3. **Run the test suite (offline, CPU)**: `CLOUDA_LOCAL_OCR_ENABLED=false CLOUDA_NO_NETWORK_TESTS=true python -m pytest tests -q` *(executed during this audit — see §18 for the live result)*
4. **Environment doctor**: `python -m clouda_data.pipeline.cli doctor --deep` *(executed: all deep checks PASS — backend smoke, offline MockTrainer dry-run, canonical PDF self-test; exit 0. One machine-specific FAIL observed — free-disk-space threshold — demonstrating the check works.)*
5. **Demo smoke**: `python scripts/demo.py` *(executed: produces `outputs/demo/digital_text.docx` + `outputs/demo/page_statuses.json` with page states `direct_pdf_text`, `pending_ocr_model`, `blank_page`)*
6. **Benchmark integrity**: `python benchmarks/ocr_arabic/scripts/validate_release.py` *(executed: "OCR Arabic benchmark metadata validation: PASS"); independent re-hash: `python -c "import hashlib;print(hashlib.sha256(open('benchmarks/ocr_arabic/benchmark_manifest.jsonl','rb').read()).hexdigest())"` → must equal `2a499ed0…fb893` *(executed: match)*
7. **Inspect CI**: `.github/workflows/ci.yml` (16-step validate job + optional jobs); live status via the GitHub Actions tab — latest `main` run **success** (2026-09-19, re-checked via API 2026-09-20)
8. **Static counts** (if you distrust quoted numbers): `grep -r "def test_" tests --include="*.py" | wc -l` → 1,895; `find tests -name "*.py" | wc -l` → 202 *(executed)*
9. **Security scan**: `python -m tools.validation.repository_scan --root .` *(CI-enforced on every push; secrets/large-file/forbidden-path)*
10. **Full CI parity**: `ruff check .`, `black --check app.py pdfword clouda_contracts clouda_data clouda_models clouda_training tests scripts tools`, `mypy app.py pdfword clouda_contracts clouda_data clouda_training clouda_models tools`, `python -m build` *(all executed by the green CI run of the audited SHA)*

## 18. Live verification log (this audit, 2026-09-20)

| Command | Result |
|---|---|
| `git fetch origin` + `git log origin/main -1` | `45d0078e…` (2026-09-19) |
| GitHub API `actions/runs` check | latest `main` run: `completed / success` |
| `python benchmarks/ocr_arabic/scripts/validate_release.py` | `PASS` |
| Manifest re-hash (177 rows) | `2a499ed0268a5c583a1b174ef241e05affb9c4ba757593eb9ee3fc6f3e4fb893` — matches pinned value |
| `python scripts/demo.py` | DOCX + page statuses produced; routes `direct_pdf_text` / `pending_ocr_model` / `blank_page` |
| `python -m clouda_data.pipeline.cli doctor --deep` | Deep checks PASS; overall PARTIALLY_READY solely due to machine-specific disk-space threshold; exit 0 |
| `python -m pytest tests -q` (offline flags, CPU-only) | **2,071 passed, 15 skipped, 0 failed** in 648.64 s |
| Static counts | 202 test files; 1,895 test functions |

Live pytest result (2026-09-20, audit machine): **2,071 passed, 15 skipped, 0 failed, 17 warnings, 648.64 s** — zero failures on the full offline suite. The 15 skips are consistent with the suite's env-gated skip design (e.g. the CUDA smoke test requires `CLOUDA_CUDA_SMOKE`) on a GPU-less CPU-only Windows machine.

## 19. Investor technical Q&A

**Is this a wrapper?** No — the wrapper accusation applies when the only value is forwarding pages to a model API. Clouda's runtime never forwards anything today: it performs page-level evidence analysis, a 12-check trusted-text gate, digest-bound extraction, categorical OCR self-review with render-identity-locked selective re-read, and RTL DOCX export — all tested, none of which a hosted OCR API provides. That said, the honest corollary (§4, §13): the OCR model itself is not Clouda's yet, so the *current* end-to-end value for scanned pages is pending the model stage.

**Why Arabic-first?** The pipeline has Arabic-specific engineering that general-purpose tools lack: isolated-character and pathological-spacing rejection checks, diacritic-aware normalization as the primary benchmark metric, RTL DOCX (`w:bidi`/`w:rtl`, Arabic page-number footers), Arabic-specific synthetic degradations (`faint_diacritics`, `merged_strokes`), and a dedicated Arabic benchmark. The repo does not claim a market-size thesis in writing; the technical focus is demonstrable.

**Why not just use a cloud OCR API?** Documented posture: self-hosted/offline orientation, loopback-only services, no model downloads at runtime, user documents never used for training by default (two-gate consent), and per-page review states instead of opaque API confidence. For documents where data control matters, the architecture is built for local processing; the trade-off is that scanned-page quality awaits the model stage.

**What is proprietary?** By the README's own scope statement: training data, private reference texts, final model weights/adapters/checkpoints, production service code, private training recipes, deployment configuration, and local permission evidence — explicitly excluded from the Apache-2.0 grant and "may be licensed, hosted, or distributed separately."

**What is open source?** Everything in the repo: application structure, model-agnostic interfaces, engine registry, page routing, quality/review workflow, evaluation utilities, the deterministic data factory, quality gate, benchmark metadata, tests, and documentation — Apache-2.0 (original code), fonts OFL 1.1.

**What is trained by Clouda?** Nothing yet. The training framework executes deterministic mock runs only; a real PyTorch backend exists but has only been proven on a synthetic adapter.

**What is not trained yet?** Any OCR model. HunyuanOCR-1.5 and Qwen3-VL exist as fail-closed SFT adapters and benchmark candidates (`real_weights_validated=False`, `gpu_validated=False`).

**How do you validate quality?** Three layers: the per-page trust gate and categorical self-review in the runtime; the deterministic CER/WER evaluation stack (`clouda_data/evaluation/`); and the frozen 177-page Arabic benchmark with pinned manifest hash and published methodology. No user-facing confidence percentages are exposed by design.

**How do you prevent benchmark leakage?** The data platform implements L0–L6 cross-split leakage detection where any CRITICAL finding fails the gate, plus derived-manifest exclusion planning. Honest caveat: this has not been documented as applied between the benchmark cohort and future training splits — that gate must be run before adaptation results are comparable (§9, §15).

**How is security handled?** See §6: fail-closed storage roots, worker claim-token fencing with two-phase finalization, loopback-only Lab, bounded uploads, archive/XML/image validation, secret scanning + Bandit + pip-audit in CI, an adversarial-hardening test suite including a Hypothesis property test on path containment — and explicitly no external audit yet.

**What is the next technical milestone?** Secure GPU compute and complete licensing verification, then adapt/integrate the leading candidate behind the existing `ExtractionEngine` contract, with measured accuracy/latency/memory and the documented 6-step activation gate (§14).

**What does funding unlock?** Per the repo's own repeated statement: GPU access is the primary constraint on the entire next stage (adaptation, training, hardware validation, runtime integration). The AWS plan (Spot P-family, checkpointing, interruption handling) shows the spend is planned to be interrupt-tolerant. No budget figures are published, so per-item capital mapping beyond §16 is not possible from evidence.

## 20. Appendix — verification commands

```bash
# Identity
git clone https://github.com/sahrasayed3-crypto/clouda-ocr && cd clouda-ocr
git rev-parse HEAD                     # expect 45d0078e551267535f85c79badd432fdc8514bef on main

# Install (CPU, offline-safe)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Tests (offline, matches CI flags)
CLOUDA_LOCAL_OCR_ENABLED=false CLOUDA_NO_NETWORK_TESTS=true python -m pytest tests -q

# Environment doctor (deep = offline release self-test)
python -m clouda_data.pipeline.cli doctor --deep

# Demo smoke (no network, no model)
python scripts/demo.py                 # → outputs/demo/digital_text.docx + page_statuses.json

# Benchmark integrity
python benchmarks/ocr_arabic/scripts/validate_release.py
python -c "import hashlib;print(hashlib.sha256(open('benchmarks/ocr_arabic/benchmark_manifest.jsonl','rb').read()).hexdigest())"
# expect: 2a499ed0268a5c583a1b174ef241e05affb9c4ba757593eb9ee3fc6f3e4fb893

# Security scans (as run in CI)
python -m tools.validation.repository_scan --root .
python -m bandit -c pyproject.toml -r pdfword tools scripts -ll -q
python -m pip_audit -r constraints/py311.txt

# Quality gates (as run in CI)
python -m ruff check .
black --check app.py pdfword clouda_contracts clouda_data clouda_models clouda_training tests scripts tools
python -m mypy app.py pdfword clouda_contracts clouda_data clouda_training clouda_models tools --ignore-missing-imports --no-incremental

# Static test counts
grep -r "def test_" tests --include="*.py" | wc -l    # 1,895 at audited SHA
find tests -name "*.py" | wc -l                       # 202 at audited SHA
```

---

*This document was generated from read-only inspection plus CPU-only local verification. It contains no invented results, customers, revenue, partnerships, certifications, or dates. Claims it cannot prove are qualified or omitted. Regenerate counts and re-run §18 commands against a fresh clone before circulating, since static counts drift with the tree.*
