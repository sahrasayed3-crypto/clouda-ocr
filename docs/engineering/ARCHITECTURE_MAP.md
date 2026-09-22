# Architecture Map (derived from code, session of 2026-09-21)

Root: `F:\PROJECT\CLOUDA_UNIFIED_PROJECT`, package `clouda-pdf` v0.2.0. Arabic-first PDF→DOCX runtime, single-machine Windows deployment.

## Subsystems

| Package | Role | Notes |
|---|---|---|
| `pdfword/` | Product runtime: FastAPI server, RQ worker, Streamlit UI components, SQLite, auth, OCR routing, DOCX export | ~22.5k LOC. Two dormant subsystems: `key_router/` (multi-provider key routing) and `teacher_pipeline/` |
| `clouda_contracts/` | Dependency-light shared contracts: `storage.py` (StorageRoots), checksums, manifests, archive security, queues, evaluation policy | 1.3k LOC |
| `clouda_data/` | Dataset construction: ingestion → factory render/distort → pretraining → quality gate → evaluation → results store | ~30k LOC; largest subsystem |
| `clouda_models/` | Model metadata registry (JSON records), provider protocol, checkpoint resolution | 195 LOC; no weights |
| `clouda_training/` | Training planning/export/resume: planner, adapters, runtime (torch backend, checkpoints), hunyuan/qwen export | ~11.7k LOC |
| `clouda_lab/` | Local analysis + loopback-only FastAPI dashboard; error analysis, active learning, training orchestration | ~8.5k LOC |
| `app.py` | Streamlit Arabic UI; thin client over FastAPI via `_api_request` (`app.py:103`) | |
| `tools/` | Operator scripts: benchmarks, synthetic acceptance, env configuration, poppler binaries | |
| `configs/`, `schemas/`, `dataset_catalog/`, `benchmarks/` | Config, JSON schemas, dataset source catalog, local benchmark scripts | |

## Entry points

- Console scripts (`pyproject.toml:87-92`):
  - `clouda-data` → `clouda_data/pipeline/cli.py:main` (argparse hub ~1368 lines: inspect/ingest, dataset sources/download, render/distort, factory-*, dataset-*, results-*, doctor)
  - `clouda-quality` → `clouda_data/quality/cli.py` (scan/verify/report/clean/cluster, exit 0/1/2 contract)
  - `clouda-lab` → `clouda_lab/cli.py:main` (serve loopback-only + analysis subcommands)
  - `clouda-training` → `clouda_training/cli.py:main` (plan/export/validate/split/runs/resume/checkpoints + hunyuan sub-CLI)
- FastAPI: `pdfword/worker_api.py:118` (main server, ~60 routes, TrustedHostMiddleware, maintenance thread in `_lifespan`); `clouda_lab/dashboard/app.py:96` (`create_app`, loopback-gated).
- Streamlit: `app.py` (port 8501).
- Worker: `pdfword/worker.py:main` (requires `APP_ROLE=worker`, Windows `SimpleWorker` at `worker.py:82`, job callable `worker_tasks.run_remote_job`).
- Launch: `start_clouda_all.ps1` starts api (uvicorn 127.0.0.1:8000) + ui (8501) + worker; sets `CLOUDA_STATE_HOME=_state`, `LOCAL_PROCESSING_ENABLED=false`, PATH prepend for vendored poppler.

## Core data flow (PDF → DOCX)

1. Upload: Streamlit → `/user/documents/upload` (`worker_api.py:1195`): extension/streaming-size/magic/page-count caps, `TenantStorage` under server-generated uuid name, `conversions` row in SQLite.
2. Dispatch: `_dispatch_conversion_job` (`worker_api.py:1116`) — local in-process processing is hard-rejected; always RQ enqueue via `job_queue.py:84-130`.
3. Worker: RQ → `run_remote_job` → `conversion_service.execute_worker_conversion` (`conversion_service.py:184`), cost-budget guards, then `process_pdf` (`ocr_pipeline.py:225`).
4. Routing: `page_routing.analyze_pdf_page` → digital-text gate → `decide_page_route` (`page_routing.py:876`): `NO_EXTRACTION` / `MANUAL_REVIEW` / `PENDING_OCR_MODEL` / `DIRECT_PDF_TEXT`. Engines (`engines.py`): `DirectPdfTextEngine`, `FutureOcrEngine` (placeholder always returns `OCR_STATUS_PENDING_MODEL` — no approved OCR model selected yet), `FeatureFlaggedLocalModelEngine`. Cloud OCR args are intentionally discarded in `process_pdf` (`ocr_pipeline.py:253-255`); scanned pages park as "requires future OCR".
5. Correction: enabled DB correction rules applied by string replace; `markdown_to_docx` (`docx_export.py`, RTL-aware) → DOCX bytes → `POST /internal/jobs/{id}/result`; server publishes via unique `.docx.part` + fsync + `os.replace` (`worker_api.py:2197-2258`).

## Persistence

- State roots: `clouda_contracts/storage.py:77` `StorageRoots.from_env()` (`CLOUDA_STATE_HOME`, default `~/.clouda_pdf_word`; launch scripts use `_state/`).
- SQLite: `pdfword/database.py:97` (users/sessions/conversions/attempts/quotas/guest/corrections); `key_router/repository.py`; `clouda_data/storage/manifest_store.py`; `clouda_data/distortion/checkpoints.py`. Backups via online-backup API (`pdfword/backup.py`).
- Secrets: `pdfword/constants.py:16` `SECRETS_DIR` (`openrouter_api_key.txt`, `ocr_learning_store.json`, `model_router_store.json`).
- Registries/manifests: model registry JSON, dataset catalog, ingestion manifests (schema-validated), pretraining manifest/handoff, factory run manifests, quality run state + clean manifests, results store.

## Network boundaries

- External: OpenRouter (`openrouter_client.py:121,300`; `provider_client.py` OpenAI-compatible providers), Firebase auth (`auth.py`: local / emulator / staging modes), HF dataset downloads (`clouda_data/factory/ingest/hf_datasets.py`).
- Loopback: Streamlit→API, worker→API job protocol (`worker_client.py`), Lab dashboard (bind enforced loopback, `require_loopback` IP check), local OCR HTTP provider (loopback-gated, `local_ocr_adapters.py:131-137`).

## Optional-dependency boundaries (lazy imports)

- redis/rq: `worker.py:40-41`, `job_queue.py:95-96`, `worker_api.py:653` (tests use fakeredis).
- torch: `ocr_self_review.py:535`, `doctor/training.py:316`, `training_data/torch_adapter.py`, `clouda_training/runtime/` backends.
- fastapi/uvicorn: module-level in `worker_api.py`; lazy in `clouda_lab/cli.py:44-52`.
- weasyprint/numpy/cv2/img2pdf/pikepdf: factory extras, imported inside backends.
- transformers: inside `TransformersVisionLanguageProvider._load` (`local_ocr_adapters.py:359`).

## clouda_data pipeline DAG (from CLI)

Source catalog → ingest (schema-validated manifests) → pretraining build (register→scan→validate→normalize→dedupe→split[seeded, holdout]→export) → [optional handoff] → Data Factory (render backends → distort profiles w/ sqlite checkpoints → ground truth → run manifest) → quality gate (dup/fingerprint/leakage checks → PASS/PASS_WITH_WARNINGS/FAIL + clean manifest + quarantine) → evaluation (CER/WER) → results store → training handoff (clouda_training export/pack). `doctor` reports stage readiness throughout.
