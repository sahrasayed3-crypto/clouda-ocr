from __future__ import annotations

import hmac
import secrets
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .benchmarks import BenchmarkWorkspaceService
from .catalog import DatasetCatalog
from .datasets import DatasetOperationsService
from .document_intelligence import DocumentIntelligenceService, MAX_PDF_BYTES
from .models import ModelCatalogService
from .observability import ObservabilityService
from .security import browser_safe, require_loopback
from .settings import LabSettings
from .storage import StorageService
from .tasks import OperationTaskService
from .training import TrainingService

_STATIC = Path(__file__).with_name("static")
_CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; "
    "script-src 'self'; connect-src 'self'; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
)


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QualityRequest(StrictRequest):
    dataset_id: str
    max_samples: int = Field(default=0, ge=0, le=5_000)


class DeriveRequest(StrictRequest):
    dataset_id: str
    output_label: str


class PlanRequest(StrictRequest):
    experiment_name: str | None = None
    adapter_type: str
    dataset_id: str
    precision: str = "bf16"
    seed: int = 20260723
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    epochs: int = Field(default=1, gt=0)
    max_steps: int = Field(default=100, gt=0)
    learning_rate: float = Field(default=5e-5, gt=0)
    checkpoint_frequency: int = Field(default=100, gt=0)


class PreflightRequest(StrictRequest):
    plan_id: str
    write_probe: bool = False


class DoctorRequest(StrictRequest):
    deep: bool = False


class EmptyRequest(StrictRequest):
    pass


class ConfirmationRequest(StrictRequest):
    plan_id: str
    confirmation: str


class ModelAssetRequest(StrictRequest):
    asset_id: str


class BenchmarkPlanRequest(StrictRequest):
    model_ids: list[str] = Field(min_length=1, max_length=8)


def create_app(settings: LabSettings | None = None) -> FastAPI:
    resolved = settings or LabSettings.from_repo(Path.cwd())
    app = FastAPI(
        title="Clouda Lab",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        dependencies=[Depends(require_loopback)],
    )
    app.state.lab_settings = resolved
    catalog = DatasetCatalog(resolved)
    tasks = OperationTaskService(
        resolved.tasks_root, browser_roots=(resolved.repo_root,), max_workers=2
    )
    dataset_operations = DatasetOperationsService(resolved, tasks)
    model_catalog = ModelCatalogService(resolved, tasks)
    training = TrainingService(resolved, catalog, tasks=tasks, models=model_catalog)
    storage = StorageService(resolved, tasks=tasks)
    benchmark_workspace = BenchmarkWorkspaceService(resolved, model_catalog)
    observability = ObservabilityService(
        resolved, catalog, training, storage=storage, tasks=tasks
    )
    document_intelligence = DocumentIntelligenceService(resolved)
    app.state.lab_catalog = catalog
    app.state.lab_tasks = tasks
    app.state.lab_dataset_operations = dataset_operations
    app.state.lab_model_catalog = model_catalog
    app.state.lab_training = training
    app.state.lab_storage = storage
    app.state.lab_benchmark_workspace = benchmark_workspace
    app.state.lab_observability = observability
    app.state.lab_document_intelligence = document_intelligence
    app.state.lab_action_token = secrets.token_urlsafe(32)

    def require_action_token(
        request: Request,
        supplied: str | None = Header(default=None, alias="X-Clouda-Lab-Action"),
    ) -> None:
        expected = request.app.state.lab_action_token
        if supplied is None or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=403, detail="Local action token required")

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = _CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Exception)
    async def sanitized_error(_request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": browser_safe(
                    f"{type(exc).__name__}: {exc}", (resolved.repo_root,)
                ),
            },
        )

    @app.exception_handler(KeyError)
    async def not_found(_request, exc: KeyError):
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "detail": browser_safe(str(exc), (resolved.repo_root,)),
            },
        )

    @app.exception_handler(PermissionError)
    async def protection_error(_request, exc: PermissionError):
        return JSONResponse(
            status_code=409,
            content={
                "error": "protected",
                "detail": browser_safe(str(exc), (resolved.repo_root,)),
            },
        )

    @app.exception_handler(ValueError)
    async def invalid_request(_request, exc: ValueError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_request",
                "detail": browser_safe(str(exc), (resolved.repo_root,)),
            },
        )

    app.mount("/lab/assets", StaticFiles(directory=_STATIC), name="lab-assets")

    @app.get("/api/lab/offline")
    def offline_status() -> dict[str, object]:
        return {
            "schema_version": "clouda.lab.offline.v1",
            "offline": True,
            "network_required": False,
            "automatic_model_download": False,
            "automatic_dataset_download": False,
            "remote_provider_calls": False,
            "binding": "loopback-only" if resolved.local_only else "configured",
        }

    @app.get("/api/lab/session")
    def session(request: Request) -> dict[str, str]:
        return {"action_token": request.app.state.lab_action_token}

    @app.post(
        "/api/lab/document-intelligence/analyze",
        dependencies=[Depends(require_action_token)],
    )
    async def analyze_document_intelligence(
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        try:
            payload = await file.read(MAX_PDF_BYTES + 1)
        finally:
            await file.close()
        return document_intelligence.analyze(payload)

    @app.get("/api/lab/overview")
    def overview() -> dict[str, Any]:
        return observability.overview()

    @app.get("/api/lab/tasks")
    def operation_tasks(
        kind: str | None = None,
        limit: int = Query(default=200, ge=1, le=500),
    ) -> dict[str, Any]:
        return {"tasks": tasks.list_tasks(kind=kind)[:limit]}

    @app.get("/api/lab/tasks/{task_id}")
    def operation_task(task_id: str) -> dict[str, Any]:
        return tasks.get_task(task_id)

    @app.post(
        "/api/lab/tasks/{task_id}/cancel",
        dependencies=[Depends(require_action_token)],
    )
    def cancel_operation(task_id: str, _payload: EmptyRequest) -> dict[str, Any]:
        return tasks.cancel(task_id)

    @app.get("/api/lab/downloads")
    def downloads(limit: int = Query(default=200, ge=1, le=500)) -> dict[str, Any]:
        return {
            "downloads": tasks.list_tasks(kind="DATASET_DOWNLOAD")[:limit],
            "automatic_downloads": False,
        }

    @app.get("/api/lab/sources")
    def operational_sources() -> dict[str, Any]:
        return {"sources": dataset_operations.list_sources()}

    @app.get("/api/lab/sources/{source_id}")
    def operational_source(source_id: str) -> dict[str, Any]:
        return dataset_operations.source_detail(source_id)

    @app.post(
        "/api/lab/sources/{source_id}/download-plan",
        dependencies=[Depends(require_action_token)],
    )
    def dataset_download_plan(source_id: str, _payload: EmptyRequest) -> dict[str, Any]:
        return dataset_operations.create_download_plan(source_id)

    @app.post(
        "/api/lab/dataset-downloads",
        dependencies=[Depends(require_action_token)],
    )
    def start_dataset_download(payload: ConfirmationRequest) -> dict[str, Any]:
        return dataset_operations.start_download(payload.plan_id, payload.confirmation)

    @app.post(
        "/api/lab/sources/{source_id}/verify",
        dependencies=[Depends(require_action_token)],
    )
    def verify_dataset_source(source_id: str, _payload: EmptyRequest) -> dict[str, Any]:
        return dataset_operations.verify_source(source_id)

    @app.post(
        "/api/lab/sources/{source_id}/removal-plan",
        dependencies=[Depends(require_action_token)],
    )
    def dataset_removal_plan(source_id: str, _payload: EmptyRequest) -> dict[str, Any]:
        return dataset_operations.create_removal_plan(source_id)

    @app.post(
        "/api/lab/dataset-removals",
        dependencies=[Depends(require_action_token)],
    )
    def remove_dataset_download(payload: ConfirmationRequest) -> dict[str, Any]:
        return dataset_operations.remove_download(payload.plan_id, payload.confirmation)

    @app.get("/api/lab/imports")
    def imports() -> dict[str, Any]:
        return {"imports": dataset_operations.list_imports()}

    @app.post(
        "/api/lab/imports/{import_id}/validate",
        dependencies=[Depends(require_action_token)],
    )
    def validate_import(import_id: str, _payload: EmptyRequest) -> dict[str, Any]:
        return dataset_operations.validate_import(import_id)

    @app.post(
        "/api/lab/imports/{import_id}/register",
        dependencies=[Depends(require_action_token)],
    )
    def register_import(import_id: str, _payload: EmptyRequest) -> dict[str, Any]:
        return dataset_operations.register_import(import_id)

    @app.get("/api/lab/datasets")
    def datasets(limit: int = Query(default=200, ge=1, le=500)) -> dict[str, Any]:
        return {"datasets": catalog.list_datasets()[:limit]}

    @app.get("/api/lab/dataset-sources")
    def dataset_sources(
        limit: int = Query(default=200, ge=1, le=500),
    ) -> dict[str, Any]:
        return {"sources": catalog.list_sources()[:limit]}

    @app.get("/api/lab/datasets/{dataset_id}")
    def dataset_detail(dataset_id: str) -> dict[str, Any]:
        return catalog.get_dataset(dataset_id)

    @app.get("/api/lab/datasets/{dataset_id}/preview")
    def dataset_preview(
        dataset_id: str, limit: int = Query(default=10, ge=1, le=100)
    ) -> dict[str, Any]:
        return catalog.preview(dataset_id, limit)

    @app.get("/api/lab/quality")
    def quality(dataset_id: str) -> dict[str, Any]:
        catalog.get_dataset(dataset_id)
        return catalog.quality_summary(dataset_id)

    @app.post("/api/lab/quality/check", dependencies=[Depends(require_action_token)])
    def quality_check(payload: QualityRequest) -> dict[str, Any]:
        catalog.get_dataset(payload.dataset_id)

        def worker(context):
            context.update(phase="ANALYZING", detail="Running canonical quality gate")
            return catalog.run_quality(
                payload.dataset_id,
                max_samples=payload.max_samples,
                duplicates_only=False,
            )

        return tasks.enqueue("DATASET_QUALITY", payload.dataset_id, worker)

    @app.post(
        "/api/lab/quality/duplicates", dependencies=[Depends(require_action_token)]
    )
    def quality_duplicates(payload: QualityRequest) -> dict[str, Any]:
        catalog.get_dataset(payload.dataset_id)

        def worker(context):
            context.update(phase="DEDUPLICATING", detail="Running canonical dedup")
            return catalog.run_quality(
                payload.dataset_id,
                max_samples=payload.max_samples,
                duplicates_only=True,
            )

        return tasks.enqueue("DATASET_DEDUP", payload.dataset_id, worker)

    @app.post("/api/lab/quality/derive", dependencies=[Depends(require_action_token)])
    def quality_derive(payload: DeriveRequest) -> dict[str, Any]:
        catalog.get_dataset(payload.dataset_id)

        def worker(context):
            context.update(
                phase="DERIVING", detail="Creating immutable canonical derived dataset"
            )
            return catalog.derive(payload.dataset_id, payload.output_label)

        return tasks.enqueue(
            "DATASET_DERIVE",
            payload.dataset_id,
            worker,
            metadata={"output_label": payload.output_label},
        )

    @app.get("/api/lab/models")
    def models(limit: int = Query(default=100, ge=1, le=200)) -> dict[str, Any]:
        return {"models": training.list_models()[:limit]}

    @app.get("/api/lab/model-catalog")
    def published_model_catalog() -> dict[str, Any]:
        return {"models": model_catalog.list_models()}

    @app.get("/api/lab/model-catalog/{catalog_id}")
    def published_model_detail(catalog_id: str) -> dict[str, Any]:
        return model_catalog.get_model(catalog_id)

    @app.post(
        "/api/lab/model-catalog/{catalog_id}/assets",
        dependencies=[Depends(require_action_token)],
    )
    def configure_model_assets(
        catalog_id: str, payload: ModelAssetRequest
    ) -> dict[str, Any]:
        return model_catalog.configure_assets(catalog_id, payload.asset_id)

    @app.post(
        "/api/lab/model-catalog/{catalog_id}/verify",
        dependencies=[Depends(require_action_token)],
    )
    def verify_model_assets(catalog_id: str, _payload: EmptyRequest) -> dict[str, Any]:
        return model_catalog.verify_assets(catalog_id)

    @app.post(
        "/api/lab/model-catalog/{catalog_id}/removal-plan",
        dependencies=[Depends(require_action_token)],
    )
    def model_removal_plan(catalog_id: str, _payload: EmptyRequest) -> dict[str, Any]:
        return model_catalog.create_removal_plan(catalog_id)

    @app.post(
        "/api/lab/model-removals",
        dependencies=[Depends(require_action_token)],
    )
    def remove_model_assets(payload: ConfirmationRequest) -> dict[str, Any]:
        return model_catalog.remove_assets(payload.plan_id, payload.confirmation)

    @app.get("/api/lab/storage")
    def storage_status() -> dict[str, Any]:
        return storage.status()

    @app.get("/api/lab/planner/options")
    def planner_options() -> dict[str, Any]:
        return training.planner_options()

    @app.get("/api/lab/plans")
    def plans(limit: int = Query(default=200, ge=1, le=500)) -> dict[str, Any]:
        return {"plans": training.list_plans()[:limit]}

    @app.post("/api/lab/plans", dependencies=[Depends(require_action_token)])
    def create_plan(payload: PlanRequest) -> dict[str, Any]:
        return training.create_plan(payload.model_dump(exclude_none=True))

    @app.get("/api/lab/plans/{plan_id}")
    def plan_detail(plan_id: str) -> dict[str, Any]:
        return training.get_plan(plan_id)

    @app.post("/api/lab/preflight", dependencies=[Depends(require_action_token)])
    def preflight(payload: PreflightRequest) -> dict[str, Any]:
        return training.run_preflight(payload.plan_id, write_probe=payload.write_probe)

    @app.post(
        "/api/lab/training/{plan_id}/start-plan",
        dependencies=[Depends(require_action_token)],
    )
    def training_start_plan(plan_id: str, _payload: EmptyRequest) -> dict[str, Any]:
        return training.create_start_plan(plan_id)

    @app.post(
        "/api/lab/training-starts",
        dependencies=[Depends(require_action_token)],
    )
    def start_training(payload: ConfirmationRequest) -> dict[str, Any]:
        return training.confirm_start(payload.plan_id, payload.confirmation)

    @app.get("/api/lab/training/capabilities")
    def training_capabilities() -> dict[str, Any]:
        return {"stop": training.stop_capability()}

    @app.get("/api/lab/runs")
    def runs(limit: int = Query(default=200, ge=1, le=500)) -> dict[str, Any]:
        return {"runs": training.list_runs()[:limit]}

    @app.get("/api/lab/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        return training.get_run(run_id)

    @app.get("/api/lab/runs/{run_id}/checkpoints")
    def checkpoints(run_id: str) -> dict[str, Any]:
        return {"checkpoints": training.list_checkpoints(run_id)}

    @app.post(
        "/api/lab/runs/{run_id}/resume-check",
        dependencies=[Depends(require_action_token)],
    )
    def resume_check(run_id: str) -> dict[str, Any]:
        return training.resume_check(run_id)

    @app.post(
        "/api/lab/runs/{run_id}/resume",
        dependencies=[Depends(require_action_token)],
    )
    def resume_training(run_id: str, _payload: EmptyRequest) -> dict[str, Any]:
        return training.resume_training(run_id)

    @app.get("/api/lab/results")
    def results(
        model: str | None = None,
        dataset: str | None = None,
        run_type: str | None = None,
        experiment: str | None = None,
        benchmark: str | None = None,
        date: str | None = None,
        status: str | None = None,
        limit: int = Query(default=200, ge=1, le=500),
    ) -> dict[str, Any]:
        return observability.results(
            {
                "model": model,
                "dataset": dataset,
                "run_type": run_type,
                "experiment": experiment,
                "benchmark": benchmark,
                "date": date,
                "status": status,
                "limit": str(limit),
            }
        )

    @app.get("/api/lab/results/{run_id}")
    def result_detail(
        run_id: str, metric_limit: int = Query(default=200, ge=1, le=500)
    ) -> dict[str, Any]:
        return observability.result_detail(run_id, metric_limit=metric_limit)

    @app.get("/api/lab/benchmarks")
    def benchmarks(
        model: str | None = None,
        status: str | None = None,
        limit: int = Query(default=200, ge=1, le=500),
    ) -> dict[str, Any]:
        return observability.benchmarks(
            {"model": model, "status": status, "limit": str(limit)}
        )

    @app.get("/api/lab/benchmark-workspace")
    def benchmark_inventory() -> dict[str, Any]:
        return benchmark_workspace.inventory()

    @app.get("/api/lab/benchmark-results")
    def benchmark_results() -> dict[str, Any]:
        return benchmark_workspace.results()

    @app.post(
        "/api/lab/benchmark-plans",
        dependencies=[Depends(require_action_token)],
    )
    def create_benchmark_plan(payload: BenchmarkPlanRequest) -> dict[str, Any]:
        return benchmark_workspace.create_plan(payload.model_dump())

    @app.get("/api/lab/benchmark-plans/{plan_id}")
    def benchmark_plan(plan_id: str) -> dict[str, Any]:
        return benchmark_workspace.get_plan(plan_id)

    @app.get("/api/lab/benchmark-comparison")
    def benchmark_comparison(left: str, right: str) -> dict[str, Any]:
        return benchmark_workspace.compare(left, right)

    @app.get("/api/lab/doctor/latest")
    def doctor_latest() -> dict[str, Any]:
        return observability.latest_doctor()

    @app.post("/api/lab/doctor/run", dependencies=[Depends(require_action_token)])
    def doctor_run(payload: DoctorRequest) -> dict[str, Any]:
        return observability.run_doctor(deep=payload.deep)

    @app.get("/api/lab/hardware")
    def hardware() -> dict[str, Any]:
        return observability.hardware()

    @app.post(
        "/api/lab/hardware/validate",
        dependencies=[Depends(require_action_token)],
    )
    def validate_hardware(_payload: EmptyRequest) -> dict[str, Any]:
        return observability.validate_hardware()

    @app.get("/lab", include_in_schema=False)
    @app.get("/lab/{page:path}", include_in_schema=False)
    def lab_shell(page: str = ""):
        return FileResponse(_STATIC / "index.html")

    return app


app = create_app()

__all__ = ["app", "create_app"]
