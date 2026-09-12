from __future__ import annotations

import hmac
import secrets
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .catalog import DatasetCatalog
from .observability import ObservabilityService
from .security import browser_safe, require_loopback
from .settings import LabSettings
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
    model_id: str | None = None
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
    training = TrainingService(resolved, catalog)
    observability = ObservabilityService(resolved, catalog, training)
    app.state.lab_catalog = catalog
    app.state.lab_training = training
    app.state.lab_observability = observability
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
        }

    @app.get("/api/lab/session")
    def session(request: Request) -> dict[str, str]:
        return {"action_token": request.app.state.lab_action_token}

    @app.get("/api/lab/overview")
    def overview() -> dict[str, Any]:
        return observability.overview()

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
        return catalog.run_quality(
            payload.dataset_id, max_samples=payload.max_samples, duplicates_only=False
        )

    @app.post(
        "/api/lab/quality/duplicates", dependencies=[Depends(require_action_token)]
    )
    def quality_duplicates(payload: QualityRequest) -> dict[str, Any]:
        return catalog.run_quality(
            payload.dataset_id, max_samples=payload.max_samples, duplicates_only=True
        )

    @app.post("/api/lab/quality/derive", dependencies=[Depends(require_action_token)])
    def quality_derive(payload: DeriveRequest) -> dict[str, Any]:
        return catalog.derive(payload.dataset_id, payload.output_label)

    @app.get("/api/lab/models")
    def models(limit: int = Query(default=100, ge=1, le=200)) -> dict[str, Any]:
        return {"models": training.list_models()[:limit]}

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

    @app.get("/api/lab/results")
    def results(
        model: str | None = None,
        dataset: str | None = None,
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

    @app.get("/api/lab/doctor/latest")
    def doctor_latest() -> dict[str, Any]:
        return observability.latest_doctor()

    @app.post("/api/lab/doctor/run", dependencies=[Depends(require_action_token)])
    def doctor_run(payload: DoctorRequest) -> dict[str, Any]:
        return observability.run_doctor(deep=payload.deep)

    @app.get("/api/lab/hardware")
    def hardware() -> dict[str, Any]:
        return observability.hardware()

    @app.get("/lab", include_in_schema=False)
    @app.get("/lab/{page:path}", include_in_schema=False)
    def lab_shell(page: str = ""):
        return FileResponse(_STATIC / "index.html")

    return app


app = create_app()

__all__ = ["app", "create_app"]
