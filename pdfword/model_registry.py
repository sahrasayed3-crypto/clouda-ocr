from dataclasses import dataclass
from contextlib import closing

from .database import Database
from .openrouter_client import discover_openrouter_models


def _price(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class RegisteredModel:
    id: str
    name: str
    provider: str
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    supports_vision: bool
    context_length: int
    prompt_price: float
    completion_price: float
    image_price: float
    is_free: bool
    available: bool
    updated_at: str
    failures: int
    average_speed: float
    average_quality: float
    category: str
    exclusion_reason: str | None = None


class ModelRegistry:
    def __init__(self, database: Database | None = None) -> None:
        self.database = database or Database()

    def refresh(self, force: bool = False) -> list[RegisteredModel]:
        raw_models = discover_openrouter_models(force=force)
        performance = self._performance()
        registered: list[RegisteredModel] = []
        for raw in raw_models:
            model_id = raw["id"]
            prompt = _price(raw.get("prompt_price"))
            completion = _price(raw.get("completion_price"))
            image = _price(raw.get("image_price"))
            vision = bool(raw.get("supports_vision"))
            pricing_known = (
                prompt is not None and completion is not None and image is not None
            )
            stats = performance.get(model_id, {})
            free = pricing_known and prompt == 0 and completion == 0 and image == 0
            lowered = f"{model_id} {raw.get('name') or ''}".lower()
            unsuitable = next(
                (
                    reason
                    for keyword, reason in {
                        "content-safety": "content safety model, not OCR",
                        "content safety": "content safety model, not OCR",
                        "moderation": "moderation model, not OCR",
                        "embedding": "embedding model, not generative OCR",
                        "guard": "guard model, not OCR",
                        "lyria": "audio generation model, not OCR",
                        "clip-preview": "media model, not OCR transcription",
                    }.items()
                    if keyword in lowered
                ),
                None,
            )
            if unsuitable:
                category, exclusion = "excluded", unsuitable
            elif not vision:
                category, exclusion = "excluded", "does not support image input"
            elif not pricing_known:
                category, exclusion = "excluded", "pricing metadata is unavailable"
            elif free:
                category, exclusion = "free_vision", None
            elif ((prompt or 0.0) + (completion or 0.0)) <= 0.00001:
                category, exclusion = "low_cost_vision", None
            else:
                category, exclusion = "high_quality_paid_vision", None
            registered.append(
                RegisteredModel(
                    id=model_id,
                    name=raw.get("name") or model_id,
                    provider=(
                        model_id.split("/", 1)[0] if "/" in model_id else "unknown"
                    ),
                    input_modalities=tuple(
                        raw.get("input_modalities")
                        or (["text", "image"] if vision else ["text"])
                    ),
                    output_modalities=tuple(raw.get("output_modalities") or ["text"]),
                    supports_vision=vision,
                    context_length=int(raw.get("context_length") or 0),
                    prompt_price=prompt or 0.0,
                    completion_price=completion or 0.0,
                    image_price=image or 0.0,
                    is_free=free,
                    available=True,
                    updated_at=raw.get("updated_at") or "",
                    failures=int(stats.get("failures") or 0),
                    average_speed=float(stats.get("average_speed") or 0),
                    average_quality=float(stats.get("average_quality") or 0),
                    category=category,
                    exclusion_reason=exclusion,
                )
            )
        return registered

    def ranked(self, force: bool = False) -> dict[str, list[RegisteredModel]]:
        models = self.refresh(force=force)
        eligible = [model for model in models if model.category != "excluded"]
        eligible.sort(
            key=lambda model: (
                -model.average_quality,
                model.failures,
                model.prompt_price + model.completion_price + model.image_price,
                model.average_speed or float("inf"),
            )
        )
        return {
            "free_vision": [
                model for model in eligible if model.category == "free_vision"
            ][:10],
            "paid_vision": [
                model for model in eligible if model.category != "free_vision"
            ][:5],
            "excluded": [model for model in models if model.category == "excluded"],
        }

    def _performance(self) -> dict[str, dict]:
        with closing(self.database.connect()) as connection:
            rows = connection.execute("""
                SELECT model_name,
                       SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failures,
                       AVG(processing_time) AS average_speed,
                       AVG(quality_score) AS average_quality
                FROM attempts
                WHERE model_name IS NOT NULL AND model_name != ''
                GROUP BY model_name
                """).fetchall()
        return {row["model_name"]: dict(row) for row in rows}
