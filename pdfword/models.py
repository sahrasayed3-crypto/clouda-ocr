from dataclasses import dataclass


@dataclass
class PageResult:
    page_no: int
    model_used: str
    markdown: str
    quality_score: float | None = None
    text_quality_score: float | None = None
    layout_quality_score: float | None = None
    direction_quality_score: float | None = None
    completeness_score: float | None = None
    quality_label: str = "درجة جودة تقديرية"
    requires_manual_review: bool = False
    review_reason: str | None = None
    engines_attempted: tuple[str, ...] = ()
    route_used: str | None = None
    accepted: bool | None = None
    attempts_count: int | None = None
    cost: float | None = None
    elapsed_time: float | None = None
    corruption_diagnostics: dict | None = None
    selection_reason: str | None = None
    metadata: dict | None = None

    @property
    def final_score(self) -> float | None:
        return (
            self.text_quality_score
            if self.text_quality_score is not None
            else self.quality_score
        )

    @property
    def reason(self) -> str | None:
        return self.review_reason


class ModelEndpointError(RuntimeError):
    pass


class OpenRouterEmptyContentError(RuntimeError):
    pass
