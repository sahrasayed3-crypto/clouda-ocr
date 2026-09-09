from __future__ import annotations

from pathlib import Path

from .runs import RunHandle, RunStatus, list_runs, load_run


class ExperimentRegistry:
    """Filesystem-derived registry; run artifacts remain the only source of truth."""

    def __init__(self, runs_root: str | Path) -> None:
        self.runs_root = Path(runs_root)

    def experiments(self) -> list[str]:
        return sorted({run.path.parent.name for run in list_runs(self.runs_root)})

    def runs(
        self, *, status: RunStatus | None = None, tags: set[str] | None = None
    ) -> list[RunHandle]:
        return list_runs(self.runs_root, status=status, tags=tags)

    def get(self, run_id: str) -> RunHandle:
        return load_run(run_id, self.runs_root)

    def latest(self, experiment: str) -> RunHandle | None:
        runs = [run for run in self.runs() if run.path.parent.name == experiment]
        return runs[0] if runs else None

    def best(
        self, experiment: str, metric: str, *, greater_is_better: bool = False
    ) -> RunHandle | None:
        candidates = [
            run
            for run in self.runs(status=RunStatus.COMPLETED)
            if run.path.parent.name == experiment
            and metric in run.summary().get("best_metrics", {})
        ]
        if not candidates:
            return None

        def metric_value(run: RunHandle) -> float:
            return float(run.summary()["best_metrics"][metric])

        return (
            max(candidates, key=metric_value)
            if greater_is_better
            else min(candidates, key=metric_value)
        )
