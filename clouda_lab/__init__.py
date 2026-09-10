"""Clouda Lab backend.

Backend domain services powering the future Clouda Lab web interface:

- :mod:`error_analysis` — OCR error analysis engine (char/word alignment,
  Arabic-aware classification)
- :mod:`batch_analysis` — aggregation across samples by model/run/dataset/
  profile/distortion/source/split
- :mod:`dataset_selection` — reproducible, provenance-preserving dataset
  subset selection
- :mod:`holdout_guard` — fail-closed holdout protection
- :mod:`failure_analysis` — before/after comparison between models/runs
- :mod:`failure_buckets` — structured failure bucket taxonomy
- :mod:`hard_examples` — transparent weighted hard-example mining
- :mod:`active_learning` — deterministic next-batch recommendation
- :mod:`selection_history` — lightweight usage registry
- :mod:`distortion_experiments` — experiment layer over the Data Factory
- :mod:`evaluation_service` — canonical backend evaluation entry point
- :mod:`training_orchestrator` — facade over the Training Experiment Framework
- :mod:`run_pipeline` — post-run analysis pipeline

No web UI, no HTTP API, no presentation code — backend correctness only.
Real training remains disabled (mock/dry-run only).
"""

from __future__ import annotations

__version__ = "0.1.0"

from .active_learning import recommend_next_batch
from .batch_analysis import analyze_batch
from .dataset_selection import (
    SelectionCriteria,
    SelectionResult,
    select_rows,
    select_samples,
    write_selection_manifest,
)
from .error_analysis import analyze_sample
from .evaluation_service import EvaluationService
from .failure_analysis import SampleMetrics, compare_failure
from .hard_examples import rank_hard_examples, select_hard_examples
from .models import (  # noqa: F401  (re-exported contracts)
    BatchReport,
    BatchSummary,
    ErrorAnalysis,
    ErrorRecord,
    FailureBucket,
    FailureComparison,
    FailureReport,
    HardExampleScore,
    OCRSample,
    RunAnalysis,
    SampleSelection,
    TrainingBatchRecommendation,
)
from .selection_history import SelectionHistory
from .results_service import StoredResultsAnalysisService
from .training_orchestrator import TrainingOrchestrator

__all__ = [
    "BatchReport",
    "BatchSummary",
    "ErrorAnalysis",
    "ErrorRecord",
    "EvaluationService",
    "FailureBucket",
    "FailureComparison",
    "FailureReport",
    "HardExampleScore",
    "OCRSample",
    "RunAnalysis",
    "SampleMetrics",
    "SampleSelection",
    "SelectionCriteria",
    "SelectionHistory",
    "SelectionResult",
    "StoredResultsAnalysisService",
    "TrainingBatchRecommendation",
    "TrainingOrchestrator",
    "analyze_batch",
    "analyze_sample",
    "compare_failure",
    "rank_hard_examples",
    "recommend_next_batch",
    "select_hard_examples",
    "select_rows",
    "select_samples",
    "write_selection_manifest",
]
