"""Training experiment planner package.

Includes typed planning domain models, transparent VRAM estimates,
storage/runtime/cost planning, and the canonical planning orchestrator.

Legacy API bridge
-----------------
Before this package existed the planner was the single module
``clouda_training/planner.py`` exposing ``TrainingPlan`` and
``plan_training`` (dataset-inventory planning used by
``clouda_training.cli`` and ``tests/training/test_planning.py``). This
package shadows that module (Python prefers packages over modules in
the same namespace), so the compatibility API is bridged here by explicitly
loading the original file. The root package's
``from .planner import TrainingPlan, plan_training`` keeps working
unchanged.

Note: ``import clouda_training.planner`` yields this package, and the
legacy symbols are re-exported on it, so both consumers
(``from clouda_training.planner import plan_training``) and the root
re-export see the legacy API.  The legacy module's own imports
(``clouda_contracts.storage``, ``clouda_training.config.models``,
``clouda_training.datasets.approved``) are stdlib-plus-local only and
load lazily here so that importing the planner package never pulls them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from .models import (  # noqa: F401
    BYTES_PER_GIB,
    CHECKPOINT_INTERVAL_DEFAULTS,
    CheckpointPlan,
    CostEstimate,
    DEFAULT_PROFILES,
    EPOCHS_DEFAULTS,
    Estimate,
    EstimateConfidence,
    EstimateSource,
    ExperimentPlan,
    ExperimentProfile,
    GRADIENT_ACCUMULATION_DEFAULTS,
    HardwareEnvelope,
    HardwareFit,
    MAX_STEPS_DEFAULTS,
    MICRO_BATCH_DEFAULTS,
    MemoryComponentEstimates,
    PLANNING_POLICY_VERSION,
    PRECISION_DEFAULTS,
    ParameterMetadata,
    PlanningAssumption,
    PlanningIdentity,
    PlanningWarning,
    ResourceEstimate,
    RuntimeEstimate,
    SAMPLE_COUNT_DEFAULTS,
    StepPlan,
    StorageEstimate,
    StorageKind,
    TrainingMode,
    TrainingScale,
    WORLD_SIZE_DEFAULTS,
    default_profile,
    worst_confidence,
)

__all__ = [
    "BYTES_PER_GIB",
    "CHECKPOINT_INTERVAL_DEFAULTS",
    "DEFAULT_PROFILES",
    "EPOCHS_DEFAULTS",
    "Estimate",
    "EstimateConfidence",
    "EstimateSource",
    "ExperimentPlan",
    "ExperimentProfile",
    "GRADIENT_ACCUMULATION_DEFAULTS",
    "HardwareEnvelope",
    "HardwareFit",
    "MAX_STEPS_DEFAULTS",
    "MICRO_BATCH_DEFAULTS",
    "PLANNING_POLICY_VERSION",
    "PRECISION_DEFAULTS",
    "ParameterMetadata",
    "PlanningAssumption",
    "PlanningIdentity",
    "PlanningWarning",
    "ResourceEstimate",
    "RuntimeEstimate",
    "SAMPLE_COUNT_DEFAULTS",
    "StepPlan",
    "StorageEstimate",
    "StorageKind",
    "TrainingMode",
    "TrainingScale",
    "WORLD_SIZE_DEFAULTS",
    "TrainingPlan",
    "default_profile",
    "plan_training",
    "worst_confidence",
]


def _load_legacy_module() -> ModuleType | None:
    """Explicitly load the shadowed legacy ``planner.py`` module.

    The legacy file lives next to this package.  It is loaded under a
    private name (not ``clouda_training.planner`` — that resolves to this
    package) and its public symbols are re-exported below.
    """
    legacy_path = Path(__file__).resolve().parent.parent / "planner.py"
    if not legacy_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "_clouda_training_planner_legacy", legacy_path
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass/typing introspection inside the
    # legacy module resolves consistently.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_module()

if _legacy is not None:
    TrainingPlan = _legacy.TrainingPlan  # type: ignore[attr-defined]
    plan_training = _legacy.plan_training  # type: ignore[attr-defined]
else:  # pragma: no cover - only when the legacy file is absent
    TrainingPlan = None  # type: ignore[assignment,misc]
    plan_training = None  # type: ignore[assignment]
