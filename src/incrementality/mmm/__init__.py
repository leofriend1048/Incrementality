"""Google Meridian Marketing Mix Modeling engine for Michael Todd Beauty.

This package wraps the Google Meridian MMM library and provides utilities
for tensor preparation, calibration event management, posterior diagnostics,
budget optimisation, model registry, and three-gate validation.

Exported classes
----------------
MeridianConfig        - Pydantic model holding all model hyper-parameters.
MeridianTensorBuilder - Assembles Meridian-ready numpy tensors from raw DataFrames.
MeridianMMM           - Core Meridian wrapper with fit / predict / diagnostics.
CalibrationEvent      - Pydantic model representing a single lift-test calibration.
CalibrationStore      - Persistent store for CalibrationEvents with Meridian export.
BudgetOptimizer       - CVXPY / scipy budget allocator with Hill-curve saturation.
OptimizationResult    - Pydantic result model for BudgetOptimizer.
ModelRegistry         - MLflow (or local JSON) run & artifact registry.
MMMValidator          - Three-gate validation framework (PRD Section 10).
ValidationReport      - Pydantic model for the full three-gate validation report.
"""

from __future__ import annotations

from incrementality.mmm.calibration import CalibrationEvent, CalibrationStore
from incrementality.mmm.config import MeridianConfig
from incrementality.mmm.model import MeridianMMM
from incrementality.mmm.optimizer import BudgetOptimizer, OptimizationResult
from incrementality.mmm.registry import ModelRegistry
from incrementality.mmm.tensors import MeridianTensorBuilder
from incrementality.mmm.validation import (
    MMMValidator,
    ValidationReport,
    compute_mape,
    check_contribution_sum,
)

__all__ = [
    # Existing
    "MeridianConfig",
    "MeridianTensorBuilder",
    "MeridianMMM",
    "CalibrationEvent",
    "CalibrationStore",
    # New
    "BudgetOptimizer",
    "OptimizationResult",
    "ModelRegistry",
    "MMMValidator",
    "ValidationReport",
    "compute_mape",
    "check_contribution_sum",
]
