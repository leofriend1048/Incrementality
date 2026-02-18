"""
Pipelines package for the Michael Todd Beauty MMM platform.

Contains Prefect-based orchestration flows for daily ingestion
and weekly MCMC model refitting. Falls back to synchronous
execution when Prefect is not installed.
"""

from incrementality.pipelines.daily_ingest import mmm_daily_ingest
from incrementality.pipelines.weekly_refit import mmm_weekly_refit

__all__ = ["mmm_daily_ingest", "mmm_weekly_refit"]
