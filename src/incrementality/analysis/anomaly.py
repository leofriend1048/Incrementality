"""Pre-analysis anomaly detection for data quality assurance.

Automatically detects data quality issues BEFORE running causal inference.
Catching problems early prevents wasting compute on unreliable data and
ensures the test design integrity.

Checks:
1. Missing data: gaps in the daily panel (DMAs missing days)
2. Z-score outliers: individual DMA-day observations far from the mean
3. Sudden drops/spikes: day-over-day changes that indicate data issues
4. Zero-revenue periods: DMAs with sustained zero revenue
5. Coverage: whether treatment/holdout groups have adequate data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnomalyAlert:
    """A single detected anomaly."""
    severity: str  # "warning" or "blocker"
    category: str  # "missing_data", "outlier", "spike", "zero_revenue", "coverage"
    message: str
    affected_dmas: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class AnomalyReport:
    """Full anomaly detection report."""
    alerts: list[AnomalyAlert] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def has_blockers(self) -> bool:
        return any(a.severity == "blocker" for a in self.alerts)

    @property
    def warnings(self) -> list[str]:
        return [a.message for a in self.alerts if a.severity == "warning"]

    @property
    def blockers(self) -> list[str]:
        return [a.message for a in self.alerts if a.severity == "blocker"]

    @property
    def n_warnings(self) -> int:
        return sum(1 for a in self.alerts if a.severity == "warning")

    @property
    def n_blockers(self) -> int:
        return sum(1 for a in self.alerts if a.severity == "blocker")


def detect_anomalies(
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    revenue_col: str = "revenue",
    dma_col: str = "dma_code",
    date_col: str = "date",
    zscore_threshold: float = 3.5,
    spike_threshold: float = 3.0,
    max_missing_pct: float = 0.10,
    max_zero_pct: float = 0.20,
) -> AnomalyReport:
    """Run all anomaly detection checks on pre and post data.

    Args:
        pre_data: Pre-treatment period data.
        post_data: Post-treatment period data.
        treatment_dmas: Treatment group DMA codes.
        holdout_dmas: Holdout group DMA codes.
        revenue_col: Revenue column name.
        dma_col: DMA identifier column.
        date_col: Date column.
        zscore_threshold: Z-score threshold for outlier detection.
        spike_threshold: Multiplier for day-over-day spike detection.
        max_missing_pct: Max fraction of missing observations before warning.
        max_zero_pct: Max fraction of zero-revenue observations before warning.

    Returns:
        AnomalyReport with all detected issues.
    """
    report = AnomalyReport()
    all_dmas = treatment_dmas + holdout_dmas

    # Combine for comprehensive checks
    combined = pd.concat([pre_data, post_data], ignore_index=True)

    if combined.empty or revenue_col not in combined.columns:
        report.alerts.append(AnomalyAlert(
            severity="blocker",
            category="coverage",
            message="No data available for analysis. Cannot proceed.",
        ))
        return report

    # --- Check 1: Missing data ---
    _check_missing_data(
        combined, all_dmas, treatment_dmas, holdout_dmas,
        revenue_col, dma_col, date_col, max_missing_pct, report,
    )

    # --- Check 2: Z-score outliers ---
    _check_zscore_outliers(
        combined, all_dmas, revenue_col, dma_col, date_col,
        zscore_threshold, report,
    )

    # --- Check 3: Sudden spikes/drops ---
    _check_spikes(
        combined, all_dmas, revenue_col, dma_col, date_col,
        spike_threshold, report,
    )

    # --- Check 4: Zero-revenue periods ---
    _check_zero_revenue(
        combined, all_dmas, revenue_col, dma_col, date_col,
        max_zero_pct, report,
    )

    # --- Check 5: Coverage ---
    _check_coverage(
        pre_data, post_data, treatment_dmas, holdout_dmas,
        dma_col, date_col, report,
    )

    # Summary stats
    report.summary = {
        "total_observations": len(combined),
        "unique_dmas": combined[dma_col].nunique(),
        "unique_dates": combined[date_col].nunique(),
        "revenue_mean": float(combined[revenue_col].mean()),
        "revenue_median": float(combined[revenue_col].median()),
        "revenue_std": float(combined[revenue_col].std()),
        "n_warnings": report.n_warnings,
        "n_blockers": report.n_blockers,
    }

    logger.info(
        f"Anomaly detection: {report.n_blockers} blockers, "
        f"{report.n_warnings} warnings across "
        f"{combined[dma_col].nunique()} DMAs"
    )

    return report


def _check_missing_data(
    data: pd.DataFrame,
    all_dmas: list[str],
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    revenue_col: str,
    dma_col: str,
    date_col: str,
    max_missing_pct: float,
    report: AnomalyReport,
) -> None:
    """Check for missing observations in the panel."""
    dates = sorted(data[date_col].unique())
    n_dates = len(dates)

    if n_dates == 0:
        return

    # Expected: every DMA should have data for every date
    expected_per_dma = n_dates
    missing_dmas = []

    for dma in all_dmas:
        dma_data = data[data[dma_col] == dma]
        n_obs = len(dma_data)
        missing_pct = 1 - (n_obs / expected_per_dma)

        if missing_pct > max_missing_pct:
            missing_dmas.append(dma)

    if len(missing_dmas) > len(all_dmas) * 0.5:
        report.alerts.append(AnomalyAlert(
            severity="blocker",
            category="missing_data",
            message=(
                f"More than 50% of DMAs ({len(missing_dmas)}/{len(all_dmas)}) "
                f"have significant missing data (>{max_missing_pct:.0%} of days). "
                f"Data pipeline may be broken."
            ),
            affected_dmas=missing_dmas,
        ))
    elif missing_dmas:
        # Check if missing DMAs are concentrated in treatment or holdout
        missing_treatment = [d for d in missing_dmas if d in treatment_dmas]
        missing_holdout = [d for d in missing_dmas if d in holdout_dmas]

        report.alerts.append(AnomalyAlert(
            severity="warning",
            category="missing_data",
            message=(
                f"{len(missing_dmas)} DMAs have >{max_missing_pct:.0%} missing days "
                f"({len(missing_treatment)} treatment, {len(missing_holdout)} holdout). "
                f"Missing data is filled forward but may affect precision."
            ),
            affected_dmas=missing_dmas,
            details={
                "missing_treatment": len(missing_treatment),
                "missing_holdout": len(missing_holdout),
            },
        ))


def _check_zscore_outliers(
    data: pd.DataFrame,
    all_dmas: list[str],
    revenue_col: str,
    dma_col: str,
    date_col: str,
    threshold: float,
    report: AnomalyReport,
) -> None:
    """Detect DMA-day observations with extreme z-scores."""
    # Compute per-DMA z-scores relative to that DMA's own distribution
    outlier_dmas = set()
    total_outliers = 0

    for dma in all_dmas:
        dma_data = data[data[dma_col] == dma][revenue_col].values
        if len(dma_data) < 7:
            continue

        mean = np.mean(dma_data)
        std = np.std(dma_data, ddof=1)  # Sample std for proper z-scores

        if std < 1e-8:
            continue

        zscores = np.abs((dma_data - mean) / std)
        n_outliers = int(np.sum(zscores > threshold))

        if n_outliers > 0:
            outlier_dmas.add(dma)
            total_outliers += n_outliers

    if total_outliers > 0:
        report.alerts.append(AnomalyAlert(
            severity="warning",
            category="outlier",
            message=(
                f"Detected {total_outliers} outlier observations (|z| > {threshold}) "
                f"across {len(outlier_dmas)} DMAs. Winsorized analysis recommended "
                f"to assess robustness."
            ),
            affected_dmas=list(outlier_dmas),
            details={"total_outliers": total_outliers},
        ))


def _check_spikes(
    data: pd.DataFrame,
    all_dmas: list[str],
    revenue_col: str,
    dma_col: str,
    date_col: str,
    threshold: float,
    report: AnomalyReport,
) -> None:
    """Detect sudden day-over-day spikes or drops in revenue."""
    spike_dmas = set()
    total_spikes = 0

    for dma in all_dmas:
        dma_data = (
            data[data[dma_col] == dma]
            .sort_values(date_col)[revenue_col]
            .values
        )
        if len(dma_data) < 7:
            continue

        # Day-over-day changes
        diffs = np.abs(np.diff(dma_data))
        if len(diffs) == 0:
            continue

        median_diff = np.median(diffs)
        if median_diff < 1e-8:
            # Use mean if median is zero
            median_diff = np.mean(diffs)
            if median_diff < 1e-8:
                continue

        # Flag days where the change is > threshold * median change
        spikes = diffs > threshold * median_diff
        n_spikes = int(np.sum(spikes))

        if n_spikes > 0:
            spike_dmas.add(dma)
            total_spikes += n_spikes

    if total_spikes > len(all_dmas) * 2:
        report.alerts.append(AnomalyAlert(
            severity="warning",
            category="spike",
            message=(
                f"Detected {total_spikes} sudden revenue spikes/drops "
                f"(>{threshold}x median daily change) across {len(spike_dmas)} DMAs. "
                f"This could indicate promotional events, data pipeline issues, "
                f"or bot traffic."
            ),
            affected_dmas=list(spike_dmas),
            details={"total_spikes": total_spikes},
        ))


def _check_zero_revenue(
    data: pd.DataFrame,
    all_dmas: list[str],
    revenue_col: str,
    dma_col: str,
    date_col: str,
    max_zero_pct: float,
    report: AnomalyReport,
) -> None:
    """Detect DMAs with excessive zero-revenue days."""
    zero_dmas = []

    for dma in all_dmas:
        dma_data = data[data[dma_col] == dma][revenue_col].values
        if len(dma_data) == 0:
            continue

        zero_pct = np.mean(dma_data <= 0)
        if zero_pct > max_zero_pct:
            zero_dmas.append(dma)

    if zero_dmas:
        report.alerts.append(AnomalyAlert(
            severity="warning",
            category="zero_revenue",
            message=(
                f"{len(zero_dmas)} DMAs have >{max_zero_pct:.0%} zero-revenue days. "
                f"These DMAs may not have enough signal for reliable causal inference. "
                f"Consider excluding them or increasing test duration."
            ),
            affected_dmas=zero_dmas,
        ))


def _check_coverage(
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    dma_col: str,
    date_col: str,
    report: AnomalyReport,
) -> None:
    """Check that treatment and holdout groups have adequate coverage."""
    # Treatment DMAs present in data
    pre_treatment = set(pre_data[pre_data[dma_col].isin(treatment_dmas)][dma_col].unique())
    post_treatment = set(post_data[post_data[dma_col].isin(treatment_dmas)][dma_col].unique())
    pre_holdout = set(pre_data[pre_data[dma_col].isin(holdout_dmas)][dma_col].unique())
    post_holdout = set(post_data[post_data[dma_col].isin(holdout_dmas)][dma_col].unique())

    missing_treatment_pre = set(treatment_dmas) - pre_treatment
    missing_holdout_pre = set(holdout_dmas) - pre_holdout
    missing_treatment_post = set(treatment_dmas) - post_treatment
    missing_holdout_post = set(holdout_dmas) - post_holdout

    total_missing = len(missing_treatment_pre | missing_holdout_pre |
                        missing_treatment_post | missing_holdout_post)

    if len(post_holdout) < 3:
        report.alerts.append(AnomalyAlert(
            severity="blocker",
            category="coverage",
            message=(
                f"Only {len(post_holdout)} holdout DMAs have post-period data "
                f"(minimum 3 required). Cannot construct reliable counterfactual."
            ),
            affected_dmas=list(set(holdout_dmas) - post_holdout),
        ))
    elif total_missing > 0:
        report.alerts.append(AnomalyAlert(
            severity="warning",
            category="coverage",
            message=(
                f"{total_missing} DMAs are missing data in one or more periods. "
                f"Analysis will use only DMAs with complete coverage."
            ),
            details={
                "missing_treatment_pre": list(missing_treatment_pre),
                "missing_holdout_pre": list(missing_holdout_pre),
                "missing_treatment_post": list(missing_treatment_post),
                "missing_holdout_post": list(missing_holdout_post),
            },
        ))

    # Check pre-period length
    pre_dates = sorted(pre_data[date_col].unique())
    post_dates = sorted(post_data[date_col].unique())

    if len(pre_dates) < 14:
        report.alerts.append(AnomalyAlert(
            severity="warning",
            category="coverage",
            message=(
                f"Pre-period has only {len(pre_dates)} days (recommend >= 28). "
                f"Short pre-periods reduce counterfactual accuracy."
            ),
        ))
    if len(post_dates) < 7:
        report.alerts.append(AnomalyAlert(
            severity="warning",
            category="coverage",
            message=(
                f"Post-period has only {len(post_dates)} days (recommend >= 14). "
                f"Short test periods reduce statistical power."
            ),
        ))
