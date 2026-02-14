"""Statistical power analysis for geo holdout tests.

Determines minimum detectable effect size, required number of DMAs per cell,
and recommended test duration based on historical data variance.

The key challenge in geo tests is that the unit of analysis is the DMA,
not the individual customer, so we have relatively few units (~210 DMAs max)
and must design tests that are adequately powered despite this constraint.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from incrementality.config import StatisticalConfig
from incrementality.models import PowerAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class HistoricalVarianceEstimate:
    """Variance components estimated from historical DMA-level data."""
    between_dma_variance: float  # Variance across DMAs (cross-sectional)
    within_dma_variance: float  # Variance within DMAs over time (temporal)
    total_variance: float
    mean_daily_revenue: float
    mean_weekly_revenue: float
    coefficient_of_variation: float
    num_dmas: int
    num_periods: int
    autocorrelation_lag1: float  # Temporal autocorrelation


def estimate_historical_variance(
    daily_data: pd.DataFrame,
    revenue_col: str = "revenue",
    dma_col: str = "dma_code",
    date_col: str = "date",
) -> HistoricalVarianceEstimate:
    """Estimate variance components from historical DMA-level daily data.

    Uses a random-effects decomposition to separate between-DMA variance
    (persistent differences across markets) from within-DMA variance
    (temporal fluctuations).

    Args:
        daily_data: DataFrame with columns [date, dma_code, revenue]
        revenue_col: Column name for revenue
        dma_col: Column name for DMA identifier
        date_col: Column name for date
    """
    # Aggregate to weekly to reduce noise
    df = daily_data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["week"] = df[date_col].dt.isocalendar().week.astype(int)
    df["year_week"] = df[date_col].dt.strftime("%Y-%W")

    weekly = (
        df.groupby([dma_col, "year_week"])
        .agg(revenue=(revenue_col, "sum"))
        .reset_index()
    )

    # Between-DMA variance: variance of DMA means
    dma_means = weekly.groupby(dma_col)["revenue"].mean()
    between_var = dma_means.var()

    # Within-DMA variance: average variance within each DMA over time
    within_vars = weekly.groupby(dma_col)["revenue"].var()
    within_var = within_vars.mean()

    total_var = between_var + within_var
    grand_mean = dma_means.mean()
    cv = math.sqrt(total_var) / grand_mean if grand_mean > 0 else float("inf")

    # Estimate lag-1 autocorrelation (averaged across DMAs)
    autocorrs = []
    for dma, group in weekly.groupby(dma_col):
        if len(group) >= 4:
            series = group.sort_values("year_week")["revenue"]
            if series.std() > 0:
                ac = series.autocorr(lag=1)
                if not np.isnan(ac):
                    autocorrs.append(ac)
    mean_autocorr = np.mean(autocorrs) if autocorrs else 0.0

    # Daily mean
    daily_mean = df.groupby(dma_col)[revenue_col].mean().mean()

    return HistoricalVarianceEstimate(
        between_dma_variance=float(between_var),
        within_dma_variance=float(within_var),
        total_variance=float(total_var),
        mean_daily_revenue=float(daily_mean),
        mean_weekly_revenue=float(grand_mean),
        coefficient_of_variation=float(cv),
        num_dmas=int(dma_means.shape[0]),
        num_periods=int(weekly["year_week"].nunique()),
        autocorrelation_lag1=float(mean_autocorr),
    )


def compute_mde(
    variance_estimate: HistoricalVarianceEstimate,
    n_treatment: int,
    n_holdout: int,
    duration_weeks: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Compute the Minimum Detectable Effect (MDE) for a geo holdout test.

    For a two-sample t-test at the DMA-week level:
        MDE = (z_alpha/2 + z_beta) * sqrt(var_treatment/n_t + var_holdout/n_h)

    We adjust for:
    - Temporal autocorrelation (effective sample size reduction)
    - Clustering at DMA level
    - Pre-post design (difference-in-differences reduces variance)

    Returns MDE as a proportion of the mean (relative effect size).
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Effective number of independent time periods per DMA
    # Autocorrelation reduces effective sample size
    rho = variance_estimate.autocorrelation_lag1
    if rho > 0:
        # Effective sample size adjustment for AR(1) process
        effective_periods = duration_weeks * (1 - rho) / (1 + rho)
    else:
        effective_periods = duration_weeks

    effective_periods = max(effective_periods, 1)

    # Variance of the DMA-level mean over the test period
    # Combines between-DMA variance and within-DMA variance / effective_periods
    var_dma_mean = (
        variance_estimate.between_dma_variance
        + variance_estimate.within_dma_variance / effective_periods
    )

    # For DiD estimator, we subtract pre-period means, which removes between-DMA
    # variance (the persistent differences). This is a major variance reduction.
    # Residual variance is primarily within-DMA temporal variance.
    did_variance = variance_estimate.within_dma_variance / effective_periods

    # Standard error of the treatment effect estimate
    se = math.sqrt(did_variance / n_treatment + did_variance / n_holdout)

    # MDE in absolute terms
    mde_absolute = (z_alpha + z_beta) * se

    # Convert to relative effect (proportion of mean)
    mean = variance_estimate.mean_weekly_revenue
    mde_relative = mde_absolute / mean if mean > 0 else float("inf")

    return mde_relative


def compute_required_duration(
    variance_estimate: HistoricalVarianceEstimate,
    n_treatment: int,
    n_holdout: int,
    target_mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    max_weeks: int = 12,
) -> int:
    """Find minimum test duration to detect a given effect size.

    Binary searches over duration to find the shortest test that achieves
    the target MDE with the specified power.

    Args:
        target_mde: Target minimum detectable effect as a proportion (e.g. 0.10)
    """
    for weeks in range(1, max_weeks + 1):
        mde = compute_mde(
            variance_estimate, n_treatment, n_holdout, weeks, alpha, power
        )
        if mde <= target_mde:
            return weeks
    return max_weeks


def run_power_analysis(
    variance_estimate: HistoricalVarianceEstimate,
    n_treatment: int,
    n_holdout: int,
    config: StatisticalConfig,
    target_mde: float | None = None,
) -> PowerAnalysisResult:
    """Run complete power analysis and return results.

    If target_mde is provided, computes required duration.
    Otherwise, computes MDE for a range of durations and picks optimal.
    """
    alpha = config.significance_level
    power = config.target_power

    if target_mde is not None:
        # Find duration needed for target MDE
        duration = compute_required_duration(
            variance_estimate, n_treatment, n_holdout,
            target_mde, alpha, power, config.max_test_duration_weeks,
        )
        actual_mde = compute_mde(
            variance_estimate, n_treatment, n_holdout, duration, alpha, power
        )
    else:
        # Evaluate MDE at different durations, pick one with good MDE
        best_duration = config.min_test_duration_weeks
        best_mde = float("inf")
        for weeks in range(config.min_test_duration_weeks,
                           config.max_test_duration_weeks + 1):
            mde = compute_mde(
                variance_estimate, n_treatment, n_holdout, weeks, alpha, power
            )
            # Target: MDE ≤ 15% (detectable lift). If achievable, pick shortest.
            if mde <= 0.15:
                best_duration = weeks
                best_mde = mde
                break
            if mde < best_mde:
                best_mde = mde
                best_duration = weeks

        duration = best_duration
        actual_mde = best_mde

    # Compute Cohen's d
    pooled_sd = math.sqrt(variance_estimate.total_variance)
    effect_abs = actual_mde * variance_estimate.mean_weekly_revenue
    cohen_d = effect_abs / pooled_sd if pooled_sd > 0 else 0

    # Verify achieved power at chosen duration and MDE
    achieved_power = _compute_achieved_power(
        variance_estimate, n_treatment, n_holdout,
        actual_mde, duration, alpha,
    )

    return PowerAnalysisResult(
        minimum_detectable_effect=actual_mde,
        recommended_duration_weeks=duration,
        required_sample_size_per_cell=max(n_treatment, n_holdout),
        statistical_power=achieved_power,
        significance_level=alpha,
        baseline_variance=variance_estimate.total_variance,
        baseline_mean_revenue=variance_estimate.mean_weekly_revenue,
        effect_size_cohen_d=cohen_d,
    )


def _compute_achieved_power(
    variance_estimate: HistoricalVarianceEstimate,
    n_treatment: int,
    n_holdout: int,
    effect_size: float,
    duration_weeks: int,
    alpha: float,
) -> float:
    """Compute statistical power for a given effect size and design.

    Power = P(reject H0 | H1 is true)
         = 1 - Φ(z_α/2 - δ/SE)
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)

    rho = variance_estimate.autocorrelation_lag1
    effective_periods = duration_weeks * (1 - rho) / (1 + rho) if rho > 0 else duration_weeks
    effective_periods = max(effective_periods, 1)

    did_variance = variance_estimate.within_dma_variance / effective_periods
    se = math.sqrt(did_variance / n_treatment + did_variance / n_holdout)

    effect_absolute = effect_size * variance_estimate.mean_weekly_revenue
    noncentrality = effect_absolute / se if se > 0 else 0

    power = 1 - stats.norm.cdf(z_alpha - noncentrality)
    return float(min(power, 1.0))
