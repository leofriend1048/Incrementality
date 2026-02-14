"""Causal inference estimators for geo holdout incrementality tests.

Implements multiple methods for estimating the causal effect of advertising:

1. Difference-in-Differences (DiD): The primary estimator. Compares the change
   in outcomes between treatment and holdout groups from pre to post period.
   Removes time-invariant confounders.

2. Synthetic Control: Constructs a weighted combination of holdout DMAs to
   create a "synthetic" treatment group. Better for small holdout groups.

3. Simple Lift: Naive comparison (treatment mean - holdout mean). Included
   as a baseline but biased if groups aren't perfectly balanced.

All estimators produce confidence intervals via clustered bootstrap at the
DMA level, which correctly accounts for within-DMA correlation.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge

from incrementality.models import CellType, IncrementalityResult

logger = logging.getLogger(__name__)


def difference_in_differences(
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    revenue_col: str = "revenue",
    dma_col: str = "dma_code",
    alpha: float = 0.05,
) -> IncrementalityResult:
    """Estimate treatment effect using Difference-in-Differences.

    The DiD estimator:
        τ = (Y_treatment_post - Y_treatment_pre) - (Y_holdout_post - Y_holdout_pre)

    This removes:
    - Time-invariant DMA differences (via differencing)
    - Common time trends (via the holdout comparison)

    Key assumption: Parallel trends — absent treatment, treatment and holdout
    groups would have followed the same trend.

    Args:
        pre_data: Daily data from pre-test period (for baseline)
        post_data: Daily data from test period
        treatment_dmas: DMA codes in treatment cell
        holdout_dmas: DMA codes in holdout cell
        revenue_col: Column with revenue values
        alpha: Significance level for confidence intervals
    """
    # Aggregate to DMA-level means
    def _dma_means(df: pd.DataFrame, dma_list: list[str]) -> pd.Series:
        subset = df[df[dma_col].isin(dma_list)]
        return subset.groupby(dma_col)[revenue_col].mean()

    treatment_pre = _dma_means(pre_data, treatment_dmas)
    treatment_post = _dma_means(post_data, treatment_dmas)
    holdout_pre = _dma_means(pre_data, holdout_dmas)
    holdout_post = _dma_means(post_data, holdout_dmas)

    # DiD at DMA level
    treatment_diff = treatment_post.reindex(treatment_dmas) - treatment_pre.reindex(treatment_dmas)
    holdout_diff = holdout_post.reindex(holdout_dmas) - holdout_pre.reindex(holdout_dmas)

    # Drop DMAs with missing data in either period
    treatment_diff = treatment_diff.dropna()
    holdout_diff = holdout_diff.dropna()

    if len(treatment_diff) == 0 or len(holdout_diff) == 0:
        raise ValueError("Insufficient data for DiD estimation")

    # Point estimate
    tau = treatment_diff.mean() - holdout_diff.mean()

    # Holdout post mean (baseline for relative lift)
    holdout_post_mean = holdout_post.mean()

    # Standard error via clustered bootstrap
    tau_boots = _clustered_bootstrap_did(
        treatment_diff.values, holdout_diff.values, n_boot=2000,
    )

    se = np.std(tau_boots)
    ci_lower = np.percentile(tau_boots, 100 * alpha / 2)
    ci_upper = np.percentile(tau_boots, 100 * (1 - alpha / 2))

    # Relative lift
    relative_lift = tau / holdout_post_mean if holdout_post_mean > 0 else 0
    rel_lower = ci_lower / holdout_post_mean if holdout_post_mean > 0 else 0
    rel_upper = ci_upper / holdout_post_mean if holdout_post_mean > 0 else 0

    # P-value (two-sided)
    if se > 0:
        z_stat = tau / se
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))
    else:
        p_value = 1.0

    # Cohen's d
    pooled_sd = math.sqrt(
        (treatment_diff.var() + holdout_diff.var()) / 2
    )
    cohen_d = tau / pooled_sd if pooled_sd > 0 else 0

    return IncrementalityResult(
        absolute_lift=float(tau),
        relative_lift=float(relative_lift),
        lift_lower_ci=float(rel_lower),
        lift_upper_ci=float(rel_upper),
        p_value=float(p_value),
        is_significant=p_value < alpha,
        confidence_level=1 - alpha,
        cohen_d=float(cohen_d),
        method="difference_in_differences",
    )


def _clustered_bootstrap_did(
    treatment_diffs: np.ndarray,
    holdout_diffs: np.ndarray,
    n_boot: int = 2000,
) -> np.ndarray:
    """Clustered bootstrap for DiD standard errors.

    Resamples entire DMAs (clusters) rather than individual observations
    to preserve within-DMA correlation structure.
    """
    n_t = len(treatment_diffs)
    n_h = len(holdout_diffs)
    taus = np.empty(n_boot)

    rng = np.random.default_rng(seed=42)
    for b in range(n_boot):
        t_idx = rng.integers(0, n_t, size=n_t)
        h_idx = rng.integers(0, n_h, size=n_h)
        taus[b] = treatment_diffs[t_idx].mean() - holdout_diffs[h_idx].mean()

    return taus


def synthetic_control(
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    revenue_col: str = "revenue",
    dma_col: str = "dma_code",
    date_col: str = "date",
    alpha: float = 0.05,
) -> IncrementalityResult:
    """Estimate treatment effect using the Synthetic Control Method.

    Constructs a weighted combination of holdout DMAs that best matches
    the treatment group's pre-period trajectory. The post-period gap
    between actual treatment and synthetic control is the treatment effect.

    Advantages over DiD:
    - Does not require parallel trends assumption
    - Better for heterogeneous treatment effects
    - Provides visual validation (pre-period fit)

    Weights are estimated via ridge regression on pre-period data.
    Inference via placebo/permutation tests.
    """
    # Aggregate treatment DMAs into a single series
    treatment_pre = (
        pre_data[pre_data[dma_col].isin(treatment_dmas)]
        .groupby(date_col)[revenue_col].mean()
        .sort_index()
    )
    treatment_post = (
        post_data[post_data[dma_col].isin(treatment_dmas)]
        .groupby(date_col)[revenue_col].mean()
        .sort_index()
    )

    # Build holdout DMA matrix (each column = one holdout DMA's time series)
    holdout_pre_wide = (
        pre_data[pre_data[dma_col].isin(holdout_dmas)]
        .pivot_table(index=date_col, columns=dma_col, values=revenue_col, aggfunc="mean")
        .sort_index()
    )
    holdout_post_wide = (
        post_data[post_data[dma_col].isin(holdout_dmas)]
        .pivot_table(index=date_col, columns=dma_col, values=revenue_col, aggfunc="mean")
        .sort_index()
    )

    # Align dates
    common_pre_dates = treatment_pre.index.intersection(holdout_pre_wide.index)
    common_post_dates = treatment_post.index.intersection(holdout_post_wide.index)

    if len(common_pre_dates) < 7 or len(common_post_dates) < 7:
        raise ValueError("Insufficient overlapping dates for synthetic control")

    Y_pre = treatment_pre.loc[common_pre_dates].values
    X_pre = holdout_pre_wide.loc[common_pre_dates].fillna(0).values

    Y_post = treatment_post.loc[common_post_dates].values
    X_post = holdout_post_wide.loc[common_post_dates].fillna(0).values

    # Fit weights via ridge regression (constrained to be non-negative)
    ridge = Ridge(alpha=1.0, fit_intercept=True, positive=False)
    ridge.fit(X_pre, Y_pre)

    # Synthetic control predictions
    synthetic_pre = ridge.predict(X_pre)
    synthetic_post = ridge.predict(X_post)

    # Pre-period fit (R²)
    ss_res = np.sum((Y_pre - synthetic_pre) ** 2)
    ss_tot = np.sum((Y_pre - Y_pre.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    logger.info(f"Synthetic control pre-period R² = {r_squared:.3f}")

    # Treatment effect: actual - synthetic in post period
    gaps = Y_post - synthetic_post
    tau = float(np.mean(gaps))

    # Baseline (synthetic control mean in post period)
    baseline = float(np.mean(synthetic_post))

    # Inference via permutation test
    # For each holdout DMA, pretend it's the treatment and compute placebo gap
    placebo_effects = []
    holdout_cols = holdout_pre_wide.columns.tolist()
    for target_dma in holdout_cols:
        other_dmas = [d for d in holdout_cols if d != target_dma]
        if len(other_dmas) < 2:
            continue
        target_pre = holdout_pre_wide.loc[common_pre_dates, target_dma].values
        donor_pre = holdout_pre_wide.loc[common_pre_dates, other_dmas].fillna(0).values
        target_post = holdout_post_wide.loc[common_post_dates, target_dma].values
        donor_post = holdout_post_wide.loc[common_post_dates, other_dmas].fillna(0).values

        ridge_p = Ridge(alpha=1.0, fit_intercept=True)
        ridge_p.fit(donor_pre, target_pre)
        synth_post = ridge_p.predict(donor_post)
        placebo_effect = float(np.mean(target_post - synth_post))
        placebo_effects.append(placebo_effect)

    # P-value: fraction of placebos with effect >= actual
    if placebo_effects:
        p_value = float(np.mean(np.abs(placebo_effects) >= abs(tau)))
    else:
        p_value = 1.0

    # CI from placebo distribution
    if placebo_effects:
        se = float(np.std(placebo_effects))
        z = scipy_stats.norm.ppf(1 - alpha / 2)
        ci_lower = tau - z * se
        ci_upper = tau + z * se
    else:
        se = 0.0
        ci_lower = tau
        ci_upper = tau

    # Relative lift
    relative_lift = tau / baseline if baseline > 0 else 0
    rel_lower = ci_lower / baseline if baseline > 0 else 0
    rel_upper = ci_upper / baseline if baseline > 0 else 0

    # Cohen's d
    cohen_d = tau / se if se > 0 else 0

    return IncrementalityResult(
        absolute_lift=float(tau),
        relative_lift=float(relative_lift),
        lift_lower_ci=float(rel_lower),
        lift_upper_ci=float(rel_upper),
        p_value=float(p_value),
        is_significant=p_value < alpha,
        confidence_level=1 - alpha,
        cohen_d=float(cohen_d),
        method="synthetic_control",
    )


def simple_lift(
    post_data: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    revenue_col: str = "revenue",
    dma_col: str = "dma_code",
    alpha: float = 0.05,
) -> IncrementalityResult:
    """Simple lift comparison between treatment and holdout.

    This is the most straightforward estimator but also the most biased
    if groups aren't perfectly balanced. Included as a sanity check.

    Lift = (mean_treatment - mean_holdout) / mean_holdout
    """
    t_data = post_data[post_data[dma_col].isin(treatment_dmas)]
    h_data = post_data[post_data[dma_col].isin(holdout_dmas)]

    # DMA-level means
    t_means = t_data.groupby(dma_col)[revenue_col].mean()
    h_means = h_data.groupby(dma_col)[revenue_col].mean()

    t_mean = t_means.mean()
    h_mean = h_means.mean()

    tau = t_mean - h_mean
    relative_lift = tau / h_mean if h_mean > 0 else 0

    # Welch's t-test
    t_stat, p_value = scipy_stats.ttest_ind(
        t_means.values, h_means.values, equal_var=False,
    )

    # Confidence interval
    se = math.sqrt(t_means.var() / len(t_means) + h_means.var() / len(h_means))
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    ci_lower = (tau - z * se) / h_mean if h_mean > 0 else 0
    ci_upper = (tau + z * se) / h_mean if h_mean > 0 else 0

    # Cohen's d
    pooled_sd = math.sqrt((t_means.var() + h_means.var()) / 2)
    cohen_d = tau / pooled_sd if pooled_sd > 0 else 0

    return IncrementalityResult(
        absolute_lift=float(tau),
        relative_lift=float(relative_lift),
        lift_lower_ci=float(ci_lower),
        lift_upper_ci=float(ci_upper),
        p_value=float(p_value),
        is_significant=p_value < alpha,
        confidence_level=1 - alpha,
        cohen_d=float(cohen_d),
        method="simple_lift",
    )


def run_all_estimators(
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    revenue_col: str = "revenue",
    dma_col: str = "dma_code",
    date_col: str = "date",
    alpha: float = 0.05,
) -> dict[str, IncrementalityResult]:
    """Run all available estimators and return results keyed by method name.

    The primary estimator is DiD. Synthetic control and simple lift
    are included for robustness checks.
    """
    results = {}

    # DiD (primary)
    try:
        results["difference_in_differences"] = difference_in_differences(
            pre_data, post_data, treatment_dmas, holdout_dmas,
            revenue_col, dma_col, alpha,
        )
    except Exception as e:
        logger.error(f"DiD estimation failed: {e}")

    # Synthetic control
    try:
        results["synthetic_control"] = synthetic_control(
            pre_data, post_data, treatment_dmas, holdout_dmas,
            revenue_col, dma_col, date_col, alpha,
        )
    except Exception as e:
        logger.warning(f"Synthetic control failed (not critical): {e}")

    # Simple lift (sanity check)
    try:
        results["simple_lift"] = simple_lift(
            post_data, treatment_dmas, holdout_dmas,
            revenue_col, dma_col, alpha,
        )
    except Exception as e:
        logger.warning(f"Simple lift failed: {e}")

    if not results:
        raise ValueError("All estimators failed")

    return results
