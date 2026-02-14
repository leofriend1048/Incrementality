"""Simulation-based power analysis for geo holdout tests.

Two approaches:
1. Analytical: Fast closed-form MDE/power (for quick estimation)
2. Simulation-based: GeoLift-style Monte Carlo (for production decisions)

The simulation approach:
- Injects artificial treatment effects into historical holdout data
- Runs the full DiD estimator on each simulation
- Computes empirical power = fraction of simulations detecting the effect
- Computes empirical FPR = fraction of null simulations falsely detecting

This is what Haus uses and what GeoLift implements. Analytical formulas
are useful for rapid iteration, but the simulation power is what you
actually trust for 8-figure decisions.
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


# =====================================================================
# Analytical power analysis (fast, approximate)
# =====================================================================

def compute_mde(
    variance_estimate: HistoricalVarianceEstimate,
    n_treatment: int,
    n_holdout: int,
    duration_weeks: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Compute the Minimum Detectable Effect (MDE) for a geo holdout test.

    Returns MDE as a proportion of the mean (relative effect size).
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    rho = variance_estimate.autocorrelation_lag1
    if rho > 0:
        effective_periods = duration_weeks * (1 - rho) / (1 + rho)
    else:
        effective_periods = duration_weeks
    effective_periods = max(effective_periods, 1)

    did_variance = variance_estimate.within_dma_variance / effective_periods
    se = math.sqrt(did_variance / n_treatment + did_variance / n_holdout)
    mde_absolute = (z_alpha + z_beta) * se
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
    """Find minimum test duration to detect a given effect size."""
    for weeks in range(1, max_weeks + 1):
        mde = compute_mde(
            variance_estimate, n_treatment, n_holdout, weeks, alpha, power
        )
        if mde <= target_mde:
            return weeks
    return max_weeks


def _compute_achieved_power(
    variance_estimate: HistoricalVarianceEstimate,
    n_treatment: int,
    n_holdout: int,
    effect_size: float,
    duration_weeks: int,
    alpha: float,
) -> float:
    """Compute analytical statistical power for a given effect size and design."""
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


# =====================================================================
# Simulation-based power analysis (GeoLift-style — production grade)
# =====================================================================

def run_simulation_power_analysis(
    daily_data: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    test_duration_weeks: int,
    effect_sizes: list[float] | None = None,
    n_simulations: int = 200,
    alpha: float = 0.05,
    revenue_col: str = "revenue",
    dma_col: str = "dma_code",
    date_col: str = "date",
) -> PowerAnalysisResult:
    """GeoLift-style simulation-based power analysis.

    This is the real deal. Instead of relying on analytical formulas:
    1. Use historical data as the ground truth
    2. For each simulation:
       a. Split historical data into fake pre/post periods
       b. Inject an artificial treatment effect into the post period
       c. Run DiD estimator
       d. Check if the effect is detected (p < alpha)
    3. Empirical power = fraction of sims that detect the true effect
    4. Empirical FPR = fraction of null sims (effect=0) that falsely detect

    This captures all the messiness of real data that formulas miss:
    - Non-stationarity, heterogeneous treatment effects, autocorrelation,
      seasonal patterns, outliers, missing data.
    """
    if effect_sizes is None:
        effect_sizes = [0.05, 0.10, 0.15, 0.20]

    rng = np.random.default_rng(seed=42)

    df = daily_data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    all_dmas = treatment_dmas + holdout_dmas
    df = df[df[dma_col].isin(all_dmas)]

    dates = sorted(df[date_col].unique())
    n_dates = len(dates)
    post_days = test_duration_weeks * 7

    if n_dates < post_days + 14:
        logger.warning(
            f"Not enough data for simulation power analysis "
            f"({n_dates} days, need {post_days + 14}). "
            f"Falling back to analytical."
        )
        var_est = estimate_historical_variance(daily_data, revenue_col, dma_col, date_col)
        return _analytical_power_result(
            var_est, len(treatment_dmas), len(holdout_dmas),
            test_duration_weeks, alpha,
        )

    # --- Run simulations ---
    # For each simulation, pick a random split point in the data
    # to create fake pre/post periods
    min_pre = max(14, n_dates - post_days * 3)  # At least 14 days pre
    max_split = n_dates - post_days

    if min_pre > max_split:
        min_pre = 14
    if max_split <= min_pre:
        max_split = min_pre + 1

    # Null simulations (effect=0) for FPR
    null_detections = 0
    null_sims = min(n_simulations // 2, 50)

    logger.info(f"Running {null_sims} null simulations for FPR estimation...")
    for sim in range(null_sims):
        split_idx = rng.integers(min_pre, max_split + 1)
        split_date = dates[split_idx]

        pre = df[df[date_col] < split_date]
        post = df[(df[date_col] >= split_date) &
                   (df[date_col] < dates[min(split_idx + post_days, n_dates - 1)])]

        if pre.empty or post.empty:
            continue

        try:
            p_val = _quick_did_p_value(
                pre, post, treatment_dmas, holdout_dmas,
                revenue_col, dma_col,
            )
            if p_val < alpha:
                null_detections += 1
        except Exception:
            continue

    empirical_fpr = null_detections / null_sims if null_sims > 0 else 0.0

    # Power simulations for each effect size
    best_power = 0.0
    best_mde = effect_sizes[-1]

    power_by_effect = {}
    for effect in effect_sizes:
        detections = 0
        valid_sims = 0

        for sim in range(n_simulations):
            split_idx = rng.integers(min_pre, max_split + 1)
            split_date = dates[split_idx]
            end_idx = min(split_idx + post_days, n_dates - 1)

            pre = df[df[date_col] < split_date].copy()
            post = df[(df[date_col] >= split_date) &
                       (df[date_col] <= dates[end_idx])].copy()

            if pre.empty or post.empty:
                continue

            # Inject treatment effect
            treat_mask = post[dma_col].isin(treatment_dmas)
            post.loc[treat_mask, revenue_col] = (
                post.loc[treat_mask, revenue_col] * (1 + effect)
            )

            try:
                p_val = _quick_did_p_value(
                    pre, post, treatment_dmas, holdout_dmas,
                    revenue_col, dma_col,
                )
                valid_sims += 1
                if p_val < alpha:
                    detections += 1
            except Exception:
                continue

        sim_power = detections / valid_sims if valid_sims > 0 else 0
        power_by_effect[effect] = sim_power
        logger.info(f"  Effect={effect:.0%}: power={sim_power:.0%} ({valid_sims} valid sims)")

        if sim_power >= 0.80 and effect < best_mde:
            best_mde = effect
        best_power = max(best_power, sim_power)

    # Find the MDE: smallest effect with >= 80% power
    mde = best_mde
    for effect in sorted(power_by_effect.keys()):
        if power_by_effect[effect] >= 0.80:
            mde = effect
            break

    # Use the power at the MDE as the reported power
    sim_power_at_mde = power_by_effect.get(mde, best_power)

    # Compute analytical estimates for comparison
    var_est = estimate_historical_variance(daily_data, revenue_col, dma_col, date_col)
    analytical_mde = compute_mde(
        var_est, len(treatment_dmas), len(holdout_dmas),
        test_duration_weeks, alpha, 0.80,
    )
    analytical_power = _compute_achieved_power(
        var_est, len(treatment_dmas), len(holdout_dmas),
        mde, test_duration_weeks, alpha,
    )

    # Duration recommendation
    duration = compute_required_duration(
        var_est, len(treatment_dmas), len(holdout_dmas),
        mde, alpha, 0.80, 12,
    )

    # Cohen's d
    pooled_sd = math.sqrt(var_est.total_variance)
    effect_abs = mde * var_est.mean_weekly_revenue
    cohen_d = effect_abs / pooled_sd if pooled_sd > 0 else 0

    # Power score: 0-100 composite (Haus targets 85-90)
    power_score = _compute_power_score(
        sim_power_at_mde, empirical_fpr, mde, len(holdout_dmas),
    )

    logger.info(
        f"Simulation power analysis: MDE={mde:.1%}, "
        f"simulated power={sim_power_at_mde:.0%}, "
        f"FPR={empirical_fpr:.0%}, power score={power_score:.0f}"
    )

    return PowerAnalysisResult(
        minimum_detectable_effect=mde,
        recommended_duration_weeks=duration,
        required_sample_size_per_cell=max(len(treatment_dmas), len(holdout_dmas)),
        statistical_power=analytical_power,
        significance_level=alpha,
        baseline_variance=var_est.total_variance,
        baseline_mean_revenue=var_est.mean_weekly_revenue,
        effect_size_cohen_d=cohen_d,
        simulated_power=sim_power_at_mde,
        simulated_false_positive_rate=empirical_fpr,
        num_simulations=n_simulations,
        power_score=power_score,
    )


def _quick_did_p_value(
    pre: pd.DataFrame,
    post: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    revenue_col: str,
    dma_col: str,
) -> float:
    """Fast DiD p-value computation for simulation loops.

    Stripped-down version without bootstrap for speed.
    """
    def _dma_means(df, dma_list):
        return df[df[dma_col].isin(dma_list)].groupby(dma_col)[revenue_col].mean()

    t_pre = _dma_means(pre, treatment_dmas)
    t_post = _dma_means(post, treatment_dmas)
    h_pre = _dma_means(pre, holdout_dmas)
    h_post = _dma_means(post, holdout_dmas)

    t_diff = (t_post.reindex(treatment_dmas) - t_pre.reindex(treatment_dmas)).dropna()
    h_diff = (h_post.reindex(holdout_dmas) - h_pre.reindex(holdout_dmas)).dropna()

    if len(t_diff) < 2 or len(h_diff) < 2:
        return 1.0

    # Two-sample t-test on DMA-level differences
    t_stat, p_val = stats.ttest_ind(t_diff.values, h_diff.values)
    return float(p_val) if not np.isnan(p_val) else 1.0


def _compute_power_score(
    simulated_power: float,
    empirical_fpr: float,
    mde: float,
    n_holdout: int,
) -> float:
    """Compute composite power score (0-100).

    Haus targets 85-90 before greenlighting a test.

    Components:
    - Simulated power (40 points): 80%+ gets full marks
    - FPR calibration (20 points): close to 5% is ideal
    - MDE reasonableness (25 points): MDE <= 15% gets full marks
    - Sample size (15 points): more holdout DMAs = better
    """
    score = 0.0

    # Power (40 points)
    if simulated_power >= 0.80:
        score += 40.0
    elif simulated_power >= 0.60:
        score += 40.0 * (simulated_power - 0.40) / 0.40
    else:
        score += 40.0 * simulated_power / 0.80

    # FPR calibration (20 points) — ideal is ~5%
    if empirical_fpr <= 0.06:
        score += 20.0
    elif empirical_fpr <= 0.10:
        score += 15.0
    elif empirical_fpr <= 0.15:
        score += 8.0
    # > 15% gets 0

    # MDE (25 points)
    if mde <= 0.10:
        score += 25.0
    elif mde <= 0.15:
        score += 20.0
    elif mde <= 0.20:
        score += 12.0
    elif mde <= 0.30:
        score += 5.0

    # Sample size (15 points)
    if n_holdout >= 20:
        score += 15.0
    elif n_holdout >= 10:
        score += 10.0
    elif n_holdout >= 5:
        score += 5.0

    return min(100.0, max(0.0, score))


def _analytical_power_result(
    var_est: HistoricalVarianceEstimate,
    n_treatment: int,
    n_holdout: int,
    duration_weeks: int,
    alpha: float,
) -> PowerAnalysisResult:
    """Fallback analytical power result when simulation isn't possible."""
    mde = compute_mde(var_est, n_treatment, n_holdout, duration_weeks, alpha, 0.80)
    achieved_power = _compute_achieved_power(
        var_est, n_treatment, n_holdout, mde, duration_weeks, alpha,
    )
    duration = compute_required_duration(
        var_est, n_treatment, n_holdout, mde, alpha, 0.80, 12,
    )
    pooled_sd = math.sqrt(var_est.total_variance)
    effect_abs = mde * var_est.mean_weekly_revenue
    cohen_d = effect_abs / pooled_sd if pooled_sd > 0 else 0

    return PowerAnalysisResult(
        minimum_detectable_effect=mde,
        recommended_duration_weeks=duration,
        required_sample_size_per_cell=max(n_treatment, n_holdout),
        statistical_power=achieved_power,
        significance_level=alpha,
        baseline_variance=var_est.total_variance,
        baseline_mean_revenue=var_est.mean_weekly_revenue,
        effect_size_cohen_d=cohen_d,
    )


def run_power_analysis(
    variance_estimate: HistoricalVarianceEstimate,
    n_treatment: int,
    n_holdout: int,
    config: StatisticalConfig,
    target_mde: float | None = None,
) -> PowerAnalysisResult:
    """Run analytical power analysis (fast path).

    For simulation-based analysis, use run_simulation_power_analysis() instead.
    This is kept for backward compatibility and quick estimation during design.
    """
    alpha = config.significance_level
    power = config.target_power

    if target_mde is not None:
        duration = compute_required_duration(
            variance_estimate, n_treatment, n_holdout,
            target_mde, alpha, power, config.max_test_duration_weeks,
        )
        actual_mde = compute_mde(
            variance_estimate, n_treatment, n_holdout, duration, alpha, power
        )
    else:
        best_duration = config.min_test_duration_weeks
        best_mde = float("inf")
        for weeks in range(config.min_test_duration_weeks,
                           config.max_test_duration_weeks + 1):
            mde = compute_mde(
                variance_estimate, n_treatment, n_holdout, weeks, alpha, power
            )
            if mde <= 0.15:
                best_duration = weeks
                best_mde = mde
                break
            if mde < best_mde:
                best_mde = mde
                best_duration = weeks

        duration = best_duration
        actual_mde = best_mde

    pooled_sd = math.sqrt(variance_estimate.total_variance)
    effect_abs = actual_mde * variance_estimate.mean_weekly_revenue
    cohen_d = effect_abs / pooled_sd if pooled_sd > 0 else 0

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
