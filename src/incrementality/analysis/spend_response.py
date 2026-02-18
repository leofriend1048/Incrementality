"""Spend response curve estimation from geo incrementality test data.

Uses a dose-response framework with pre-period-adjusted DMA counterfactuals
to estimate the revenue response function: revenue = f(spend).

Methodology
-----------
1. For each treatment DMA, compute its organic counterfactual revenue using
   its own pre-period baseline scaled by the holdout group's organic growth
   rate.  This removes DMA-level heterogeneity (market size, demographics).

2. Compute DMA-level incremental revenue = actual post revenue - counterfactual.
   This isolates the ad-driven component for each DMA.

3. Fit a Hill saturation function to the (spend_i, incremental_i) pairs:
       incremental_revenue = R_max * spend^alpha / (K^alpha + spend^alpha)

4. Calibrate: if the causal iROAS from the primary analysis (ASCM/BSTS/DiD
   ensemble) is provided, scale R_max so that the curve's implied iROAS
   matches the rigorous experimental estimate.  This anchors the dose-
   response shape to the causal point rather than relying solely on cross-
   sectional variation.

5. Bootstrap confidence intervals on curve parameters and optimal spend.

From this curve, we derive:
    - Marginal ROAS at any spend level: d(revenue) / d(spend)
    - Optimal spend: where marginal ROAS = target (default 1.0 = breakeven)
    - Confidence intervals on the optimal spend recommendation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass
class SpendResponseCurve:
    """Estimated spend response curve and derived metrics."""

    # Hill function parameters
    r_max: float          # Saturation level (max incremental revenue per DMA per day)
    k: float              # Half-saturation spend (per DMA per day)
    alpha: float          # Shape parameter

    # Derived metrics
    current_spend: float        # Current total daily spend
    current_iroas: float        # iROAS at current spend
    current_marginal_roas: float  # Marginal ROAS at current spend

    optimal_spend: float        # Spend where marginal ROAS = target
    optimal_iroas: float        # iROAS at optimal spend
    optimal_spend_lower: float  # 95% CI lower bound on optimal spend
    optimal_spend_upper: float  # 95% CI upper bound on optimal spend
    target_marginal_roas: float  # The target used (default 1.0)

    # Curve data for plotting
    spend_curve: list[float] = field(default_factory=list)
    revenue_curve: list[float] = field(default_factory=list)
    revenue_curve_lower: list[float] = field(default_factory=list)  # Bootstrap CI lower
    revenue_curve_upper: list[float] = field(default_factory=list)  # Bootstrap CI upper
    marginal_roas_curve: list[float] = field(default_factory=list)
    avg_roas_curve: list[float] = field(default_factory=list)

    # Quality metrics
    r_squared: float = 0.0
    n_dmas_used: int = 0
    converged: bool = False
    calibrated: bool = False      # Was curve anchored to causal iROAS?
    observed_spend_max: float = 0.0  # Max observed per-DMA spend (for extrapolation warning)

    # Recommendations
    spend_change_pct: float = 0.0
    spend_change_direction: str = ""  # "increase", "decrease", or "maintain"
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Hill function helpers
# ---------------------------------------------------------------------------

def _hill_function(spend: np.ndarray, r_max: float, k: float, alpha: float) -> np.ndarray:
    """Hill saturation function: R_max * s^a / (K^a + s^a)."""
    s_a = np.power(np.maximum(spend, 1e-10), alpha)
    k_a = np.power(max(k, 1e-10), alpha)
    return r_max * s_a / (k_a + s_a)


def _hill_marginal(spend: float, r_max: float, k: float, alpha: float) -> float:
    """Derivative of Hill function: marginal revenue at a given spend level."""
    s = max(spend, 1e-10)
    k_a = k ** alpha
    s_a = s ** alpha
    denominator = (k_a + s_a) ** 2
    if denominator < 1e-20:
        return 0.0
    return r_max * alpha * k_a * (s ** (alpha - 1)) / denominator


def _fit_hill(
    x: np.ndarray, y: np.ndarray,
) -> tuple[tuple[float, float, float] | None, bool]:
    """Fit Hill function to spend/revenue data.

    Returns (params, converged) where params = (r_max, k, alpha) or None.
    """
    r_max_guess = np.max(y) * 1.5 if np.max(y) > 0 else max(np.mean(y) * 3, 1.0)
    x_pos = x[x > 0]
    k_guess = float(np.median(x_pos)) if len(x_pos) > 0 else 100.0
    alpha_guess = 0.7

    try:
        popt, _ = curve_fit(
            _hill_function,
            x, y,
            p0=[r_max_guess, k_guess, alpha_guess],
            bounds=([0, 1e-6, 0.1], [np.inf, np.inf, 3.0]),
            maxfev=10000,
        )
        return (float(popt[0]), float(popt[1]), float(popt[2])), True
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Hill function fit failed: {e}. Trying log-linear fallback.")
        try:
            x_log = np.log1p(x)
            coeffs = np.polyfit(x_log, y, 1)
            r_max = float(coeffs[0] * np.log1p(np.max(x)) * 2)
            k = float(np.median(x_pos)) if len(x_pos) > 0 else 100.0
            alpha = 0.5
            return (r_max, k, alpha), False
        except Exception:
            logger.warning("Fallback log-linear also failed")
            return None, False


def _find_optimal_spend(
    r_max: float,
    k: float,
    alpha: float,
    current_spend: float,
    target_marginal_roas: float,
) -> float:
    """Find spend level where marginal ROAS = target."""
    spend_range = np.linspace(0.01, max(current_spend * 3, k * 5), 10000)
    for s in spend_range:
        if _hill_marginal(s, r_max, k, alpha) < target_marginal_roas:
            return float(s)
    return float(spend_range[-1])


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------

def estimate_spend_response(
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    ad_spend_data: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    causal_iroas: float | None = None,
    revenue_col: str = "revenue",
    target_marginal_roas: float = 1.0,
    n_curve_points: int = 100,
    n_bootstrap: int = 200,
) -> SpendResponseCurve | None:
    """Estimate the spend response curve from geo test data.

    Uses pre-period-adjusted DMA counterfactuals to remove market-size
    confounding, then fits a Hill dose-response curve.

    Args:
        pre_data: Pre-test period daily revenue data (DMA-level).
        post_data: Test period daily revenue data (DMA-level).
        ad_spend_data: Ad spend data during test period [date, dma_code, spend].
        treatment_dmas: Treatment cell DMA codes.
        holdout_dmas: Holdout cell DMA codes.
        causal_iroas: The causal iROAS from the primary analysis (ensemble).
            If provided, the curve is calibrated to match this estimate.
        revenue_col: Column name for revenue.
        target_marginal_roas: Target marginal ROAS for optimal spend (default 1.0).
        n_curve_points: Number of points on the output curve.
        n_bootstrap: Number of bootstrap resamples for confidence intervals.

    Returns:
        SpendResponseCurve or None if estimation fails.
    """
    # --- Step 1: Compute DMA-level averages ---
    pre_dma = (
        pre_data.groupby("dma_code")[revenue_col]
        .mean()
        .rename("pre_avg_revenue")
    )
    post_dma = (
        post_data.groupby("dma_code")[revenue_col]
        .mean()
        .rename("post_avg_revenue")
    )

    if ad_spend_data.empty or "spend" not in ad_spend_data.columns:
        logger.warning("No spend data available for response curve estimation")
        return None

    dma_spend = (
        ad_spend_data.groupby("dma_code")["spend"]
        .mean()
        .rename("avg_daily_spend")
    )

    # Merge into DMA-level panel
    dma_data = pd.DataFrame({
        "pre_avg_revenue": pre_dma,
        "post_avg_revenue": post_dma,
        "avg_daily_spend": dma_spend,
    }).fillna(0.0)

    # --- Step 2: Organic growth rate from holdout ---
    holdout_in_data = [d for d in holdout_dmas if d in dma_data.index]
    treatment_in_data = [d for d in treatment_dmas if d in dma_data.index]

    if len(holdout_in_data) < 2 or len(treatment_in_data) < 3:
        logger.warning(
            f"Not enough DMAs for response curve: "
            f"{len(holdout_in_data)} holdout, {len(treatment_in_data)} treatment"
        )
        return None

    holdout_pre_mean = dma_data.loc[holdout_in_data, "pre_avg_revenue"].mean()
    holdout_post_mean = dma_data.loc[holdout_in_data, "post_avg_revenue"].mean()

    if holdout_pre_mean < 1e-6:
        logger.warning("Holdout pre-period revenue near zero — cannot estimate growth")
        return None

    organic_growth = holdout_post_mean / holdout_pre_mean
    logger.info(f"Organic growth rate (from holdout): {organic_growth:.4f}")

    # --- Step 3: DMA-specific counterfactuals & incremental revenue ---
    treatment_df = dma_data.loc[treatment_in_data].copy()

    # Each DMA's counterfactual = its own pre-period revenue × organic growth
    treatment_df["counterfactual"] = treatment_df["pre_avg_revenue"] * organic_growth
    treatment_df["incremental_revenue"] = (
        treatment_df["post_avg_revenue"] - treatment_df["counterfactual"]
    )

    # Keep DMAs with positive spend for curve fitting
    fit_data = treatment_df[treatment_df["avg_daily_spend"] > 0].copy()

    if len(fit_data) < 3:
        logger.warning(f"Only {len(fit_data)} DMAs with spend — need at least 3")
        return None

    # --- Step 4: Fit Hill function ---
    x = fit_data["avg_daily_spend"].values
    y = fit_data["incremental_revenue"].values

    # Anchor at zero spend = zero incremental revenue
    x_fit = np.concatenate([[0.0], x])
    y_fit = np.concatenate([[0.0], y])

    params, converged = _fit_hill(x_fit, y_fit)
    if params is None:
        return None

    r_max, k, alpha = params

    # --- Step 5: Calibrate to causal iROAS ---
    calibrated = False
    if causal_iroas is not None and causal_iroas > 0:
        total_spend = float(x.sum())
        total_predicted_inc = float(_hill_function(x, r_max, k, alpha).sum())
        predicted_iroas = total_predicted_inc / total_spend if total_spend > 0 else 0

        if predicted_iroas > 1e-6:
            calibration_factor = causal_iroas / predicted_iroas
            r_max *= calibration_factor
            calibrated = True
            logger.info(
                f"Calibrated to causal iROAS: curve predicted {predicted_iroas:.3f}, "
                f"experiment measured {causal_iroas:.3f} (scale factor: {calibration_factor:.2f})"
            )

    # --- Step 6: Fit quality ---
    y_pred = _hill_function(x_fit, r_max, k, alpha)
    ss_res = float(np.sum((y_fit - y_pred) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # --- Step 7: Current state ---
    n_treatment = len(fit_data)
    current_avg_per_dma = float(x.mean())
    current_inc_rev = float(_hill_function(np.array([current_avg_per_dma]), r_max, k, alpha)[0])
    current_iroas = current_inc_rev / current_avg_per_dma if current_avg_per_dma > 0 else 0.0
    current_marginal = _hill_marginal(current_avg_per_dma, r_max, k, alpha)

    # --- Step 8: Find optimal spend ---
    optimal_per_dma = _find_optimal_spend(r_max, k, alpha, current_avg_per_dma, target_marginal_roas)
    optimal_total = optimal_per_dma * n_treatment
    optimal_inc_rev = float(_hill_function(np.array([optimal_per_dma]), r_max, k, alpha)[0])
    optimal_iroas = optimal_inc_rev / optimal_per_dma if optimal_per_dma > 0 else 0.0

    # --- Step 9: Bootstrap confidence intervals ---
    optimal_lower, optimal_upper, param_samples = _bootstrap_ci(
        fit_data, organic_growth, causal_iroas,
        target_marginal_roas, n_treatment, n_bootstrap,
    )

    # Revenue CI bands from bootstrap parameter samples
    observed_spend_max = float(x.max())
    max_chart = max(current_avg_per_dma * 2.5, optimal_per_dma * 1.5, observed_spend_max * 1.5)
    spend_pts = np.linspace(0.01, max_chart, n_curve_points)
    rev_pts = _hill_function(spend_pts, r_max, k, alpha)
    marg_pts = np.array([_hill_marginal(s, r_max, k, alpha) for s in spend_pts])
    avg_roas_pts = np.where(spend_pts > 0, rev_pts / spend_pts, 0)

    rev_lower, rev_upper = _bootstrap_revenue_bands(spend_pts, param_samples)

    # Scale to total daily across all treatment DMAs
    spend_curve_total = (spend_pts * n_treatment).tolist()
    rev_curve_total = (rev_pts * n_treatment).tolist()
    rev_lower_total = (rev_lower * n_treatment).tolist()
    rev_upper_total = (rev_upper * n_treatment).tolist()

    # --- Step 10: Recommendation ---
    current_total = current_avg_per_dma * n_treatment
    spend_change_pct = (
        (optimal_total - current_total) / current_total * 100
        if current_total > 0 else 0.0
    )

    if spend_change_pct < -10:
        direction = "decrease"
        recommendation = (
            f"Reduce daily spend from ${current_total:,.0f} to "
            f"${optimal_total:,.0f} ({spend_change_pct:+.0f}%). "
            f"Current marginal ROAS is ${current_marginal:.2f} — "
            f"below the ${target_marginal_roas:.2f} breakeven threshold. "
            f"You're spending past the point of diminishing returns."
        )
    elif spend_change_pct > 10:
        direction = "increase"
        recommendation = (
            f"Increase daily spend from ${current_total:,.0f} to "
            f"${optimal_total:,.0f} ({spend_change_pct:+.0f}%). "
            f"Current marginal ROAS is ${current_marginal:.2f} — "
            f"still above ${target_marginal_roas:.2f}, meaning each "
            f"additional dollar returns more than it costs."
        )
    else:
        direction = "maintain"
        recommendation = (
            f"Current spend of ${current_total:,.0f}/day is near optimal. "
            f"Marginal ROAS is ${current_marginal:.2f}, close to the "
            f"${target_marginal_roas:.2f} target."
        )

    # Extrapolation warning
    if optimal_per_dma > observed_spend_max * 1.2:
        recommendation += (
            f" Note: optimal spend is {optimal_per_dma / observed_spend_max:.0%} "
            f"of the max observed spend per DMA — this is an extrapolation "
            f"beyond the data, so treat with caution."
        )

    return SpendResponseCurve(
        r_max=float(r_max),
        k=float(k),
        alpha=float(alpha),
        current_spend=float(current_total),
        current_iroas=float(current_iroas),
        current_marginal_roas=float(current_marginal),
        optimal_spend=float(optimal_total),
        optimal_iroas=float(optimal_iroas),
        optimal_spend_lower=float(optimal_lower * n_treatment),
        optimal_spend_upper=float(optimal_upper * n_treatment),
        target_marginal_roas=float(target_marginal_roas),
        spend_curve=spend_curve_total,
        revenue_curve=rev_curve_total,
        revenue_curve_lower=rev_lower_total,
        revenue_curve_upper=rev_upper_total,
        marginal_roas_curve=marg_pts.tolist(),
        avg_roas_curve=avg_roas_pts.tolist(),
        r_squared=float(r_squared),
        n_dmas_used=len(fit_data),
        converged=converged,
        calibrated=calibrated,
        observed_spend_max=float(observed_spend_max * n_treatment),
        spend_change_pct=float(spend_change_pct),
        spend_change_direction=direction,
        recommendation=recommendation,
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    fit_data: pd.DataFrame,
    organic_growth: float,
    causal_iroas: float | None,
    target_marginal_roas: float,
    n_treatment: int,
    n_bootstrap: int,
) -> tuple[float, float, list[tuple[float, float, float]]]:
    """Bootstrap confidence intervals on optimal spend and curve params.

    Resamples treatment DMAs with replacement, re-fits Hill function
    each time, and collects the distribution of optimal spend levels.

    Returns:
        (optimal_lower, optimal_upper, list of (r_max, k, alpha) samples)
    """
    rng = np.random.default_rng(42)
    dma_indices = np.arange(len(fit_data))
    optimal_samples: list[float] = []
    param_samples: list[tuple[float, float, float]] = []

    x_orig = fit_data["avg_daily_spend"].values
    y_orig = fit_data["incremental_revenue"].values
    current_avg = float(x_orig.mean())

    for _ in range(n_bootstrap):
        idx = rng.choice(dma_indices, size=len(dma_indices), replace=True)
        x_b = x_orig[idx]
        y_b = y_orig[idx]

        # Anchor at zero
        x_bf = np.concatenate([[0.0], x_b])
        y_bf = np.concatenate([[0.0], y_b])

        params, _ = _fit_hill(x_bf, y_bf)
        if params is None:
            continue
        r_max_b, k_b, alpha_b = params

        # Calibrate if available
        if causal_iroas is not None and causal_iroas > 0:
            total_spend = float(x_b.sum())
            pred_inc = float(_hill_function(x_b, r_max_b, k_b, alpha_b).sum())
            pred_iroas = pred_inc / total_spend if total_spend > 0 else 0
            if pred_iroas > 1e-6:
                r_max_b *= causal_iroas / pred_iroas

        opt = _find_optimal_spend(r_max_b, k_b, alpha_b, current_avg, target_marginal_roas)
        optimal_samples.append(opt)
        param_samples.append((r_max_b, k_b, alpha_b))

    if len(optimal_samples) < 10:
        # Not enough successful fits for reliable CI
        return current_avg * 0.5, current_avg * 2.0, param_samples

    optimal_arr = np.array(optimal_samples)
    return (
        float(np.percentile(optimal_arr, 2.5)),
        float(np.percentile(optimal_arr, 97.5)),
        param_samples,
    )


def _bootstrap_revenue_bands(
    spend_pts: np.ndarray,
    param_samples: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 95% CI bands on the revenue curve from bootstrap samples."""
    if len(param_samples) < 10:
        # Not enough samples — return point estimate as both bounds
        return np.zeros_like(spend_pts), np.full_like(spend_pts, np.inf)

    all_curves = np.array([
        _hill_function(spend_pts, r, k, a)
        for r, k, a in param_samples
    ])

    return (
        np.percentile(all_curves, 2.5, axis=0),
        np.percentile(all_curves, 97.5, axis=0),
    )
