"""Incremental ROAS (iROAS) calculation.

Computes the incremental return on ad spend by combining:
- Incremental revenue estimates from causal inference
- Ad spend data from the treatment group during the test period

Supports:
- Overall iROAS (combined Shopify + Amazon)
- Shopify-only iROAS
- Amazon-only iROAS
- Cross-platform halo effect measurement

Uses ensemble estimator for production-grade estimates. Falls back to
DiD only when ensemble is not available.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from incrementality.models import (
    IncrementalityResult,
    IncrementalROAS,
    MeasurementScope,
)

logger = logging.getLogger(__name__)


def compute_incremental_revenue(
    lift_result: IncrementalityResult,
    num_treatment_dmas: int,
    test_duration_days: int,
) -> tuple[float, float, float]:
    """Scale the per-DMA-per-day lift to total incremental revenue.

    Returns: (incremental_revenue, lower_ci, upper_ci)
    """
    scale = num_treatment_dmas * test_duration_days
    total = lift_result.absolute_lift * scale

    # Compute baseline safely — guard against near-zero relative_lift
    if abs(lift_result.relative_lift) > 1e-6:
        baseline_per_dma_day = lift_result.absolute_lift / lift_result.relative_lift
    else:
        # Cannot reliably derive baseline from near-zero relative lift.
        # Use absolute lift bounds directly (they are already per-DMA-per-day).
        baseline_per_dma_day = 0.0

    if baseline_per_dma_day > 0:
        total_baseline = baseline_per_dma_day * scale
        # CI bounds are relative — convert to absolute
        lower_abs = lift_result.lift_lower_ci * total_baseline
        upper_abs = lift_result.lift_upper_ci * total_baseline
    else:
        # Fallback: no reliable baseline, report total with no CI spread
        lower_abs = total
        upper_abs = total

    return total, lower_abs, upper_abs


def compute_iroas(
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    ad_spend_data: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    measurement_scope: MeasurementScope,
    test_duration_days: int,
    primary_result: IncrementalityResult | None = None,
    alpha: float = 0.05,
    attributed_conversions: float = 0.0,
    orders_col: str | None = None,
) -> IncrementalROAS:
    """Compute incremental ROAS from test data.

    Args:
        pre_data: Pre-test period daily data
        post_data: Test period daily data
        ad_spend_data: Ad spend during test period [date, dma_code, spend]
        treatment_dmas: Treatment cell DMA codes
        holdout_dmas: Holdout cell DMA codes
        measurement_scope: Which revenue to measure
        test_duration_days: Number of days in test period
        primary_result: Pre-computed primary incrementality result (from ensemble).
                        If None, falls back to running DiD.
        alpha: Significance level
        attributed_conversions: Platform-reported conversions (from ad platform).
                                Used to compute Incrementality Factor (IF).
        orders_col: Column name for order counts in post_data. Used to compute
                    incremental conversions for IF and CPIA.
    """
    # Total ad spend in treatment during test
    treatment_spend = ad_spend_data[
        ad_spend_data["dma_code"].isin(treatment_dmas)
    ]["spend"].sum()

    if treatment_spend <= 0:
        logger.warning("No ad spend in treatment group -- cannot compute iROAS")
        return IncrementalROAS(
            incremental_revenue=0,
            total_ad_spend=0,
            iroas=0,
            iroas_lower_ci=0,
            iroas_upper_ci=0,
        )

    n_treatment = len(treatment_dmas)

    # --- Overall incrementality ---
    if primary_result is not None:
        overall_result = primary_result
    else:
        # Fallback: run DiD
        from incrementality.analysis.estimators import difference_in_differences
        revenue_col = _get_revenue_col(measurement_scope)
        overall_result = difference_in_differences(
            pre_data, post_data, treatment_dmas, holdout_dmas,
            revenue_col=revenue_col, alpha=alpha,
        )

    inc_rev, inc_lower, inc_upper = compute_incremental_revenue(
        overall_result, n_treatment, test_duration_days,
    )

    iroas = inc_rev / treatment_spend
    iroas_lower = inc_lower / treatment_spend
    iroas_upper = inc_upper / treatment_spend

    result = IncrementalROAS(
        incremental_revenue=float(inc_rev),
        total_ad_spend=float(treatment_spend),
        iroas=float(iroas),
        iroas_lower_ci=float(iroas_lower),
        iroas_upper_ci=float(iroas_upper),
    )

    # --- Platform-specific breakdown ---
    if measurement_scope == MeasurementScope.SHOPIFY_AND_AMAZON:
        _compute_platform_breakdown(
            result, pre_data, post_data, treatment_dmas, holdout_dmas,
            n_treatment, test_duration_days, treatment_spend, alpha,
        )
    elif measurement_scope == MeasurementScope.SHOPIFY_ONLY:
        result.shopify_incremental_revenue = float(inc_rev)
        result.shopify_iroas = float(iroas)
    elif measurement_scope == MeasurementScope.AMAZON_ONLY:
        result.amazon_incremental_revenue = float(inc_rev)
        result.amazon_iroas = float(iroas)

    # --- IF and CPIA computation ---
    _compute_if_cpia(
        result, post_data, treatment_dmas, holdout_dmas,
        treatment_spend, overall_result, test_duration_days,
        attributed_conversions, orders_col,
    )

    return result


def _compute_if_cpia(
    result: IncrementalROAS,
    post_data: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    treatment_spend: float,
    primary_result: IncrementalityResult,
    test_duration_days: int,
    attributed_conversions: float = 0.0,
    orders_col: str | None = None,
) -> None:
    """Compute Incrementality Factor (IF) and Cost Per Incremental Acquisition (CPIA).

    IF = incremental_conversions / attributed_conversions
    - IF > 1.0: Ads drive MORE conversions than the platform reports (under-attribution)
    - IF = 1.0: Perfect attribution
    - IF < 1.0: Platform over-counts (some conversions would happen anyway)
    - IF = 0.0: No incremental conversions; all are organic

    CPIA = total_ad_spend / incremental_conversions
    - The true cost to acquire each incremental customer
    - Use for cross-channel comparison (compare Facebook CPIA vs YouTube CPIA)

    Incremental conversions are estimated from the experiment: we apply
    the measured lift to the holdout group's order rate to estimate how
    many extra orders the treatment caused.
    """
    # Try to compute incremental conversions from order data
    incremental_conversions = 0.0

    # Auto-detect orders column
    if orders_col is None:
        for candidate in ["orders", "shopify_orders", "total_orders"]:
            if candidate in post_data.columns:
                orders_col = candidate
                break

    if orders_col and orders_col in post_data.columns:
        # Treatment group orders per DMA per day
        treatment_orders = post_data[
            post_data["dma_code"].isin(treatment_dmas)
        ].groupby("dma_code")[orders_col].mean()

        # Holdout group orders per DMA per day (the counterfactual)
        holdout_orders = post_data[
            post_data["dma_code"].isin(holdout_dmas)
        ].groupby("dma_code")[orders_col].mean()

        if len(treatment_orders) > 0 and len(holdout_orders) > 0:
            treatment_mean = treatment_orders.mean()
            holdout_mean = holdout_orders.mean()

            # Incremental orders per DMA per day
            incremental_per_dma_day = treatment_mean - holdout_mean

            # Scale to total incremental conversions
            incremental_conversions = max(
                0.0,
                incremental_per_dma_day * len(treatment_dmas) * test_duration_days,
            )

            logger.info(
                f"Incremental conversions: {incremental_conversions:.0f} "
                f"(treatment avg: {treatment_mean:.1f}/DMA/day, "
                f"holdout avg: {holdout_mean:.1f}/DMA/day)"
            )
    elif primary_result.relative_lift > 0:
        # Fallback: estimate from lift and total treatment conversions
        # If we don't have orders data, we can't compute this
        logger.debug(
            "No orders column found in data. IF/CPIA requires order-level data."
        )

    # Populate results
    if incremental_conversions > 0:
        result.incremental_conversions = float(incremental_conversions)
        result.cpia = float(treatment_spend / incremental_conversions)

        if attributed_conversions > 0:
            result.attributed_conversions = float(attributed_conversions)
            result.incrementality_factor = float(
                incremental_conversions / attributed_conversions
            )
            logger.info(
                f"IF = {result.incrementality_factor:.2f} "
                f"({incremental_conversions:.0f} incremental / "
                f"{attributed_conversions:.0f} attributed), "
                f"CPIA = ${result.cpia:.2f}"
            )
        else:
            logger.info(
                f"CPIA = ${result.cpia:.2f} "
                f"(no attributed conversions provided for IF)"
            )


def _compute_platform_breakdown(
    result: IncrementalROAS,
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    n_treatment: int,
    test_duration_days: int,
    treatment_spend: float,
    alpha: float,
) -> None:
    """Compute Shopify and Amazon iROAS breakdown."""
    from incrementality.analysis.estimators import difference_in_differences

    if "shopify_revenue" in post_data.columns:
        try:
            shopify_result = difference_in_differences(
                pre_data, post_data, treatment_dmas, holdout_dmas,
                revenue_col="shopify_revenue", alpha=alpha,
            )
            shopify_inc, _, _ = compute_incremental_revenue(
                shopify_result, n_treatment, test_duration_days,
            )
            result.shopify_incremental_revenue = float(shopify_inc)
            result.shopify_iroas = float(shopify_inc / treatment_spend)
        except Exception as e:
            logger.warning(f"Shopify-specific iROAS failed: {e}")

    if "amazon_revenue" in post_data.columns:
        try:
            amazon_result = difference_in_differences(
                pre_data, post_data, treatment_dmas, holdout_dmas,
                revenue_col="amazon_revenue", alpha=alpha,
            )
            amazon_inc, _, _ = compute_incremental_revenue(
                amazon_result, n_treatment, test_duration_days,
            )
            result.amazon_incremental_revenue = float(amazon_inc)
            result.amazon_iroas = float(amazon_inc / treatment_spend)
        except Exception as e:
            logger.warning(f"Amazon-specific iROAS failed: {e}")


def _get_revenue_col(scope: MeasurementScope) -> str:
    """Get the revenue column name for a measurement scope."""
    if scope == MeasurementScope.SHOPIFY_ONLY:
        return "shopify_revenue"
    elif scope == MeasurementScope.AMAZON_ONLY:
        return "amazon_revenue"
    else:
        return "revenue"
