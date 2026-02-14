"""Incremental ROAS (iROAS) calculation.

Computes the incremental return on ad spend by combining:
- Incremental revenue estimates from causal inference
- Ad spend data from the treatment group during the test period

Supports:
- Overall iROAS (combined Shopify + Amazon)
- Shopify-only iROAS
- Amazon-only iROAS
- Cross-platform halo effect measurement
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy import stats

from incrementality.analysis.estimators import difference_in_differences
from incrementality.models import (
    CellType,
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
    # absolute_lift is per-DMA average daily lift
    total = lift_result.absolute_lift * num_treatment_dmas * test_duration_days
    lower = lift_result.lift_lower_ci  # These are relative
    upper = lift_result.lift_upper_ci

    # Need absolute CI bounds
    # lift_lower_ci and lift_upper_ci are relative lifts
    # Convert back: absolute = relative * baseline * n_dmas * days
    # We approximate using the same scaling
    baseline_per_dma_day = (
        lift_result.absolute_lift / lift_result.relative_lift
        if lift_result.relative_lift != 0
        else 0
    )
    total_baseline = baseline_per_dma_day * num_treatment_dmas * test_duration_days

    lower_abs = lower * total_baseline
    upper_abs = upper * total_baseline

    return total, lower_abs, upper_abs


def compute_iroas(
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    ad_spend_data: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    measurement_scope: MeasurementScope,
    test_duration_days: int,
    alpha: float = 0.05,
) -> IncrementalROAS:
    """Compute incremental ROAS from test data.

    Args:
        pre_data: Pre-test period daily data with columns
                  [date, dma_code, revenue, shopify_revenue, amazon_revenue]
        post_data: Test period daily data (same columns)
        ad_spend_data: Ad spend during test period
                       [date, dma_code, spend]
        treatment_dmas: Treatment cell DMA codes
        holdout_dmas: Holdout cell DMA codes
        measurement_scope: Which revenue to measure
        test_duration_days: Number of days in test period
        alpha: Significance level
    """
    # Total ad spend in treatment during test
    treatment_spend = ad_spend_data[
        ad_spend_data["dma_code"].isin(treatment_dmas)
    ]["spend"].sum()

    if treatment_spend <= 0:
        logger.warning("No ad spend in treatment group — cannot compute iROAS")
        return IncrementalROAS(
            incremental_revenue=0,
            total_ad_spend=0,
            iroas=0,
            iroas_lower_ci=0,
            iroas_upper_ci=0,
        )

    n_treatment = len(treatment_dmas)

    # --- Overall incrementality ---
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
        # Shopify-only iROAS
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

        # Amazon-only iROAS (halo effect!)
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

    elif measurement_scope == MeasurementScope.SHOPIFY_ONLY:
        result.shopify_incremental_revenue = float(inc_rev)
        result.shopify_iroas = float(iroas)

    elif measurement_scope == MeasurementScope.AMAZON_ONLY:
        result.amazon_incremental_revenue = float(inc_rev)
        result.amazon_iroas = float(iroas)

    return result


def _get_revenue_col(scope: MeasurementScope) -> str:
    """Get the revenue column name for a measurement scope."""
    if scope == MeasurementScope.SHOPIFY_ONLY:
        return "shopify_revenue"
    elif scope == MeasurementScope.AMAZON_ONLY:
        return "amazon_revenue"
    else:
        return "revenue"  # Combined total
