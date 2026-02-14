"""Test design optimizer.

Automatically determines the optimal test design by analyzing historical data:
- How many DMAs to assign to treatment vs. holdout
- Test duration
- Which DMAs go in which cell
- Whether the test is feasible given data variance

Balances statistical power against opportunity cost of the holdout.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from incrementality.config import StatisticalConfig
from incrementality.design.matching import (
    build_test_cells,
    compute_balance_score,
    match_dmas_mahalanobis,
    match_dmas_rerandomization,
    prepare_matching_data,
    validate_balance,
)
from incrementality.design.power_analysis import (
    HistoricalVarianceEstimate,
    compute_mde,
    estimate_historical_variance,
    run_power_analysis,
)
from incrementality.dma import get_all_dmas
from incrementality.models import (
    AdChannel,
    DMAHistoricalMetrics,
    MeasurementScope,
    PowerAnalysisResult,
    TestDesign,
    TestScope,
    TestStatus,
)

logger = logging.getLogger(__name__)


def compute_dma_historical_metrics(
    shopify_daily: pd.DataFrame | None,
    amazon_daily: pd.DataFrame | None,
    facebook_daily: pd.DataFrame | None,
    youtube_daily: pd.DataFrame | None,
    period_start: date,
    period_end: date,
) -> list[DMAHistoricalMetrics]:
    """Compute historical metrics per DMA from raw daily data.

    Each input DataFrame should have columns: date, dma_code, revenue/spend, orders
    """
    # Collect all DMA codes that appear in any dataset
    all_codes = set()
    dfs = {
        "shopify": shopify_daily,
        "amazon": amazon_daily,
        "facebook": facebook_daily,
        "youtube": youtube_daily,
    }
    for name, df in dfs.items():
        if df is not None and not df.empty and "dma_code" in df.columns:
            all_codes.update(df["dma_code"].unique())

    if not all_codes:
        return []

    metrics = []
    for code in sorted(all_codes):
        m = DMAHistoricalMetrics(
            dma_code=code,
            period_start=period_start,
            period_end=period_end,
        )

        # Shopify
        if shopify_daily is not None and not shopify_daily.empty:
            mask = shopify_daily["dma_code"] == code
            subset = shopify_daily[mask]
            if not subset.empty:
                m.shopify_revenue = float(subset["revenue"].sum())
                m.shopify_orders = int(subset["orders"].sum())

        # Amazon
        if amazon_daily is not None and not amazon_daily.empty:
            mask = amazon_daily["dma_code"] == code
            subset = amazon_daily[mask]
            if not subset.empty:
                m.amazon_revenue = float(subset["revenue"].sum())
                m.amazon_orders = int(subset["orders"].sum())

        # Facebook spend
        if facebook_daily is not None and not facebook_daily.empty:
            mask = facebook_daily["dma_code"] == code
            subset = facebook_daily[mask]
            if not subset.empty:
                m.facebook_spend = float(subset["spend"].sum())

        # YouTube spend
        if youtube_daily is not None and not youtube_daily.empty:
            mask = youtube_daily["dma_code"] == code
            subset = youtube_daily[mask]
            if not subset.empty:
                m.youtube_spend = float(subset["spend"].sum())

        # Totals
        m.total_revenue = m.shopify_revenue + m.amazon_revenue
        m.total_orders = m.shopify_orders + m.amazon_orders
        m.total_ad_spend = m.facebook_spend + m.youtube_spend
        m.aov = m.total_revenue / m.total_orders if m.total_orders > 0 else 0

        # Trend and volatility from Shopify daily (primary signal)
        rev_source = shopify_daily if shopify_daily is not None else amazon_daily
        if rev_source is not None and not rev_source.empty:
            subset = rev_source[rev_source["dma_code"] == code].sort_values("date")
            if len(subset) >= 7:
                weekly = subset.groupby(
                    pd.Grouper(key="date", freq="W")
                )["revenue"].sum()
                if len(weekly) >= 2:
                    # Trend: average week-over-week growth
                    pct_changes = weekly.pct_change().dropna()
                    m.revenue_trend = float(pct_changes.mean()) if len(pct_changes) > 0 else 0
                    # Volatility: coefficient of variation
                    m.revenue_volatility = (
                        float(weekly.std() / weekly.mean()) if weekly.mean() > 0 else 0
                    )

        metrics.append(m)

    return metrics


def determine_optimal_holdout_size(
    variance_estimate: HistoricalVarianceEstimate,
    total_dmas: int,
    config: StatisticalConfig,
    target_mde: float = 0.15,
) -> tuple[int, int]:
    """Determine optimal number of holdout DMAs.

    Balances:
    - Statistical power (more holdout = more power)
    - Opportunity cost (more holdout = more lost revenue)
    - Minimum detectable effect (smaller holdout = larger MDE)

    The optimal holdout fraction for a two-sample test is typically
    n_holdout ≈ n_total * (1 - sqrt(cost_ratio)) where cost_ratio reflects
    the relative cost of adding a holdout vs treatment unit.

    For our case, holdout is more costly (lost revenue), so we prefer
    smaller holdout groups while maintaining adequate power.
    """
    max_holdout = int(total_dmas * config.max_holdout_fraction)
    min_holdout = config.min_dmas_per_cell
    target_holdout = int(total_dmas * config.target_holdout_fraction)

    best_n_holdout = target_holdout
    best_mde = float("inf")
    best_score = -float("inf")

    for n_h in range(min_holdout, max_holdout + 1):
        n_t = total_dmas - n_h
        if n_t < config.min_dmas_per_cell:
            continue

        # Compute MDE at 4 weeks (reasonable default)
        mde = compute_mde(
            variance_estimate, n_t, n_h,
            duration_weeks=4,
            alpha=config.significance_level,
            power=config.target_power,
        )

        # Score: penalize large holdout (opportunity cost) and large MDE
        holdout_penalty = n_h / total_dmas  # Fraction of market not advertised to
        mde_score = max(0, 1 - mde / 0.30)  # 0 if MDE > 30%, 1 if MDE = 0%

        # Combined score: weight power more than cost
        score = 0.7 * mde_score - 0.3 * holdout_penalty

        if score > best_score:
            best_score = score
            best_n_holdout = n_h
            best_mde = mde

    n_treatment = total_dmas - best_n_holdout
    logger.info(
        f"Optimal design: {n_treatment} treatment, {best_n_holdout} holdout "
        f"(MDE={best_mde:.1%} at 4 weeks)"
    )
    return n_treatment, best_n_holdout


def auto_design_test(
    shopify_daily: pd.DataFrame | None,
    amazon_daily: pd.DataFrame | None,
    facebook_daily: pd.DataFrame | None,
    youtube_daily: pd.DataFrame | None,
    ad_channel: AdChannel,
    test_scope: TestScope,
    measurement_scope: MeasurementScope,
    campaign_ids: list[str] | None = None,
    config: StatisticalConfig | None = None,
    test_name: str = "Incrementality Test",
    target_mde: float | None = None,
) -> TestDesign:
    """Automatically design an optimal geo holdout test.

    This is the main entry point for test design. It:
    1. Analyzes historical data to estimate variance
    2. Determines optimal holdout size
    3. Matches DMAs into balanced treatment/holdout cells
    4. Runs power analysis
    5. Recommends test duration
    6. Returns a complete TestDesign ready to execute

    Args:
        shopify_daily: Shopify revenue by DMA by day
        amazon_daily: Amazon revenue by DMA by day
        facebook_daily: Facebook spend by DMA by day
        youtube_daily: YouTube spend by DMA by day
        ad_channel: Which channel to hold out
        test_scope: Channel-level or campaign-level holdout
        measurement_scope: What revenue to measure
        campaign_ids: Specific campaigns (for campaign-level tests)
        config: Statistical configuration
        test_name: Human-readable name for the test
        target_mde: Target minimum detectable effect (optional)
    """
    config = config or StatisticalConfig()

    # Determine date range from available data
    all_dates = []
    for df in [shopify_daily, amazon_daily, facebook_daily, youtube_daily]:
        if df is not None and not df.empty and "date" in df.columns:
            all_dates.extend(pd.to_datetime(df["date"]).dt.date)
    if not all_dates:
        raise ValueError("No historical data provided")

    period_start = min(all_dates)
    period_end = max(all_dates)

    # Step 1: Compute per-DMA historical metrics
    logger.info("Computing historical DMA metrics...")
    dma_metrics = compute_dma_historical_metrics(
        shopify_daily, amazon_daily, facebook_daily, youtube_daily,
        period_start, period_end,
    )
    if len(dma_metrics) < config.min_dmas_per_cell * 2:
        raise ValueError(
            f"Need at least {config.min_dmas_per_cell * 2} DMAs with data, "
            f"found {len(dma_metrics)}"
        )

    # Step 2: Estimate variance from historical data
    logger.info("Estimating historical variance...")
    # Build combined daily revenue data based on measurement scope
    rev_frames = []
    if measurement_scope in (MeasurementScope.SHOPIFY_ONLY, MeasurementScope.SHOPIFY_AND_AMAZON):
        if shopify_daily is not None and not shopify_daily.empty:
            rev_frames.append(shopify_daily[["date", "dma_code", "revenue"]])
    if measurement_scope in (MeasurementScope.AMAZON_ONLY, MeasurementScope.SHOPIFY_AND_AMAZON):
        if amazon_daily is not None and not amazon_daily.empty:
            rev_frames.append(amazon_daily[["date", "dma_code", "revenue"]])

    if not rev_frames:
        raise ValueError("No revenue data for the specified measurement scope")

    combined_daily = pd.concat(rev_frames).groupby(["date", "dma_code"]).sum().reset_index()
    variance_estimate = estimate_historical_variance(combined_daily)

    # Step 3: Get DMA populations
    all_dmas = get_all_dmas()
    dma_populations = {d.dma_code: d.population for d in all_dmas}
    dma_regions = {d.dma_code: d.region for d in all_dmas}

    # Step 4: Determine optimal holdout size
    total_dmas = len(dma_metrics)
    n_treatment, n_holdout = determine_optimal_holdout_size(
        variance_estimate, total_dmas, config, target_mde or 0.15,
    )

    # Step 5: Match DMAs into balanced cells
    logger.info(f"Matching DMAs: {n_treatment} treatment, {n_holdout} holdout...")
    matching_df = prepare_matching_data(dma_metrics, dma_populations)

    # Try re-randomization first (gold standard), fall back to Mahalanobis
    try:
        treatment_codes, holdout_codes = match_dmas_rerandomization(
            matching_df, n_holdout,
            n_iterations=10000,
            balance_threshold=config.balance_tolerance,
        )
    except ValueError:
        logger.warning("Re-randomization failed, falling back to Mahalanobis matching")
        treatment_codes, holdout_codes = match_dmas_mahalanobis(
            matching_df, n_holdout,
        )

    # Step 6: Validate balance
    is_balanced, balance_score, smds = validate_balance(
        matching_df, treatment_codes, holdout_codes, config.balance_tolerance,
    )
    if not is_balanced:
        logger.warning(
            f"Assignment balance is suboptimal (score={balance_score:.2f}). "
            f"Covariate SMDs: {smds}"
        )

    # Step 7: Build cells
    treatment_cell, holdout_cell = build_test_cells(
        matching_df, treatment_codes, holdout_codes,
    )

    # Step 8: Run power analysis
    logger.info("Running power analysis...")
    power_result = run_power_analysis(
        variance_estimate, n_treatment, n_holdout, config, target_mde,
    )

    # Step 9: Determine timing
    duration_weeks = power_result.recommended_duration_weeks
    # Recommend starting on a Monday
    today = date.today()
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    start_date = today + timedelta(days=days_until_monday)
    end_date = start_date + timedelta(weeks=duration_weeks)

    # Step 10: Assemble test design
    import uuid
    test_id = f"test_{uuid.uuid4().hex[:8]}"

    design = TestDesign(
        test_id=test_id,
        name=test_name,
        description=(
            f"Geo holdout test for {ad_channel.value} "
            f"({'campaign-level' if test_scope == TestScope.CAMPAIGN else 'channel-level'}) "
            f"measuring {measurement_scope.value} revenue"
        ),
        test_scope=test_scope,
        ad_channel=ad_channel,
        campaign_ids=campaign_ids or [],
        measurement_scope=measurement_scope,
        lookback_weeks=config.default_lookback_weeks,
        recommended_start_date=start_date,
        recommended_end_date=end_date,
        duration_weeks=duration_weeks,
        treatment_cell=treatment_cell,
        holdout_cell=holdout_cell,
        num_treatment_dmas=n_treatment,
        num_holdout_dmas=n_holdout,
        power_analysis=power_result,
        balance_score=balance_score,
        status=TestStatus.DESIGNED,
    )

    logger.info(
        f"Test designed: {n_treatment} treatment / {n_holdout} holdout DMAs, "
        f"{duration_weeks} weeks, MDE={power_result.minimum_detectable_effect:.1%}"
    )

    return design
