"""Test design optimizer.

Automatically determines the optimal test design by analyzing historical data:
- How many DMAs to assign to treatment vs. holdout
- Test duration
- Which DMAs go in which cell
- Whether the test is feasible given data variance
- Spillover risk assessment

Includes a pre-test feasibility gate: rejects underpowered tests before
they run. Haus targets a power score of 85-90 before greenlighting.

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
    run_simulation_power_analysis,
)
from incrementality.design.spillover import (
    apply_geographic_buffer,
    compute_spillover_risk,
)
from incrementality.dma import get_all_dmas
from incrementality.models import (
    AdChannel,
    DMAHistoricalMetrics,
    FeasibilityResult,
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
    """Compute historical metrics per DMA from raw daily data."""
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

        if shopify_daily is not None and not shopify_daily.empty:
            mask = shopify_daily["dma_code"] == code
            subset = shopify_daily[mask]
            if not subset.empty:
                m.shopify_revenue = float(subset["revenue"].sum())
                m.shopify_orders = int(subset["orders"].sum())

        if amazon_daily is not None and not amazon_daily.empty:
            mask = amazon_daily["dma_code"] == code
            subset = amazon_daily[mask]
            if not subset.empty:
                m.amazon_revenue = float(subset["revenue"].sum())
                m.amazon_orders = int(subset["orders"].sum())

        if facebook_daily is not None and not facebook_daily.empty:
            mask = facebook_daily["dma_code"] == code
            subset = facebook_daily[mask]
            if not subset.empty:
                m.facebook_spend = float(subset["spend"].sum())

        if youtube_daily is not None and not youtube_daily.empty:
            mask = youtube_daily["dma_code"] == code
            subset = youtube_daily[mask]
            if not subset.empty:
                m.youtube_spend = float(subset["spend"].sum())

        m.total_revenue = m.shopify_revenue + m.amazon_revenue
        m.total_orders = m.shopify_orders + m.amazon_orders
        m.total_ad_spend = m.facebook_spend + m.youtube_spend
        m.aov = m.total_revenue / m.total_orders if m.total_orders > 0 else 0

        # Compute trend/volatility from combined revenue sources (not just one)
        rev_frames = []
        if shopify_daily is not None and not shopify_daily.empty:
            s = shopify_daily[shopify_daily["dma_code"] == code][["date", "revenue"]].copy()
            if not s.empty:
                rev_frames.append(s)
        if amazon_daily is not None and not amazon_daily.empty:
            a = amazon_daily[amazon_daily["dma_code"] == code][["date", "revenue"]].copy()
            if not a.empty:
                rev_frames.append(a)

        if rev_frames:
            combined = pd.concat(rev_frames, ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.groupby("date")["revenue"].sum().reset_index()
            combined = combined.sort_values("date")
            if len(combined) >= 7:
                weekly = combined.groupby(
                    pd.Grouper(key="date", freq="W")
                )["revenue"].sum()
                if len(weekly) >= 2:
                    pct_changes = weekly.pct_change().dropna()
                    m.revenue_trend = float(pct_changes.mean()) if len(pct_changes) > 0 else 0
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

    Balances statistical power against opportunity cost.
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

        mde = compute_mde(
            variance_estimate, n_t, n_h,
            duration_weeks=4,
            alpha=config.significance_level,
            power=config.target_power,
        )

        holdout_penalty = n_h / total_dmas
        mde_score = max(0, 1 - mde / 0.30)

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


# =====================================================================
# Pre-test feasibility gate
# =====================================================================

def run_feasibility_check(
    variance_estimate: HistoricalVarianceEstimate,
    n_treatment: int,
    n_holdout: int,
    power_result: PowerAnalysisResult,
    holdout_historical_revenue: float,
    test_duration_weeks: int,
    min_power_score: float = 70.0,
) -> FeasibilityResult:
    """Pre-test feasibility gate.

    Haus doesn't run tests they know will fail. Neither should you.
    This gate checks whether the test design has enough statistical
    power to produce actionable results.

    Checks:
    1. Enough DMAs in both cells (>= 5 each)
    2. Enough historical data (>= 4 weeks)
    3. Acceptable MDE (<= 30%)
    4. Acceptable simulated power (>= 60%)
    5. Power score >= threshold (default 70, Haus targets 85-90)
    6. Estimated opportunity cost of holdout
    """
    reasons = []
    recommendations = []
    is_feasible = True
    mde = power_result.minimum_detectable_effect
    power_score = power_result.power_score

    if power_score == 0:
        power_score = _estimate_power_score_analytical(
            power_result.statistical_power,
            mde,
            n_holdout,
        )

    # Check 1: Minimum DMAs
    if n_holdout < 5:
        is_feasible = False
        reasons.append(
            f"BLOCK: Only {n_holdout} holdout DMAs. Need at least 5 for "
            f"valid inference. Synthetic control needs donor pool diversity."
        )
    if n_treatment < 5:
        is_feasible = False
        reasons.append(
            f"BLOCK: Only {n_treatment} treatment DMAs. Need at least 5."
        )

    # Check 2: Data sufficiency
    if variance_estimate.num_periods < 4:
        is_feasible = False
        reasons.append(
            f"BLOCK: Only {variance_estimate.num_periods} weeks of historical data. "
            f"Need at least 4 weeks for reliable variance estimation."
        )
        recommendations.append("Collect more historical data before running test.")

    # Check 3: MDE
    if mde > 0.30:
        is_feasible = False
        reasons.append(
            f"BLOCK: MDE is {mde:.0%}. You can only detect effects larger than "
            f"30%. This is too coarse for actionable decisions. Most ad channels "
            f"have true lifts of 5-20%."
        )
        recommendations.append(
            "Increase holdout size, extend test duration, or wait for "
            "more stable revenue patterns."
        )
    elif mde > 0.20:
        reasons.append(
            f"WARNING: MDE is {mde:.0%}. You'll only detect large effects. "
            f"Consider extending test duration."
        )

    # Check 4: Power
    sim_power = power_result.simulated_power
    if sim_power > 0:
        if sim_power < 0.50:
            is_feasible = False
            reasons.append(
                f"BLOCK: Simulated power is only {sim_power:.0%}. "
                f"More than half the time, you'd miss a real effect. "
                f"This test is a waste of time and money."
            )
        elif sim_power < 0.70:
            reasons.append(
                f"WARNING: Simulated power is {sim_power:.0%}. "
                f"Adequate but not ideal. Consider running longer."
            )
    elif power_result.statistical_power < 0.60:
        reasons.append(
            f"WARNING: Analytical power is {power_result.statistical_power:.0%}. "
            f"Run simulation-based power analysis for a more accurate estimate."
        )

    # Check 5: Power score
    if power_score < min_power_score:
        if power_score < 50:
            is_feasible = False
            reasons.append(
                f"BLOCK: Power score is {power_score:.0f}/100 "
                f"(need >= {min_power_score:.0f}). Test design is inadequate."
            )
        else:
            reasons.append(
                f"WARNING: Power score is {power_score:.0f}/100 "
                f"(target >= {min_power_score:.0f}). Test may produce ambiguous results."
            )

    # Check 6: FPR
    fpr = power_result.simulated_false_positive_rate
    if fpr > 0.15:
        is_feasible = False
        reasons.append(
            f"BLOCK: False positive rate is {fpr:.0%} (expected ~5%). "
            f"The methodology produces too many spurious results on this data."
        )
    elif fpr > 0.10:
        reasons.append(
            f"WARNING: False positive rate is {fpr:.0%}. Elevated but acceptable."
        )

    # Opportunity cost
    weekly_holdout_revenue = holdout_historical_revenue / max(variance_estimate.num_periods, 1)
    opportunity_cost = weekly_holdout_revenue * test_duration_weeks * mde

    if is_feasible:
        reasons.append(
            f"PASS: Test design is feasible. Power score: {power_score:.0f}/100. "
            f"Estimated opportunity cost: ${opportunity_cost:,.0f}."
        )
        recommendations.append(
            f"Proceed with test. Expected to detect effects >= {mde:.0%} "
            f"with {max(sim_power, power_result.statistical_power):.0%} probability."
        )

    return FeasibilityResult(
        is_feasible=is_feasible,
        power_score=power_score,
        estimated_mde=mde,
        estimated_duration_weeks=test_duration_weeks,
        min_holdout_dmas=n_holdout,
        estimated_opportunity_cost=float(opportunity_cost),
        reasons=reasons,
        recommendations=recommendations,
    )


def _estimate_power_score_analytical(
    analytical_power: float,
    mde: float,
    n_holdout: int,
) -> float:
    """Rough power score estimate from analytical results."""
    score = 0.0

    if analytical_power >= 0.80:
        score += 40.0
    else:
        score += 40.0 * analytical_power / 0.80

    score += 15.0

    if mde <= 0.10:
        score += 25.0
    elif mde <= 0.15:
        score += 20.0
    elif mde <= 0.20:
        score += 12.0
    elif mde <= 0.30:
        score += 5.0

    if n_holdout >= 20:
        score += 15.0
    elif n_holdout >= 10:
        score += 10.0
    elif n_holdout >= 5:
        score += 5.0

    return min(100.0, max(0.0, score))


# =====================================================================
# Main design entry point
# =====================================================================

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
    run_simulation: bool = True,
) -> TestDesign:
    """Automatically design an optimal geo holdout test.

    This is the main entry point for test design. It:
    1. Analyzes historical data to estimate variance
    2. Determines optimal holdout size
    3. Matches DMAs into balanced treatment/holdout cells
    4. Assesses spillover risk and applies geographic buffer
    5. Runs power analysis (analytical + simulation)
    6. Runs feasibility check
    7. Recommends test duration
    8. Returns a complete TestDesign ready to execute
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

    # Step 4: Determine optimal holdout size
    total_dmas = len(dma_metrics)
    n_treatment, n_holdout = determine_optimal_holdout_size(
        variance_estimate, total_dmas, config, target_mde or 0.15,
    )

    # Step 5: Match DMAs into balanced cells
    logger.info(f"Matching DMAs: {n_treatment} treatment, {n_holdout} holdout...")
    matching_df = prepare_matching_data(dma_metrics, dma_populations)

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

    # Step 6: Spillover risk assessment and geographic buffer
    logger.info("Assessing spillover risk...")
    spillover = compute_spillover_risk(treatment_codes, holdout_codes)
    if spillover["risk_score"] > 0:
        logger.info(
            f"Spillover risk: {spillover['risk_score']:.0%} "
            f"({len(spillover['border_pairs'])} border pairs)"
        )
        for rec in spillover["recommendations"]:
            logger.info(f"  {rec}")

    # Apply geographic buffer for analysis (not execution)
    analysis_treatment, analysis_holdout, buffer_dmas = apply_geographic_buffer(
        treatment_codes, holdout_codes,
        min_holdout=config.min_dmas_per_cell,
    )
    if buffer_dmas:
        logger.info(f"Geographic buffer: {len(buffer_dmas)} DMAs excluded from analysis")

    # Step 7: Validate balance
    is_balanced, balance_score, smds = validate_balance(
        matching_df, treatment_codes, holdout_codes, config.balance_tolerance,
    )
    if not is_balanced:
        logger.warning(
            f"Assignment balance is suboptimal (score={balance_score:.2f}). "
            f"Covariate SMDs: {smds}"
        )

    # Step 8: Build cells
    treatment_cell, holdout_cell = build_test_cells(
        matching_df, treatment_codes, holdout_codes,
    )

    # Step 9: Run power analysis (using post-buffer DMA counts for accuracy)
    logger.info("Running power analysis...")
    n_analysis_treatment = len(analysis_treatment)
    n_analysis_holdout = len(analysis_holdout)
    power_result = run_power_analysis(
        variance_estimate, n_analysis_treatment, n_analysis_holdout, config, target_mde,
    )

    # Step 9b: Run simulation-based power analysis if enough data
    if run_simulation and len(combined_daily) > 0:
        try:
            logger.info("Running simulation-based power analysis...")
            sim_power = run_simulation_power_analysis(
                combined_daily,
                analysis_treatment,
                analysis_holdout,
                test_duration_weeks=power_result.recommended_duration_weeks,
                n_simulations=50,
                alpha=config.significance_level,
            )
            power_result.simulated_power = sim_power.simulated_power
            power_result.simulated_false_positive_rate = sim_power.simulated_false_positive_rate
            power_result.num_simulations = sim_power.num_simulations
            power_result.power_score = sim_power.power_score
        except Exception as e:
            logger.warning(f"Simulation power analysis failed: {e}")

    # Step 10: Feasibility check (using post-buffer counts)
    logger.info("Running feasibility check...")
    feasibility = run_feasibility_check(
        variance_estimate,
        n_analysis_treatment,
        n_analysis_holdout,
        power_result,
        holdout_cell.historical_revenue,
        power_result.recommended_duration_weeks,
    )
    for reason in feasibility.reasons:
        logger.info(f"  {reason}")
    if not feasibility.is_feasible:
        logger.warning(
            "TEST DESIGN IS NOT FEASIBLE. Proceeding anyway but results "
            "may not be actionable. Review reasons above."
        )

    # Step 11: Determine timing
    duration_weeks = power_result.recommended_duration_weeks
    today = date.today()
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    start_date = today + timedelta(days=days_until_monday)
    end_date = start_date + timedelta(weeks=duration_weeks)

    # Step 12: Assemble test design
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
        f"{duration_weeks} weeks, MDE={power_result.minimum_detectable_effect:.1%}, "
        f"power score={power_result.power_score:.0f}"
    )

    return design
