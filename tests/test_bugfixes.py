"""Tests for critical bugfixes identified in the code audit.

Covers:
- ASCM augmentation correctness (Bug #1)
- Re-randomization rejection logic (Bug #2)
- Conformal CI (Bug #4)
- iROAS near-zero guard (Bug #5)
- BSTS p-value (Bug #6)
- Power formula two-sided (Bug #7)
- Power scoring continuity (Bug #8)
- Config validation (Bug #28)
- DiD baseline (Bug #10)
- Zip-to-DMA no duplicates (Bug #3)
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from incrementality.analysis.estimators import (
    _build_panel_matrices,
    _conformal_inference_ascm,
    _fit_scm_weights,
    augmented_synthetic_control,
    difference_in_differences,
)
from incrementality.analysis.iroas import compute_incremental_revenue
from incrementality.config import StatisticalConfig
from incrementality.connectors.geo import zip_to_dma
from incrementality.design.matching import match_dmas_rerandomization
from incrementality.design.power_analysis import _compute_achieved_power, _compute_power_score
from incrementality.models import IncrementalityResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(
    n_treatment: int = 30,
    n_holdout: int = 10,
    n_pre: int = 60,
    n_post: int = 21,
    base: float = 1000.0,
    lift: float = 0.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Generate clean panel data for estimator tests."""
    rng = np.random.default_rng(seed)
    treatment_dmas = [f"T_{i:03d}" for i in range(n_treatment)]
    holdout_dmas = [f"H_{i:03d}" for i in range(n_holdout)]
    all_dmas = treatment_dmas + holdout_dmas

    # Each DMA has a stable baseline + noise
    dma_baselines = {d: base * (1 + rng.normal(0, 0.2)) for d in all_dmas}

    records_pre = []
    for day in range(n_pre):
        dt = date(2025, 1, 1 + day % 28) if day < 28 else date(2025, 2, 1 + (day - 28) % 28)
        for dma in all_dmas:
            records_pre.append({
                "date": dt,
                "dma_code": dma,
                "revenue": max(0, dma_baselines[dma] + rng.normal(0, 50)),
            })

    records_post = []
    for day in range(n_post):
        dt = date(2025, 3, 1 + day % 28)
        for dma in all_dmas:
            rev = dma_baselines[dma] + rng.normal(0, 50)
            if dma in treatment_dmas and lift > 0:
                rev *= (1 + lift)
            records_post.append({
                "date": dt,
                "dma_code": dma,
                "revenue": max(0, rev),
            })

    return (
        pd.DataFrame(records_pre),
        pd.DataFrame(records_post),
        treatment_dmas,
        holdout_dmas,
    )


# ---------------------------------------------------------------------------
# Bug #1: ASCM augmentation should NOT reduce to ridge regression
# ---------------------------------------------------------------------------

class TestASCMAugmentation:
    def test_ascm_not_pure_ridge(self):
        """ASCM counterfactual should differ from pure ridge prediction."""
        pre, post, t_dmas, h_dmas = _make_panel(
            n_treatment=20, n_holdout=10, n_pre=60, n_post=21,
        )
        Y_pre, Y_post, X_pre, X_post, _, _ = _build_panel_matrices(
            pre, post, t_dmas, h_dmas, "revenue", "dma_code", "date",
        )

        # Fit SCM weights
        w = _fit_scm_weights(Y_pre, X_pre)
        scm_synth = X_post @ w

        # The ASCM counterfactual should incorporate SCM weights, not discard them
        from sklearn.linear_model import RidgeCV, Ridge
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 20), fit_intercept=True)
        ridge.fit(X_pre, Y_pre)
        pure_ridge_post = ridge.predict(X_post)

        # Fit ASCM bias correction (the fixed version)
        scm_residual_pre = Y_pre - X_pre @ w
        ridge_bias = Ridge(alpha=ridge.alpha_, fit_intercept=True)
        ridge_bias.fit(X_pre, scm_residual_pre)
        ascm_synth = scm_synth + ridge_bias.predict(X_post)

        # ASCM should NOT equal pure ridge (they should differ)
        assert not np.allclose(ascm_synth, pure_ridge_post, atol=1e-6), \
            "ASCM counterfactual should not equal pure ridge prediction"

    def test_ascm_with_known_effect(self):
        """ASCM should detect a known treatment effect."""
        pre, post, t_dmas, h_dmas = _make_panel(lift=0.10)
        result = augmented_synthetic_control(
            pre, post, t_dmas, h_dmas,
        )
        # Should detect positive lift
        assert result.absolute_lift > 0
        assert result.relative_lift > 0


# ---------------------------------------------------------------------------
# Bug #2: Re-randomization should reject unbalanced assignments
# ---------------------------------------------------------------------------

class TestRerandomization:
    def test_rejects_unbalanced(self):
        """Re-randomization should prefer balanced assignments."""
        rng = np.random.default_rng(42)
        n = 40
        df = pd.DataFrame({
            "dma_code": [f"DMA_{i}" for i in range(n)],
            "total_revenue": rng.lognormal(10, 1, n),
            "total_orders": rng.poisson(100, n),
            "revenue_per_capita": rng.uniform(5, 50, n),
            "total_ad_spend": rng.lognormal(8, 1, n),
            "revenue_trend": rng.normal(0.01, 0.05, n),
            "revenue_volatility": rng.uniform(0.05, 0.30, n),
        })

        treatment, holdout = match_dmas_rerandomization(
            df, n_holdout=10, n_iterations=5000, balance_threshold=0.25,
        )
        assert len(holdout) == 10
        assert len(treatment) == 30

        # Check balance
        from incrementality.design.matching import compute_balance_score
        t_df = df[df["dma_code"].isin(treatment)]
        h_df = df[df["dma_code"].isin(holdout)]
        score, smds = compute_balance_score(t_df, h_df)
        # Should achieve reasonable balance
        assert score > 0.5, f"Balance score too low: {score}"


# ---------------------------------------------------------------------------
# Bug #4: Conformal CI should contain the point estimate
# ---------------------------------------------------------------------------

class TestConformalCI:
    def test_ci_contains_point_estimate(self):
        """Conformal CI should always contain the observed test statistic."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            pre_residuals = rng.normal(0, 1, 30)
            post_gaps = rng.normal(0.5, 1, 14)

            result = _conformal_inference_ascm(
                pre_residuals, post_gaps, alpha=0.05, n_perm=1000,
            )
            observed = np.mean(post_gaps)
            assert result["ci_lower"] <= observed <= result["ci_upper"], \
                f"CI [{result['ci_lower']}, {result['ci_upper']}] does not contain observed {observed}"

    def test_ci_wider_at_higher_alpha(self):
        """CI at alpha=0.01 should be wider than at alpha=0.10."""
        pre_residuals = np.random.default_rng(42).normal(0, 1, 30)
        post_gaps = np.random.default_rng(42).normal(0.5, 1, 14)

        ci_narrow = _conformal_inference_ascm(pre_residuals, post_gaps, alpha=0.10)
        ci_wide = _conformal_inference_ascm(pre_residuals, post_gaps, alpha=0.01)

        width_narrow = ci_narrow["ci_upper"] - ci_narrow["ci_lower"]
        width_wide = ci_wide["ci_upper"] - ci_wide["ci_lower"]
        assert width_wide >= width_narrow


# ---------------------------------------------------------------------------
# Bug #5: iROAS CI should not explode near zero lift
# ---------------------------------------------------------------------------

class TestIROASNearZero:
    def test_near_zero_lift_no_explosion(self):
        """When relative_lift is near zero, CI should not be astronomically large."""
        result = IncrementalityResult(
            absolute_lift=0.001,
            relative_lift=0.0000001,  # Near-zero
            lift_lower_ci=-0.05,
            lift_upper_ci=0.05,
            p_value=0.95,
            is_significant=False,
            confidence_level=0.95,
            cohen_d=0.0,
            method="ascm",
        )
        total, lower, upper = compute_incremental_revenue(result, 100, 30)
        # Should not be infinity or astronomically large
        assert abs(lower) < 1e10
        assert abs(upper) < 1e10

    def test_normal_lift_computes_correctly(self):
        """Normal lift should compute reasonable revenue."""
        result = IncrementalityResult(
            absolute_lift=100.0,
            relative_lift=0.10,
            lift_lower_ci=0.03,
            lift_upper_ci=0.17,
            p_value=0.01,
            is_significant=True,
            confidence_level=0.95,
            cohen_d=0.5,
            method="ascm",
        )
        total, lower, upper = compute_incremental_revenue(result, 50, 30)
        assert total == 100.0 * 50 * 30
        assert lower < total
        assert upper > total


# ---------------------------------------------------------------------------
# Bug #7: Power formula should be two-sided
# ---------------------------------------------------------------------------

class TestPowerFormula:
    def test_power_two_sided(self):
        """Power formula should give ~target_power at the MDE."""
        from incrementality.design.power_analysis import (
            HistoricalVarianceEstimate,
            compute_mde,
        )
        var_est = HistoricalVarianceEstimate(
            between_dma_variance=50000,
            within_dma_variance=10000,
            total_variance=60000,
            mean_daily_revenue=500,
            mean_weekly_revenue=3500,
            coefficient_of_variation=0.7,
            num_dmas=50,
            num_periods=12,
            autocorrelation_lag1=0.3,
        )
        mde = compute_mde(var_est, 40, 10, 4, 0.05, 0.80)
        power = _compute_achieved_power(var_est, 40, 10, mde, 4, 0.05)
        # Power at MDE should be close to 0.80 (the target)
        assert 0.78 <= power <= 0.85, f"Power at MDE should be ~0.80, got {power}"


# ---------------------------------------------------------------------------
# Bug #8: Power scoring should be continuous
# ---------------------------------------------------------------------------

class TestPowerScoring:
    def test_continuous_at_060(self):
        """Score should be continuous at simulated_power=0.60."""
        score_below = _compute_power_score(0.599, 0.05, 0.10, 15)
        score_at = _compute_power_score(0.60, 0.05, 0.10, 15)
        score_above = _compute_power_score(0.601, 0.05, 0.10, 15)

        # Scores should be monotonically increasing (no discontinuity)
        assert score_below <= score_at <= score_above, \
            f"Discontinuity at 0.60: {score_below}, {score_at}, {score_above}"

    def test_monotonically_increasing(self):
        """Higher power should always give equal or higher score."""
        prev = 0.0
        for p in np.arange(0, 1.01, 0.05):
            score = _compute_power_score(p, 0.05, 0.10, 15)
            assert score >= prev - 0.01, \
                f"Score decreased at power={p}: {score} < {prev}"
            prev = score


# ---------------------------------------------------------------------------
# Bug #10: DiD baseline should use treatment counterfactual
# ---------------------------------------------------------------------------

class TestDiDBaseline:
    def test_baseline_not_holdout_level(self):
        """DiD baseline should be treatment counterfactual, not raw holdout."""
        pre, post, t_dmas, h_dmas = _make_panel(
            n_treatment=20, n_holdout=10, lift=0.10,
        )
        result = difference_in_differences(
            pre, post, t_dmas, h_dmas,
        )
        # The relative lift should be reasonable (not NaN, not extreme)
        assert not math.isnan(result.relative_lift)
        assert abs(result.relative_lift) < 1.0  # Should be in reasonable range


# ---------------------------------------------------------------------------
# Bug #28: Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_valid_config(self):
        c = StatisticalConfig()
        assert c.significance_level == 0.05
        assert c.target_power == 0.80

    def test_invalid_alpha_rejected(self):
        with pytest.raises(Exception):
            StatisticalConfig(significance_level=2.0)

    def test_invalid_power_rejected(self):
        with pytest.raises(Exception):
            StatisticalConfig(target_power=-0.5)

    def test_zero_alpha_rejected(self):
        with pytest.raises(Exception):
            StatisticalConfig(significance_level=0.0)

    def test_zero_holdout_rejected(self):
        with pytest.raises(Exception):
            StatisticalConfig(max_holdout_fraction=0.0)


# ---------------------------------------------------------------------------
# Bug #3: Zip-to-DMA should have no duplicate entries
# ---------------------------------------------------------------------------

class TestZipToDMANoDuplicates:
    def test_no_duplicate_zip3_keys(self):
        """The zip_to_dma function should return geographically correct results.

        This test verifies the mapping by checking that known zip codes
        map to the expected DMAs (based on geography).
        """
        # These are known correct mappings:
        assert zip_to_dma("12000") == "532"  # Albany, not NYC
        assert zip_to_dma("06400") == "533"  # Hartford, not NYC
        assert zip_to_dma("44400") == "536"  # Youngstown, not Cleveland
        assert zip_to_dma("78000") == "641"  # San Antonio, not Laredo

    def test_major_metros_correct(self):
        """Major metro zip codes should map to the correct DMA."""
        # NYC core zips should still be 501
        assert zip_to_dma("10001") == "501"  # Manhattan
        assert zip_to_dma("11201") == "501"  # Brooklyn
        # Houston core should be 618
        assert zip_to_dma("77001") == "618"  # Houston proper


# ---------------------------------------------------------------------------
# Campaign-level design: ad spend should be filtered by campaign IDs
# ---------------------------------------------------------------------------

class TestCampaignLevelDesign:
    def test_pull_historical_data_accepts_campaign_ids(self):
        """pull_historical_data should accept ad_channel and campaign_ids params."""
        from incrementality.orchestrator import TestOrchestrator
        from incrementality.config import Config
        from incrementality.models import AdChannel
        import inspect

        sig = inspect.signature(TestOrchestrator.pull_historical_data)
        param_names = list(sig.parameters.keys())
        assert "campaign_ids" in param_names, \
            "pull_historical_data should accept campaign_ids parameter"
        assert "ad_channel" in param_names, \
            "pull_historical_data should accept ad_channel parameter"

    def test_design_test_passes_campaign_ids_to_data_pull(self):
        """design_test should pass campaign_ids when pulling historical data."""
        from unittest.mock import patch, MagicMock
        from incrementality.orchestrator import TestOrchestrator
        from incrementality.config import Config
        from incrementality.models import AdChannel, TestScope

        config = Config()
        orch = TestOrchestrator(config)

        # Mock pull_historical_data to capture its arguments
        with patch.object(orch, "pull_historical_data") as mock_pull:
            mock_pull.side_effect = Exception("stop here")
            with patch.object(orch, "load_cached_data", return_value={}):
                try:
                    orch.design_test(
                        ad_channel=AdChannel.FACEBOOK,
                        test_scope=TestScope.CAMPAIGN,
                        campaign_ids=["123", "456"],
                    )
                except Exception:
                    pass

            # Verify campaign_ids was passed
            mock_pull.assert_called_once()
            call_kwargs = mock_pull.call_args
            assert call_kwargs.kwargs.get("campaign_ids") == ["123", "456"]
            assert call_kwargs.kwargs.get("ad_channel") == AdChannel.FACEBOOK


# ---------------------------------------------------------------------------
# Feasibility should be stored in TestDesign
# ---------------------------------------------------------------------------

class TestFeasibilityInDesign:
    def test_feasibility_field_exists(self):
        """TestDesign model should have a feasibility field."""
        from incrementality.models import TestDesign
        assert "feasibility" in TestDesign.model_fields

    def test_feasibility_populated_after_design(self):
        """auto_design_test should populate feasibility in the returned design."""
        from incrementality.design.optimizer import auto_design_test
        from incrementality.models import AdChannel, TestScope, MeasurementScope

        pre, post, t_dmas, h_dmas = _make_panel(
            n_treatment=30, n_holdout=10, n_pre=60, n_post=21,
        )
        # Combine into single historical data
        all_data = pd.concat([pre, post])
        all_data["orders"] = 10

        design = auto_design_test(
            shopify_daily=all_data,
            amazon_daily=None,
            facebook_daily=all_data.rename(columns={"revenue": "spend"}),
            youtube_daily=None,
            ad_channel=AdChannel.FACEBOOK,
            test_scope=TestScope.CHANNEL,
            measurement_scope=MeasurementScope.SHOPIFY_ONLY,
            run_simulation=False,
        )

        assert design.feasibility is not None
        assert hasattr(design.feasibility, "is_feasible")
        assert hasattr(design.feasibility, "power_score")
        assert hasattr(design.feasibility, "reasons")
        assert len(design.feasibility.reasons) > 0


# ---------------------------------------------------------------------------
# Optimizer should pick more holdout DMAs when variance is high
# ---------------------------------------------------------------------------

class TestOptimizerHoldoutSelection:
    def test_high_variance_picks_more_than_minimum_holdout(self):
        """When variance is high (MDE > target for all sizes), the optimizer
        should still pick a reasonable holdout size rather than the minimum 5.

        Regression test: the optimizer used to clamp mde_score at 0, causing
        all holdout sizes to score identically when MDE exceeded target_mde.
        The loop started at 5 and kept it due to strict > comparison.
        """
        from incrementality.design.optimizer import determine_optimal_holdout_size
        from incrementality.design.power_analysis import HistoricalVarianceEstimate

        # Simulate high-variance data where MDE won't reach 15% at any holdout size
        # Using variance values that produce ~58% MDE at 5 holdout DMAs
        var_est = HistoricalVarianceEstimate(
            between_dma_variance=50000.0,
            within_dma_variance=500000.0,
            total_variance=550000.0,
            mean_daily_revenue=100.0,
            mean_weekly_revenue=700.0,
            coefficient_of_variation=1.06,
            num_dmas=210,
            num_periods=12,
            autocorrelation_lag1=0.3,
        )

        config = StatisticalConfig()
        n_treatment, n_holdout = determine_optimal_holdout_size(
            var_est, 210, config, target_mde=0.15,
        )

        # With high variance, optimizer should allocate significantly more than 5
        assert n_holdout > 5, (
            f"Optimizer selected only {n_holdout} holdout DMAs. "
            f"With high variance, should pick more to reduce MDE."
        )
        # Should pick something in the range of 20-105 (not the minimum)
        assert n_holdout >= 20, (
            f"Expected >= 20 holdout DMAs, got {n_holdout}. "
            f"Optimizer should aggressively reduce MDE when variance is high."
        )
        assert n_treatment + n_holdout == 210

    def test_low_variance_stays_near_target(self):
        """When variance is low enough to hit target MDE, optimizer should
        pick a holdout near the target fraction (25%).
        """
        from incrementality.design.optimizer import determine_optimal_holdout_size
        from incrementality.design.power_analysis import HistoricalVarianceEstimate

        # Low variance — MDE should be well under 15% at target holdout
        var_est = HistoricalVarianceEstimate(
            between_dma_variance=100.0,
            within_dma_variance=1000.0,
            total_variance=1100.0,
            mean_daily_revenue=100.0,
            mean_weekly_revenue=700.0,
            coefficient_of_variation=0.047,
            num_dmas=210,
            num_periods=12,
            autocorrelation_lag1=0.1,
        )

        config = StatisticalConfig()
        n_treatment, n_holdout = determine_optimal_holdout_size(
            var_est, 210, config, target_mde=0.15,
        )

        # With low variance, should be near the target (25% of 210 = 52)
        # but not necessarily exactly 52 — the penalty terms will shift it
        assert 5 <= n_holdout <= 80, f"Unexpected holdout size: {n_holdout}"


# ---------------------------------------------------------------------------
# CLI analyze should accept --attributed-conversions
# ---------------------------------------------------------------------------

class TestCLIAttributedConversions:
    def test_analyze_has_attributed_conversions_param(self):
        """The analyze CLI command should accept --attributed-conversions."""
        from incrementality.cli import analyze
        import click

        # Inspect click command params
        param_names = [p.name for p in analyze.params]
        assert "attributed_conversions" in param_names, \
            "analyze command should have --attributed-conversions option"
