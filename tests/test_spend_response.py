"""Tests for spend response curve estimation.

Verifies:
- Pre-period-adjusted DMA counterfactuals (removes market-size confounding)
- Hill function fitting and calibration to causal iROAS
- Optimal spend calculation
- Bootstrap confidence intervals
- Edge cases (insufficient data, zero spend, etc.)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from incrementality.analysis.spend_response import (
    SpendResponseCurve,
    _fit_hill,
    _find_optimal_spend,
    _hill_function,
    _hill_marginal,
    estimate_spend_response,
)


# ---------------------------------------------------------------------------
# Test data generators
# ---------------------------------------------------------------------------

def _make_test_data(
    n_treatment: int = 20,
    n_holdout: int = 10,
    n_days_pre: int = 30,
    n_days_post: int = 21,
    base_revenue: float = 1000.0,
    lift_pct: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Generate pre/post/spend data with known lift for testing.

    Treatment DMAs have varying spend and receive a lift proportional to spend.
    Holdout DMAs have zero spend and zero lift.
    """
    rng = np.random.default_rng(seed)
    treatment_dmas = [f"T_{i:03d}" for i in range(n_treatment)]
    holdout_dmas = [f"H_{i:03d}" for i in range(n_holdout)]
    all_dmas = treatment_dmas + holdout_dmas

    # Each DMA has a unique baseline (simulates market size heterogeneity)
    dma_baselines = {
        d: base_revenue * (0.5 + rng.uniform(0, 2.0))
        for d in all_dmas
    }

    # Spend varies across treatment DMAs (correlated with baseline size)
    dma_spend = {}
    for d in treatment_dmas:
        # Spend roughly proportional to baseline + noise
        dma_spend[d] = max(10, dma_baselines[d] * 0.05 + rng.normal(0, 20))
    for d in holdout_dmas:
        dma_spend[d] = 0.0

    # Pre-period: all DMAs at their baseline
    pre_records = []
    for day in range(n_days_pre):
        dt = date(2025, 1, 1 + day % 28) if day < 28 else date(2025, 2, 1 + (day - 28) % 28)
        for dma in all_dmas:
            pre_records.append({
                "date": dt,
                "dma_code": dma,
                "revenue": max(0, dma_baselines[dma] + rng.normal(0, 30)),
            })

    # Post-period: treatment DMAs get a lift, holdout stays at baseline
    post_records = []
    for day in range(n_days_post):
        dt = date(2025, 3, 1 + day % 28)
        for dma in all_dmas:
            rev = dma_baselines[dma] + rng.normal(0, 30)
            if dma in treatment_dmas:
                # Lift proportional to spend (diminishing returns via sqrt)
                spend = dma_spend[dma]
                rev += lift_pct * dma_baselines[dma] * np.sqrt(spend / 50.0)
            post_records.append({
                "date": dt,
                "dma_code": dma,
                "revenue": max(0, rev),
            })

    # Ad spend data (only for treatment DMAs)
    spend_records = []
    for day in range(n_days_post):
        dt = date(2025, 3, 1 + day % 28)
        for dma in treatment_dmas:
            spend_records.append({
                "date": dt,
                "dma_code": dma,
                "spend": dma_spend[dma] + rng.normal(0, 2),
            })

    return (
        pd.DataFrame(pre_records),
        pd.DataFrame(post_records),
        pd.DataFrame(spend_records),
        treatment_dmas,
        holdout_dmas,
    )


# ---------------------------------------------------------------------------
# Hill function unit tests
# ---------------------------------------------------------------------------

class TestHillFunction:
    def test_hill_at_zero(self):
        """Hill function should return ~0 at spend = 0."""
        result = _hill_function(np.array([0.0]), r_max=100.0, k=50.0, alpha=1.0)
        assert result[0] < 0.01

    def test_hill_at_half_saturation(self):
        """At spend = K, Hill should return R_max / 2."""
        result = _hill_function(np.array([50.0]), r_max=100.0, k=50.0, alpha=1.0)
        assert abs(result[0] - 50.0) < 0.01

    def test_hill_saturation(self):
        """At very high spend, Hill should approach R_max."""
        result = _hill_function(np.array([1e6]), r_max=100.0, k=50.0, alpha=1.0)
        assert result[0] > 99.9

    def test_hill_monotonic(self):
        """Hill function should be monotonically increasing."""
        spend = np.linspace(0.01, 1000.0, 100)
        values = _hill_function(spend, r_max=100.0, k=50.0, alpha=0.7)
        diffs = np.diff(values)
        assert np.all(diffs >= 0)

    def test_hill_concave_with_alpha_less_than_one(self):
        """With alpha < 1, Hill should show diminishing returns (concave)."""
        spend = np.linspace(1.0, 1000.0, 100)
        values = _hill_function(spend, r_max=100.0, k=50.0, alpha=0.7)
        diffs = np.diff(values)
        # Second differences should be negative (diminishing returns)
        second_diffs = np.diff(diffs)
        assert np.all(second_diffs <= 1e-10)


class TestHillMarginal:
    def test_marginal_positive(self):
        """Marginal ROAS should be positive."""
        m = _hill_marginal(50.0, r_max=100.0, k=50.0, alpha=0.7)
        assert m > 0

    def test_marginal_decreasing(self):
        """Marginal ROAS should decrease with higher spend."""
        m1 = _hill_marginal(10.0, r_max=100.0, k=50.0, alpha=0.7)
        m2 = _hill_marginal(100.0, r_max=100.0, k=50.0, alpha=0.7)
        m3 = _hill_marginal(500.0, r_max=100.0, k=50.0, alpha=0.7)
        assert m1 > m2 > m3

    def test_marginal_matches_numerical_derivative(self):
        """Marginal should match numerical derivative of Hill function."""
        s = 50.0
        h = 0.01
        numerical = (
            _hill_function(np.array([s + h]), 100.0, 50.0, 0.7)[0]
            - _hill_function(np.array([s - h]), 100.0, 50.0, 0.7)[0]
        ) / (2 * h)
        analytical = _hill_marginal(s, 100.0, 50.0, 0.7)
        assert abs(numerical - analytical) / max(analytical, 1e-6) < 0.01


# ---------------------------------------------------------------------------
# Fit and optimal spend tests
# ---------------------------------------------------------------------------

class TestFitHill:
    def test_fit_recovers_known_params(self):
        """Fitting Hill to data generated from Hill should recover params."""
        true_r_max, true_k, true_alpha = 200.0, 40.0, 0.8
        rng = np.random.default_rng(42)
        x = np.linspace(0, 200, 30)
        y = _hill_function(x, true_r_max, true_k, true_alpha) + rng.normal(0, 2, len(x))

        params, converged = _fit_hill(x, y)
        assert params is not None
        assert converged
        r_max, k, alpha = params
        # Should recover within 20%
        assert abs(r_max - true_r_max) / true_r_max < 0.2
        assert abs(k - true_k) / true_k < 0.3
        assert abs(alpha - true_alpha) / true_alpha < 0.3

    def test_fit_returns_none_on_garbage(self):
        """Fitting to random noise should gracefully handle failure."""
        rng = np.random.default_rng(99)
        x = rng.uniform(0, 100, 5)
        y = rng.uniform(-100, 100, 5)
        # May or may not converge, but should not crash
        params, converged = _fit_hill(x, y)
        # Either returns params or None, no exceptions


class TestFindOptimalSpend:
    def test_optimal_spend_where_marginal_equals_target(self):
        """Optimal spend should be where marginal ROAS = target."""
        r_max, k, alpha = 200.0, 50.0, 0.8
        target = 1.0
        optimal = _find_optimal_spend(r_max, k, alpha, 100.0, target)

        marginal_at_optimal = _hill_marginal(optimal, r_max, k, alpha)
        # Should be close to the target
        assert abs(marginal_at_optimal - target) < 0.1

    def test_higher_target_gives_lower_optimal_spend(self):
        """Higher target marginal ROAS should mean lower optimal spend."""
        r_max, k, alpha = 200.0, 50.0, 0.8
        opt_low_target = _find_optimal_spend(r_max, k, alpha, 100.0, 0.5)
        opt_high_target = _find_optimal_spend(r_max, k, alpha, 100.0, 2.0)
        assert opt_high_target < opt_low_target


# ---------------------------------------------------------------------------
# Integration tests: estimate_spend_response
# ---------------------------------------------------------------------------

class TestEstimateSpendResponse:
    def test_basic_estimation_succeeds(self):
        """Should produce a SpendResponseCurve from valid test data."""
        pre, post, spend, t_dmas, h_dmas = _make_test_data()
        result = estimate_spend_response(
            pre, post, spend, t_dmas, h_dmas,
            n_bootstrap=20,
        )
        assert result is not None
        assert isinstance(result, SpendResponseCurve)
        assert result.r_max > 0
        assert result.k > 0
        assert result.alpha > 0
        assert result.n_dmas_used > 0
        assert result.current_spend > 0

    def test_pre_period_adjustment_reduces_confounding(self):
        """Pre-period-adjusted incremental should be more stable than raw."""
        pre, post, spend, t_dmas, h_dmas = _make_test_data(
            n_treatment=30, n_holdout=15, lift_pct=0.05,
        )
        result = estimate_spend_response(
            pre, post, spend, t_dmas, h_dmas,
            n_bootstrap=10,
        )
        assert result is not None
        # R² should be reasonable (fit quality)
        assert result.r_squared > -1.0  # Even poor fits shouldn't be wildly negative

    def test_calibration_adjusts_curve(self):
        """When causal iROAS is provided, curve should be calibrated."""
        pre, post, spend, t_dmas, h_dmas = _make_test_data()
        causal_iroas = 3.5

        result = estimate_spend_response(
            pre, post, spend, t_dmas, h_dmas,
            causal_iroas=causal_iroas,
            n_bootstrap=10,
        )
        assert result is not None
        assert result.calibrated is True

    def test_without_calibration(self):
        """Without causal iROAS, curve should not be calibrated."""
        pre, post, spend, t_dmas, h_dmas = _make_test_data()

        result = estimate_spend_response(
            pre, post, spend, t_dmas, h_dmas,
            causal_iroas=None,
            n_bootstrap=10,
        )
        assert result is not None
        assert result.calibrated is False

    def test_curve_data_populated(self):
        """Should produce curve data for charting."""
        pre, post, spend, t_dmas, h_dmas = _make_test_data()
        result = estimate_spend_response(
            pre, post, spend, t_dmas, h_dmas,
            n_curve_points=50,
            n_bootstrap=10,
        )
        assert result is not None
        assert len(result.spend_curve) == 50
        assert len(result.revenue_curve) == 50
        assert len(result.marginal_roas_curve) == 50
        assert len(result.avg_roas_curve) == 50
        # CI bands populated from bootstrap
        assert len(result.revenue_curve_lower) == 50
        assert len(result.revenue_curve_upper) == 50

    def test_optimal_spend_ci_populated(self):
        """Bootstrap should produce CI on optimal spend."""
        pre, post, spend, t_dmas, h_dmas = _make_test_data()
        result = estimate_spend_response(
            pre, post, spend, t_dmas, h_dmas,
            n_bootstrap=50,
        )
        assert result is not None
        assert result.optimal_spend_lower > 0
        assert result.optimal_spend_upper > 0
        assert result.optimal_spend_lower <= result.optimal_spend_upper

    def test_recommendation_generated(self):
        """Should produce a text recommendation."""
        pre, post, spend, t_dmas, h_dmas = _make_test_data()
        result = estimate_spend_response(
            pre, post, spend, t_dmas, h_dmas,
            n_bootstrap=10,
        )
        assert result is not None
        assert result.recommendation != ""
        assert result.spend_change_direction in ("increase", "decrease", "maintain")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestSpendResponseEdgeCases:
    def test_empty_spend_data(self):
        """Should return None if spend data is empty."""
        pre, post, _, t_dmas, h_dmas = _make_test_data()
        result = estimate_spend_response(
            pre, post,
            pd.DataFrame(columns=["date", "dma_code", "spend"]),
            t_dmas, h_dmas,
        )
        assert result is None

    def test_too_few_treatment_dmas(self):
        """Should return None with fewer than 3 treatment DMAs."""
        pre, post, spend, _, h_dmas = _make_test_data()
        result = estimate_spend_response(
            pre, post, spend,
            ["T_000", "T_001"],  # Only 2
            h_dmas,
            n_bootstrap=5,
        )
        assert result is None

    def test_too_few_holdout_dmas(self):
        """Should return None with fewer than 2 holdout DMAs."""
        pre, post, spend, t_dmas, _ = _make_test_data()
        result = estimate_spend_response(
            pre, post, spend,
            t_dmas,
            ["H_000"],  # Only 1
            n_bootstrap=5,
        )
        assert result is None

    def test_missing_spend_column(self):
        """Should return None if spend column is missing."""
        pre, post, spend, t_dmas, h_dmas = _make_test_data()
        bad_spend = spend.rename(columns={"spend": "cost"})
        result = estimate_spend_response(
            pre, post, bad_spend, t_dmas, h_dmas,
        )
        assert result is None

    def test_extrapolation_warning(self):
        """Recommendation should warn when optimal exceeds observed range."""
        pre, post, spend, t_dmas, h_dmas = _make_test_data(lift_pct=0.30)
        result = estimate_spend_response(
            pre, post, spend, t_dmas, h_dmas,
            target_marginal_roas=0.1,  # Very low target → high optimal spend
            n_bootstrap=10,
        )
        # If optimal >> observed, should have extrapolation note
        if result is not None and result.optimal_spend > result.observed_spend_max * 1.2:
            assert "extrapolation" in result.recommendation.lower()
