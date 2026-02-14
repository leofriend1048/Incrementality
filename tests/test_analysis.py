"""Tests for the analysis pipeline: winsorization, anomaly detection, IF/CPIA."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from incrementality.analysis.anomaly import AnomalyReport, detect_anomalies
from incrementality.analysis.iroas import compute_iroas
from incrementality.analysis.winsorize import (
    compare_winsorized_results,
    winsorize_panel,
    winsorize_series,
)
from incrementality.models import IncrementalROAS, IncrementalityResult, MeasurementScope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(
    n_dmas: int = 10,
    n_days: int = 30,
    treatment_frac: float = 0.7,
    base_revenue: float = 1000.0,
    lift: float = 0.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Generate synthetic panel data for testing.

    Returns: (pre_data, post_data, treatment_dmas, holdout_dmas)
    """
    rng = np.random.default_rng(seed)
    n_treatment = max(1, int(n_dmas * treatment_frac))
    n_holdout = n_dmas - n_treatment
    treatment_dmas = [f"DMA_{i:03d}" for i in range(n_treatment)]
    holdout_dmas = [f"DMA_{i:03d}" for i in range(n_treatment, n_dmas)]

    all_dmas = treatment_dmas + holdout_dmas
    pre_days = n_days
    post_days = max(7, n_days // 2)

    records = []
    for dma in all_dmas:
        dma_base = base_revenue * (1 + rng.normal(0, 0.1))
        for d in range(pre_days):
            records.append({
                "date": date(2025, 6, 1 + d % 28) if d < 28 else date(2025, 7, 1 + (d - 28) % 28),
                "dma_code": dma,
                "revenue": max(0, dma_base + rng.normal(0, base_revenue * 0.05)),
                "orders": max(0, int(10 + rng.normal(0, 2))),
            })

    pre_data = pd.DataFrame(records)

    records = []
    for dma in all_dmas:
        dma_base = base_revenue * (1 + rng.normal(0, 0.1))
        is_treatment = dma in treatment_dmas
        for d in range(post_days):
            rev = dma_base + rng.normal(0, base_revenue * 0.05)
            if is_treatment and lift > 0:
                rev *= (1 + lift)
            orders = max(0, int(10 + rng.normal(0, 2)))
            if is_treatment and lift > 0:
                orders = int(orders * (1 + lift * 0.8))
            records.append({
                "date": date(2025, 7, 1 + d % 28),
                "dma_code": dma,
                "revenue": max(0, rev),
                "orders": orders,
            })

    post_data = pd.DataFrame(records)
    return pre_data, post_data, treatment_dmas, holdout_dmas


# ---------------------------------------------------------------------------
# Winsorization Tests
# ---------------------------------------------------------------------------

class TestWinsorizeSeries:
    def test_basic_winsorization(self):
        # Use enough data points so the interior values aren't affected
        rng = np.random.default_rng(42)
        values = np.concatenate([rng.normal(50, 5, size=98), [1.0, 500.0]])
        result = winsorize_series(values, lower_pct=0.01, upper_pct=0.99)
        # The extreme values should be clipped
        assert result[-1] < 500  # Upper extreme clipped
        # Interior values should be unchanged
        assert result[10] == values[10]
        assert result[50] == values[50]

    def test_empty_array(self):
        result = winsorize_series(np.array([]))
        assert len(result) == 0

    def test_single_value(self):
        result = winsorize_series(np.array([42.0]))
        assert result[0] == 42.0

    def test_preserves_length(self):
        values = np.random.default_rng(42).normal(100, 20, size=1000)
        result = winsorize_series(values, lower_pct=0.01, upper_pct=0.99)
        assert len(result) == len(values)

    def test_clips_both_tails(self):
        values = np.array([-100, 1, 2, 3, 4, 5, 200])
        result = winsorize_series(values, lower_pct=0.10, upper_pct=0.90)
        assert result[0] > -100  # Lower tail clipped
        assert result[-1] < 200  # Upper tail clipped


class TestWinsorizePanel:
    def test_basic_panel_winsorization(self):
        pre, post, t_dmas, h_dmas = _make_panel(n_dmas=20, n_days=30)
        # Add an extreme outlier DMA
        outlier_mask = post["dma_code"] == t_dmas[0]
        post.loc[outlier_mask, "revenue"] *= 10  # 10x outlier

        result = winsorize_panel(post, revenue_col="revenue")
        # Outlier DMA should be scaled down
        outlier_rev = result.loc[outlier_mask, "revenue"].mean()
        original_rev = post.loc[outlier_mask, "revenue"].mean()
        assert outlier_rev < original_rev

    def test_preserves_shape(self):
        pre, post, _, _ = _make_panel()
        result = winsorize_panel(post)
        assert result.shape == post.shape
        assert set(result.columns) == set(post.columns)

    def test_too_few_dmas_skips(self):
        pre, post, _, _ = _make_panel(n_dmas=3)
        result = winsorize_panel(post)
        # Should return unchanged with < 5 DMAs
        pd.testing.assert_frame_equal(result, post)

    def test_missing_column_returns_unchanged(self):
        pre, post, _, _ = _make_panel()
        result = winsorize_panel(post, revenue_col="nonexistent")
        pd.testing.assert_frame_equal(result, post)


class TestCompareWinsorizedResults:
    def test_robust_results(self):
        result = compare_winsorized_results(
            raw_lift=0.10, winsorized_lift=0.098,
            raw_p=0.01, winsorized_p=0.012,
        )
        assert not result["is_outlier_driven"]
        assert not result["significance_flips"]
        assert result["divergence"] < 0.30

    def test_divergent_results(self):
        result = compare_winsorized_results(
            raw_lift=0.15, winsorized_lift=0.04,
            raw_p=0.01, winsorized_p=0.08,
        )
        assert result["is_outlier_driven"]
        assert result["significance_flips"]

    def test_moderate_divergence(self):
        result = compare_winsorized_results(
            raw_lift=0.12, winsorized_lift=0.06,
            raw_p=0.02, winsorized_p=0.03,
        )
        assert result["is_outlier_driven"]
        assert not result["significance_flips"]


# ---------------------------------------------------------------------------
# Anomaly Detection Tests
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    def test_clean_data_no_blockers(self):
        pre, post, t_dmas, h_dmas = _make_panel(n_dmas=15, n_days=30)
        report = detect_anomalies(pre, post, t_dmas, h_dmas)
        assert not report.has_blockers
        assert isinstance(report.summary, dict)

    def test_empty_data_blocker(self):
        empty = pd.DataFrame(columns=["date", "dma_code", "revenue"])
        report = detect_anomalies(
            empty, empty,
            ["DMA_001"], ["DMA_002"],
        )
        assert report.has_blockers
        assert report.n_blockers > 0

    def test_outlier_detection(self):
        pre, post, t_dmas, h_dmas = _make_panel(n_dmas=15, n_days=30)
        # Inject extreme outliers
        for dma in t_dmas[:2]:
            mask = post["dma_code"] == dma
            idx = post.loc[mask].index[0]
            post.loc[idx, "revenue"] = 100_000  # Extreme spike

        report = detect_anomalies(
            pre, post, t_dmas, h_dmas,
            zscore_threshold=3.0,
        )
        # Should detect outliers
        outlier_alerts = [a for a in report.alerts if a.category == "outlier"]
        assert len(outlier_alerts) > 0

    def test_coverage_warning(self):
        pre, post, t_dmas, h_dmas = _make_panel(n_dmas=10, n_days=10)
        # Remove some holdout DMAs from post data
        post = post[~post["dma_code"].isin(h_dmas[:2])]

        report = detect_anomalies(pre, post, t_dmas, h_dmas)
        coverage_alerts = [a for a in report.alerts if a.category == "coverage"]
        assert len(coverage_alerts) > 0

    def test_anomaly_report_properties(self):
        report = AnomalyReport()
        assert not report.has_blockers
        assert report.warnings == []
        assert report.blockers == []
        assert report.n_warnings == 0
        assert report.n_blockers == 0


# ---------------------------------------------------------------------------
# IF / CPIA Computation Tests
# ---------------------------------------------------------------------------

class TestIFCPIA:
    def test_compute_if_cpia_with_orders(self):
        """Test that IF and CPIA are computed when orders data is available."""
        pre, post, t_dmas, h_dmas = _make_panel(
            n_dmas=10, n_days=30, lift=0.15,
        )

        # Create ad spend data
        spend_records = []
        for dma in t_dmas:
            for d in range(15):
                spend_records.append({
                    "date": date(2025, 7, 1 + d),
                    "dma_code": dma,
                    "spend": 100.0,
                })
        ad_spend = pd.DataFrame(spend_records)

        # A primary result with positive lift
        primary = IncrementalityResult(
            absolute_lift=150.0,
            relative_lift=0.15,
            lift_lower_ci=0.05,
            lift_upper_ci=0.25,
            p_value=0.01,
            is_significant=True,
            confidence_level=0.95,
            cohen_d=0.5,
            method="ensemble",
        )

        result = compute_iroas(
            pre, post, ad_spend,
            t_dmas, h_dmas,
            MeasurementScope.SHOPIFY_ONLY,
            test_duration_days=15,
            primary_result=primary,
            attributed_conversions=100.0,
        )

        # Should have IF and CPIA populated
        assert result.incremental_conversions > 0
        assert result.cpia > 0
        assert result.attributed_conversions == 100.0
        assert result.incrementality_factor > 0

    def test_no_orders_no_if(self):
        """Without orders data, IF/CPIA should stay at 0."""
        pre, post, t_dmas, h_dmas = _make_panel(n_dmas=10, n_days=30)
        # Remove orders column
        pre = pre.drop(columns=["orders"])
        post = post.drop(columns=["orders"])

        spend_records = []
        for dma in t_dmas:
            for d in range(15):
                spend_records.append({
                    "date": date(2025, 7, 1 + d),
                    "dma_code": dma,
                    "spend": 100.0,
                })
        ad_spend = pd.DataFrame(spend_records)

        primary = IncrementalityResult(
            absolute_lift=100.0,
            relative_lift=0.10,
            lift_lower_ci=0.02,
            lift_upper_ci=0.18,
            p_value=0.02,
            is_significant=True,
            confidence_level=0.95,
            cohen_d=0.4,
            method="did",
        )

        result = compute_iroas(
            pre, post, ad_spend,
            t_dmas, h_dmas,
            MeasurementScope.SHOPIFY_ONLY,
            test_duration_days=15,
            primary_result=primary,
        )

        # IF/CPIA should remain at 0
        assert result.incrementality_factor == 0.0
        assert result.cpia == 0.0

    def test_no_attributed_conversions_still_has_cpia(self):
        """CPIA should be computed even without attributed conversions."""
        pre, post, t_dmas, h_dmas = _make_panel(
            n_dmas=10, n_days=30, lift=0.15,
        )

        spend_records = []
        for dma in t_dmas:
            for d in range(15):
                spend_records.append({
                    "date": date(2025, 7, 1 + d),
                    "dma_code": dma,
                    "spend": 100.0,
                })
        ad_spend = pd.DataFrame(spend_records)

        primary = IncrementalityResult(
            absolute_lift=150.0,
            relative_lift=0.15,
            lift_lower_ci=0.05,
            lift_upper_ci=0.25,
            p_value=0.01,
            is_significant=True,
            confidence_level=0.95,
            cohen_d=0.5,
            method="did",
        )

        result = compute_iroas(
            pre, post, ad_spend,
            t_dmas, h_dmas,
            MeasurementScope.SHOPIFY_ONLY,
            test_duration_days=15,
            primary_result=primary,
            attributed_conversions=0.0,  # No attributed conversions
        )

        # CPIA should be computed from experiment data
        assert result.cpia > 0
        assert result.incremental_conversions > 0
        # But IF should be 0 (no attributed to compare against)
        assert result.incrementality_factor == 0.0


# ---------------------------------------------------------------------------
# Sample Report Generation Test
# ---------------------------------------------------------------------------

class TestSampleReport:
    def test_sample_report_generates(self):
        """Test that the sample report generator still works with new fields."""
        from scripts.generate_sample_report import build_sample_report

        report = build_sample_report()
        assert report.test_id == "test_mtb_fb_q1_2026"
        assert report.winsorized_comparison is not None
        assert len(report.winsorized_comparison) > 0
        assert isinstance(report.anomaly_warnings, list)
        assert isinstance(report.anomaly_blockers, list)
        assert report.iroas.incrementality_factor == 1.20
        assert report.iroas.cpia == 125.38

    def test_html_report_renders(self):
        """Test that the HTML report renders without errors."""
        from scripts.generate_sample_report import build_sample_report
        from incrementality.report_pdf import _render_html

        report = build_sample_report()
        html = _render_html(report)
        assert "LIFT" in html
        assert "Winsorized" in html
        assert "Anomaly" in html or "Data Quality" in html
        assert "Incrementality Factor" in html
        assert "CPIA" in html or "Cost Per Incr" in html
