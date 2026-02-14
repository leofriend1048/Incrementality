"""Generate a sample LIFT report with realistic dummy data."""

from datetime import date, datetime

from incrementality.models import (
    AdChannel,
    IncrementalityResult,
    IncrementalROAS,
    MeasurementScope,
    PlaceboTestResult,
    TestReport,
    TestScope,
    ValidationReport,
)
from incrementality.report_pdf import save_report_html


def build_sample_report() -> TestReport:
    """Build a comprehensive sample report mimicking a real Facebook test."""

    # Primary ensemble result
    primary = IncrementalityResult(
        absolute_lift=847.32,
        relative_lift=0.134,
        lift_lower_ci=0.068,
        lift_upper_ci=0.201,
        p_value=0.0012,
        is_significant=True,
        confidence_level=0.95,
        cohen_d=0.412,
        method="ensemble_weighted",
        l2_imbalance=0.023,
        pre_period_r_squared=0.961,
        lift_likelihood=0.994,
    )

    # Individual estimator results
    ascm_result = IncrementalityResult(
        absolute_lift=891.50,
        relative_lift=0.141,
        lift_lower_ci=0.072,
        lift_upper_ci=0.210,
        p_value=0.0008,
        is_significant=True,
        confidence_level=0.95,
        cohen_d=0.438,
        method="ascm",
        l2_imbalance=0.019,
        pre_period_r_squared=0.968,
        lift_likelihood=0.997,
    )

    bsts_result = IncrementalityResult(
        absolute_lift=812.10,
        relative_lift=0.128,
        lift_lower_ci=0.054,
        lift_upper_ci=0.203,
        p_value=0.0031,
        is_significant=True,
        confidence_level=0.95,
        cohen_d=0.389,
        method="bsts",
        l2_imbalance=0.031,
        pre_period_r_squared=0.952,
        lift_likelihood=0.989,
    )

    did_result = IncrementalityResult(
        absolute_lift=838.67,
        relative_lift=0.132,
        lift_lower_ci=0.061,
        lift_upper_ci=0.204,
        p_value=0.0019,
        is_significant=True,
        confidence_level=0.95,
        cohen_d=0.401,
        method="did",
        l2_imbalance=0.0,
        pre_period_r_squared=0.0,
        lift_likelihood=0.991,
    )

    # iROAS
    iroas = IncrementalROAS(
        incremental_revenue=287_431.58,
        total_ad_spend=112_840.00,
        iroas=2.55,
        iroas_lower_ci=1.31,
        iroas_upper_ci=3.78,
        shopify_incremental_revenue=241_842.30,
        amazon_incremental_revenue=45_589.28,
        shopify_iroas=2.14,
        amazon_iroas=0.40,
        # IF and CPIA: Facebook reports 750 orders, but experiment shows 900 incremental
        attributed_conversions=750,
        incremental_conversions=900,
        incrementality_factor=1.20,  # 900 / 750
        cpia=125.38,  # $112,840 / 900
    )

    # Shopify-specific incrementality
    shopify_inc = IncrementalityResult(
        absolute_lift=711.89,
        relative_lift=0.142,
        lift_lower_ci=0.074,
        lift_upper_ci=0.211,
        p_value=0.0009,
        is_significant=True,
        confidence_level=0.95,
        cohen_d=0.445,
        method="ensemble_weighted",
        lift_likelihood=0.996,
    )

    # Amazon halo
    amazon_inc = IncrementalityResult(
        absolute_lift=135.43,
        relative_lift=0.067,
        lift_lower_ci=0.008,
        lift_upper_ci=0.126,
        p_value=0.0284,
        is_significant=True,
        confidence_level=0.95,
        cohen_d=0.198,
        method="did",
        lift_likelihood=0.971,
    )

    # Validation
    validation = ValidationReport(
        l2_imbalance=0.023,
        pre_period_r_squared=0.961,
        pre_period_mape=0.034,
        num_placebo_tests=20,
        placebo_pass_rate=0.95,
        false_positive_rate=0.05,
        placebo_results=[
            PlaceboTestResult(
                placebo_type="in_time",
                placebo_date=date(2025, 11, 15),
                estimated_effect=0.008,
                p_value=0.72,
                is_false_positive=False,
            ),
            PlaceboTestResult(
                placebo_type="in_space",
                target_dma="501",
                estimated_effect=-0.012,
                p_value=0.58,
                is_false_positive=False,
            ),
        ],
        aa_test_p_value=0.681,
        aa_test_passed=True,
        estimator_agreement=0.89,
        is_trustworthy=True,
        trust_score=87.0,
        warnings=[
            "3 DMAs in the holdout group had partial ad exposure during week 2 "
            "(spend < $50). Effect on results is negligible.",
        ],
        blockers=[],
    )

    # Full report
    report = TestReport(
        test_id="test_mtb_fb_q1_2026",
        test_name="Michael Todd Beauty — Facebook Incrementality Q1",
        ad_channel=AdChannel.FACEBOOK,
        test_scope=TestScope.CHANNEL,
        measurement_scope=MeasurementScope.SHOPIFY_AND_AMAZON,
        campaign_ids=[],
        duration_weeks=6,
        num_treatment_dmas=48,
        num_holdout_dmas=16,
        test_start=date(2026, 1, 6),
        test_end=date(2026, 2, 16),
        incrementality=primary,
        iroas=iroas,
        treatment_total_revenue=2_432_891.47,
        holdout_total_revenue=714_280.33,
        treatment_avg_daily_revenue=1206.40,
        holdout_avg_daily_revenue=1063.51,
        organic_baseline_revenue=2_142_872.00,
        shopify_incrementality=shopify_inc,
        amazon_incrementality=amazon_inc,
        validation=validation,
        estimator_results={
            "ascm": ascm_result,
            "bsts": bsts_result,
            "did": did_result,
        },
        estimator_weights={
            "ascm": 0.45,
            "bsts": 0.35,
            "did": 0.20,
        },
        # Winsorized robustness comparison
        winsorized_comparison={
            "raw_lift": 0.134,
            "winsorized_lift": 0.128,
            "raw_p": 0.0012,
            "winsorized_p": 0.0018,
            "divergence": 0.046,
            "is_outlier_driven": False,
            "significance_flips": False,
            "recommendation": (
                "Results are robust to winsorization (divergence: 5%). "
                "Outliers are not driving the result."
            ),
        },
        # Anomaly detection results
        anomaly_warnings=[
            "Detected 7 outlier observations (|z| > 3.5) across 3 DMAs. "
            "Winsorized analysis recommended to assess robustness.",
            "3 DMAs in the holdout group had partial ad exposure during week 2 "
            "(spend < $50). Effect on results is negligible.",
        ],
        anomaly_blockers=[],
        recommendations=[
            "The test reached statistical significance (p=0.001). There is strong "
            "evidence that Facebook ads drive +13.4% incremental lift across "
            "Shopify and Amazon combined. This result is corroborated by all three "
            "estimators (ASCM, BSTS, DiD) with high agreement (89%).",

            "iROAS of 2.55x is profitable — every $1 spent on Facebook generates "
            "$2.55 in incremental revenue. Current spend level of ~$112K over 6 weeks "
            "appears efficient. Consider testing a 15-20% budget increase to find "
            "the diminishing returns curve.",

            "Significant Amazon halo effect detected: +6.7% incremental lift on "
            "Amazon (p=0.028). Facebook ads are driving cross-platform conversions "
            "that platform-reported ROAS cannot capture. Amazon accounts for 16% of "
            "total incremental returns — this is revenue you'd miss with "
            "platform-only measurement.",

            "Cross-platform iROAS breakdown: Shopify $2.14x + Amazon $0.40x = "
            "Combined $2.55x. Without measuring the Amazon halo, you'd undervalue "
            "Facebook by 19%. Factor this into budget allocation decisions.",

            "Trust score: 87/100 (Trustworthy). All validation checks passed — "
            "pre-period balance is excellent (L2=0.023, R²=0.961), placebo tests "
            "show 5% false positive rate (calibrated), and all three estimators "
            "agree on direction and magnitude. Safe to make budget decisions "
            "based on these results.",
        ],
        generated_at=datetime(2026, 2, 16, 14, 30, 0),
    )

    return report


if __name__ == "__main__":
    report = build_sample_report()
    path = save_report_html(report, "./output")
    print(f"Report saved to: {path}")
