"""LIFT — Incrementality test reporting.

Generates comprehensive branded reports from test results including:
- Test design summary
- Incrementality results with confidence intervals
- iROAS by platform (Shopify, Amazon, cross-platform)
- Validation results (trust score, blockers, warnings)
- Ensemble model details and weights
- Recommendations for budget allocation
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import date
from pathlib import Path

from incrementality.models import (
    IncrementalityResult,
    IncrementalROAS,
    MeasurementScope,
    TestDesign,
    TestReport,
    TestScope,
)
from incrementality.ui import (
    branded_table,
    console,
    iroas_value,
    kv,
    lift_value,
    money,
    pass_fail,
    quality_badge,
    score_bar,
    section,
    sig_badge,
    spacer,
    trust_meter,
)

logger = logging.getLogger(__name__)


def generate_report(
    design: TestDesign,
    incrementality: IncrementalityResult,
    iroas: IncrementalROAS,
    treatment_total_revenue: float,
    holdout_total_revenue: float,
    test_duration_days: int,
    shopify_incrementality: IncrementalityResult | None = None,
    amazon_incrementality: IncrementalityResult | None = None,
) -> TestReport:
    """Generate a complete test report from analysis results."""

    treatment_avg_daily = (
        treatment_total_revenue / (design.num_treatment_dmas * test_duration_days)
        if design.num_treatment_dmas > 0 and test_duration_days > 0
        else 0
    )
    holdout_avg_daily = (
        holdout_total_revenue / (design.num_holdout_dmas * test_duration_days)
        if design.num_holdout_dmas > 0 and test_duration_days > 0
        else 0
    )

    organic_baseline = holdout_avg_daily * design.num_treatment_dmas * test_duration_days

    recommendations = _generate_recommendations(
        incrementality, iroas, design, shopify_incrementality, amazon_incrementality,
    )

    return TestReport(
        test_id=design.test_id,
        test_name=design.name,
        ad_channel=design.ad_channel,
        test_scope=design.test_scope,
        measurement_scope=design.measurement_scope,
        campaign_ids=design.campaign_ids,
        duration_weeks=design.duration_weeks,
        num_treatment_dmas=design.num_treatment_dmas,
        num_holdout_dmas=design.num_holdout_dmas,
        test_start=(
            design.recommended_start_date
            or (design.deployed_at.date() if design.deployed_at else date.today())
        ),
        test_end=(
            design.recommended_end_date
            or date.today()
        ),
        incrementality=incrementality,
        iroas=iroas,
        treatment_total_revenue=treatment_total_revenue,
        holdout_total_revenue=holdout_total_revenue,
        treatment_avg_daily_revenue=treatment_avg_daily,
        holdout_avg_daily_revenue=holdout_avg_daily,
        organic_baseline_revenue=organic_baseline,
        shopify_incrementality=shopify_incrementality,
        amazon_incrementality=amazon_incrementality,
        recommendations=recommendations,
    )


def _generate_recommendations(
    incrementality: IncrementalityResult,
    iroas: IncrementalROAS,
    design: TestDesign,
    shopify_inc: IncrementalityResult | None,
    amazon_inc: IncrementalityResult | None,
) -> list[str]:
    """Generate actionable recommendations from test results."""
    recs = []

    if not incrementality.is_significant:
        recs.append(
            f"The test did NOT reach statistical significance (p={incrementality.p_value:.3f}). "
            f"The measured lift of {incrementality.relative_lift:.1%} could be due to chance. "
            f"Consider running a longer test or increasing the holdout group size."
        )
    else:
        recs.append(
            f"The test reached statistical significance (p={incrementality.p_value:.3f}). "
            f"There is strong evidence that {design.ad_channel.value} ads drive "
            f"{incrementality.relative_lift:.1%} incremental lift."
        )

    if iroas.iroas > 0:
        if iroas.iroas >= 3.0:
            recs.append(
                f"iROAS of {iroas.iroas:.2f}x is excellent. "
                f"Every $1 spent generates ${iroas.iroas:.2f} in incremental revenue. "
                f"Consider increasing budget on {design.ad_channel.value}."
            )
        elif iroas.iroas >= 1.0:
            recs.append(
                f"iROAS of {iroas.iroas:.2f}x is profitable. "
                f"Ads are generating positive incremental returns. "
                f"Current spend level appears efficient."
            )
        else:
            recs.append(
                f"iROAS of {iroas.iroas:.2f}x is below breakeven. "
                f"Ad spend exceeds incremental revenue. "
                f"Consider reducing {design.ad_channel.value} budget or "
                f"optimizing creative/targeting."
            )
    else:
        recs.append(
            f"Negative iROAS ({iroas.iroas:.2f}x) suggests ads may be cannibalizing "
            f"organic revenue. Investigate audience overlap and frequency capping."
        )

    if (
        design.measurement_scope == MeasurementScope.SHOPIFY_AND_AMAZON
        and shopify_inc
        and amazon_inc
    ):
        if amazon_inc.relative_lift > 0 and amazon_inc.is_significant:
            recs.append(
                f"Significant Amazon halo effect detected: "
                f"{amazon_inc.relative_lift:.1%} incremental lift on Amazon. "
                f"{design.ad_channel.value} ads are driving cross-platform conversions. "
                f"Platform-reported ROAS underestimates true value."
            )
        elif amazon_inc.relative_lift > 0:
            recs.append(
                f"Positive but non-significant Amazon halo ({amazon_inc.relative_lift:.1%}). "
                f"There may be cross-platform effects -- run a longer test to confirm."
            )

        if iroas.shopify_iroas > 0 and iroas.amazon_iroas > 0:
            total_iroas = iroas.shopify_iroas + iroas.amazon_iroas
            amazon_share = iroas.amazon_iroas / total_iroas * 100
            recs.append(
                f"Cross-platform iROAS breakdown: "
                f"Shopify {iroas.shopify_iroas:.2f}x + Amazon {iroas.amazon_iroas:.2f}x. "
                f"Amazon accounts for {amazon_share:.0f}% of total incremental returns."
            )

    if design.test_scope == TestScope.CAMPAIGN and design.campaign_ids:
        recs.append(
            f"This was a campaign-level test for campaigns: {', '.join(design.campaign_ids)}. "
            f"Results apply specifically to these campaigns. "
            f"Other campaigns in {design.ad_channel.value} may have different incrementality."
        )

    return recs


# ── Report Printer ────────────────────────────────────────────────────────────

def print_report(report: TestReport) -> None:
    """Print a branded LIFT report to the terminal."""

    # ── Header ──
    section(f"Incrementality Report: {report.test_name}")
    kv("Test ID", f"[accent]{report.test_id}[/accent]")
    kv("Ad Channel", f"[accent]{report.ad_channel.value.title()}[/accent]")
    kv("Test Scope", report.test_scope.value.replace("_", " ").title())
    kv("Measurement", report.measurement_scope.value.replace("_", " ").title())
    if report.campaign_ids:
        kv("Campaigns", ", ".join(report.campaign_ids))
    kv("Duration", f"{report.duration_weeks} weeks")
    kv("Test Period", f"{report.test_start} → {report.test_end}")
    kv("Treatment DMAs", str(report.num_treatment_dmas))
    kv("Holdout DMAs", str(report.num_holdout_dmas))

    # ── Primary Results ──
    inc = report.incrementality
    section("Incrementality Results")

    kv("Incremental Lift", lift_value(
        inc.relative_lift, inc.lift_lower_ci, inc.lift_upper_ci, inc.is_significant,
    ))
    kv("Significance", sig_badge(inc.is_significant, inc.p_value))
    kv("Method", f"[muted]{inc.method.replace('_', ' ').title()}[/muted]")
    kv("Effect Size (d)", f"{inc.cohen_d:.3f}")
    if inc.lift_likelihood > 0:
        ll_bar = score_bar(inc.lift_likelihood * 100, 100, 15)
        kv("P(true lift > 0)", f"{ll_bar}  {inc.lift_likelihood:.1%}")

    # ── Ensemble Details ──
    if report.estimator_results and len(report.estimator_results) > 1:
        _print_ensemble_details(report)

    # ── iROAS ──
    iroas = report.iroas
    section("Incremental ROAS")

    kv("Incremental Revenue", money(iroas.incremental_revenue))
    kv("Total Ad Spend", money(iroas.total_ad_spend))
    kv("iROAS", iroas_value(iroas.iroas))
    kv("iROAS 95% CI", f"[muted][{iroas.iroas_lower_ci:.2f}x, {iroas.iroas_upper_ci:.2f}x][/muted]")

    if iroas.shopify_incremental_revenue > 0 or iroas.amazon_incremental_revenue > 0:
        spacer()
        kv("Shopify Incremental", money(iroas.shopify_incremental_revenue))
        kv("Shopify iROAS", f"{iroas.shopify_iroas:.2f}x")
        kv("Amazon Incremental", money(iroas.amazon_incremental_revenue))
        kv("Amazon iROAS", f"{iroas.amazon_iroas:.2f}x")

    # ── Spend Response Curve ──
    if report.spend_response:
        _print_spend_response(report.spend_response)

    # ── Revenue Summary ──
    section("Revenue Summary")

    table = branded_table("", show_header=True)
    table.add_column("Metric", style="label")
    table.add_column("Treatment", justify="right")
    table.add_column("Holdout", justify="right")
    table.add_row(
        "Total Revenue",
        money(report.treatment_total_revenue),
        money(report.holdout_total_revenue),
    )
    table.add_row(
        "Avg Daily / DMA",
        money(report.treatment_avg_daily_revenue),
        money(report.holdout_avg_daily_revenue),
    )
    table.add_row(
        "Organic Baseline",
        money(report.organic_baseline_revenue),
        "",
    )
    console.print(table)

    # ── Cross-Platform ──
    if report.shopify_incrementality or report.amazon_incrementality:
        section("Cross-Platform Incrementality")

        table = branded_table("", show_header=True)
        table.add_column("Platform", style="label")
        table.add_column("Lift", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("p-value", justify="right")

        if report.shopify_incrementality:
            si = report.shopify_incrementality
            table.add_row(
                "Shopify",
                f"{si.relative_lift:+.1%}",
                pass_fail(si.is_significant),
                f"{si.p_value:.4f}",
            )
        if report.amazon_incrementality:
            ai = report.amazon_incrementality
            table.add_row(
                "Amazon",
                f"{ai.relative_lift:+.1%}",
                pass_fail(ai.is_significant),
                f"{ai.p_value:.4f}",
            )
        console.print(table)

    # ── Validation ──
    if report.validation:
        _print_validation(report)

    # ── Recommendations ──
    section("Recommendations")
    for i, rec in enumerate(report.recommendations, 1):
        console.print(f"    [accent]{i}.[/accent]  {rec}")
        spacer()


def _print_ensemble_details(report: TestReport) -> None:
    """Print ensemble model breakdown."""
    section("Model Ensemble")

    table = branded_table("", show_header=True)
    table.add_column("Model", style="accent")
    table.add_column("Weight", justify="right")
    table.add_column("Lift", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("L2", justify="right", style="muted")
    table.add_column("R\u00b2", justify="right", style="muted")

    for name, result in report.estimator_results.items():
        weight = report.estimator_weights.get(name, 0)

        weight_bar = score_bar(weight * 100, 100, 8)

        table.add_row(
            name.upper(),
            f"{weight_bar} {weight:.0%}",
            f"{result.relative_lift:+.1%}",
            f"{result.p_value:.4f}",
            pass_fail(result.is_significant),
            f"{result.l2_imbalance:.4f}" if result.l2_imbalance > 0 else "—",
            f"{result.pre_period_r_squared:.3f}" if result.pre_period_r_squared > 0 else "—",
        )

    console.print(table)


def _print_validation(report: TestReport) -> None:
    """Print validation results with visual trust score."""
    v = report.validation

    section("Validation & Trust Score")

    # Trust score meter
    trust_meter(v.trust_score)
    spacer()

    # Individual checks
    kv("AA Test (pre-period)",
       f"p = {v.aa_test_p_value:.4f}  {pass_fail(v.aa_test_passed)}")

    kv("Pre-period L2",
       f"{v.l2_imbalance:.4f}  {quality_badge(v.l2_imbalance, 0.05, 0.10, reverse=True)}")

    kv("Pre-period R\u00b2",
       f"{v.pre_period_r_squared:.3f}  {quality_badge(v.pre_period_r_squared, 0.90, 0.80)}")

    kv("Placebo Tests",
       f"{v.num_placebo_tests} tests, {v.false_positive_rate:.0%} FPR  "
       f"{quality_badge(v.false_positive_rate, 0.10, 0.15, reverse=True)}")

    kv("Estimator Agreement",
       f"{v.estimator_agreement:.0%}  "
       f"{quality_badge(v.estimator_agreement, 0.70, 0.50)}")

    # Blockers
    if v.blockers:
        spacer()
        console.print("    [bad]BLOCKERS:[/bad]")
        for b in v.blockers:
            console.print(f"    [bad]✗[/bad]  {b}")

    # Warnings
    if v.warnings:
        spacer()
        console.print("    [warn]WARNINGS:[/warn]")
        for w in v.warnings:
            console.print(f"    [warn]⚠[/warn]  {w}")

    spacer()


def _print_spend_response(sr: dict) -> None:
    """Print spend response curve analysis."""
    section("Spend Response Curve")

    kv("Current Daily Spend", money(sr.get("current_spend", 0)))
    kv("Current Marginal ROAS", f"${sr.get('current_marginal_roas', 0):.2f}")
    spacer()
    kv("Optimal Daily Spend", money(sr.get("optimal_spend", 0)))
    kv("Optimal iROAS", f"${sr.get('optimal_iroas', 0):.2f}")

    lower = sr.get("optimal_spend_lower", 0)
    upper = sr.get("optimal_spend_upper", 0)
    if lower > 0 and upper > 0:
        kv("Optimal Spend 95% CI", f"[muted][${lower:,.0f}, ${upper:,.0f}][/muted]")

    spacer()
    direction = sr.get("spend_change_direction", "")
    pct = sr.get("spend_change_pct", 0)
    if direction == "decrease":
        kv("Recommendation", f"[bad]Decrease spend {pct:+.0f}%[/bad]")
    elif direction == "increase":
        kv("Recommendation", f"[ok]Increase spend {pct:+.0f}%[/ok]")
    else:
        kv("Recommendation", "[ok]Maintain current spend[/ok]")

    rec = sr.get("recommendation", "")
    if rec:
        console.print(f"    [muted]{rec}[/muted]")

    spacer()
    n_dmas = sr.get("n_dmas_used", 0)
    r_sq = sr.get("r_squared", 0)
    calibrated = sr.get("calibrated", False)
    kv("Fit Quality",
       f"R\u00b2={r_sq:.3f}, {n_dmas} DMAs"
       f"{', calibrated to causal iROAS' if calibrated else ''}")


# ── File Output ───────────────────────────────────────────────────────────────

def save_report_json(report: TestReport, output_dir: str | Path) -> Path:
    """Save report as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.test_id}_report.json"
    with open(path, "w") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
    logger.info(f"Report saved to {path}")
    return path


def save_report_csv(report: TestReport, output_dir: str | Path) -> Path:
    """Save key metrics as a flat CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.test_id}_summary.csv"

    rows = [
        {"metric": "test_id", "value": report.test_id},
        {"metric": "test_name", "value": report.test_name},
        {"metric": "ad_channel", "value": report.ad_channel.value},
        {"metric": "test_scope", "value": report.test_scope.value},
        {"metric": "measurement_scope", "value": report.measurement_scope.value},
        {"metric": "duration_weeks", "value": str(report.duration_weeks)},
        {"metric": "num_treatment_dmas", "value": str(report.num_treatment_dmas)},
        {"metric": "num_holdout_dmas", "value": str(report.num_holdout_dmas)},
        {"metric": "method", "value": report.incrementality.method},
        {"metric": "incremental_lift", "value": f"{report.incrementality.relative_lift:.4f}"},
        {"metric": "lift_lower_ci", "value": f"{report.incrementality.lift_lower_ci:.4f}"},
        {"metric": "lift_upper_ci", "value": f"{report.incrementality.lift_upper_ci:.4f}"},
        {"metric": "p_value", "value": f"{report.incrementality.p_value:.6f}"},
        {"metric": "is_significant", "value": str(report.incrementality.is_significant)},
        {"metric": "lift_likelihood", "value": f"{report.incrementality.lift_likelihood:.4f}"},
        {"metric": "iroas", "value": f"{report.iroas.iroas:.4f}"},
        {"metric": "iroas_lower_ci", "value": f"{report.iroas.iroas_lower_ci:.4f}"},
        {"metric": "iroas_upper_ci", "value": f"{report.iroas.iroas_upper_ci:.4f}"},
        {"metric": "incremental_revenue", "value": f"{report.iroas.incremental_revenue:.2f}"},
        {"metric": "total_ad_spend", "value": f"{report.iroas.total_ad_spend:.2f}"},
        {"metric": "treatment_total_revenue", "value": f"{report.treatment_total_revenue:.2f}"},
        {"metric": "holdout_total_revenue", "value": f"{report.holdout_total_revenue:.2f}"},
        {"metric": "organic_baseline_revenue", "value": f"{report.organic_baseline_revenue:.2f}"},
    ]

    # Validation metrics
    if report.validation:
        v = report.validation
        rows.extend([
            {"metric": "trust_score", "value": f"{v.trust_score:.1f}"},
            {"metric": "is_trustworthy", "value": str(v.is_trustworthy)},
            {"metric": "aa_test_passed", "value": str(v.aa_test_passed)},
            {"metric": "aa_test_p_value", "value": f"{v.aa_test_p_value:.4f}"},
            {"metric": "false_positive_rate", "value": f"{v.false_positive_rate:.4f}"},
            {"metric": "num_placebo_tests", "value": str(v.num_placebo_tests)},
            {"metric": "estimator_agreement", "value": f"{v.estimator_agreement:.4f}"},
            {"metric": "l2_imbalance", "value": f"{v.l2_imbalance:.4f}"},
            {"metric": "pre_period_r_squared", "value": f"{v.pre_period_r_squared:.4f}"},
            {"metric": "num_blockers", "value": str(len(v.blockers))},
        ])

    # Ensemble details
    for name, weight in report.estimator_weights.items():
        rows.append({"metric": f"weight_{name}", "value": f"{weight:.4f}"})
    for name, result in report.estimator_results.items():
        rows.append({"metric": f"lift_{name}", "value": f"{result.relative_lift:.4f}"})
        rows.append({"metric": f"p_value_{name}", "value": f"{result.p_value:.6f}"})

    if report.iroas.shopify_iroas > 0:
        rows.append({"metric": "shopify_iroas", "value": f"{report.iroas.shopify_iroas:.4f}"})
        rows.append({
            "metric": "shopify_incremental_revenue",
            "value": f"{report.iroas.shopify_incremental_revenue:.2f}",
        })
    if report.iroas.amazon_iroas > 0:
        rows.append({"metric": "amazon_iroas", "value": f"{report.iroas.amazon_iroas:.4f}"})
        rows.append({
            "metric": "amazon_incremental_revenue",
            "value": f"{report.iroas.amazon_incremental_revenue:.2f}",
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Summary CSV saved to {path}")
    return path
