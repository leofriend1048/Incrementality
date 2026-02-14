"""Incrementality test reporting.

Generates comprehensive reports from test results including:
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
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from incrementality.models import (
    IncrementalityResult,
    IncrementalROAS,
    MeasurementScope,
    TestDesign,
    TestReport,
    TestScope,
)

logger = logging.getLogger(__name__)
console = Console()


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
        test_start=design.recommended_start_date,
        test_end=design.recommended_end_date,
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


def print_report(report: TestReport) -> None:
    """Print a formatted report to the terminal."""

    # Header
    console.print()
    console.print(Panel(
        f"[bold]{report.test_name}[/bold]\n"
        f"Test ID: {report.test_id}",
        title="Incrementality Test Report",
        border_style="blue",
    ))

    # Design summary
    design_table = Table(title="Test Design", show_header=False, border_style="dim")
    design_table.add_column("Parameter", style="bold")
    design_table.add_column("Value")
    design_table.add_row("Ad Channel", report.ad_channel.value.title())
    design_table.add_row("Test Scope", report.test_scope.value.replace("_", " ").title())
    design_table.add_row("Measurement", report.measurement_scope.value.replace("_", " ").title())
    if report.campaign_ids:
        design_table.add_row("Campaigns", ", ".join(report.campaign_ids))
    design_table.add_row("Duration", f"{report.duration_weeks} weeks")
    design_table.add_row("Test Period", f"{report.test_start} to {report.test_end}")
    design_table.add_row("Treatment DMAs", str(report.num_treatment_dmas))
    design_table.add_row("Holdout DMAs", str(report.num_holdout_dmas))
    console.print(design_table)
    console.print()

    # Incrementality results
    inc = report.incrementality
    result_style = "green" if inc.is_significant and inc.relative_lift > 0 else "red"
    sig_label = "SIGNIFICANT" if inc.is_significant else "NOT SIGNIFICANT"

    results_table = Table(title="Incrementality Results", border_style="dim")
    results_table.add_column("Metric", style="bold")
    results_table.add_column("Value", justify="right")
    results_table.add_column("CI (95%)", justify="right")
    results_table.add_row(
        "Incremental Lift",
        f"[{result_style}]{inc.relative_lift:+.1%}[/{result_style}]",
        f"[{inc.lift_lower_ci:+.1%}, {inc.lift_upper_ci:+.1%}]",
    )
    results_table.add_row(
        "Statistical Significance",
        f"[{result_style}]{sig_label}[/{result_style}]",
        f"p = {inc.p_value:.4f}",
    )
    results_table.add_row("Method", inc.method.replace("_", " ").title(), "")
    results_table.add_row("Effect Size (Cohen's d)", f"{inc.cohen_d:.3f}", "")
    if inc.lift_likelihood > 0:
        results_table.add_row(
            "P(true lift > 0)",
            f"{inc.lift_likelihood:.1%}",
            "",
        )
    console.print(results_table)
    console.print()

    # Ensemble details
    if report.estimator_results and len(report.estimator_results) > 1:
        _print_ensemble_details(report)

    # iROAS
    iroas = report.iroas
    iroas_style = "green" if iroas.iroas >= 1.0 else "red"

    iroas_table = Table(title="Incremental ROAS", border_style="dim")
    iroas_table.add_column("Metric", style="bold")
    iroas_table.add_column("Value", justify="right")
    iroas_table.add_row(
        "Total Incremental Revenue",
        f"${iroas.incremental_revenue:,.2f}",
    )
    iroas_table.add_row(
        "Total Ad Spend (Treatment)",
        f"${iroas.total_ad_spend:,.2f}",
    )
    iroas_table.add_row(
        "iROAS",
        f"[{iroas_style}]{iroas.iroas:.2f}x[/{iroas_style}]",
    )
    iroas_table.add_row(
        "iROAS 95% CI",
        f"[{iroas.iroas_lower_ci:.2f}x, {iroas.iroas_upper_ci:.2f}x]",
    )

    if iroas.shopify_incremental_revenue > 0 or iroas.amazon_incremental_revenue > 0:
        iroas_table.add_row("", "")
        iroas_table.add_row(
            "Shopify Incremental Revenue",
            f"${iroas.shopify_incremental_revenue:,.2f}",
        )
        iroas_table.add_row("Shopify iROAS", f"{iroas.shopify_iroas:.2f}x")
        iroas_table.add_row(
            "Amazon Incremental Revenue",
            f"${iroas.amazon_incremental_revenue:,.2f}",
        )
        iroas_table.add_row("Amazon iROAS", f"{iroas.amazon_iroas:.2f}x")

    console.print(iroas_table)
    console.print()

    # Revenue summary
    rev_table = Table(title="Revenue Summary", border_style="dim")
    rev_table.add_column("Metric", style="bold")
    rev_table.add_column("Treatment", justify="right")
    rev_table.add_column("Holdout", justify="right")
    rev_table.add_row(
        "Total Revenue",
        f"${report.treatment_total_revenue:,.2f}",
        f"${report.holdout_total_revenue:,.2f}",
    )
    rev_table.add_row(
        "Avg Daily Rev / DMA",
        f"${report.treatment_avg_daily_revenue:,.2f}",
        f"${report.holdout_avg_daily_revenue:,.2f}",
    )
    rev_table.add_row(
        "Organic Baseline (est.)",
        f"${report.organic_baseline_revenue:,.2f}",
        "",
    )
    console.print(rev_table)
    console.print()

    # Cross-platform incrementality
    if report.shopify_incrementality or report.amazon_incrementality:
        cross_table = Table(
            title="Cross-Platform Incrementality", border_style="dim"
        )
        cross_table.add_column("Platform", style="bold")
        cross_table.add_column("Lift", justify="right")
        cross_table.add_column("Significant?", justify="right")
        cross_table.add_column("p-value", justify="right")

        if report.shopify_incrementality:
            si = report.shopify_incrementality
            cross_table.add_row(
                "Shopify",
                f"{si.relative_lift:+.1%}",
                "Yes" if si.is_significant else "No",
                f"{si.p_value:.4f}",
            )
        if report.amazon_incrementality:
            ai = report.amazon_incrementality
            cross_table.add_row(
                "Amazon",
                f"{ai.relative_lift:+.1%}",
                "Yes" if ai.is_significant else "No",
                f"{ai.p_value:.4f}",
            )
        console.print(cross_table)
        console.print()

    # Validation report
    if report.validation:
        _print_validation(report)

    # Recommendations
    console.print(Panel(
        "\n".join(f"  {i+1}. {rec}" for i, rec in enumerate(report.recommendations)),
        title="Recommendations",
        border_style="yellow",
    ))
    console.print()


def _print_ensemble_details(report: TestReport) -> None:
    """Print ensemble model details."""
    ensemble_table = Table(title="Ensemble Model Details", border_style="dim")
    ensemble_table.add_column("Estimator", style="bold")
    ensemble_table.add_column("Weight", justify="right")
    ensemble_table.add_column("Lift", justify="right")
    ensemble_table.add_column("p-value", justify="right")
    ensemble_table.add_column("Significant?", justify="right")
    ensemble_table.add_column("L2", justify="right")
    ensemble_table.add_column("R\u00b2", justify="right")

    for name, result in report.estimator_results.items():
        weight = report.estimator_weights.get(name, 0)
        sig_style = "green" if result.is_significant else "red"
        ensemble_table.add_row(
            name.upper(),
            f"{weight:.0%}",
            f"{result.relative_lift:+.1%}",
            f"{result.p_value:.4f}",
            f"[{sig_style}]{'Yes' if result.is_significant else 'No'}[/{sig_style}]",
            f"{result.l2_imbalance:.4f}" if result.l2_imbalance > 0 else "--",
            f"{result.pre_period_r_squared:.3f}" if result.pre_period_r_squared > 0 else "--",
        )

    console.print(ensemble_table)
    console.print()


def _print_validation(report: TestReport) -> None:
    """Print validation results with trust score."""
    v = report.validation

    # Trust score header
    if v.is_trustworthy:
        trust_style = "green"
        trust_label = "TRUSTWORTHY"
    else:
        trust_style = "red"
        trust_label = "NOT TRUSTWORTHY"

    val_table = Table(title="Validation Results", border_style="dim")
    val_table.add_column("Check", style="bold")
    val_table.add_column("Result", justify="right")
    val_table.add_column("Status", justify="right")

    # Trust score
    val_table.add_row(
        "Trust Score",
        f"[{trust_style}]{v.trust_score:.0f}/100[/{trust_style}]",
        f"[{trust_style}]{trust_label}[/{trust_style}]",
    )

    # AA test
    aa_style = "green" if v.aa_test_passed else "red"
    val_table.add_row(
        "AA Test (pre-period)",
        f"p = {v.aa_test_p_value:.4f}",
        f"[{aa_style}]{'PASS' if v.aa_test_passed else 'FAIL'}[/{aa_style}]",
    )

    # Pre-period fit
    l2_style = "green" if v.l2_imbalance < 0.05 else ("yellow" if v.l2_imbalance < 0.10 else "red")
    val_table.add_row(
        "Pre-period L2 Imbalance",
        f"{v.l2_imbalance:.4f}",
        f"[{l2_style}]{'Good' if v.l2_imbalance < 0.05 else ('OK' if v.l2_imbalance < 0.10 else 'Poor')}[/{l2_style}]",
    )

    r2_style = "green" if v.pre_period_r_squared > 0.90 else ("yellow" if v.pre_period_r_squared > 0.80 else "red")
    val_table.add_row(
        "Pre-period R\u00b2",
        f"{v.pre_period_r_squared:.3f}",
        f"[{r2_style}]{'Good' if v.pre_period_r_squared > 0.90 else ('OK' if v.pre_period_r_squared > 0.80 else 'Poor')}[/{r2_style}]",
    )

    # Placebo tests
    fpr_style = "green" if v.false_positive_rate < 0.10 else ("yellow" if v.false_positive_rate < 0.15 else "red")
    val_table.add_row(
        "Placebo Tests",
        f"{v.num_placebo_tests} tests, {v.false_positive_rate:.0%} FPR",
        f"[{fpr_style}]{'Good' if v.false_positive_rate < 0.10 else ('Elevated' if v.false_positive_rate < 0.15 else 'High')}[/{fpr_style}]",
    )

    # Estimator agreement
    agree_style = "green" if v.estimator_agreement > 0.7 else ("yellow" if v.estimator_agreement > 0.5 else "red")
    val_table.add_row(
        "Estimator Agreement",
        f"{v.estimator_agreement:.0%}",
        f"[{agree_style}]{'Good' if v.estimator_agreement > 0.7 else ('Partial' if v.estimator_agreement > 0.5 else 'Disagree')}[/{agree_style}]",
    )

    console.print(val_table)

    # Blockers
    if v.blockers:
        blocker_text = "\n".join(f"  [bold red]X[/bold red] {b}" for b in v.blockers)
        console.print(Panel(
            blocker_text,
            title="BLOCKERS (Hard Stops)",
            border_style="red",
        ))

    # Warnings
    if v.warnings:
        warning_text = "\n".join(f"  [yellow]![/yellow] {w}" for w in v.warnings)
        console.print(Panel(
            warning_text,
            title="Warnings",
            border_style="yellow",
        ))

    console.print()


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
