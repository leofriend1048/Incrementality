"""LIFT — PDF/HTML report generation.

Generates a polished, self-contained HTML report from test results that can
be opened in any browser or converted to PDF via weasyprint.

The report includes:
- Executive summary with key metrics
- Incrementality results with confidence intervals
- iROAS breakdown (Shopify, Amazon, cross-platform)
- Model ensemble details and weights
- Validation trust score with individual checks
- Actionable recommendations
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from incrementality.models import TestReport

logger = logging.getLogger(__name__)


def _pct(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value:+.1%}" if value != 0 else "0.0%"


def _money(amount: float) -> str:
    """Format a dollar amount."""
    return f"${amount:,.2f}"


def _iroas_class(iroas: float) -> str:
    """Return a CSS class for iROAS quality."""
    if iroas >= 3.0:
        return "excellent"
    if iroas >= 1.0:
        return "good"
    if iroas >= 0:
        return "marginal"
    return "bad"


def _iroas_label(iroas: float) -> str:
    """Return a human label for iROAS quality."""
    if iroas >= 3.0:
        return "Excellent"
    if iroas >= 1.0:
        return "Profitable"
    if iroas >= 0:
        return "Below Breakeven"
    return "Negative"


def _trust_class(score: float) -> str:
    """Return a CSS class for trust score."""
    if score >= 80:
        return "excellent"
    if score >= 60:
        return "marginal"
    return "bad"


def _trust_label(score: float) -> str:
    """Return a human label for trust level."""
    if score >= 80:
        return "Trustworthy"
    if score >= 60:
        return "Marginal"
    return "Not Trustworthy"


# ── HTML Template ────────────────────────────────────────────────────────────

_CSS = """\
:root {
    --brand: #06b6d4;
    --brand-dark: #0891b2;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #eab308;
    --bg: #0f172a;
    --surface: #1e293b;
    --surface-2: #334155;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --border: #475569;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 40px;
    max-width: 900px;
    margin: 0 auto;
}

/* Header */
.header {
    border-bottom: 2px solid var(--brand);
    padding-bottom: 24px;
    margin-bottom: 32px;
}
.header .brand {
    font-size: 28px;
    font-weight: 800;
    color: var(--brand);
    letter-spacing: 2px;
}
.header .brand .mark { margin-right: 8px; }
.header .subtitle {
    color: var(--text-muted);
    font-size: 14px;
    margin-top: 4px;
}
.header .test-name {
    font-size: 20px;
    font-weight: 600;
    margin-top: 16px;
}
.header .meta {
    color: var(--text-muted);
    font-size: 13px;
    margin-top: 4px;
}

/* Sections */
.section {
    margin-bottom: 32px;
}
.section-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--brand);
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid var(--surface-2);
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}
.metric-card {
    background: var(--surface);
    border-radius: 8px;
    padding: 20px;
    border-left: 3px solid var(--brand);
}
.metric-card.highlight {
    border-left-color: var(--green);
}
.metric-card.warn {
    border-left-color: var(--yellow);
}
.metric-card.bad {
    border-left-color: var(--red);
}
.metric-label {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
}
.metric-detail {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
}
.metric-value.positive { color: var(--green); }
.metric-value.negative { color: var(--red); }
.metric-value.neutral { color: var(--text); }

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 16px;
    background: var(--surface);
    border-radius: 8px;
    overflow: hidden;
}
th {
    background: var(--surface-2);
    color: var(--brand);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
}
td {
    padding: 10px 16px;
    border-top: 1px solid var(--surface-2);
    font-size: 14px;
}
tr:hover { background: rgba(6, 182, 212, 0.05); }

/* Trust meter */
.trust-meter {
    background: var(--surface);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
}
.trust-bar-bg {
    background: var(--surface-2);
    border-radius: 4px;
    height: 24px;
    overflow: hidden;
    margin: 12px 0;
}
.trust-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
}
.trust-bar-fill.excellent { background: var(--green); }
.trust-bar-fill.marginal { background: var(--yellow); }
.trust-bar-fill.bad { background: var(--red); }

.trust-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.trust-score {
    font-size: 24px;
    font-weight: 700;
}
.trust-verdict {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.trust-verdict.excellent { color: var(--green); }
.trust-verdict.marginal { color: var(--yellow); }
.trust-verdict.bad { color: var(--red); }

/* Validation checks */
.check-item {
    display: flex;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--surface-2);
}
.check-item:last-child { border-bottom: none; }
.check-icon { width: 24px; font-size: 16px; }
.check-icon.pass { color: var(--green); }
.check-icon.fail { color: var(--red); }
.check-name {
    flex: 1;
    font-size: 14px;
}
.check-value {
    font-size: 14px;
    color: var(--text-muted);
    font-family: 'SF Mono', 'Fira Code', monospace;
}

/* Recommendations */
.rec-item {
    background: var(--surface);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 3px solid var(--brand);
}
.rec-number {
    color: var(--brand);
    font-weight: 700;
    margin-right: 8px;
}

/* Blockers/Warnings */
.alert {
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 14px;
}
.alert.blocker {
    background: rgba(239, 68, 68, 0.1);
    border-left: 3px solid var(--red);
    color: var(--red);
}
.alert.warning {
    background: rgba(234, 179, 8, 0.1);
    border-left: 3px solid var(--yellow);
    color: var(--yellow);
}

/* Footer */
.footer {
    margin-top: 48px;
    padding-top: 16px;
    border-top: 1px solid var(--surface-2);
    color: var(--text-muted);
    font-size: 12px;
    text-align: center;
}

@media print {
    body { background: white; color: #1e293b; padding: 20px; }
    .metric-card, table, .trust-meter, .rec-item, .alert {
        background: #f8fafc;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }
    .section { page-break-inside: avoid; }
}
"""


def _render_html(report: TestReport) -> str:
    """Render the full HTML report."""
    inc = report.incrementality
    iroas = report.iroas
    lift_class = "positive" if inc.relative_lift > 0 and inc.is_significant else (
        "negative" if inc.relative_lift < 0 else "neutral"
    )
    iroas_cls = _iroas_class(iroas.iroas)

    # Build sections
    parts = []

    # ── Header ────────────────────────────────────────────────────────
    parts.append(f"""
    <div class="header">
        <div class="brand"><span class="mark">&#9650;</span>LIFT</div>
        <div class="subtitle">Geo Incrementality Platform &mdash; Test Report</div>
        <div class="test-name">{_esc(report.test_name)}</div>
        <div class="meta">
            {report.ad_channel.value.title()} &middot;
            {report.test_scope.value.replace('_', ' ').title()} &middot;
            {report.measurement_scope.value.replace('_', ' ').title()} &middot;
            {report.duration_weeks} weeks &middot;
            {report.test_start} &rarr; {report.test_end}
        </div>
        <div class="meta">
            Test ID: {_esc(report.test_id)} &middot;
            Treatment DMAs: {report.num_treatment_dmas} &middot;
            Holdout DMAs: {report.num_holdout_dmas}
        </div>
    </div>
    """)

    # ── Executive Summary ─────────────────────────────────────────────
    sig_text = "Significant" if inc.is_significant else "Not Significant"
    sig_card_class = "highlight" if inc.is_significant else "warn"
    iroas_card_class = "highlight" if iroas.iroas >= 1.0 else ("warn" if iroas.iroas >= 0 else "bad")

    parts.append(f"""
    <div class="section">
        <div class="section-title">Executive Summary</div>
        <div class="metric-grid">
            <div class="metric-card highlight">
                <div class="metric-label">Incremental Lift</div>
                <div class="metric-value {lift_class}">{_pct(inc.relative_lift)}</div>
                <div class="metric-detail">95% CI: [{_pct(inc.lift_lower_ci)}, {_pct(inc.lift_upper_ci)}]</div>
            </div>
            <div class="metric-card {iroas_card_class}">
                <div class="metric-label">Incremental ROAS</div>
                <div class="metric-value">{iroas.iroas:.2f}x</div>
                <div class="metric-detail">{_iroas_label(iroas.iroas)} &middot; CI: [{iroas.iroas_lower_ci:.2f}x, {iroas.iroas_upper_ci:.2f}x]</div>
            </div>
            <div class="metric-card {sig_card_class}">
                <div class="metric-label">Statistical Significance</div>
                <div class="metric-value" style="font-size:22px">{sig_text}</div>
                <div class="metric-detail">p = {inc.p_value:.4f} &middot; Cohen's d = {inc.cohen_d:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Incremental Revenue</div>
                <div class="metric-value" style="font-size:22px">{_money(iroas.incremental_revenue)}</div>
                <div class="metric-detail">on {_money(iroas.total_ad_spend)} ad spend</div>
            </div>
        </div>
    </div>
    """)

    # ── Revenue Breakdown ─────────────────────────────────────────────
    parts.append(f"""
    <div class="section">
        <div class="section-title">Revenue Breakdown</div>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th style="text-align:right">Treatment</th>
                    <th style="text-align:right">Holdout</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Total Revenue</td>
                    <td style="text-align:right">{_money(report.treatment_total_revenue)}</td>
                    <td style="text-align:right">{_money(report.holdout_total_revenue)}</td>
                </tr>
                <tr>
                    <td>Avg Daily / DMA</td>
                    <td style="text-align:right">{_money(report.treatment_avg_daily_revenue)}</td>
                    <td style="text-align:right">{_money(report.holdout_avg_daily_revenue)}</td>
                </tr>
                <tr>
                    <td>Organic Baseline</td>
                    <td style="text-align:right">{_money(report.organic_baseline_revenue)}</td>
                    <td style="text-align:right">&mdash;</td>
                </tr>
            </tbody>
        </table>
    </div>
    """)

    # ── Cross-Platform iROAS ──────────────────────────────────────────
    if iroas.shopify_incremental_revenue > 0 or iroas.amazon_incremental_revenue > 0:
        cross_rows = ""
        if iroas.shopify_incremental_revenue > 0:
            cross_rows += f"""
                <tr>
                    <td>Shopify</td>
                    <td style="text-align:right">{_money(iroas.shopify_incremental_revenue)}</td>
                    <td style="text-align:right">{iroas.shopify_iroas:.2f}x</td>
                </tr>"""
        if iroas.amazon_incremental_revenue > 0:
            cross_rows += f"""
                <tr>
                    <td>Amazon</td>
                    <td style="text-align:right">{_money(iroas.amazon_incremental_revenue)}</td>
                    <td style="text-align:right">{iroas.amazon_iroas:.2f}x</td>
                </tr>"""
        cross_rows += f"""
                <tr>
                    <td><strong>Combined</strong></td>
                    <td style="text-align:right"><strong>{_money(iroas.incremental_revenue)}</strong></td>
                    <td style="text-align:right"><strong>{iroas.iroas:.2f}x</strong></td>
                </tr>"""

        parts.append(f"""
        <div class="section">
            <div class="section-title">Cross-Platform iROAS</div>
            <table>
                <thead>
                    <tr>
                        <th>Platform</th>
                        <th style="text-align:right">Incremental Revenue</th>
                        <th style="text-align:right">iROAS</th>
                    </tr>
                </thead>
                <tbody>{cross_rows}</tbody>
            </table>
        </div>
        """)

    # ── Cross-Platform Incrementality ─────────────────────────────────
    if report.shopify_incrementality or report.amazon_incrementality:
        plat_rows = ""
        if report.shopify_incrementality:
            si = report.shopify_incrementality
            sig_icon = "&#10003;" if si.is_significant else "&#10007;"
            sig_cls = "pass" if si.is_significant else "fail"
            plat_rows += f"""
                <tr>
                    <td>Shopify</td>
                    <td style="text-align:right">{_pct(si.relative_lift)}</td>
                    <td style="text-align:center"><span class="check-icon {sig_cls}">{sig_icon}</span></td>
                    <td style="text-align:right">{si.p_value:.4f}</td>
                </tr>"""
        if report.amazon_incrementality:
            ai = report.amazon_incrementality
            sig_icon = "&#10003;" if ai.is_significant else "&#10007;"
            sig_cls = "pass" if ai.is_significant else "fail"
            plat_rows += f"""
                <tr>
                    <td>Amazon</td>
                    <td style="text-align:right">{_pct(ai.relative_lift)}</td>
                    <td style="text-align:center"><span class="check-icon {sig_cls}">{sig_icon}</span></td>
                    <td style="text-align:right">{ai.p_value:.4f}</td>
                </tr>"""

        parts.append(f"""
        <div class="section">
            <div class="section-title">Platform-Level Incrementality</div>
            <table>
                <thead>
                    <tr>
                        <th>Platform</th>
                        <th style="text-align:right">Lift</th>
                        <th style="text-align:center">Significant</th>
                        <th style="text-align:right">p-value</th>
                    </tr>
                </thead>
                <tbody>{plat_rows}</tbody>
            </table>
        </div>
        """)

    # ── Ensemble Model Details ────────────────────────────────────────
    if report.estimator_results and len(report.estimator_results) > 1:
        model_rows = ""
        for name, result in report.estimator_results.items():
            weight = report.estimator_weights.get(name, 0)
            sig_icon = "&#10003;" if result.is_significant else "&#10007;"
            sig_cls = "pass" if result.is_significant else "fail"
            l2 = f"{result.l2_imbalance:.4f}" if result.l2_imbalance > 0 else "&mdash;"
            r2 = f"{result.pre_period_r_squared:.3f}" if result.pre_period_r_squared > 0 else "&mdash;"
            model_rows += f"""
                <tr>
                    <td><strong>{_esc(name.upper())}</strong></td>
                    <td style="text-align:right">{weight:.0%}</td>
                    <td style="text-align:right">{_pct(result.relative_lift)}</td>
                    <td style="text-align:right">{result.p_value:.4f}</td>
                    <td style="text-align:center"><span class="check-icon {sig_cls}">{sig_icon}</span></td>
                    <td style="text-align:right">{l2}</td>
                    <td style="text-align:right">{r2}</td>
                </tr>"""

        parts.append(f"""
        <div class="section">
            <div class="section-title">Model Ensemble</div>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th style="text-align:right">Weight</th>
                        <th style="text-align:right">Lift</th>
                        <th style="text-align:right">p-value</th>
                        <th style="text-align:center">Sig.</th>
                        <th style="text-align:right">L2</th>
                        <th style="text-align:right">R&sup2;</th>
                    </tr>
                </thead>
                <tbody>{model_rows}</tbody>
            </table>
            <div style="color: var(--text-muted); font-size: 12px; padding: 4px 0;">
                Method: {_esc(inc.method.replace('_', ' ').title())}
                {f' &middot; P(true lift > 0): {inc.lift_likelihood:.1%}' if inc.lift_likelihood > 0 else ''}
            </div>
        </div>
        """)

    # ── Validation & Trust Score ──────────────────────────────────────
    if report.validation:
        v = report.validation
        tc = _trust_class(v.trust_score)
        tl = _trust_label(v.trust_score)

        aa_icon, aa_cls = ("&#10003;", "pass") if v.aa_test_passed else ("&#10007;", "fail")

        fpr_pass = v.false_positive_rate <= 0.15
        fpr_icon, fpr_cls = ("&#10003;", "pass") if fpr_pass else ("&#10007;", "fail")

        agree_pass = v.estimator_agreement >= 0.50
        agree_icon, agree_cls = ("&#10003;", "pass") if agree_pass else ("&#10007;", "fail")

        blockers_html = ""
        if v.blockers:
            blockers_html = "".join(
                f'<div class="alert blocker">&#10007; {_esc(b)}</div>' for b in v.blockers
            )

        warnings_html = ""
        if v.warnings:
            warnings_html = "".join(
                f'<div class="alert warning">&#9888; {_esc(w)}</div>' for w in v.warnings
            )

        parts.append(f"""
        <div class="section">
            <div class="section-title">Validation &amp; Trust Score</div>
            <div class="trust-meter">
                <div class="trust-label">
                    <span class="trust-score">{v.trust_score:.0f}<span style="font-size:14px;color:var(--text-muted)">/100</span></span>
                    <span class="trust-verdict {tc}">{tl}</span>
                </div>
                <div class="trust-bar-bg">
                    <div class="trust-bar-fill {tc}" style="width:{v.trust_score:.0f}%"></div>
                </div>
            </div>

            <div style="background: var(--surface); border-radius: 8px; padding: 16px;">
                <div class="check-item">
                    <span class="check-icon {aa_cls}">{aa_icon}</span>
                    <span class="check-name">AA Test (pre-period balance)</span>
                    <span class="check-value">p = {v.aa_test_p_value:.4f}</span>
                </div>
                <div class="check-item">
                    <span class="check-icon">&#8226;</span>
                    <span class="check-name">Pre-period L2 imbalance</span>
                    <span class="check-value">{v.l2_imbalance:.4f}</span>
                </div>
                <div class="check-item">
                    <span class="check-icon">&#8226;</span>
                    <span class="check-name">Pre-period R&sup2;</span>
                    <span class="check-value">{v.pre_period_r_squared:.3f}</span>
                </div>
                <div class="check-item">
                    <span class="check-icon {fpr_cls}">{fpr_icon}</span>
                    <span class="check-name">Placebo Tests ({v.num_placebo_tests} tests)</span>
                    <span class="check-value">FPR = {v.false_positive_rate:.0%}</span>
                </div>
                <div class="check-item">
                    <span class="check-icon {agree_cls}">{agree_icon}</span>
                    <span class="check-name">Estimator Agreement</span>
                    <span class="check-value">{v.estimator_agreement:.0%}</span>
                </div>
            </div>

            {blockers_html}
            {warnings_html}
        </div>
        """)

    # ── Recommendations ───────────────────────────────────────────────
    if report.recommendations:
        rec_html = ""
        for i, rec in enumerate(report.recommendations, 1):
            rec_html += f"""
            <div class="rec-item">
                <span class="rec-number">{i}.</span> {_esc(rec)}
            </div>"""

        parts.append(f"""
        <div class="section">
            <div class="section-title">Recommendations</div>
            {rec_html}
        </div>
        """)

    # ── Footer ────────────────────────────────────────────────────────
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    parts.append(f"""
    <div class="footer">
        Generated by LIFT &mdash; Geo Incrementality Platform &middot; {now}
    </div>
    """)

    # Assemble full HTML
    body = "\n".join(parts)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LIFT Report &mdash; {_esc(report.test_name)}</title>
    <style>{_CSS}</style>
</head>
<body>
{body}
</body>
</html>"""


def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ── Public API ───────────────────────────────────────────────────────────────

def save_report_html(report: TestReport, output_dir: str | Path) -> Path:
    """Save the report as a self-contained HTML file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.test_id}_report.html"
    html = _render_html(report)
    path.write_text(html, encoding="utf-8")
    logger.info(f"HTML report saved to {path}")
    return path


def save_report_pdf(report: TestReport, output_dir: str | Path) -> Path:
    """Save the report as a PDF file.

    Requires weasyprint: pip install 'incrementality[pdf]'
    Falls back to HTML if weasyprint is not installed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html = _render_html(report)

    try:
        from weasyprint import HTML as WeasyHTML
        pdf_path = output_dir / f"{report.test_id}_report.pdf"
        WeasyHTML(string=html).write_pdf(pdf_path)
        logger.info(f"PDF report saved to {pdf_path}")
        return pdf_path
    except ImportError:
        logger.warning("weasyprint not installed — saving as HTML instead")
        return save_report_html(report, output_dir)
