"""LIFT — PDF/HTML report generation (Haus-inspired design).

Generates a polished, self-contained HTML report from test results that can
be opened in any browser or converted to PDF via weasyprint.

The report includes:
- Key takeaway callout
- Executive summary with hero metrics
- Lift confidence interval visualization
- Incrementality Factor and CPIA
- iROAS breakdown (Shopify, Amazon, cross-platform)
- Model ensemble with visual weight bars
- Validation trust score with individual checks
- Methodology overview
- Actionable recommendations
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from incrementality.models import TestReport

logger = logging.getLogger(__name__)


def _pct(value: float) -> str:
    return f"{value:+.1%}" if value != 0 else "0.0%"


def _money(amount: float) -> str:
    return f"${amount:,.2f}"


def _iroas_label(iroas: float) -> str:
    if iroas >= 3.0:
        return "Excellent"
    if iroas >= 1.0:
        return "Profitable"
    if iroas >= 0:
        return "Below Breakeven"
    return "Negative"


def _trust_class(score: float) -> str:
    if score >= 80:
        return "excellent"
    if score >= 60:
        return "marginal"
    return "bad"


def _trust_label(score: float) -> str:
    if score >= 80:
        return "Trustworthy"
    if score >= 60:
        return "Marginal"
    return "Not Trustworthy"


# ── CSS ──────────────────────────────────────────────────────────────────────

_CSS = """\
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --brand: #0251F3;
    --brand-light: #84CBFF;
    --brand-glow: rgba(2, 81, 243, 0.10);
    --brand-border: rgba(2, 81, 243, 0.25);
    --green: #34D399;
    --green-dim: rgba(52, 211, 153, 0.12);
    --red: #F87171;
    --red-dim: rgba(248, 113, 113, 0.10);
    --yellow: #FBBF24;
    --yellow-dim: rgba(251, 191, 36, 0.10);
    --bg: #0B1120;
    --surface: #111827;
    --surface-2: #1C2B4A;
    --surface-3: #243A63;
    --text: #E8EDF5;
    --text-muted: #7B93B8;
    --border: #1E3054;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 0;
    max-width: 960px;
    margin: 0 auto;
    -webkit-font-smoothing: antialiased;
}

/* ── Header ── */
.header {
    background: linear-gradient(135deg, #0B1120 0%, #111D35 50%, #0B1120 100%);
    padding: 48px 48px 40px;
    border-bottom: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}
.header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--brand), var(--brand-light), var(--brand));
}
.brand {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 4px;
    background: linear-gradient(135deg, var(--brand), var(--brand-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.subtitle {
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 500;
    margin-top: 4px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.test-name {
    font-size: 24px;
    font-weight: 700;
    margin-top: 24px;
    color: var(--text);
    line-height: 1.3;
}
.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 16px;
}
.meta-tag {
    display: inline-flex;
    align-items: center;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-muted);
}
.meta-tag .tag-value {
    color: var(--text);
    margin-left: 4px;
}

/* ── Content ── */
.content { padding: 40px 48px 48px; }

/* ── Sections ── */
.section { margin-bottom: 40px; }
.section-title {
    font-size: 11px;
    font-weight: 700;
    color: var(--brand-light);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Key Takeaway ── */
.takeaway {
    background: linear-gradient(135deg, rgba(2, 81, 243, 0.08), rgba(132, 203, 255, 0.04));
    border: 1px solid var(--brand-border);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 40px;
    position: relative;
}
.takeaway::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, var(--brand), var(--brand-light));
    border-radius: 12px 0 0 12px;
}
.takeaway-label {
    font-size: 10px;
    font-weight: 700;
    color: var(--brand-light);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
}
.takeaway p {
    font-size: 15px;
    line-height: 1.7;
    color: var(--text-muted);
}
.takeaway strong { color: var(--text); }
.takeaway .hl { color: var(--green); font-weight: 700; }
.takeaway .hl-brand { color: var(--brand-light); font-weight: 700; }

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-bottom: 16px;
}
.metric-grid.three-col {
    grid-template-columns: repeat(3, 1fr);
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--brand);
    opacity: 0;
    transition: opacity 0.2s;
}
.metric-card.green::before { background: var(--green); opacity: 1; }
.metric-card.yellow::before { background: var(--yellow); opacity: 1; }
.metric-card.red::before { background: var(--red); opacity: 1; }
.metric-card.brand::before { background: linear-gradient(90deg, var(--brand), var(--brand-light)); opacity: 1; }
.metric-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 36px;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.5px;
}
.metric-value.sm { font-size: 26px; }
.metric-sub {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 8px;
    font-weight: 500;
}
.positive { color: var(--green); }
.negative { color: var(--red); }
.neutral { color: var(--text); }

/* ── CI Bar ── */
.ci-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}
.ci-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
    margin-bottom: 16px;
}
.ci-bar-track {
    position: relative;
    height: 36px;
    background: var(--surface-2);
    border-radius: 8px;
    overflow: visible;
    margin: 0 0 10px 0;
}
.ci-bar-range {
    position: absolute;
    height: 100%;
    border-radius: 8px;
    opacity: 0.25;
}
.ci-bar-range.positive { background: var(--green); }
.ci-bar-range.negative { background: var(--red); }
.ci-bar-range.mixed { background: var(--yellow); }
.ci-bar-point {
    position: absolute;
    top: 50%;
    width: 4px;
    height: 26px;
    border-radius: 2px;
    transform: translateY(-50%) translateX(-50%);
    background: var(--text);
}
.ci-bar-zero {
    position: absolute;
    top: 0;
    height: 100%;
    width: 1px;
    background: var(--text-muted);
    opacity: 0.4;
}
.ci-ticks {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text-muted);
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-weight: 500;
}

/* ── Likelihood Ring ── */
.likelihood-row {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
}
.likelihood-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.ring-container {
    position: relative;
    width: 68px;
    height: 68px;
    flex-shrink: 0;
}
.ring-container svg {
    transform: rotate(-90deg);
    width: 68px;
    height: 68px;
}
.ring-container .ring-bg {
    fill: none;
    stroke: var(--surface-2);
    stroke-width: 4.5;
}
.ring-container .ring-fill {
    fill: none;
    stroke-width: 4.5;
    stroke-linecap: round;
}
.ring-container .ring-fill.green { stroke: var(--green); }
.ring-container .ring-fill.yellow { stroke: var(--yellow); }
.ring-container .ring-fill.brand { stroke: var(--brand); }
.ring-label {
    font-size: 15px;
    font-weight: 800;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}
.likelihood-text .ll-title {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}
.likelihood-text .ll-value {
    font-size: 20px;
    font-weight: 700;
    margin-top: 4px;
}
.likelihood-text .ll-detail {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 2px;
}

/* ── Tables ── */
table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}
th {
    background: var(--surface-2);
    color: var(--brand-light);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 14px 18px;
    text-align: left;
    font-weight: 700;
}
td {
    padding: 14px 18px;
    border-top: 1px solid var(--border);
    font-size: 14px;
    font-weight: 500;
}
tr:first-child td { border-top: none; }

/* ── Weight bars ── */
.weight-bar-bg {
    display: inline-block;
    width: 60px;
    height: 6px;
    background: var(--surface-3);
    border-radius: 3px;
    overflow: hidden;
    vertical-align: middle;
    margin-right: 8px;
}
.weight-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--brand), var(--brand-light));
}

/* ── Trust Meter ── */
.trust-meter {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px;
    margin-bottom: 16px;
}
.trust-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}
.trust-score-num {
    font-size: 32px;
    font-weight: 800;
}
.trust-score-num span { font-size: 15px; color: var(--text-muted); font-weight: 500; }
.trust-verdict {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 6px 14px;
    border-radius: 6px;
}
.trust-verdict.excellent { background: var(--green-dim); color: var(--green); }
.trust-verdict.marginal { background: var(--yellow-dim); color: var(--yellow); }
.trust-verdict.bad { background: var(--red-dim); color: var(--red); }
.trust-bar-bg {
    background: var(--surface-2);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.trust-bar-fill {
    height: 100%;
    border-radius: 4px;
}
.trust-bar-fill.excellent { background: linear-gradient(90deg, var(--green), #6EE7B7); }
.trust-bar-fill.marginal { background: linear-gradient(90deg, var(--yellow), #FDE68A); }
.trust-bar-fill.bad { background: linear-gradient(90deg, var(--red), #FCA5A5); }

/* ── Validation Checks ── */
.checks {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 8px 20px;
}
.check-item {
    display: flex;
    align-items: center;
    padding: 14px 0;
    border-bottom: 1px solid var(--border);
}
.check-item:last-child { border-bottom: none; }
.check-icon {
    width: 28px;
    font-size: 13px;
    flex-shrink: 0;
    font-weight: 700;
}
.check-icon.pass { color: var(--green); }
.check-icon.fail { color: var(--red); }
.check-icon.info { color: var(--text-muted); }
.check-name {
    flex: 1;
    font-size: 13px;
    font-weight: 500;
}
.check-value {
    font-size: 12px;
    color: var(--text-muted);
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-weight: 500;
}
.check-badge {
    font-size: 10px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    margin-left: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.check-badge.good { background: var(--green-dim); color: var(--green); }
.check-badge.ok { background: var(--yellow-dim); color: var(--yellow); }
.check-badge.poor { background: var(--red-dim); color: var(--red); }

/* ── Recommendations ── */
.rec-list { display: flex; flex-direction: column; gap: 12px; }
.rec-item {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    font-size: 14px;
    line-height: 1.7;
    position: relative;
    padding-left: 56px;
}
.rec-number {
    position: absolute;
    left: 20px;
    top: 20px;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: var(--brand-glow);
    border: 1px solid var(--brand-border);
    color: var(--brand-light);
    font-weight: 800;
    font-size: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* ── Alerts ── */
.alert {
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 12px;
    font-size: 13px;
    line-height: 1.6;
    font-weight: 500;
    border: 1px solid;
}
.alert.blocker {
    background: var(--red-dim);
    border-color: rgba(248, 113, 113, 0.25);
    color: #FCA5A5;
}
.alert.warning {
    background: var(--yellow-dim);
    border-color: rgba(251, 191, 36, 0.25);
    color: #FDE68A;
}

/* ── Methodology ── */
.method-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}
.method-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
}
.method-card .method-name {
    font-size: 14px;
    font-weight: 700;
    color: var(--brand-light);
    margin-bottom: 8px;
}
.method-card .method-desc {
    font-size: 12px;
    color: var(--text-muted);
    line-height: 1.6;
}

/* ── Footer ── */
.footer {
    padding: 24px 48px;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 11px;
    text-align: center;
    font-weight: 500;
    letter-spacing: 0.3px;
}
.footer strong {
    background: linear-gradient(135deg, var(--brand), var(--brand-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 40px 0;
}

@media print {
    body { background: #fff; color: #1a1a2e; padding: 0; }
    .header { background: #f8fafc; }
    .header::before { background: var(--brand); }
    .brand {
        -webkit-text-fill-color: var(--brand);
        color: var(--brand);
    }
    .metric-card, table, .trust-meter, .checks, .rec-item,
    .alert, .takeaway, .ci-container, .likelihood-card, .method-card {
        background: #f8fafc;
        border-color: #e2e8f0;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }
    .ci-bar-track { background: #e2e8f0; }
    .section { page-break-inside: avoid; }
    .footer strong {
        -webkit-text-fill-color: var(--brand);
        color: var(--brand);
    }
}
"""


def _render_html(report: TestReport) -> str:
    """Render the full HTML report."""
    inc = report.incrementality
    iroas = report.iroas

    lift_class = "positive" if inc.relative_lift > 0 and inc.is_significant else (
        "negative" if inc.relative_lift < 0 else "neutral"
    )

    parts = []

    # ── Header ────────────────────────────────────────────────────────
    channel_name = report.ad_channel.value.title()
    scope_name = report.measurement_scope.value.replace('_', ' ').title()
    test_type = report.test_scope.value.replace('_', ' ').title()

    parts.append(f"""
    <div class="header">
        <div class="brand">LIFT</div>
        <div class="subtitle">Geo Incrementality Platform</div>
        <div class="test-name">{_esc(report.test_name)}</div>
        <div class="meta-row">
            <span class="meta-tag">Channel<span class="tag-value">{channel_name}</span></span>
            <span class="meta-tag">Scope<span class="tag-value">{test_type}</span></span>
            <span class="meta-tag">Measurement<span class="tag-value">{scope_name}</span></span>
            <span class="meta-tag">Duration<span class="tag-value">{report.duration_weeks} weeks</span></span>
            <span class="meta-tag">Period<span class="tag-value">{report.test_start} &rarr; {report.test_end}</span></span>
        </div>
        <div class="meta-row" style="margin-top:8px">
            <span class="meta-tag">Test ID<span class="tag-value">{_esc(report.test_id)}</span></span>
            <span class="meta-tag">Treatment<span class="tag-value">{report.num_treatment_dmas} DMAs</span></span>
            <span class="meta-tag">Holdout<span class="tag-value">{report.num_holdout_dmas} DMAs</span></span>
        </div>
    </div>
    """)

    parts.append('<div class="content">')

    # ── Key Takeaway ──────────────────────────────────────────────────
    if inc.is_significant and iroas.iroas >= 1.0:
        takeaway = (
            f"<strong>{channel_name} ads are driving real, measurable growth.</strong> "
            f"The test detected a statistically significant "
            f"<span class='hl'>{_pct(inc.relative_lift)}</span> incremental lift "
            f"with an iROAS of <span class='hl'>{iroas.iroas:.2f}x</span> "
            f"&mdash; every $1 spent generated "
            f"<span class='hl'>${iroas.iroas:.2f}</span> in incremental revenue."
        )
    elif inc.is_significant and iroas.iroas < 1.0:
        takeaway = (
            f"<strong>{channel_name} ads show a measurable effect, but returns are below breakeven.</strong> "
            f"The test detected a significant "
            f"<span class='hl'>{_pct(inc.relative_lift)}</span> lift, "
            f"but iROAS of <span style='color:var(--yellow)'>{iroas.iroas:.2f}x</span> "
            f"means ad spend exceeds incremental revenue. Consider optimizing creative and targeting."
        )
    else:
        takeaway = (
            f"<strong>{channel_name} ads did not show a statistically significant effect.</strong> "
            f"Measured lift of {_pct(inc.relative_lift)} (p={inc.p_value:.3f}) "
            f"could be due to chance. Consider running a longer test or increasing holdout size."
        )

    parts.append(f"""
    <div class="takeaway">
        <div class="takeaway-label">Key Takeaway</div>
        <p>{takeaway}</p>
    </div>
    """)

    # ── Executive Summary Cards ───────────────────────────────────────
    sig_text = "Significant" if inc.is_significant else "Not Significant"
    sig_class = "green" if inc.is_significant else "yellow"
    iroas_class = "green" if iroas.iroas >= 1.0 else ("yellow" if iroas.iroas >= 0 else "red")

    parts.append(f"""
    <div class="section">
        <div class="section-title">Executive Summary</div>
        <div class="metric-grid">
            <div class="metric-card green">
                <div class="metric-label">Incremental Lift</div>
                <div class="metric-value {lift_class}">{_pct(inc.relative_lift)}</div>
                <div class="metric-sub">95% CI: [{_pct(inc.lift_lower_ci)}, {_pct(inc.lift_upper_ci)}]</div>
            </div>
            <div class="metric-card {iroas_class}">
                <div class="metric-label">Incremental ROAS</div>
                <div class="metric-value">{iroas.iroas:.2f}x</div>
                <div class="metric-sub">{_iroas_label(iroas.iroas)} &middot; CI: [{iroas.iroas_lower_ci:.2f}x, {iroas.iroas_upper_ci:.2f}x]</div>
            </div>
            <div class="metric-card {sig_class}">
                <div class="metric-label">Statistical Significance</div>
                <div class="metric-value sm">{sig_text}</div>
                <div class="metric-sub">p = {inc.p_value:.4f} &middot; Cohen&rsquo;s d = {inc.cohen_d:.3f}</div>
            </div>
            <div class="metric-card brand">
                <div class="metric-label">Incremental Revenue</div>
                <div class="metric-value sm">{_money(iroas.incremental_revenue)}</div>
                <div class="metric-sub">on {_money(iroas.total_ad_spend)} total ad spend</div>
            </div>
        </div>
    """)

    # ── IF and CPIA cards (if available) ──────────────────────────────
    if iroas.incrementality_factor > 0 or iroas.cpia > 0:
        if_class = "green" if 0.5 <= iroas.incrementality_factor <= 1.5 else "yellow"
        if_label = (
            "Highly Incremental" if iroas.incrementality_factor >= 1.0
            else "Partially Incremental" if iroas.incrementality_factor >= 0.5
            else "Low Incrementality"
        )
        cpia_display = f"${iroas.cpia:,.0f}" if iroas.cpia > 0 else "N/A"

        parts.append(f"""
        <div class="metric-grid three-col" style="margin-top:0">
            <div class="metric-card {if_class}">
                <div class="metric-label">Incrementality Factor</div>
                <div class="metric-value">{iroas.incrementality_factor:.2f}</div>
                <div class="metric-sub">{if_label} &middot; {iroas.incremental_conversions:,.0f} incr. / {iroas.attributed_conversions:,.0f} attr.</div>
            </div>
            <div class="metric-card brand">
                <div class="metric-label">Cost Per Incr. Acquisition</div>
                <div class="metric-value sm">{cpia_display}</div>
                <div class="metric-sub">{_money(iroas.total_ad_spend)} / {iroas.incremental_conversions:,.0f} conversions</div>
            </div>
            <div class="metric-card brand">
                <div class="metric-label">Incremental Orders</div>
                <div class="metric-value sm">{iroas.incremental_conversions:,.0f}</div>
                <div class="metric-sub">Platform reported: {iroas.attributed_conversions:,.0f}</div>
            </div>
        </div>
        """)

    # ── Confidence Interval Visualization ─────────────────────────────
    ci_low = inc.lift_lower_ci
    ci_high = inc.lift_upper_ci
    point = inc.relative_lift

    vis_min = min(ci_low, -0.05)
    vis_max = max(ci_high, 0.25)
    vis_range = vis_max - vis_min
    if vis_range == 0:
        vis_range = 0.30

    def to_pct(v: float) -> float:
        return max(0, min(100, (v - vis_min) / vis_range * 100))

    zero_pos = to_pct(0)
    ci_left = to_pct(ci_low)
    ci_right = to_pct(ci_high)
    point_pos = to_pct(point)

    ci_color = "positive" if ci_low > 0 else ("negative" if ci_high < 0 else "mixed")

    parts.append(f"""
        <div class="ci-container">
            <div class="ci-label">Lift 95% Confidence Interval</div>
            <div class="ci-bar-track">
                <div class="ci-bar-zero" style="left:{zero_pos:.1f}%"></div>
                <div class="ci-bar-range {ci_color}" style="left:{ci_left:.1f}%;width:{ci_right - ci_left:.1f}%"></div>
                <div class="ci-bar-point" style="left:{point_pos:.1f}%"></div>
            </div>
            <div class="ci-ticks">
                <span>{_pct(vis_min)}</span>
                <span>0%</span>
                <span>{_pct(vis_max)}</span>
            </div>
        </div>
    """)

    # ── Lift Likelihood + Effect Size ─────────────────────────────────
    if inc.lift_likelihood > 0:
        ll = inc.lift_likelihood
        circumference = 2 * 3.14159 * 28  # r=28 for 68px svg
        fill_len = ll * circumference
        gap_len = circumference - fill_len
        ring_cls = "green" if ll >= 0.90 else "yellow"

        parts.append(f"""
        <div class="likelihood-row" style="margin-top:16px">
            <div class="likelihood-card">
                <div class="ring-container">
                    <svg viewBox="0 0 68 68">
                        <circle class="ring-bg" cx="34" cy="34" r="28"/>
                        <circle class="ring-fill {ring_cls}" cx="34" cy="34" r="28"
                            stroke-dasharray="{fill_len:.1f} {gap_len:.1f}"/>
                    </svg>
                    <div class="ring-label">{ll:.0%}</div>
                </div>
                <div class="likelihood-text">
                    <div class="ll-title">Probability of True Lift &gt; 0</div>
                    <div class="ll-value">{ll:.1%}</div>
                    <div class="ll-detail">Bayesian posterior probability</div>
                </div>
            </div>
            <div class="likelihood-card">
                <div class="ring-container">
                    <svg viewBox="0 0 68 68">
                        <circle class="ring-bg" cx="34" cy="34" r="28"/>
                        <circle class="ring-fill {'green' if inc.cohen_d >= 0.3 else 'yellow'}" cx="34" cy="34" r="28"
                            stroke-dasharray="{min(inc.cohen_d / 0.8, 1.0) * circumference:.1f} {circumference - min(inc.cohen_d / 0.8, 1.0) * circumference:.1f}"/>
                    </svg>
                    <div class="ring-label">{inc.cohen_d:.2f}</div>
                </div>
                <div class="likelihood-text">
                    <div class="ll-title">Effect Size (Cohen&rsquo;s d)</div>
                    <div class="ll-value">{'Large' if inc.cohen_d >= 0.8 else ('Medium' if inc.cohen_d >= 0.5 else ('Small-Medium' if inc.cohen_d >= 0.3 else 'Small'))}</div>
                    <div class="ll-detail">{'Strong practical significance' if inc.cohen_d >= 0.5 else 'Detectable practical significance'}</div>
                </div>
            </div>
        </div>
        """)

    parts.append("</div>")  # close section

    # ── Revenue Breakdown ─────────────────────────────────────────────
    parts.append(f"""
    <div class="section">
        <div class="section-title">Revenue Breakdown</div>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th style="text-align:right">Treatment ({report.num_treatment_dmas} DMAs)</th>
                    <th style="text-align:right">Holdout ({report.num_holdout_dmas} DMAs)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Total Revenue</td>
                    <td style="text-align:right;font-weight:700">{_money(report.treatment_total_revenue)}</td>
                    <td style="text-align:right;font-weight:700">{_money(report.holdout_total_revenue)}</td>
                </tr>
                <tr>
                    <td>Avg Daily / DMA</td>
                    <td style="text-align:right">{_money(report.treatment_avg_daily_revenue)}</td>
                    <td style="text-align:right">{_money(report.holdout_avg_daily_revenue)}</td>
                </tr>
                <tr>
                    <td>Organic Baseline (estimated)</td>
                    <td style="text-align:right;color:var(--text-muted)">{_money(report.organic_baseline_revenue)}</td>
                    <td style="text-align:right">&mdash;</td>
                </tr>
            </tbody>
        </table>
    </div>
    """)

    # ── Cross-Platform iROAS ──────────────────────────────────────────
    if iroas.shopify_incremental_revenue > 0 or iroas.amazon_incremental_revenue > 0:
        total_inc = iroas.incremental_revenue or 1
        shopify_share = iroas.shopify_incremental_revenue / total_inc * 100
        amazon_share = iroas.amazon_incremental_revenue / total_inc * 100

        cross_rows = ""
        if iroas.shopify_incremental_revenue > 0:
            cross_rows += f"""
                <tr>
                    <td>Shopify</td>
                    <td style="text-align:right;font-weight:700">{_money(iroas.shopify_incremental_revenue)}</td>
                    <td style="text-align:right">{iroas.shopify_iroas:.2f}x</td>
                    <td style="text-align:right">{shopify_share:.0f}%</td>
                </tr>"""
        if iroas.amazon_incremental_revenue > 0:
            cross_rows += f"""
                <tr>
                    <td>Amazon</td>
                    <td style="text-align:right;font-weight:700">{_money(iroas.amazon_incremental_revenue)}</td>
                    <td style="text-align:right">{iroas.amazon_iroas:.2f}x</td>
                    <td style="text-align:right">{amazon_share:.0f}%</td>
                </tr>"""
        cross_rows += f"""
                <tr style="border-top:2px solid var(--border)">
                    <td><strong>Combined</strong></td>
                    <td style="text-align:right;font-weight:800">{_money(iroas.incremental_revenue)}</td>
                    <td style="text-align:right;font-weight:800">{iroas.iroas:.2f}x</td>
                    <td style="text-align:right;font-weight:800">100%</td>
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
                        <th style="text-align:right">Share</th>
                    </tr>
                </thead>
                <tbody>{cross_rows}</tbody>
            </table>
        </div>
        """)

    # ── Platform-Level Incrementality ─────────────────────────────────
    if report.shopify_incrementality or report.amazon_incrementality:
        plat_rows = ""
        for name, result in [("Shopify", report.shopify_incrementality), ("Amazon", report.amazon_incrementality)]:
            if result:
                sig_icon = "&#10003;" if result.is_significant else "&#10007;"
                sig_cls = "pass" if result.is_significant else "fail"
                lift_color = "color:var(--green)" if result.relative_lift > 0 and result.is_significant else ""
                plat_rows += f"""
                <tr>
                    <td>{name}</td>
                    <td style="text-align:right;font-weight:700;{lift_color}">{_pct(result.relative_lift)}</td>
                    <td style="text-align:right">[{_pct(result.lift_lower_ci)}, {_pct(result.lift_upper_ci)}]</td>
                    <td style="text-align:center"><span class="check-icon {sig_cls}">{sig_icon}</span></td>
                    <td style="text-align:right">{result.p_value:.4f}</td>
                </tr>"""

        parts.append(f"""
        <div class="section">
            <div class="section-title">Platform-Level Incrementality</div>
            <table>
                <thead>
                    <tr>
                        <th>Platform</th>
                        <th style="text-align:right">Lift</th>
                        <th style="text-align:right">95% CI</th>
                        <th style="text-align:center">Sig.</th>
                        <th style="text-align:right">p-value</th>
                    </tr>
                </thead>
                <tbody>{plat_rows}</tbody>
            </table>
        </div>
        """)

    # ── Model Ensemble ────────────────────────────────────────────────
    if report.estimator_results and len(report.estimator_results) > 1:
        model_rows = ""
        for name, result in report.estimator_results.items():
            weight = report.estimator_weights.get(name, 0)
            sig_icon = "&#10003;" if result.is_significant else "&#10007;"
            sig_cls = "pass" if result.is_significant else "fail"
            l2 = f"{result.l2_imbalance:.4f}" if result.l2_imbalance > 0 else "&mdash;"
            r2 = f"{result.pre_period_r_squared:.3f}" if result.pre_period_r_squared > 0 else "&mdash;"
            bar_w = weight * 100

            model_rows += f"""
                <tr>
                    <td><strong>{_esc(name.upper())}</strong></td>
                    <td style="text-align:right">
                        <span class="weight-bar-bg"><span class="weight-bar-fill" style="width:{bar_w:.0f}%"></span></span>
                        {weight:.0%}
                    </td>
                    <td style="text-align:right;font-weight:700">{_pct(result.relative_lift)}</td>
                    <td style="text-align:right">{result.p_value:.4f}</td>
                    <td style="text-align:center"><span class="check-icon {sig_cls}">{sig_icon}</span></td>
                    <td style="text-align:right;color:var(--text-muted)">{l2}</td>
                    <td style="text-align:right;color:var(--text-muted)">{r2}</td>
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
            <div style="color: var(--text-muted); font-size: 12px; padding: 12px 0 0 0; font-weight: 500;">
                Final estimate: {_esc(inc.method.replace('_', ' ').title())}
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

        l2_badge = "good" if v.l2_imbalance < 0.05 else ("ok" if v.l2_imbalance < 0.10 else "poor")
        l2_label = "Good" if v.l2_imbalance < 0.05 else ("OK" if v.l2_imbalance < 0.10 else "Poor")

        r2_badge = "good" if v.pre_period_r_squared > 0.90 else ("ok" if v.pre_period_r_squared > 0.80 else "poor")
        r2_label = "Good" if v.pre_period_r_squared > 0.90 else ("OK" if v.pre_period_r_squared > 0.80 else "Poor")

        fpr_pass = v.false_positive_rate <= 0.15
        fpr_icon, fpr_cls = ("&#10003;", "pass") if fpr_pass else ("&#10007;", "fail")
        fpr_badge = "good" if v.false_positive_rate <= 0.10 else ("ok" if v.false_positive_rate <= 0.15 else "poor")
        fpr_label = "Calibrated" if v.false_positive_rate <= 0.10 else ("OK" if v.false_positive_rate <= 0.15 else "High")

        agree_pass = v.estimator_agreement >= 0.50
        agree_icon, agree_cls = ("&#10003;", "pass") if agree_pass else ("&#10007;", "fail")
        agree_badge = "good" if v.estimator_agreement >= 0.70 else ("ok" if v.estimator_agreement >= 0.50 else "poor")
        agree_label = "Strong" if v.estimator_agreement >= 0.70 else ("Moderate" if v.estimator_agreement >= 0.50 else "Weak")

        alerts_html = ""
        if v.blockers:
            alerts_html += "".join(
                f'<div class="alert blocker">&#10007; <strong>Blocker:</strong> {_esc(b)}</div>'
                for b in v.blockers
            )
        if v.warnings:
            alerts_html += "".join(
                f'<div class="alert warning">&#9888; {_esc(w)}</div>'
                for w in v.warnings
            )

        parts.append(f"""
        <div class="section">
            <div class="section-title">Validation &amp; Trust Score</div>
            <div class="trust-meter">
                <div class="trust-header">
                    <span class="trust-score-num">{v.trust_score:.0f}<span>/100</span></span>
                    <span class="trust-verdict {tc}">{tl}</span>
                </div>
                <div class="trust-bar-bg">
                    <div class="trust-bar-fill {tc}" style="width:{v.trust_score:.0f}%"></div>
                </div>
            </div>

            <div class="checks">
                <div class="check-item">
                    <span class="check-icon {aa_cls}">{aa_icon}</span>
                    <span class="check-name">AA Test (pre-period balance)</span>
                    <span class="check-value">p = {v.aa_test_p_value:.4f}</span>
                </div>
                <div class="check-item">
                    <span class="check-icon info">&#9679;</span>
                    <span class="check-name">Pre-period L2 Imbalance</span>
                    <span class="check-value">{v.l2_imbalance:.4f}</span>
                    <span class="check-badge {l2_badge}">{l2_label}</span>
                </div>
                <div class="check-item">
                    <span class="check-icon info">&#9679;</span>
                    <span class="check-name">Pre-period R&sup2;</span>
                    <span class="check-value">{v.pre_period_r_squared:.3f}</span>
                    <span class="check-badge {r2_badge}">{r2_label}</span>
                </div>
                <div class="check-item">
                    <span class="check-icon {fpr_cls}">{fpr_icon}</span>
                    <span class="check-name">Placebo False Positive Rate ({v.num_placebo_tests} tests)</span>
                    <span class="check-value">{v.false_positive_rate:.0%}</span>
                    <span class="check-badge {fpr_badge}">{fpr_label}</span>
                </div>
                <div class="check-item">
                    <span class="check-icon {agree_cls}">{agree_icon}</span>
                    <span class="check-name">Estimator Agreement</span>
                    <span class="check-value">{v.estimator_agreement:.0%}</span>
                    <span class="check-badge {agree_badge}">{agree_label}</span>
                </div>
            </div>
            {alerts_html}
        </div>
        """)

    # ── Methodology ───────────────────────────────────────────────────
    parts.append("""
    <div class="section">
        <div class="section-title">Methodology</div>
        <div class="method-grid">
            <div class="method-card">
                <div class="method-name">ASCM</div>
                <div class="method-desc">
                    Augmented Synthetic Control Method.
                    Builds a weighted combination of holdout DMAs to create a
                    synthetic treatment counterfactual, with ridge-augmented bias correction.
                </div>
            </div>
            <div class="method-card">
                <div class="method-name">BSTS</div>
                <div class="method-desc">
                    Bayesian Structural Time Series.
                    Models the counterfactual using a state-space time series
                    with spike-and-slab variable selection on control DMAs.
                </div>
            </div>
            <div class="method-card">
                <div class="method-name">DiD</div>
                <div class="method-desc">
                    Difference-in-Differences.
                    Compares pre-to-post changes in treatment vs. holdout groups
                    to estimate the causal effect, controlling for time trends.
                </div>
            </div>
        </div>
        <div style="color: var(--text-muted); font-size: 12px; margin-top: 16px; line-height: 1.7; font-weight: 500;">
            The final estimate is a weighted ensemble of all three methods. Weights are
            assigned based on pre-period fit quality (L2 imbalance, R&sup2;) and cross-validation
            performance. Results are validated with placebo tests, AA balance checks, and
            estimator agreement analysis.
        </div>
    </div>
    """)

    # ── Recommendations ───────────────────────────────────────────────
    if report.recommendations:
        rec_html = ""
        for i, rec in enumerate(report.recommendations, 1):
            rec_html += f"""
            <div class="rec-item">
                <div class="rec-number">{i}</div>
                {_esc(rec)}
            </div>"""

        parts.append(f"""
        <div class="section">
            <div class="section-title">Recommendations</div>
            <div class="rec-list">{rec_html}</div>
        </div>
        """)

    parts.append("</div>")  # close .content

    # ── Footer ────────────────────────────────────────────────────────
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    parts.append(f"""
    <div class="footer">
        Generated by <strong>LIFT</strong> &mdash; Geo Incrementality Platform &middot; {now}
    </div>
    """)

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
