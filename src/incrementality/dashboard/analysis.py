"""LIFT Dashboard — Analysis Report page.

Executive-grade report card with platform iROAS comparison,
revenue waterfall, cross-platform lift, inline spend curve,
and full model validation — organised into tabs for quick
consumption by both CMOs and data scientists.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from incrementality.dashboard.data_loader import scan_reports
from incrementality.dashboard.theme import (
    C, CHART_PALETTE, empty_state, metric_card, score_bar,
    section_header, status_badge,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' to 'R, G, B' for rgba()."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}"


def _fmt_pct(v: float) -> str:
    return f"{v:+.1%}"


def _fmt_dollar(v: float) -> str:
    return f"${v:,.0f}"


# ═════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═════════════════════════════════════════════════════════════════════════════

def render(output_dir: str):
    st.title("Analysis Report")

    reports = scan_reports(output_dir)

    if not reports:
        empty_state(
            "📊", "No reports available",
            "Analyze a completed test: incrementality analyze --test-id <ID>",
        )
        return

    # ── Report Selector ───────────────────────────────────────────────────
    options = {f"{r.test_name}  ({r.test_id})": r for r in reports}
    selected_key = st.selectbox("Select a report", list(options.keys()))
    report = options[selected_key]

    # ── Key Takeaway Card (hero banner) ────────────────────────────────────
    _render_key_takeaway(report)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_summary, tab_platform, tab_models, tab_spend = st.tabs([
        "Executive Summary",
        "Platform Breakdown",
        "Models & Validation",
        "Spend Optimization",
    ])

    with tab_summary:
        _render_executive_summary(report)

    with tab_platform:
        _render_platform_tab(report)

    with tab_models:
        _render_models_tab(report)

    with tab_spend:
        _render_spend_tab(report)

    # ── Recommendations (always visible) ──────────────────────────────────
    if report.recommendations:
        _render_recommendations(report)


# ═════════════════════════════════════════════════════════════════════════════
#  Key Takeaway — the hero card
# ═════════════════════════════════════════════════════════════════════════════

def _render_key_takeaway(report):
    inc = report.incrementality
    iroas = report.iroas

    # Determine headline
    if inc.is_significant and iroas.iroas >= 1.0:
        headline = "Ad spend is driving profitable incremental growth"
        icon = "✓"
        border_color = C["ok"]
        headline_color = C["ok"]
        detail_parts = [
            f"Every $1 spent returns **${iroas.iroas:.2f}** in incremental revenue",
            f"with a **{inc.relative_lift:+.1%}** lift over organic baseline",
        ]
        if iroas.shopify_iroas > 0 and iroas.amazon_iroas > 0:
            detail_parts.append(
                f"Shopify iROAS **{iroas.shopify_iroas:.2f}x** · "
                f"Amazon halo **{iroas.amazon_iroas:.2f}x**"
            )
    elif inc.is_significant and iroas.iroas < 1.0:
        headline = "Ads drive lift, but spend exceeds incremental returns"
        icon = "⚠"
        border_color = C["warn"]
        headline_color = C["warn"]
        detail_parts = [
            f"Lift of **{inc.relative_lift:+.1%}** is statistically significant",
            f"but iROAS of **{iroas.iroas:.2f}x** is below breakeven — "
            f"consider reducing spend or optimising creative",
        ]
    else:
        headline = "Results are not yet statistically significant"
        icon = "—"
        border_color = C["text_muted"]
        headline_color = C["text_muted"]
        detail_parts = [
            f"Observed lift of **{inc.relative_lift:+.1%}** (p={inc.p_value:.3f}) "
            f"does not meet the significance threshold",
        ]
        if inc.lift_likelihood > 0:
            detail_parts.append(
                f"Bayesian probability of positive lift: **{inc.lift_likelihood:.0%}**"
            )

    detail_html = "<br>".join(detail_parts)

    st.markdown(f"""
    <div class="lift-card" style="border-color:{border_color}; border-width:2px;
                padding:1.5rem 2rem;">
        <div style="display:flex; align-items:flex-start; gap:1.25rem;">
            <div style="font-size:2.2rem; color:{border_color}; line-height:1;">{icon}</div>
            <div style="flex:1;">
                <div style="color:{headline_color}; font-size:1.35rem; font-weight:700;
                            margin-bottom:0.5rem; line-height:1.3;">
                    {headline}</div>
                <div style="color:{C['text_secondary']}; font-size:0.95rem; line-height:1.7;">
                    {detail_html}</div>
            </div>
            <div style="text-align:right; min-width:120px;">
                <div style="color:{C['text_muted']}; font-size:0.7rem;
                            text-transform:uppercase; letter-spacing:0.06em;">
                    Overall iROAS</div>
                <div style="color:{C['text']}; font-size:2rem; font-weight:800;
                            line-height:1.2;">{iroas.iroas:.2f}x</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Executive Summary
# ═════════════════════════════════════════════════════════════════════════════

def _render_executive_summary(report):
    inc = report.incrementality
    iroas = report.iroas

    # ── Hero metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    lift_color = C["ok"] if inc.is_significant and inc.relative_lift > 0 else (
        C["bad"] if inc.relative_lift < 0 else C["text_muted"]
    )
    with c1:
        st.markdown(f"""
        <div class="lift-card-accent">
            <h4>Incremental Lift</h4>
            <div class="value" style="color:{lift_color};">{inc.relative_lift:+.1%}</div>
            <div class="sub">[{inc.lift_lower_ci:+.1%}, {inc.lift_upper_ci:+.1%}]</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        iroas_color = C["ok"] if iroas.iroas >= 1.0 else C["warn"] if iroas.iroas >= 0 else C["bad"]
        st.markdown(f"""
        <div class="lift-card-accent">
            <h4>Incremental ROAS</h4>
            <div class="value" style="color:{iroas_color};">{iroas.iroas:.2f}x</div>
            <div class="sub">CI: [{iroas.iroas_lower_ci:.2f}x, {iroas.iroas_upper_ci:.2f}x]</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        sig_color = C["ok"] if inc.is_significant else C["bad"]
        sig_label = "Significant" if inc.is_significant else "Not significant"
        st.markdown(f"""
        <div class="lift-card">
            <h4>Statistical Significance</h4>
            <div class="value" style="color:{sig_color}; font-size:1.4rem;">
                p = {inc.p_value:.4f}
            </div>
            <div class="sub">{status_badge(sig_label)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="lift-card">
            <h4>Incremental Revenue</h4>
            <div class="value">{_fmt_dollar(iroas.incremental_revenue)}</div>
            <div class="sub">on {_fmt_dollar(iroas.total_ad_spend)} spend</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Context row ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Test Period", f"{report.duration_weeks}w",
                     f"{report.test_start} → {report.test_end}")
    with c2:
        metric_card("Channel", report.ad_channel.value.title(),
                     report.test_scope.value.title())
    with c3:
        metric_card("DMAs", f"{report.num_treatment_dmas}T / {report.num_holdout_dmas}H",
                     report.measurement_scope.value.replace("_", " ").title())
    with c4:
        if inc.lift_likelihood > 0:
            metric_card("P(lift > 0)", f"{inc.lift_likelihood:.1%}",
                         f"Cohen's d = {inc.cohen_d:.3f}")

    # ── Lift forest plot ──────────────────────────────────────────────────
    section_header("Incremental Lift")
    _render_lift_chart(report)

    # ── iROAS gauge ───────────────────────────────────────────────────────
    section_header("iROAS Gauge")
    _render_iroas_gauge(report)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Platform Breakdown
# ═════════════════════════════════════════════════════════════════════════════

def _render_platform_tab(report):
    iroas = report.iroas
    has_shopify = iroas.shopify_incremental_revenue > 0 or iroas.shopify_iroas > 0
    has_amazon = iroas.amazon_incremental_revenue > 0 or iroas.amazon_iroas > 0

    if not has_shopify and not has_amazon:
        empty_state(
            "📦", "Single-platform test",
            "This test measured a single platform — no cross-platform breakdown available.",
        )
        return

    # ── 1) Platform iROAS Comparison Bar Chart ────────────────────────────
    section_header("Platform iROAS Comparison")
    _render_platform_iroas_bars(report)

    # ── 2) Revenue Waterfall ──────────────────────────────────────────────
    section_header("Revenue Attribution Waterfall")
    _render_revenue_waterfall(report)

    # ── 3) Cross-platform Lift Comparison ─────────────────────────────────
    section_header("Cross-Platform Lift")
    _render_cross_platform_lift(report)

    # ── 4) Platform metric cards ──────────────────────────────────────────
    section_header("Platform Detail Cards")
    _render_platform_detail_cards(report)


def _render_platform_iroas_bars(report):
    """Bar chart: Overall vs Shopify vs Amazon iROAS with breakeven reference."""
    iroas = report.iroas

    labels = []
    values = []
    colors = []
    cis_lower = []
    cis_upper = []

    # Overall
    labels.append("Overall")
    values.append(iroas.iroas)
    colors.append(C["primary"])
    cis_lower.append(iroas.iroas_lower_ci)
    cis_upper.append(iroas.iroas_upper_ci)

    # Shopify
    if iroas.shopify_iroas > 0 or iroas.shopify_incremental_revenue > 0:
        labels.append("Shopify")
        values.append(iroas.shopify_iroas)
        colors.append(C["ok"])
        # Approximate CI proportionally
        if iroas.iroas > 0:
            ratio = iroas.shopify_iroas / iroas.iroas
            cis_lower.append(iroas.iroas_lower_ci * ratio)
            cis_upper.append(iroas.iroas_upper_ci * ratio)
        else:
            cis_lower.append(0)
            cis_upper.append(0)

    # Amazon
    if iroas.amazon_iroas > 0 or iroas.amazon_incremental_revenue > 0:
        labels.append("Amazon")
        values.append(iroas.amazon_iroas)
        colors.append(C["purple"])
        if iroas.iroas > 0:
            ratio = iroas.amazon_iroas / iroas.iroas
            cis_lower.append(iroas.iroas_lower_ci * ratio)
            cis_upper.append(iroas.iroas_upper_ci * ratio)
        else:
            cis_lower.append(0)
            cis_upper.append(0)

    fig = go.Figure()

    # Bars
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.2f}x" for v in values],
        textposition="outside",
        textfont=dict(color=C["text"], size=14, family="Inter, sans-serif"),
        error_y=dict(
            type="data",
            symmetric=False,
            array=[u - v for v, u in zip(values, cis_upper)],
            arrayminus=[v - lo for v, lo in zip(values, cis_lower)],
            color=C["text_muted"],
            thickness=1.5,
            width=6,
        ),
        hovertemplate="%{x}: %{y:.2f}x iROAS<extra></extra>",
    ))

    # Breakeven line
    fig.add_hline(
        y=1.0, line_dash="dash", line_color=C["bad"], line_width=1.5,
        annotation_text="Breakeven (1.0x)",
        annotation_font_color=C["bad"],
        annotation_position="top right",
    )

    fig.update_layout(
        height=380,
        yaxis_title="Incremental ROAS",
        yaxis=dict(rangemode="tozero"),
        xaxis_title="",
        showlegend=False,
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Caption
    st.caption(
        "**Overall iROAS** is total incremental revenue (all platforms) ÷ ad spend. "
        "**Platform-specific iROAS** isolates lift to Shopify or Amazon revenue only. "
        "Amazon 'halo' iROAS captures cross-platform effects where Facebook ads drive Amazon purchases."
    )


def _render_revenue_waterfall(report):
    """Waterfall: Organic Baseline → +Shopify Incremental → +Amazon Incremental → Total."""
    iroas = report.iroas

    organic = report.organic_baseline_revenue
    shopify_inc = iroas.shopify_incremental_revenue
    amazon_inc = iroas.amazon_incremental_revenue

    # If platform breakdown isn't available, show simple two-step
    if shopify_inc == 0 and amazon_inc == 0:
        shopify_inc = iroas.incremental_revenue
        labels = ["Organic Baseline", "Incremental Revenue", "Total Revenue"]
        measures = ["absolute", "relative", "total"]
        values = [organic, shopify_inc, organic + shopify_inc]
        bar_colors = [C["text_muted"], C["primary"], C["ok"]]
    else:
        labels = ["Organic Baseline", "Shopify Incremental", "Amazon Incremental",
                  "Total Revenue"]
        measures = ["absolute", "relative", "relative", "total"]
        values = [organic, shopify_inc, amazon_inc,
                 organic + shopify_inc + amazon_inc]
        bar_colors = [C["text_muted"], C["ok"], C["purple"], C["primary"]]

    fig = go.Figure(go.Waterfall(
        x=labels,
        y=values,
        measure=measures,
        connector=dict(line=dict(color=C["border"], width=1.5)),
        decreasing=dict(marker=dict(color=C["bad"])),
        increasing=dict(marker=dict(color=C["ok"])),
        totals=dict(marker=dict(color=C["primary"],
                                line=dict(color=C["primary_dark"], width=1))),
        text=[_fmt_dollar(v) for v in values],
        textposition="outside",
        textfont=dict(color=C["text"], size=12),
        hovertemplate="%{x}: %{y:$,.0f}<extra></extra>",
    ))

    fig.update_layout(
        height=420,
        yaxis_title="Revenue ($)",
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        xaxis_title="",
        showlegend=False,
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Percentage breakdown
    total = organic + shopify_inc + amazon_inc
    if total > 0 and (shopify_inc > 0 or amazon_inc > 0):
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Organic Baseline", _fmt_dollar(organic),
                         f"{organic / total:.0%} of total")
        with c2:
            metric_card("Shopify Incremental", _fmt_dollar(shopify_inc),
                         f"{shopify_inc / total:.0%} of total" if shopify_inc > 0 else "—")
        with c3:
            metric_card("Amazon Incremental", _fmt_dollar(amazon_inc),
                         f"{amazon_inc / total:.0%} of total" if amazon_inc > 0 else "—")


def _render_cross_platform_lift(report):
    """Side-by-side lift comparison: Shopify vs Amazon with significance badges."""
    shopify_inc = report.shopify_incrementality
    amazon_inc = report.amazon_incrementality

    if not shopify_inc and not amazon_inc:
        st.info("No cross-platform lift breakdown available for this test.")
        return

    # Build data for forest-style chart
    methods = []
    lifts = []
    lowers = []
    uppers = []
    sig_flags = []
    bar_colors = []

    # Ensemble (overall)
    overall = report.incrementality
    methods.append("Overall (Ensemble)")
    lifts.append(overall.relative_lift)
    lowers.append(overall.lift_lower_ci)
    uppers.append(overall.lift_upper_ci)
    sig_flags.append(overall.is_significant)
    bar_colors.append(C["primary"])

    if shopify_inc:
        methods.append("Shopify")
        lifts.append(shopify_inc.relative_lift)
        lowers.append(shopify_inc.lift_lower_ci)
        uppers.append(shopify_inc.lift_upper_ci)
        sig_flags.append(shopify_inc.is_significant)
        bar_colors.append(C["ok"])

    if amazon_inc:
        methods.append("Amazon")
        lifts.append(amazon_inc.relative_lift)
        lowers.append(amazon_inc.lift_lower_ci)
        uppers.append(amazon_inc.lift_upper_ci)
        sig_flags.append(amazon_inc.is_significant)
        bar_colors.append(C["purple"])

    fig = go.Figure()

    for i, method in enumerate(methods):
        color = bar_colors[i]
        ci_alpha = "0.15" if sig_flags[i] else "0.08"

        # CI bar
        fig.add_trace(go.Bar(
            x=[uppers[i] - lowers[i]],
            y=[method],
            base=[lowers[i]],
            orientation="h",
            marker=dict(
                color=f"rgba({_hex_to_rgb(color)}, {ci_alpha})",
                line=dict(color=color, width=1.5),
            ),
            showlegend=False,
            hovertemplate=(
                f"{method}: {lifts[i]:+.1%} "
                f"[{lowers[i]:+.1%}, {uppers[i]:+.1%}]<extra></extra>"
            ),
        ))

        # Point estimate
        fig.add_trace(go.Scatter(
            x=[lifts[i]], y=[method],
            mode="markers",
            marker=dict(color=color, size=14, symbol="diamond",
                        line=dict(color=C["text"], width=1)),
            showlegend=False,
            hovertemplate=f"{method}: {lifts[i]:+.1%}<extra></extra>",
        ))

    # Zero reference
    fig.add_vline(x=0, line_dash="dash", line_color=C["bad"], line_width=1)

    fig.update_layout(
        height=60 + len(methods) * 80,
        xaxis_title="Relative Lift",
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=130, r=20, t=20, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Significance badges row
    cols = st.columns(len(methods))
    for i, col in enumerate(cols):
        with col:
            p_val = [overall.p_value,
                     shopify_inc.p_value if shopify_inc else None,
                     amazon_inc.p_value if amazon_inc else None][i]
            sig = sig_flags[i]
            label = "Significant" if sig else "Not significant"
            p_str = f"p = {p_val:.4f}" if p_val is not None else ""
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="color:{C['text']}; font-weight:600;
                            margin-bottom:0.3rem;">{methods[i]}</div>
                {status_badge(label)}
                <div style="color:{C['text_muted']}; font-size:0.8rem;
                            margin-top:0.25rem;">{p_str}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_platform_detail_cards(report):
    """Metric cards for each platform's performance."""
    iroas = report.iroas
    shopify_inc = report.shopify_incrementality
    amazon_inc = report.amazon_incrementality

    c1, c2 = st.columns(2)

    # Shopify card
    with c1:
        shopify_color = C["ok"]
        s_lift = f"{shopify_inc.relative_lift:+.1%}" if shopify_inc else "—"
        s_sig = shopify_inc.is_significant if shopify_inc else False
        s_label = "Significant" if s_sig else "Not significant"

        st.markdown(f"""
        <div class="lift-card" style="border-left: 3px solid {shopify_color};">
            <h4 style="color:{shopify_color} !important;">Shopify</h4>
            <div style="display:flex; justify-content:space-between; align-items:baseline;
                        margin-bottom:0.75rem;">
                <div>
                    <div class="value">{iroas.shopify_iroas:.2f}x iROAS</div>
                    <div class="sub">{_fmt_dollar(iroas.shopify_incremental_revenue)} incremental</div>
                </div>
                <div style="text-align:right;">
                    <div style="color:{C['text']}; font-size:1.1rem;
                                font-weight:600;">{s_lift} lift</div>
                    <div style="margin-top:0.25rem;">{status_badge(s_label)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Amazon card
    with c2:
        amazon_color = C["purple"]
        a_lift = f"{amazon_inc.relative_lift:+.1%}" if amazon_inc else "—"
        a_sig = amazon_inc.is_significant if amazon_inc else False
        a_label = "Significant" if a_sig else "Not significant"

        st.markdown(f"""
        <div class="lift-card" style="border-left: 3px solid {amazon_color};">
            <h4 style="color:{amazon_color} !important;">Amazon (Halo Effect)</h4>
            <div style="display:flex; justify-content:space-between; align-items:baseline;
                        margin-bottom:0.75rem;">
                <div>
                    <div class="value">{iroas.amazon_iroas:.2f}x iROAS</div>
                    <div class="sub">{_fmt_dollar(iroas.amazon_incremental_revenue)} incremental</div>
                </div>
                <div style="text-align:right;">
                    <div style="color:{C['text']}; font-size:1.1rem;
                                font-weight:600;">{a_lift} lift</div>
                    <div style="margin-top:0.25rem;">{status_badge(a_label)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Attribution quality (IF/CPIA)
    if iroas.incrementality_factor > 0:
        section_header("Attribution Quality")
        c1, c2, c3 = st.columns(3)
        with c1:
            if_val = iroas.incrementality_factor
            if_color = C["ok"] if 0.5 <= if_val <= 1.5 else C["warn"]
            metric_card("Incrementality Factor", f"{if_val:.2f}",
                         "1.0 = perfect attribution")
        with c2:
            metric_card("Attributed (Platform)", f"{iroas.attributed_conversions:,.0f}",
                         "From Ads Manager")
        with c3:
            metric_card("Incremental (Measured)", f"{iroas.incremental_conversions:,.0f}",
                         "From experiment")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Models & Validation
# ═════════════════════════════════════════════════════════════════════════════

def _render_models_tab(report):
    # ── Ensemble comparison ──────────────────────────────────────────────
    if report.estimator_results and len(report.estimator_results) > 1:
        _render_ensemble(report)

    # ── iROAS deep dive ──────────────────────────────────────────────────
    section_header("iROAS Deep Dive")
    _render_iroas_detail(report)

    # ── Validation ────────────────────────────────────────────────────────
    if report.validation:
        _render_validation(report)

    # ── Data quality ──────────────────────────────────────────────────────
    if report.anomaly_warnings or report.anomaly_blockers:
        _render_anomalies(report)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 4 — Spend Optimization
# ═════════════════════════════════════════════════════════════════════════════

def _render_spend_tab(report):
    sr = report.spend_response

    if not sr:
        empty_state(
            "💰", "No spend response data",
            "Re-run the analysis with ad spend data to generate the response curve. "
            "The spend optimizer requires DMA-level spend variation.",
        )
        return

    # Summary cards
    _render_spend_summary(sr)

    # Response curve
    section_header("Response Curve")
    _render_inline_response_curve(sr)

    # Marginal ROAS
    section_header("Marginal ROAS — Diminishing Returns")
    _render_inline_marginal_curve(sr)

    # Recommendation
    _render_spend_recommendation(sr)


# ═════════════════════════════════════════════════════════════════════════════
#  Components
# ═════════════════════════════════════════════════════════════════════════════

# ── Lift chart ────────────────────────────────────────────────────────────

def _render_lift_chart(report):
    inc = report.incrementality

    fig = go.Figure()

    # CI bar
    fig.add_trace(go.Bar(
        x=[inc.lift_upper_ci - inc.lift_lower_ci],
        y=["Ensemble"],
        base=[inc.lift_lower_ci],
        orientation="h",
        marker=dict(color=C["primary_bg"], line=dict(color=C["primary"], width=1)),
        name="95% CI",
        hovertemplate=f"CI: [{inc.lift_lower_ci:+.1%}, {inc.lift_upper_ci:+.1%}]<extra></extra>",
    ))

    # Point estimate
    fig.add_trace(go.Scatter(
        x=[inc.relative_lift], y=["Ensemble"],
        mode="markers",
        marker=dict(color=C["primary"], size=16, symbol="diamond"),
        name=f"Lift: {inc.relative_lift:+.1%}",
        hovertemplate=f"Lift: {inc.relative_lift:+.1%}<extra></extra>",
    ))

    fig.add_vline(x=0, line_dash="dash", line_color=C["bad"], line_width=1)

    fig.update_layout(
        height=180,
        xaxis_title="Relative Lift",
        xaxis=dict(tickformat=".0%"),
        showlegend=True,
        legend=dict(orientation="h", y=-0.3),
        margin=dict(l=80, r=20, t=20, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    if inc.lift_likelihood > 0:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(f"**P(true lift > 0):** {inc.lift_likelihood:.1%}")
        with c2:
            st.markdown(score_bar(inc.lift_likelihood * 100, 100, 12), unsafe_allow_html=True)


# ── iROAS gauge ───────────────────────────────────────────────────────────

def _render_iroas_gauge(report):
    iroas = report.iroas

    c1, c2 = st.columns([2, 1])

    with c1:
        gauge_max = max(iroas.iroas * 2, 5)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=iroas.iroas,
            number=dict(suffix="x", font=dict(color=C["text"], size=36)),
            gauge=dict(
                axis=dict(range=[0, gauge_max],
                          tickfont=dict(color=C["text_muted"])),
                bar=dict(color=C["primary"]),
                bgcolor=C["surface2"],
                borderwidth=0,
                steps=[
                    dict(range=[0, 1], color=C["bad_bg"]),
                    dict(range=[1, 3], color=C["ok_bg"]),
                    dict(range=[3, gauge_max], color=C["primary_bg"]),
                ],
                threshold=dict(
                    line=dict(color=C["text"], width=2),
                    thickness=0.8,
                    value=1.0,
                ),
            ),
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        metric_card("Incremental Revenue", _fmt_dollar(iroas.incremental_revenue),
                     f"on {_fmt_dollar(iroas.total_ad_spend)} spend", accent=True)
        if iroas.incremental_conversions > 0:
            metric_card("CPIA", f"${iroas.cpia:,.2f}",
                         f"{iroas.incremental_conversions:,.0f} conversions")


# ── iROAS deep dive ──────────────────────────────────────────────────────

def _render_iroas_detail(report):
    iroas = report.iroas
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("iROAS", f"{iroas.iroas:.2f}x",
                     f"CI: [{iroas.iroas_lower_ci:.2f}x, {iroas.iroas_upper_ci:.2f}x]",
                     accent=True)
    with c2:
        metric_card("Incremental Revenue", _fmt_dollar(iroas.incremental_revenue),
                     f"Total Ad Spend: {_fmt_dollar(iroas.total_ad_spend)}")
    with c3:
        if iroas.incremental_conversions > 0:
            metric_card("CPIA", f"${iroas.cpia:,.2f}",
                         f"{iroas.incremental_conversions:,.0f} incremental conversions")
        elif iroas.total_ad_spend > 0 and iroas.incremental_revenue > 0:
            eff = iroas.incremental_revenue / iroas.total_ad_spend
            metric_card("Efficiency", f"{eff:.2f}x", "Revenue per $ spent")


# ── Ensemble ──────────────────────────────────────────────────────────────

def _render_ensemble(report):
    section_header("Model Ensemble")

    methods = list(report.estimator_results.keys())
    lifts = [report.estimator_results[m].relative_lift for m in methods]
    pvals = [report.estimator_results[m].p_value for m in methods]
    weights = [report.estimator_weights.get(m, 0) for m in methods]

    # Forest plot
    fig = go.Figure()
    method_colors = [C["primary"], C["ok"], C["warn"], C["purple"]]

    for i, method in enumerate(methods):
        r = report.estimator_results[method]
        color = method_colors[i % len(method_colors)]
        alpha = "0.15" if r.is_significant else "0.08"

        fig.add_trace(go.Bar(
            x=[r.lift_upper_ci - r.lift_lower_ci],
            y=[method.upper()],
            base=[r.lift_lower_ci],
            orientation="h",
            marker=dict(
                color=f"rgba({_hex_to_rgb(color)}, {alpha})",
                line=dict(color=color, width=1.5),
            ),
            showlegend=False,
            hovertemplate=(
                f"{method.upper()}: {r.relative_lift:+.1%} "
                f"[{r.lift_lower_ci:+.1%}, {r.lift_upper_ci:+.1%}]<extra></extra>"
            ),
        ))

        fig.add_trace(go.Scatter(
            x=[r.relative_lift], y=[method.upper()],
            mode="markers",
            marker=dict(color=color, size=12, symbol="diamond"),
            showlegend=False,
        ))

    fig.add_vline(x=0, line_dash="dash", line_color=C["bad"], line_width=1)

    fig.update_layout(
        title="Estimator Forest Plot",
        height=60 + len(methods) * 70,
        xaxis_title="Relative Lift",
        xaxis=dict(tickformat=".0%"),
        margin=dict(l=80, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Weights & detail
    c1, c2 = st.columns(2)

    with c1:
        fig_pie = go.Figure(data=[go.Pie(
            labels=[m.upper() for m in methods],
            values=weights,
            marker=dict(colors=method_colors[:len(methods)]),
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(color=C["text"]),
        )])
        fig_pie.update_layout(title="Ensemble Weights", height=300, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        border_color = C["border"]
        text_sec = C["text_secondary"]
        text_main = C["text"]
        rows_html = ""
        for i, m in enumerate(methods):
            sig_badge = status_badge(
                "significant" if report.estimator_results[m].is_significant
                else "not significant"
            )
            rows_html += (
                f'<tr style="border-bottom: 1px solid {border_color};">'
                f'<td style="padding:0.5rem;color:{text_main};">{m.upper()}</td>'
                f'<td style="text-align:right;padding:0.5rem;">{weights[i]:.0%}</td>'
                f'<td style="text-align:right;padding:0.5rem;">{lifts[i]:+.1%}</td>'
                f'<td style="text-align:right;padding:0.5rem;">{pvals[i]:.4f}</td>'
                f'<td style="text-align:center;padding:0.5rem;">{sig_badge}</td>'
                f'</tr>'
            )

        st.markdown(f"""
        <div class="lift-card">
            <h4>Model Details</h4>
            <table style="width:100%; color:{text_sec}; font-size:0.85rem;">
                <tr style="border-bottom: 1px solid {border_color};">
                    <th style="text-align:left;padding:0.5rem;">Model</th>
                    <th style="text-align:right;padding:0.5rem;">Weight</th>
                    <th style="text-align:right;padding:0.5rem;">Lift</th>
                    <th style="text-align:right;padding:0.5rem;">p-value</th>
                    <th style="text-align:center;padding:0.5rem;">Status</th>
                </tr>
                {rows_html}
            </table>
        </div>
        """, unsafe_allow_html=True)


# ── Validation ────────────────────────────────────────────────────────────

def _render_validation(report):
    v = report.validation
    section_header("Validation & Trust Score")

    trust_color = C["ok"] if v.trust_score >= 80 else C["warn"] if v.trust_score >= 60 else C["bad"]
    verdict = ("Trustworthy" if v.trust_score >= 80
               else "Marginal" if v.trust_score >= 60
               else "Not trustworthy")

    c1, c2 = st.columns([1, 2])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=v.trust_score,
            number=dict(suffix="/100", font=dict(color=trust_color, size=32)),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(color=C["text_muted"])),
                bar=dict(color=trust_color),
                bgcolor=C["surface2"],
                borderwidth=0,
                steps=[
                    dict(range=[0, 60], color=C["bad_bg"]),
                    dict(range=[60, 80], color=C["warn_bg"]),
                    dict(range=[80, 100], color=C["ok_bg"]),
                ],
            ),
        ))
        fig.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<div style='text-align:center;'>{status_badge(verdict)}</div>",
                    unsafe_allow_html=True)

    with c2:
        checks = [
            ("AA Test (pre-period)", f"p = {v.aa_test_p_value:.4f}", v.aa_test_passed),
            ("Pre-period L2", f"{v.l2_imbalance:.4f}", v.l2_imbalance < 0.10),
            ("Pre-period R²", f"{v.pre_period_r_squared:.3f}", v.pre_period_r_squared > 0.90),
            ("Placebo FPR", f"{v.false_positive_rate:.0%}", v.false_positive_rate < 0.15),
            ("Estimator Agreement", f"{v.estimator_agreement:.0%}", v.estimator_agreement > 0.70),
        ]

        for name, value, passed in checks:
            icon = "✓" if passed else "✗"
            color = C["ok"] if passed else C["bad"]
            st.markdown(f"""
            <div style="display:flex; align-items:center; padding:0.4rem 0;
                        border-bottom: 1px solid {C['border']};">
                <span style="color:{color}; font-size:1.1rem; margin-right:0.75rem;
                             width:1.5rem; text-align:center;">{icon}</span>
                <span style="flex:1; color:{C['text_secondary']};">{name}</span>
                <span style="color:{C['text']}; font-weight:600;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

    # Blockers
    if v.blockers:
        bad_color = C["bad"]
        st.markdown(f"<div class='section-header' style='color:{bad_color};'>Blockers</div>",
                    unsafe_allow_html=True)
        for b in v.blockers:
            st.error(b)

    if v.warnings:
        with st.expander(f"Warnings ({len(v.warnings)})"):
            for w in v.warnings:
                st.warning(w)


# ── Data quality ──────────────────────────────────────────────────────────

def _render_anomalies(report):
    section_header("Data Quality")
    if report.anomaly_blockers:
        for b in report.anomaly_blockers:
            st.error(f"**BLOCKER:** {b}")
    if report.anomaly_warnings:
        with st.expander(f"Data Quality Warnings ({len(report.anomaly_warnings)})"):
            for w in report.anomaly_warnings:
                st.warning(w)


# ── Recommendations ───────────────────────────────────────────────────────

def _render_recommendations(report):
    section_header("Recommendations")
    for i, rec in enumerate(report.recommendations, 1):
        st.markdown(f"""
        <div class="rec-card">
            <div style="color:{C['primary']};font-weight:700;margin-bottom:0.25rem;">
                {i}.</div>
            <p>{rec}</p>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Inline Spend Response (Tab 4)
# ═════════════════════════════════════════════════════════════════════════════

def _render_spend_summary(sr: dict):
    c1, c2, c3, c4 = st.columns(4)

    direction = sr.get("spend_change_direction", "")
    pct = sr.get("spend_change_pct", 0)
    dir_color = (C["bad"] if direction == "decrease"
                 else C["ok"] if direction == "increase"
                 else C["text"])

    with c1:
        metric_card("Current Spend", f"${sr.get('current_spend', 0):,.0f}/day",
                     f"Marginal ROAS: ${sr.get('current_marginal_roas', 0):.2f}")
    with c2:
        metric_card("Optimal Spend", f"${sr.get('optimal_spend', 0):,.0f}/day",
                     f"CI: [${sr.get('optimal_spend_lower', 0):,.0f}, "
                     f"${sr.get('optimal_spend_upper', 0):,.0f}]", accent=True)
    with c3:
        st.markdown(f"""
        <div class="lift-card">
            <h4>Recommended Change</h4>
            <div class="value" style="color:{dir_color};">{pct:+.0f}%</div>
            <div class="sub">{direction.title()} spend</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        metric_card("iROAS at Optimal", f"${sr.get('optimal_iroas', 0):.2f}",
                     f"Current: ${sr.get('current_iroas', 0):.2f}")


def _render_inline_response_curve(sr: dict):
    """Spend response curve embedded in the analysis report."""
    spend = sr.get("spend_curve", [])
    rev = sr.get("revenue_curve", [])
    rev_lower = sr.get("revenue_curve_lower", [])
    rev_upper = sr.get("revenue_curve_upper", [])

    if not spend or not rev:
        st.info("Curve data not available. Re-run analysis with DMA-level spend data.")
        return

    fig = go.Figure()

    # CI band
    if rev_lower and rev_upper and len(rev_lower) == len(spend):
        fig.add_trace(go.Scatter(
            x=spend + spend[::-1],
            y=rev_upper + rev_lower[::-1],
            fill="toself",
            fillcolor=f"rgba({_hex_to_rgb(C['primary'])}, 0.08)",
            line=dict(width=0),
            name="95% CI",
            hoverinfo="skip",
        ))

    # Main curve
    fig.add_trace(go.Scatter(
        x=spend, y=rev,
        mode="lines",
        name="Incremental Revenue",
        line=dict(color=C["primary"], width=3),
        hovertemplate="Spend: $%{x:,.0f}/day<br>Revenue: $%{y:,.0f}/day<extra></extra>",
    ))

    # Current marker
    current = sr.get("current_spend", 0)
    if current > 0 and spend:
        idx = min(range(len(spend)), key=lambda i: abs(spend[i] - current))
        fig.add_trace(go.Scatter(
            x=[current], y=[rev[idx] if idx < len(rev) else 0],
            mode="markers+text",
            marker=dict(color=C["warn"], size=14, symbol="circle",
                        line=dict(color=C["text"], width=2)),
            text=["Current"],
            textposition="top center",
            textfont=dict(color=C["warn"], size=12),
            name="Current Spend",
        ))

    # Optimal marker
    optimal = sr.get("optimal_spend", 0)
    if optimal > 0 and spend:
        idx = min(range(len(spend)), key=lambda i: abs(spend[i] - optimal))
        fig.add_trace(go.Scatter(
            x=[optimal], y=[rev[idx] if idx < len(rev) else 0],
            mode="markers+text",
            marker=dict(color=C["ok"], size=14, symbol="diamond",
                        line=dict(color=C["text"], width=2)),
            text=["Optimal"],
            textposition="top center",
            textfont=dict(color=C["ok"], size=12),
            name="Optimal Spend",
        ))

    # Observed boundary
    obs_max = sr.get("observed_spend_max", 0)
    if obs_max > 0:
        fig.add_vline(
            x=obs_max, line_dash="dot", line_color=C["text_muted"],
            annotation_text="Observed max",
            annotation_font_color=C["text_muted"],
            annotation_position="bottom right",
        )

    fig.update_layout(
        xaxis_title="Daily Ad Spend",
        yaxis_title="Daily Incremental Revenue",
        xaxis=dict(tickprefix="$", tickformat=",.0f"),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        height=450,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_inline_marginal_curve(sr: dict):
    """Marginal ROAS diminishing returns chart."""
    spend = sr.get("spend_curve", [])
    marginal = sr.get("marginal_roas_curve", [])

    if not spend or not marginal:
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=spend, y=marginal,
        mode="lines",
        name="Marginal ROAS",
        line=dict(color=C["purple"], width=3),
        hovertemplate="Spend: $%{x:,.0f}/day<br>Marginal ROAS: $%{y:.2f}<extra></extra>",
    ))

    # Target line
    target = sr.get("target_marginal_roas", 1.0)
    fig.add_hline(
        y=target, line_dash="dash", line_color=C["bad"],
        annotation_text=f"Target: ${target:.2f}",
        annotation_font_color=C["bad"],
    )

    # Profitable zone fill
    fig.add_trace(go.Scatter(
        x=spend, y=[max(m - target, 0) for m in marginal],
        fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(C['ok'])}, 0.10)",
        line=dict(width=0),
        name="Profitable zone",
        hoverinfo="skip",
    ))

    # Current
    current = sr.get("current_spend", 0)
    current_marg = sr.get("current_marginal_roas", 0)
    if current > 0:
        fig.add_trace(go.Scatter(
            x=[current], y=[current_marg],
            mode="markers",
            marker=dict(color=C["warn"], size=14, symbol="circle",
                        line=dict(color=C["text"], width=2)),
            name=f"Current (${current_marg:.2f})",
        ))

    # Optimal
    optimal = sr.get("optimal_spend", 0)
    if optimal > 0 and spend:
        idx = min(range(len(spend)), key=lambda i: abs(spend[i] - optimal))
        marg_at_opt = marginal[idx] if idx < len(marginal) else target
        fig.add_trace(go.Scatter(
            x=[optimal], y=[marg_at_opt],
            mode="markers",
            marker=dict(color=C["ok"], size=14, symbol="diamond",
                        line=dict(color=C["text"], width=2)),
            name=f"Optimal (${marg_at_opt:.2f})",
        ))

    fig.update_layout(
        xaxis_title="Daily Ad Spend",
        yaxis_title="Marginal ROAS ($/$ spent)",
        xaxis=dict(tickprefix="$", tickformat=",.0f"),
        yaxis=dict(tickprefix="$"),
        height=400,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The marginal ROAS curve shows the return on each additional dollar of spend. "
        "When marginal ROAS falls below the target (typically $1.00 = breakeven), "
        "you're spending past the point of diminishing returns."
    )


def _render_spend_recommendation(sr: dict):
    direction = sr.get("spend_change_direction", "")
    rec = sr.get("recommendation", "")

    if direction == "decrease":
        border = C["bad"]
        icon = "↓"
    elif direction == "increase":
        border = C["ok"]
        icon = "↑"
    else:
        border = C["primary"]
        icon = "→"

    st.markdown(f"""
    <div class="lift-card" style="border-color:{border}; border-width:2px;">
        <div style="display:flex; align-items:flex-start; gap:1rem;">
            <div style="font-size:2rem; color:{border};">{icon}</div>
            <div>
                <div style="color:{C['text']}; font-size:1.1rem; font-weight:600;
                            margin-bottom:0.5rem;">
                    {direction.title()} Spend {sr.get('spend_change_pct', 0):+.0f}%
                </div>
                <div style="color:{C['text_secondary']}; line-height:1.6;">
                    {rec}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
