"""LIFT Dashboard — Analysis Report page with rich visualizations."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from incrementality.dashboard.data_loader import scan_reports
from incrementality.dashboard.theme import (
    C, empty_state, metric_card, score_bar, section_header, status_badge,
)


def render(output_dir: str):
    st.title("Analysis Reports")

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

    st.divider()

    # ── Executive Summary Cards ───────────────────────────────────────────
    _render_executive_summary(report)

    # ── Lift Visualization ────────────────────────────────────────────────
    _render_lift_chart(report)

    # ── Ensemble Comparison ───────────────────────────────────────────────
    if report.estimator_results and len(report.estimator_results) > 1:
        _render_ensemble(report)

    # ── iROAS Deep Dive ───────────────────────────────────────────────────
    _render_iroas(report)

    # ── Spend Response ────────────────────────────────────────────────────
    if report.spend_response:
        _render_spend_response_preview(report)

    # ── Validation ────────────────────────────────────────────────────────
    if report.validation:
        _render_validation(report)

    # ── Data Quality ──────────────────────────────────────────────────────
    if report.anomaly_warnings or report.anomaly_blockers:
        _render_anomalies(report)

    # ── Recommendations ───────────────────────────────────────────────────
    if report.recommendations:
        _render_recommendations(report)


# ── Executive Summary ─────────────────────────────────────────────────────────

def _render_executive_summary(report):
    inc = report.incrementality
    iroas = report.iroas

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        lift_color = C["ok"] if inc.is_significant and inc.relative_lift > 0 else (
            C["bad"] if inc.relative_lift < 0 else C["text_muted"]
        )
        st.markdown(f"""
        <div class="lift-card-accent">
            <h4>Incremental Lift</h4>
            <div class="value" style="color:{lift_color};">{inc.relative_lift:+.1%}</div>
            <div class="sub">[{inc.lift_lower_ci:+.1%}, {inc.lift_upper_ci:+.1%}]</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        iroas_color = C["ok"] if iroas.iroas >= 1.0 else C["warn"] if iroas.iroas >= 0 else C["bad"]
        iroas_label = "Excellent" if iroas.iroas >= 3.0 else "Profitable" if iroas.iroas >= 1.0 else "Below breakeven"
        st.markdown(f"""
        <div class="lift-card-accent">
            <h4>Incremental ROAS</h4>
            <div class="value" style="color:{iroas_color};">{iroas.iroas:.2f}x</div>
            <div class="sub">{iroas_label}</div>
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
            <div class="value">${iroas.incremental_revenue:,.0f}</div>
            <div class="sub">on ${iroas.total_ad_spend:,.0f} spend</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Additional metrics row ────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Test Period",
                    f"{report.duration_weeks}w",
                    f"{report.test_start} → {report.test_end}")
    with c2:
        metric_card("Channel",
                    report.ad_channel.value.title(),
                    report.test_scope.value.title())
    with c3:
        metric_card("DMAs",
                    f"{report.num_treatment_dmas}T / {report.num_holdout_dmas}H",
                    report.measurement_scope.value.replace("_", " ").title())
    with c4:
        if inc.lift_likelihood > 0:
            metric_card("P(lift > 0)", f"{inc.lift_likelihood:.1%}",
                        f"Cohen's d = {inc.cohen_d:.3f}")


# ── Lift Chart ────────────────────────────────────────────────────────────────

def _render_lift_chart(report):
    section_header("Incremental Lift")
    inc = report.incrementality

    fig = go.Figure()

    # CI range as a horizontal bar
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

    # Zero line
    fig.add_vline(x=0, line_dash="dash", line_color=C["bad"], line_width=1)

    fig.update_layout(
        height=180,
        xaxis_title="Relative Lift",
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(visible=True),
        showlegend=True,
        legend=dict(orientation="h", y=-0.3),
        margin=dict(l=80, r=20, t=20, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Lift likelihood bar
    if inc.lift_likelihood > 0:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(f"**P(true lift > 0):** {inc.lift_likelihood:.1%}")
        with c2:
            st.markdown(score_bar(inc.lift_likelihood * 100, 100, 12), unsafe_allow_html=True)


# ── Ensemble ──────────────────────────────────────────────────────────────────

def _render_ensemble(report):
    section_header("Model Ensemble")

    methods = list(report.estimator_results.keys())
    lifts = [report.estimator_results[m].relative_lift for m in methods]
    pvals = [report.estimator_results[m].p_value for m in methods]
    weights = [report.estimator_weights.get(m, 0) for m in methods]

    # Forest plot — all methods
    fig = go.Figure()

    for i, method in enumerate(methods):
        r = report.estimator_results[method]
        color = C["ok"] if r.is_significant else C["text_muted"]

        # CI bar
        fig.add_trace(go.Bar(
            x=[r.lift_upper_ci - r.lift_lower_ci],
            y=[method.upper()],
            base=[r.lift_lower_ci],
            orientation="h",
            marker=dict(color=f"rgba({_hex_to_rgb(color)}, 0.15)",
                       line=dict(color=color, width=1)),
            showlegend=False,
            hovertemplate=f"{method.upper()}: {r.relative_lift:+.1%} "
                         f"[{r.lift_lower_ci:+.1%}, {r.lift_upper_ci:+.1%}]<extra></extra>",
        ))

        # Point estimate
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

    # Weights and details table
    c1, c2 = st.columns(2)

    with c1:
        # Weight pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=[m.upper() for m in methods],
            values=weights,
            marker=dict(colors=[C["primary"], C["ok"], C["warn"], C["purple"]][:len(methods)]),
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(color=C["text"]),
        )])
        fig_pie.update_layout(
            title="Ensemble Weights",
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        # Details table
        st.markdown(f"""
        <div class="lift-card">
            <h4>Model Details</h4>
            <table style="width:100%; color:{C['text_secondary']}; font-size:0.85rem;">
                <tr style="border-bottom: 1px solid {C['border']};">
                    <th style="text-align:left;padding:0.5rem;">Model</th>
                    <th style="text-align:right;padding:0.5rem;">Weight</th>
                    <th style="text-align:right;padding:0.5rem;">Lift</th>
                    <th style="text-align:right;padding:0.5rem;">p-value</th>
                    <th style="text-align:center;padding:0.5rem;">Status</th>
                </tr>
                {"".join(
                    f'''<tr style="border-bottom: 1px solid {C['border']};">
                        <td style="padding:0.5rem;color:{C['text']};">{m.upper()}</td>
                        <td style="text-align:right;padding:0.5rem;">{weights[i]:.0%}</td>
                        <td style="text-align:right;padding:0.5rem;">{lifts[i]:+.1%}</td>
                        <td style="text-align:right;padding:0.5rem;">{pvals[i]:.4f}</td>
                        <td style="text-align:center;padding:0.5rem;">
                            {status_badge("significant" if report.estimator_results[m].is_significant else "not significant")}
                        </td>
                    </tr>'''
                    for i, m in enumerate(methods)
                )}
            </table>
        </div>
        """, unsafe_allow_html=True)


# ── iROAS ─────────────────────────────────────────────────────────────────────

def _render_iroas(report):
    section_header("Incremental ROAS")
    iroas = report.iroas

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("iROAS", f"{iroas.iroas:.2f}x",
                    f"95% CI: [{iroas.iroas_lower_ci:.2f}x, {iroas.iroas_upper_ci:.2f}x]",
                    accent=True)
    with c2:
        metric_card("Incremental Revenue", f"${iroas.incremental_revenue:,.0f}",
                    f"Total Ad Spend: ${iroas.total_ad_spend:,.0f}")
    with c3:
        if iroas.incremental_conversions > 0:
            metric_card("CPIA", f"${iroas.cpia:,.2f}",
                        f"{iroas.incremental_conversions:,.0f} incremental conversions")
        elif iroas.total_ad_spend > 0 and iroas.incremental_revenue > 0:
            eff = iroas.incremental_revenue / iroas.total_ad_spend
            metric_card("Efficiency", f"{eff:.2f}x",
                        "Revenue per $ spent")

    # iROAS CI visualization — gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=iroas.iroas,
        number=dict(suffix="x", font=dict(color=C["text"], size=36)),
        gauge=dict(
            axis=dict(range=[0, max(iroas.iroas * 2, 5)],
                     tickfont=dict(color=C["text_muted"])),
            bar=dict(color=C["primary"]),
            bgcolor=C["surface2"],
            borderwidth=0,
            steps=[
                dict(range=[0, 1], color=C["bad_bg"]),
                dict(range=[1, 3], color=C["ok_bg"]),
                dict(range=[3, max(iroas.iroas * 2, 5)], color=C["primary_bg"]),
            ],
            threshold=dict(
                line=dict(color=C["text"], width=2),
                thickness=0.8,
                value=1.0,  # breakeven line
            ),
        ),
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Platform breakdown
    if iroas.shopify_incremental_revenue > 0 or iroas.amazon_incremental_revenue > 0:
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Shopify iROAS", f"{iroas.shopify_iroas:.2f}x",
                        f"${iroas.shopify_incremental_revenue:,.0f} incremental")
        with c2:
            metric_card("Amazon iROAS", f"{iroas.amazon_iroas:.2f}x",
                        f"${iroas.amazon_incremental_revenue:,.0f} incremental")

    # IF / CPIA section
    if iroas.incrementality_factor > 0:
        section_header("Attribution Quality")
        c1, c2, c3 = st.columns(3)
        with c1:
            if_color = C["ok"] if 0.5 <= iroas.incrementality_factor <= 1.5 else C["warn"]
            metric_card("Incrementality Factor", f"{iroas.incrementality_factor:.2f}",
                        "1.0 = perfect attribution")
        with c2:
            metric_card("Attributed (Platform)", f"{iroas.attributed_conversions:,.0f}",
                        "From Ads Manager")
        with c3:
            metric_card("Incremental (Measured)", f"{iroas.incremental_conversions:,.0f}",
                        "From experiment")


# ── Spend Response Preview ────────────────────────────────────────────────────

def _render_spend_response_preview(report):
    """Quick spend response summary — full version on Spend Optimizer page."""
    sr = report.spend_response
    section_header("Spend Response Curve")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Current Daily Spend", f"${sr.get('current_spend', 0):,.0f}",
                    f"Marginal ROAS: ${sr.get('current_marginal_roas', 0):.2f}")
    with c2:
        metric_card("Optimal Daily Spend", f"${sr.get('optimal_spend', 0):,.0f}",
                    f"iROAS at optimal: ${sr.get('optimal_iroas', 0):.2f}",
                    accent=True)
    with c3:
        direction = sr.get("spend_change_direction", "")
        pct = sr.get("spend_change_pct", 0)
        dir_color = C["bad"] if direction == "decrease" else C["ok"]
        metric_card("Recommendation",
                    f"{pct:+.0f}%",
                    f"{direction.title()} spend")

    st.info("See the **Spend Optimizer** page for the full interactive response curve.")


# ── Validation ────────────────────────────────────────────────────────────────

def _render_validation(report):
    v = report.validation
    section_header("Validation & Trust Score")

    # Trust score gauge
    trust_color = C["ok"] if v.trust_score >= 80 else C["warn"] if v.trust_score >= 60 else C["bad"]
    verdict = "Trustworthy" if v.trust_score >= 80 else "Marginal" if v.trust_score >= 60 else "Not trustworthy"

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
        # Individual checks
        checks = [
            ("AA Test (pre-period)", f"p = {v.aa_test_p_value:.4f}",
             v.aa_test_passed),
            ("Pre-period L2", f"{v.l2_imbalance:.4f}",
             v.l2_imbalance < 0.10),
            ("Pre-period R²", f"{v.pre_period_r_squared:.3f}",
             v.pre_period_r_squared > 0.90),
            ("Placebo FPR", f"{v.false_positive_rate:.0%}",
             v.false_positive_rate < 0.15),
            ("Estimator Agreement", f"{v.estimator_agreement:.0%}",
             v.estimator_agreement > 0.70),
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

    # Blockers and warnings
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


# ── Anomalies ─────────────────────────────────────────────────────────────────

def _render_anomalies(report):
    section_header("Data Quality")

    if report.anomaly_blockers:
        for b in report.anomaly_blockers:
            st.error(f"**BLOCKER:** {b}")

    if report.anomaly_warnings:
        with st.expander(f"Data Quality Warnings ({len(report.anomaly_warnings)})"):
            for w in report.anomaly_warnings:
                st.warning(w)


# ── Recommendations ───────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' to 'R, G, B' for rgba()."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}"
