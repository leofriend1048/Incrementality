"""LIFT Dashboard — Spend Optimizer page with interactive response curves."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from incrementality.dashboard.data_loader import scan_reports
from incrementality.dashboard.theme import (
    C, empty_state, metric_card, section_header, status_badge,
)


def render(output_dir: str):
    st.title("Spend Optimizer")

    reports = scan_reports(output_dir)
    reports_with_spend = [r for r in reports if r.spend_response]

    if not reports_with_spend:
        empty_state(
            "💰", "No spend response data available",
            "Run an analysis with ad spend data to generate spend optimization curves.",
        )
        return

    # ── Report Selector ───────────────────────────────────────────────────
    options = {f"{r.test_name}  ({r.test_id})": r for r in reports_with_spend}
    selected_key = st.selectbox("Select a report", list(options.keys()))
    report = options[selected_key]
    sr = report.spend_response

    st.divider()

    # ── Executive Cards ───────────────────────────────────────────────────
    _render_summary_cards(sr)

    # ── Interactive Response Curve ────────────────────────────────────────
    _render_response_curve(sr)

    # ── Marginal ROAS Curve ───────────────────────────────────────────────
    _render_marginal_curve(sr)

    # ── Average vs Marginal ROAS ──────────────────────────────────────────
    _render_avg_vs_marginal(sr)

    # ── Recommendation ────────────────────────────────────────────────────
    _render_recommendation(sr)

    # ── Technical Details ─────────────────────────────────────────────────
    _render_details(sr)


def _render_summary_cards(sr: dict):
    c1, c2, c3, c4 = st.columns(4)

    direction = sr.get("spend_change_direction", "")
    pct = sr.get("spend_change_pct", 0)
    dir_color = C["bad"] if direction == "decrease" else C["ok"] if direction == "increase" else C["text"]

    with c1:
        metric_card("Current Spend", f"${sr.get('current_spend', 0):,.0f}/day",
                    f"Marginal ROAS: ${sr.get('current_marginal_roas', 0):.2f}")
    with c2:
        metric_card("Optimal Spend", f"${sr.get('optimal_spend', 0):,.0f}/day",
                    f"95% CI: [${sr.get('optimal_spend_lower', 0):,.0f}, "
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


def _render_response_curve(sr: dict):
    section_header("Spend Response Curve")

    spend = sr.get("spend_curve", [])
    rev = sr.get("revenue_curve", [])
    rev_lower = sr.get("revenue_curve_lower", [])
    rev_upper = sr.get("revenue_curve_upper", [])

    if not spend or not rev:
        st.warning("No curve data available.")
        return

    fig = go.Figure()

    # Confidence band
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

    # Current spend marker
    current = sr.get("current_spend", 0)
    if current > 0 and spend:
        # Find closest point
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
            hovertemplate=f"Current: ${current:,.0f}/day<extra></extra>",
        ))

    # Optimal spend marker
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
            hovertemplate=f"Optimal: ${optimal:,.0f}/day<extra></extra>",
        ))

    # Optimal CI band
    lower = sr.get("optimal_spend_lower", 0)
    upper = sr.get("optimal_spend_upper", 0)
    if lower > 0 and upper > 0:
        fig.add_vrect(
            x0=lower, x1=upper,
            fillcolor=f"rgba({_hex_to_rgb(C['ok'])}, 0.06)",
            line=dict(color=C["ok"], width=1, dash="dot"),
            annotation_text="Optimal range",
            annotation_font_color=C["ok"],
            annotation_position="top",
        )

    # Observed data boundary
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
        height=500,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_marginal_curve(sr: dict):
    section_header("Marginal ROAS — Diminishing Returns")

    spend = sr.get("spend_curve", [])
    marginal = sr.get("marginal_roas_curve", [])

    if not spend or not marginal:
        return

    fig = go.Figure()

    # Marginal ROAS line
    fig.add_trace(go.Scatter(
        x=spend, y=marginal,
        mode="lines",
        name="Marginal ROAS",
        line=dict(color=C["purple"], width=3),
        hovertemplate="Spend: $%{x:,.0f}/day<br>Marginal ROAS: $%{y:.2f}<extra></extra>",
    ))

    # Breakeven line
    target = sr.get("target_marginal_roas", 1.0)
    fig.add_hline(
        y=target, line_dash="dash", line_color=C["bad"],
        annotation_text=f"Target: ${target:.2f}",
        annotation_font_color=C["bad"],
    )

    # Fill profitable region
    fig.add_trace(go.Scatter(
        x=spend, y=[max(m - target, 0) for m in marginal],
        fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(C['ok'])}, 0.10)",
        line=dict(width=0),
        name="Profitable zone",
        hoverinfo="skip",
    ))

    # Current marker
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

    # Optimal marker
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
        height=450,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The marginal ROAS curve shows the return on each additional dollar of spend. "
        "When marginal ROAS falls below the target (typically $1.00 = breakeven), "
        "you're spending past the point of diminishing returns."
    )


def _render_avg_vs_marginal(sr: dict):
    section_header("Average vs. Marginal ROAS")

    spend = sr.get("spend_curve", [])
    marginal = sr.get("marginal_roas_curve", [])
    avg = sr.get("avg_roas_curve", [])

    if not spend or not marginal or not avg:
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=spend, y=avg,
        mode="lines",
        name="Average iROAS",
        line=dict(color=C["primary"], width=2),
        hovertemplate="Avg iROAS: $%{y:.2f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=spend, y=marginal,
        mode="lines",
        name="Marginal ROAS",
        line=dict(color=C["purple"], width=2, dash="dash"),
        hovertemplate="Marginal ROAS: $%{y:.2f}<extra></extra>",
    ))

    fig.add_hline(y=1.0, line_dash="dot", line_color=C["bad"],
                  annotation_text="Breakeven",
                  annotation_font_color=C["bad"])

    fig.update_layout(
        xaxis_title="Daily Ad Spend",
        yaxis_title="ROAS ($/$ spent)",
        xaxis=dict(tickprefix="$", tickformat=",.0f"),
        yaxis=dict(tickprefix="$"),
        height=400,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Average iROAS is total incremental revenue divided by total spend. "
        "Marginal ROAS is the return on the *next* dollar. "
        "Average can look profitable while marginal is already below breakeven — "
        "that's the diminishing returns trap."
    )


def _render_recommendation(sr: dict):
    section_header("Optimization Recommendation")

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


def _render_details(sr: dict):
    with st.expander("Technical Details"):
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("R²", f"{sr.get('r_squared', 0):.3f}", "Fit quality")
        with c2:
            metric_card("DMAs Used", str(sr.get("n_dmas_used", 0)),
                        "In curve estimation")
        with c3:
            calibrated = sr.get("calibrated", False)
            converged = sr.get("converged", False)
            st.markdown(f"""
            <div class="lift-card">
                <h4>Model Status</h4>
                <div style="margin-top:0.5rem;">
                    {status_badge("pass" if converged else "fail")}
                    <span style="color:{C['text_secondary']}; margin-left:0.5rem;">
                        Convergence</span>
                </div>
                <div style="margin-top:0.5rem;">
                    {status_badge("pass" if calibrated else "warning")}
                    <span style="color:{C['text_secondary']}; margin-left:0.5rem;">
                        Causal calibration</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Hill function parameters
        st.markdown(f"""
        <div class="lift-card">
            <h4>Hill Function Parameters</h4>
            <table style="width:100%; color:{C['text_secondary']}; font-size:0.85rem;">
                <tr><td>R_max (saturation)</td>
                    <td style="text-align:right; color:{C['text']};">
                        {sr.get('r_max', 0):,.2f}</td></tr>
                <tr><td>K (half-saturation)</td>
                    <td style="text-align:right; color:{C['text']};">
                        {sr.get('k', 0):,.2f}</td></tr>
                <tr><td>Alpha (shape)</td>
                    <td style="text-align:right; color:{C['text']};">
                        {sr.get('alpha', 0):.3f}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}"
