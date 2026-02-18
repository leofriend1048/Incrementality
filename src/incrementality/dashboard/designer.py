"""LIFT Dashboard — Test Designer page with power analysis visualization."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from incrementality.dashboard.data_loader import scan_designs
from incrementality.dashboard.theme import (
    C, empty_state, metric_card, score_bar, section_header,
)


def render(data_dir: str):
    st.title("Test Designer")

    designs = scan_designs(data_dir)

    if not designs:
        empty_state(
            "📐", "No test designs found",
            "Create one via CLI: incrementality design --channel facebook --name \"My Test\"",
        )
        return

    # ── Design Selector ───────────────────────────────────────────────────
    options = {f"{d.name}  ({d.test_id})": d for d in designs}
    selected_key = st.selectbox("Select a test design", list(options.keys()))
    design = options[selected_key]

    st.divider()

    # ── Design Overview ───────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Channel", design.ad_channel.value.title(), accent=True)
    with c2:
        metric_card("Scope", design.test_scope.value.title(),
                    f"{len(design.campaign_ids)} campaigns" if design.campaign_ids else "Full channel")
    with c3:
        metric_card("Duration", f"{design.duration_weeks} weeks",
                    f"{design.recommended_start_date} → {design.recommended_end_date}" if design.recommended_start_date else "")
    with c4:
        metric_card("Measurement", design.measurement_scope.value.replace("_", " ").title())

    # ── Cell Assignment ───────────────────────────────────────────────────
    section_header("Cell Assignment")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Treatment DMAs", str(design.num_treatment_dmas),
                    f"${design.treatment_cell.historical_revenue:,.0f} historical revenue")
    with c2:
        metric_card("Holdout DMAs", str(design.num_holdout_dmas),
                    f"${design.holdout_cell.historical_revenue:,.0f} historical revenue")
    with c3:
        bal = design.balance_score
        metric_card("Balance Score", f"{bal:.1%}",
                    "Treatment ≈ Holdout on key covariates")

    # ── Cell Comparison Chart ─────────────────────────────────────────────
    fig = go.Figure()
    labels = ["Total Revenue", "Total Orders", "Population"]
    treat_vals = [
        design.treatment_cell.historical_revenue,
        design.treatment_cell.historical_orders,
        design.treatment_cell.total_population,
    ]
    holdout_vals = [
        design.holdout_cell.historical_revenue,
        design.holdout_cell.historical_orders,
        design.holdout_cell.total_population,
    ]
    # Normalize for comparison (per-DMA)
    n_t = max(design.num_treatment_dmas, 1)
    n_h = max(design.num_holdout_dmas, 1)
    treat_norm = [v / n_t for v in treat_vals]
    holdout_norm = [v / n_h for v in holdout_vals]

    fig.add_trace(go.Bar(
        name="Treatment (per DMA)", x=labels, y=treat_norm,
        marker_color=C["primary"], marker_line_width=0,
    ))
    fig.add_trace(go.Bar(
        name="Holdout (per DMA)", x=labels, y=holdout_norm,
        marker_color=C["ok"], marker_line_width=0,
    ))
    fig.update_layout(
        title="Cell Balance — Per-DMA Averages",
        barmode="group", height=350,
        yaxis_title="Value per DMA",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Power Analysis ────────────────────────────────────────────────────
    if design.power_analysis:
        _render_power_analysis(design)

    # ── Feasibility ───────────────────────────────────────────────────────
    if design.feasibility:
        _render_feasibility(design)


def _render_power_analysis(design):
    pa = design.power_analysis
    section_header("Power Analysis")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("MDE", f"{pa.minimum_detectable_effect:.1%}",
                    "Minimum detectable effect")
    with c2:
        metric_card("Duration", f"{pa.recommended_duration_weeks}w",
                    "Recommended test length")
    with c3:
        power_val = pa.simulated_power if pa.simulated_power > 0 else pa.statistical_power
        metric_card("Power", f"{power_val:.0%}",
                    "Simulated" if pa.simulated_power > 0 else "Analytical")
    with c4:
        metric_card("Power Score", f"{pa.power_score:.0f}/100", accent=True)

    st.markdown(score_bar(pa.power_score, 100), unsafe_allow_html=True)

    # ── Power Curve ── MDE vs Duration ────────────────────────────────────
    _render_power_curve(pa, design)


def _render_power_curve(pa, design):
    """Visualize how MDE changes with test duration."""
    durations = list(range(1, 13))
    # Approximate MDE scaling: MDE ∝ 1/sqrt(duration)
    # Anchor to the actual computed values
    base_dur = pa.recommended_duration_weeks
    base_mde = pa.minimum_detectable_effect
    mdes = [base_mde * np.sqrt(base_dur / max(d, 0.5)) for d in durations]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=durations, y=[m * 100 for m in mdes],
        mode="lines+markers",
        name="MDE",
        line=dict(color=C["primary"], width=3),
        marker=dict(size=8, color=C["primary"]),
        hovertemplate="Week %{x}: MDE = %{y:.1f}%<extra></extra>",
    ))

    # Mark the recommended duration
    fig.add_vline(
        x=design.duration_weeks, line_dash="dash",
        line_color=C["ok"], annotation_text="Selected",
        annotation_font_color=C["ok"],
    )

    # Mark common MDE thresholds
    for thresh, label in [(5, "5% MDE"), (10, "10% MDE"), (15, "15% MDE")]:
        fig.add_hline(
            y=thresh, line_dash="dot", line_color=C["border_light"],
            annotation_text=label, annotation_font_color=C["text_muted"],
            annotation_position="bottom right",
        )

    fig.update_layout(
        title="Minimum Detectable Effect vs. Test Duration",
        xaxis_title="Duration (weeks)",
        yaxis_title="MDE (%)",
        height=400,
        yaxis=dict(rangemode="tozero"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_feasibility(design):
    feas = design.feasibility
    section_header("Feasibility Assessment")

    # Status card
    if feas.is_feasible:
        st.markdown(f"""
        <div class="lift-card" style="border-color:{C['ok_dark']};">
            <div style="display:flex;align-items:center;gap:1rem;">
                <div style="font-size:2rem;">✓</div>
                <div>
                    <div style="color:{C['ok']};font-size:1.2rem;font-weight:700;">
                        Test is FEASIBLE</div>
                    <div style="color:{C['text_muted']};">
                        Power score: {feas.power_score:.0f}/100</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="lift-card" style="border-color:{C['bad_dark']};">
            <div style="display:flex;align-items:center;gap:1rem;">
                <div style="font-size:2rem;">✗</div>
                <div>
                    <div style="color:{C['bad']};font-size:1.2rem;font-weight:700;">
                        Test is NOT FEASIBLE</div>
                    <div style="color:{C['text_muted']};">
                        Power score: {feas.power_score:.0f}/100</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        metric_card("Opportunity Cost", f"${feas.estimated_opportunity_cost:,.0f}",
                    f"Over {feas.estimated_duration_weeks} weeks")
    with c2:
        metric_card("Min Holdout DMAs", str(feas.min_holdout_dmas),
                    f"Actual: {design.num_holdout_dmas}")

    # Reasons
    if feas.reasons:
        with st.expander("Feasibility Details", expanded=True):
            ok_color = C["ok"]
            bad_color = C["bad"]
            warn_color = C["warn"]
            for r in feas.reasons:
                if "PASS" in r.upper():
                    st.markdown(f"<span style='color:{ok_color}'>✓</span> {r}",
                               unsafe_allow_html=True)
                elif "BLOCK" in r.upper():
                    st.markdown(f"<span style='color:{bad_color}'>✗</span> {r}",
                               unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:{warn_color}'>⚠</span> {r}",
                               unsafe_allow_html=True)

    # Recommendations
    if feas.recommendations:
        with st.expander("Recommendations"):
            for rec in feas.recommendations:
                st.markdown(f"""
                <div class="rec-card"><p>{rec}</p></div>
                """, unsafe_allow_html=True)
