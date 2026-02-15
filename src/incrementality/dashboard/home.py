"""LIFT Dashboard — Home / Overview page."""

from __future__ import annotations

import streamlit as st

from incrementality.dashboard.theme import (
    C, empty_state, metric_card, section_header, status_badge,
)
from incrementality.dashboard.data_loader import scan_designs, scan_reports


def render(data_dir: str, output_dir: str):
    st.title("Dashboard Overview")

    designs = scan_designs(data_dir)
    reports = scan_reports(output_dir)

    # ── Summary Metrics ───────────────────────────────────────────────────
    designed = [d for d in designs if d.status.value == "designed"]
    running = [d for d in designs if d.status.value == "running"]
    analyzed = [d for d in designs if d.status.value == "analyzed"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Tests", str(len(designs)), f"{len(reports)} analyzed", accent=True)
    with c2:
        metric_card("Designed", str(len(designed)), "Ready to deploy")
    with c3:
        metric_card("Running", str(len(running)), "Active holdouts")
    with c4:
        metric_card("Analyzed", str(len(analyzed)), "Reports available")

    # ── Recent Tests ──────────────────────────────────────────────────────
    section_header("Recent Tests")

    if not designs:
        empty_state(
            "🔬", "No tests yet",
            "Design your first test with the CLI: incrementality design --channel facebook",
        )
        return

    for design in designs[:8]:
        with st.container():
            cols = st.columns([3, 2, 2, 1.5, 1.5])
            with cols[0]:
                st.markdown(f"**{design.name}**")
                st.caption(f"`{design.test_id}`")
            with cols[1]:
                st.markdown(f"<small>{design.ad_channel.value.title()} &middot; "
                           f"{design.test_scope.value.title()}</small>",
                           unsafe_allow_html=True)
            with cols[2]:
                if design.recommended_start_date and design.recommended_end_date:
                    st.caption(f"{design.recommended_start_date} → {design.recommended_end_date}")
                else:
                    st.caption(f"{design.duration_weeks}w")
            with cols[3]:
                st.markdown(f"{design.num_treatment_dmas}T / {design.num_holdout_dmas}H DMAs")
            with cols[4]:
                st.markdown(status_badge(design.status.value), unsafe_allow_html=True)
            st.divider()

    # ── Report Highlights ─────────────────────────────────────────────────
    if reports:
        section_header("Latest Results")

        for report in reports[:3]:
            with st.container():
                st.markdown(f"""
                <div class="lift-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <h4 style="margin:0; color:{C['text']}!important;
                                       font-size:1rem;">{report.test_name}</h4>
                            <div style="color:{C['text_muted']}; font-size:0.8rem;">
                                {report.ad_channel.value.title()} &middot;
                                {report.test_start} → {report.test_end}
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div style="color:{C['ok'] if report.incrementality.is_significant else C['bad']};
                                        font-size:1.5rem; font-weight:700;">
                                {report.incrementality.relative_lift:+.1%}
                            </div>
                            <div style="color:{C['text_muted']}; font-size:0.8rem;">
                                iROAS: {report.iroas.iroas:.2f}x
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
