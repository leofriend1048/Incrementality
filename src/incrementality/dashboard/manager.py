"""LIFT Dashboard — Test Manager page (list, status, lifecycle)."""

from __future__ import annotations

import streamlit as st

from incrementality.dashboard.data_loader import scan_designs, scan_reports
from incrementality.dashboard.theme import (
    C, empty_state, metric_card, section_header, status_badge,
)


def render(data_dir: str, output_dir: str):
    st.title("Test Manager")

    designs = scan_designs(data_dir)
    reports = scan_reports(output_dir)
    report_ids = {r.test_id for r in reports}

    if not designs:
        empty_state(
            "📋", "No tests to manage",
            "Design your first test with the CLI: incrementality design --channel facebook",
        )
        return

    # ── Status Filter ─────────────────────────────────────────────────────
    statuses = sorted({d.status.value for d in designs})
    selected_status = st.multiselect(
        "Filter by status", statuses, default=statuses,
    )

    filtered = [d for d in designs if d.status.value in selected_status]

    # ── Test Table ────────────────────────────────────────────────────────
    section_header(f"Tests ({len(filtered)})")

    for design in filtered:
        has_report = design.test_id in report_ids

        with st.container():
            st.markdown(f"""
            <div class="lift-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="flex:1;">
                        <div style="color:{C['text']}; font-size:1.1rem; font-weight:600;">
                            {design.name}
                        </div>
                        <div style="color:{C['text_muted']}; font-size:0.8rem; margin-top:0.25rem;">
                            <code>{design.test_id}</code>
                            &nbsp;&middot;&nbsp; {design.ad_channel.value.title()}
                            &nbsp;&middot;&nbsp; {design.test_scope.value.title()}
                            &nbsp;&middot;&nbsp; {design.duration_weeks}w
                            &nbsp;&middot;&nbsp; {design.num_treatment_dmas}T / {design.num_holdout_dmas}H DMAs
                        </div>
                    </div>
                    <div style="text-align:right;">
                        {status_badge(design.status.value)}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Action buttons
            cols = st.columns([2, 2, 2, 2, 4])

            if design.status.value == "designed":
                with cols[0]:
                    if st.button("Deploy", key=f"deploy_{design.test_id}", type="primary"):
                        st.info(f"Run: `incrementality execute --test-id {design.test_id}`")
                with cols[1]:
                    st.button("View Design", key=f"view_{design.test_id}",
                             help="Go to Test Designer page")

            elif design.status.value == "running":
                with cols[0]:
                    if st.button("Update", key=f"update_{design.test_id}"):
                        st.info(f"Run: `incrementality execute --test-id {design.test_id} --update`")
                with cols[1]:
                    if st.button("Revert", key=f"revert_{design.test_id}",
                                type="secondary"):
                        st.warning(f"Run: `incrementality revert --test-id {design.test_id}`")
                with cols[2]:
                    if st.button("Analyze", key=f"analyze_run_{design.test_id}",
                                type="primary"):
                        st.info(f"Run: `incrementality analyze --test-id {design.test_id}`")

            elif design.status.value in ("completed", "analyzed"):
                with cols[0]:
                    if has_report:
                        st.button("View Report", key=f"report_{design.test_id}",
                                 type="primary",
                                 help="Go to Analysis page")
                    else:
                        if st.button("Analyze", key=f"analyze_{design.test_id}",
                                    type="primary"):
                            st.info(f"Run: `incrementality analyze --test-id {design.test_id}`")

            st.markdown("---")

    # ── Lifecycle Guide ───────────────────────────────────────────────────
    with st.expander("Test Lifecycle Guide"):
        st.markdown(f"""
        <div style="color:{C['text_secondary']};">

        **1. Designed** → Test plan created with power analysis and DMA assignment.

        **2. Running** → Holdout exclusions deployed to ad platform. Ads are being held
        back in holdout DMAs. Use <code>--update</code> to catch new ad sets.

        **3. Completed** → Test period ended. Holdout reverted.

        **4. Analyzed** → Causal inference run. Report available with iROAS,
        validation, and spend optimization.

        </div>
        """, unsafe_allow_html=True)
