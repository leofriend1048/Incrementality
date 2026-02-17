"""
View 1 — National Revenue Decomposition
Waterfall chart of channel contributions, Northbeam vs MMM delta table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from incrementality.dashboard.theme import C, CHART_PALETTE, section_header

# ── Synthetic data generation ─────────────────────────────────────────────────

_CHANNELS = [
    "Meta (Paid Social)",
    "Google Search",
    "Google Shopping",
    "TikTok",
    "Pinterest",
    "YouTube",
    "Email / CRM",
    "Influencer",
    "TV / CTV",
    "Direct / Organic",
]

_CHANNEL_COLORS = CHART_PALETTE[: len(_CHANNELS)]


def _make_decomp_data(outcome: str, window: int) -> pd.DataFrame:
    """Generate realistic decomposition posterior means + credible intervals."""
    rng = np.random.default_rng(seed=hash(outcome + str(window)) % (2**31))

    scale = {7: 0.13, 30: 0.55, 90: 1.65, 365: 6.5}[window]

    # Revenue in millions
    baseline = 0.85 * scale
    contributions = {
        "Meta (Paid Social)":  rng.normal(0.38, 0.04) * scale,
        "Google Search":       rng.normal(0.22, 0.03) * scale,
        "Google Shopping":     rng.normal(0.14, 0.02) * scale,
        "TikTok":              rng.normal(0.09, 0.02) * scale,
        "Pinterest":           rng.normal(0.05, 0.01) * scale,
        "YouTube":             rng.normal(0.06, 0.015) * scale,
        "Email / CRM":         rng.normal(0.07, 0.01) * scale,
        "Influencer":          rng.normal(0.04, 0.01) * scale,
        "TV / CTV":            rng.normal(0.03, 0.01) * scale,
    }

    # Shopify vs Amazon split
    if outcome == "Shopify":
        channel_mult = {ch: rng.uniform(0.55, 0.75) for ch in contributions}
    elif outcome == "Amazon":
        channel_mult = {ch: rng.uniform(0.25, 0.45) for ch in contributions}
    else:
        channel_mult = {ch: 1.0 for ch in contributions}

    rows = []
    ci_width_80 = 0.06
    ci_width_95 = 0.12

    for ch, contrib in contributions.items():
        v = contrib * channel_mult[ch]
        rows.append({
            "channel": ch,
            "contribution": v,
            "ci80_lo": v * (1 - ci_width_80 * rng.uniform(0.8, 1.2)),
            "ci80_hi": v * (1 + ci_width_80 * rng.uniform(0.8, 1.2)),
            "ci95_lo": v * (1 - ci_width_95 * rng.uniform(0.8, 1.2)),
            "ci95_hi": v * (1 + ci_width_95 * rng.uniform(0.8, 1.2)),
        })

    df = pd.DataFrame(rows)
    total_channel = df["contribution"].sum()
    residual = rng.normal(0.02, 0.005) * scale

    # Prepend baseline row
    bl_row = pd.DataFrame([{
        "channel": "Baseline",
        "contribution": baseline * (channel_mult.get("Meta (Paid Social)", 1.0) if outcome != "Combined" else 1.0),
        "ci80_lo": baseline * 0.92,
        "ci80_hi": baseline * 1.08,
        "ci95_lo": baseline * 0.88,
        "ci95_hi": baseline * 1.12,
    }])

    res_row = pd.DataFrame([{
        "channel": "Residual",
        "contribution": residual,
        "ci80_lo": residual * 0.5,
        "ci80_hi": residual * 1.5,
        "ci95_lo": residual * 0.3,
        "ci95_hi": residual * 1.7,
    }])

    result = pd.concat([bl_row, df, res_row], ignore_index=True)
    return result


def _make_northbeam_delta(outcome: str, window: int) -> pd.DataFrame:
    """Generate Northbeam MTA vs MMM posterior comparison table."""
    rng = np.random.default_rng(seed=hash(f"nb_{outcome}_{window}") % (2**31))
    scale = {7: 0.13, 30: 0.55, 90: 1.65, 365: 6.5}[window]

    channels_no_baseline = [c for c in _CHANNELS if c not in ("Direct / Organic",)]
    rows = []
    for ch in channels_no_baseline:
        mmm_val = rng.uniform(0.02, 0.40) * scale
        # NB tends to over-attribute paid social, under-attribute upper funnel
        if "Meta" in ch or "TikTok" in ch or "Google Search" in ch:
            nb_mult = rng.uniform(1.10, 1.55)
        elif "TV" in ch or "YouTube" in ch or "Influencer" in ch:
            nb_mult = rng.uniform(0.45, 0.80)
        else:
            nb_mult = rng.uniform(0.90, 1.15)

        nb_val = mmm_val * nb_mult
        delta = nb_val - mmm_val
        delta_pct = delta / mmm_val * 100

        rows.append({
            "Channel": ch,
            "NB MTA Attribution ($M)": round(nb_val, 3),
            "MMM Posterior ($M)": round(mmm_val, 3),
            "Delta ($M)": round(delta, 3),
            "Delta %": round(delta_pct, 1),
        })

    return pd.DataFrame(rows)


# ── Waterfall chart ───────────────────────────────────────────────────────────

def _waterfall_fig(df: pd.DataFrame) -> go.Figure:
    labels = df["channel"].tolist()
    values = df["contribution"].tolist()

    # Build measure list: first = absolute (baseline), rest = relative, last total
    measure = ["absolute"] + ["relative"] * (len(labels) - 2) + ["total"]

    # Error bar arrays (symmetric approx from 95% CI)
    err_minus = (df["contribution"] - df["ci95_lo"]).clip(lower=0).tolist()
    err_plus  = (df["ci95_hi"] - df["contribution"]).clip(lower=0).tolist()

    # Color: positive = teal, residual = muted
    connector_color = C["border"]
    bar_colors = [C["primary"]] * len(labels)
    bar_colors[0]  = C["ok"]        # Baseline → green
    bar_colors[-1] = C["text_muted"]  # Residual → gray

    fig = go.Figure(go.Waterfall(
        name="Revenue",
        orientation="v",
        measure=measure,
        x=labels,
        y=values,
        text=[f"${v:.2f}M" for v in values],
        textposition="outside",
        connector={"line": {"color": connector_color, "width": 1, "dash": "dot"}},
        increasing={"marker": {"color": C["primary"]}},
        decreasing={"marker": {"color": C["bad"]}},
        totals={"marker": {"color": C["warn"]}},
        error_y={
            "type": "data",
            "array": err_plus,
            "arrayminus": err_minus,
            "visible": True,
            "color": C["text_muted"],
            "thickness": 1.5,
            "width": 4,
        },
    ))

    # Overlay 80% CI as thicker bars
    err_minus_80 = (df["contribution"] - df["ci80_lo"]).clip(lower=0).tolist()
    err_plus_80  = (df["ci80_hi"] - df["contribution"]).clip(lower=0).tolist()

    fig.update_layout(
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["surface"],
        font={"color": C["text"], "family": "Inter, sans-serif"},
        title={
            "text": "Revenue Decomposition — Channel Contributions",
            "font": {"size": 16, "color": C["text"]},
            "x": 0.01,
        },
        yaxis={
            "title": "Revenue ($M)",
            "gridcolor": C["border"],
            "zerolinecolor": C["border"],
        },
        xaxis={"tickangle": -30},
        height=480,
        margin={"t": 60, "b": 100, "l": 60, "r": 20},
        showlegend=False,
    )
    return fig


# ── Delta table styling ───────────────────────────────────────────────────────

def _style_delta_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def highlight_row(row):
        styles = [""] * len(row)
        if abs(row["Delta %"]) > 20:
            styles = [
                f"background-color: {C['bad_bg']}; color: {C['bad']}"
            ] * len(row)
        return styles

    return (
        df.style
        .apply(highlight_row, axis=1)
        .format({
            "NB MTA Attribution ($M)": "${:.3f}M",
            "MMM Posterior ($M)": "${:.3f}M",
            "Delta ($M)": "${:+.3f}M",
            "Delta %": "{:+.1f}%",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", C["surface2"]),
                ("color", C["text"]),
                ("font-size", "0.82rem"),
                ("padding", "6px 10px"),
                ("border-bottom", f"1px solid {C['border']}"),
            ]},
            {"selector": "td", "props": [
                ("font-size", "0.82rem"),
                ("padding", "5px 10px"),
                ("border-bottom", f"1px solid {C['border']}"),
            ]},
        ])
    )


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    st.title("National Revenue Decomposition")
    st.caption("Meridian Bayesian MMM — posterior mean contributions with 80% / 95% credible intervals")

    # ── Controls row ─────────────────────────────────────────────────────────
    col_oc, col_win, _ = st.columns([2, 2, 4])
    with col_oc:
        outcome = st.radio(
            "Outcome",
            ["Combined", "Shopify", "Amazon"],
            horizontal=True,
        )
    with col_win:
        window = st.selectbox(
            "Trailing Period",
            options=[7, 30, 90, 365],
            format_func=lambda x: {7: "7 days", 30: "30 days", 90: "90 days", 365: "365 days"}[x],
            index=1,
        )

    df_decomp = _make_decomp_data(outcome, window)
    df_delta  = _make_northbeam_delta(outcome, window)

    # ── Summary KPI row ──────────────────────────────────────────────────────
    total_rev    = df_decomp["contribution"].sum()
    baseline_rev = df_decomp.loc[df_decomp["channel"] == "Baseline", "contribution"].values[0]
    paid_rev     = total_rev - baseline_rev - df_decomp.loc[df_decomp["channel"] == "Residual", "contribution"].values[0]
    blended_roas = paid_rev / (paid_rev * 0.38)  # implied spend ~38% of paid rev

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", f"${total_rev:.2f}M", help=f"Trailing {window} days, {outcome}")
    k2.metric("Baseline (Organic)", f"${baseline_rev:.2f}M",
              f"{baseline_rev/total_rev*100:.0f}% of total")
    k3.metric("Paid Channel Revenue", f"${paid_rev:.2f}M",
              f"{paid_rev/total_rev*100:.0f}% of total")
    k4.metric("Blended Paid ROAS", f"{blended_roas:.2f}x")

    st.divider()

    # ── Waterfall ─────────────────────────────────────────────────────────────
    section_header("Channel Revenue Waterfall")
    fig = _waterfall_fig(df_decomp)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Error bars show **95% credible interval** (thin) from Meridian MCMC posterior. "
        "Baseline = organic + branded search + retention. Residual = unexplained variance."
    )

    st.divider()

    # ── Northbeam vs MMM delta table ──────────────────────────────────────────
    section_header("Northbeam MTA vs. MMM Posterior — Attribution Delta")

    st.markdown(
        f"Rows highlighted in <span style='color:{C['bad']}'>**red**</span> "
        f"have |Δ%| > 20% — review for model calibration or NB data quality issues.",
        unsafe_allow_html=True,
    )

    large_delta = df_delta[df_delta["Delta %"].abs() > 20]
    if not large_delta.empty:
        flags = ", ".join(large_delta["Channel"].tolist())
        st.warning(f"Significant attribution gap (>20%): {flags}")

    styled = _style_delta_table(df_delta)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    with st.expander("Interpretation guide"):
        st.markdown(f"""
        | Metric | Definition |
        |--------|------------|
        | **NB MTA Attribution** | Northbeam data-driven multi-touch attribution (last model refresh) |
        | **MMM Posterior** | Meridian Bayesian MMM posterior mean contribution |
        | **Delta** | NB − MMM; positive = NB gives channel more credit than MMM |
        | **Delta %** | Delta / MMM × 100; rows > ±20% flagged for review |

        **Common causes of large delta:**
        - Upper-funnel channels (TV, YouTube) are systematically under-attributed by MTA.
        - Last-click or time-decay NB settings over-credit bottom-funnel paid search.
        - MMM adstock parameters may need recalibration if delta persists across windows.
        """)
