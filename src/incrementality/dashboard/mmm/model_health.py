"""
View 4 — Model Health
R-hat convergence, posterior predictive check, time-varying betas,
calibration status, and validation gate indicators.
"""
from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from incrementality.dashboard.theme import C, CHART_PALETTE, section_header

# ── Constants ──────────────────────────────────────────────────────────────────

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
]

_ADSTOCK_PARAMS = [f"adstock_decay[{ch}]" for ch in _CHANNELS]
_SAT_PARAMS     = [f"sat_slope[{ch}]" for ch in _CHANNELS] + [f"sat_ec[{ch}]" for ch in _CHANNELS]
_BASELINE_PARAMS = ["intercept", "trend_slope", "seasonality_amp", "day_of_week[0]",
                    "day_of_week[1]", "day_of_week[2]", "noise_sigma"]
_BETA_PARAMS    = [f"beta[{ch}]" for ch in _CHANNELS]


# ── Synthetic data generators ──────────────────────────────────────────────────

def _make_rhat_table() -> pd.DataFrame:
    rng = np.random.default_rng(42)

    all_params = (
        [(f"beta[{ch}]", "Channel β") for ch in _CHANNELS]
        + [(f"adstock_decay[{ch}]", "Adstock") for ch in _CHANNELS]
        + [(f"sat_slope[{ch}]", "Saturation γ") for ch in _CHANNELS]
        + [(f"sat_ec[{ch}]", "Saturation κ") for ch in _CHANNELS]
        + [(p, "Baseline") for p in _BASELINE_PARAMS]
    )

    rows = []
    for param, group in all_params:
        # Most params converge; inject a couple of marginal/bad ones
        if "TV" in param and "beta" in param:
            rhat = rng.uniform(1.06, 1.12)
        elif "Influencer" in param and "adstock" in param:
            rhat = rng.uniform(1.03, 1.08)
        elif "noise_sigma" in param:
            rhat = rng.uniform(1.00, 1.02)
        else:
            rhat = rng.uniform(1.00, 1.04)

        n_eff = int(rng.uniform(300, 2000))
        rows.append({
            "Parameter": param,
            "Group": group,
            "R-hat": round(rhat, 4),
            "N_eff": n_eff,
            "Status": (
                "OK"      if rhat < 1.05 else
                "Warning" if rhat < 1.10 else
                "Bad"
            ),
        })

    return pd.DataFrame(rows)


def _make_ppc_data(n_days: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    dates = pd.date_range(end=dt.date(2026, 2, 17), periods=n_days, freq="D")

    # Shopify
    shop_actual = 80_000 + rng.normal(0, 8_000, n_days).cumsum() * 0.1 + \
                  15_000 * np.sin(np.linspace(0, 2 * np.pi, n_days))
    shop_actual = np.clip(shop_actual, 40_000, 180_000)
    shop_pred   = shop_actual * rng.uniform(0.88, 1.12, n_days)
    shop_lo80   = shop_pred * 0.88
    shop_hi80   = shop_pred * 1.12
    shop_lo95   = shop_pred * 0.80
    shop_hi95   = shop_pred * 1.20

    # Amazon
    amz_actual  = 40_000 + rng.normal(0, 5_000, n_days).cumsum() * 0.05 + \
                  8_000 * np.sin(np.linspace(np.pi / 4, 2.25 * np.pi, n_days))
    amz_actual  = np.clip(amz_actual, 20_000, 90_000)
    amz_pred    = amz_actual * rng.uniform(0.90, 1.10, n_days)
    amz_lo80    = amz_pred * 0.87
    amz_hi80    = amz_pred * 1.13
    amz_lo95    = amz_pred * 0.79
    amz_hi95    = amz_pred * 1.21

    return pd.DataFrame({
        "date": dates,
        "shop_actual": shop_actual.astype(int),
        "shop_pred":   shop_pred.astype(int),
        "shop_lo80":   shop_lo80.astype(int),
        "shop_hi80":   shop_hi80.astype(int),
        "shop_lo95":   shop_lo95.astype(int),
        "shop_hi95":   shop_hi95.astype(int),
        "amz_actual":  amz_actual.astype(int),
        "amz_pred":    amz_pred.astype(int),
        "amz_lo80":    amz_lo80.astype(int),
        "amz_hi80":    amz_hi80.astype(int),
        "amz_lo95":    amz_lo95.astype(int),
        "amz_hi95":    amz_hi95.astype(int),
    })


def _make_beta_timeseries(n_weeks: int = 52) -> pd.DataFrame:
    rng = np.random.default_rng(77)
    dates = pd.date_range(end=dt.date(2026, 2, 17), periods=n_weeks, freq="W")

    rows = {"date": dates}
    for ch in _CHANNELS:
        base_beta = rng.uniform(0.4, 2.5)
        trend = rng.uniform(-0.005, 0.003)
        noise = rng.normal(0, 0.03, n_weeks)
        seasonal = 0.1 * np.sin(np.linspace(0, 4 * np.pi, n_weeks))
        betas = base_beta + trend * np.arange(n_weeks) + noise + seasonal
        rows[ch] = np.clip(betas, 0.05, 4.0).round(4)

    return pd.DataFrame(rows)


def _make_calibration_table() -> pd.DataFrame:
    today = dt.date(2026, 2, 17)
    rows = []
    calibration_data = [
        ("Meta (Paid Social)",   dt.date(2026, 1,  8), "Holdout",    True),
        ("Google Search",        dt.date(2026, 1, 22), "NB iROAS",   True),
        ("Google Shopping",      dt.date(2025, 12, 15), "Holdout",   True),
        ("TikTok",               dt.date(2026, 2,  1), "NB iROAS",   True),
        ("Pinterest",            dt.date(2025, 11, 10), "NB iROAS",  False),
        ("YouTube",              dt.date(2026, 1, 15), "Holdout",    True),
        ("Email / CRM",          dt.date(2025, 10,  5), "Holdout",   False),
        ("Influencer",           dt.date(2026, 2,  3), "NB iROAS",   True),
        ("TV / CTV",             dt.date(2025, 12,  1), "Holdout",   True),
    ]
    for ch, last_date, source, status_ok in calibration_data:
        days_since = (today - last_date).days
        rows.append({
            "Channel": ch,
            "Last Geo Holdout Date": last_date.strftime("%Y-%m-%d"),
            "Days Since Calibration": days_since,
            "Prior Source": source,
            "Status": "Current" if status_ok else "Stale",
        })
    return pd.DataFrame(rows)


# ── R-hat table styling ────────────────────────────────────────────────────────

def _style_rhat(df: pd.DataFrame):
    def color_rhat(val):
        if val < 1.05:
            return f"color: {C['ok']}; font-weight: 700"
        elif val < 1.10:
            return f"color: {C['warn']}; font-weight: 700"
        else:
            return f"color: {C['bad']}; font-weight: 700"

    def color_status(val):
        cmap = {"OK": C["ok"], "Warning": C["warn"], "Bad": C["bad"]}
        c = cmap.get(val, C["text_muted"])
        return f"color: {c}; font-weight: 600"

    return (
        df.style
        .applymap(color_rhat, subset=["R-hat"])
        .applymap(color_status, subset=["Status"])
        .format({"R-hat": "{:.4f}", "N_eff": "{:,.0f}"})
        .bar(subset=["R-hat"], color=[C["ok_bg"], C["bad_bg"]], vmin=1.0, vmax=1.15)
    )


# ── PPC chart ─────────────────────────────────────────────────────────────────

def _ppc_fig(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["Shopify", "Amazon"],
        vertical_spacing=0.12,
    )

    for row, (actual_col, pred_col, lo80_col, hi80_col, lo95_col, hi95_col, label, color) in enumerate([
        ("shop_actual", "shop_pred", "shop_lo80", "shop_hi80", "shop_lo95", "shop_hi95", "Shopify", C["primary"]),
        ("amz_actual",  "amz_pred",  "amz_lo80",  "amz_hi80",  "amz_lo95",  "amz_hi95",  "Amazon",  C["ok"]),
    ], start=1):
        # 95% CI band
        fig.add_trace(go.Scatter(
            x=list(df["date"]) + list(df["date"])[::-1],
            y=list(df[hi95_col]) + list(df[lo95_col])[::-1],
            fill="toself",
            fillcolor=f"{color}18",
            line={"width": 0},
            showlegend=(row == 1),
            name="95% CI",
            legendgroup="95ci",
        ), row=row, col=1)

        # 80% CI band
        fig.add_trace(go.Scatter(
            x=list(df["date"]) + list(df["date"])[::-1],
            y=list(df[hi80_col]) + list(df[lo80_col])[::-1],
            fill="toself",
            fillcolor=f"{color}30",
            line={"width": 0},
            showlegend=(row == 1),
            name="80% CI",
            legendgroup="80ci",
        ), row=row, col=1)

        # Predicted line
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[pred_col],
            mode="lines",
            name=f"MMM Predicted ({label})",
            line={"color": color, "width": 2},
            legendgroup=f"pred{label}",
        ), row=row, col=1)

        # Actual line
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[actual_col],
            mode="lines+markers",
            name=f"Actual ({label})",
            line={"color": C["text"], "width": 1.5, "dash": "dot"},
            marker={"size": 3, "color": C["text"]},
            legendgroup=f"act{label}",
        ), row=row, col=1)

    fig.update_layout(
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["surface"],
        font={"color": C["text"], "family": "Inter, sans-serif"},
        height=520,
        margin={"t": 60, "b": 40, "l": 80, "r": 20},
        legend={"bgcolor": C["surface2"], "bordercolor": C["border"], "borderwidth": 1, "x": 1.01, "y": 1},
    )
    for i in range(1, 3):
        fig.update_xaxes(gridcolor=C["border"], row=i, col=1)
        fig.update_yaxes(gridcolor=C["border"], tickprefix="$", row=i, col=1)
    # Update subplot title colors
    for ann in fig.layout.annotations:
        ann.font.color = C["text_secondary"]
    return fig


# ── Beta time-series chart ─────────────────────────────────────────────────────

def _beta_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, ch in enumerate(_CHANNELS):
        color = CHART_PALETTE[i % len(CHART_PALETTE)]
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df[ch],
            mode="lines",
            name=ch,
            line={"color": color, "width": 1.8},
        ))

    fig.update_layout(
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["surface"],
        font={"color": C["text"], "family": "Inter, sans-serif"},
        title={"text": "Time-Varying Channel Effectiveness (β) — Trailing 52 Weeks",
               "font": {"size": 14, "color": C["text"]}, "x": 0.01},
        yaxis={"title": "β Coefficient", "gridcolor": C["border"]},
        xaxis={"gridcolor": C["border"]},
        legend={"bgcolor": C["surface2"], "bordercolor": C["border"], "borderwidth": 1},
        height=400,
        margin={"t": 60, "b": 40, "l": 70, "r": 20},
    )
    return fig


def _beta_alerts(df: pd.DataFrame) -> list[str]:
    """Return list of channels where beta dropped >25% vs 90-day avg."""
    alerts = []
    last_13 = df.tail(13)  # ~13 weeks ≈ 90 days
    last_1  = df.tail(1)

    for ch in _CHANNELS:
        avg_90 = last_13[ch].mean()
        current = last_1[ch].values[0]
        if avg_90 > 0 and (avg_90 - current) / avg_90 > 0.25:
            drop_pct = (avg_90 - current) / avg_90 * 100
            alerts.append(f"{ch}: β dropped {drop_pct:.0f}% vs 90-day avg (current={current:.3f}, avg={avg_90:.3f})")

    return alerts


# ── Calibration table styling ──────────────────────────────────────────────────

def _style_calibration(df: pd.DataFrame):
    def color_days(val):
        if val <= 30:
            return f"color: {C['ok']}"
        elif val <= 60:
            return f"color: {C['warn']}"
        else:
            return f"color: {C['bad']}"

    def color_status(val):
        cmap = {"Current": C["ok"], "Stale": C["bad"]}
        c = cmap.get(val, C["text_muted"])
        return f"color: {c}; font-weight: 600"

    return (
        df.style
        .applymap(color_days, subset=["Days Since Calibration"])
        .applymap(color_status, subset=["Status"])
    )


# ── Validation gate badges ─────────────────────────────────────────────────────

def _gate_badge(label: str, passed: bool, detail: str) -> str:
    c = C["ok"] if passed else C["bad"]
    bg = C["ok_bg"] if passed else C["bad_bg"]
    symbol = "PASS" if passed else "FAIL"
    return f"""
    <div style="background:{bg};border:1px solid {c};border-radius:8px;
                padding:12px 16px;margin:4px 0;">
        <span style="color:{c};font-weight:700;font-size:1.0rem;">{symbol}</span>
        <span style="color:{C['text']};font-weight:600;margin-left:8px;">{label}</span>
        <div style="color:{C['text_muted']};font-size:0.78rem;margin-top:4px;">{detail}</div>
    </div>
    """


# ── Model metadata panel ───────────────────────────────────────────────────────

def _metadata_panel() -> None:
    meta = {
        "Last Refit":       "2026-02-16 03:14 UTC",
        "Training Days":    "730 days (2024-02-17 – 2026-02-16)",
        "N DMAs":           "210",
        "Sampler":          "NUTS (JAX backend)",
        "Chains":           "4",
        "Warmup / Samples": "1,000 / 1,000 per chain",
        "Total Draws":      "4,000",
        "Model Framework":  "Meridian v1.2.1",
        "Calibration":      "Northbeam iROAS priors + geo holdouts",
    }
    cols = st.columns(3)
    items = list(meta.items())
    for i, col in enumerate(cols):
        for label, val in items[i * 3 : (i + 1) * 3]:
            col.markdown(
                f"<div style='margin-bottom:8px;'>"
                f"<span style='color:{C['text_muted']};font-size:0.78rem;'>{label}</span><br>"
                f"<span style='color:{C['text']};font-size:0.9rem;font-weight:600;'>{val}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("Model Health")
    st.caption("MCMC convergence · posterior predictive checks · calibration status · validation gates")

    # ── Model metadata ────────────────────────────────────────────────────────
    section_header("Model Metadata")
    _metadata_panel()

    st.divider()

    # ── Validation gates ──────────────────────────────────────────────────────
    section_header("Validation Gates")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(
            _gate_badge(
                "Gate 1: MCMC Convergence",
                passed=True,
                detail="All R-hat < 1.05 (except TV β = 1.09 — within tolerance). "
                       "N_eff > 300 for all parameters.",
            ),
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            _gate_badge(
                "Gate 2: Posterior Predictive",
                passed=True,
                detail="Shopify MAPE = 6.2% | Amazon MAPE = 8.7%. "
                       "Both below 10% threshold. 95% CI coverage = 94%.",
            ),
            unsafe_allow_html=True,
        )
    with g3:
        st.markdown(
            _gate_badge(
                "Gate 3: Calibration",
                passed=False,
                detail="Pinterest NB prior stale (99 days). Email holdout stale (135 days). "
                       "2 of 9 channels require recalibration.",
            ),
            unsafe_allow_html=True,
        )

    st.divider()

    # ── R-hat convergence ─────────────────────────────────────────────────────
    section_header("R-hat Convergence Diagnostics")

    rhat_df = _make_rhat_table()
    bad_count  = (rhat_df["Status"] == "Bad").sum()
    warn_count = (rhat_df["Status"] == "Warning").sum()
    ok_count   = (rhat_df["Status"] == "OK").sum()

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Converged (R-hat < 1.05)", f"{ok_count}/{len(rhat_df)}", help="Parameters with R-hat below 1.05")
    rc2.metric("Warning (1.05 – 1.10)", f"{warn_count}", help="Monitor these parameters")
    rc3.metric("Diverged (> 1.10)", f"{bad_count}", help="Consider reconfiguring sampler or priors")

    group_filter = st.multiselect(
        "Filter by parameter group",
        options=rhat_df["Group"].unique().tolist(),
        default=rhat_df["Group"].unique().tolist(),
    )
    filtered_rhat = rhat_df[rhat_df["Group"].isin(group_filter)]
    st.dataframe(_style_rhat(filtered_rhat), use_container_width=True, hide_index=True)

    st.divider()

    # ── Posterior predictive check ────────────────────────────────────────────
    section_header("Posterior Predictive Check — Last 30 Days")

    ppc_df = _make_ppc_data(30)

    # MAPE calculation
    shop_mape = np.mean(np.abs((ppc_df["shop_actual"] - ppc_df["shop_pred"]) / ppc_df["shop_actual"])) * 100
    amz_mape  = np.mean(np.abs((ppc_df["amz_actual"]  - ppc_df["amz_pred"])  / ppc_df["amz_actual"]))  * 100

    pm1, pm2, pm3 = st.columns(3)
    pm1.metric(
        "Shopify MAPE",
        f"{shop_mape:.1f}%",
        "Good" if shop_mape < 10 else "Review",
        delta_color="inverse",
    )
    pm2.metric(
        "Amazon MAPE",
        f"{amz_mape:.1f}%",
        "Good" if amz_mape < 10 else "Review",
        delta_color="inverse",
    )
    pm3.metric(
        "95% CI Coverage",
        "93.3%",
        help="Fraction of actual observations falling within the 95% posterior predictive interval (target ≥ 90%)",
    )

    fig_ppc = _ppc_fig(ppc_df)
    st.plotly_chart(fig_ppc, use_container_width=True)

    st.divider()

    # ── Time-varying betas ────────────────────────────────────────────────────
    section_header("Time-Varying Channel Effectiveness (β) — Trailing 52 Weeks")

    beta_df = _make_beta_timeseries(52)
    alerts = _beta_alerts(beta_df)

    if alerts:
        with st.container():
            for alert in alerts:
                st.warning(f"β Drop Alert: {alert}", icon="⚠️")
    else:
        st.success("No channel β values have dropped >25% vs 90-day average.", icon="✓")

    fig_beta = _beta_fig(beta_df)
    st.plotly_chart(fig_beta, use_container_width=True)

    with st.expander("Beta interpretation"):
        st.markdown("""
        **β (beta)** represents the effectiveness of each channel's spend in driving revenue,
        after accounting for adstock carry-over and saturation.

        - **Rising β** → channel becoming more efficient (e.g. improved creative, better targeting)
        - **Falling β** → channel fatigue, audience saturation, or competitive pressure
        - **Alert threshold:** >25% drop vs trailing 90-day average triggers review

        Time-varying betas are estimated using a Gaussian random walk prior over weekly periods.
        """)

    st.divider()

    # ── Calibration status ────────────────────────────────────────────────────
    section_header("Calibration Status")
    calib_df = _make_calibration_table()
    styled_calib = _style_calibration(calib_df)
    st.dataframe(styled_calib, use_container_width=True, hide_index=True)

    st.markdown(
        f"Channels with **Days Since Calibration > 60** "
        f"(highlighted in <span style='color:{C['bad']}'>red</span>) "
        f"should be prioritized for geo holdout experiments or NB iROAS prior refresh.",
        unsafe_allow_html=True,
    )

    with st.expander("Calibration methodology"):
        st.markdown("""
        **Calibration sources:**

        | Source | Method | Frequency |
        |--------|--------|-----------|
        | **Geo Holdout** | Randomized matched-market test; holdout DMA group withholds spend | Per campaign or quarterly |
        | **NB iROAS** | Northbeam incremental ROAS estimate used as Bayesian prior mean | Rolling — NB refreshes weekly |

        **Recalibration triggers:**
        - Days since last holdout > 60 days
        - Channel β drops >25% vs 90-day avg (possible prior drift)
        - Delta between NB MTA and MMM posterior exceeds 30%
        - Major creative/targeting change on channel (flag manually)
        """)
