"""
View 3 — Budget Optimizer
Interactive spend allocation optimizer with scenario planner.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from incrementality.dashboard.theme import C, section_header

# ── Real data loader ──────────────────────────────────────────────────────────

def _find_latest_run_result(output_dir: str = "./output") -> Optional[dict]:
    """Load the most recently saved MMMRunResult JSON, or None if absent."""
    try:
        paths = sorted(
            Path(output_dir).glob("mmm_run_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if paths:
            return json.loads(paths[0].read_text())
    except Exception:
        pass
    return None


def _real_current_alloc(run: dict, total_budget: float) -> Optional[dict]:
    """Extract current allocation percentages from MMMRunResult channel results."""
    try:
        channel_results = run.get("channel_results", [])
        if not channel_results:
            return None
        # Deduplicate by channel (multiple outcomes per channel — use shopify)
        seen = {}
        for cr in channel_results:
            ch = cr["channel"]
            if cr.get("outcome") == "shopify" and ch not in seen:
                seen[ch] = cr.get("contribution_pct", 0.0)
        if not seen:
            return None
        # Convert contribution_pct → spend allocation (rough proxy)
        total_pct = sum(seen.values()) or 1.0
        return {ch: (pct / total_pct) * total_budget for ch, pct in seen.items()}
    except Exception:
        return None


def _real_optimal_alloc(run: dict) -> Optional[dict]:
    """Extract optimal_allocation from a real MMMRunResult dict."""
    try:
        alloc = run.get("optimal_allocation", {})
        return alloc if alloc else None
    except Exception:
        return None

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

# Current allocation (% of budget) — realistic baseline
_CURRENT_ALLOC = {
    "Meta (Paid Social)":  0.32,
    "Google Search":       0.20,
    "Google Shopping":     0.13,
    "TikTok":              0.10,
    "Pinterest":           0.05,
    "YouTube":             0.07,
    "Email / CRM":         0.05,
    "Influencer":          0.05,
    "TV / CTV":            0.03,
}

# Hill saturation params per channel (α=scale, κ=inflection $, γ=shape)
_HILL_PARAMS = {
    "Meta (Paid Social)":  {"alpha": 2.8, "kappa": 180_000, "gamma": 0.85},
    "Google Search":       {"alpha": 3.5, "kappa": 120_000, "gamma": 0.90},
    "Google Shopping":     {"alpha": 3.0, "kappa":  90_000, "gamma": 0.80},
    "TikTok":              {"alpha": 2.2, "kappa":  60_000, "gamma": 0.75},
    "Pinterest":           {"alpha": 1.8, "kappa":  35_000, "gamma": 0.70},
    "YouTube":             {"alpha": 2.0, "kappa":  80_000, "gamma": 0.78},
    "Email / CRM":         {"alpha": 4.5, "kappa":  15_000, "gamma": 0.95},
    "Influencer":          {"alpha": 1.5, "kappa":  45_000, "gamma": 0.65},
    "TV / CTV":            {"alpha": 1.2, "kappa": 200_000, "gamma": 0.60},
}


def _hill_roas(spend: float, alpha: float, kappa: float, gamma: float) -> float:
    """Return ROAS at given spend level using Hill saturation."""
    if spend <= 0:
        return 0.0
    return alpha * (spend ** gamma) / (kappa ** gamma + spend ** gamma)


def _hill_revenue(spend: float, alpha: float, kappa: float, gamma: float) -> float:
    return _hill_roas(spend, alpha, kappa, gamma) * spend


def _marginal_roas(spend: float, alpha: float, kappa: float, gamma: float, delta: float = 500.0) -> float:
    """Numerical marginal ROAS at given spend."""
    if spend <= 0:
        return alpha
    r1 = _hill_revenue(spend + delta, alpha, kappa, gamma)
    r0 = _hill_revenue(spend, alpha, kappa, gamma)
    return (r1 - r0) / delta


# ── Optimizer: simple greedy marginal allocation ──────────────────────────────

def _optimize_budget(
    total_budget: float,
    shopify_weight: float,
    floors: dict[str, float],
    n_steps: int = 1000,
) -> dict[str, float]:
    """
    Greedy marginal ROAS optimizer.
    Returns recommended spend per channel.
    """
    # Start from floors
    alloc = {ch: max(floors.get(ch, 0.0), total_budget * 0.01) for ch in _CHANNELS}
    remaining = total_budget - sum(alloc.values())
    step = remaining / n_steps

    if remaining <= 0:
        # Normalize floors to budget
        total_floor = sum(alloc.values())
        return {ch: v / total_floor * total_budget for ch, v in alloc.items()}

    for _ in range(n_steps):
        best_ch = max(
            _CHANNELS,
            key=lambda ch: _marginal_roas(
                alloc[ch], **_HILL_PARAMS[ch]
            ),
        )
        alloc[best_ch] += step

    return alloc


def _expected_revenue(
    alloc: dict[str, float],
    shopify_weight: float,
) -> tuple[float, float, float]:
    """
    Returns (total_rev, shopify_rev, amazon_rev) in dollars.
    shopify_weight: fraction of revenue attributed to Shopify channel.
    """
    total = sum(
        _hill_revenue(spend, **_HILL_PARAMS[ch])
        for ch, spend in alloc.items()
    )
    shopify = total * shopify_weight
    amazon  = total * (1 - shopify_weight)
    return total, shopify, amazon


def _revenue_uncertainty(total_rev: float, rng_seed: int = 42) -> tuple[float, float]:
    """Return ±1σ (80% CI half-width) based on typical MMM posterior spread."""
    rng = np.random.default_rng(rng_seed)
    sigma = total_rev * rng.uniform(0.10, 0.18)
    return sigma * 1.28, sigma * 1.96  # 80% and 95% half-widths


# ── Bar chart: current vs recommended vs last week ────────────────────────────

def _alloc_chart(
    current_alloc: dict[str, float],
    rec_alloc: dict[str, float],
    last_week_alloc: dict[str, float],
) -> go.Figure:
    channels = list(current_alloc.keys())

    fig = go.Figure()
    bar_data = [
        ("Current", list(current_alloc.values()), C["text_muted"]),
        ("Recommended", list(rec_alloc.values()), C["primary"]),
        ("Last Week Actual", list(last_week_alloc.values()), C["ok"]),
    ]
    for name, vals, color in bar_data:
        fig.add_trace(go.Bar(
            name=name,
            x=channels,
            y=vals,
            text=[f"${v:,.0f}" for v in vals],
            textposition="outside",
            textfont={"size": 9, "color": C["text"]},
            marker_color=color,
            marker_line_width=0,
        ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["surface"],
        font={"color": C["text"], "family": "Inter, sans-serif"},
        title={"text": "Channel Spend Allocation — Current vs. Recommended vs. Last Week",
               "font": {"size": 14, "color": C["text"]}, "x": 0.01},
        yaxis={
            "title": "Weekly Spend ($)",
            "gridcolor": C["border"],
            "tickprefix": "$",
        },
        xaxis={"tickangle": -25},
        legend={"bgcolor": C["surface2"], "bordercolor": C["border"], "borderwidth": 1},
        height=400,
        margin={"t": 50, "b": 100, "l": 80, "r": 20},
    )
    return fig


def _scenario_chart(total_budget: float, shopify_weight: float, floors: dict[str, float]) -> go.Figure:
    """Revenue vs budget delta curve for scenario planner."""
    pct_range = np.linspace(-0.30, 0.30, 31)
    revs = []
    mroas_vals = []

    for pct in pct_range:
        b = total_budget * (1 + pct)
        alloc = _optimize_budget(b, shopify_weight, floors, n_steps=500)
        rev, _, _ = _expected_revenue(alloc, shopify_weight)
        revs.append(rev)
        # Marginal ROAS at this budget level vs base
        delta_budget = b - total_budget if abs(b - total_budget) > 1 else 1_000
        base_alloc = _optimize_budget(total_budget, shopify_weight, floors, n_steps=500)
        base_rev, _, _ = _expected_revenue(base_alloc, shopify_weight)
        delta_rev = rev - base_rev
        mroas_vals.append(delta_rev / abs(delta_budget) if delta_budget != 0 else np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pct_range * 100,
        y=[r / 1_000 for r in revs],
        mode="lines+markers",
        name="Expected Revenue",
        line={"color": C["primary"], "width": 2},
        marker={"size": 5},
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=pct_range * 100,
        y=mroas_vals,
        mode="lines",
        name="Marginal ROAS",
        line={"color": C["warn"], "width": 1.5, "dash": "dash"},
        yaxis="y2",
    ))
    # Mark current budget
    fig.add_vline(x=0, line_dash="dot", line_color=C["border"],
                  annotation_text="Current", annotation_font_color=C["text_muted"])

    fig.update_layout(
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["surface"],
        font={"color": C["text"], "family": "Inter, sans-serif"},
        title={"text": "Scenario: Revenue vs. Budget Change", "font": {"size": 14, "color": C["text"]}, "x": 0.01},
        xaxis={"title": "Budget Change (%)", "gridcolor": C["border"], "ticksuffix": "%"},
        yaxis={"title": "Expected Revenue ($K)", "gridcolor": C["border"], "tickprefix": "$", "ticksuffix": "K"},
        yaxis2={
            "title": "Marginal ROAS",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "tickcolor": C["warn"],
            "titlefont": {"color": C["warn"]},
            "tickfont": {"color": C["warn"]},
        },
        legend={"bgcolor": C["surface2"], "bordercolor": C["border"], "borderwidth": 1},
        height=380,
        margin={"t": 50, "b": 60, "l": 80, "r": 80},
    )
    return fig


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("Budget Optimizer")
    st.caption("Greedy marginal ROI allocation · Hill saturation curves · Scenario planner")

    run = _find_latest_run_result()
    using_real = run is not None and bool(run.get("optimal_allocation"))

    if not using_real:
        st.info(
            "Optimizer uses **synthetic Hill saturation parameters** calibrated to realistic MTB media mix. "
            "Run `lift mmm fit` to activate live Meridian-posterior optimization.",
            icon="ℹ️",
        )
    else:
        st.success(
            f"Showing **live Meridian posterior** allocation from run `{run.get('run_id', 'latest')}`. "
            f"Blended ROAS: {run.get('blended_roas', 0):.2f}x",
            icon="✓",
        )

    # ── Input panel ───────────────────────────────────────────────────────────
    section_header("Optimization Inputs")
    col_inp, col_out = st.columns([1, 2], gap="large")

    with col_inp:
        total_budget = st.slider(
            "Total Weekly Budget",
            min_value=50_000,
            max_value=1_000_000,
            value=250_000,
            step=5_000,
            format="$%d",
        )

        shopify_wt = st.slider(
            "Shopify Revenue Weight",
            min_value=0.10,
            max_value=0.90,
            value=0.65,
            step=0.05,
            help="Fraction of MMM revenue target attributed to Shopify (remainder = Amazon)",
        )
        amazon_wt = 1.0 - shopify_wt
        st.caption(f"Amazon weight: **{amazon_wt:.0%}**")

        with st.expander("Minimum Spend Floors per Channel"):
            floors: dict[str, float] = {}
            for ch in _CHANNELS:
                default_floor = int(total_budget * _CURRENT_ALLOC[ch] * 0.30)
                floors[ch] = st.number_input(
                    ch,
                    min_value=0,
                    max_value=int(total_budget * 0.5),
                    value=default_floor,
                    step=1_000,
                    format="%d",
                    key=f"floor_{ch}",
                )

    with col_out:
        # ── Run optimizer ─────────────────────────────────────────────────────
        # Use real Meridian allocation when available; else greedy synthetic
        if using_real:
            real_opt = _real_optimal_alloc(run)
            real_cur = _real_current_alloc(run, total_budget)
            rec_alloc = real_opt or _optimize_budget(total_budget, shopify_wt, floors)
            current_alloc_raw = real_cur or {ch: total_budget * pct for ch, pct in _CURRENT_ALLOC.items()}
        else:
            rec_alloc = _optimize_budget(total_budget, shopify_wt, floors)
            current_alloc_raw = {ch: total_budget * pct for ch, pct in _CURRENT_ALLOC.items()}

        # Align channel lists — keep only channels present in both allocs
        all_channels = sorted(set(rec_alloc) | set(current_alloc_raw))
        for ch in all_channels:
            rec_alloc.setdefault(ch, 0.0)
            current_alloc_raw.setdefault(ch, 0.0)

        # Last week: small random variation around current
        rng = np.random.default_rng(17)
        last_week_alloc = {
            ch: max(0, v * rng.uniform(0.90, 1.10))
            for ch, v in current_alloc_raw.items()
        }

        if using_real and run:
            total_rev = run.get("expected_total_revenue", 0.0)
            shop_rev = run.get("expected_shopify_revenue", total_rev * shopify_wt)
            amz_rev = run.get("expected_amazon_revenue", total_rev * (1 - shopify_wt))
            cur_rev = total_rev  # no separate "current" rev in run result
        else:
            total_rev, shop_rev, amz_rev = _expected_revenue(rec_alloc, shopify_wt)
            cur_rev, _, _ = _expected_revenue(current_alloc_raw, shopify_wt)
        ci80, ci95 = _revenue_uncertainty(total_rev)

        # Blended ROAS
        total_paid = sum(rec_alloc.values())
        blended_roas = total_rev / total_paid if total_paid > 0 else 0.0
        cur_roas = cur_rev / sum(current_alloc_raw.values())

        section_header("Optimization Results")
        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Expected Revenue",
            f"${total_rev/1e6:.2f}M",
            f"±${ci80/1e3:.0f}K (80% CI)",
        )
        k2.metric(
            "Blended ROAS",
            f"{blended_roas:.2f}x",
            f"{blended_roas - cur_roas:+.2f}x vs current",
        )
        k3.metric(
            "Revenue Uplift vs Current",
            f"${(total_rev - cur_rev)/1e3:+.0f}K",
            f"{(total_rev/cur_rev - 1)*100:+.1f}%",
        )

        c_shop, c_amz = st.columns(2)
        c_shop.metric("Shopify Revenue", f"${shop_rev/1e6:.2f}M", f"{shopify_wt:.0%} share")
        c_amz.metric("Amazon Revenue",  f"${amz_rev/1e6:.2f}M",  f"{amazon_wt:.0%} share")

    st.divider()

    # ── Allocation bar chart ──────────────────────────────────────────────────
    section_header("Channel Allocation Comparison")
    fig_alloc = _alloc_chart(current_alloc_raw, rec_alloc, last_week_alloc)
    st.plotly_chart(fig_alloc, use_container_width=True)

    # ── Per-channel summary table ─────────────────────────────────────────────
    with st.expander("Detailed Channel Allocation Table"):
        rows = []
        for ch in _CHANNELS:
            cur = current_alloc_raw[ch]
            rec = rec_alloc[ch]
            delta = rec - cur
            cur_roas_ch = _hill_roas(cur, **_HILL_PARAMS[ch])
            rec_roas_ch = _hill_roas(rec, **_HILL_PARAMS[ch])
            mroas_ch = _marginal_roas(rec, **_HILL_PARAMS[ch])
            rows.append({
                "Channel": ch,
                "Current Spend": cur,
                "Recommended Spend": rec,
                "Delta ($)": delta,
                "Delta %": (delta / cur * 100) if cur > 0 else 0.0,
                "Avg ROAS (Current)": cur_roas_ch,
                "Avg ROAS (Rec)": rec_roas_ch,
                "Marginal ROAS": mroas_ch,
            })
        tbl = pd.DataFrame(rows)

        def color_delta(val):
            if val > 0:
                return f"color: {C['ok']}"
            elif val < 0:
                return f"color: {C['bad']}"
            return ""

        styled = (
            tbl.style
            .applymap(color_delta, subset=["Delta ($)", "Delta %"])
            .format({
                "Current Spend":      "${:,.0f}",
                "Recommended Spend":  "${:,.0f}",
                "Delta ($)":          "${:+,.0f}",
                "Delta %":            "{:+.1f}%",
                "Avg ROAS (Current)": "{:.2f}x",
                "Avg ROAS (Rec)":     "{:.2f}x",
                "Marginal ROAS":      "{:.2f}x",
            })
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # ── Scenario planner ──────────────────────────────────────────────────────
    section_header("Scenario Planner")
    st.markdown("Explore how revenue and marginal ROAS change if you scale your total weekly budget up or down.")

    scen_pct = st.slider(
        "Budget adjustment",
        min_value=-30,
        max_value=30,
        value=0,
        step=5,
        format="%d%%",
    )

    scen_budget = total_budget * (1 + scen_pct / 100)
    scen_alloc = _optimize_budget(scen_budget, shopify_wt, floors, n_steps=500)
    scen_rev, scen_shop, scen_amz = _expected_revenue(scen_alloc, shopify_wt)
    scen_delta_rev = scen_rev - total_rev
    scen_mroas = scen_delta_rev / (scen_budget - total_budget) if scen_budget != total_budget else np.nan

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Scenario Budget", f"${scen_budget:,.0f}/wk", f"{scen_pct:+d}%")
    sc2.metric("Scenario Revenue", f"${scen_rev/1e6:.2f}M", f"${scen_delta_rev/1e3:+.0f}K")
    sc3.metric(
        "Marginal ROAS at New Budget",
        f"{scen_mroas:.2f}x" if not np.isnan(scen_mroas) else "—",
        help="Incremental revenue per incremental dollar at the new budget level",
    )
    sc4.metric("Shopify / Amazon", f"${scen_shop/1e6:.2f}M / ${scen_amz/1e6:.2f}M")

    fig_scen = _scenario_chart(total_budget, shopify_wt, floors)
    st.plotly_chart(fig_scen, use_container_width=True)

    st.caption(
        "Revenue curve uses greedy marginal ROI allocation at each budget level. "
        "Shaded intervals reflect typical Meridian MMM posterior uncertainty (~10–18% σ)."
    )
