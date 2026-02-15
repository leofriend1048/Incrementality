"""LIFT Dashboard — Brand theme, color palette, and reusable components.

Provides the visual identity for the Streamlit dashboard,
matching the LIFT CLI's teal/cyan branding with a dark, professional look.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# ── Brand Identity ────────────────────────────────────────────────────────────

BRAND = "LIFT"
MARK = "▲"
TAGLINE = "Geo Incrementality Platform"
VERSION = "0.3.0"

# ── Color Palette ─────────────────────────────────────────────────────────────

C = {
    # Backgrounds
    "bg": "#0B1120",
    "surface": "#111827",
    "surface2": "#1F2937",
    "border": "#374151",
    "border_light": "#4B5563",
    # Text
    "text": "#F8FAFC",
    "text_secondary": "#CBD5E1",
    "text_muted": "#94A3B8",
    # Brand
    "primary": "#06B6D4",
    "primary_dark": "#0891B2",
    "primary_light": "#22D3EE",
    "primary_bg": "rgba(6, 182, 212, 0.10)",
    # Semantic
    "ok": "#34D399",
    "ok_dark": "#059669",
    "ok_bg": "rgba(52, 211, 153, 0.10)",
    "warn": "#FBBF24",
    "warn_dark": "#D97706",
    "warn_bg": "rgba(251, 191, 36, 0.10)",
    "bad": "#F87171",
    "bad_dark": "#DC2626",
    "bad_bg": "rgba(248, 113, 113, 0.10)",
    # Chart accents
    "purple": "#A78BFA",
    "pink": "#F472B6",
    "sky": "#38BDF8",
    "orange": "#FB923C",
}

CHART_PALETTE = [
    C["primary"], C["ok"], C["warn"], C["bad"],
    C["purple"], C["pink"], C["sky"], C["orange"],
]


# ── Plotly Template ───────────────────────────────────────────────────────────

def _build_plotly_template() -> go.layout.Template:
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font=dict(color=C["text_secondary"], family="Inter, system-ui, sans-serif", size=13),
        title=dict(font=dict(color=C["text"], size=16)),
        xaxis=dict(
            gridcolor=C["border"], zerolinecolor=C["border"],
            tickfont=dict(color=C["text_muted"]),
        ),
        yaxis=dict(
            gridcolor=C["border"], zerolinecolor=C["border"],
            tickfont=dict(color=C["text_muted"]),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(color=C["text_secondary"]),
        ),
        colorway=CHART_PALETTE,
        margin=dict(l=50, r=20, t=50, b=40),
        hoverlabel=dict(
            bgcolor=C["surface2"], font_color=C["text"],
            bordercolor=C["border"],
        ),
    )
    return tpl


def setup_plotly():
    """Register the LIFT Plotly template as default."""
    pio.templates["lift"] = _build_plotly_template()
    pio.templates.default = "lift"


# ── CSS Injection ─────────────────────────────────────────────────────────────

def inject_css():
    """Inject LIFT brand CSS into the Streamlit page."""
    st.markdown(f"""
    <style>
    /* ── Global ────────────────────────────────── */
    .stApp {{
        background-color: {C["bg"]};
    }}
    .stApp header {{
        background-color: {C["bg"]} !important;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {C["surface"]};
        border-right: 1px solid {C["border"]};
    }}
    section[data-testid="stSidebar"] .stMarkdown p {{
        color: {C["text_secondary"]};
    }}

    /* ── Typography ────────────────────────────── */
    h1, h2, h3 {{
        color: {C["text"]} !important;
        font-family: "Inter", system-ui, sans-serif;
    }}
    h1 {{
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }}
    .stMarkdown p {{
        color: {C["text_secondary"]};
    }}

    /* ── Metrics ───────────────────────────────── */
    [data-testid="stMetricValue"] {{
        color: {C["text"]} !important;
        font-weight: 700;
    }}
    [data-testid="stMetricDelta"] svg {{
        display: inline;
    }}

    /* ── Cards ─────────────────────────────────── */
    .lift-card {{
        background: {C["surface"]};
        border: 1px solid {C["border"]};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
    }}
    .lift-card-accent {{
        background: {C["surface"]};
        border: 1px solid {C["primary_dark"]};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
    }}
    .lift-card h4 {{
        color: {C["text_muted"]} !important;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 0.5rem 0;
    }}
    .lift-card .value {{
        color: {C["text"]};
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.2;
    }}
    .lift-card .sub {{
        color: {C["text_muted"]};
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }}

    /* ── Status Badges ─────────────────────────── */
    .badge {{
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .badge-ok {{
        background: {C["ok_bg"]};
        color: {C["ok"]};
        border: 1px solid {C["ok_dark"]};
    }}
    .badge-warn {{
        background: {C["warn_bg"]};
        color: {C["warn"]};
        border: 1px solid {C["warn_dark"]};
    }}
    .badge-bad {{
        background: {C["bad_bg"]};
        color: {C["bad"]};
        border: 1px solid {C["bad_dark"]};
    }}
    .badge-info {{
        background: {C["primary_bg"]};
        color: {C["primary"]};
        border: 1px solid {C["primary_dark"]};
    }}

    /* ── Section Dividers ──────────────────────── */
    .section-header {{
        color: {C["text"]} !important;
        font-size: 1.1rem;
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {C["border"]};
        margin: 1.5rem 0 1rem 0;
    }}

    /* ── Score Bar ─────────────────────────────── */
    .score-bar-container {{
        background: {C["surface2"]};
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
        margin: 0.25rem 0;
    }}
    .score-bar-fill {{
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease;
    }}

    /* ── Sidebar nav ───────────────────────────── */
    section[data-testid="stSidebar"] .stRadio > div {{
        gap: 0.25rem;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        padding: 0.5rem 0.75rem !important;
        border-radius: 8px;
        transition: background 0.2s;
    }}
    section[data-testid="stSidebar"] .stRadio label:hover {{
        background: {C["surface2"]};
    }}

    /* ── Tables ────────────────────────────────── */
    .stDataFrame {{
        border: 1px solid {C["border"]} !important;
        border-radius: 8px;
    }}

    /* ── Plotly chart border ───────────────────── */
    .stPlotlyChart {{
        border: 1px solid {C["border"]};
        border-radius: 12px;
        overflow: hidden;
    }}

    /* ── Recommendation cards ──────────────────── */
    .rec-card {{
        background: {C["surface"]};
        border-left: 3px solid {C["primary"]};
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }}
    .rec-card p {{
        margin: 0;
        color: {C["text_secondary"]};
    }}
    </style>
    """, unsafe_allow_html=True)


# ── Reusable Components ───────────────────────────────────────────────────────

def brand_header():
    """Render the LIFT brand header in the sidebar."""
    st.sidebar.markdown(f"""
    <div style="padding: 1rem 0 1.5rem 0; text-align: center;">
        <span style="color: {C['primary']}; font-size: 2rem; font-weight: 800;
                      letter-spacing: 0.05em;">{MARK} {BRAND}</span>
        <div style="color: {C['text_muted']}; font-size: 0.8rem; margin-top: 0.25rem;">
            {TAGLINE} &middot; v{VERSION}
        </div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "", accent: bool = False):
    """Render a branded metric card."""
    cls = "lift-card-accent" if accent else "lift-card"
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="{cls}">
        <h4>{label}</h4>
        <div class="value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def status_badge(status: str) -> str:
    """Return HTML for a status badge."""
    style_map = {
        "designed": "badge-info",
        "running": "badge-warn",
        "completed": "badge-ok",
        "analyzed": "badge-ok",
        "draft": "badge-info",
        "feasible": "badge-ok",
        "not feasible": "badge-bad",
        "significant": "badge-ok",
        "not significant": "badge-bad",
        "pass": "badge-ok",
        "fail": "badge-bad",
        "warning": "badge-warn",
        "trustworthy": "badge-ok",
        "marginal": "badge-warn",
        "not trustworthy": "badge-bad",
    }
    cls = style_map.get(status.lower(), "badge-info")
    return f'<span class="badge {cls}">{status}</span>'


def section_header(title: str):
    """Render a branded section divider."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def score_bar(value: float, max_val: float = 100.0, height: int = 10) -> str:
    """Return HTML for a horizontal score bar."""
    pct = min(max(value / max_val * 100, 0), 100)
    if pct >= 80:
        color = C["ok"]
    elif pct >= 60:
        color = C["warn"]
    else:
        color = C["bad"]
    return f"""
    <div class="score-bar-container" style="height:{height}px">
        <div class="score-bar-fill" style="width:{pct:.0f}%;background:{color}"></div>
    </div>
    """


def empty_state(icon: str, title: str, message: str):
    """Render an empty state placeholder."""
    st.markdown(f"""
    <div style="text-align:center; padding: 3rem 1rem;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="color: {C['text']}; font-size: 1.2rem; font-weight: 600;">{title}</div>
        <div style="color: {C['text_muted']}; margin-top: 0.5rem;">{message}</div>
    </div>
    """, unsafe_allow_html=True)
