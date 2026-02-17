"""LIFT Dashboard — Main entry point.

Run with:
    streamlit run src/incrementality/dashboard/app.py

Or if installed as a package:
    python -m incrementality.dashboard.app
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is on the path for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from incrementality.dashboard.theme import (
    BRAND, C, MARK, TAGLINE, VERSION, brand_header, inject_css, setup_plotly,
)

# ── Page Config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title=f"{BRAND} — {TAGLINE}",
    page_icon=MARK,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme Setup ───────────────────────────────────────────────────────────────

inject_css()
setup_plotly()

# ── Sidebar Navigation ───────────────────────────────────────────────────────

brand_header()

PAGES = {
    "Home": "home",
    "Configuration": "config",
    # ── Geo Holdout Testing ───────────────────────────────────────────────
    "Test Designer": "designer",
    "Test Manager": "manager",
    "Analysis Report": "analysis",
    "Spend Optimizer": "spend",
    # ── Marketing Mix Model (Meridian) ────────────────────────────────────
    "MMM: Decomposition": "mmm_decomposition",
    "MMM: DMA Intelligence": "mmm_dma",
    "MMM: Budget Optimizer": "mmm_budget",
    "MMM: Model Health": "mmm_health",
}

PAGE_ICONS = {
    "Home": "🏠",
    "Configuration": "⚙️",
    "Test Designer": "🔬",
    "Test Manager": "📋",
    "Analysis Report": "📊",
    "Spend Optimizer": "💰",
    "MMM: Decomposition": "📈",
    "MMM: DMA Intelligence": "🗺️",
    "MMM: Budget Optimizer": "🎯",
    "MMM: Model Health": "🩺",
}

st.sidebar.markdown(f"""
<div style="color:{C['text_muted']}; font-size:0.75rem; text-transform:uppercase;
            letter-spacing:0.08em; padding: 0 0 0.5rem 0.5rem;">
Navigation
</div>
""", unsafe_allow_html=True)

GEO_PAGES = ["Home", "Configuration", "Test Designer", "Test Manager",
             "Analysis Report", "Spend Optimizer"]
MMM_PAGES = ["MMM: Decomposition", "MMM: DMA Intelligence",
             "MMM: Budget Optimizer", "MMM: Model Health"]


def _page_label(p: str) -> str:
    icon = PAGE_ICONS.get(p, "")
    if p in MMM_PAGES and p == MMM_PAGES[0]:
        return f"\n{icon}  {p}"
    return f"{icon}  {p}"


selected_page = st.sidebar.radio(
    "Navigation",
    list(PAGES.keys()),
    format_func=_page_label,
    label_visibility="collapsed",
)

# ── Settings (sidebar bottom) ────────────────────────────────────────────────

st.sidebar.markdown("---")

with st.sidebar.expander("Directories", expanded=False):
    data_dir = st.text_input("Data directory", value="./data", key="data_dir",
                             help="Where test designs are saved")
    output_dir = st.text_input("Output directory", value="./output", key="output_dir",
                               help="Where reports are saved")
    config_path = st.text_input("Config file", value="./config.yaml", key="config_path",
                                help="Path to config.yaml")

# ── Page Routing ──────────────────────────────────────────────────────────────

page_key = PAGES[selected_page]

if page_key == "home":
    from incrementality.dashboard.home import render
    render(data_dir, output_dir)

elif page_key == "config":
    from incrementality.dashboard.config_page import render
    render(config_path)

elif page_key == "designer":
    from incrementality.dashboard.designer import render
    render(data_dir)

elif page_key == "manager":
    from incrementality.dashboard.manager import render
    render(data_dir, output_dir)

elif page_key == "analysis":
    from incrementality.dashboard.analysis import render
    render(output_dir)

elif page_key == "spend":
    from incrementality.dashboard.spend import render
    render(output_dir)

elif page_key == "mmm_decomposition":
    from incrementality.dashboard.mmm.decomposition import render
    render()

elif page_key == "mmm_dma":
    from incrementality.dashboard.mmm.dma_intel import render
    render()

elif page_key == "mmm_budget":
    from incrementality.dashboard.mmm.budget_opt import render
    render()

elif page_key == "mmm_health":
    from incrementality.dashboard.mmm.model_health import render
    render()

# ── Footer ────────────────────────────────────────────────────────────────────

st.sidebar.markdown(f"""
<div style="position:fixed; bottom:0; padding:1rem; color:{C['text_muted']};
            font-size:0.7rem; text-align:center; width:inherit;">
    {MARK} {BRAND} v{VERSION} &middot; {TAGLINE}
</div>
""", unsafe_allow_html=True)
