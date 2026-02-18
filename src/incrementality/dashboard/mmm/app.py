"""
MTB MMM Dashboard — Main entry point
Views: National Decomposition | DMA Intelligence | Budget Optimizer | Model Health
"""
import streamlit as st

# Page config must be first
st.set_page_config(
    page_title="MTB MMM Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from incrementality.dashboard.mmm import decomposition, dma_intel, budget_opt, model_health  # noqa: E402
from incrementality.dashboard.theme import apply_theme  # noqa: E402

def main():
    apply_theme()

    st.sidebar.title("MTB MMM Platform")
    st.sidebar.markdown("*Meridian · DMA Panel · Northbeam-Calibrated*")

    view = st.sidebar.radio(
        "View",
        ["National Decomposition", "DMA Intelligence", "Budget Optimizer", "Model Health"],
    )

    if view == "National Decomposition":
        decomposition.render()
    elif view == "DMA Intelligence":
        dma_intel.render()
    elif view == "Budget Optimizer":
        budget_opt.render()
    elif view == "Model Health":
        model_health.render()

if __name__ == "__main__":
    main()
