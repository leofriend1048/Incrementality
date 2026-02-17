"""
View 2 — DMA Intelligence
Choropleth map of marginal ROI by DMA, ranking table, saturation badges.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from incrementality.dashboard.theme import C, CHART_PALETTE, section_header

# ── DMA synthetic data ─────────────────────────────────────────────────────────

# 210 US DMAs with realistic names (subset of Nielsen DMAs)
_DMA_NAMES = [
    "New York NY", "Los Angeles CA", "Chicago IL", "Philadelphia PA",
    "Dallas-Ft. Worth TX", "San Francisco-Oakland-San Jose CA",
    "Boston MA-Manchester NH", "Atlanta GA", "Washington DC",
    "Houston TX", "Tampa-St. Petersburg FL", "Seattle-Tacoma WA",
    "Minneapolis-St. Paul MN", "Miami-Ft. Lauderdale FL", "Detroit MI",
    "Phoenix AZ", "Cleveland-Akron OH", "Denver CO", "Orlando FL",
    "Sacramento-Stockton-Modesto CA", "St. Louis MO", "Portland OR",
    "Indianapolis IN", "Pittsburgh PA", "Baltimore MD", "San Diego CA",
    "Nashville TN", "Charlotte NC", "Raleigh-Durham NC", "Hartford-New Haven CT",
    "Kansas City MO", "Columbus OH", "Cincinnati OH", "Milwaukee WI",
    "Salt Lake City UT", "Las Vegas NV", "Grand Rapids-Kalamazoo MI",
    "Norfolk-Portsmouth VA", "Memphis TN", "Louisville KY",
    "New Orleans LA", "Buffalo NY", "Oklahoma City OK", "Greensboro NC",
    "Providence RI-New Bedford MA", "Albuquerque-Santa Fe NM", "Jacksonville FL",
    "Richmond VA", "Austin TX", "West Palm Beach FL", "Birmingham AL",
    "Albany-Schenectady-Troy NY", "Tulsa OK", "Fresno-Visalia CA",
    "Dayton OH", "Little Rock AR", "Knoxville TN", "Charleston SC",
    "Lexington KY", "Spokane WA", "Des Moines-Ames IA", "Wichita KS",
    "Flint-Saginaw-Bay City MI", "Omaha NE", "Rochester NY", "Tucson AZ",
    "Toledo OH", "Honolulu HI", "El Paso TX", "Shreveport LA",
    "Roanoke-Lynchburg VA", "Portland-Auburn ME", "Paducah KY",
    "Columbia SC", "Syracuse NY", "Champaign-Springfield IL",
    "Madison WI", "South Bend-Elkhart IN", "Tri-Cities TN-VA",
    "Huntsville-Decatur AL", "Sioux Falls SD", "Springfield MO",
    "Waco-Temple-Bryan TX", "Baton Rouge LA", "Augusta GA",
    "Savannah GA", "Mobile AL-Pensacola FL", "Chattanooga TN",
    "Jackson MS", "Cedar Rapids-Waterloo IA", "Ft. Smith-Fayetteville AR",
    "Burlington VT-Plattsburgh NY", "Colorado Springs-Pueblo CO",
    "Greenville-Spartanburg SC", "Johnson City TN", "Evansville IN",
    "Columbus GA", "Tallahassee FL-Thomasville GA", "Macon GA",
    "Wilmington NC", "Boise ID", "Springfield MA", "Beaumont-Port Arthur TX",
    "Greenville-New Bern-Washington NC", "Corpus Christi TX",
    "Peoria-Bloomington IL", "Lansing MI", "Ft. Myers-Naples FL",
    "Harlingen-Weslaco TX", "Topeka KS", "Youngstown OH",
    "Lubbock TX", "Davenport IA-Rock Island-Moline IL",
    "Meridian MS", "Columbus-Tupelo MS", "Monroe LA-El Dorado AR",
    "Odessa-Midland TX", "Amarillo TX", "Anchorage AK",
    "Fargo-Valley City ND", "Medford-Klamath Falls OR",
    "Yakima-Pasco WA", "Wausau-Rhinelander WI",
    "Duluth MN-Superior WI", "Billings MT", "Ft. Wayne IN",
    "Eugene OR", "Gainesville FL", "Rapid City SD",
    "Bakersfield CA", "Santa Barbara CA", "Monterey-Salinas CA",
    "Palm Springs CA", "Chico-Redding CA", "Eureka CA",
    "Reno NV", "Yuma AZ-El Centro CA", "Missoula MT",
    "Great Falls MT", "Butte-Bozeman MT", "Helena MT",
    "Idaho Falls-Pocatello ID", "Twin Falls ID", "Bend OR",
    "Wichita Falls TX", "Abilene TX", "San Angelo TX",
    "Tyler-Longview TX", "Laredo TX", "Brownsville TX",
    "Victoria TX", "Laramie WY", "Cheyenne WY", "Casper WY",
    "Rock Springs WY", "Grand Junction CO", "Pueblo CO",
    "Scottsbluff NE", "North Platte NE", "Lincoln NE",
    "Sioux City IA", "Quincy IL-Hannibal MO", "Springfield-Holyoke MA",
    "Manchester NH", "Bangor ME", "Presque Isle ME",
    "Glendive MT", "Miles City MT", "Minot ND",
    "Bismarck ND", "Grand Forks ND", "Mankato MN",
    "La Crosse-Eau Claire WI", "Marquette MI", "Green Bay WI",
    "Traverse City-Cadillac MI", "Alpena MI", "Salisbury MD",
    "Bluefield-Beckley-Oak Hill WV", "Clarksburg-Weston WV",
    "Parkersburg WV", "Wheeling WV-Steubenville OH",
    "Lima OH", "Zanesville OH", "Terre Haute IN",
    "Bowling Green KY", "Harrisonburg VA", "Charlottesville VA",
    "Staunton VA", "Harrisburg-Lancaster PA", "Scranton-Wilkes-Barre PA",
    "Wilkes Barre PA", "Elmira NY", "Binghamton NY",
    "Utica NY", "Watertown NY", "Plattsburgh-Massena NY",
    "Burlington VT", "Rutland VT", "Zanesville OH",
    "Parkersburg WV", "Huntington WV-Ashland KY", "Lexington KY",
    "Paducah KY-Cape Girardeau MO", "Jackson TN", "Dothan AL",
    "Biloxi-Gulfport MS", "Hattiesburg-Laurel MS",
    "Lake Charles LA", "Alexandria LA", "Lafayette LA",
    "Jonesboro AR", "Texarkana TX", "Wichita Falls TX-Lawton OK",
    "Sherman TX-Ada OK", "Ardmore OK", "Joplin MO-Pittsburg KS",
]

# Deduplicate to 210
_DMA_NAMES = list(dict.fromkeys(_DMA_NAMES))[:210]

_STATE_COORDS = {
    "NY": (43.3, -74.2), "CA": (36.7, -119.4), "IL": (40.0, -89.2),
    "PA": (41.2, -77.2), "TX": (31.5, -99.5), "MA": (42.1, -71.6),
    "GA": (32.7, -83.2), "DC": (38.9, -77.0), "FL": (27.8, -81.5),
    "WA": (47.4, -121.5), "MN": (45.7, -94.3), "MI": (44.3, -85.4),
    "AZ": (34.3, -111.1), "OH": (40.4, -82.7), "CO": (39.1, -105.4),
    "OR": (43.8, -120.6), "IN": (39.8, -86.3), "MD": (38.8, -76.8),
    "SD": (44.3, -100.3), "MO": (37.9, -91.8), "CT": (41.6, -72.7),
    "WI": (44.3, -89.8), "UT": (39.3, -111.1), "NV": (38.5, -117.0),
    "NC": (35.5, -79.4), "TN": (35.9, -86.3), "VA": (37.5, -79.5),
    "KY": (37.5, -85.3), "LA": (30.9, -91.8), "AL": (32.7, -86.8),
    "SC": (33.8, -81.2), "OK": (35.6, -97.5), "AR": (34.8, -92.2),
    "IA": (42.0, -93.6), "KS": (38.5, -98.4), "NE": (41.5, -99.9),
    "ND": (47.5, -100.2), "MS": (32.4, -89.7), "ID": (44.2, -114.5),
    "NM": (34.4, -106.1), "AK": (64.2, -153.4), "HI": (20.8, -157.0),
    "MT": (46.9, -110.4), "WY": (43.0, -107.6), "VT": (44.1, -72.7),
    "NH": (43.7, -71.6), "ME": (45.3, -69.4), "RI": (41.7, -71.5),
    "WV": (38.5, -80.5), "DE": (39.0, -75.5),
}

_CHANNELS = [
    "Meta (Paid Social)",
    "Google Search",
    "Google Shopping",
    "TikTok",
    "Pinterest",
    "YouTube",
    "Email / CRM",
    "TV / CTV",
]


def _extract_state(dma_name: str) -> str:
    """Extract state abbreviation from DMA name."""
    parts = dma_name.split()
    for p in reversed(parts):
        p = p.rstrip(".,")
        if len(p) == 2 and p.upper() in _STATE_COORDS:
            return p.upper()
    return "TX"


def _make_dma_data(channel: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed=hash(channel) % (2**31))
    n = len(_DMA_NAMES)

    # Base spend per week — power law distribution
    base_spend = rng.lognormal(mean=9.5, sigma=1.2, size=n).clip(200, 120_000)

    # Marginal ROI — inversely related to spend (Hill saturation)
    # High spend → low mROI (saturated), low spend → high mROI (under-served)
    spend_rank = base_spend.argsort().argsort() / n  # normalized rank 0-1
    mroi_base = rng.lognormal(mean=0.5, sigma=0.6, size=n)
    mroi = mroi_base * (2.5 - spend_rank * 2.0)  # invert with rank
    mroi = mroi.clip(0.05, 8.0)

    # Saturation status based on Hill inflection
    hill_inflection = rng.lognormal(mean=9.0, sigma=0.8, size=n)
    ratio = base_spend / hill_inflection

    status = np.where(ratio < 0.6, "Under-Invested",
             np.where(ratio < 1.3, "Optimal", "Saturated"))

    # Recommended delta
    rec_delta = np.where(
        ratio < 0.6,  rng.uniform(0.15, 0.40, size=n) * base_spend,
        np.where(ratio < 1.3, rng.uniform(-0.05, 0.05, size=n) * base_spend,
                 -rng.uniform(0.10, 0.35, size=n) * base_spend)
    )

    states = [_extract_state(d) for d in _DMA_NAMES]
    lats = [_STATE_COORDS.get(s, (37.5, -96.0))[0] + rng.uniform(-1.5, 1.5) for s in states]
    lons = [_STATE_COORDS.get(s, (37.5, -96.0))[1] + rng.uniform(-1.5, 1.5) for s in states]

    df = pd.DataFrame({
        "DMA": _DMA_NAMES,
        "State": states,
        "lat": lats,
        "lon": lons,
        "Spend/Week ($)": base_spend.astype(int),
        "Marginal ROI": mroi.round(2),
        "Status": status,
        "Hill Ratio": ratio.round(2),
        "Recommended Δ ($)": rec_delta.astype(int),
    })
    df["Rank"] = df["Marginal ROI"].rank(ascending=False).astype(int)
    return df.sort_values("Rank")


# ── Badge HTML ─────────────────────────────────────────────────────────────────

def _status_badge_html(status: str) -> str:
    color_map = {
        "Under-Invested": C["ok"],
        "Optimal":        C["warn"],
        "Saturated":      C["bad"],
    }
    bg_map = {
        "Under-Invested": C["ok_bg"],
        "Optimal":        C["warn_bg"],
        "Saturated":      C["bad_bg"],
    }
    c = color_map.get(status, C["text_muted"])
    bg = bg_map.get(status, "transparent")
    return (
        f'<span style="background:{bg};color:{c};padding:2px 8px;'
        f'border-radius:4px;font-size:0.78rem;font-weight:600;">{status}</span>'
    )


# ── Choropleth scatter map ─────────────────────────────────────────────────────

def _map_fig(df: pd.DataFrame, channel: str) -> go.Figure:
    fig = go.Figure()

    # Color: red (high mROI=under-served) → yellow → green (low mROI=saturated)
    # We invert so green = high ROI opportunity
    color_scale = [
        [0.0, C["bad"]],
        [0.4, C["warn"]],
        [1.0, C["ok"]],
    ]

    fig.add_trace(go.Scattergeo(
        lat=df["lat"],
        lon=df["lon"],
        mode="markers",
        marker=dict(
            size=np.clip(np.log1p(df["Spend/Week ($)"]) * 1.8, 5, 18),
            color=df["Marginal ROI"],
            colorscale=color_scale,
            cmin=df["Marginal ROI"].quantile(0.05),
            cmax=df["Marginal ROI"].quantile(0.95),
            showscale=True,
            colorbar=dict(
                title="Marginal ROI",
                titlefont=dict(color=C["text"]),
                tickfont=dict(color=C["text"]),
                bgcolor=C["surface"],
                bordercolor=C["border"],
                x=1.01,
            ),
            line=dict(width=0.5, color=C["bg"]),
        ),
        text=df.apply(
            lambda r: (
                f"<b>{r['DMA']}</b><br>"
                f"mROI: {r['Marginal ROI']:.2f}<br>"
                f"Spend/wk: ${r['Spend/Week ($)']:,.0f}<br>"
                f"Status: {r['Status']}<br>"
                f"Rec Δ: ${r['Recommended Δ ($)']:+,.0f}"
            ),
            axis=1,
        ),
        hoverinfo="text",
        name="",
    ))

    fig.update_geos(
        scope="usa",
        bgcolor=C["bg"],
        lakecolor=C["surface"],
        landcolor=C["surface2"],
        subunitcolor=C["border"],
        countrycolor=C["border"],
        showlakes=True,
        showsubunits=True,
    )

    fig.update_layout(
        geo=dict(
            scope="usa",
            projection_type="albers usa",
        ),
        paper_bgcolor=C["bg"],
        font={"color": C["text"], "family": "Inter, sans-serif"},
        title={
            "text": f"Marginal ROI by DMA — {channel}",
            "font": {"size": 15, "color": C["text"]},
            "x": 0.01,
        },
        height=480,
        margin={"t": 50, "b": 10, "l": 10, "r": 10},
    )
    return fig


# ── Ranking table ──────────────────────────────────────────────────────────────

def _ranking_table(df: pd.DataFrame, top_n: int = 20, bottom: bool = False) -> pd.DataFrame:
    if bottom:
        sub = df.nlargest(top_n, "Rank")[
            ["Rank", "DMA", "Spend/Week ($)", "Marginal ROI", "Status", "Recommended Δ ($)"]
        ].sort_values("Rank", ascending=False)
    else:
        sub = df.nsmallest(top_n, "Rank")[
            ["Rank", "DMA", "Spend/Week ($)", "Marginal ROI", "Status", "Recommended Δ ($)"]
        ].sort_values("Rank")
    return sub


def _style_ranking(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def color_status(val):
        cm = {
            "Under-Invested": f"color: {C['ok']}; font-weight: 600",
            "Optimal":        f"color: {C['warn']}; font-weight: 600",
            "Saturated":      f"color: {C['bad']}; font-weight: 600",
        }
        return cm.get(val, "")

    def color_mroi(val):
        if val >= 3.0:
            return f"color: {C['ok']}; font-weight: 700"
        elif val >= 1.5:
            return f"color: {C['warn']}"
        else:
            return f"color: {C['bad']}"

    def color_delta(val):
        if val > 0:
            return f"color: {C['ok']}"
        elif val < 0:
            return f"color: {C['bad']}"
        return ""

    return (
        df.style
        .applymap(color_status, subset=["Status"])
        .applymap(color_mroi, subset=["Marginal ROI"])
        .applymap(color_delta, subset=["Recommended Δ ($)"])
        .format({
            "Spend/Week ($)": "${:,.0f}",
            "Marginal ROI": "{:.2f}x",
            "Recommended Δ ($)": "${:+,.0f}",
        })
    )


# ── Summary saturation badges ──────────────────────────────────────────────────

def _saturation_summary(df: pd.DataFrame) -> None:
    counts = df["Status"].value_counts()
    total = len(df)

    cols = st.columns(3)
    for col, (status, color) in zip(cols, [
        ("Under-Invested", C["ok"]),
        ("Optimal",        C["warn"]),
        ("Saturated",      C["bad"]),
    ]):
        n = counts.get(status, 0)
        pct = n / total * 100
        with col:
            st.markdown(
                f"""
                <div style="background:{color}1A;border:1px solid {color};
                            border-radius:8px;padding:12px 16px;text-align:center;">
                    <div style="font-size:1.6rem;font-weight:800;color:{color};">{n}</div>
                    <div style="color:{color};font-size:0.9rem;font-weight:600;">{status}</div>
                    <div style="color:{C['text_muted']};font-size:0.75rem;">{pct:.0f}% of DMAs</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("DMA Intelligence")
    st.caption("Marginal ROI by DMA · Hill saturation model · Spend reallocation recommendations")

    st.info(
        "Displaying **synthetic data** representative of a Meridian DMA panel model. "
        "Connect real model outputs to replace.",
        icon="ℹ️",
    )

    # Channel filter
    channel = st.selectbox("Channel", _CHANNELS, index=0)

    df = _make_dma_data(channel)

    # ── Saturation summary ────────────────────────────────────────────────────
    section_header("Saturation Summary")
    _saturation_summary(df)

    st.divider()

    # ── Map ───────────────────────────────────────────────────────────────────
    section_header("Marginal ROI Map")
    fig = _map_fig(df, channel)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Bubble size = weekly spend. Color = marginal ROI "
        "(green = high opportunity / under-served, red = saturated / over-invested)."
    )

    st.divider()

    # ── Ranking tables ────────────────────────────────────────────────────────
    section_header("DMA Rankings")
    tab_top, tab_bot = st.tabs(["Top 20 — Highest mROI (Under-served)", "Bottom 20 — Lowest mROI (Saturated)"])

    with tab_top:
        top_df = _ranking_table(df, top_n=20, bottom=False)
        st.dataframe(_style_ranking(top_df), use_container_width=True, hide_index=True)

    with tab_bot:
        bot_df = _ranking_table(df, top_n=20, bottom=True)
        st.dataframe(_style_ranking(bot_df), use_container_width=True, hide_index=True)

    st.divider()

    # ── Spend reallocation opportunity ────────────────────────────────────────
    section_header("Reallocation Opportunity")
    total_rec_increase = df.loc[df["Recommended Δ ($)"] > 0, "Recommended Δ ($)"].sum()
    total_rec_decrease = df.loc[df["Recommended Δ ($)"] < 0, "Recommended Δ ($)"].sum()
    net_delta = total_rec_increase + total_rec_decrease

    c1, c2, c3 = st.columns(3)
    c1.metric("Recommended Increase", f"+${total_rec_increase:,.0f}/wk",
              help="Sum of recommended spend increases across under-invested DMAs")
    c2.metric("Recommended Decrease", f"${total_rec_decrease:,.0f}/wk",
              help="Sum of recommended spend decreases across saturated DMAs")
    c3.metric("Net Budget Delta", f"${net_delta:+,.0f}/wk",
              help="Net change if all recommendations applied (budget-neutral if ~$0)")

    with st.expander("Saturation model details"):
        st.markdown("""
        **Hill Saturation Function:** `contribution(spend) = α · spend^γ / (κ^γ + spend^γ)`

        | Symbol | Meaning |
        |--------|---------|
        | α | Channel-level asymptote (max weekly contribution) |
        | κ | Inflection point — spend at 50% of max response |
        | γ | Shape parameter — steepness of saturation curve |
        | **Hill Ratio** | `spend / κ` — ratio > 1 means past inflection (diminishing returns) |

        **Status thresholds:**
        - **Under-Invested:** Hill Ratio < 0.6
        - **Optimal:** 0.6 ≤ Hill Ratio < 1.3
        - **Saturated:** Hill Ratio ≥ 1.3
        """)
