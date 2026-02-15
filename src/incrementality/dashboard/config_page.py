"""LIFT Dashboard — Configuration page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from incrementality.config import (
    AmazonConfig,
    Config,
    FacebookConfig,
    ShopifyConfig,
    StatisticalConfig,
    YouTubeConfig,
)
from incrementality.dashboard.theme import C, metric_card, section_header, status_badge


def render(config_path: str):
    st.title("Configuration")

    config = _load_or_default(config_path)

    tab_connectors, tab_stats, tab_export = st.tabs([
        "Connectors", "Statistical Parameters", "Import / Export",
    ])

    # ── Connectors ────────────────────────────────────────────────────────
    with tab_connectors:
        _render_connector_status(config)
        st.divider()
        _render_connector_forms(config, config_path)

    # ── Statistical Parameters ────────────────────────────────────────────
    with tab_stats:
        _render_stats_config(config, config_path)

    # ── Import / Export ───────────────────────────────────────────────────
    with tab_export:
        _render_export(config, config_path)


def _load_or_default(path: str) -> Config:
    p = Path(path)
    if p.exists():
        try:
            return Config.from_yaml(p)
        except Exception:
            pass
    return Config()


def _render_connector_status(config: Config):
    section_header("Connector Status")
    c1, c2, c3, c4 = st.columns(4)
    connectors = [
        ("Shopify", config.shopify, c1),
        ("Amazon", config.amazon, c2),
        ("Facebook", config.facebook, c3),
        ("YouTube", config.youtube, c4),
    ]
    for name, cfg, col in connectors:
        with col:
            connected = cfg is not None
            badge = status_badge("pass" if connected else "fail")
            label = "Connected" if connected else "Not configured"
            st.markdown(f"""
            <div class="lift-card" style="text-align:center;">
                <h4>{name}</h4>
                <div style="margin: 0.5rem 0;">{badge}</div>
                <div class="sub">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_connector_forms(config: Config, config_path: str):
    section_header("Connector Settings")

    with st.expander("Shopify", expanded=False):
        shop = config.shopify or ShopifyConfig(shop_domain="", access_token="")
        domain = st.text_input("Shop Domain", value=shop.shop_domain,
                               placeholder="my-store.myshopify.com",
                               key="shopify_domain")
        token = st.text_input("Access Token", value=shop.access_token,
                              type="password", key="shopify_token")
        api_ver = st.text_input("API Version", value=shop.api_version,
                                key="shopify_api_ver")
        if st.button("Save Shopify", key="save_shopify"):
            if domain and token:
                config.shopify = ShopifyConfig(
                    shop_domain=domain, access_token=token, api_version=api_ver,
                )
                _save_config(config, config_path)
                st.success("Shopify configuration saved.")

    with st.expander("Amazon", expanded=False):
        amz = config.amazon or AmazonConfig(
            marketplace_id="", seller_id="", refresh_token="",
            client_id="", client_secret="",
        )
        mkid = st.text_input("Marketplace ID", value=amz.marketplace_id,
                             placeholder="ATVPDKIKX0DER", key="amz_mkid")
        sid = st.text_input("Seller ID", value=amz.seller_id, key="amz_sid")
        rt = st.text_input("Refresh Token", value=amz.refresh_token,
                           type="password", key="amz_rt")
        cid = st.text_input("Client ID", value=amz.client_id, key="amz_cid")
        csec = st.text_input("Client Secret", value=amz.client_secret,
                             type="password", key="amz_csec")
        if st.button("Save Amazon", key="save_amazon"):
            if mkid and sid and rt and cid and csec:
                config.amazon = AmazonConfig(
                    marketplace_id=mkid, seller_id=sid,
                    refresh_token=rt, client_id=cid, client_secret=csec,
                )
                _save_config(config, config_path)
                st.success("Amazon configuration saved.")

    with st.expander("Facebook / Meta", expanded=False):
        fb = config.facebook or FacebookConfig(
            app_id="", app_secret="", access_token="", ad_account_id="",
        )
        fb_appid = st.text_input("App ID", value=fb.app_id, key="fb_appid")
        fb_secret = st.text_input("App Secret", value=fb.app_secret,
                                  type="password", key="fb_secret")
        fb_token = st.text_input("Access Token", value=fb.access_token,
                                 type="password", key="fb_token")
        fb_adacct = st.text_input("Ad Account ID", value=fb.ad_account_id,
                                  placeholder="act_123456", key="fb_adacct")
        if st.button("Save Facebook", key="save_fb"):
            if fb_appid and fb_token and fb_adacct:
                config.facebook = FacebookConfig(
                    app_id=fb_appid, app_secret=fb_secret,
                    access_token=fb_token, ad_account_id=fb_adacct,
                )
                _save_config(config, config_path)
                st.success("Facebook configuration saved.")

    with st.expander("YouTube / Google Ads", expanded=False):
        yt = config.youtube or YouTubeConfig(
            client_id="", client_secret="", refresh_token="", customer_id="",
        )
        yt_cid = st.text_input("Client ID", value=yt.client_id, key="yt_cid")
        yt_csec = st.text_input("Client Secret", value=yt.client_secret,
                                type="password", key="yt_csec")
        yt_rt = st.text_input("Refresh Token", value=yt.refresh_token,
                              type="password", key="yt_rt")
        yt_custid = st.text_input("Customer ID", value=yt.customer_id, key="yt_custid")
        yt_devtok = st.text_input("Developer Token", value=yt.developer_token,
                                  type="password", key="yt_devtok")
        if st.button("Save YouTube", key="save_yt"):
            if yt_cid and yt_rt and yt_custid:
                config.youtube = YouTubeConfig(
                    client_id=yt_cid, client_secret=yt_csec,
                    refresh_token=yt_rt, customer_id=yt_custid,
                    developer_token=yt_devtok,
                )
                _save_config(config, config_path)
                st.success("YouTube configuration saved.")


def _render_stats_config(config: Config, config_path: str):
    section_header("Statistical Parameters")
    st.caption("These control test design defaults. Changes apply to future designs.")

    s = config.statistical

    c1, c2 = st.columns(2)
    with c1:
        alpha = st.slider("Significance Level (alpha)", 0.01, 0.20, float(s.significance_level),
                          step=0.01, key="stat_alpha",
                          help="Probability of a false positive. Lower = stricter.")
        power = st.slider("Target Power (1 - beta)", 0.70, 0.99, float(s.target_power),
                          step=0.01, key="stat_power",
                          help="Probability of detecting a real effect. Higher = better.")
        holdout_frac = st.slider("Target Holdout Fraction", 0.10, 0.50,
                                 float(s.target_holdout_fraction), step=0.05,
                                 key="stat_holdout",
                                 help="Fraction of DMAs assigned to holdout.")

    with c2:
        min_dur = st.number_input("Min Duration (weeks)", 1, 8,
                                  int(s.min_test_duration_weeks), key="stat_minw")
        max_dur = st.number_input("Max Duration (weeks)", 4, 24,
                                  int(s.max_test_duration_weeks), key="stat_maxw")
        lookback = st.number_input("Lookback (weeks)", 4, 52,
                                   int(s.default_lookback_weeks), key="stat_lookback")
        min_dmas = st.number_input("Min DMAs per Cell", 3, 20,
                                   int(s.min_dmas_per_cell), key="stat_mindma")

    if st.button("Save Statistical Parameters", type="primary"):
        config.statistical = StatisticalConfig(
            significance_level=alpha,
            target_power=power,
            target_holdout_fraction=holdout_frac,
            min_test_duration_weeks=min_dur,
            max_test_duration_weeks=max_dur,
            default_lookback_weeks=lookback,
            min_dmas_per_cell=min_dmas,
        )
        _save_config(config, config_path)
        st.success("Statistical parameters saved.")


def _render_export(config: Config, config_path: str):
    section_header("Import / Export Configuration")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Export")
        st.download_button(
            "Download config.yaml",
            data=_config_to_yaml_str(config),
            file_name="config.yaml",
            mime="text/yaml",
        )

    with c2:
        st.subheader("Import")
        uploaded = st.file_uploader("Upload config.yaml", type=["yaml", "yml"])
        if uploaded is not None:
            try:
                import yaml
                raw = yaml.safe_load(uploaded)
                new_config = Config.model_validate(raw)
                _save_config(new_config, config_path)
                st.success("Configuration imported successfully. Please refresh.")
            except Exception as e:
                st.error(f"Failed to import: {e}")


def _save_config(config: Config, path: str):
    config.to_yaml(path)


def _config_to_yaml_str(config: Config) -> str:
    import yaml
    return yaml.dump(config.model_dump(mode="json", exclude_none=True),
                     default_flow_style=False, sort_keys=False)
