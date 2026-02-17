"""Configuration management for the incrementality platform."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ShopifyConfig(BaseModel):
    shop_domain: str  # e.g., "my-store.myshopify.com"
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    api_version: str = "2025-01"


class AmazonConfig(BaseModel):
    marketplace_id: str  # e.g., "ATVPDKIKX0DER" for US
    seller_id: str
    refresh_token: str
    client_id: str
    client_secret: str
    region: str = "na"  # na, eu, fe


class FacebookConfig(BaseModel):
    app_id: str
    app_secret: str
    access_token: str
    ad_account_id: str  # e.g., "act_123456"


class YouTubeConfig(BaseModel):
    client_id: str
    client_secret: str
    refresh_token: str
    customer_id: str  # Google Ads customer ID
    developer_token: str = ""  # Required — get from Google Ads API Center


class TikTokConfig(BaseModel):
    app_id: str
    secret: str
    access_token: str
    advertiser_id: str = ""


class NorthbeamConfig(BaseModel):
    api_key: str
    account_id: str
    attribution_window: int = 28  # days; 7 or 28 per PRD
    # Channel discount factors applied when computing prior means for Meridian
    discount_factors: dict[str, float] = Field(default_factory=lambda: {
        "meta_perf": 0.65,
        "meta_aware": 0.60,
        "google_brand": 0.85,
        "google_nonbrand": 0.75,
        "tiktok": 0.65,
        "amz_sponsored": 1.0,
        "email_sms": 0.80,
    })


class MMMSettings(BaseModel):
    """Settings for the Google Meridian MMM engine."""
    # MCMC
    mcmc_chains: int = 4
    mcmc_warmup: int = 1000
    mcmc_samples: int = 2000
    # Channels to include in the model (default: all 7 from PRD)
    channels: list[str] = Field(default_factory=lambda: [
        "meta_perf", "meta_aware", "google_brand", "google_nonbrand",
        "tiktok", "amz_sponsored", "email_sms",
    ])
    # Budget optimizer constraints
    max_channel_concentration: float = 0.55  # No channel > 55% of budget
    shopify_revenue_weight: float = 0.7
    amazon_revenue_weight: float = 0.3
    # Alerting thresholds (per PRD §9.3)
    mroi_drop_alert_threshold: float = 0.20   # Alert if mROI drops >20% WoW
    nb_mmm_delta_alert_threshold: float = 0.40  # Alert if NB vs MMM delta >40%
    posterior_mape_alert_threshold: float = 0.12  # Alert if 7-day MAPE >12%
    beta_decay_alert_threshold: float = 0.25  # Alert if β drops >25% vs 90-day avg
    # BigQuery / warehouse (optional — uses local SQLite fallback if absent)
    gcp_project_id: str = ""
    bq_dataset_id: str = "mmm_prod"
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    # Slack / PagerDuty
    slack_webhook_url: str = ""
    pagerduty_routing_key: str = ""
    # Refit schedule
    weekly_refit_day: str = "monday"
    daily_update_hour_et: int = 7  # 07:00 AM ET


class StatisticalConfig(BaseModel):
    """Statistical parameters for test design."""
    significance_level: float = 0.05  # Alpha
    target_power: float = 0.80  # 1 - Beta
    min_test_duration_weeks: int = 2
    max_test_duration_weeks: int = 12
    default_lookback_weeks: int = 12
    min_dmas_per_cell: int = 5
    max_holdout_fraction: float = 0.50  # Max fraction of DMAs in holdout
    target_holdout_fraction: float = 0.25  # Ideal holdout fraction
    balance_tolerance: float = 0.10  # Max standardized mean difference for balance

    @field_validator("significance_level")
    @classmethod
    def _validate_alpha(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError(f"significance_level must be between 0 and 1, got {v}")
        return v

    @field_validator("target_power")
    @classmethod
    def _validate_power(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError(f"target_power must be between 0 and 1, got {v}")
        return v

    @field_validator("max_holdout_fraction", "target_holdout_fraction")
    @classmethod
    def _validate_fraction(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError(f"holdout fraction must be between 0 and 1, got {v}")
        return v


class Config(BaseModel):
    """Top-level configuration."""
    shopify: ShopifyConfig | None = None
    amazon: AmazonConfig | None = None
    facebook: FacebookConfig | None = None
    youtube: YouTubeConfig | None = None
    tiktok: TikTokConfig | None = None
    northbeam: NorthbeamConfig | None = None
    statistical: StatisticalConfig = Field(default_factory=StatisticalConfig)
    mmm: MMMSettings = Field(default_factory=MMMSettings)
    data_dir: str = "./data"
    output_dir: str = "./output"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(exclude_none=True), f, default_flow_style=False)
