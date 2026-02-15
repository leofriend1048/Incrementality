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
    statistical: StatisticalConfig = Field(default_factory=StatisticalConfig)
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
