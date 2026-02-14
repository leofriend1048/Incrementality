"""Facebook/Meta Ads data connector.

Pulls ad spend and delivery data from the Meta Marketing API,
broken down by DMA-level geo targeting where available.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
import requests

from incrementality.config import FacebookConfig

logger = logging.getLogger(__name__)

_GRAPH_API_BASE = "https://graph.facebook.com/v19.0"


class FacebookConnector:
    """Connects to Meta Marketing API for ad spend by DMA."""

    def __init__(self, config: FacebookConfig):
        self.config = config
        self.session = requests.Session()
        self.session.params = {"access_token": config.access_token}  # type: ignore

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        url = f"{_GRAPH_API_BASE}/{path}"
        resp = self.session.get(url, params=params or {})
        resp.raise_for_status()
        return resp.json()

    def fetch_campaigns(self) -> pd.DataFrame:
        """Fetch all campaigns in the ad account.

        Returns columns: campaign_id, name, status, objective, daily_budget
        """
        data = self._get(
            f"{self.config.ad_account_id}/campaigns",
            params={
                "fields": "id,name,status,objective,daily_budget",
                "limit": 500,
            },
        )
        records = []
        for c in data.get("data", []):
            records.append({
                "campaign_id": c["id"],
                "name": c.get("name", ""),
                "status": c.get("status", ""),
                "objective": c.get("objective", ""),
                "daily_budget": float(c.get("daily_budget", 0)) / 100,
            })
        return pd.DataFrame(records)

    def fetch_spend_by_dma(
        self,
        start_date: date,
        end_date: date,
        campaign_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch daily ad spend broken down by DMA (geo region).

        Uses the Marketing API insights endpoint with region breakdown.

        Returns columns: date, dma_code, dma_name, spend, impressions, clicks
        """
        params: dict[str, Any] = {
            "fields": "spend,impressions,clicks",
            "breakdowns": "region",
            "time_range": f'{{"since":"{start_date}","until":"{end_date}"}}',
            "time_increment": 1,  # Daily granularity
            "limit": 5000,
        }
        if campaign_ids:
            # Filter to specific campaigns
            filtering = [
                {"field": "campaign.id", "operator": "IN", "value": campaign_ids}
            ]
            params["filtering"] = str(filtering)

        all_records = []
        path = f"{self.config.ad_account_id}/insights"

        while path:
            data = self._get(path, params)
            for row in data.get("data", []):
                region = row.get("region", "")
                dma_code = self._region_to_dma_code(region)
                if dma_code:
                    all_records.append({
                        "date": pd.Timestamp(row["date_start"]).date(),
                        "dma_code": dma_code,
                        "dma_name": region,
                        "spend": float(row.get("spend", 0)),
                        "impressions": int(row.get("impressions", 0)),
                        "clicks": int(row.get("clicks", 0)),
                    })
            # Pagination
            paging = data.get("paging", {})
            next_url = paging.get("next")
            if next_url:
                path = next_url.replace(_GRAPH_API_BASE + "/", "")
                params = {}
            else:
                path = None  # type: ignore

        return pd.DataFrame(all_records)

    def get_campaign_spend_by_dma(
        self,
        start_date: date,
        end_date: date,
        campaign_ids: list[str],
    ) -> pd.DataFrame:
        """Convenience: fetch spend for specific campaigns by DMA."""
        return self.fetch_spend_by_dma(start_date, end_date, campaign_ids)

    @staticmethod
    def _region_to_dma_code(region_name: str) -> str | None:
        """Map a Meta region name to a Nielsen DMA code.

        Meta uses state/region names which we map to DMA codes.
        This is an approximate mapping — for metro-level precision,
        use geo_locations targeting with DMA codes directly.
        """
        # Meta provides DMA-level targeting via geo_locations API
        # When breakdown=region, it returns state-level data.
        # For DMA-level: use the custom geo breakdown approach.
        # This mapping handles the most common cases.
        _REGION_MAP = {
            "New York": "501",
            "Los Angeles": "803",
            "Chicago": "602",
            "Philadelphia": "504",
            "Dallas-Fort Worth": "623",
            "San Francisco": "807",
            "Washington": "511",
            "Houston": "618",
            "Boston": "506",
            "Atlanta": "524",
            "Tampa": "539",
            "Phoenix": "753",
            "Seattle": "819",
            "Detroit": "505",
            "Minneapolis": "613",
            "Miami": "528",
            "Denver": "751",
            "Cleveland": "510",
            "Orlando": "534",
            "Portland": "820",
            "Sacramento": "828",
            "St. Louis": "609",
            "Pittsburgh": "508",
            "Charlotte": "517",
            "Indianapolis": "527",
            "San Diego": "825",
            "Nashville": "659",
            "Kansas City": "616",
            "Las Vegas": "839",
            "Austin": "635",
            "San Antonio": "641",
        }
        return _REGION_MAP.get(region_name)


def get_dma_geo_targeting_spec(dma_codes: list[str]) -> dict:
    """Build a Meta geo_locations targeting spec for specific DMAs.

    Use this when setting up holdout tests to exclude DMAs from ad delivery.
    """
    return {
        "geo_locations": {
            "geo_markets": [
                {"key": code, "market_type": "dma"}
                for code in dma_codes
            ],
        },
    }


def get_exclusion_targeting_spec(holdout_dma_codes: list[str]) -> dict:
    """Build an exclusion targeting spec to hold out specific DMAs.

    Apply this to campaigns during the test period to create the holdout.
    """
    return {
        "excluded_geo_locations": {
            "geo_markets": [
                {"key": code, "market_type": "dma"}
                for code in holdout_dma_codes
            ],
        },
    }
