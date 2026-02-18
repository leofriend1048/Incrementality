"""AppLovin Ads (Axon engine) Campaign Management API connector.

AppLovin Ads is AppLovin's demand-side advertising platform, powered by the
Axon machine-learning engine.  This is the advertiser-facing product — NOT
AppLovin MAX, which is the publisher-side monetization SDK.

AppLovin Ads does not offer DMA-level geographic breakdowns.
Country-level (US national) data is returned and fed into the MMM as the
``applovin`` channel.

Campaign Management API docs:
    https://developers.applovin.com/en/api/campaign-management

Typical usage::

    connector = AppLovinConnector(config)
    df = connector.get_daily_national_spend(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

from incrementality.config import AppLovinConfig
from incrementality.connectors.retry import request_with_retry

logger = logging.getLogger(__name__)

# AppLovin Campaign Management API base URL
_BASE_URL = "https://api.applovin.com"

# Reporting endpoint — returns campaign-level daily metrics
_REPORT_PATH = "campaign/report"

# Max date range per request (AppLovin recommends ≤ 90 days)
_MAX_WINDOW_DAYS = 90

# Default breakdown columns for the report
_DEFAULT_COLUMNS = "day,country,campaign_name,campaign_id,impressions,clicks,installs,spend"


class AppLovinAPIError(Exception):
    """Raised for AppLovin Campaign Management API errors."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"AppLovin Ads API error {status_code}: {message}")


class AppLovinConnector:
    """Connects to the AppLovin Ads (Axon) Campaign Management API.

    Pulls daily spend, impressions, clicks, and conversion data at the
    US national level for use as an MMM channel input.

    Args:
        config: AppLovinConfig with api_key (the AppLovin Ads API key,
                found in the AppLovin Ads dashboard under Account → API).
    """

    DEFAULT_TIMEOUT = 30

    def __init__(self, config: AppLovinConfig) -> None:
        self.config = config
        self.session = requests.Session()
        # AppLovin Ads uses the api_key as a header (not MAX's report_key query param)
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        })

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        url = f"{_BASE_URL}/{path.lstrip('/')}"
        resp = request_with_retry(
            self.session, "GET", url,
            params=params,
            timeout=self.DEFAULT_TIMEOUT,
        )
        if not resp.ok:
            raise AppLovinAPIError(resp.status_code, resp.text[:500])
        return resp

    def get_campaigns(self) -> list[dict[str, Any]]:
        """Return all campaigns in the AppLovin Ads account.

        Returns:
            List of campaign dicts with id, name, status, budget, goal.
        """
        resp = self._get("campaign/list", params={})
        data = resp.json()
        campaigns = data.get("campaigns", data) if isinstance(data, dict) else data
        logger.info("AppLovin Ads: %d campaigns found.", len(campaigns))
        return campaigns if isinstance(campaigns, list) else []

    def get_daily_national_spend(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch daily national (US) spend from AppLovin Ads (Axon).

        Uses the Campaign Management reporting endpoint.  Filters to
        country='US' and aggregates across all campaigns.

        Args:
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            DataFrame with columns:
                date, spend, impressions, clicks, installs
        """
        all_rows: list[dict] = []
        chunk_start = start_date

        while chunk_start <= end_date:
            chunk_end = min(
                chunk_start + timedelta(days=_MAX_WINDOW_DAYS - 1), end_date
            )
            logger.info(
                "AppLovin Ads: fetching %s → %s", chunk_start, chunk_end
            )

            params: dict[str, Any] = {
                "start": chunk_start.isoformat(),
                "end": chunk_end.isoformat(),
                "columns": _DEFAULT_COLUMNS,
                "format": "json",
                "filter_country": "US",
            }

            resp = self._get(_REPORT_PATH, params)
            data = resp.json()

            # Response shape: {"results": [...]} or a bare list
            results = (
                data.get("results", data.get("data", data))
                if isinstance(data, dict)
                else data
            )
            if isinstance(results, list):
                for row in results:
                    day_str = row.get("day") or row.get("date", "")
                    if not day_str:
                        continue
                    all_rows.append({
                        "date": pd.to_datetime(day_str).date(),
                        "spend": float(row.get("spend", 0) or 0),
                        "impressions": int(row.get("impressions", 0) or 0),
                        "clicks": int(row.get("clicks", 0) or 0),
                        "installs": int(row.get("installs", 0) or 0),
                    })

            chunk_start = chunk_end + timedelta(days=1)

        if not all_rows:
            logger.warning(
                "AppLovin Ads: no data for %s → %s", start_date, end_date
            )
            return pd.DataFrame(
                columns=["date", "spend", "impressions", "clicks", "installs"]
            )

        df = pd.DataFrame(all_rows)
        # Aggregate across campaigns for the same date
        df = (
            df.groupby("date", as_index=False)
            .agg({"spend": "sum", "impressions": "sum",
                  "clicks": "sum", "installs": "sum"})
            .sort_values("date")
            .reset_index(drop=True)
        )
        logger.info(
            "AppLovin Ads: %d rows, spend=%.2f, %s → %s",
            len(df), df["spend"].sum(),
            df["date"].min(), df["date"].max(),
        )
        return df
