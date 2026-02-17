"""AppLovin MAX Reporting API connector.

Pulls ad spend and delivery data from the AppLovin MAX platform, including
connected TV (CTV), in-app, and interstitial placements.

AppLovin does not offer DMA-level geographic breakdowns.
The finest granularity is country-level; all spend data returned here
is national (US) and fed into the MMM as a national channel.

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

_BASE_URL = "https://r.applovin.com"

# AppLovin report columns
_DEFAULT_COLUMNS = "day,impressions,clicks,installs,revenue"

# Max 90 days per request
_MAX_WINDOW_DAYS = 90


class AppLovinAPIError(Exception):
    """Raised for AppLovin API-level errors."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"AppLovin API error {status_code}: {message}")


class AppLovinConnector:
    """Connects to the AppLovin MAX Reporting API.

    Pulls daily spend, impressions, and conversion data at the national level
    for use as an MMM channel input.

    Args:
        config: AppLovinConfig with report_key and optional sdk_key.
    """

    DEFAULT_TIMEOUT = 30

    def __init__(self, config: AppLovinConfig) -> None:
        self.config = config
        self.session = requests.Session()

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

    def get_daily_national_spend(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch daily national spend from AppLovin MAX.

        Uses the AppLovin report API endpoint which returns CSV data
        by default.  Dates are chunked to respect the 90-day limit.

        Args:
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            DataFrame with columns:
                date, spend, impressions, clicks, conversions
        """
        all_rows: list[dict] = []
        chunk_start = start_date

        while chunk_start <= end_date:
            chunk_end = min(
                chunk_start + timedelta(days=_MAX_WINDOW_DAYS - 1), end_date
            )
            logger.info(
                "AppLovin: fetching %s → %s", chunk_start, chunk_end
            )

            params = {
                "api_key": self.config.report_key,
                "start": chunk_start.isoformat(),
                "end": chunk_end.isoformat(),
                "columns": _DEFAULT_COLUMNS,
                "format": "json",
                "country": "US",
            }

            resp = self._get("report", params)
            data = resp.json()

            # AppLovin report endpoint returns {"results": [...]} or CSV
            results = data if isinstance(data, list) else data.get("results", [])
            for row in results:
                all_rows.append({
                    "date": pd.to_datetime(row.get("day")).date(),
                    "spend": float(row.get("revenue", 0) or 0),
                    "impressions": int(row.get("impressions", 0) or 0),
                    "clicks": int(row.get("clicks", 0) or 0),
                    "conversions": int(row.get("installs", 0) or 0),
                })

            chunk_start = chunk_end + timedelta(days=1)

        if not all_rows:
            logger.warning(
                "AppLovin: no data for %s → %s", start_date, end_date
            )
            return pd.DataFrame(
                columns=["date", "spend", "impressions", "clicks", "conversions"]
            )

        df = pd.DataFrame(all_rows)
        df = df.sort_values("date").reset_index(drop=True)
        logger.info(
            "AppLovin: %d rows, spend=%.2f, %s → %s",
            len(df), df["spend"].sum(),
            df["date"].min(), df["date"].max(),
        )
        return df
