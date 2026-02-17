"""Pinterest Ads API v5 connector.

Pulls ad spend and delivery metrics from the Pinterest Business API,
broken down by country/region.  Pinterest does not offer DMA-level
geographic breakdowns; the finest granularity is region (state).

Typical usage::

    connector = PinterestConnector(config)
    df = connector.get_daily_spend_by_region(
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

from incrementality.config import PinterestConfig
from incrementality.connectors.retry import request_with_retry

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.pinterest.com/v5"

# Pinterest reports are capped at 90 days per request
_MAX_WINDOW_DAYS = 90

# Granularity options: DAY, WEEK, MONTH, TOTAL
_GRANULARITY = "DAY"


class PinterestAPIError(Exception):
    """Raised for Pinterest API-level errors."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Pinterest API error {status_code}: {message}")


class PinterestConnector:
    """Connects to the Pinterest Business API v5 for ad spend by region.

    Args:
        config: PinterestConfig with app credentials and ad account ID.
    """

    DEFAULT_TIMEOUT = 30  # seconds

    def __init__(self, config: PinterestConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.access_token}",
            "Content-Type": "application/json",
        })

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{_BASE_URL}/{path.lstrip('/')}"
        resp = request_with_retry(
            self.session, "GET", url,
            params=params or {},
            timeout=self.DEFAULT_TIMEOUT,
        )
        if not resp.ok:
            raise PinterestAPIError(resp.status_code, resp.text[:500])
        return resp.json()

    def get_campaigns(self) -> pd.DataFrame:
        """Return all campaigns in the ad account.

        Returns columns: campaign_id, name, status, objective_type, budget.
        """
        data = self._get(
            f"ad_accounts/{self.config.ad_account_id}/campaigns",
            params={"page_size": 250},
        )
        records = []
        for c in data.get("items", []):
            records.append({
                "campaign_id": c.get("id", ""),
                "name": c.get("name", ""),
                "status": c.get("status", ""),
                "objective_type": c.get("objective_type", ""),
                "budget": float(c.get("daily_spend_cap", 0)) / 1_000_000,  # microcurrency
            })
        return pd.DataFrame(records)

    def get_daily_spend_by_region(
        self,
        start_date: date,
        end_date: date,
        campaign_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch daily ad spend broken down by region (state).

        Pinterest provides region-level geographic breakdowns via the
        analytics/report endpoint with ``granularity=DAY`` and
        ``breakdown_types=['REGION']``.

        Args:
            start_date: Inclusive start date.
            end_date: Inclusive end date (auto-chunked at 90-day windows).
            campaign_ids: Optional filter — specific campaign IDs only.

        Returns:
            DataFrame with columns:
                date, region, spend, impressions, clicks, saves, pin_clicks
        """
        all_rows: list[dict] = []
        chunk_start = start_date

        while chunk_start <= end_date:
            chunk_end = min(
                chunk_start + timedelta(days=_MAX_WINDOW_DAYS - 1), end_date
            )
            logger.info(
                "Pinterest: fetching spend %s → %s", chunk_start, chunk_end
            )

            payload: dict[str, Any] = {
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "granularity": _GRANULARITY,
                "columns": [
                    "DATE", "CAMPAIGN_ID", "SPEND_IN_MICRO_DOLLAR",
                    "IMPRESSION_1", "OUTBOUND_CLICK_1", "SAVE", "PIN_CLICK",
                    "REGION",
                ],
                "level": "CAMPAIGN",
                "click_window_days": 30,
                "view_window_days": 1,
                "conversion_report_time": "TIME_OF_AD_ACTION",
            }
            if campaign_ids:
                payload["campaign_ids"] = campaign_ids

            data = self._get(
                f"ad_accounts/{self.config.ad_account_id}/analytics",
                params=payload,
            )

            for row in data.get("data", []):
                all_rows.append({
                    "date": pd.to_datetime(row.get("DATE")).date(),
                    "region": row.get("REGION", ""),
                    "spend": float(row.get("SPEND_IN_MICRO_DOLLAR", 0)) / 1_000_000,
                    "impressions": int(row.get("IMPRESSION_1", 0) or 0),
                    "clicks": int(row.get("OUTBOUND_CLICK_1", 0) or 0),
                    "saves": int(row.get("SAVE", 0) or 0),
                    "pin_clicks": int(row.get("PIN_CLICK", 0) or 0),
                })

            chunk_start = chunk_end + timedelta(days=1)

        if not all_rows:
            logger.warning(
                "Pinterest: no data for %s → %s", start_date, end_date
            )
            return pd.DataFrame(
                columns=["date", "region", "spend", "impressions",
                         "clicks", "saves", "pin_clicks"]
            )

        df = pd.DataFrame(all_rows)
        df = df.sort_values(["date", "region"]).reset_index(drop=True)
        logger.info(
            "Pinterest: %d rows, spend=%.2f, %s → %s",
            len(df), df["spend"].sum(),
            df["date"].min(), df["date"].max(),
        )
        return df

    def get_daily_national_spend(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch national-level daily spend (no geo breakdown).

        Convenience method for MMM channels that need national totals.

        Returns columns: date, spend, impressions, clicks.
        """
        df = self.get_daily_spend_by_region(start_date, end_date)
        if df.empty:
            return pd.DataFrame(columns=["date", "spend", "impressions", "clicks"])
        return (
            df.groupby("date", as_index=False)[["spend", "impressions", "clicks"]]
            .sum()
            .sort_values("date")
            .reset_index(drop=True)
        )
