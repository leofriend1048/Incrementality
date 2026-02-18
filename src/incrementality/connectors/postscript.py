"""Postscript SMS marketing API connector.

Postscript is an SMS marketing platform built for Shopify merchants.
This connector pulls campaign-level performance metrics including sends,
clicks, conversions, and attributed revenue.

SMS data is national (no DMA breakdown available) and is fed into the
MMM as the ``postscript_sms`` channel.

Postscript API docs: https://docs.postscript.io/reference/introduction

Typical usage::

    connector = PostscriptConnector(config)
    df = connector.get_daily_performance(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
import requests

from incrementality.config import PostscriptConfig
from incrementality.connectors.retry import request_with_retry

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.postscript.io/api/v2"

# Max campaigns per page
_PAGE_SIZE = 250


class PostscriptAPIError(Exception):
    """Raised for Postscript API-level errors."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Postscript API error {status_code}: {message}")


class PostscriptConnector:
    """Connects to the Postscript SMS API.

    Pulls daily campaign sends, clicks, attributed revenue, and
    estimated spend (from carrier costs + platform fees).

    Args:
        config: PostscriptConfig with api_key and optional shop_id.
    """

    DEFAULT_TIMEOUT = 30

    def __init__(self, config: PostscriptConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
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
            raise PostscriptAPIError(resp.status_code, resp.text[:500])
        return resp.json()

    def get_campaigns(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict]:
        """Return all SMS campaigns (popups and broadcasts).

        Args:
            start_date: Filter campaigns sent on or after this date.
            end_date: Filter campaigns sent on or before this date.

        Returns:
            List of campaign dicts with id, title, type, status, sent_at.
        """
        params: dict[str, Any] = {"limit": _PAGE_SIZE, "page": 1}
        if start_date:
            params["sent_at[gte]"] = start_date.isoformat()
        if end_date:
            params["sent_at[lte]"] = end_date.isoformat()

        all_campaigns: list[dict] = []
        while True:
            data = self._get("campaigns", params)
            items = data.get("data", [])
            all_campaigns.extend(items)

            meta = data.get("meta", {})
            total_pages = meta.get("total_pages", 1)
            if params["page"] >= total_pages:
                break
            params["page"] += 1

        logger.info("Postscript: %d campaigns retrieved.", len(all_campaigns))
        return all_campaigns

    def get_campaign_analytics(self, campaign_id: str) -> dict[str, Any]:
        """Return analytics for a single campaign.

        Returns:
            Dict with sent, delivered, clicked, revenue_attributed, unsubscribes.
        """
        return self._get(f"campaigns/{campaign_id}/analytics")

    def get_daily_performance(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Aggregate daily SMS performance across all campaigns.

        Fetches all campaigns in the window and sums their metrics by
        the date the campaign was sent.  Postscript does not expose a
        single daily-aggregate endpoint, so this is computed client-side.

        Args:
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            DataFrame with columns:
                date, campaigns_sent, total_sends, clicks, revenue,
                unsubscribes, estimated_spend
        """
        campaigns = self.get_campaigns(start_date=start_date, end_date=end_date)

        rows: list[dict] = []
        for campaign in campaigns:
            sent_at_str = campaign.get("sent_at") or campaign.get("created_at", "")
            if not sent_at_str:
                continue
            try:
                sent_date = pd.to_datetime(sent_at_str).date()
            except Exception:
                continue

            if not (start_date <= sent_date <= end_date):
                continue

            try:
                analytics = self.get_campaign_analytics(campaign["id"])
            except PostscriptAPIError as exc:
                logger.warning(
                    "Failed to get analytics for campaign %s: %s",
                    campaign["id"], exc,
                )
                continue

            sends = int(analytics.get("sent", 0) or 0)
            clicks = int(analytics.get("clicked", 0) or 0)
            revenue = float(analytics.get("revenue_attributed", 0) or 0)
            unsubs = int(analytics.get("unsubscribes", 0) or 0)
            # Rough spend estimate: ~$0.01 per SMS segment
            estimated_spend = sends * 0.01

            rows.append({
                "date": sent_date,
                "campaigns_sent": 1,
                "total_sends": sends,
                "clicks": clicks,
                "revenue": revenue,
                "unsubscribes": unsubs,
                "estimated_spend": estimated_spend,
            })

        if not rows:
            logger.warning(
                "Postscript: no campaign data for %s → %s", start_date, end_date
            )
            return pd.DataFrame(
                columns=["date", "campaigns_sent", "total_sends", "clicks",
                         "revenue", "unsubscribes", "estimated_spend"]
            )

        df = pd.DataFrame(rows)
        df = (
            df.groupby("date", as_index=False)
            .agg({
                "campaigns_sent": "sum",
                "total_sends": "sum",
                "clicks": "sum",
                "revenue": "sum",
                "unsubscribes": "sum",
                "estimated_spend": "sum",
            })
            .sort_values("date")
            .reset_index(drop=True)
        )
        logger.info(
            "Postscript: %d days, spend≈%.2f, revenue=%.2f",
            len(df),
            df["estimated_spend"].sum(),
            df["revenue"].sum(),
        )
        return df
