"""Klaviyo email (and SMS) marketing API connector.

Klaviyo is the primary email marketing platform for DTC brands.
This connector pulls campaign and flow performance metrics including
sends, opens, clicks, and attributed revenue.

Email data is national (no DMA breakdown available) and is fed into
the MMM as the ``klaviyo_email`` channel.

Klaviyo API: https://developers.klaviyo.com/en/reference/api-overview
Uses API revision 2024-10-15.

Typical usage::

    connector = KlaviyoConnector(config)
    df = connector.get_daily_performance(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import requests

from incrementality.config import KlaviyoConfig
from incrementality.connectors.retry import request_with_retry

logger = logging.getLogger(__name__)

_BASE_URL = "https://a.klaviyo.com/api"
_API_REVISION = "2024-10-15"

# Page size for list endpoints
_PAGE_SIZE = 50  # Klaviyo max is 50 for campaigns


class KlaviyoAPIError(Exception):
    """Raised for Klaviyo API-level errors."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Klaviyo API error {status_code}: {message}")


class KlaviyoConnector:
    """Connects to the Klaviyo Marketing API.

    Pulls daily email campaign and flow performance for use as an
    MMM channel input.

    Args:
        config: KlaviyoConfig with private_key.
    """

    DEFAULT_TIMEOUT = 30

    def __init__(self, config: KlaviyoConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Klaviyo-API-Key {config.private_key}",
            "revision": _API_REVISION,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{_BASE_URL}/{path.lstrip('/')}"
        resp = request_with_retry(
            self.session, "GET", url,
            params=params or {},
            timeout=self.DEFAULT_TIMEOUT,
        )
        if not resp.ok:
            raise KlaviyoAPIError(resp.status_code, resp.text[:500])
        return resp.json()

    def get_campaigns(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict]:
        """Return all email campaigns.

        Args:
            start_date: Filter campaigns sent on or after this date.
            end_date: Filter campaigns sent on or before this date.

        Returns:
            List of Klaviyo campaign resource objects.
        """
        params: dict[str, Any] = {
            "page[size]": _PAGE_SIZE,
            "filter": "equals(messages.channel,'email')",
            "sort": "-send_time",
        }
        if start_date:
            # ISO 8601 with timezone
            start_iso = datetime.combine(start_date, datetime.min.time()).replace(
                tzinfo=timezone.utc
            ).isoformat()
            params["filter"] = (
                f"greater-or-equal(send_time,{start_iso}),"
                f"equals(messages.channel,'email')"
            )

        all_campaigns: list[dict] = []
        next_cursor: str | None = None

        while True:
            if next_cursor:
                params["page[cursor]"] = next_cursor

            data = self._get("campaigns", params)
            items = data.get("data", [])
            all_campaigns.extend(items)

            # Klaviyo uses cursor-based pagination
            links = data.get("links", {})
            next_link = links.get("next")
            if not next_link:
                break
            # Extract cursor from next URL
            if "page%5Bcursor%5D=" in next_link:
                next_cursor = next_link.split("page%5Bcursor%5D=")[1].split("&")[0]
            elif "page[cursor]=" in next_link:
                next_cursor = next_link.split("page[cursor]=")[1].split("&")[0]
            else:
                break

        # Filter by end_date client-side if needed
        if end_date:
            all_campaigns = [
                c for c in all_campaigns
                if _campaign_send_date(c) and _campaign_send_date(c) <= end_date  # type: ignore[operator]
            ]

        logger.info("Klaviyo: %d campaigns retrieved.", len(all_campaigns))
        return all_campaigns

    def get_campaign_aggregate_metrics(self, campaign_id: str) -> dict[str, Any]:
        """Return aggregate send metrics for a single campaign.

        Uses the campaign-message endpoint to get delivered/opened/clicked/revenue.

        Returns:
            Dict with delivered, opened, clicked, revenue, unsubscribed.
        """
        # Get campaign messages (Klaviyo splits campaign → message → metrics)
        data = self._get(
            f"campaigns/{campaign_id}/campaign-messages",
        )
        messages = data.get("data", [])
        if not messages:
            return {}

        # Aggregate across all messages (usually 1 per campaign)
        totals: dict[str, float] = {
            "delivered": 0, "opened": 0, "clicked": 0,
            "revenue": 0, "unsubscribed": 0,
        }
        for msg in messages:
            attrs = msg.get("attributes", {})
            counts = attrs.get("counts", {})
            totals["delivered"] += int(counts.get("delivered", 0) or 0)
            totals["opened"] += int(counts.get("opened_unique", 0) or 0)
            totals["clicked"] += int(counts.get("clicked_unique", 0) or 0)
            totals["revenue"] += float(counts.get("revenue", 0) or 0)
            totals["unsubscribed"] += int(counts.get("unsubscribed", 0) or 0)

        return totals

    def get_daily_performance(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Aggregate daily email performance across all campaigns.

        Fetches campaigns in the window and groups metrics by send date.
        Klaviyo does not offer a single daily aggregate endpoint, so this
        is computed client-side from per-campaign metric lookups.

        Args:
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            DataFrame with columns:
                date, campaigns_sent, delivered, opened, clicked,
                revenue, unsubscribed, estimated_spend
        """
        campaigns = self.get_campaigns(start_date=start_date, end_date=end_date)

        rows: list[dict] = []
        for campaign in campaigns:
            send_date = _campaign_send_date(campaign)
            if send_date is None:
                continue
            if not (start_date <= send_date <= end_date):
                continue

            campaign_id = campaign.get("id", "")
            try:
                metrics = self.get_campaign_aggregate_metrics(campaign_id)
            except KlaviyoAPIError as exc:
                logger.warning(
                    "Failed to get metrics for campaign %s: %s", campaign_id, exc
                )
                continue

            delivered = int(metrics.get("delivered", 0))
            # Klaviyo pricing: ~$20/1000 emails = $0.02 per email
            estimated_spend = delivered * 0.02

            rows.append({
                "date": send_date,
                "campaigns_sent": 1,
                "delivered": delivered,
                "opened": int(metrics.get("opened", 0)),
                "clicked": int(metrics.get("clicked", 0)),
                "revenue": float(metrics.get("revenue", 0)),
                "unsubscribed": int(metrics.get("unsubscribed", 0)),
                "estimated_spend": estimated_spend,
            })

        if not rows:
            logger.warning(
                "Klaviyo: no campaign data for %s → %s", start_date, end_date
            )
            return pd.DataFrame(
                columns=["date", "campaigns_sent", "delivered", "opened",
                         "clicked", "revenue", "unsubscribed", "estimated_spend"]
            )

        df = pd.DataFrame(rows)
        df = (
            df.groupby("date", as_index=False)
            .agg({
                "campaigns_sent": "sum",
                "delivered": "sum",
                "opened": "sum",
                "clicked": "sum",
                "revenue": "sum",
                "unsubscribed": "sum",
                "estimated_spend": "sum",
            })
            .sort_values("date")
            .reset_index(drop=True)
        )
        logger.info(
            "Klaviyo: %d days, delivered=%d, revenue=%.2f",
            len(df),
            int(df["delivered"].sum()),
            df["revenue"].sum(),
        )
        return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _campaign_send_date(campaign: dict) -> date | None:
    """Extract the send date from a Klaviyo campaign resource object."""
    attrs = campaign.get("attributes", {})
    send_time = attrs.get("send_time") or attrs.get("scheduled_at")
    if not send_time:
        return None
    try:
        return pd.to_datetime(send_time).date()
    except Exception:
        return None
