"""Tatari TV measurement & attribution API connector.

Tatari is a TV advertising measurement platform that provides DMA-level
spend, airtime, and attribution data for linear, streaming (OTT/CTV),
and addressable TV placements.

Unlike digital channels, Tatari provides true DMA-level geographic
breakdowns for TV airings and their estimated reach/impact.

Typical usage::

    connector = TatariConnector(config)
    df = connector.get_daily_spend_by_dma(
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

from incrementality.config import TatariConfig
from incrementality.connectors.retry import request_with_retry

logger = logging.getLogger(__name__)

# Tatari API uses versioned endpoints under /v1
_API_VERSION = "v1"

# Maximum date range per request
_MAX_WINDOW_DAYS = 90


class TatariAPIError(Exception):
    """Raised for Tatari API-level errors."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Tatari API error {status_code}: {message}")


class TatariConnector:
    """Connects to the Tatari measurement API for TV spend by DMA.

    Tatari provides daily TV spend and estimated reach broken down by
    DMA, making it a first-class geo channel in the MMM alongside digital.

    Args:
        config: TatariConfig with api_key and account_id.
    """

    DEFAULT_TIMEOUT = 30

    def __init__(self, config: TatariConfig) -> None:
        self.config = config
        self.base_url = f"{config.base_url.rstrip('/')}/{_API_VERSION}"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "X-Account-ID": config.account_id,
        })

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = request_with_retry(
            self.session, "GET", url,
            params=params or {},
            timeout=self.DEFAULT_TIMEOUT,
        )
        if not resp.ok:
            raise TatariAPIError(resp.status_code, resp.text[:500])
        return resp.json()

    def get_daily_spend_by_dma(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch daily TV spend and airtime broken down by DMA.

        Retrieves linear + streaming TV airings with their estimated
        local-market spend, GRPs, and reach per DMA.

        Args:
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            DataFrame with columns:
                date, dma_code, dma_name, spend, grps, impressions,
                airings, placement_type (linear|streaming|addressable)
        """
        all_rows: list[dict] = []
        chunk_start = start_date

        while chunk_start <= end_date:
            chunk_end = min(
                chunk_start + timedelta(days=_MAX_WINDOW_DAYS - 1), end_date
            )
            logger.info("Tatari: fetching %s → %s", chunk_start, chunk_end)

            data = self._get(
                f"accounts/{self.config.account_id}/airings",
                params={
                    "start_date": chunk_start.isoformat(),
                    "end_date": chunk_end.isoformat(),
                    "breakdown": "dma",
                    "granularity": "daily",
                },
            )

            for row in data.get("data", []):
                all_rows.append({
                    "date": pd.to_datetime(row.get("date")).date(),
                    "dma_code": str(row.get("dma_code", "")),
                    "dma_name": row.get("dma_name", ""),
                    "spend": float(row.get("spend", 0) or 0),
                    "grps": float(row.get("grps", 0) or 0),
                    "impressions": int(row.get("impressions", 0) or 0),
                    "airings": int(row.get("airings", 0) or 0),
                    "placement_type": row.get("placement_type", "linear"),
                })

            chunk_start = chunk_end + timedelta(days=1)

        if not all_rows:
            logger.warning("Tatari: no data for %s → %s", start_date, end_date)
            return pd.DataFrame(
                columns=["date", "dma_code", "dma_name", "spend", "grps",
                         "impressions", "airings", "placement_type"]
            )

        df = pd.DataFrame(all_rows)
        df = df.sort_values(["date", "dma_code"]).reset_index(drop=True)
        logger.info(
            "Tatari: %d rows, spend=%.2f, %s → %s",
            len(df), df["spend"].sum(),
            df["date"].min(), df["date"].max(),
        )
        return df

    def get_daily_national_spend(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Aggregate Tatari DMA spend to national totals.

        Returns columns: date, spend, grps, impressions, airings.
        """
        df = self.get_daily_spend_by_dma(start_date, end_date)
        if df.empty:
            return pd.DataFrame(
                columns=["date", "spend", "grps", "impressions", "airings"]
            )
        return (
            df.groupby("date", as_index=False)[
                ["spend", "grps", "impressions", "airings"]
            ]
            .sum()
            .sort_values("date")
            .reset_index(drop=True)
        )
