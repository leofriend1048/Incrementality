"""TikTok Shop API connector.

TikTok Shop is TikTok's native commerce platform, distinct from TikTok Ads.
This connector pulls GMV (Gross Merchandise Value), order counts, and
affiliated creator spend from the TikTok Shop Open Platform API.

Data is national (US) — no DMA breakdown is available from TikTok Shop.
This feeds the ``tiktok_shop`` MMM channel.

API docs: https://partner.tiktokshop.com/docv2/page/

Typical usage::

    connector = TikTokShopConnector(config)
    df = connector.get_daily_gmv(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

from incrementality.config import TikTokShopConfig
from incrementality.connectors.retry import request_with_retry

logger = logging.getLogger(__name__)

_BASE_URL = "https://open-api.tiktokglobalshop.com"
_API_VERSION = "202309"

# Max date range per request (TikTok Shop analytics)
_MAX_WINDOW_DAYS = 30


class TikTokShopAPIError(Exception):
    """Raised for TikTok Shop API-level errors."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"TikTok Shop API error {code}: {message}")


class TikTokShopConnector:
    """Connects to the TikTok Shop Open Platform API.

    Pulls GMV, orders, and affiliate-attributed spend for use as the
    ``tiktok_shop`` MMM channel.

    Args:
        config: TikTokShopConfig with app credentials.
    """

    DEFAULT_TIMEOUT = 30

    def __init__(self, config: TikTokShopConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _sign_request(
        self,
        path: str,
        params: dict[str, Any],
        timestamp: int,
    ) -> str:
        """Generate HMAC-SHA256 signature for TikTok Shop API requests.

        TikTok Shop requires every request to be signed using the
        app_secret as the HMAC key and a canonical string derived from
        the path and sorted query parameters.

        Args:
            path: API path (e.g. '/api/analytics/overview').
            params: Query parameters to include in signature.
            timestamp: Unix epoch seconds.

        Returns:
            Hex-encoded HMAC-SHA256 signature string.
        """
        sorted_params = sorted(params.items())
        param_str = "".join(f"{k}{v}" for k, v in sorted_params)
        sign_str = f"{self.config.app_secret}{path}{param_str}{timestamp}{self.config.app_secret}"
        return hmac.new(
            self.config.app_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = params or {}
        timestamp = int(time.time())
        params.update({
            "app_key": self.config.app_key,
            "access_token": self.config.access_token,
            "timestamp": timestamp,
            "version": _API_VERSION,
        })
        params["sign"] = self._sign_request(path, params, timestamp)

        url = f"{_BASE_URL}{path}"
        resp = request_with_retry(
            self.session, "GET", url,
            params=params,
            timeout=self.DEFAULT_TIMEOUT,
        )
        if not resp.ok:
            raise TikTokShopAPIError(resp.status_code, resp.text[:500])

        data = resp.json()
        code = data.get("code", 0)
        if code != 0:
            raise TikTokShopAPIError(code, data.get("message", "Unknown error"))
        return data.get("data", {})

    def get_daily_gmv(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch daily Gross Merchandise Value and order metrics.

        Pulls from the TikTok Shop analytics overview endpoint for the
        shop linked to the access token.

        Args:
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            DataFrame with columns:
                date, gmv, orders, units_sold, affiliate_commission,
                creator_count, estimated_spend
        """
        all_rows: list[dict] = []
        chunk_start = start_date

        while chunk_start <= end_date:
            chunk_end = min(
                chunk_start + timedelta(days=_MAX_WINDOW_DAYS - 1), end_date
            )
            logger.info("TikTok Shop: fetching %s → %s", chunk_start, chunk_end)

            data = self._get(
                "/api/analytics/overview",
                params={
                    "start_date": chunk_start.strftime("%Y%m%d"),
                    "end_date": chunk_end.strftime("%Y%m%d"),
                    "granularity": "DAILY",
                    "shop_id": self.config.shop_id,
                },
            )

            for row in data.get("list", []):
                gmv = float(row.get("gmv", 0) or 0)
                # Estimate spend from affiliate commission rate (~15% typical)
                commission = float(row.get("affiliate_commission", gmv * 0.15) or 0)

                all_rows.append({
                    "date": pd.to_datetime(str(row.get("date", ""))).date(),
                    "gmv": gmv,
                    "orders": int(row.get("orders", 0) or 0),
                    "units_sold": int(row.get("units_sold", 0) or 0),
                    "affiliate_commission": commission,
                    "creator_count": int(row.get("creator_count", 0) or 0),
                    "estimated_spend": commission,
                })

            chunk_start = chunk_end + timedelta(days=1)

        if not all_rows:
            logger.warning(
                "TikTok Shop: no data for %s → %s", start_date, end_date
            )
            return pd.DataFrame(
                columns=["date", "gmv", "orders", "units_sold",
                         "affiliate_commission", "creator_count", "estimated_spend"]
            )

        df = pd.DataFrame(all_rows)
        df = df.sort_values("date").reset_index(drop=True)
        logger.info(
            "TikTok Shop: %d days, GMV=%.2f, estimated_spend=%.2f",
            len(df), df["gmv"].sum(), df["estimated_spend"].sum(),
        )
        return df
