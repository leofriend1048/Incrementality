"""TikTok Ads API v1.3 connector.

Pulls ad spend and delivery data from the TikTok Business API broken down
by region/state.  TikTok does not provide DMA-level geo breakdowns; the
finest geographic granularity available is region (state).

Typical usage::

    connector = TikTokConnector(
        app_id="my_app_id",
        secret="my_secret",
        access_token="my_access_token",
    )
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

from incrementality.connectors.retry import request_with_retry, RetryableRequestError

logger = logging.getLogger(__name__)

_BASE_URL = "https://business-api.tiktok.com/open_api/v1.3/"

# TikTok-specific error codes that should trigger token-related handling
_TOKEN_EXPIRED_CODES = {40001, 40002, 40100, 40101, 40102}
# TikTok rate-limit error code
_RATE_LIMIT_CODE = 40100

# Maximum rows TikTok returns per page
_PAGE_SIZE = 1000

# Dimension/metric names used in the reporting API
_GEO_DIMENSIONS = ["stat_time_day", "province_id", "province_name"]
_SPEND_METRICS = ["spend", "impressions", "video_play_actions", "video_watched_2s"]


class TikTokAPIError(Exception):
    """Raised for TikTok API-level errors returned inside a 200 response body."""

    def __init__(self, code: int, message: str, request_id: str | None = None):
        self.code = code
        self.message = message
        self.request_id = request_id
        super().__init__(f"TikTok API error {code}: {message} (request_id={request_id})")


class TikTokTokenExpiredError(TikTokAPIError):
    """Raised when the TikTok access token has expired or is invalid."""


class TikTokConnector:
    """Connector for the TikTok Business API v1.3.

    Pulls spend, impressions, and video engagement metrics at the
    region/state level.  TikTok only provides region (state) granularity
    for geographic breakdowns — DMA-level data is not available.

    Args:
        app_id: TikTok app ID from the TikTok for Business developer portal.
        secret: App secret corresponding to ``app_id``.
        access_token: OAuth access token for the ad account.
    """

    BASE_URL = _BASE_URL
    DEFAULT_TIMEOUT = 30  # seconds per request

    def __init__(self, app_id: str, secret: str, access_token: str) -> None:
        self.app_id = app_id
        self.secret = secret
        self.access_token = access_token

        self.session = requests.Session()
        self.session.headers.update({
            "Access-Token": access_token,
            "Content-Type": "application/json",
        })

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        """Build a full API URL from a relative path."""
        return f"{self.BASE_URL}{path.lstrip('/')}"

    def _check_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Inspect the TikTok JSON envelope and raise on API-level errors.

        TikTok wraps every response in::

            {
                "code": 0,          # 0 = success
                "message": "OK",
                "request_id": "...",
                "data": { ... }
            }

        Args:
            data: Parsed JSON body from a TikTok API response.

        Returns:
            The inner ``data`` dict on success.

        Raises:
            TikTokTokenExpiredError: If the error code indicates token expiry.
            TikTokAPIError: For any other non-zero error code.
        """
        code = data.get("code", 0)
        message = data.get("message", "")
        request_id = data.get("request_id")

        if code == 0:
            return data.get("data", {})

        if code in _TOKEN_EXPIRED_CODES:
            logger.error(
                "TikTok access token expired or invalid (code=%s, request_id=%s).",
                code, request_id,
            )
            raise TikTokTokenExpiredError(code, message, request_id)

        logger.error(
            "TikTok API error: code=%s message=%r request_id=%s",
            code, message, request_id,
        )
        raise TikTokAPIError(code, message, request_id)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a GET request and return the unwrapped ``data`` payload.

        Args:
            path: API path relative to ``BASE_URL``.
            params: Query parameters.

        Returns:
            The ``data`` field from the TikTok JSON envelope.

        Raises:
            TikTokAPIError: On API-level errors.
            RetryableRequestError: When retries are exhausted on network/HTTP errors.
        """
        url = self._url(path)
        resp = request_with_retry(
            self.session, "GET", url,
            params=params,
            timeout=self.DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return self._check_response(resp.json())

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a POST request and return the unwrapped ``data`` payload.

        Args:
            path: API path relative to ``BASE_URL``.
            payload: JSON body.

        Returns:
            The ``data`` field from the TikTok JSON envelope.

        Raises:
            TikTokAPIError: On API-level errors.
            RetryableRequestError: When retries are exhausted on network/HTTP errors.
        """
        url = self._url(path)
        resp = request_with_retry(
            self.session, "POST", url,
            json=payload,
            timeout=self.DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return self._check_response(resp.json())

    def _get_ad_account_id(self) -> str:
        """Retrieve the first advertiser ID associated with the access token.

        Returns:
            Advertiser (ad account) ID as a string.

        Raises:
            ValueError: If no advertiser accounts are found.
        """
        data = self._get(
            "oauth2/advertiser/get/",
            params={"app_id": self.app_id, "secret": self.secret},
        )
        accounts: list[dict] = data.get("list", [])
        if not accounts:
            raise ValueError(
                "No advertiser accounts found for the provided app_id/secret/access_token."
            )
        account_id = str(accounts[0]["advertiser_id"])
        logger.debug("Using TikTok advertiser_id=%s", account_id)
        return account_id

    def _paginate_report(
        self,
        advertiser_id: str,
        start_date: date,
        end_date: date,
        dimensions: list[str],
        metrics: list[str],
    ) -> list[dict[str, Any]]:
        """Fetch all pages of a TikTok integrated reporting request.

        TikTok's reporting endpoint uses cursor-based pagination via
        ``page`` / ``page_size``.  All pages are fetched and concatenated.

        Args:
            advertiser_id: The TikTok advertiser (ad account) ID.
            start_date: Inclusive start of the reporting window.
            end_date: Inclusive end of the reporting window.
            dimensions: List of dimension keys (e.g. ``["stat_time_day"]``).
            metrics: List of metric keys (e.g. ``["spend"]``).

        Returns:
            Flat list of row dicts from all pages.
        """
        all_rows: list[dict[str, Any]] = []
        page = 1

        while True:
            payload = {
                "advertiser_id": advertiser_id,
                "report_type": "AUDIENCE",
                "data_level": "AUCTION_ADVERTISER",
                "dimensions": dimensions,
                "metrics": metrics,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "page": page,
                "page_size": _PAGE_SIZE,
                "lifetime": False,
            }
            data = self._post("report/integrated/get/", payload)

            rows: list[dict] = data.get("list", [])
            all_rows.extend(rows)

            page_info: dict = data.get("page_info", {})
            total_pages = page_info.get("total_page", 1)

            logger.debug(
                "TikTok report: page %d/%d, fetched %d rows so far.",
                page, total_pages, len(all_rows),
            )

            if page >= total_pages:
                break
            page += 1

        return all_rows

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_daily_spend_by_objective(
        self,
        start_date: date,
        end_date: date,
        objective_type: str = "GMV_MAX",
    ) -> pd.DataFrame:
        """Fetch daily spend filtered to a specific campaign objective type.

        Use this to isolate GMV Max campaigns (objective_type='GMV_MAX')
        from general TikTok performance campaigns for separate MMM channels.

        Args:
            start_date: Inclusive start date.
            end_date: Inclusive end date.
            objective_type: TikTok campaign objective (e.g. 'GMV_MAX',
                'VIDEO_VIEWS', 'CONVERSIONS', 'REACH').

        Returns:
            DataFrame with columns:
                date, region, objective_type, spend, impressions,
                video_views, video_completions
        """
        advertiser_id = self._get_ad_account_id()

        # First get campaign IDs matching the objective
        campaigns = self.get_campaign_list()
        matching_ids = [
            str(c["campaign_id"])
            for c in campaigns
            if c.get("objective_type", "").upper() == objective_type.upper()
        ]
        if not matching_ids:
            logger.info(
                "TikTok: no campaigns found with objective=%s", objective_type
            )
            return pd.DataFrame(
                columns=["date", "region", "objective_type", "spend",
                         "impressions", "video_views", "video_completions"]
            )

        all_rows: list[dict] = []
        chunk_start = start_date

        while chunk_start <= end_date:
            chunk_end = min(chunk_start + timedelta(days=29), end_date)

            rows = self._paginate_report(
                advertiser_id=advertiser_id,
                start_date=chunk_start,
                end_date=chunk_end,
                dimensions=["stat_time_day", "province_id", "province_name",
                            "campaign_id"],
                metrics=["spend", "impressions", "video_play_actions",
                         "video_views_p100"],
            )

            for row in rows:
                dims = row.get("dimensions", {})
                if str(dims.get("campaign_id", "")) not in matching_ids:
                    continue
                mets = row.get("metrics", {})
                all_rows.append({
                    "date": pd.to_datetime(dims.get("stat_time_day")).date(),
                    "region": dims.get("province_name", ""),
                    "objective_type": objective_type,
                    "spend": float(mets.get("spend", 0) or 0),
                    "impressions": int(mets.get("impressions", 0) or 0),
                    "video_views": int(mets.get("video_play_actions", 0) or 0),
                    "video_completions": int(mets.get("video_views_p100", 0) or 0),
                })

            chunk_start = chunk_end + timedelta(days=1)

        if not all_rows:
            return pd.DataFrame(
                columns=["date", "region", "objective_type", "spend",
                         "impressions", "video_views", "video_completions"]
            )

        df = pd.DataFrame(all_rows)
        # Aggregate across campaigns for same date+region
        df = (
            df.groupby(["date", "region", "objective_type"], as_index=False)
            .agg({
                "spend": "sum",
                "impressions": "sum",
                "video_views": "sum",
                "video_completions": "sum",
            })
            .sort_values(["date", "region"])
            .reset_index(drop=True)
        )
        logger.info(
            "TikTok %s: %d rows, spend=%.2f", objective_type, len(df), df["spend"].sum()
        )
        return df

    def get_ad_account_info(self) -> dict[str, Any]:
        """Return metadata for the ad account linked to this access token.

        Retrieves account name, status, currency, timezone, and other
        top-level attributes from the TikTok advertiser info endpoint.

        Returns:
            Dict of account metadata keyed by TikTok field names.

        Raises:
            TikTokAPIError: On API-level errors.
            ValueError: If no accounts are associated with the token.
        """
        data = self._get(
            "oauth2/advertiser/get/",
            params={"app_id": self.app_id, "secret": self.secret},
        )
        accounts: list[dict] = data.get("list", [])
        if not accounts:
            raise ValueError(
                "No advertiser accounts found for the provided app_id/secret/access_token."
            )
        account = accounts[0]
        logger.info(
            "TikTok ad account: advertiser_id=%s name=%r status=%s currency=%s",
            account.get("advertiser_id"),
            account.get("advertiser_name"),
            account.get("status"),
            account.get("currency"),
        )
        return account

    def get_campaign_list(self) -> list[dict[str, Any]]:
        """Return all campaigns for the ad account.

        Paginates through the campaign list endpoint and returns every
        campaign record.

        Returns:
            List of campaign dicts with keys such as ``campaign_id``,
            ``campaign_name``, ``status``, ``budget``, ``objective_type``.

        Raises:
            TikTokAPIError: On API-level errors.
        """
        advertiser_id = self._get_ad_account_id()
        all_campaigns: list[dict[str, Any]] = []
        page = 1

        while True:
            data = self._get(
                "campaign/get/",
                params={
                    "advertiser_id": advertiser_id,
                    "page": page,
                    "page_size": _PAGE_SIZE,
                    "fields": (
                        '["campaign_id","campaign_name","status",'
                        '"budget","objective_type","create_time","modify_time"]'
                    ),
                },
            )
            campaigns: list[dict] = data.get("list", [])
            all_campaigns.extend(campaigns)

            page_info: dict = data.get("page_info", {})
            total_pages = page_info.get("total_page", 1)

            logger.debug(
                "TikTok campaign list: page %d/%d, %d campaigns total.",
                page, total_pages, len(all_campaigns),
            )

            if page >= total_pages:
                break
            page += 1

        logger.info("Retrieved %d TikTok campaigns.", len(all_campaigns))
        return all_campaigns

    def get_daily_spend_by_region(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Pull daily spend and video metrics broken down by region/state.

        TikTok provides geographic data at the region (state) level only.
        DMA-level breakdown is not available from the TikTok Ads API.
        Hourly data is aggregated to daily totals server-side by passing
        ``stat_time_day`` as the time dimension.

        Args:
            start_date: Inclusive start date for the reporting window.
            end_date: Inclusive end date for the reporting window (max 30
                days per request; longer windows are split automatically).

        Returns:
            DataFrame with columns:
                - ``date`` (datetime.date)
                - ``region`` (str) — province/state name
                - ``spend`` (float) — total spend in account currency
                - ``impressions`` (int)
                - ``video_views`` (int) — 2-second video views
                - ``video_completions`` (int) — complete video views

        Raises:
            TikTokAPIError: On API-level errors.
            RetryableRequestError: When retries are exhausted.
        """
        advertiser_id = self._get_ad_account_id()

        # TikTok reporting windows are capped at 30 days per request.
        # Split longer date ranges into 30-day chunks.
        all_rows: list[dict[str, Any]] = []
        chunk_start = start_date
        max_window_days = 30

        while chunk_start <= end_date:
            chunk_end = min(chunk_start + timedelta(days=max_window_days - 1), end_date)
            logger.info(
                "Fetching TikTok region spend: %s → %s", chunk_start, chunk_end
            )

            dimensions = ["stat_time_day", "province_id", "province_name"]
            metrics = [
                "spend",
                "impressions",
                "video_play_actions",
                "video_watched_2s",
                "video_views_p100",
            ]

            rows = self._paginate_report(
                advertiser_id=advertiser_id,
                start_date=chunk_start,
                end_date=chunk_end,
                dimensions=dimensions,
                metrics=metrics,
            )
            all_rows.extend(rows)
            chunk_start = chunk_end + timedelta(days=1)

        if not all_rows:
            logger.warning(
                "TikTok returned no spend data for %s → %s.", start_date, end_date
            )
            return pd.DataFrame(
                columns=[
                    "date", "region", "spend", "impressions",
                    "video_views", "video_completions",
                ]
            )

        records = []
        for row in all_rows:
            dims: dict = row.get("dimensions", {})
            mets: dict = row.get("metrics", {})
            records.append({
                "date": pd.to_datetime(dims.get("stat_time_day")).date(),
                "region": dims.get("province_name", ""),
                "spend": float(mets.get("spend", 0) or 0),
                "impressions": int(mets.get("impressions", 0) or 0),
                # video_play_actions = 2-second views (TikTok's primary view metric)
                "video_views": int(mets.get("video_play_actions", 0) or 0),
                # video_views_p100 = 100% completion plays
                "video_completions": int(mets.get("video_views_p100", 0) or 0),
            })

        df = pd.DataFrame(records)
        df = df.sort_values(["date", "region"]).reset_index(drop=True)

        logger.info(
            "TikTok region spend: %d rows, date range %s → %s, "
            "total spend=%.2f",
            len(df),
            df["date"].min() if not df.empty else "N/A",
            df["date"].max() if not df.empty else "N/A",
            df["spend"].sum() if not df.empty else 0.0,
        )
        return df
