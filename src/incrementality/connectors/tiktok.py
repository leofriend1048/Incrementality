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

from incrementality.connectors.retry import request_with_retry

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


# ---------------------------------------------------------------------------
# State → DMA crosswalk
# ---------------------------------------------------------------------------

# Population-weighted mapping: US state (TikTok province_name values) →
# list of (DMA name, population_weight) tuples.
# Weights are derived from 2020 US Census population estimates.
# DMAs that span multiple states have fractional weights.
# Source: Nielsen DMA rankings + Census Bureau population data.
_STATE_TO_DMA: dict[str, list[tuple[str, float]]] = {
    "Alabama":           [("Birmingham AL", 0.52), ("Mobile AL-Pensacola FL", 0.27), ("Huntsville-Decatur AL", 0.21)],
    "Alaska":            [("Anchorage AK", 1.00)],
    "Arizona":           [("Phoenix AZ", 0.81), ("Tucson AZ", 0.19)],
    "Arkansas":          [("Little Rock AR", 0.52), ("Ft. Smith-Fayetteville AR", 0.30), ("Memphis TN", 0.18)],
    "California":        [("Los Angeles CA", 0.53), ("San Francisco-Oakland-San Jose CA", 0.22),
                          ("Sacramento-Stockton-Modesto CA", 0.12), ("San Diego CA", 0.13)],
    "Colorado":          [("Denver CO", 0.82), ("Colorado Springs-Pueblo CO", 0.18)],
    "Connecticut":       [("Hartford-New Haven CT", 0.80), ("New York NY", 0.20)],
    "Delaware":          [("Philadelphia PA", 0.85), ("Baltimore MD", 0.15)],
    "Florida":           [("Miami-Ft. Lauderdale FL", 0.33), ("Tampa-St. Petersburg FL", 0.28),
                          ("Orlando FL", 0.21), ("Jacksonville FL", 0.10), ("West Palm Beach FL", 0.08)],
    "Georgia":           [("Atlanta GA", 0.73), ("Savannah GA", 0.09), ("Augusta GA", 0.09), ("Columbus GA", 0.09)],
    "Hawaii":            [("Honolulu HI", 1.00)],
    "Idaho":             [("Boise ID", 0.72), ("Spokane WA", 0.28)],
    "Illinois":          [("Chicago IL", 0.82), ("Champaign-Springfield IL", 0.12), ("Rockford IL", 0.06)],
    "Indiana":           [("Indianapolis IN", 0.57), ("South Bend-Elkhart IN", 0.21), ("Evansville IN", 0.22)],
    "Iowa":              [("Des Moines-Ames IA", 0.47), ("Cedar Rapids-Waterloo IA", 0.32), ("Davenport IA", 0.21)],
    "Kansas":            [("Kansas City MO", 0.54), ("Wichita KS", 0.46)],
    "Kentucky":          [("Louisville KY", 0.47), ("Lexington KY", 0.30), ("Cincinnati OH", 0.14), ("Paducah KY", 0.09)],
    "Louisiana":         [("New Orleans LA", 0.48), ("Baton Rouge LA", 0.32), ("Shreveport LA", 0.20)],
    "Maine":             [("Portland-Auburn ME", 0.80), ("Bangor ME", 0.20)],
    "Maryland":          [("Baltimore MD", 0.73), ("Washington DC", 0.27)],
    "Massachusetts":     [("Boston MA-Manchester NH", 0.88), ("Springfield MA", 0.12)],
    "Michigan":          [("Detroit MI", 0.59), ("Grand Rapids-Kalamazoo MI", 0.24), ("Flint-Saginaw-Bay City MI", 0.17)],
    "Minnesota":         [("Minneapolis-St. Paul MN", 0.88), ("Duluth MN-Superior WI", 0.12)],
    "Mississippi":       [("Jackson MS", 0.60), ("Mobile AL-Pensacola FL", 0.25), ("Memphis TN", 0.15)],
    "Missouri":          [("St. Louis MO", 0.47), ("Kansas City MO", 0.43), ("Springfield MO", 0.10)],
    "Montana":           [("Billings MT", 0.60), ("Missoula MT", 0.40)],
    "Nebraska":          [("Omaha NE", 0.72), ("Lincoln NE", 0.28)],
    "Nevada":            [("Las Vegas NV", 0.82), ("Reno NV", 0.18)],
    "New Hampshire":     [("Boston MA-Manchester NH", 0.75), ("Burlington VT-Plattsburgh NY", 0.25)],
    "New Jersey":        [("New York NY", 0.80), ("Philadelphia PA", 0.20)],
    "New Mexico":        [("Albuquerque-Santa Fe NM", 0.88), ("El Paso TX", 0.12)],
    "New York":          [("New York NY", 0.73), ("Albany-Schenectady-Troy NY", 0.10),
                          ("Buffalo NY", 0.09), ("Rochester NY", 0.08)],
    "North Carolina":    [("Charlotte NC", 0.31), ("Raleigh-Durham NC", 0.28), ("Greensboro NC", 0.21),
                          ("Wilmington NC", 0.11), ("Greenville-New Bern-Washington NC", 0.09)],
    "North Dakota":      [("Fargo-Valley City ND", 0.72), ("Minot-Bismarck ND", 0.28)],
    "Ohio":              [("Cleveland-Akron OH", 0.35), ("Columbus OH", 0.27), ("Cincinnati OH", 0.21),
                          ("Dayton OH", 0.12), ("Toledo OH", 0.05)],
    "Oklahoma":          [("Oklahoma City OK", 0.58), ("Tulsa OK", 0.42)],
    "Oregon":            [("Portland OR", 0.82), ("Eugene OR", 0.18)],
    "Pennsylvania":      [("Philadelphia PA", 0.47), ("Pittsburgh PA", 0.32), ("Harrisburg-Lancaster PA", 0.13),
                          ("Wilkes Barre-Scranton PA", 0.08)],
    "Rhode Island":      [("Providence RI-New Bedford MA", 1.00)],
    "South Carolina":    [("Charleston SC", 0.35), ("Greenville-Spartanburg SC", 0.35),
                          ("Columbia SC", 0.30)],
    "South Dakota":      [("Sioux Falls SD", 0.72), ("Rapid City SD", 0.28)],
    "Tennessee":         [("Nashville TN", 0.39), ("Memphis TN", 0.25), ("Knoxville TN", 0.18),
                          ("Chattanooga TN", 0.10), ("Tri-Cities TN-VA", 0.08)],
    "Texas":             [("Dallas-Ft. Worth TX", 0.38), ("Houston TX", 0.30), ("San Antonio TX", 0.12),
                          ("Austin TX", 0.10), ("El Paso TX", 0.05), ("Waco-Temple-Bryan TX", 0.03),
                          ("Corpus Christi TX", 0.02)],
    "Utah":              [("Salt Lake City UT", 0.88), ("St. George UT", 0.12)],
    "Vermont":           [("Burlington VT-Plattsburgh NY", 1.00)],
    "Virginia":          [("Washington DC", 0.45), ("Norfolk-Portsmouth VA", 0.28),
                          ("Richmond VA", 0.20), ("Roanoke-Lynchburg VA", 0.07)],
    "Washington":        [("Seattle-Tacoma WA", 0.80), ("Spokane WA", 0.20)],
    "West Virginia":     [("Charleston-Huntington WV", 0.58), ("Pittsburgh PA", 0.25), ("Roanoke-Lynchburg VA", 0.17)],
    "Wisconsin":         [("Milwaukee WI", 0.56), ("Madison WI", 0.24), ("Green Bay WI", 0.20)],
    "Wyoming":           [("Denver CO", 0.55), ("Salt Lake City UT", 0.30), ("Billings MT", 0.15)],
}

# Alias map for TikTok province_name variations that differ from census names
_PROVINCE_ALIAS: dict[str, str] = {
    "District of Columbia": "Maryland",  # route DC spend to MD→DC DMA
    "DC": "Maryland",
    "Puerto Rico": "Miami-Ft. Lauderdale FL",  # not a state; map to closest major market
}


def state_spend_to_dma(
    state_df: "pd.DataFrame",
    state_col: str = "region",
    spend_col: str = "spend",
    date_col: str = "date",
) -> "pd.DataFrame":
    """Disaggregate state-level TikTok spend to DMA level using population weights.

    Takes the region-level spend DataFrame returned by
    ``TikTokConnector.get_daily_spend_by_region()`` and distributes each
    state's daily spend across the DMAs that overlap that state, weighted
    by the relative population each DMA draws from the state.

    Parameters
    ----------
    state_df : pd.DataFrame
        State-level spend DataFrame with at least ``date``, ``region``, and
        ``spend`` columns (as returned by the TikTok connector).
    state_col : str
        Name of the column containing state names.
    spend_col : str
        Name of the spend column.
    date_col : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        DMA-level spend with columns: ``date``, ``dma``, ``spend``, and
        all original numeric columns proportionally distributed.

    Notes
    -----
    - States not in the crosswalk are allocated 100 % to a synthetic
      "Unknown" DMA so that total spend is conserved.
    - This is a population-weight approximation; geo holdout experiments
      on specific DMAs provide causal ground truth for calibration.
    """
    import pandas as pd

    numeric_cols = [c for c in state_df.columns if c not in (date_col, state_col)
                    and pd.api.types.is_numeric_dtype(state_df[c])]

    rows = []
    for _, row in state_df.iterrows():
        state = str(row[state_col])
        state = _PROVINCE_ALIAS.get(state, state)
        dma_weights = _STATE_TO_DMA.get(state)

        if not dma_weights:
            # Unknown state: preserve spend in an "Unknown" bucket
            new_row: dict = {date_col: row[date_col], "dma": f"Unknown ({state})"}
            for c in numeric_cols:
                new_row[c] = float(row[c])
            rows.append(new_row)
            continue

        for dma_name, weight in dma_weights:
            new_row = {date_col: row[date_col], "dma": dma_name}
            for c in numeric_cols:
                new_row[c] = float(row[c]) * weight
            rows.append(new_row)

    if not rows:
        return pd.DataFrame(columns=[date_col, "dma"] + numeric_cols)

    result = pd.DataFrame(rows)
    # Aggregate: multiple states may feed the same DMA on the same date
    agg = {c: "sum" for c in numeric_cols}
    result = result.groupby([date_col, "dma"], as_index=False).agg(agg)
    return result.sort_values([date_col, "dma"]).reset_index(drop=True)
