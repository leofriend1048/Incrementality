"""Northbeam MTA API connector.

Pulls multi-touch attribution (MTA) output from the Northbeam API and
translates it into Meridian MMM prior calibration inputs.

Northbeam provides channel-level attributed revenue and conversions across
configurable attribution windows (7-day, 28-day, etc.) as well as new vs.
returning customer splits.

Channel discount factors (from PRD) applied in ``compute_prior_means``:

    Meta Performance    ×0.65
    Meta Awareness      ×0.60
    Google Brand Search ×0.85
    Google Non-Brand    ×0.75
    TikTok              ×0.65

These factors deflate MTA-derived ROI priors to correct for the known
over-attribution bias in last-touch-adjacent MTA models.

Typical usage::

    nb = NorthbeamConnector(api_key="...", account_id="...")
    attribution_df = nb.get_channel_attribution(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        attribution_window=28,
    )
    prior_means = nb.compute_prior_means(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        total_revenue=500_000.0,
    )
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
import requests

from incrementality.connectors.retry import request_with_retry, RetryableRequestError

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.northbeam.io/v1/"

# Attribution windows (days) supported by the Northbeam API
_VALID_ATTRIBUTION_WINDOWS = {1, 7, 14, 28, 30, 90}

# Channel discount factors per PRD: applied to MTA-derived ROI estimates
# to correct for MTA over-attribution bias before use as Meridian priors.
_CHANNEL_DISCOUNT_FACTORS: dict[str, float] = {
    "meta_performance":     0.65,
    "meta_awareness":       0.60,
    "google_brand_search":  0.85,
    "google_non_brand":     0.75,
    "tiktok":               0.65,
}

# Page size for paginated list endpoints
_PAGE_SIZE = 500


class NorthbeamAPIError(Exception):
    """Raised for Northbeam API-level errors (non-2xx or error payload)."""

    def __init__(self, status_code: int, message: str, request_id: str | None = None):
        self.status_code = status_code
        self.message = message
        self.request_id = request_id
        super().__init__(
            f"Northbeam API error {status_code}: {message} (request_id={request_id})"
        )


class NorthbeamConnector:
    """Connector for the Northbeam MTA API.

    Provides methods to retrieve channel-level attributed revenue,
    new/returning customer splits, and to compute Meridian-compatible
    ROI prior means from Northbeam MTA output.

    Args:
        api_key: Northbeam API key (found in account settings).
        account_id: Northbeam account/brand identifier.
    """

    BASE_URL = _BASE_URL
    DEFAULT_TIMEOUT = 45  # seconds; Northbeam aggregation queries can be slow

    def __init__(self, api_key: str, account_id: str) -> None:
        self.api_key = api_key
        self.account_id = account_id

        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        """Build a full API URL from a relative path."""
        return f"{self.BASE_URL}{path.lstrip('/')}"

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a GET request and return the parsed JSON body.

        Args:
            path: API path relative to ``BASE_URL``.
            params: Query parameters to append.

        Returns:
            Parsed JSON response dict.

        Raises:
            NorthbeamAPIError: On non-2xx HTTP responses.
            RetryableRequestError: When retries are exhausted on transient errors.
        """
        url = self._url(path)
        resp = request_with_retry(
            self.session, "GET", url,
            params=params,
            timeout=self.DEFAULT_TIMEOUT,
        )
        if not resp.ok:
            try:
                body = resp.json()
                message = body.get("message") or body.get("error") or resp.text
            except Exception:
                message = resp.text
            request_id = resp.headers.get("x-request-id")
            raise NorthbeamAPIError(resp.status_code, message, request_id)

        return resp.json()

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a POST request and return the parsed JSON body.

        Args:
            path: API path relative to ``BASE_URL``.
            payload: JSON body to send.

        Returns:
            Parsed JSON response dict.

        Raises:
            NorthbeamAPIError: On non-2xx HTTP responses.
            RetryableRequestError: When retries are exhausted on transient errors.
        """
        url = self._url(path)
        resp = request_with_retry(
            self.session, "POST", url,
            json=payload,
            timeout=self.DEFAULT_TIMEOUT,
        )
        if not resp.ok:
            try:
                body = resp.json()
                message = body.get("message") or body.get("error") or resp.text
            except Exception:
                message = resp.text
            request_id = resp.headers.get("x-request-id")
            raise NorthbeamAPIError(resp.status_code, message, request_id)

        return resp.json()

    @staticmethod
    def _normalize_channel_key(channel_name: str) -> str:
        """Convert a Northbeam channel display name to a canonical snake_case key.

        The canonical keys must align with ``_CHANNEL_DISCOUNT_FACTORS``.

        Examples::

            "Meta - Performance"  → "meta_performance"
            "Google Brand Search" → "google_brand_search"
            "TikTok"              → "tiktok"

        Args:
            channel_name: Raw channel name returned by the Northbeam API.

        Returns:
            Lowercase, underscore-separated channel key.
        """
        return (
            channel_name
            .lower()
            .replace(" - ", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_touchpoint_model_type(self) -> str:
        """Return the attribution model type configured for this account.

        Queries the account settings endpoint to determine whether Northbeam
        is using a data-driven (algorithmic) or a rules-based (e.g. linear)
        attribution model.

        Returns:
            ``'data_driven'`` or ``'linear'``.  Returns ``'data_driven'`` as
            the default if the model type is unrecognised, because Northbeam
            defaults to their proprietary data-driven model.

        Raises:
            NorthbeamAPIError: On API errors.
        """
        data = self._get(f"accounts/{self.account_id}/settings")
        model_type: str = (
            data.get("attribution_model", "data_driven").lower().strip()
        )

        if "data" in model_type or "algorithmic" in model_type:
            resolved = "data_driven"
        elif "linear" in model_type:
            resolved = "linear"
        else:
            logger.warning(
                "Unrecognised Northbeam attribution model type %r; "
                "defaulting to 'data_driven'.",
                model_type,
            )
            resolved = "data_driven"

        logger.info(
            "Northbeam attribution model type: %r (raw=%r)", resolved, model_type
        )
        return resolved

    def get_channel_attribution(
        self,
        start_date: date,
        end_date: date,
        attribution_window: int = 28,
    ) -> pd.DataFrame:
        """Pull daily channel-level attributed revenue and conversions.

        Args:
            start_date: Inclusive start date.
            end_date: Inclusive end date.
            attribution_window: Lookback window in days for credit assignment.
                Northbeam supports 1, 7, 14, 28, 30, and 90.

        Returns:
            DataFrame with columns:
                - ``date`` (datetime.date)
                - ``channel`` (str) — canonical snake_case channel name
                - ``attributed_revenue`` (float)
                - ``attributed_conversions`` (int)
                - ``attribution_window`` (int)

        Raises:
            ValueError: If ``attribution_window`` is not supported.
            NorthbeamAPIError: On API errors.
        """
        if attribution_window not in _VALID_ATTRIBUTION_WINDOWS:
            raise ValueError(
                f"attribution_window={attribution_window} is not supported. "
                f"Valid windows: {sorted(_VALID_ATTRIBUTION_WINDOWS)}"
            )

        logger.info(
            "Fetching Northbeam channel attribution: %s → %s, window=%dd",
            start_date, end_date, attribution_window,
        )

        payload = {
            "account_id": self.account_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "attribution_window": attribution_window,
            "granularity": "daily",
            "breakdown": "channel",
        }
        data = self._post("reports/attribution", payload)

        rows_raw: list[dict] = data.get("data", data.get("rows", []))
        if not rows_raw:
            logger.warning(
                "Northbeam returned no attribution data for %s → %s (window=%dd).",
                start_date, end_date, attribution_window,
            )
            return pd.DataFrame(
                columns=[
                    "date", "channel", "attributed_revenue",
                    "attributed_conversions", "attribution_window",
                ]
            )

        records = []
        for row in rows_raw:
            channel_raw: str = row.get("channel") or row.get("channel_name", "unknown")
            records.append({
                "date": pd.to_datetime(row.get("date")).date(),
                "channel": self._normalize_channel_key(channel_raw),
                "attributed_revenue": float(row.get("revenue", 0) or 0),
                "attributed_conversions": int(row.get("conversions", 0) or 0),
                "attribution_window": attribution_window,
            })

        df = pd.DataFrame(records)
        df = df.sort_values(["date", "channel"]).reset_index(drop=True)

        logger.info(
            "Northbeam attribution: %d rows, %d channels, "
            "total attributed_revenue=%.2f",
            len(df),
            df["channel"].nunique(),
            df["attributed_revenue"].sum(),
        )
        return df

    def get_new_returning_split(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Pull daily new vs. returning customer revenue split by channel.

        Args:
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            DataFrame with columns:
                - ``date`` (datetime.date)
                - ``channel`` (str) — canonical snake_case channel name
                - ``new_revenue`` (float)
                - ``returning_revenue`` (float)
                - ``new_pct`` (float) — fraction of revenue from new customers [0, 1]

        Raises:
            NorthbeamAPIError: On API errors.
        """
        logger.info(
            "Fetching Northbeam new/returning split: %s → %s",
            start_date, end_date,
        )

        payload = {
            "account_id": self.account_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "granularity": "daily",
            "breakdown": "channel",
            "segment": "new_vs_returning",
        }
        data = self._post("reports/customer_type", payload)

        rows_raw: list[dict] = data.get("data", data.get("rows", []))
        if not rows_raw:
            logger.warning(
                "Northbeam returned no new/returning data for %s → %s.",
                start_date, end_date,
            )
            return pd.DataFrame(
                columns=[
                    "date", "channel", "new_revenue",
                    "returning_revenue", "new_pct",
                ]
            )

        records = []
        for row in rows_raw:
            channel_raw: str = row.get("channel") or row.get("channel_name", "unknown")
            new_rev = float(row.get("new_revenue", 0) or 0)
            ret_rev = float(row.get("returning_revenue", 0) or 0)
            total_rev = new_rev + ret_rev
            new_pct = (new_rev / total_rev) if total_rev > 0 else 0.0

            records.append({
                "date": pd.to_datetime(row.get("date")).date(),
                "channel": self._normalize_channel_key(channel_raw),
                "new_revenue": new_rev,
                "returning_revenue": ret_rev,
                "new_pct": new_pct,
            })

        df = pd.DataFrame(records)
        df = df.sort_values(["date", "channel"]).reset_index(drop=True)

        logger.info(
            "Northbeam new/returning split: %d rows, %d channels. "
            "Avg new_pct=%.1f%%",
            len(df),
            df["channel"].nunique(),
            df["new_pct"].mean() * 100 if not df.empty else 0.0,
        )
        return df

    def compute_nb_mta_shares(
        self,
        start_date: date,
        end_date: date,
        attribution_window: int = 28,
    ) -> dict[str, float]:
        """Compute each channel's share of total MTA-attributed revenue.

        Sums attributed revenue over the date range (using the specified
        attribution window) and divides by the grand total to produce
        channel revenue shares.  Channels with zero total revenue are
        included with a share of 0.0.

        Args:
            start_date: Inclusive start date.
            end_date: Inclusive end date.
            attribution_window: Northbeam attribution lookback window (days).

        Returns:
            Dict mapping canonical channel name → revenue share [0.0, 1.0].
            Shares sum to 1.0 (within floating-point tolerance) when all
            channels are present.

        Raises:
            NorthbeamAPIError: On API errors.
        """
        df = self.get_channel_attribution(start_date, end_date, attribution_window)

        if df.empty:
            logger.warning(
                "No attribution data available; returning empty MTA shares."
            )
            return {}

        channel_totals = (
            df.groupby("channel")["attributed_revenue"]
            .sum()
            .to_dict()
        )
        grand_total = sum(channel_totals.values())

        if grand_total <= 0:
            logger.warning(
                "Grand total MTA revenue is zero; cannot compute revenue shares."
            )
            return {ch: 0.0 for ch in channel_totals}

        shares = {ch: rev / grand_total for ch, rev in channel_totals.items()}

        logger.info(
            "Northbeam MTA revenue shares (window=%dd): %s",
            attribution_window,
            {ch: f"{s:.3f}" for ch, s in sorted(shares.items())},
        )
        return shares

    def compute_prior_means(
        self,
        start_date: date,
        end_date: date,
        total_revenue: float,
        attribution_window: int = 28,
    ) -> dict[str, float]:
        """Compute Meridian ROI prior means from Northbeam MTA output.

        The procedure is:
        1. Retrieve channel-level attributed revenue from Northbeam.
        2. Retrieve channel-level spend from the attribution report.
        3. Compute raw MTA ROI = attributed_revenue / spend per channel.
        4. Apply the PRD-specified discount factor per channel to correct
           for MTA over-attribution bias.
        5. Return the discounted ROI as the prior mean for each channel.

        When spend is not directly available from the Northbeam report,
        the method falls back to computing implied ROI from revenue shares
        and ``total_revenue``:

            implied_revenue  = share × total_revenue
            raw_roi          = implied_revenue / attributed_revenue  (≈1.0 baseline)

        Then the discount factor is applied to obtain the prior mean.

        Discount factors (from PRD):

            meta_performance    ×0.65
            meta_awareness      ×0.60
            google_brand_search ×0.85
            google_non_brand    ×0.75
            tiktok              ×0.65

        Args:
            start_date: Inclusive start date for the MTA query.
            end_date: Inclusive end date for the MTA query.
            total_revenue: Observed total revenue over the period (used to
                scale MTA shares into absolute revenue estimates when
                spend data is not available in the Northbeam response).
            attribution_window: Northbeam attribution lookback window (days).

        Returns:
            Dict mapping canonical channel name → prior mean ROI (float).
            Channels not in ``_CHANNEL_DISCOUNT_FACTORS`` are returned with
            a discount factor of 1.0 (no adjustment) and a warning logged.

        Raises:
            NorthbeamAPIError: On API errors.
        """
        df = self.get_channel_attribution(start_date, end_date, attribution_window)

        if df.empty:
            logger.warning(
                "No Northbeam attribution data; returning empty prior means."
            )
            return {}

        # Aggregate to channel-level totals over the full date range
        agg = (
            df.groupby("channel")
            .agg(
                attributed_revenue=("attributed_revenue", "sum"),
                attributed_conversions=("attributed_conversions", "sum"),
            )
            .reset_index()
        )

        grand_total_mta = float(agg["attributed_revenue"].sum())
        if grand_total_mta <= 0:
            logger.warning("Total MTA attributed revenue is 0; cannot compute priors.")
            return {row["channel"]: 0.0 for _, row in agg.iterrows()}

        # Check whether the attribution report includes spend
        has_spend_col = "spend" in df.columns and df["spend"].sum() > 0

        prior_means: dict[str, float] = {}

        for _, row in agg.iterrows():
            channel: str = row["channel"]
            attr_rev: float = float(row["attributed_revenue"])

            if has_spend_col:
                channel_spend = float(
                    df[df["channel"] == channel]["spend"].sum()
                )
                if channel_spend > 0:
                    raw_roi = attr_rev / channel_spend
                else:
                    logger.warning(
                        "Channel %r has zero spend in Northbeam data; "
                        "skipping ROI computation.",
                        channel,
                    )
                    prior_means[channel] = 0.0
                    continue
            else:
                # Fallback: use revenue-share approach.
                # MTA share of total observed revenue → implied spend
                # is approximated by assuming a baseline ROI of 1.0
                # (i.e. implied_spend ≈ implied_revenue).  The discount
                # factor then directly scales this to a below-1 prior mean.
                mta_share = attr_rev / grand_total_mta
                implied_revenue = mta_share * total_revenue
                # With no spend data, set raw_roi = implied_revenue / attr_rev
                # so that prior_mean = discount × (implied / attr).
                raw_roi = implied_revenue / attr_rev if attr_rev > 0 else 0.0

            discount = _CHANNEL_DISCOUNT_FACTORS.get(channel)
            if discount is None:
                logger.warning(
                    "Channel %r not found in discount factor table; "
                    "applying no discount (factor=1.0). "
                    "Add it to _CHANNEL_DISCOUNT_FACTORS if needed.",
                    channel,
                )
                discount = 1.0

            prior_mean = raw_roi * discount
            prior_means[channel] = prior_mean

            logger.debug(
                "Channel %r: raw_roi=%.4f × discount=%.2f → prior_mean=%.4f",
                channel, raw_roi, discount, prior_mean,
            )

        logger.info(
            "Northbeam prior means computed for %d channels (window=%dd): %s",
            len(prior_means),
            attribution_window,
            {ch: f"{v:.4f}" for ch, v in sorted(prior_means.items())},
        )
        return prior_means
