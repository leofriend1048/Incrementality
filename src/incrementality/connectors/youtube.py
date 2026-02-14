"""YouTube / Google Ads data connector.

Pulls ad spend and delivery data from the Google Ads API,
broken down by DMA-level geo targeting.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

from incrementality.config import YouTubeConfig

logger = logging.getLogger(__name__)


class YouTubeConnector:
    """Connects to Google Ads API for YouTube ad spend by DMA.

    Google Ads natively supports DMA (Metro) level geo targeting and reporting,
    making it ideal for geo holdout tests.
    """

    def __init__(self, config: YouTubeConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        """Initialize the Google Ads client."""
        if self._client is not None:
            return self._client

        from google.ads.googleads.client import GoogleAdsClient

        self._client = GoogleAdsClient.load_from_dict({
            "developer_token": "",  # Set via env or config
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "refresh_token": self.config.refresh_token,
            "login_customer_id": self.config.customer_id,
            "use_proto_plus": True,
        })
        return self._client

    def fetch_campaigns(self) -> pd.DataFrame:
        """Fetch YouTube/video campaigns from Google Ads.

        Returns columns: campaign_id, name, status, type, budget
        """
        client = self._get_client()
        service = client.get_service("GoogleAdsService")

        query = """
            SELECT
                campaign.id,
                campaign.name,
                campaign.status,
                campaign.advertising_channel_type,
                campaign_budget.amount_micros
            FROM campaign
            WHERE campaign.advertising_channel_type = 'VIDEO'
            AND campaign.status != 'REMOVED'
        """

        records = []
        response = service.search(customer_id=self.config.customer_id, query=query)
        for row in response:
            records.append({
                "campaign_id": str(row.campaign.id),
                "name": row.campaign.name,
                "status": row.campaign.status.name,
                "type": row.campaign.advertising_channel_type.name,
                "budget": row.campaign_budget.amount_micros / 1_000_000,
            })
        return pd.DataFrame(records)

    def fetch_spend_by_dma(
        self,
        start_date: date,
        end_date: date,
        campaign_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch daily ad spend by DMA (metro area) from Google Ads.

        Google Ads uses 'metro' criterion for DMA-level geo data.

        Returns columns: date, dma_code, dma_name, spend, impressions,
                         views, view_rate, clicks, cost_per_view
        """
        client = self._get_client()
        service = client.get_service("GoogleAdsService")

        campaign_filter = ""
        if campaign_ids:
            ids_str = ", ".join(campaign_ids)
            campaign_filter = f"AND campaign.id IN ({ids_str})"

        query = f"""
            SELECT
                segments.date,
                geographic_view.country_criterion_id,
                geographic_view.location_type,
                geo_target_constant.canonical_name,
                geo_target_constant.id,
                metrics.cost_micros,
                metrics.impressions,
                metrics.video_views,
                metrics.video_view_rate,
                metrics.clicks
            FROM geographic_view
            WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
            AND campaign.advertising_channel_type = 'VIDEO'
            AND geographic_view.location_type = 'LOCATION_OF_PRESENCE'
            {campaign_filter}
        """

        records = []
        response = service.search(customer_id=self.config.customer_id, query=query)
        for row in response:
            geo_id = str(row.geo_target_constant.id)
            dma_code = _google_geo_id_to_dma(geo_id)
            if dma_code:
                cost = row.metrics.cost_micros / 1_000_000
                views = row.metrics.video_views
                records.append({
                    "date": pd.Timestamp(row.segments.date).date(),
                    "dma_code": dma_code,
                    "dma_name": row.geo_target_constant.canonical_name,
                    "spend": cost,
                    "impressions": row.metrics.impressions,
                    "views": views,
                    "view_rate": row.metrics.video_view_rate,
                    "clicks": row.metrics.clicks,
                    "cost_per_view": cost / views if views > 0 else 0,
                })

        return pd.DataFrame(records)

    def get_campaign_spend_by_dma(
        self,
        start_date: date,
        end_date: date,
        campaign_ids: list[str],
    ) -> pd.DataFrame:
        """Convenience: fetch spend for specific campaigns by DMA."""
        return self.fetch_spend_by_dma(start_date, end_date, campaign_ids)


def get_dma_location_targeting(dma_codes: list[str]) -> list[dict]:
    """Build Google Ads location targeting for specific DMAs.

    Returns location criteria to use for campaign geo targeting.
    """
    return [
        {
            "geo_target_constant": f"geoTargetConstants/{_dma_to_google_geo_id(code)}",
            "negative": False,
        }
        for code in dma_codes
        if _dma_to_google_geo_id(code)
    ]


def get_dma_exclusion_targeting(holdout_dma_codes: list[str]) -> list[dict]:
    """Build Google Ads exclusion targeting for holdout DMAs."""
    return [
        {
            "geo_target_constant": f"geoTargetConstants/{_dma_to_google_geo_id(code)}",
            "negative": True,
        }
        for code in holdout_dma_codes
        if _dma_to_google_geo_id(code)
    ]


# ---------------------------------------------------------------------------
# Google Ads geo criterion ID ↔ Nielsen DMA code mapping
# Google Ads uses its own geo IDs for metro areas.
# Top 50 DMAs mapped here; extend as needed.
# ---------------------------------------------------------------------------
_GOOGLE_TO_DMA: dict[str, str] = {
    "200501": "501",  # New York
    "200803": "803",  # Los Angeles
    "200602": "602",  # Chicago
    "200504": "504",  # Philadelphia
    "200807": "807",  # San Francisco
    "200511": "511",  # Washington DC
    "200623": "623",  # Dallas
    "200618": "618",  # Houston
    "200506": "506",  # Boston
    "200524": "524",  # Atlanta
    "200539": "539",  # Tampa
    "200753": "753",  # Phoenix
    "200819": "819",  # Seattle
    "200505": "505",  # Detroit
    "200613": "613",  # Minneapolis
    "200528": "528",  # Miami
    "200751": "751",  # Denver
    "200510": "510",  # Cleveland
    "200508": "508",  # Pittsburgh
    "200534": "534",  # Orlando
    "200527": "527",  # Indianapolis
    "200825": "825",  # San Diego
    "200820": "820",  # Portland
    "200616": "616",  # Kansas City
    "200659": "659",  # Nashville
    "200512": "512",  # Baltimore
    "200560": "560",  # Raleigh
    "200517": "517",  # Charlotte
    "200641": "641",  # San Antonio
    "200515": "515",  # Cincinnati
    "200535": "535",  # Columbus OH
    "200609": "609",  # St. Louis
    "200635": "635",  # Austin
    "200622": "622",  # New Orleans
    "200650": "650",  # Oklahoma City
    "200671": "671",  # Tulsa
    "200630": "630",  # Birmingham
    "200561": "561",  # Jacksonville
    "200529": "529",  # Louisville
    "200770": "770",  # Salt Lake City
    "200839": "839",  # Las Vegas
    "200828": "828",  # Sacramento
    "200652": "652",  # Omaha
    "200757": "757",  # Boise
    "200790": "790",  # Albuquerque
    "200789": "789",  # Tucson
    "200765": "765",  # El Paso
    "200693": "693",  # Little Rock
    "200716": "716",  # Baton Rouge
}

_DMA_TO_GOOGLE = {v: k for k, v in _GOOGLE_TO_DMA.items()}


def _google_geo_id_to_dma(geo_id: str) -> str | None:
    return _GOOGLE_TO_DMA.get(geo_id)


def _dma_to_google_geo_id(dma_code: str) -> str | None:
    return _DMA_TO_GOOGLE.get(dma_code)
