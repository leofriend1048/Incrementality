"""YouTube / Google Ads data connector.

Pulls ad spend and delivery data from the Google Ads API,
broken down by DMA-level geo targeting.

YouTube campaigns in Google Ads use advertising_channel_type = VIDEO.
Google Ads natively supports DMA (Metro) level geo targeting and reporting.
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

        config_dict = {
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "refresh_token": self.config.refresh_token,
            "login_customer_id": self.config.customer_id,
            "use_proto_plus": True,
        }

        # developer_token is required for all Google Ads API calls
        if self.config.developer_token:
            config_dict["developer_token"] = self.config.developer_token
        else:
            raise ValueError(
                "Google Ads developer_token is required. "
                "Get one at https://ads.google.com/nav/selectaccount?authuser=0 "
                "under Tools & Settings > API Center."
            )

        self._client = GoogleAdsClient.load_from_dict(config_dict)
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

        Uses the user_location_view resource which provides DMA-level
        geo data via geo_target_constant of type METRO.

        Returns columns: date, dma_code, dma_name, spend, impressions,
                         views, view_rate, clicks, cost_per_view
        """
        client = self._get_client()
        service = client.get_service("GoogleAdsService")

        campaign_filter = ""
        if campaign_ids:
            ids_str = ", ".join(campaign_ids)
            campaign_filter = f"AND campaign.id IN ({ids_str})"

        # Use user_location_view for location-based reporting.
        # Filter to metro-level (DMA) geo targets only.
        query = f"""
            SELECT
                segments.date,
                campaign.id,
                user_location_view.country_criterion_id,
                user_location_view.targeting_location,
                metrics.cost_micros,
                metrics.impressions,
                metrics.video_views,
                metrics.video_view_rate,
                metrics.clicks
            FROM user_location_view
            WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
            AND campaign.advertising_channel_type = 'VIDEO'
            {campaign_filter}
        """

        records = []
        response = service.search(customer_id=self.config.customer_id, query=query)
        for row in response:
            # targeting_location is a geo_target_constant resource name
            geo_resource = row.user_location_view.targeting_location
            # Extract the geo ID from the resource name
            # Format: "geoTargetConstants/XXXXXX"
            geo_id = geo_resource.split("/")[-1] if geo_resource else ""
            dma_code = _google_geo_id_to_dma(geo_id)
            if dma_code:
                cost = row.metrics.cost_micros / 1_000_000
                views = row.metrics.video_views
                records.append({
                    "date": pd.Timestamp(row.segments.date).date(),
                    "dma_code": dma_code,
                    "dma_name": "",  # Name resolved from mapping
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

    # ------------------------------------------------------------------
    # Targeting deployment (execute / revert)
    # ------------------------------------------------------------------

    def get_all_campaign_ids(self) -> list[str]:
        """Get all active VIDEO campaign IDs."""
        client = self._get_client()
        service = client.get_service("GoogleAdsService")
        query = """
            SELECT campaign.id
            FROM campaign
            WHERE campaign.advertising_channel_type = 'VIDEO'
            AND campaign.status IN ('ENABLED', 'PAUSED')
        """
        ids = []
        response = service.search(customer_id=self.config.customer_id, query=query)
        for row in response:
            ids.append(str(row.campaign.id))
        return ids

    def get_campaign_geo_criteria(self, campaign_id: str) -> list[dict]:
        """Read existing geo-targeting criteria for a campaign."""
        client = self._get_client()
        service = client.get_service("GoogleAdsService")
        query = f"""
            SELECT
                campaign_criterion.resource_name,
                campaign_criterion.location.geo_target_constant,
                campaign_criterion.negative
            FROM campaign_criterion
            WHERE campaign_criterion.type = 'LOCATION'
            AND campaign.id = {campaign_id}
        """
        criteria = []
        response = service.search(customer_id=self.config.customer_id, query=query)
        for row in response:
            criteria.append({
                "resource_name": row.campaign_criterion.resource_name,
                "geo_target_constant": row.campaign_criterion.location.geo_target_constant,
                "negative": row.campaign_criterion.negative,
            })
        return criteria

    def deploy_holdout(
        self,
        holdout_dma_codes: list[str],
        campaign_ids: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """Deploy DMA holdout exclusions to Google Ads campaigns.

        Adds negative location criteria for holdout DMAs. Saves the
        resource names of created criteria so they can be removed on revert.

        Returns:
            Dict mapping campaign IDs to created criterion resource names.
        """
        client = self._get_client()
        campaign_service = client.get_service("CampaignService")
        criterion_service = client.get_service("CampaignCriterionService")

        if campaign_ids is None:
            campaign_ids = self.get_all_campaign_ids()
            logger.info(
                f"Channel-level test: found {len(campaign_ids)} VIDEO campaigns"
            )

        created_criteria: dict[str, list[str]] = {}

        for campaign_id in campaign_ids:
            existing = self.get_campaign_geo_criteria(campaign_id)
            existing_constants = {
                c["geo_target_constant"] for c in existing if c["negative"]
            }

            operations = []
            for dma_code in holdout_dma_codes:
                google_geo_id = _dma_to_google_geo_id(dma_code)
                if not google_geo_id:
                    logger.warning(f"No Google geo ID for DMA {dma_code}")
                    continue

                geo_constant = client.get_service(
                    "GeoTargetConstantService"
                ).geo_target_constant_path(google_geo_id)

                if geo_constant in existing_constants:
                    continue

                operation = client.get_type("CampaignCriterionOperation")
                criterion = operation.create
                criterion.campaign = campaign_service.campaign_path(
                    self.config.customer_id, campaign_id,
                )
                criterion.location.geo_target_constant = geo_constant
                criterion.negative = True
                operations.append(operation)

            if operations:
                response = criterion_service.mutate_campaign_criteria(
                    customer_id=self.config.customer_id,
                    operations=operations,
                )
                created_criteria[campaign_id] = [
                    r.resource_name for r in response.results
                ]
                logger.info(
                    f"Campaign {campaign_id}: excluded "
                    f"{len(created_criteria[campaign_id])} holdout DMAs"
                )
            else:
                created_criteria[campaign_id] = []

        total = sum(len(v) for v in created_criteria.values())
        logger.info(f"Deployed {total} DMA exclusions across {len(campaign_ids)} campaigns")
        return created_criteria

    def deploy_holdout_update(
        self,
        holdout_dma_codes: list[str],
        already_excluded_campaign_ids: set[str],
    ) -> tuple[dict[str, list[str]], int]:
        """Scan all VIDEO campaigns and apply holdout exclusions to any new ones.

        Only processes campaigns NOT already in already_excluded_campaign_ids.

        Args:
            holdout_dma_codes: Nielsen DMA codes to exclude.
            already_excluded_campaign_ids: Campaign IDs already tracked.

        Returns:
            Tuple of (new criteria mapping, count of newly excluded criteria).
        """
        client = self._get_client()
        campaign_service = client.get_service("CampaignService")
        criterion_service = client.get_service("CampaignCriterionService")

        all_campaign_ids = self.get_all_campaign_ids()
        new_criteria: dict[str, list[str]] = {}
        n_new = 0

        for campaign_id in all_campaign_ids:
            if campaign_id in already_excluded_campaign_ids:
                continue  # Already managed

            existing = self.get_campaign_geo_criteria(campaign_id)
            existing_constants = {
                c["geo_target_constant"] for c in existing if c["negative"]
            }

            operations = []
            for dma_code in holdout_dma_codes:
                google_geo_id = _dma_to_google_geo_id(dma_code)
                if not google_geo_id:
                    continue

                geo_constant = client.get_service(
                    "GeoTargetConstantService"
                ).geo_target_constant_path(google_geo_id)

                if geo_constant in existing_constants:
                    continue

                operation = client.get_type("CampaignCriterionOperation")
                criterion = operation.create
                criterion.campaign = campaign_service.campaign_path(
                    self.config.customer_id, campaign_id,
                )
                criterion.location.geo_target_constant = geo_constant
                criterion.negative = True
                operations.append(operation)

            if operations:
                response = criterion_service.mutate_campaign_criteria(
                    customer_id=self.config.customer_id,
                    operations=operations,
                )
                new_criteria[campaign_id] = [
                    r.resource_name for r in response.results
                ]
                n_new += len(new_criteria[campaign_id])
                logger.info(
                    f"NEW campaign {campaign_id}: excluded "
                    f"{len(new_criteria[campaign_id])} holdout DMAs"
                )
            else:
                new_criteria[campaign_id] = []

        logger.info(
            f"Holdout update: scanned {len(all_campaign_ids)} campaigns, "
            f"found {n_new} new exclusions to apply."
        )
        return new_criteria, n_new

    def revert_holdout(self, created_criteria: dict[str, list[str]]) -> int:
        """Remove holdout DMA exclusions added by deploy_holdout().

        Returns number of criteria successfully removed.
        """
        client = self._get_client()
        criterion_service = client.get_service("CampaignCriterionService")
        n_reverted = 0

        for campaign_id, resource_names in created_criteria.items():
            if not resource_names:
                continue
            operations = []
            for rn in resource_names:
                op = client.get_type("CampaignCriterionOperation")
                op.remove = rn
                operations.append(op)
            try:
                criterion_service.mutate_campaign_criteria(
                    customer_id=self.config.customer_id,
                    operations=operations,
                )
                n_reverted += len(resource_names)
                logger.info(f"Campaign {campaign_id}: removed {len(resource_names)} exclusions")
            except Exception as e:
                logger.error(f"FAILED to revert campaign {campaign_id}: {e}")

        logger.info(f"Reverted {n_reverted} criteria total")
        return n_reverted


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
# Google Ads geo criterion ID <-> Nielsen DMA code mapping
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
