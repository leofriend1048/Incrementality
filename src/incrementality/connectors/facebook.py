"""Facebook/Meta Ads data connector.

Pulls ad spend and delivery data from the Meta Marketing API,
broken down by DMA-level geo targeting.

Uses breakdowns=dma for native DMA-level reporting (requires Graph API v17+).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import pandas as pd
import requests

from incrementality.config import FacebookConfig

logger = logging.getLogger(__name__)

_GRAPH_API_BASE = "https://graph.facebook.com/v19.0"

# Meta DMA name → Nielsen DMA code mapping
# Meta returns DMA names like "New York, NY" when breakdowns=dma
_META_DMA_TO_NIELSEN: dict[str, str] = {
    "New York, NY": "501",
    "Binghamton, NY": "502",
    "Philadelphia, PA": "504",
    "Detroit, MI": "505",
    "Boston, MA-Manchester, NH": "506",
    "Boston, MA": "506",
    "Pittsburgh, PA": "508",
    "Ft. Wayne, IN": "509",
    "Cleveland-Akron (Canton), OH": "510",
    "Cleveland, OH": "510",
    "Washington, DC (Hagerstown, MD)": "511",
    "Washington, DC": "511",
    "Baltimore, MD": "512",
    "Buffalo, NY": "514",
    "Cincinnati, OH": "515",
    "Charlotte, NC": "517",
    "Greensboro-High Point-Winston Salem, NC": "518",
    "Greensboro, NC": "518",
    "Charleston, SC": "519",
    "Providence, RI-New Bedford, MA": "521",
    "Providence, RI": "521",
    "Atlanta, GA": "524",
    "Indianapolis, IN": "527",
    "Miami-Ft. Lauderdale, FL": "528",
    "Miami, FL": "528",
    "Louisville, KY": "529",
    "Hartford & New Haven, CT": "533",
    "Hartford, CT": "533",
    "Orlando-Daytona Beach-Melbourne, FL": "534",
    "Orlando, FL": "534",
    "Columbus, OH": "535",
    "Rochester, NY": "538",
    "Tampa-St. Petersburg (Sarasota), FL": "539",
    "Tampa, FL": "539",
    "Dayton, OH": "542",
    "Norfolk-Portsmouth-Newport News, VA": "544",
    "Norfolk, VA": "544",
    "Greenville-Spartanburg-Asheville-Anderson, SC": "545",
    "Greenville-Spartanburg, SC": "545",
    "Columbia, SC": "546",
    "West Palm Beach-Ft. Pierce, FL": "548",
    "West Palm Beach, FL": "548",
    "Richmond-Petersburg, VA": "556",
    "Richmond, VA": "556",
    "Knoxville, TN": "557",
    "Raleigh-Durham (Fayetteville), NC": "560",
    "Raleigh-Durham, NC": "560",
    "Jacksonville, FL": "561",
    "Grand Rapids-Kalamazoo-Battle Creek, MI": "563",
    "Grand Rapids, MI": "563",
    "Florence-Myrtle Beach, SC": "570",
    "Ft. Myers-Naples, FL": "571",
    "Chattanooga, TN": "575",
    "Chicago, IL": "602",
    "St. Louis, MO": "609",
    "Minneapolis-St. Paul, MN": "613",
    "Minneapolis, MN": "613",
    "Kansas City, MO": "616",
    "Milwaukee, WI": "617",
    "Houston, TX": "618",
    "New Orleans, LA": "622",
    "Dallas-Ft. Worth, TX": "623",
    "Dallas, TX": "623",
    "Birmingham (Anniston and Tuscaloosa), AL": "630",
    "Birmingham, AL": "630",
    "Austin, TX": "635",
    "Memphis, TN": "640",
    "San Antonio, TX": "641",
    "Oklahoma City, OK": "650",
    "Omaha, NE": "652",
    "Green Bay-Appleton, WI": "658",
    "Nashville, TN": "659",
    "Madison, WI": "669",
    "Tulsa, OK": "671",
    "Wichita-Hutchinson Plus, KS": "678",
    "Wichita, KS": "678",
    "Des Moines-Ames, IA": "679",
    "Des Moines, IA": "679",
    "Mobile-Pensacola (Ft. Walton Beach), FL": "686",
    "Mobile, AL": "686",
    "Little Rock-Pine Bluff, AR": "693",
    "Little Rock, AR": "693",
    "Baton Rouge, LA": "716",
    "Jackson, MS": "718",
    "Denver, CO": "751",
    "Colorado Springs-Pueblo, CO": "752",
    "Colorado Springs, CO": "752",
    "Phoenix (Prescott), AZ": "753",
    "Phoenix, AZ": "753",
    "Boise, ID": "757",
    "El Paso (Las Cruces), TX": "765",
    "El Paso, TX": "765",
    "Salt Lake City, UT": "770",
    "Tucson (Sierra Vista), AZ": "789",
    "Tucson, AZ": "789",
    "Albuquerque-Santa Fe, NM": "790",
    "Albuquerque, NM": "790",
    "Los Angeles, CA": "803",
    "San Francisco-Oakland-San Jose, CA": "807",
    "San Francisco, CA": "807",
    "Seattle-Tacoma, WA": "819",
    "Seattle, WA": "819",
    "Portland, OR": "820",
    "San Diego, CA": "825",
    "Sacramento-Stockton-Modesto, CA": "862",
    "Sacramento, CA": "862",
    "Las Vegas, NV": "839",
    "Honolulu, HI": "744",
    "Anchorage, AK": "743",
    "Fresno-Visalia, CA": "866",
    "Fresno, CA": "866",
    "Bakersfield, CA": "800",
    "Monterey-Salinas, CA": "828",
    "Palm Springs, CA": "804",
    "Santa Barbara-Santa Maria-San Luis Obispo, CA": "855",
    "Santa Barbara, CA": "855",
    "Spokane, WA": "881",
    "Reno, NV": "811",
    "Portland-Auburn, ME": "500",
    "Savannah, GA": "507",
    "Macon, GA": "503",
}


class FacebookConnector:
    """Connects to Meta Marketing API for ad spend by DMA."""

    DEFAULT_TIMEOUT = 30  # seconds per request

    def __init__(self, config: FacebookConfig):
        self.config = config
        self.session = requests.Session()
        self.session.params = {"access_token": config.access_token}  # type: ignore

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        url = f"{_GRAPH_API_BASE}/{path}"
        resp = self.session.get(url, params=params or {}, timeout=self.DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, data: dict[str, Any] | None = None) -> dict:
        url = f"{_GRAPH_API_BASE}/{path}"
        resp = self.session.post(url, json=data or {}, timeout=self.DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def fetch_campaigns(self) -> pd.DataFrame:
        """Fetch all campaigns in the ad account.

        Returns columns: campaign_id, name, status, objective, daily_budget
        """
        data = self._get(
            f"{self.config.ad_account_id}/campaigns",
            params={
                "fields": "id,name,status,objective,daily_budget",
                "limit": 500,
            },
        )
        records = []
        for c in data.get("data", []):
            records.append({
                "campaign_id": c["id"],
                "name": c.get("name", ""),
                "status": c.get("status", ""),
                "objective": c.get("objective", ""),
                "daily_budget": float(c.get("daily_budget", 0)) / 100,
            })
        return pd.DataFrame(records)

    def fetch_spend_by_dma(
        self,
        start_date: date,
        end_date: date,
        campaign_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch daily ad spend broken down by DMA.

        Uses the Marketing API insights endpoint with breakdowns=dma
        for native DMA-level reporting.

        Returns columns: date, dma_code, dma_name, spend, impressions, clicks
        """
        params: dict[str, Any] = {
            "fields": "spend,impressions,clicks",
            "breakdowns": "dma",
            "time_range": json.dumps({
                "since": str(start_date),
                "until": str(end_date),
            }),
            "time_increment": 1,  # Daily granularity
            "limit": 5000,
        }
        if campaign_ids:
            filtering = [
                {"field": "campaign.id", "operator": "IN", "value": campaign_ids}
            ]
            params["filtering"] = json.dumps(filtering)

        all_records = []
        unmapped_dmas: set[str] = set()
        total_rows = 0
        path = f"{self.config.ad_account_id}/insights"

        while path:
            data = self._get(path, params)
            for row in data.get("data", []):
                total_rows += 1
                dma_name = row.get("dma", "")
                dma_code = self._dma_name_to_code(dma_name)
                if dma_code:
                    all_records.append({
                        "date": pd.Timestamp(row["date_start"]).date(),
                        "dma_code": dma_code,
                        "dma_name": dma_name,
                        "spend": float(row.get("spend", 0)),
                        "impressions": int(row.get("impressions", 0)),
                        "clicks": int(row.get("clicks", 0)),
                    })
                else:
                    unmapped_dmas.add(dma_name)
            # Pagination
            paging = data.get("paging", {})
            next_url = paging.get("next")
            if next_url:
                path = next_url.replace(_GRAPH_API_BASE + "/", "")
                params = {}
            else:
                path = None  # type: ignore

        # Log mapping coverage
        if total_rows > 0:
            n_mapped = len(all_records)
            coverage_pct = n_mapped / total_rows
            if unmapped_dmas:
                logger.warning(
                    f"Facebook DMA mapping: {n_mapped}/{total_rows} rows mapped "
                    f"({coverage_pct:.0%}). {len(unmapped_dmas)} DMA names unmapped: "
                    f"{sorted(unmapped_dmas)[:5]}"
                )
            else:
                logger.info(
                    f"Facebook: {n_mapped} rows, all DMAs mapped successfully"
                )

        return pd.DataFrame(all_records)

    def get_campaign_spend_by_dma(
        self,
        start_date: date,
        end_date: date,
        campaign_ids: list[str],
    ) -> pd.DataFrame:
        """Convenience: fetch spend for specific campaigns by DMA."""
        return self.fetch_spend_by_dma(start_date, end_date, campaign_ids)

    @staticmethod
    def _dma_name_to_code(dma_name: str) -> str | None:
        """Map a Meta DMA name to a Nielsen DMA code.

        Meta returns DMA names when breakdowns=dma. We map these
        to the standard Nielsen DMA codes used throughout the platform.
        """
        if not dma_name:
            return None

        # Direct lookup
        code = _META_DMA_TO_NIELSEN.get(dma_name)
        if code:
            return code

        # Try fuzzy match: strip extra qualifiers
        # Meta sometimes includes extra city names
        base_name = dma_name.split(",")[0].strip() if "," in dma_name else dma_name
        for meta_name, nielsen_code in _META_DMA_TO_NIELSEN.items():
            if base_name in meta_name or meta_name.startswith(base_name):
                return nielsen_code

        logger.debug(f"No DMA code mapping for: {dma_name}")
        return None

    # ------------------------------------------------------------------
    # Targeting deployment (execute / revert)
    # ------------------------------------------------------------------

    def get_campaign_targeting(self, campaign_id: str) -> dict:
        """Read current geo-targeting for a campaign.

        Returns the full targeting spec including geo_locations and
        excluded_geo_locations so it can be restored after the test.
        """
        data = self._get(campaign_id, params={"fields": "targeting,name,status"})
        return data.get("targeting", {})

    def get_adset_targeting(self, adset_id: str) -> dict:
        """Read current geo-targeting for an ad set."""
        data = self._get(adset_id, params={"fields": "targeting,name,status"})
        return data.get("targeting", {})

    def get_active_adsets(self, campaign_id: str) -> list[dict]:
        """Get all active ad sets for a campaign.

        Facebook targeting lives at the ad set level, not campaign level.
        We need to modify each ad set to apply DMA exclusions.
        """
        data = self._get(
            f"{campaign_id}/adsets",
            params={
                "fields": "id,name,status,targeting",
                "limit": 500,
                "filtering": json.dumps([
                    {"field": "effective_status", "operator": "IN",
                     "value": ["ACTIVE", "PAUSED"]}
                ]),
            },
        )
        return data.get("data", [])

    def get_all_campaign_ids(self) -> list[str]:
        """Get all active campaign IDs in the ad account."""
        data = self._get(
            f"{self.config.ad_account_id}/campaigns",
            params={
                "fields": "id,name,status",
                "limit": 500,
                "filtering": json.dumps([
                    {"field": "effective_status", "operator": "IN",
                     "value": ["ACTIVE", "PAUSED"]}
                ]),
            },
        )
        return [c["id"] for c in data.get("data", [])]

    def deploy_holdout(
        self,
        holdout_dma_codes: list[str],
        campaign_ids: list[str] | None = None,
    ) -> dict[str, dict]:
        """Deploy DMA holdout exclusions to Facebook ad sets.

        Saves original targeting for each ad set before modifying, so it
        can be reverted later. For channel-level tests, applies to ALL
        active campaigns. For campaign-level, applies only to specified campaigns.

        Args:
            holdout_dma_codes: Nielsen DMA codes to exclude from ad delivery.
            campaign_ids: Specific campaign IDs (None = all active campaigns).

        Returns:
            Dict mapping ad set IDs to their original targeting.
        """
        if campaign_ids is None:
            campaign_ids = self.get_all_campaign_ids()
            logger.info(f"Channel-level test: found {len(campaign_ids)} active campaigns")

        original_targeting: dict[str, dict] = {}
        exclusion_spec = get_exclusion_targeting_spec(holdout_dma_codes)
        n_updated = 0

        for campaign_id in campaign_ids:
            adsets = self.get_active_adsets(campaign_id)
            logger.info(
                f"Campaign {campaign_id}: {len(adsets)} ad sets to update"
            )

            for adset in adsets:
                adset_id = adset["id"]
                current_targeting = adset.get("targeting", {})

                # Save original targeting state
                original_targeting[adset_id] = {
                    "targeting": current_targeting.copy(),
                    "campaign_id": campaign_id,
                    "adset_name": adset.get("name", ""),
                }

                # Merge holdout exclusions into existing targeting
                updated_targeting = current_targeting.copy()
                existing_excluded = updated_targeting.get("excluded_geo_locations", {})
                existing_markets = existing_excluded.get("geo_markets", [])

                # Add holdout DMA exclusions (avoid duplicates)
                existing_keys = {m.get("key") for m in existing_markets}
                new_markets = [
                    m for m in exclusion_spec["excluded_geo_locations"]["geo_markets"]
                    if m["key"] not in existing_keys
                ]
                all_markets = existing_markets + new_markets

                updated_targeting["excluded_geo_locations"] = {
                    "geo_markets": all_markets,
                }

                # Apply the update
                self._post(adset_id, data={
                    "targeting": json.dumps(updated_targeting),
                })
                n_updated += 1
                logger.info(
                    f"  Updated ad set {adset_id} ({adset.get('name', '')}): "
                    f"excluded {len(new_markets)} holdout DMAs"
                )

        logger.info(
            f"Deployed holdout to {n_updated} ad sets across "
            f"{len(campaign_ids)} campaigns. "
            f"Excluding {len(holdout_dma_codes)} DMAs."
        )
        return original_targeting

    def deploy_holdout_update(
        self,
        holdout_dma_codes: list[str],
        already_excluded_adset_ids: set[str],
    ) -> tuple[dict[str, dict], int]:
        """Scan all active ad sets and apply holdout exclusions to any new ones.

        Only processes ad sets NOT already in already_excluded_adset_ids.
        This catches new campaigns/ad sets created after the initial deploy.

        Args:
            holdout_dma_codes: Nielsen DMA codes to exclude.
            already_excluded_adset_ids: Ad set IDs already tracked from deploy.

        Returns:
            Tuple of (new original_targeting entries, count of newly updated ad sets).
        """
        all_campaign_ids = self.get_all_campaign_ids()
        exclusion_spec = get_exclusion_targeting_spec(holdout_dma_codes)
        new_targeting: dict[str, dict] = {}
        n_new = 0

        for campaign_id in all_campaign_ids:
            adsets = self.get_active_adsets(campaign_id)
            for adset in adsets:
                adset_id = adset["id"]
                if adset_id in already_excluded_adset_ids:
                    continue  # Already managed

                current_targeting = adset.get("targeting", {})

                # Check if holdout DMAs are already excluded
                existing_excluded = current_targeting.get("excluded_geo_locations", {})
                existing_keys = {
                    m.get("key") for m in existing_excluded.get("geo_markets", [])
                }
                holdout_set = set(holdout_dma_codes)
                if holdout_set.issubset(existing_keys):
                    continue  # Already has all exclusions

                # Save original and apply exclusions
                new_targeting[adset_id] = {
                    "targeting": current_targeting.copy(),
                    "campaign_id": campaign_id,
                    "adset_name": adset.get("name", ""),
                }

                updated_targeting = current_targeting.copy()
                existing_markets = existing_excluded.get("geo_markets", [])
                new_markets = [
                    m for m in exclusion_spec["excluded_geo_locations"]["geo_markets"]
                    if m["key"] not in existing_keys
                ]
                updated_targeting["excluded_geo_locations"] = {
                    "geo_markets": existing_markets + new_markets,
                }

                self._post(adset_id, data={
                    "targeting": json.dumps(updated_targeting),
                })
                n_new += 1
                logger.info(
                    f"  NEW ad set {adset_id} ({adset.get('name', '')}): "
                    f"excluded {len(new_markets)} holdout DMAs"
                )

        logger.info(
            f"Holdout update: scanned {len(all_campaign_ids)} campaigns, "
            f"found {n_new} new ad sets to exclude."
        )
        return new_targeting, n_new

    def revert_holdout(self, original_targeting: dict[str, dict]) -> int:
        """Revert ad sets back to their pre-test targeting.

        Args:
            original_targeting: Dict from deploy_holdout() mapping ad set IDs
                                to their original targeting state.

        Returns:
            Number of ad sets successfully reverted.
        """
        n_reverted = 0
        n_failed = 0

        for adset_id, state in original_targeting.items():
            try:
                self._post(adset_id, data={
                    "targeting": json.dumps(state["targeting"]),
                })
                n_reverted += 1
                logger.info(
                    f"Reverted ad set {adset_id} ({state.get('adset_name', '')})"
                )
            except Exception as e:
                n_failed += 1
                logger.error(
                    f"FAILED to revert ad set {adset_id}: {e}. "
                    f"Manual revert required!"
                )

        logger.info(
            f"Reverted {n_reverted}/{len(original_targeting)} ad sets. "
            f"{n_failed} failures."
        )
        if n_failed > 0:
            logger.error(
                f"WARNING: {n_failed} ad sets could not be reverted automatically. "
                f"Check Facebook Ads Manager and restore geo targeting manually."
            )
        return n_reverted


def get_dma_geo_targeting_spec(dma_codes: list[str]) -> dict:
    """Build a Meta geo_locations targeting spec for specific DMAs.

    Use this when setting up holdout tests to exclude DMAs from ad delivery.
    """
    return {
        "geo_locations": {
            "geo_markets": [
                {"key": code, "market_type": "dma"}
                for code in dma_codes
            ],
        },
    }


def get_exclusion_targeting_spec(holdout_dma_codes: list[str]) -> dict:
    """Build an exclusion targeting spec to hold out specific DMAs.

    Apply this to campaigns during the test period to create the holdout.
    """
    return {
        "excluded_geo_locations": {
            "geo_markets": [
                {"key": code, "market_type": "dma"}
                for code in holdout_dma_codes
            ],
        },
    }
