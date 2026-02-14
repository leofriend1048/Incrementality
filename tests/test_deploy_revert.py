"""Tests for deploy/revert holdout targeting on Facebook and Google Ads.

Uses mocked API responses to verify:
- Original targeting is saved before modifications
- Holdout DMA exclusions are applied correctly
- Revert restores original targeting state
- Channel-level vs campaign-level scoping works
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from incrementality.config import FacebookConfig
from incrementality.connectors.facebook import FacebookConnector, get_exclusion_targeting_spec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fb_connector() -> FacebookConnector:
    config = FacebookConfig(
        app_id="test_app_id",
        app_secret="test_app_secret",
        access_token="test_access_token",
        ad_account_id="act_123456",
    )
    return FacebookConnector(config)


# ---------------------------------------------------------------------------
# Facebook Deploy/Revert Tests
# ---------------------------------------------------------------------------

class TestFacebookDeployHoldout:
    def test_deploy_saves_original_targeting(self):
        """Verify deploy saves original targeting before modifying."""
        c = _make_fb_connector()

        adsets_response = {
            "data": [
                {
                    "id": "adset_001",
                    "name": "Test Ad Set 1",
                    "status": "ACTIVE",
                    "targeting": {
                        "geo_locations": {
                            "countries": ["US"],
                        },
                    },
                },
                {
                    "id": "adset_002",
                    "name": "Test Ad Set 2",
                    "status": "ACTIVE",
                    "targeting": {
                        "geo_locations": {
                            "countries": ["US"],
                        },
                        "excluded_geo_locations": {
                            "geo_markets": [
                                {"key": "999", "market_type": "dma"},
                            ],
                        },
                    },
                },
            ],
        }

        with patch.object(c, "_get") as mock_get, \
             patch.object(c, "_post") as mock_post:
            mock_get.return_value = adsets_response
            mock_post.return_value = {"success": True}

            original = c.deploy_holdout(
                holdout_dma_codes=["501", "803"],
                campaign_ids=["camp_001"],
            )

        # Should have saved original targeting for both ad sets
        assert "adset_001" in original
        assert "adset_002" in original

        # Original targeting should contain the pre-deploy state
        assert original["adset_001"]["targeting"] == adsets_response["data"][0]["targeting"]
        assert original["adset_002"]["targeting"] == adsets_response["data"][1]["targeting"]

        # Should have called _post twice (once per ad set)
        assert mock_post.call_count == 2

    def test_deploy_avoids_duplicate_exclusions(self):
        """If a DMA is already excluded, don't add it again."""
        c = _make_fb_connector()

        adsets_response = {
            "data": [
                {
                    "id": "adset_001",
                    "name": "Test",
                    "status": "ACTIVE",
                    "targeting": {
                        "excluded_geo_locations": {
                            "geo_markets": [
                                {"key": "501", "market_type": "dma"},
                            ],
                        },
                    },
                },
            ],
        }

        with patch.object(c, "_get") as mock_get, \
             patch.object(c, "_post") as mock_post:
            mock_get.return_value = adsets_response
            mock_post.return_value = {"success": True}

            c.deploy_holdout(
                holdout_dma_codes=["501", "803"],
                campaign_ids=["camp_001"],
            )

        # Check the targeting that was posted
        call_args = mock_post.call_args
        posted_targeting = json.loads(call_args[1]["data"]["targeting"])
        markets = posted_targeting["excluded_geo_locations"]["geo_markets"]

        # Should have 2 total (original 501 + new 803), not 3
        keys = [m["key"] for m in markets]
        assert len(keys) == 2
        assert "501" in keys
        assert "803" in keys

    def test_deploy_channel_level_gets_all_campaigns(self):
        """Channel-level test should fetch all campaigns automatically."""
        c = _make_fb_connector()

        campaigns_response = {
            "data": [
                {"id": "camp_001"},
                {"id": "camp_002"},
                {"id": "camp_003"},
            ],
        }
        adsets_response = {"data": []}

        with patch.object(c, "_get") as mock_get, \
             patch.object(c, "_post"):
            # First call gets campaigns, subsequent calls get ad sets
            mock_get.side_effect = [campaigns_response,
                                    adsets_response, adsets_response, adsets_response]

            c.deploy_holdout(
                holdout_dma_codes=["501"],
                campaign_ids=None,  # Channel-level = all campaigns
            )

        # Should have made 4 _get calls: 1 for campaigns + 3 for ad sets
        assert mock_get.call_count == 4


class TestFacebookRevertHoldout:
    def test_revert_restores_original_targeting(self):
        """Verify revert sends back the exact original targeting."""
        c = _make_fb_connector()

        original_targeting = {
            "adset_001": {
                "targeting": {"geo_locations": {"countries": ["US"]}},
                "campaign_id": "camp_001",
                "adset_name": "Test Ad Set 1",
            },
            "adset_002": {
                "targeting": {
                    "geo_locations": {"countries": ["US"]},
                    "excluded_geo_locations": {
                        "geo_markets": [{"key": "999", "market_type": "dma"}],
                    },
                },
                "campaign_id": "camp_001",
                "adset_name": "Test Ad Set 2",
            },
        }

        with patch.object(c, "_post") as mock_post:
            mock_post.return_value = {"success": True}
            n_reverted = c.revert_holdout(original_targeting)

        assert n_reverted == 2
        assert mock_post.call_count == 2

        # Verify exact targeting was posted back
        first_call = mock_post.call_args_list[0]
        assert first_call[0][0] == "adset_001"
        posted = json.loads(first_call[1]["data"]["targeting"])
        assert posted == {"geo_locations": {"countries": ["US"]}}

    def test_revert_handles_partial_failure(self):
        """If one ad set fails to revert, continue with others."""
        c = _make_fb_connector()

        original_targeting = {
            "adset_001": {
                "targeting": {"geo_locations": {"countries": ["US"]}},
                "campaign_id": "camp_001",
                "adset_name": "Success",
            },
            "adset_002": {
                "targeting": {"geo_locations": {"countries": ["US"]}},
                "campaign_id": "camp_001",
                "adset_name": "Failure",
            },
        }

        with patch.object(c, "_post") as mock_post:
            # First succeeds, second fails
            mock_post.side_effect = [
                {"success": True},
                Exception("API error"),
            ]
            n_reverted = c.revert_holdout(original_targeting)

        assert n_reverted == 1
        assert mock_post.call_count == 2


# ---------------------------------------------------------------------------
# Exclusion Spec Tests
# ---------------------------------------------------------------------------

class TestExclusionSpec:
    def test_get_exclusion_targeting_spec(self):
        spec = get_exclusion_targeting_spec(["501", "803", "602"])
        markets = spec["excluded_geo_locations"]["geo_markets"]
        assert len(markets) == 3
        assert all(m["market_type"] == "dma" for m in markets)
        keys = {m["key"] for m in markets}
        assert keys == {"501", "803", "602"}

    def test_empty_holdout(self):
        spec = get_exclusion_targeting_spec([])
        assert spec["excluded_geo_locations"]["geo_markets"] == []
