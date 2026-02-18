"""Tests for the Shopify connector.

Unit tests use mocked HTTP responses to validate the full code path:
  API request construction → pagination → order parsing → DMA aggregation.

Integration test (test_shopify_live) hits the real API. Run with:
  SHOPIFY_DOMAIN=my-store.myshopify.com SHOPIFY_TOKEN=shpat_xxx pytest tests/test_shopify_connector.py -k live
"""

from __future__ import annotations

import os
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from incrementality.config import ShopifyConfig
from incrementality.connectors.shopify import ShopifyConnector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ORDERS = {
    "orders": [
        {
            "id": 1001,
            "created_at": "2025-06-15T10:30:00-04:00",
            "total_price": "79.99",
            "subtotal_price": "69.99",
            "financial_status": "paid",
            "line_items": [{"id": 1}],
            "shipping_address": {
                "zip": "10001",
                "province_code": "NY",
                "city": "New York",
            },
        },
        {
            "id": 1002,
            "created_at": "2025-06-15T14:00:00-04:00",
            "total_price": "149.50",
            "subtotal_price": "139.50",
            "financial_status": "paid",
            "line_items": [{"id": 2}, {"id": 3}],
            "shipping_address": {
                "zip": "90210",
                "province_code": "CA",
                "city": "Beverly Hills",
            },
        },
        {
            "id": 1003,
            "created_at": "2025-06-16T09:15:00-04:00",
            "total_price": "24.99",
            "subtotal_price": "24.99",
            "financial_status": "paid",
            "line_items": [{"id": 4}],
            "shipping_address": {
                "zip": "60601",
                "province_code": "IL",
                "city": "Chicago",
            },
        },
        {
            "id": 1004,
            "created_at": "2025-06-16T11:00:00-04:00",
            "total_price": "0.00",
            "subtotal_price": "0.00",
            "financial_status": "refunded",
            "line_items": [],
            "shipping_address": {
                "zip": "10001",
                "province_code": "NY",
                "city": "New York",
            },
        },
        {
            "id": 1005,
            "created_at": "2025-06-16T12:30:00-04:00",
            "total_price": "55.00",
            "subtotal_price": "55.00",
            "financial_status": "paid",
            "line_items": [{"id": 5}],
            "shipping_address": None,  # No shipping address
        },
    ]
}

# Second page for pagination test
SAMPLE_ORDERS_PAGE2 = {
    "orders": [
        {
            "id": 1006,
            "created_at": "2025-06-17T08:00:00-04:00",
            "total_price": "199.99",
            "subtotal_price": "199.99",
            "financial_status": "paid",
            "line_items": [{"id": 6}],
            "shipping_address": {
                "zip": "33101",
                "province_code": "FL",
                "city": "Miami",
            },
        }
    ]
}


def _make_connector() -> ShopifyConnector:
    config = ShopifyConfig(
        shop_domain="test-store.myshopify.com",
        access_token="shpat_test_token_123",
    )
    return ShopifyConnector(config)


def _mock_response(data: dict, status: int = 200, link: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.ok = status < 400
    resp.json.return_value = data
    resp.headers = {"Link": link} if link else {}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestShopifyConnectorInit:
    def test_base_url_construction(self):
        c = _make_connector()
        assert c.base_url == "https://test-store.myshopify.com/admin/api/2025-01"

    def test_auth_header(self):
        c = _make_connector()
        assert c.session.headers["X-Shopify-Access-Token"] == "shpat_test_token_123"

    def test_optional_api_key_secret(self):
        """api_key and api_secret should be optional."""
        config = ShopifyConfig(
            shop_domain="x.myshopify.com",
            access_token="shpat_abc",
        )
        assert config.api_key == ""
        assert config.api_secret == ""


class TestFetchOrders:
    @patch.object(ShopifyConnector, "_get_paginated")
    def test_basic_order_parsing(self, mock_paginated):
        mock_paginated.return_value = SAMPLE_ORDERS["orders"]
        c = _make_connector()
        df = c.fetch_orders(date(2025, 6, 15), date(2025, 6, 17))

        assert len(df) == 5
        assert "order_id" in df.columns
        assert "shipping_zip" in df.columns
        assert "date" in df.columns
        assert df["total_price"].dtype == float

    @patch.object(ShopifyConnector, "_get_paginated")
    def test_zip_extraction(self, mock_paginated):
        mock_paginated.return_value = SAMPLE_ORDERS["orders"]
        c = _make_connector()
        df = c.fetch_orders(date(2025, 6, 15), date(2025, 6, 17))

        assert df.iloc[0]["shipping_zip"] == "10001"
        assert df.iloc[1]["shipping_zip"] == "90210"
        assert df.iloc[2]["shipping_zip"] == "60601"

    @patch.object(ShopifyConnector, "_get_paginated")
    def test_null_shipping_address(self, mock_paginated):
        mock_paginated.return_value = SAMPLE_ORDERS["orders"]
        c = _make_connector()
        df = c.fetch_orders(date(2025, 6, 15), date(2025, 6, 17))

        # Order 1005 has no shipping address
        null_row = df[df["order_id"] == 1005].iloc[0]
        assert null_row["shipping_zip"] == ""
        assert null_row["shipping_state"] == ""

    @patch.object(ShopifyConnector, "_get_paginated")
    def test_empty_response(self, mock_paginated):
        mock_paginated.return_value = []
        c = _make_connector()
        df = c.fetch_orders(date(2025, 6, 15), date(2025, 6, 17))
        assert df.empty


class TestPagination:
    def test_single_page(self):
        c = _make_connector()
        with patch.object(c.session, "request") as mock_request:
            mock_request.return_value = _mock_response(SAMPLE_ORDERS)
            result = c._get_paginated("orders", {"limit": 250})
            assert len(result) == 5
            mock_request.assert_called_once()

    def test_multi_page(self):
        c = _make_connector()
        page1_resp = _mock_response(
            SAMPLE_ORDERS,
            link='<https://test-store.myshopify.com/admin/api/2025-01/orders.json?page_info=abc>; rel="next"',
        )
        page2_resp = _mock_response(SAMPLE_ORDERS_PAGE2)

        with patch.object(c.session, "request") as mock_request:
            mock_request.side_effect = [page1_resp, page2_resp]
            result = c._get_paginated("orders", {"limit": 250})
            assert len(result) == 6  # 5 from page 1 + 1 from page 2
            assert mock_request.call_count == 2


class TestDMAggregation:
    @patch.object(ShopifyConnector, "fetch_orders")
    def test_dma_mapping(self, mock_fetch):
        orders = pd.DataFrame([
            {"order_id": 1, "created_at": pd.Timestamp("2025-06-15"), "total_price": 100.0,
             "shipping_zip": "10001", "financial_status": "paid", "date": date(2025, 6, 15)},
            {"order_id": 2, "created_at": pd.Timestamp("2025-06-15"), "total_price": 200.0,
             "shipping_zip": "10002", "financial_status": "paid", "date": date(2025, 6, 15)},
            {"order_id": 3, "created_at": pd.Timestamp("2025-06-15"), "total_price": 150.0,
             "shipping_zip": "90210", "financial_status": "paid", "date": date(2025, 6, 15)},
        ])
        mock_fetch.return_value = orders

        c = _make_connector()
        daily = c.get_daily_revenue_by_dma(date(2025, 6, 15), date(2025, 6, 15))

        assert len(daily) == 2  # Two DMAs: NYC (501) and LA (803)
        nyc = daily[daily["dma_code"] == "501"]
        la = daily[daily["dma_code"] == "803"]
        assert nyc["revenue"].iloc[0] == 300.0  # $100 + $200
        assert la["revenue"].iloc[0] == 150.0
        assert nyc["orders"].iloc[0] == 2
        assert la["orders"].iloc[0] == 1

    @patch.object(ShopifyConnector, "fetch_orders")
    def test_refunded_orders_excluded(self, mock_fetch):
        orders = pd.DataFrame([
            {"order_id": 1, "created_at": pd.Timestamp("2025-06-15"), "total_price": 100.0,
             "shipping_zip": "10001", "financial_status": "paid", "date": date(2025, 6, 15)},
            {"order_id": 2, "created_at": pd.Timestamp("2025-06-15"), "total_price": 50.0,
             "shipping_zip": "10002", "financial_status": "refunded", "date": date(2025, 6, 15)},
        ])
        mock_fetch.return_value = orders

        c = _make_connector()
        daily = c.get_daily_revenue_by_dma(date(2025, 6, 15), date(2025, 6, 15))

        # Only the paid order should be counted
        assert daily["revenue"].sum() == 100.0

    @patch.object(ShopifyConnector, "fetch_orders")
    def test_unmapped_zips_dropped(self, mock_fetch):
        orders = pd.DataFrame([
            {"order_id": 1, "created_at": pd.Timestamp("2025-06-15"), "total_price": 100.0,
             "shipping_zip": "10001", "financial_status": "paid", "date": date(2025, 6, 15)},
            {"order_id": 2, "created_at": pd.Timestamp("2025-06-15"), "total_price": 200.0,
             "shipping_zip": "", "financial_status": "paid", "date": date(2025, 6, 15)},
        ])
        mock_fetch.return_value = orders

        c = _make_connector()
        daily = c.get_daily_revenue_by_dma(date(2025, 6, 15), date(2025, 6, 15))

        # Only zip 10001 maps to DMA 501
        assert len(daily) == 1
        assert daily["dma_code"].iloc[0] == "501"


# ---------------------------------------------------------------------------
# Integration Test (requires real Shopify credentials)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("SHOPIFY_DOMAIN") or not os.environ.get("SHOPIFY_TOKEN"),
    reason="Set SHOPIFY_DOMAIN and SHOPIFY_TOKEN env vars to run live test",
)
class TestShopifyLive:
    def test_shopify_live(self):
        """Live integration test against real Shopify store.

        Run with:
            SHOPIFY_DOMAIN=michael-todd-beauty-2017.myshopify.com \
            SHOPIFY_TOKEN=shpat_xxx \
            pytest tests/test_shopify_connector.py -k live -v
        """
        config = ShopifyConfig(
            shop_domain=os.environ["SHOPIFY_DOMAIN"],
            access_token=os.environ["SHOPIFY_TOKEN"],
        )
        connector = ShopifyConnector(config)

        end = date.today()
        start = end - pd.Timedelta(days=30)

        # Test order fetching
        orders = connector.fetch_orders(start.date(), end)
        assert isinstance(orders, pd.DataFrame)
        print(f"\nOrders fetched: {len(orders)}")

        if not orders.empty:
            assert "order_id" in orders.columns
            assert "shipping_zip" in orders.columns
            assert "total_price" in orders.columns
            print(f"Revenue range: ${orders['total_price'].min():.2f} - ${orders['total_price'].max():.2f}")
            print(f"Total revenue: ${orders['total_price'].sum():,.2f}")

            # Test DMA aggregation
            daily = connector.get_daily_revenue_by_dma(start.date(), end)
            assert isinstance(daily, pd.DataFrame)
            print(f"DMA rows: {len(daily)}, unique DMAs: {daily['dma_code'].nunique()}")
            print(f"Top DMAs:\n{daily.groupby('dma_code')['revenue'].sum().nlargest(5)}")
