"""Shopify data connector.

Pulls order-level data from the Shopify Admin API and aggregates
to DMA-level daily metrics using shipping address zip-to-DMA mapping.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

from incrementality.config import ShopifyConfig
from incrementality.connectors.geo import zip_to_dma

logger = logging.getLogger(__name__)


class ShopifyConnector:
    """Connects to a Shopify store and pulls order data by DMA."""

    BASE_URL = "https://{shop_domain}/admin/api/{api_version}"

    def __init__(self, config: ShopifyConfig):
        self.config = config
        self.base_url = self.BASE_URL.format(
            shop_domain=config.shop_domain,
            api_version=config.api_version,
        )
        self.session = requests.Session()
        self.session.headers.update({
            "X-Shopify-Access-Token": config.access_token,
            "Content-Type": "application/json",
        })

    def _get_paginated(self, endpoint: str, params: dict[str, Any]) -> list[dict]:
        """Fetch all pages of a paginated Shopify endpoint."""
        results = []
        url = f"{self.base_url}/{endpoint}.json"
        while url:
            resp = self.session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            key = endpoint.split("/")[-1]
            results.extend(data.get(key, []))
            # Handle cursor-based pagination
            link_header = resp.headers.get("Link", "")
            url = None
            if 'rel="next"' in link_header:
                for part in link_header.split(","):
                    if 'rel="next"' in part:
                        url = part.split("<")[1].split(">")[0]
                        params = {}  # URL already contains params
            logger.debug(f"Fetched {len(results)} records from {endpoint}")
        return results

    def fetch_orders(
        self,
        start_date: date,
        end_date: date,
        status: str = "any",
    ) -> pd.DataFrame:
        """Fetch orders within date range and return as DataFrame.

        Returns columns:
            order_id, created_at, total_price, subtotal_price,
            shipping_zip, shipping_state, shipping_city,
            line_items_count, financial_status
        """
        params = {
            "created_at_min": f"{start_date}T00:00:00-00:00",
            "created_at_max": f"{end_date}T23:59:59-00:00",
            "status": status,
            "limit": 250,
            "fields": (
                "id,created_at,total_price,subtotal_price,"
                "shipping_address,line_items,financial_status"
            ),
        }
        raw_orders = self._get_paginated("orders", params)
        records = []
        for order in raw_orders:
            shipping = order.get("shipping_address") or {}
            records.append({
                "order_id": order["id"],
                "created_at": pd.Timestamp(order["created_at"]),
                "total_price": float(order.get("total_price", 0)),
                "subtotal_price": float(order.get("subtotal_price", 0)),
                "shipping_zip": str(shipping.get("zip", "")).strip()[:5],
                "shipping_state": shipping.get("province_code", ""),
                "shipping_city": shipping.get("city", ""),
                "line_items_count": len(order.get("line_items", [])),
                "financial_status": order.get("financial_status", ""),
            })
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df["date"] = df["created_at"].dt.date
        return df

    def get_daily_revenue_by_dma(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Aggregate Shopify orders to daily revenue per DMA.

        Returns columns: date, dma_code, revenue, orders
        """
        orders = self.fetch_orders(start_date, end_date, status="any")
        if orders.empty:
            return pd.DataFrame(columns=["date", "dma_code", "revenue", "orders"])

        # Map zip codes to DMA codes
        orders["dma_code"] = orders["shipping_zip"].apply(zip_to_dma)
        # Drop orders without DMA mapping
        orders = orders.dropna(subset=["dma_code"])
        # Only count paid orders
        orders = orders[orders["financial_status"].isin(["paid", "partially_refunded"])]

        daily = (
            orders
            .groupby(["date", "dma_code"])
            .agg(
                revenue=("total_price", "sum"),
                orders=("order_id", "nunique"),
            )
            .reset_index()
        )
        return daily
