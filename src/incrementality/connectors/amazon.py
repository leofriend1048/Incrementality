"""Amazon data connector.

Pulls order-level data from the Amazon Selling Partner (SP-API)
and aggregates to DMA-level daily metrics.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

from incrementality.config import AmazonConfig
from incrementality.connectors.geo import zip_to_dma

logger = logging.getLogger(__name__)

# SP-API endpoints by region
_ENDPOINTS = {
    "na": "https://sellingpartnerapi-na.amazon.com",
    "eu": "https://sellingpartnerapi-eu.amazon.com",
    "fe": "https://sellingpartnerapi-fe.amazon.com",
}

_TOKEN_URL = "https://api.amazon.com/auth/o2/token"


class AmazonConnector:
    """Connects to Amazon SP-API and pulls order data by DMA."""

    def __init__(self, config: AmazonConfig):
        self.config = config
        self.base_url = _ENDPOINTS[config.region]
        self.session = requests.Session()
        self._access_token: str | None = None

    def _refresh_access_token(self) -> str:
        """Get or refresh the LWA access token."""
        resp = requests.post(_TOKEN_URL, data={
            "grant_type": "refresh_token",
            "refresh_token": self.config.refresh_token,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        })
        resp.raise_for_status()
        self._access_token = resp.json()["access_token"]
        self.session.headers.update({
            "x-amz-access-token": self._access_token,
            "Content-Type": "application/json",
        })
        return self._access_token

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        """Make an authenticated GET request to SP-API."""
        if not self._access_token:
            self._refresh_access_token()
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params)
        if resp.status_code == 403:
            # Token might be expired, refresh and retry
            self._refresh_access_token()
            resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def fetch_orders(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch orders from Amazon SP-API within date range.

        Returns columns:
            order_id, purchase_date, order_total, shipping_zip,
            shipping_state, order_status, num_items
        """
        records = []
        next_token = None
        while True:
            params: dict[str, Any] = {
                "MarketplaceIds": self.config.marketplace_id,
                "CreatedAfter": f"{start_date}T00:00:00Z",
                "CreatedBefore": f"{end_date}T23:59:59Z",
                "OrderStatuses": "Shipped,Delivered",
            }
            if next_token:
                params = {"NextToken": next_token}
            data = self._get("/orders/v0/orders", params)
            payload = data.get("payload", {})
            for order in payload.get("Orders", []):
                shipping = order.get("ShippingAddress") or {}
                records.append({
                    "order_id": order["AmazonOrderId"],
                    "purchase_date": pd.Timestamp(order.get("PurchaseDate", "")),
                    "order_total": float(
                        order.get("OrderTotal", {}).get("Amount", 0)
                    ),
                    "shipping_zip": str(
                        shipping.get("PostalCode", "")
                    ).strip()[:5],
                    "shipping_state": shipping.get("StateOrRegion", ""),
                    "order_status": order.get("OrderStatus", ""),
                    "num_items": order.get("NumberOfItemsShipped", 0),
                })
            next_token = payload.get("NextToken")
            if not next_token:
                break
            logger.debug(f"Fetched {len(records)} Amazon orders so far")

        df = pd.DataFrame(records)
        if df.empty:
            return df
        df["date"] = df["purchase_date"].dt.date
        return df

    def get_daily_revenue_by_dma(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Aggregate Amazon orders to daily revenue per DMA.

        Returns columns: date, dma_code, revenue, orders
        """
        orders = self.fetch_orders(start_date, end_date)
        if orders.empty:
            return pd.DataFrame(columns=["date", "dma_code", "revenue", "orders"])

        orders["dma_code"] = orders["shipping_zip"].apply(zip_to_dma)
        orders = orders.dropna(subset=["dma_code"])

        daily = (
            orders
            .groupby(["date", "dma_code"])
            .agg(
                revenue=("order_total", "sum"),
                orders=("order_id", "nunique"),
            )
            .reset_index()
        )
        return daily
