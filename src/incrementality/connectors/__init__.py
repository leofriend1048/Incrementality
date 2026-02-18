"""Data connectors for all supported ad platforms and revenue sources.

Revenue:
    ShopifyConnector, AmazonConnector

Paid Social:
    FacebookConnector, YouTubeConnector,
    TikTokConnector, TikTokShopConnector,
    PinterestConnector

Mobile / CTV:
    AppLovinConnector

TV:
    TatariConnector

Owned / Retention:
    PostscriptConnector, KlaviyoConnector

MTA Calibration:
    NorthbeamConnector
"""

from incrementality.connectors.amazon import AmazonConnector
from incrementality.connectors.applovin import AppLovinConnector
from incrementality.connectors.facebook import FacebookConnector
from incrementality.connectors.klaviyo import KlaviyoConnector
from incrementality.connectors.northbeam import NorthbeamConnector
from incrementality.connectors.pinterest import PinterestConnector
from incrementality.connectors.postscript import PostscriptConnector
from incrementality.connectors.shopify import ShopifyConnector
from incrementality.connectors.tatari import TatariConnector
from incrementality.connectors.tiktok import TikTokConnector
from incrementality.connectors.tiktok_shop import TikTokShopConnector
from incrementality.connectors.youtube import YouTubeConnector

__all__ = [
    # Revenue
    "AmazonConnector",
    "ShopifyConnector",
    # Paid social
    "FacebookConnector",
    "YouTubeConnector",
    "TikTokConnector",
    "TikTokShopConnector",
    "PinterestConnector",
    # Mobile / CTV
    "AppLovinConnector",
    # TV
    "TatariConnector",
    # Owned / retention
    "PostscriptConnector",
    "KlaviyoConnector",
    # MTA calibration
    "NorthbeamConnector",
]
