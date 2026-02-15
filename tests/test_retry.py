"""Tests for retry logic with exponential backoff.

Validates:
- Successful requests pass through immediately
- 429 and 5xx responses trigger retries with backoff
- Connection errors and timeouts trigger retries
- Non-retryable 4xx errors raise immediately
- Retry-After header is respected
- Max retries are enforced
- Integration with Shopify and Facebook connectors
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
import requests

from incrementality.connectors.retry import (
    DEFAULT_MAX_RETRIES,
    RetryableRequestError,
    _compute_delay,
    _parse_retry_after,
    request_with_retry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status: int = 200, json_data: dict | None = None,
                   headers: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.json.return_value = json_data or {}
    resp.headers = headers or {}
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            f"HTTP {status}", response=resp
        )
    return resp


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

class TestComputeDelay:
    def test_exponential_growth(self):
        # Delays should grow exponentially (ignoring jitter)
        d0 = _compute_delay(0, base_delay=1.0, max_delay=60.0)
        d1 = _compute_delay(1, base_delay=1.0, max_delay=60.0)
        d2 = _compute_delay(2, base_delay=1.0, max_delay=60.0)
        # d0 ~ 1*2^0 + jitter = 1-2, d1 ~ 1*2^1 + jitter = 2-3, d2 ~ 1*2^2 + jitter = 4-5
        assert d0 < d1 < d2

    def test_max_delay_cap(self):
        delay = _compute_delay(10, base_delay=1.0, max_delay=5.0)
        assert delay <= 5.0

    def test_retry_after_respected(self):
        delay = _compute_delay(0, base_delay=1.0, max_delay=60.0, retry_after=30.0)
        # Should be ~30 + small jitter
        assert 30.0 <= delay <= 31.0

    def test_retry_after_capped_by_max(self):
        delay = _compute_delay(0, base_delay=1.0, max_delay=10.0, retry_after=30.0)
        assert delay <= 10.0


class TestParseRetryAfter:
    def test_numeric_header(self):
        resp = MagicMock()
        resp.headers = {"Retry-After": "5"}
        assert _parse_retry_after(resp) == 5.0

    def test_missing_header(self):
        resp = MagicMock()
        resp.headers = {}
        assert _parse_retry_after(resp) is None

    def test_invalid_header(self):
        resp = MagicMock()
        resp.headers = {"Retry-After": "not-a-number"}
        assert _parse_retry_after(resp) is None


# ---------------------------------------------------------------------------
# Core retry logic tests
# ---------------------------------------------------------------------------

class TestRequestWithRetry:
    @patch("incrementality.connectors.retry.time.sleep")
    def test_success_on_first_try(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        session.request.return_value = _mock_response(200, {"ok": True})

        resp = request_with_retry(session, "GET", "https://api.example.com/test")

        assert resp.status_code == 200
        session.request.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("incrementality.connectors.retry.time.sleep")
    def test_retry_on_429(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        rate_limited = _mock_response(429, headers={"Retry-After": "1"})
        success = _mock_response(200, {"ok": True})
        session.request.side_effect = [rate_limited, success]

        resp = request_with_retry(session, "GET", "https://api.example.com/test",
                                  max_retries=3)

        assert resp.status_code == 200
        assert session.request.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("incrementality.connectors.retry.time.sleep")
    def test_retry_on_500(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        error_resp = _mock_response(500)
        success = _mock_response(200, {"ok": True})
        session.request.side_effect = [error_resp, success]

        resp = request_with_retry(session, "GET", "https://api.example.com/test",
                                  max_retries=3)

        assert resp.status_code == 200
        assert session.request.call_count == 2

    @patch("incrementality.connectors.retry.time.sleep")
    def test_retry_on_502(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        error_resp = _mock_response(502)
        success = _mock_response(200, {"ok": True})
        session.request.side_effect = [error_resp, success]

        resp = request_with_retry(session, "GET", "https://api.example.com/test",
                                  max_retries=3)

        assert resp.status_code == 200

    @patch("incrementality.connectors.retry.time.sleep")
    def test_retry_on_503(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        error_resp = _mock_response(503)
        success = _mock_response(200, {"ok": True})
        session.request.side_effect = [error_resp, success]

        resp = request_with_retry(session, "GET", "https://api.example.com/test",
                                  max_retries=3)

        assert resp.status_code == 200

    @patch("incrementality.connectors.retry.time.sleep")
    def test_retry_on_connection_error(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        session.request.side_effect = [
            requests.ConnectionError("Connection reset"),
            _mock_response(200, {"ok": True}),
        ]

        resp = request_with_retry(session, "GET", "https://api.example.com/test",
                                  max_retries=3)

        assert resp.status_code == 200
        assert session.request.call_count == 2

    @patch("incrementality.connectors.retry.time.sleep")
    def test_retry_on_timeout(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        session.request.side_effect = [
            requests.Timeout("Request timed out"),
            _mock_response(200, {"ok": True}),
        ]

        resp = request_with_retry(session, "GET", "https://api.example.com/test",
                                  max_retries=3)

        assert resp.status_code == 200
        assert session.request.call_count == 2

    @patch("incrementality.connectors.retry.time.sleep")
    def test_no_retry_on_400(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        session.request.return_value = _mock_response(400)

        with pytest.raises(requests.HTTPError):
            request_with_retry(session, "GET", "https://api.example.com/test",
                               max_retries=3)

        session.request.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("incrementality.connectors.retry.time.sleep")
    def test_no_retry_on_401(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        session.request.return_value = _mock_response(401)

        with pytest.raises(requests.HTTPError):
            request_with_retry(session, "GET", "https://api.example.com/test",
                               max_retries=3)

        session.request.assert_called_once()

    @patch("incrementality.connectors.retry.time.sleep")
    def test_no_retry_on_403(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        session.request.return_value = _mock_response(403)

        with pytest.raises(requests.HTTPError):
            request_with_retry(session, "GET", "https://api.example.com/test",
                               max_retries=3)

        session.request.assert_called_once()

    @patch("incrementality.connectors.retry.time.sleep")
    def test_exhausted_retries_raises(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        session.request.return_value = _mock_response(429)

        with pytest.raises(RetryableRequestError) as exc_info:
            request_with_retry(session, "GET", "https://api.example.com/test",
                               max_retries=2)

        assert exc_info.value.attempts == 3  # 1 initial + 2 retries
        assert session.request.call_count == 3

    @patch("incrementality.connectors.retry.time.sleep")
    def test_exhausted_retries_on_connection_error(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        session.request.side_effect = requests.ConnectionError("down")

        with pytest.raises(RetryableRequestError) as exc_info:
            request_with_retry(session, "GET", "https://api.example.com/test",
                               max_retries=2)

        assert exc_info.value.attempts == 3
        assert session.request.call_count == 3

    @patch("incrementality.connectors.retry.time.sleep")
    def test_multiple_failures_then_success(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        session.request.side_effect = [
            _mock_response(503),
            requests.ConnectionError("blip"),
            _mock_response(429),
            _mock_response(200, {"ok": True}),
        ]

        resp = request_with_retry(session, "GET", "https://api.example.com/test",
                                  max_retries=4)

        assert resp.status_code == 200
        assert session.request.call_count == 4
        assert mock_sleep.call_count == 3

    @patch("incrementality.connectors.retry.time.sleep")
    def test_post_method(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        session.request.return_value = _mock_response(200, {"id": "123"})

        resp = request_with_retry(session, "POST", "https://api.example.com/test",
                                  json={"key": "value"})

        assert resp.status_code == 200
        session.request.assert_called_once_with(
            "POST", "https://api.example.com/test",
            json={"key": "value"},
        )


# ---------------------------------------------------------------------------
# Integration: Shopify connector uses retry
# ---------------------------------------------------------------------------

class TestShopifyRetryIntegration:
    @patch("incrementality.connectors.retry.time.sleep")
    def test_shopify_pagination_retries_on_429(self, mock_sleep):
        from incrementality.config import ShopifyConfig
        from incrementality.connectors.shopify import ShopifyConnector

        config = ShopifyConfig(
            shop_domain="test.myshopify.com",
            access_token="shpat_test",
        )
        c = ShopifyConnector(config)

        rate_limited = _mock_response(429, headers={"Retry-After": "1"})
        success = _mock_response(200, {"orders": [{"id": 1}]})
        success.headers = {}

        with patch.object(c.session, "request", side_effect=[rate_limited, success]):
            result = c._get_paginated("orders", {"limit": 250})

        assert len(result) == 1
        assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# Integration: Facebook connector uses retry
# ---------------------------------------------------------------------------

class TestFacebookRetryIntegration:
    @patch("incrementality.connectors.retry.time.sleep")
    def test_facebook_get_retries_on_500(self, mock_sleep):
        from incrementality.config import FacebookConfig
        from incrementality.connectors.facebook import FacebookConnector

        config = FacebookConfig(
            app_id="test_app",
            app_secret="test_secret",
            access_token="test_token",
            ad_account_id="act_123",
        )
        c = FacebookConnector(config)

        error_resp = _mock_response(500)
        success = _mock_response(200, {"data": []})

        with patch.object(c.session, "request", side_effect=[error_resp, success]):
            result = c._get("act_123/campaigns", params={"fields": "id"})

        assert result == {"data": []}
        assert mock_sleep.call_count == 1

    @patch("incrementality.connectors.retry.time.sleep")
    def test_facebook_post_retries_on_429(self, mock_sleep):
        from incrementality.config import FacebookConfig
        from incrementality.connectors.facebook import FacebookConnector

        config = FacebookConfig(
            app_id="test_app",
            app_secret="test_secret",
            access_token="test_token",
            ad_account_id="act_123",
        )
        c = FacebookConnector(config)

        rate_limited = _mock_response(429)
        success = _mock_response(200, {"success": True})

        with patch.object(c.session, "request", side_effect=[rate_limited, success]):
            result = c._post("adset_001", data={"targeting": "{}"})

        assert result == {"success": True}
        assert mock_sleep.call_count == 1
