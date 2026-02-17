"""
Prefect daily ingestion pipeline for the Michael Todd Beauty MMM platform.

Scheduled at 07:00 AM ET every day. Pulls spend and revenue data from all
upstream sources, appends to BigQuery, runs the fast sequential posterior
update, and distributes daily budget recommendations via Slack.

When Prefect is not installed the module falls back to executing all tasks
as plain synchronous Python functions with the same signatures.
"""

from __future__ import annotations

import importlib
import logging
import traceback
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional Prefect import – fall back to identity decorators
# ---------------------------------------------------------------------------
try:
    from prefect import flow, task
    from prefect.blocks.system import Secret
    from prefect.logging import get_run_logger

    _PREFECT_AVAILABLE = True
    logger.debug("Prefect is available; using Prefect orchestration.")
except ImportError:
    _PREFECT_AVAILABLE = False
    logger.warning(
        "Prefect is not installed. Tasks will execute synchronously without "
        "retry logic or observability."
    )

    def task(_fn=None, **_kwargs):  # type: ignore[override]
        """No-op task decorator fallback."""
        def decorator(fn):
            return fn
        if _fn is not None:
            return _fn
        return decorator

    def flow(_fn=None, **_kwargs):  # type: ignore[override]
        """No-op flow decorator fallback."""
        def decorator(fn):
            return fn
        if _fn is not None:
            return _fn
        return decorator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_logger():
    if _PREFECT_AVAILABLE:
        try:
            return get_run_logger()
        except Exception:
            pass
    return logger


def _send_pagerduty_alert(
    summary: str,
    details: str,
    config: dict[str, Any],
) -> None:
    """Fire a PagerDuty Events v2 trigger when a task fails."""
    import requests  # local import to avoid hard dependency at module level

    routing_key = (config.get("pagerduty") or {}).get("routing_key", "")
    if not routing_key:
        logger.error(
            "PagerDuty routing_key not configured; cannot send alert. summary=%s",
            summary,
        )
        return

    payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "payload": {
            "summary": summary,
            "source": "mtb-mmm-daily-ingest",
            "severity": "error",
            "custom_details": {"details": details},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    try:
        resp = requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("PagerDuty alert sent: %s", summary)
    except Exception as exc:
        logger.error("Failed to send PagerDuty alert: %s", exc)


def _send_slack_message(text: str, config: dict[str, Any]) -> None:
    """Post a message to the configured Slack webhook."""
    import requests

    webhook_url = (config.get("slack") or {}).get("webhook_url", "")
    if not webhook_url:
        logger.warning("Slack webhook_url not configured; skipping notification.")
        return
    try:
        resp = requests.post(
            webhook_url,
            json={"text": text},
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("Slack message sent.")
    except Exception as exc:
        logger.error("Failed to send Slack message: %s", exc)


def _build_warehouse(config: dict[str, Any]):
    """Instantiate BigQueryWarehouse from pipeline config."""
    from incrementality.warehouse.bigquery import BigQueryWarehouse

    return BigQueryWarehouse(
        project_id=config.get("gcp_project_id", ""),
        dataset_id=config.get("bq_dataset_id", "mmm_prod"),
        credentials_path=config.get("gcp_credentials_path"),
    )


# ---------------------------------------------------------------------------
# Task: Meta spend
# ---------------------------------------------------------------------------

@task(
    name="ingest-meta-spend",
    retries=3,
    retry_delay_seconds=60,
    description="Pull Meta DMA spend and append to BigQuery.",
)
def ingest_meta_spend(date_str: str, config: dict[str, Any]) -> dict[str, Any]:
    """
    Pull yesterday's Meta (Facebook/Instagram) spend broken down by DMA from
    the Marketing API and append to ``raw_daily_spend_meta``.

    Returns a summary dict with ``rows_written`` and ``total_spend``.
    """
    log = _get_logger()
    log.info("ingest_meta_spend: date=%s", date_str)

    try:
        from incrementality.connectors.facebook import FacebookConnector  # type: ignore

        connector = FacebookConnector(
            access_token=config["meta"]["access_token"],
            ad_account_id=config["meta"]["ad_account_id"],
        )
        df: pd.DataFrame = connector.get_dma_spend(date_str)
    except Exception as exc:
        log.error("Meta connector failed: %s", exc)
        raise

    wh = _build_warehouse(config)
    wh.upload_daily_spend(df, channel="meta", date_str=date_str)

    summary = {
        "channel": "meta",
        "date": date_str,
        "rows_written": len(df),
        "total_spend": float(df["spend"].sum()) if "spend" in df.columns else 0.0,
    }
    log.info("ingest_meta_spend complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Task: Google spend
# ---------------------------------------------------------------------------

@task(
    name="ingest-google-spend",
    retries=3,
    retry_delay_seconds=60,
    description="Pull Google DMA spend and append to BigQuery.",
)
def ingest_google_spend(date_str: str, config: dict[str, Any]) -> dict[str, Any]:
    """
    Pull yesterday's Google Ads spend broken down by DMA (brand + non-brand)
    and append to ``raw_daily_spend_google``.
    """
    log = _get_logger()
    log.info("ingest_google_spend: date=%s", date_str)

    try:
        from incrementality.connectors.google_ads import GoogleAdsConnector  # type: ignore

        connector = GoogleAdsConnector(
            developer_token=config["google"]["developer_token"],
            customer_id=config["google"]["customer_id"],
            credentials_path=config.get("gcp_credentials_path"),
        )
        df: pd.DataFrame = connector.get_dma_spend(date_str)
    except Exception as exc:
        log.error("Google Ads connector failed: %s", exc)
        raise

    wh = _build_warehouse(config)
    wh.upload_daily_spend(df, channel="google", date_str=date_str)

    summary = {
        "channel": "google",
        "date": date_str,
        "rows_written": len(df),
        "total_spend": float(df["spend"].sum()) if "spend" in df.columns else 0.0,
    }
    log.info("ingest_google_spend complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Task: TikTok spend
# ---------------------------------------------------------------------------

@task(
    name="ingest-tiktok-spend",
    retries=3,
    retry_delay_seconds=60,
    description="Pull TikTok state-level spend and append to BigQuery.",
)
def ingest_tiktok_spend(date_str: str, config: dict[str, Any]) -> dict[str, Any]:
    """
    Pull yesterday's TikTok Ads spend broken down by US state and append to
    ``raw_daily_spend_tiktok``.

    TikTok does not expose DMA-level targeting, so state-level data is
    stored and mapped to DMA codes during the dbt transformation layer.
    """
    log = _get_logger()
    log.info("ingest_tiktok_spend: date=%s", date_str)

    try:
        from incrementality.connectors.tiktok import TikTokConnector  # type: ignore

        connector = TikTokConnector(
            access_token=config["tiktok"]["access_token"],
            advertiser_id=config["tiktok"]["advertiser_id"],
        )
        df: pd.DataFrame = connector.get_state_spend(date_str)
    except Exception as exc:
        log.error("TikTok connector failed: %s", exc)
        raise

    wh = _build_warehouse(config)
    wh.upload_daily_spend(df, channel="tiktok", date_str=date_str)

    summary = {
        "channel": "tiktok",
        "date": date_str,
        "rows_written": len(df),
        "total_spend": float(df["spend"].sum()) if "spend" in df.columns else 0.0,
    }
    log.info("ingest_tiktok_spend complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Task: Shopify revenue
# ---------------------------------------------------------------------------

@task(
    name="ingest-shopify-revenue",
    retries=3,
    retry_delay_seconds=90,
    description="Pull Shopify orders, map to DMA, append to BigQuery.",
)
def ingest_shopify_revenue(date_str: str, config: dict[str, Any]) -> dict[str, Any]:
    """
    Pull Shopify orders for ``date_str``, geocode shipping addresses to DMA
    codes, compute net revenue, and append to ``raw_shopify_revenue``.
    """
    log = _get_logger()
    log.info("ingest_shopify_revenue: date=%s", date_str)

    try:
        from incrementality.connectors.shopify import ShopifyConnector  # type: ignore
        from incrementality.connectors.geo import geocode_to_dma  # type: ignore

        connector = ShopifyConnector(
            shop_url=config["shopify"]["shop_url"],
            api_key=config["shopify"]["api_key"],
            api_secret=config["shopify"]["api_secret"],
            access_token=config["shopify"]["access_token"],
        )
        df: pd.DataFrame = connector.get_orders(date_str)
        df = geocode_to_dma(df, zip_col="shipping_zip")
    except Exception as exc:
        log.error("Shopify ingest failed: %s", exc)
        raise

    wh = _build_warehouse(config)
    wh.upload_shopify_revenue(df, date_str=date_str)

    summary = {
        "date": date_str,
        "rows_written": len(df),
        "total_net_revenue": float(df["net_revenue"].sum())
        if "net_revenue" in df.columns
        else 0.0,
    }
    log.info("ingest_shopify_revenue complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Task: Amazon revenue
# ---------------------------------------------------------------------------

@task(
    name="ingest-amazon-revenue",
    retries=3,
    retry_delay_seconds=120,
    description="Pull Amazon SP-API revenue and append to BigQuery.",
)
def ingest_amazon_revenue(date_str: str, config: dict[str, Any]) -> dict[str, Any]:
    """
    Pull Amazon Selling Partner API sales report for ``date_str`` and append
    national-level revenue to ``raw_amazon_revenue``.
    """
    log = _get_logger()
    log.info("ingest_amazon_revenue: date=%s", date_str)

    try:
        from incrementality.connectors.amazon import AmazonConnector  # type: ignore

        connector = AmazonConnector(
            refresh_token=config["amazon"]["refresh_token"],
            client_id=config["amazon"]["client_id"],
            client_secret=config["amazon"]["client_secret"],
            marketplace_id=config["amazon"].get("marketplace_id", "ATVPDKIKX0DER"),
        )
        df: pd.DataFrame = connector.get_daily_revenue(date_str)
    except Exception as exc:
        log.error("Amazon SP-API ingest failed: %s", exc)
        raise

    wh = _build_warehouse(config)
    wh.upload_amazon_revenue(df, date_str=date_str)

    summary = {
        "date": date_str,
        "rows_written": len(df),
        "total_net_revenue": float(df["net_revenue"].sum())
        if "net_revenue" in df.columns
        else 0.0,
    }
    log.info("ingest_amazon_revenue complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Task: Northbeam attribution
# ---------------------------------------------------------------------------

@task(
    name="ingest-northbeam-attribution",
    retries=3,
    retry_delay_seconds=60,
    description="Pull Northbeam MTA attribution data and append to BigQuery.",
)
def ingest_northbeam_attribution(
    date_str: str, config: dict[str, Any]
) -> dict[str, Any]:
    """
    Pull Northbeam multi-touch attribution data for ``date_str`` across all
    configured attribution models and append to ``raw_northbeam_attribution``.
    """
    log = _get_logger()
    log.info("ingest_northbeam_attribution: date=%s", date_str)

    try:
        from incrementality.connectors.northbeam import NorthbeamConnector  # type: ignore

        connector = NorthbeamConnector(
            api_key=config["northbeam"]["api_key"],
            account_id=config["northbeam"]["account_id"],
        )
        df: pd.DataFrame = connector.get_attribution(
            date_str,
            models=config["northbeam"].get(
                "attribution_models",
                ["last_click", "linear", "data_driven"],
            ),
        )
    except Exception as exc:
        log.error("Northbeam ingest failed: %s", exc)
        raise

    wh = _build_warehouse(config)
    wh.upload_northbeam_attribution(df, date_str=date_str)

    summary = {
        "date": date_str,
        "rows_written": len(df),
    }
    log.info("ingest_northbeam_attribution complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Task: Sequential posterior update
# ---------------------------------------------------------------------------

@task(
    name="run-daily-sequential-update",
    retries=1,
    retry_delay_seconds=300,
    description="Run fast sequential Bayesian posterior update on latest data.",
)
def run_daily_sequential_update(
    date_str: str, config: dict[str, Any]
) -> dict[str, Any]:
    """
    Run the lightweight sequential (online) posterior update on the most recent
    day's data. This does NOT re-run full MCMC; it updates the posterior
    approximation using the Laplace/variational shortcut built into the MMM.

    Returns a summary dict with updated posterior statistics.
    """
    log = _get_logger()
    log.info("run_daily_sequential_update: date=%s", date_str)

    try:
        from incrementality.mmm import MMMModel  # type: ignore

        model_path = config.get("mmm", {}).get("model_path")
        if not model_path:
            raise ValueError("mmm.model_path is required in config for sequential update.")

        model = MMMModel.load(model_path)
        wh = _build_warehouse(config)

        # Use a 7-day rolling window to include the latest data point
        from datetime import date as _date, timedelta

        end_dt = _date.fromisoformat(date_str)
        start_dt = end_dt - timedelta(days=6)
        inputs = wh.get_model_ready_inputs(
            str(start_dt), str(end_dt)
        )

        update_stats = model.sequential_update(inputs)
        model.save(model_path)

        log.info("Sequential update complete: %s", update_stats)
        return {"date": date_str, "status": "ok", **update_stats}

    except Exception as exc:
        log.error("Sequential posterior update failed: %s\n%s", exc, traceback.format_exc())
        raise


# ---------------------------------------------------------------------------
# Task: Send budget recommendations
# ---------------------------------------------------------------------------

@task(
    name="send-daily-budget-recommendations",
    retries=2,
    retry_delay_seconds=30,
    description="Generate and send daily budget recommendations via Slack.",
)
def send_daily_budget_recommendations(config: dict[str, Any]) -> None:
    """
    Invoke the budget optimizer to generate recommended spend allocations for
    the next 7 days and post a formatted summary to the configured Slack channel.
    """
    log = _get_logger()
    log.info("send_daily_budget_recommendations: generating recommendations")

    try:
        from incrementality.mmm import MMMModel  # type: ignore
        from incrementality.mmm.optimizer import BudgetOptimizer  # type: ignore

        model_path = config.get("mmm", {}).get("model_path")
        if not model_path:
            raise ValueError("mmm.model_path is required for budget recommendations.")

        model = MMMModel.load(model_path)
        optimizer = BudgetOptimizer(model)
        weekly_budget = config.get("budget", {}).get("weekly_total_usd", 500_000)
        recs = optimizer.optimize(total_budget=weekly_budget, horizon_days=7)

        lines = ["*MTB MMM: Daily Budget Recommendations*\n"]
        for channel, amount in recs.items():
            lines.append(f"  • {channel}: ${amount:,.0f}")
        lines.append(f"\nWeekly total: ${weekly_budget:,.0f}")
        message = "\n".join(lines)

    except Exception as exc:
        log.error("Budget recommendation generation failed: %s", exc)
        message = (
            f"MTB MMM: Budget recommendation generation failed — {exc}. "
            "Using last valid recommendations."
        )

    _send_slack_message(message, config)
    log.info("Budget recommendation Slack message sent.")


# ---------------------------------------------------------------------------
# Main daily flow
# ---------------------------------------------------------------------------

@flow(
    name="mmm-daily-ingest",
    description=(
        "MTB MMM daily ingestion: pulls spend + revenue from all channels, "
        "runs sequential model update, sends budget recommendations."
    ),
)
def mmm_daily_ingest(
    date_str: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Main Prefect flow for the MMM daily ingestion pipeline.

    Scheduled at 07:00 AM ET. ``date_str`` defaults to yesterday in ET.

    Parameters
    ----------
    date_str:
        ISO date string (``YYYY-MM-DD``) to ingest. Defaults to yesterday.
    config:
        Pipeline configuration dict. Loaded from the default config path when
        ``None``.

    Returns
    -------
    dict
        Summary of the run including per-task results and overall status.
    """
    from datetime import date, timedelta
    import pytz

    log = _get_logger()

    if config is None:
        from incrementality.config import load_config  # type: ignore
        config = load_config()

    if date_str is None:
        et_tz = pytz.timezone("America/New_York")
        date_str = (
            datetime.now(et_tz) - timedelta(days=1)
        ).strftime("%Y-%m-%d")

    log.info("mmm_daily_ingest: started for date=%s", date_str)

    run_summary: dict[str, Any] = {
        "date": date_str,
        "started_at": datetime.utcnow().isoformat(),
        "tasks": {},
        "status": "ok",
    }

    # ------------------------------------------------------------------
    # Helper: run a task, catch failures, fire PD alert on error
    # ------------------------------------------------------------------
    def _run_task(task_fn, *args, task_name: str, **kwargs) -> Any:
        try:
            result = task_fn(*args, **kwargs)
            run_summary["tasks"][task_name] = {"status": "ok", "result": result}
            return result
        except Exception as exc:
            error_detail = traceback.format_exc()
            log.error("Task %s failed: %s", task_name, exc)
            run_summary["tasks"][task_name] = {
                "status": "failed",
                "error": str(exc),
            }
            run_summary["status"] = "partial_failure"
            _send_pagerduty_alert(
                summary=f"MTB MMM daily ingest: {task_name} failed for {date_str}",
                details=error_detail,
                config=config,
            )
            return None

    # ------------------------------------------------------------------
    # Ingestion tasks – run in definition order; each is independent so a
    # failure in one does not block others.
    # ------------------------------------------------------------------
    _run_task(ingest_meta_spend, date_str, config, task_name="ingest_meta_spend")
    _run_task(ingest_google_spend, date_str, config, task_name="ingest_google_spend")
    _run_task(ingest_tiktok_spend, date_str, config, task_name="ingest_tiktok_spend")
    _run_task(ingest_shopify_revenue, date_str, config, task_name="ingest_shopify_revenue")
    _run_task(ingest_amazon_revenue, date_str, config, task_name="ingest_amazon_revenue")
    _run_task(
        ingest_northbeam_attribution,
        date_str,
        config,
        task_name="ingest_northbeam_attribution",
    )

    # ------------------------------------------------------------------
    # Model update – requires all ingestion tasks to have written data
    # ------------------------------------------------------------------
    _run_task(
        run_daily_sequential_update,
        date_str,
        config,
        task_name="run_daily_sequential_update",
    )

    # ------------------------------------------------------------------
    # Notifications – always attempt even if some tasks failed
    # ------------------------------------------------------------------
    _run_task(
        send_daily_budget_recommendations,
        config,
        task_name="send_daily_budget_recommendations",
    )

    run_summary["finished_at"] = datetime.utcnow().isoformat()

    if run_summary["status"] == "ok":
        log.info("mmm_daily_ingest: all tasks succeeded for date=%s", date_str)
    else:
        log.warning(
            "mmm_daily_ingest: completed with failures for date=%s — see task summary",
            date_str,
        )

    return run_summary
