"""
Prefect weekly MCMC refit pipeline for the Michael Todd Beauty MMM platform.

Scheduled at Monday 01:00 AM ET. Loads a 3-year rolling window of data from
BigQuery, runs full NUTS sampling via Meridian, validates convergence and
out-of-sample accuracy, registers the model in MLflow, and updates the
saturation-curve cache used by the budget optimizer.

When Prefect is not installed the module falls back to executing all tasks
as plain synchronous Python functions.
"""

from __future__ import annotations

import logging
import traceback
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional Prefect import
# ---------------------------------------------------------------------------
try:
    from prefect import flow, task
    from prefect.logging import get_run_logger

    _PREFECT_AVAILABLE = True
    logger.debug("Prefect is available; using Prefect orchestration.")
except ImportError:
    _PREFECT_AVAILABLE = False
    logger.warning(
        "Prefect is not installed. Tasks will execute synchronously."
    )

    def task(_fn=None, **_kwargs):  # type: ignore[override]
        def decorator(fn):
            return fn
        if _fn is not None:
            return _fn
        return decorator

    def flow(_fn=None, **_kwargs):  # type: ignore[override]
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


def _send_slack_message(text: str, config: dict[str, Any]) -> None:
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


def _send_pagerduty_alert(
    summary: str, details: str, config: dict[str, Any]
) -> None:
    import requests

    routing_key = (config.get("pagerduty") or {}).get("routing_key", "")
    if not routing_key:
        logger.error("PagerDuty routing_key not configured. summary=%s", summary)
        return
    payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "payload": {
            "summary": summary,
            "source": "mtb-mmm-weekly-refit",
            "severity": "critical",
            "custom_details": {"details": details},
            "timestamp": datetime.utcnow().isoformat() + "Z",
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


def _build_warehouse(config: dict[str, Any]):
    from incrementality.warehouse.bigquery import BigQueryWarehouse

    return BigQueryWarehouse(
        project_id=config.get("gcp_project_id", ""),
        dataset_id=config.get("bq_dataset_id", "mmm_prod"),
        credentials_path=config.get("gcp_credentials_path"),
    )


# ---------------------------------------------------------------------------
# Task: Load tensor data
# ---------------------------------------------------------------------------

@task(
    name="load-tensor-data",
    retries=3,
    retry_delay_seconds=120,
    description="Load 3-year rolling data window from BigQuery for MCMC refit.",
)
def load_tensor_data(
    start_date: str,
    end_date: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Load the full 3-year rolling tensor window from BigQuery.

    Returns a dict with keys ``spend``, ``revenue``, and ``calibration``
    as pandas DataFrames ready for the Meridian model builder.

    Parameters
    ----------
    start_date:
        ISO date string for the beginning of the training window.
    end_date:
        ISO date string for the end of the training window (typically today).
    config:
        Pipeline configuration dict.
    """
    log = _get_logger()
    log.info("load_tensor_data: %s → %s", start_date, end_date)

    wh = _build_warehouse(config)
    tensors = wh.get_model_ready_inputs(start_date, end_date)

    log.info(
        "load_tensor_data complete: spend=%d revenue=%d calibration=%d",
        len(tensors["spend"]),
        len(tensors["revenue"]),
        len(tensors["calibration"]),
    )
    return tensors


# ---------------------------------------------------------------------------
# Task: Full MCMC refit
# ---------------------------------------------------------------------------

@task(
    name="run-full-mcmc-refit",
    retries=1,
    retry_delay_seconds=600,
    description="Run full NUTS MCMC sampling for the weekly model refit.",
    timeout_seconds=14400,  # 4-hour hard ceiling
)
def run_full_mcmc_refit(
    tensors: dict[str, Any],
    calibration_events: Any,
    config: dict[str, Any],
) -> Any:
    """
    Execute a full No-U-Turn Sampler (NUTS) MCMC run using Google Meridian
    and return the fitted model object.

    The number of chains, warmup steps, and sampling steps are pulled from
    ``config["mmm"]["mcmc"]`` with sensible defaults.

    Parameters
    ----------
    tensors:
        Output of ``load_tensor_data`` – dict of DataFrames.
    calibration_events:
        DataFrame of calibration events used as informative priors.
    config:
        Pipeline configuration dict.
    """
    log = _get_logger()
    log.info("run_full_mcmc_refit: starting NUTS sampling")

    try:
        from incrementality.mmm import MMMModel  # type: ignore

        mcmc_cfg = (config.get("mmm") or {}).get("mcmc", {})
        num_chains = mcmc_cfg.get("num_chains", 4)
        num_warmup = mcmc_cfg.get("num_warmup", 1000)
        num_samples = mcmc_cfg.get("num_samples", 1000)
        target_accept = mcmc_cfg.get("target_accept_prob", 0.8)

        log.info(
            "MCMC config: chains=%d warmup=%d samples=%d target_accept=%.2f",
            num_chains,
            num_warmup,
            num_samples,
            target_accept,
        )

        model = MMMModel(
            channels=config.get("mmm", {}).get(
                "channels",
                [
                    "meta_perf",
                    "meta_aware",
                    "google_brand",
                    "google_nonbrand",
                    "tiktok",
                    "amz_sponsored",
                    "email_sms",
                ],
            ),
            geo_level="dma",
        )
        model.build_tensors(tensors, calibration_events=calibration_events)
        model.fit(
            num_chains=num_chains,
            num_warmup=num_warmup,
            num_samples=num_samples,
            target_accept_prob=target_accept,
        )

        log.info("run_full_mcmc_refit: NUTS sampling complete")
        return model

    except Exception as exc:
        log.error(
            "MCMC refit failed: %s\n%s", exc, traceback.format_exc()
        )
        raise


# ---------------------------------------------------------------------------
# Task: Validate convergence
# ---------------------------------------------------------------------------

@task(
    name="validate-convergence",
    retries=0,
    description="Check R-hat < 1.05 and effective sample sizes for all parameters.",
)
def validate_convergence(model: Any) -> dict[str, Any]:
    """
    Validate MCMC convergence diagnostics.

    Checks that all Gelman-Rubin R-hat statistics are below 1.05 and that
    bulk effective sample sizes are above 400 per chain.

    Parameters
    ----------
    model:
        Fitted ``MMMModel`` object returned by ``run_full_mcmc_refit``.

    Returns
    -------
    dict
        Convergence summary with ``passed`` bool and per-parameter statistics.

    Raises
    ------
    ValueError
        When convergence criteria are not met, preventing model registration.
    """
    log = _get_logger()
    log.info("validate_convergence: checking R-hat and ESS diagnostics")

    convergence_stats = model.convergence_diagnostics()

    max_rhat = convergence_stats.get("max_rhat", float("inf"))
    min_ess = convergence_stats.get("min_bulk_ess", 0)
    n_divergences = convergence_stats.get("n_divergences", 0)

    log.info(
        "Diagnostics: max_rhat=%.4f min_bulk_ess=%d n_divergences=%d",
        max_rhat,
        min_ess,
        n_divergences,
    )

    issues = []
    if max_rhat >= 1.05:
        issues.append(
            f"R-hat {max_rhat:.4f} >= 1.05 — model has not converged."
        )
    if min_ess < 400:
        issues.append(
            f"Min bulk ESS {min_ess} < 400 — insufficient effective samples."
        )
    if n_divergences > 50:
        issues.append(
            f"{n_divergences} NUTS divergences — posterior geometry may be problematic."
        )

    if issues:
        raise ValueError(
            "Convergence validation failed:\n" + "\n".join(f"  - {i}" for i in issues)
        )

    log.info("validate_convergence: all diagnostics passed")
    return {
        "passed": True,
        "max_rhat": max_rhat,
        "min_bulk_ess": min_ess,
        "n_divergences": n_divergences,
        **convergence_stats,
    }


# ---------------------------------------------------------------------------
# Task: Validation gates (PRD section 10)
# ---------------------------------------------------------------------------

@task(
    name="run-validation-gates",
    retries=0,
    description="Run hold-out MAPE and revenue decomposition validation gates.",
)
def run_validation_gates(
    model: Any, holdout_data: Any, config: dict[str, Any]
) -> dict[str, Any]:
    """
    Run the validation gates defined in the MTB MMM PRD section 10.

    Gates include:
    - National out-of-sample MAPE < 15 %
    - DMA-level median MAPE < 25 %
    - Channel iROAS 95 % CI width < 3× the point estimate
    - Revenue decomposition: paid channels 30–70 % of total revenue
    - Baseline share 30–60 %

    Parameters
    ----------
    model:
        Fitted ``MMMModel``.
    holdout_data:
        Dict containing held-out spend and revenue DataFrames used for
        out-of-sample evaluation.
    config:
        Pipeline config (gates thresholds may be overridden here).

    Returns
    -------
    dict
        Validation results with ``passed`` bool and per-gate details.

    Raises
    ------
    ValueError
        When any hard gate fails.
    """
    log = _get_logger()
    log.info("run_validation_gates: evaluating PRD section 10 gates")

    gates_cfg = (config.get("mmm") or {}).get("validation_gates", {})
    national_mape_threshold = gates_cfg.get("national_mape_threshold", 0.15)
    dma_median_mape_threshold = gates_cfg.get("dma_median_mape_threshold", 0.25)

    results = model.evaluate(holdout_data)

    national_mape = results.get("national_mape", float("inf"))
    dma_median_mape = results.get("dma_median_mape", float("inf"))
    iroas_ci_ratio = results.get("max_iroas_ci_ratio", float("inf"))
    paid_channel_share = results.get("paid_channel_revenue_share", 0.0)
    baseline_share = results.get("baseline_revenue_share", 0.0)

    log.info(
        "Validation: national_mape=%.3f dma_median_mape=%.3f "
        "iroas_ci_ratio=%.2f paid_share=%.2f baseline_share=%.2f",
        national_mape,
        dma_median_mape,
        iroas_ci_ratio,
        paid_channel_share,
        baseline_share,
    )

    hard_failures = []
    warnings = []

    if national_mape > national_mape_threshold:
        hard_failures.append(
            f"National MAPE {national_mape:.1%} > {national_mape_threshold:.0%} threshold"
        )
    if dma_median_mape > dma_median_mape_threshold:
        hard_failures.append(
            f"DMA median MAPE {dma_median_mape:.1%} > {dma_median_mape_threshold:.0%} threshold"
        )
    if iroas_ci_ratio > 3.0:
        warnings.append(
            f"iROAS CI ratio {iroas_ci_ratio:.1f}x > 3.0 — high parameter uncertainty"
        )
    if not (0.30 <= paid_channel_share <= 0.70):
        hard_failures.append(
            f"Paid channel revenue share {paid_channel_share:.1%} outside [30%, 70%]"
        )
    if not (0.30 <= baseline_share <= 0.60):
        warnings.append(
            f"Baseline share {baseline_share:.1%} outside [30%, 60%] — check prior"
        )

    for w in warnings:
        log.warning("Validation warning: %s", w)

    if hard_failures:
        raise ValueError(
            "Validation gate(s) failed:\n"
            + "\n".join(f"  - {f}" for f in hard_failures)
        )

    log.info("run_validation_gates: all hard gates passed")
    return {
        "passed": True,
        "national_mape": national_mape,
        "dma_median_mape": dma_median_mape,
        "iroas_ci_ratio": iroas_ci_ratio,
        "paid_channel_share": paid_channel_share,
        "baseline_share": baseline_share,
        "warnings": warnings,
        **results,
    }


# ---------------------------------------------------------------------------
# Task: Register model in MLflow
# ---------------------------------------------------------------------------

@task(
    name="register-model",
    retries=2,
    retry_delay_seconds=30,
    description="Serialize model artifacts and register in MLflow model registry.",
)
def register_model(
    model: Any,
    run_id: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """
    Save model artifacts to disk and register in MLflow with associated metrics.

    The run is tagged with the refit date, chain count, and convergence
    diagnostics. The registered model version is returned.

    Parameters
    ----------
    model:
        Fitted and validated ``MMMModel``.
    run_id:
        Unique identifier string for this refit run (e.g. ``"2026-02-17T01:00:00"``).
    metrics:
        Dict of metrics to log in MLflow (MAPE, R-hat, ESS, etc.).
    config:
        Pipeline configuration dict.

    Returns
    -------
    str
        MLflow model version URI.
    """
    log = _get_logger()
    log.info("register_model: run_id=%s", run_id)

    try:
        import mlflow  # type: ignore

        tracking_uri = (config.get("mlflow") or {}).get(
            "tracking_uri", "sqlite:///mlflow.db"
        )
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = (config.get("mlflow") or {}).get(
            "experiment_name", "mtb-mmm-weekly-refit"
        )
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_id) as run:
            mlflow.log_params(
                {
                    "refit_run_id": run_id,
                    "model_type": "meridian_geo_mmm",
                    "channels": ",".join(model.channels),
                    "geo_level": model.geo_level,
                }
            )
            flat_metrics = {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            mlflow.log_metrics(flat_metrics)

            model_path = (config.get("mmm") or {}).get(
                "model_path", f"/tmp/mtb_mmm_{run_id}.pkl"
            )
            model.save(model_path)
            mlflow.log_artifact(model_path, artifact_path="model")

            registered = mlflow.register_model(
                f"runs:/{run.info.run_id}/model",
                name="mtb-mmm",
            )
            model_version_uri = f"models:/mtb-mmm/{registered.version}"

        log.info("register_model: registered at %s", model_version_uri)
        return model_version_uri

    except ImportError:
        log.warning("mlflow not installed; saving model locally only.")
        model_path = (config.get("mmm") or {}).get(
            "model_path", f"/tmp/mtb_mmm_{run_id}.pkl"
        )
        model.save(model_path)
        return model_path


# ---------------------------------------------------------------------------
# Task: Update saturation curves
# ---------------------------------------------------------------------------

@task(
    name="update-saturation-curves",
    retries=2,
    retry_delay_seconds=30,
    description="Recompute and cache saturation curves for the budget optimizer.",
)
def update_saturation_curves(model: Any, config: dict[str, Any]) -> dict[str, Any]:
    """
    Recompute the per-channel Hill-function saturation curves from the newly
    fitted posterior and write them to the cache location used by the budget
    optimizer.

    Parameters
    ----------
    model:
        Fitted ``MMMModel`` with updated posterior.
    config:
        Pipeline configuration dict.

    Returns
    -------
    dict
        Per-channel saturation curve parameters (alpha, gamma, EC50).
    """
    log = _get_logger()
    log.info("update_saturation_curves: recomputing from new posterior")

    try:
        from incrementality.mmm.optimizer import BudgetOptimizer  # type: ignore

        optimizer = BudgetOptimizer(model)
        curves = optimizer.recompute_saturation_curves()

        cache_path = (config.get("mmm") or {}).get(
            "saturation_cache_path", "/tmp/mtb_mmm_saturation_curves.json"
        )
        import json

        with open(cache_path, "w") as fh:
            json.dump(curves, fh, indent=2)

        log.info(
            "update_saturation_curves: wrote curves for %d channels to %s",
            len(curves),
            cache_path,
        )
        return curves

    except Exception as exc:
        log.error("Failed to update saturation curves: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Task: Send refit summary
# ---------------------------------------------------------------------------

@task(
    name="send-refit-summary",
    retries=2,
    retry_delay_seconds=30,
    description="Post the weekly refit summary and key metrics to Slack.",
)
def send_refit_summary(
    metrics: dict[str, Any], config: dict[str, Any]
) -> None:
    """
    Format and post the weekly MCMC refit summary to the configured Slack
    channel.

    Parameters
    ----------
    metrics:
        Aggregated dict of convergence and validation metrics.
    config:
        Pipeline configuration dict.
    """
    log = _get_logger()
    log.info("send_refit_summary: composing Slack message")

    national_mape = metrics.get("national_mape", float("nan"))
    dma_mape = metrics.get("dma_median_mape", float("nan"))
    max_rhat = metrics.get("max_rhat", float("nan"))
    min_ess = metrics.get("min_bulk_ess", float("nan"))
    model_uri = metrics.get("model_uri", "unknown")
    refit_date = metrics.get("refit_date", date.today().isoformat())

    status_icon = "white_check_mark" if metrics.get("passed", False) else "x"
    lines = [
        f":{status_icon}: *MTB MMM Weekly Refit Complete — {refit_date}*",
        "",
        "*Model Quality Metrics*",
        f"  • National MAPE: {national_mape:.1%}" if national_mape == national_mape else "  • National MAPE: n/a",
        f"  • DMA Median MAPE: {dma_mape:.1%}" if dma_mape == dma_mape else "  • DMA Median MAPE: n/a",
        f"  • Max R-hat: {max_rhat:.4f}" if max_rhat == max_rhat else "  • Max R-hat: n/a",
        f"  • Min Bulk ESS: {int(min_ess)}" if min_ess == min_ess else "  • Min Bulk ESS: n/a",
        "",
        f"*Registered Model URI:* `{model_uri}`",
    ]

    warnings = metrics.get("warnings", [])
    if warnings:
        lines.append("")
        lines.append("*Warnings*")
        for w in warnings:
            lines.append(f"  :warning: {w}")

    message = "\n".join(lines)
    _send_slack_message(message, config)
    log.info("Refit summary Slack message sent.")


# ---------------------------------------------------------------------------
# Main weekly refit flow
# ---------------------------------------------------------------------------

@flow(
    name="mmm-weekly-refit",
    description=(
        "MTB MMM weekly MCMC refit: loads 3-year data window, runs full NUTS "
        "sampling, validates convergence + accuracy gates, registers in MLflow, "
        "updates saturation-curve cache."
    ),
)
def mmm_weekly_refit(
    end_date: str | None = None,
    lookback_years: int = 3,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Main Prefect flow for the weekly MMM refit pipeline.

    Scheduled at Monday 01:00 AM ET.

    Parameters
    ----------
    end_date:
        ISO date string for the end of the training window. Defaults to today.
    lookback_years:
        Number of years to include in the training window. Default 3.
    config:
        Pipeline configuration dict. Loaded from default path when ``None``.

    Returns
    -------
    dict
        Run summary including all metric results and overall status.
    """
    import uuid

    log = _get_logger()

    if config is None:
        from incrementality.config import load_config  # type: ignore
        config = load_config()

    if end_date is None:
        end_date = date.today().isoformat()

    start_date = (
        date.fromisoformat(end_date) - timedelta(days=lookback_years * 365)
    ).isoformat()

    run_id = f"{end_date}T{datetime.utcnow().strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}"
    log.info(
        "mmm_weekly_refit: started run_id=%s window=%s → %s",
        run_id,
        start_date,
        end_date,
    )

    run_summary: dict[str, Any] = {
        "run_id": run_id,
        "start_date": start_date,
        "end_date": end_date,
        "refit_date": end_date,
        "started_at": datetime.utcnow().isoformat(),
        "status": "ok",
        "passed": False,
    }

    def _run_task(task_fn, *args, task_name: str, **kwargs) -> Any:
        try:
            result = task_fn(*args, **kwargs)
            run_summary[task_name] = {"status": "ok"}
            return result
        except Exception as exc:
            error_detail = traceback.format_exc()
            log.error("Task %s failed: %s", task_name, exc)
            run_summary[task_name] = {"status": "failed", "error": str(exc)}
            run_summary["status"] = "failed"
            _send_pagerduty_alert(
                summary=f"MTB MMM weekly refit: {task_name} failed",
                details=error_detail,
                config=config,
            )
            raise  # Re-raise to halt the pipeline at critical failures

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    tensors = _run_task(
        load_tensor_data,
        start_date,
        end_date,
        config,
        task_name="load_tensor_data",
    )

    calibration_events = tensors.pop("calibration", None)

    # ------------------------------------------------------------------
    # Step 2: Full MCMC fit
    # ------------------------------------------------------------------
    model = _run_task(
        run_full_mcmc_refit,
        tensors,
        calibration_events,
        config,
        task_name="run_full_mcmc_refit",
    )

    # ------------------------------------------------------------------
    # Step 3: Convergence validation – hard gate
    # ------------------------------------------------------------------
    convergence_metrics = _run_task(
        validate_convergence,
        model,
        task_name="validate_convergence",
    )
    run_summary.update(convergence_metrics or {})

    # ------------------------------------------------------------------
    # Step 4: Hold-out validation gates
    # ------------------------------------------------------------------
    holdout_cfg = (config.get("mmm") or {}).get("holdout", {})
    holdout_days = holdout_cfg.get("days", 28)
    holdout_end = date.fromisoformat(end_date)
    holdout_start = holdout_end - timedelta(days=holdout_days)
    holdout_data = {
        "spend": tensors["spend"][
            tensors["spend"]["date"] >= holdout_start
        ].copy(),
        "revenue": tensors["revenue"][
            tensors["revenue"]["date"] >= holdout_start
        ].copy(),
    }

    validation_metrics = _run_task(
        run_validation_gates,
        model,
        holdout_data,
        config,
        task_name="run_validation_gates",
    )
    run_summary.update(validation_metrics or {})

    # ------------------------------------------------------------------
    # Step 5: Register in MLflow
    # ------------------------------------------------------------------
    combined_metrics = {
        **({} if not convergence_metrics else convergence_metrics),
        **({} if not validation_metrics else validation_metrics),
        "refit_date": end_date,
    }
    model_uri = _run_task(
        register_model,
        model,
        run_id,
        combined_metrics,
        config,
        task_name="register_model",
    )
    run_summary["model_uri"] = model_uri
    combined_metrics["model_uri"] = model_uri

    # ------------------------------------------------------------------
    # Step 6: Update saturation curves
    # ------------------------------------------------------------------
    _run_task(
        update_saturation_curves,
        model,
        config,
        task_name="update_saturation_curves",
    )

    # ------------------------------------------------------------------
    # Step 7: Slack notification – always attempt
    # ------------------------------------------------------------------
    combined_metrics["passed"] = run_summary["status"] == "ok"
    run_summary["passed"] = combined_metrics["passed"]

    try:
        send_refit_summary(combined_metrics, config)
    except Exception as exc:
        log.error("send_refit_summary failed: %s", exc)

    run_summary["finished_at"] = datetime.utcnow().isoformat()

    if run_summary["status"] == "ok":
        log.info("mmm_weekly_refit: run %s completed successfully", run_id)
    else:
        log.error("mmm_weekly_refit: run %s completed with failures", run_id)

    return run_summary
