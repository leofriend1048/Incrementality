"""MLflow model registry for Michael Todd Beauty MMM.

Tracks model artifacts, metrics, and run history across Meridian re-fits.
Provides a graceful fallback to a local JSON file when MLflow is unavailable.

Typical usage
-------------
>>> from incrementality.mmm.registry import ModelRegistry
>>> registry = ModelRegistry(tracking_uri="http://mlflow.internal:5000")
>>> with registry.start_run("weekly-refit-2024-W12"):
...     registry.log_model_artifacts(model, config, metrics)
...     registry.log_refit_metrics(rhat_df, 0.082, 0.094, 210, 365)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional MLflow import
# ---------------------------------------------------------------------------
try:
    import mlflow  # type: ignore
    import mlflow.pyfunc  # type: ignore

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    logger.warning(
        "mlflow is not installed.  ModelRegistry will persist runs to a "
        "local JSON file.  Install mlflow for full experiment tracking: "
        "pip install mlflow"
    )

# ---------------------------------------------------------------------------
# JSON fallback path
# ---------------------------------------------------------------------------
_FALLBACK_DIR = Path(os.environ.get("MTB_REGISTRY_DIR", "~/.mtb_registry")).expanduser()
_FALLBACK_FILE = _FALLBACK_DIR / "runs.json"


def _load_fallback_db() -> List[Dict[str, Any]]:
    _FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
    if _FALLBACK_FILE.exists():
        with open(_FALLBACK_FILE) as f:
            return json.load(f)
    return []


def _save_fallback_db(runs: List[Dict[str, Any]]) -> None:
    _FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(_FALLBACK_FILE, "w") as f:
        json.dump(runs, f, indent=2, default=str)


class _FallbackRun:
    """Lightweight in-memory run object when MLflow is absent."""

    def __init__(self, run_name: str) -> None:
        self.run_id = f"local-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        self.run_name = run_name
        self.metrics: Dict[str, float] = {}
        self.params: Dict[str, str] = {}
        self.tags: Dict[str, str] = {}
        self.artifacts: List[str] = []
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.end_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "metrics": self.metrics,
            "params": self.params,
            "tags": self.tags,
            "artifacts": self.artifacts,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Tracks Meridian MMM runs and artifacts via MLflow or a local JSON store.

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking server URI (e.g. ``http://mlflow.internal:5000``).
        Ignored when MLflow is not installed.
    experiment_name : str
        MLflow experiment name.  Created if it does not exist.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str = "mtb-meridian-mmm",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._active_run: Optional[_FallbackRun] = None

        if _MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(tracking_uri)
            try:
                self._experiment_id = mlflow.create_experiment(experiment_name)
            except mlflow.exceptions.MlflowException:
                # Already exists
                exp = mlflow.get_experiment_by_name(experiment_name)
                self._experiment_id = exp.experiment_id if exp else None
            logger.info(
                "MLflow ModelRegistry ready: uri=%s, experiment=%s (id=%s)",
                tracking_uri,
                experiment_name,
                self._experiment_id,
            )
        else:
            self._experiment_id = None
            logger.info(
                "ModelRegistry using local JSON fallback: %s", _FALLBACK_FILE
            )

    # ------------------------------------------------------------------
    # Context-manager run lifecycle
    # ------------------------------------------------------------------

    @contextmanager
    def start_run(self, run_name: str) -> Generator[None, None, None]:
        """Context manager that opens (and closes) a tracking run.

        Usage::

            with registry.start_run("weekly-refit-2024-W12"):
                registry.log_refit_metrics(...)
        """
        if _MLFLOW_AVAILABLE:
            with mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name=run_name,
            ) as run:
                self._active_mlflow_run = run
                logger.info("MLflow run started: %s (%s)", run_name, run.info.run_id)
                try:
                    yield
                finally:
                    self._active_mlflow_run = None
                    logger.info("MLflow run ended: %s", run_name)
        else:
            fallback = _FallbackRun(run_name)
            self._active_run = fallback
            logger.info("Fallback run started: %s (%s)", run_name, fallback.run_id)
            try:
                yield
            finally:
                fallback.end_time = datetime.now(timezone.utc).isoformat()
                runs = _load_fallback_db()
                runs.append(fallback.to_dict())
                _save_fallback_db(runs)
                self._active_run = None
                logger.info(
                    "Fallback run persisted: %s → %s", run_name, _FALLBACK_FILE
                )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_metric(self, key: str, value: float) -> None:
        if _MLFLOW_AVAILABLE:
            mlflow.log_metric(key, value)
        elif self._active_run is not None:
            self._active_run.metrics[key] = value

    def _log_param(self, key: str, value: Any) -> None:
        str_value = str(value)
        if _MLFLOW_AVAILABLE:
            mlflow.log_param(key, str_value)
        elif self._active_run is not None:
            self._active_run.params[key] = str_value

    def _log_tag(self, key: str, value: str) -> None:
        if _MLFLOW_AVAILABLE:
            mlflow.set_tag(key, value)
        elif self._active_run is not None:
            self._active_run.tags[key] = value

    def _log_artifact(self, local_path: str) -> None:
        if _MLFLOW_AVAILABLE:
            mlflow.log_artifact(local_path)
        elif self._active_run is not None:
            self._active_run.artifacts.append(local_path)

    # ------------------------------------------------------------------
    # Public logging API
    # ------------------------------------------------------------------

    def log_model_artifacts(
        self,
        model: Any,
        config: Any,
        metrics: Dict[str, float],
    ) -> None:
        """Persist model posterior, configuration, and evaluation metrics.

        Parameters
        ----------
        model : any
            Fitted Meridian model object (serialised via pickle).
        config : any
            MeridianConfig or dict with model hyper-parameters.
        metrics : dict
            Evaluation metrics to log (e.g. MAPE, R-hat max).
        """
        # Log scalar metrics
        for k, v in metrics.items():
            self._log_metric(k, float(v))

        # Log config params
        if hasattr(config, "model_dump"):
            cfg_dict = config.model_dump()
        elif isinstance(config, dict):
            cfg_dict = config
        else:
            cfg_dict = vars(config)

        for k, v in cfg_dict.items():
            if not isinstance(v, (dict, list)):
                self._log_param(k, v)

        # Persist model + config as artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            config_path = Path(tmpdir) / "config.json"

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            with open(config_path, "w") as f:
                json.dump(cfg_dict, f, indent=2, default=str)

            self._log_artifact(str(model_path))
            self._log_artifact(str(config_path))

        self._log_tag("artifact_type", "meridian_mmm")
        self._log_tag("logged_at", datetime.now(timezone.utc).isoformat())
        logger.info("Model artifacts logged: %d metrics, config persisted.", len(metrics))

    def log_refit_metrics(
        self,
        rhat_df: pd.DataFrame,
        mape_shopify: float,
        mape_amazon: float,
        n_dmas: int,
        n_days: int,
    ) -> None:
        """Log the standard set of Meridian re-fit quality metrics.

        Parameters
        ----------
        rhat_df : pd.DataFrame
            DataFrame with columns ``["parameter", "rhat"]`` from MCMC diagnostics.
        mape_shopify : float
            Mean absolute percentage error on the Shopify outcome.
        mape_amazon : float
            Mean absolute percentage error on the Amazon outcome.
        n_dmas : int
            Number of DMAs in the fitted model.
        n_days : int
            Number of days / time periods in the fitted model.
        """
        # R-hat summary statistics
        rhat_col = "rhat" if "rhat" in rhat_df.columns else rhat_df.columns[-1]
        rhat_max = float(rhat_df[rhat_col].max())
        rhat_mean = float(rhat_df[rhat_col].mean())
        n_diverged = int((rhat_df[rhat_col] > 1.05).sum())

        self._log_metric("rhat_max", rhat_max)
        self._log_metric("rhat_mean", rhat_mean)
        self._log_metric("rhat_n_diverged", float(n_diverged))
        self._log_metric("mape_shopify", mape_shopify)
        self._log_metric("mape_amazon", mape_amazon)
        self._log_metric("n_dmas", float(n_dmas))
        self._log_metric("n_days", float(n_days))
        self._log_metric(
            "convergence_passed", 1.0 if rhat_max < 1.05 else 0.0
        )

        # Persist full R-hat table
        with tempfile.TemporaryDirectory() as tmpdir:
            rhat_path = Path(tmpdir) / "rhat_diagnostics.csv"
            rhat_df.to_csv(rhat_path, index=False)
            self._log_artifact(str(rhat_path))

        logger.info(
            "Refit metrics logged: rhat_max=%.4f, mape_shopify=%.4f, "
            "mape_amazon=%.4f, n_dmas=%d, n_days=%d",
            rhat_max, mape_shopify, mape_amazon, n_dmas, n_days,
        )

    def log_validation_results(self, validation_dict: Dict[str, Any]) -> None:
        """Log validation gate results (Gates 1 / 2 / 3).

        Parameters
        ----------
        validation_dict : dict
            Output from MMMValidator containing gate pass/fail flags and
            per-gate metric dictionaries.
        """
        # Flatten gate metrics into scalar MLflow metrics
        for gate_key in ("gate_1", "gate_2", "gate_3"):
            gate = validation_dict.get(gate_key, {})
            if isinstance(gate, dict):
                passed = gate.get("passed", False)
                self._log_metric(
                    f"validation_{gate_key}_passed", 1.0 if passed else 0.0
                )
                for metric_key, metric_val in gate.get("metrics", {}).items():
                    if isinstance(metric_val, (int, float)):
                        self._log_metric(
                            f"validation_{gate_key}_{metric_key}", float(metric_val)
                        )

        overall_passed = validation_dict.get("all_gates_passed", False)
        self._log_metric("validation_all_gates_passed", 1.0 if overall_passed else 0.0)

        # Persist the full dict as a JSON artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            val_path = Path(tmpdir) / "validation_results.json"
            with open(val_path, "w") as f:
                json.dump(validation_dict, f, indent=2, default=str)
            self._log_artifact(str(val_path))

        logger.info(
            "Validation results logged: all_gates_passed=%s", overall_passed
        )

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_latest_run(self) -> Tuple[Optional[str], Dict[str, float]]:
        """Return the run_id and metrics dict of the most recent run.

        Returns
        -------
        tuple
            ``(run_id, metrics_dict)`` where ``run_id`` is None if no
            runs exist.
        """
        if _MLFLOW_AVAILABLE:
            runs = mlflow.search_runs(
                experiment_ids=[self._experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
                output_format="pandas",
            )
            if runs.empty:
                return None, {}
            row = runs.iloc[0]
            run_id = row["run_id"]
            metrics = {
                k.replace("metrics.", ""): v
                for k, v in row.items()
                if k.startswith("metrics.") and pd.notna(v)
            }
            return run_id, metrics
        else:
            runs = _load_fallback_db()
            if not runs:
                return None, {}
            latest = sorted(runs, key=lambda r: r.get("start_time", ""), reverse=True)[0]
            return latest["run_id"], latest.get("metrics", {})

    def get_run_history(self, n_runs: int = 10) -> pd.DataFrame:
        """Return a DataFrame of the last *n_runs* runs with their metrics.

        Parameters
        ----------
        n_runs : int
            Maximum number of runs to return.

        Returns
        -------
        pd.DataFrame
            Columns: run_id, run_name, start_time, and all logged metrics.
        """
        if _MLFLOW_AVAILABLE:
            df = mlflow.search_runs(
                experiment_ids=[self._experiment_id],
                order_by=["start_time DESC"],
                max_results=n_runs,
                output_format="pandas",
            )
            if df.empty:
                return pd.DataFrame()
            df.columns = [c.replace("metrics.", "").replace("tags.", "") for c in df.columns]
            return df.reset_index(drop=True)
        else:
            runs = _load_fallback_db()
            runs_sorted = sorted(
                runs, key=lambda r: r.get("start_time", ""), reverse=True
            )[:n_runs]
            records = []
            for r in runs_sorted:
                row: Dict[str, Any] = {
                    "run_id": r["run_id"],
                    "run_name": r["run_name"],
                    "start_time": r.get("start_time"),
                }
                row.update(r.get("metrics", {}))
                records.append(row)
            return pd.DataFrame(records)

    def compare_runs(self, run_id_a: str, run_id_b: str) -> pd.DataFrame:
        """Return a side-by-side comparison of key metrics for two runs.

        Parameters
        ----------
        run_id_a : str
            First run ID.
        run_id_b : str
            Second run ID.

        Returns
        -------
        pd.DataFrame
            Index: metric name.  Columns: run_id_a, run_id_b, delta,
            delta_pct, improved.
        """
        _KEY_METRICS = [
            "mape_shopify",
            "mape_amazon",
            "rhat_max",
            "rhat_mean",
            "rhat_n_diverged",
            "convergence_passed",
            "validation_gate_1_passed",
            "validation_gate_2_passed",
            "validation_gate_3_passed",
            "validation_all_gates_passed",
        ]

        def _get_metrics_for_run(run_id: str) -> Dict[str, float]:
            if _MLFLOW_AVAILABLE:
                client = mlflow.tracking.MlflowClient()
                run = client.get_run(run_id)
                return {k: v for k, v in run.data.metrics.items()}
            else:
                runs = _load_fallback_db()
                for r in runs:
                    if r["run_id"] == run_id:
                        return r.get("metrics", {})
                return {}

        metrics_a = _get_metrics_for_run(run_id_a)
        metrics_b = _get_metrics_for_run(run_id_b)

        all_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
        # Put key metrics first
        ordered_keys = [k for k in _KEY_METRICS if k in all_keys] + [
            k for k in all_keys if k not in _KEY_METRICS
        ]

        records = []
        for k in ordered_keys:
            v_a = metrics_a.get(k, float("nan"))
            v_b = metrics_b.get(k, float("nan"))
            try:
                delta = v_b - v_a
                delta_pct = (delta / abs(v_a) * 100) if abs(v_a) > 1e-9 else float("nan")
            except TypeError:
                delta = float("nan")
                delta_pct = float("nan")
            # "Improved" depends on metric direction (lower is better for MAPE/rhat)
            lower_is_better = any(
                kw in k for kw in ("mape", "rhat", "diverged")
            )
            if not np.isnan(delta):
                improved = delta < 0 if lower_is_better else delta > 0
            else:
                improved = None
            records.append(
                {
                    "metric": k,
                    run_id_a: v_a,
                    run_id_b: v_b,
                    "delta": delta,
                    "delta_pct": delta_pct,
                    "improved": improved,
                }
            )

        return pd.DataFrame(records).set_index("metric")
