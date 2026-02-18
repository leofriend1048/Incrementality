"""Core Meridian MMM wrapper.

This module provides ``MeridianMMM``, a high-level wrapper around the Google
Meridian library.  When the ``meridian`` package is not installed the class
falls back to a clearly-labelled numpy-based stub implementation that returns
realistic dummy data so that the rest of the platform can be developed and
tested without a full Meridian installation.

Stub mode is announced at construction time and on every call via the logger at
WARNING level so that it is never confused with real inference results.

Typical usage
-------------
>>> from incrementality.mmm.config import MeridianConfig
>>> from incrementality.mmm.model  import MeridianMMM
>>> cfg   = MeridianConfig()
>>> model = MeridianMMM(cfg)
>>> model.fit(kpi, media, spend, extra, population)
>>> contribs = model.get_channel_contributions()
"""

from __future__ import annotations

import logging
import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from incrementality.mmm.config import MeridianConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Meridian import
# ---------------------------------------------------------------------------
try:
    import meridian  # type: ignore[import-untyped]  # noqa: F401
    from meridian import data as meridian_data  # type: ignore[import-untyped]
    from meridian import model as meridian_model  # type: ignore[import-untyped]

    _MERIDIAN_AVAILABLE = True
    logger.info("Google Meridian library found; using real NUTS sampler.")
except ImportError:
    _MERIDIAN_AVAILABLE = False
    logger.warning(
        "Google Meridian library NOT found.  MeridianMMM will operate in "
        "STUB mode and return synthetic dummy data.  Install 'meridian' to "
        "enable real inference."
    )


# ---------------------------------------------------------------------------
# MeridianMMM
# ---------------------------------------------------------------------------


class MeridianMMM:
    """Wrapper around the Google Meridian MMM library.

    When the ``meridian`` package is available the class delegates all heavy
    computation to Meridian's NUTS sampler via JAX.  When it is absent the
    class operates in STUB mode and returns synthetically-generated data that
    has the same shape and column names as the real outputs.

    Parameters
    ----------
    config : MeridianConfig
        Model configuration.  All MCMC and model hyper-parameters are read
        from this object.
    """

    # Attributes set during/after fitting
    _meridian_model: Optional[Any]
    _posterior: Optional[Any]
    _fit_meta: Dict[str, Any]
    _is_fitted: bool
    _n_times: int
    _n_geos: int
    _stub_mode: bool

    def __init__(self, config: MeridianConfig) -> None:
        self.config = config
        self._meridian_model = None
        self._posterior = None
        self._fit_meta = {}
        self._is_fitted = False
        self._n_times = 0
        self._n_geos = config.n_dmas
        self._stub_mode = not _MERIDIAN_AVAILABLE

        if self._stub_mode:
            logger.warning(
                "[STUB] MeridianMMM initialised in stub mode.  "
                "All outputs are synthetic dummy data."
            )
        else:
            logger.info("MeridianMMM initialised with real Meridian backend.")

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        kpi_tensor: np.ndarray,
        media_tensor: np.ndarray,
        media_spend: np.ndarray,
        extra_features: np.ndarray,
        population: np.ndarray,
        calibration_data: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Fit the Meridian model via NUTS sampling.

        Parameters
        ----------
        kpi_tensor : np.ndarray
            Shape ``[T, G, K]`` — KPI outcomes (revenue).
        media_tensor : np.ndarray
            Shape ``[T, G, C]`` — media impressions / spend post-discount.
        media_spend : np.ndarray
            Shape ``[T, G, C]`` — raw dollar spend (used for ROI calibration
            constraints and budget optimisation; may equal *media_tensor* when
            spend = impressions).
        extra_features : np.ndarray
            Shape ``[T, G, F]`` — non-media covariates.
        population : np.ndarray
            Shape ``[G]`` — population weights summing to 1.
        calibration_data : list[dict], optional
            List of lift-test constraints in Meridian ROI calibration format.
            Each dict must contain ``lift_abs``, ``lift_lower``, ``lift_upper``,
            ``channel``, and ``outcome``.
        """
        T, G, K = kpi_tensor.shape
        self._n_times = T
        self._n_geos = G

        logger.info(
            "Fitting MMM: T=%d time-steps, G=%d geos, C=%d channels, K=%d outcomes.",
            T, G, media_tensor.shape[2], K,
        )

        if self._stub_mode:
            self._fit_stub(kpi_tensor, media_tensor, media_spend, extra_features, population)
        else:
            self._fit_meridian(
                kpi_tensor, media_tensor, media_spend,
                extra_features, population, calibration_data,
            )

        self._is_fitted = True
        logger.info("Model fitting complete (stub=%s).", self._stub_mode)

    def _fit_meridian(
        self,
        kpi_tensor: np.ndarray,
        media_tensor: np.ndarray,
        media_spend: np.ndarray,
        extra_features: np.ndarray,
        population: np.ndarray,
        calibration_data: Optional[List[Dict[str, Any]]],
    ) -> None:
        """Delegate fitting to the real Meridian library."""
        # Build Meridian InputData
        input_data = meridian_data.InputData(
            kpi=kpi_tensor,
            media=media_tensor,
            media_spend=media_spend,
            extra_features=extra_features,
            population=population,
        )

        # Build model spec from config
        model_spec = meridian_model.ModelSpec(
            n_chains=self.config.mcmc_chains,
            n_warmup=self.config.mcmc_warmup,
            n_samples=self.config.mcmc_samples,
        )

        # Optionally add calibration constraints
        if calibration_data:
            logger.info(
                "Adding %d calibration constraints to Meridian.", len(calibration_data)
            )
            for constraint in calibration_data:
                model_spec.add_roi_calibration(**constraint)

        # Instantiate and sample
        self._meridian_model = meridian_model.Meridian(
            input_data=input_data,
            model_spec=model_spec,
        )

        logger.info(
            "Starting NUTS sampling: %d chains × %d warmup + %d samples.",
            self.config.mcmc_chains,
            self.config.mcmc_warmup,
            self.config.mcmc_samples,
        )
        self._posterior = self._meridian_model.sample_posterior(
            tpu=self.config.tpu_enabled
        )
        logger.info("NUTS sampling complete.")

    def _fit_stub(
        self,
        kpi_tensor: np.ndarray,
        media_tensor: np.ndarray,
        media_spend: np.ndarray,
        extra_features: np.ndarray,
        population: np.ndarray,
    ) -> None:
        """Generate synthetic posterior metadata for stub mode."""
        logger.warning("[STUB] Skipping real MCMC; generating synthetic posterior.")
        rng = np.random.default_rng(seed=42)
        T, G, C = media_tensor.shape
        S = self.config.total_posterior_samples  # total draws

        # Store synthetic posterior as plain numpy arrays keyed by parameter name
        self._posterior = {
            "beta_media": rng.lognormal(mean=0.5, sigma=0.3, size=(S, C)),
            "roi": rng.lognormal(mean=0.7, sigma=0.25, size=(S, C)),
            "adstock_decay": rng.beta(a=2, b=5, size=(S, C)),
            "hill_alpha": rng.gamma(shape=3, scale=0.5, size=(S, C)),
            "hill_gamma": rng.gamma(shape=5, scale=0.2, size=(S, C)),
            "sigma": rng.exponential(scale=1.0, size=(S,)),
            # Time-varying betas: [S, T, C]
            "beta_tv": rng.normal(loc=0.5, scale=0.1, size=(S, T, C)),
            # Channel contributions: [S, T, G, C, K]
            "_media_contributions": rng.lognormal(
                mean=1.0, sigma=0.5, size=(S, T, G, C, self.config.n_outcomes)
            ),
        }
        self._fit_meta = {"T": T, "G": G, "C": C, "stub": True}
        logger.warning("[STUB] Synthetic posterior generated.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        media_tensor: np.ndarray,
        extra_features: np.ndarray,
    ) -> np.ndarray:
        """Generate posterior predictive samples.

        Parameters
        ----------
        media_tensor : np.ndarray
            Shape ``[T_pred, G, C]``.
        extra_features : np.ndarray
            Shape ``[T_pred, G, F]``.

        Returns
        -------
        np.ndarray
            Shape ``[S, T_pred, G, K]`` where S = total posterior samples.
        """
        self._assert_fitted()
        T_pred, G, C = media_tensor.shape
        K = self.config.n_outcomes
        S = self.config.total_posterior_samples

        if not self._stub_mode:
            logger.info("Running posterior predictive via Meridian.")
            return self._meridian_model.predict(
                media=media_tensor,
                extra_features=extra_features,
            )

        logger.warning("[STUB] Returning synthetic posterior predictive samples.")
        rng = np.random.default_rng(seed=99)
        return rng.lognormal(mean=8.0, sigma=0.4, size=(S, T_pred, G, K)).astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Diagnostics: channel contributions
    # ------------------------------------------------------------------

    def get_channel_contributions(self) -> pd.DataFrame:
        """Extract posterior channel contribution estimates.

        Returns
        -------
        pd.DataFrame
            Columns: date, dma, channel, outcome, contribution_mean,
            contribution_p10, contribution_p90.
        """
        self._assert_fitted()

        if not self._stub_mode:
            logger.info("Extracting channel contributions from Meridian posterior.")
            return self._extract_meridian_contributions()

        logger.warning("[STUB] Generating synthetic channel contributions.")
        return self._stub_contributions()

    def _extract_meridian_contributions(self) -> pd.DataFrame:
        """Extract real contributions from a fitted Meridian posterior."""
        # Meridian exposes media_contributions via the analyzer API
        analyzer = self._meridian_model.get_analyzer()
        contrib_summary = analyzer.get_media_contribution_summary()
        # contrib_summary is expected to be a DataFrame with Meridian's standard columns
        # Rename to our internal schema
        rename_map = {
            "channel": "channel",
            "geo": "dma",
            "date": "date",
            "mean": "contribution_mean",
            "p10": "contribution_p10",
            "p90": "contribution_p90",
        }
        df = contrib_summary.rename(columns=rename_map)
        # Add outcome column if missing (Meridian may return stacked outcomes)
        if "outcome" not in df.columns:
            df["outcome"] = "combined"
        return df[
            ["date", "dma", "channel", "outcome",
             "contribution_mean", "contribution_p10", "contribution_p90"]
        ]

    def _stub_contributions(self) -> pd.DataFrame:
        """Build a synthetic contributions DataFrame for stub mode."""
        rng = np.random.default_rng(seed=7)
        T = self._fit_meta.get("T", 52)
        start_date = date.today() - timedelta(weeks=T)

        rows: List[Dict[str, Any]] = []
        for t in range(T):
            d = start_date + timedelta(weeks=t)
            for dma in [f"DMA_{i:03d}" for i in range(5)]:  # abbreviated DMA list
                for channel in self.config.channels:
                    for outcome in self.config.outcomes:
                        mean_val = float(rng.lognormal(mean=3.5, sigma=0.8))
                        rows.append(
                            {
                                "date": d,
                                "dma": dma,
                                "channel": channel,
                                "outcome": outcome,
                                "contribution_mean": mean_val,
                                "contribution_p10": mean_val * 0.6,
                                "contribution_p90": mean_val * 1.5,
                            }
                        )

        df = pd.DataFrame(rows)
        logger.warning("[STUB] Contributions DataFrame has %d rows.", len(df))
        return df

    # ------------------------------------------------------------------
    # Diagnostics: R-hat convergence
    # ------------------------------------------------------------------

    def get_rhat_diagnostics(self) -> pd.DataFrame:
        """Compute R-hat (Gelman-Rubin) convergence statistics.

        Returns
        -------
        pd.DataFrame
            Columns: parameter, rhat.  Rows with rhat > config.convergence_threshold
            indicate non-convergence.
        """
        self._assert_fitted()

        if not self._stub_mode:
            logger.info("Computing R-hat diagnostics from Meridian posterior.")
            return self._extract_meridian_rhat()

        logger.warning("[STUB] Generating synthetic R-hat diagnostics.")
        return self._stub_rhat()

    def _extract_meridian_rhat(self) -> pd.DataFrame:
        """Extract real R-hat statistics from a Meridian posterior."""
        analyzer = self._meridian_model.get_analyzer()
        rhat_dict = analyzer.get_rhat()  # Expected: {param_name: float}
        rows = [{"parameter": k, "rhat": float(v)} for k, v in rhat_dict.items()]
        df = pd.DataFrame(rows).sort_values("rhat", ascending=False)
        n_bad = (df["rhat"] > self.config.convergence_threshold).sum()
        if n_bad > 0:
            logger.warning(
                "%d parameter(s) have R-hat > %.3f (non-converged).",
                n_bad, self.config.convergence_threshold,
            )
        return df

    def _stub_rhat(self) -> pd.DataFrame:
        """Generate synthetic R-hat values in [1.00, 1.04] to simulate convergence."""
        rng = np.random.default_rng(seed=11)
        param_names = (
            [f"beta_media[{ch}]" for ch in self.config.channels]
            + [f"roi[{ch}]" for ch in self.config.channels]
            + [f"adstock_decay[{ch}]" for ch in self.config.channels]
            + ["sigma", "intercept"]
        )
        rhats = 1.0 + rng.uniform(0.0, 0.04, size=len(param_names))
        df = pd.DataFrame({"parameter": param_names, "rhat": rhats})
        logger.warning("[STUB] R-hat diagnostics: %d parameters.", len(df))
        return df

    # ------------------------------------------------------------------
    # Diagnostics: time-varying betas
    # ------------------------------------------------------------------

    def get_time_varying_betas(self) -> pd.DataFrame:
        """Extract time-varying media effectiveness coefficients.

        Returns
        -------
        pd.DataFrame
            Columns: date, channel, outcome, beta_mean, beta_p10, beta_p90.
        """
        self._assert_fitted()

        if not self._stub_mode:
            logger.info("Extracting time-varying betas from Meridian posterior.")
            return self._extract_meridian_tv_betas()

        logger.warning("[STUB] Generating synthetic time-varying betas.")
        return self._stub_tv_betas()

    def _extract_meridian_tv_betas(self) -> pd.DataFrame:
        """Extract real time-varying betas from Meridian."""
        analyzer = self._meridian_model.get_analyzer()
        tv_beta_summary = analyzer.get_time_varying_beta_summary()
        rename_map = {
            "date": "date",
            "channel": "channel",
            "mean": "beta_mean",
            "p10": "beta_p10",
            "p90": "beta_p90",
        }
        df = tv_beta_summary.rename(columns=rename_map)
        if "outcome" not in df.columns:
            df["outcome"] = "combined"
        return df[["date", "channel", "outcome", "beta_mean", "beta_p10", "beta_p90"]]

    def _stub_tv_betas(self) -> pd.DataFrame:
        """Generate synthetic time-varying betas."""
        rng = np.random.default_rng(seed=13)
        T = self._fit_meta.get("T", 52)
        start_date = date.today() - timedelta(weeks=T)

        rows: List[Dict[str, Any]] = []
        for t in range(T):
            d = start_date + timedelta(weeks=t)
            for channel in self.config.channels:
                for outcome in self.config.outcomes:
                    mean_val = float(rng.lognormal(mean=0.3, sigma=0.2))
                    rows.append(
                        {
                            "date": d,
                            "channel": channel,
                            "outcome": outcome,
                            "beta_mean": mean_val,
                            "beta_p10": mean_val * 0.7,
                            "beta_p90": mean_val * 1.4,
                        }
                    )

        df = pd.DataFrame(rows)
        logger.warning("[STUB] Time-varying betas DataFrame: %d rows.", len(df))
        return df

    # ------------------------------------------------------------------
    # Saturation curves
    # ------------------------------------------------------------------

    def get_saturation_curves(
        self, n_points: int = 100
    ) -> Dict[str, np.ndarray]:
        """Compute posterior-mean saturation curves for each channel.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping channel → 2-D array of shape ``[n_points, 2]`` where
            column 0 is the spend grid and column 1 is the response.
        """
        self._assert_fitted()

        if not self._stub_mode:
            logger.info("Extracting saturation curves from Meridian posterior.")
            return self._extract_meridian_saturation_curves(n_points)

        logger.warning("[STUB] Generating synthetic saturation curves.")
        return self._stub_saturation_curves(n_points)

    def _extract_meridian_saturation_curves(
        self, n_points: int
    ) -> Dict[str, np.ndarray]:
        """Extract real saturation curves from Meridian."""
        analyzer = self._meridian_model.get_analyzer()
        curves: Dict[str, np.ndarray] = {}
        for i, channel in enumerate(self.config.channels):
            spend_grid = np.linspace(0, 1e6, n_points)
            response = analyzer.get_saturation_curve(channel_idx=i, spend_grid=spend_grid)
            curves[channel] = np.stack([spend_grid, response], axis=1)
        return curves

    def _stub_saturation_curves(self, n_points: int) -> Dict[str, np.ndarray]:
        """Generate synthetic Hill-function saturation curves."""
        rng = np.random.default_rng(seed=17)
        curves: Dict[str, np.ndarray] = {}
        spend_grid = np.linspace(0.0, 1_000_000.0, n_points)

        for channel in self.config.channels:
            alpha = rng.uniform(0.5, 2.5)
            gamma = rng.uniform(100_000.0, 600_000.0)
            x_a = spend_grid ** alpha
            g_a = gamma ** alpha
            response = x_a / (x_a + g_a)
            curves[channel] = np.stack([spend_grid, response], axis=1)

        logger.warning("[STUB] Saturation curves generated for %d channels.", len(curves))
        return curves

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_posterior(self, path: str | Path) -> None:
        """Serialise the posterior (and model metadata) to disk.

        For the real Meridian backend the posterior is saved in Meridian's
        native format.  For the stub backend a pickle file is used.

        Parameters
        ----------
        path : str or Path
            Destination file path.  The ``.pkl`` extension is added if
            ``path`` has no suffix.
        """
        self._assert_fitted()
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pkl")

        path.parent.mkdir(parents=True, exist_ok=True)

        if not self._stub_mode:
            logger.info("Saving Meridian posterior to %s.", path)
            self._meridian_model.save(str(path))
        else:
            payload = {
                "posterior": self._posterior,
                "fit_meta": self._fit_meta,
                "config": self.config.model_dump(),
                "stub_mode": True,
            }
            with open(path, "wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.warning("[STUB] Stub posterior saved to %s.", path)

        logger.info("Posterior saved to %s.", path)

    def load_posterior(self, path: str | Path) -> None:
        """Load a previously saved posterior from disk.

        Parameters
        ----------
        path : str or Path
            Path to the file created by :meth:`save_posterior`.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Posterior file not found: {path}")

        if not self._stub_mode:
            logger.info("Loading Meridian posterior from %s.", path)
            self._meridian_model = type(self._meridian_model).load(str(path))
            self._posterior = self._meridian_model.posterior
        else:
            logger.warning("[STUB] Loading stub posterior from %s.", path)
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
            self._posterior = payload.get("posterior")
            self._fit_meta = payload.get("fit_meta", {})

        self._is_fitted = True
        logger.info("Posterior loaded from %s.", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        """Raise RuntimeError if the model has not been fitted yet."""
        if not self._is_fitted:
            raise RuntimeError(
                "Model has not been fitted.  Call .fit() before accessing results."
            )

    @property
    def is_stub(self) -> bool:
        """True when operating in stub mode (Meridian not installed)."""
        return self._stub_mode

    @property
    def is_fitted(self) -> bool:
        """True after :meth:`fit` or :meth:`load_posterior` has been called."""
        return self._is_fitted

    def __repr__(self) -> str:
        backend = "stub" if self._stub_mode else "meridian"
        fitted = "fitted" if self._is_fitted else "unfitted"
        return (
            f"MeridianMMM(backend={backend!r}, status={fitted!r}, "
            f"channels={len(self.config.channels)}, "
            f"outcomes={len(self.config.outcomes)})"
        )
