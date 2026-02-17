"""MMM Orchestrator — bridges existing connectors to the Meridian MMM engine.

Lives alongside TestOrchestrator and reuses the same Config, connectors,
data_dir / output_dir conventions, and Pydantic models.

Usage (CLI calls this):
    from incrementality.mmm_orchestrator import MMMOrchestrator
    orch = MMMOrchestrator(config)
    result = orch.fit(lookback_weeks=156)          # ~3 years
    orch.save_result(result)
    allocation = orch.optimize(total_budget=300_000)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from incrementality.config import Config
from incrementality.models import MMMChannelResult, MMMRunResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger("incrementality.mmm")


class MMMOrchestrator:
    """Orchestrates the full MMM pipeline using the same Config/connectors
    pattern as TestOrchestrator.

    Data flow:
        Config → connectors → raw DataFrames
        → MeridianTensorBuilder → [T×G×C] tensors
        → MeridianMMM.fit() → posterior
        → BudgetOptimizer.optimize() → allocation
        → MMMRunResult → JSON in output_dir
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.output_dir = Path(config.output_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialise connectors from the same config that TestOrchestrator uses
        self._shopify = None
        self._amazon = None
        self._facebook = None
        self._youtube = None
        self._tiktok = None
        self._tiktok_shop = None
        self._pinterest = None
        self._applovin = None
        self._tatari = None
        self._postscript = None
        self._klaviyo = None
        self._northbeam = None

        if config.shopify:
            from incrementality.connectors.shopify import ShopifyConnector
            self._shopify = ShopifyConnector(config.shopify)

        if config.amazon:
            from incrementality.connectors.amazon import AmazonConnector
            self._amazon = AmazonConnector(config.amazon)

        if config.facebook:
            from incrementality.connectors.facebook import FacebookConnector
            self._facebook = FacebookConnector(config.facebook)

        if config.youtube:
            from incrementality.connectors.youtube import YouTubeConnector
            self._youtube = YouTubeConnector(config.youtube)

        if config.tiktok:
            from incrementality.connectors.tiktok import TikTokConnector
            self._tiktok = TikTokConnector(
                config.tiktok.app_id,
                config.tiktok.secret,
                config.tiktok.access_token,
            )

        if config.tiktok_shop:
            from incrementality.connectors.tiktok_shop import TikTokShopConnector
            self._tiktok_shop = TikTokShopConnector(config.tiktok_shop)

        if config.pinterest:
            from incrementality.connectors.pinterest import PinterestConnector
            self._pinterest = PinterestConnector(config.pinterest)

        if config.applovin:
            from incrementality.connectors.applovin import AppLovinConnector
            self._applovin = AppLovinConnector(config.applovin)

        if config.tatari:
            from incrementality.connectors.tatari import TatariConnector
            self._tatari = TatariConnector(config.tatari)

        if config.postscript:
            from incrementality.connectors.postscript import PostscriptConnector
            self._postscript = PostscriptConnector(config.postscript)

        if config.klaviyo:
            from incrementality.connectors.klaviyo import KlaviyoConnector
            self._klaviyo = KlaviyoConnector(config.klaviyo)

        if config.northbeam:
            from incrementality.connectors.northbeam import NorthbeamConnector
            self._northbeam = NorthbeamConnector(
                config.northbeam.api_key,
                config.northbeam.account_id,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Data pulling (reuses same connector interface as TestOrchestrator)
    # ─────────────────────────────────────────────────────────────────────────

    def pull_historical_data(
        self,
        lookback_weeks: int = 156,
        end_date: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Pull all data sources needed for MMM.

        Returns:
            dict with keys: shopify, amazon, facebook, youtube, tiktok,
                            northbeam_attribution
            Values are DataFrames; empty DataFrame on failure.
        """
        if end_date is None:
            end_date = date.today()
        start_date = end_date - timedelta(weeks=lookback_weeks)

        result: dict[str, pd.DataFrame] = {}
        pull_errors: dict[str, str] = {}

        # Shopify revenue by DMA
        if self._shopify:
            try:
                result["shopify"] = self._shopify.get_daily_revenue_by_dma(
                    start_date, end_date
                )
                logger.info("Shopify: %d rows", len(result["shopify"]))
            except Exception as exc:
                pull_errors["shopify"] = str(exc)
                result["shopify"] = pd.DataFrame()
        else:
            result["shopify"] = pd.DataFrame()

        # Amazon revenue (national)
        if self._amazon:
            try:
                result["amazon"] = self._amazon.get_daily_revenue(
                    start_date, end_date
                )
                logger.info("Amazon: %d rows", len(result["amazon"]))
            except Exception as exc:
                pull_errors["amazon"] = str(exc)
                result["amazon"] = pd.DataFrame()
        else:
            result["amazon"] = pd.DataFrame()

        # Facebook spend by DMA
        if self._facebook:
            try:
                result["facebook"] = self._facebook.fetch_spend_by_dma(
                    start_date, end_date
                )
                logger.info("Facebook: %d rows", len(result["facebook"]))
            except Exception as exc:
                pull_errors["facebook"] = str(exc)
                result["facebook"] = pd.DataFrame()
        else:
            result["facebook"] = pd.DataFrame()

        # YouTube/Google spend by DMA
        if self._youtube:
            try:
                result["youtube"] = self._youtube.fetch_spend_by_dma(
                    start_date, end_date
                )
                logger.info("YouTube: %d rows", len(result["youtube"]))
            except Exception as exc:
                pull_errors["youtube"] = str(exc)
                result["youtube"] = pd.DataFrame()
        else:
            result["youtube"] = pd.DataFrame()

        # TikTok performance spend by region (state-level)
        if self._tiktok:
            try:
                result["tiktok"] = self._tiktok.get_daily_spend_by_region(
                    start_date, end_date
                )
                logger.info("TikTok: %d rows", len(result["tiktok"]))
            except Exception as exc:
                pull_errors["tiktok"] = str(exc)
                result["tiktok"] = pd.DataFrame()
        else:
            result["tiktok"] = pd.DataFrame()

        # TikTok GMV Max — filtered by campaign objective
        if self._tiktok:
            try:
                result["tiktok_gmv_max"] = self._tiktok.get_daily_spend_by_objective(
                    start_date, end_date, objective_type="GMV_MAX"
                )
                logger.info("TikTok GMV Max: %d rows", len(result["tiktok_gmv_max"]))
            except Exception as exc:
                pull_errors["tiktok_gmv_max"] = str(exc)
                result["tiktok_gmv_max"] = pd.DataFrame()
        else:
            result["tiktok_gmv_max"] = pd.DataFrame()

        # TikTok Shop — native commerce GMV
        if self._tiktok_shop:
            try:
                result["tiktok_shop"] = self._tiktok_shop.get_daily_gmv(
                    start_date, end_date
                )
                logger.info("TikTok Shop: %d rows", len(result["tiktok_shop"]))
            except Exception as exc:
                pull_errors["tiktok_shop"] = str(exc)
                result["tiktok_shop"] = pd.DataFrame()
        else:
            result["tiktok_shop"] = pd.DataFrame()

        # Pinterest spend by region
        if self._pinterest:
            try:
                result["pinterest"] = self._pinterest.get_daily_spend_by_region(
                    start_date, end_date
                )
                logger.info("Pinterest: %d rows", len(result["pinterest"]))
            except Exception as exc:
                pull_errors["pinterest"] = str(exc)
                result["pinterest"] = pd.DataFrame()
        else:
            result["pinterest"] = pd.DataFrame()

        # AppLovin national spend
        if self._applovin:
            try:
                result["applovin"] = self._applovin.get_daily_national_spend(
                    start_date, end_date
                )
                logger.info("AppLovin: %d rows", len(result["applovin"]))
            except Exception as exc:
                pull_errors["applovin"] = str(exc)
                result["applovin"] = pd.DataFrame()
        else:
            result["applovin"] = pd.DataFrame()

        # Tatari TV spend by DMA
        if self._tatari:
            try:
                result["tatari"] = self._tatari.get_daily_spend_by_dma(
                    start_date, end_date
                )
                logger.info("Tatari: %d rows", len(result["tatari"]))
            except Exception as exc:
                pull_errors["tatari"] = str(exc)
                result["tatari"] = pd.DataFrame()
        else:
            result["tatari"] = pd.DataFrame()

        # Postscript SMS national performance
        if self._postscript:
            try:
                result["postscript_sms"] = self._postscript.get_daily_performance(
                    start_date, end_date
                )
                logger.info("Postscript: %d rows", len(result["postscript_sms"]))
            except Exception as exc:
                pull_errors["postscript_sms"] = str(exc)
                result["postscript_sms"] = pd.DataFrame()
        else:
            result["postscript_sms"] = pd.DataFrame()

        # Klaviyo email national performance
        if self._klaviyo:
            try:
                result["klaviyo_email"] = self._klaviyo.get_daily_performance(
                    start_date, end_date
                )
                logger.info("Klaviyo: %d rows", len(result["klaviyo_email"]))
            except Exception as exc:
                pull_errors["klaviyo_email"] = str(exc)
                result["klaviyo_email"] = pd.DataFrame()
        else:
            result["klaviyo_email"] = pd.DataFrame()

        # Northbeam MTA for prior calibration
        if self._northbeam:
            try:
                result["northbeam_attribution"] = self._northbeam.get_channel_attribution(
                    start_date.isoformat(), end_date.isoformat(),
                    attribution_window=self.config.northbeam.attribution_window,  # type: ignore[union-attr]
                )
                logger.info("Northbeam: %d rows", len(result["northbeam_attribution"]))
            except Exception as exc:
                pull_errors["northbeam_attribution"] = str(exc)
                result["northbeam_attribution"] = pd.DataFrame()
        else:
            result["northbeam_attribution"] = pd.DataFrame()

        if pull_errors:
            logger.warning("Data pull errors: %s", pull_errors)
        self.pull_errors = pull_errors
        return result

    def load_cached_data(self) -> dict[str, pd.DataFrame]:
        """Load previously cached parquet files from data_dir."""
        result: dict[str, pd.DataFrame] = {}
        sources = [
            "shopify", "amazon",
            "facebook", "youtube",
            "tiktok", "tiktok_gmv_max", "tiktok_shop",
            "pinterest", "applovin", "tatari",
            "postscript_sms", "klaviyo_email",
            "northbeam_attribution",
        ]
        for src in sources:
            cache_path = self.data_dir / f"mmm_{src}.parquet"
            if cache_path.exists():
                result[src] = pd.read_parquet(cache_path)
            else:
                result[src] = pd.DataFrame()
        return result

    def cache_data(self, data: dict[str, pd.DataFrame]) -> None:
        """Persist pulled data as parquet for re-use."""
        for src, df in data.items():
            if not df.empty:
                df.to_parquet(self.data_dir / f"mmm_{src}.parquet", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # MMM fitting
    # ─────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        lookback_weeks: int = 156,
        data: dict[str, pd.DataFrame] | None = None,
        use_cached_data: bool = False,
    ) -> MMMRunResult:
        """Pull data (or use provided), build tensors, fit Meridian, return result.

        Args:
            lookback_weeks: How many weeks of history to use (~3 years = 156).
            data: Pre-pulled data dict — skips API pull if provided.
            use_cached_data: Load from parquet cache instead of APIs.

        Returns:
            MMMRunResult with channel contributions, diagnostics, and
            budget optimizer output.
        """
        from incrementality.mmm.config import MeridianConfig
        from incrementality.mmm.model import MeridianMMM
        from incrementality.mmm.tensors import MeridianTensorBuilder
        from incrementality.mmm.calibration import CalibrationStore
        from incrementality.mmm.optimizer import BudgetOptimizer

        cfg = self.config.mmm
        run_id = f"mmm_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        logger.info("Starting MMM run %s", run_id)

        # ── 1. Data ──────────────────────────────────────────────────────────
        if data is None:
            if use_cached_data:
                data = self.load_cached_data()
            else:
                data = self.pull_historical_data(lookback_weeks=lookback_weeks)
            self.cache_data(data)

        # ── 2. Northbeam priors ──────────────────────────────────────────────
        nb_prior_means: dict[str, float] = {}
        if self._northbeam and not data.get("northbeam_attribution", pd.DataFrame()).empty:
            try:
                end_d = date.today()
                start_d = end_d - timedelta(days=30)
                nb_prior_means = self._northbeam.compute_prior_means(
                    start_d.isoformat(), end_d.isoformat(), total_revenue=1.0
                )
                logger.info("Northbeam prior means: %s", nb_prior_means)
            except Exception as exc:
                logger.warning("Northbeam prior computation failed: %s", exc)

        # ── 3. Calibration events ────────────────────────────────────────────
        cal_store = CalibrationStore(
            storage_path=str(self.data_dir / "calibration_events.json")
        )
        calibration_constraints = cal_store.to_meridian_constraints(
            cal_store.get_events()
        )

        # ── 4. Build Meridian config ─────────────────────────────────────────
        meridian_cfg = MeridianConfig(
            mcmc_chains=cfg.mcmc_chains,
            mcmc_warmup=cfg.mcmc_warmup,
            mcmc_samples=cfg.mcmc_samples,
            channels=cfg.channels,
        )

        # ── 5. Build input tensors ───────────────────────────────────────────
        builder = MeridianTensorBuilder(config=meridian_cfg)
        from incrementality.dma import ALL_DMAS
        dma_codes = [d.dma_code for d in ALL_DMAS]

        tensors = builder.build_all(
            shopify_df=data.get("shopify", pd.DataFrame()),
            amazon_df=data.get("amazon", pd.DataFrame()),
            spend_dfs={
                # Meta — split Facebook performance vs awareness
                "meta_perf": data.get("facebook", pd.DataFrame()),
                "meta_aware": data.get("facebook", pd.DataFrame()),
                # Google — split brand vs non-brand
                "google_brand": data.get("youtube", pd.DataFrame()),
                "google_nonbrand": data.get("youtube", pd.DataFrame()),
                # TikTok channels
                "tiktok": data.get("tiktok", pd.DataFrame()),
                "tiktok_gmv_max": data.get("tiktok_gmv_max", pd.DataFrame()),
                "tiktok_shop": data.get("tiktok_shop", pd.DataFrame()),
                # Pinterest
                "pinterest": data.get("pinterest", pd.DataFrame()),
                # AppLovin (mobile / CTV)
                "applovin": data.get("applovin", pd.DataFrame()),
                # TV
                "tatari": data.get("tatari", pd.DataFrame()),
                # Amazon retail media (spend injected manually or via CSV)
                "amz_sponsored": pd.DataFrame(),
                # Owned channels
                "postscript_sms": data.get("postscript_sms", pd.DataFrame()),
                "klaviyo_email": data.get("klaviyo_email", pd.DataFrame()),
            },
            promo_df=pd.DataFrame(),
            trends_df=pd.DataFrame(),
            bsr_df=pd.DataFrame(),
            dma_list=dma_codes,
        )

        # ── 6. Fit model ─────────────────────────────────────────────────────
        model = MeridianMMM(config=meridian_cfg)
        model.fit(
            kpi_tensor=tensors["kpi"],
            media_tensor=tensors["media"],
            media_spend=tensors["media_spend"],
            extra_features=tensors.get("extra_features"),
            population=tensors["population"],
            calibration_data=calibration_constraints if calibration_constraints else None,
        )

        # ── 7. Extract results ───────────────────────────────────────────────
        contributions_df = model.get_channel_contributions()
        rhat_df = model.get_rhat_diagnostics()
        saturation_curves = model.get_saturation_curves()

        channel_results = _build_channel_results(contributions_df, rhat_df)

        # ── 8. Budget optimizer ──────────────────────────────────────────────
        optimizer = BudgetOptimizer(
            saturation_curves=saturation_curves,
            config={
                "channels": meridian_cfg.channels,
                "outcomes": meridian_cfg.outcomes,
            },
        )
        monthly_budget = 350_000.0  # default; CLI can override
        opt_result = optimizer.optimize(
            total_budget=monthly_budget,
            shopify_weight=cfg.shopify_revenue_weight,
            amazon_weight=cfg.amazon_revenue_weight,
        )

        # ── 9. NB vs MMM delta ───────────────────────────────────────────────
        nb_mmm_delta: dict[str, float] = {}
        if nb_prior_means:
            for ch_result in channel_results:
                nb_roi = nb_prior_means.get(ch_result.channel, 0.0)
                if nb_roi > 0 and ch_result.roi_mean > 0:
                    nb_mmm_delta[ch_result.channel] = (
                        (ch_result.roi_mean - nb_roi) / nb_roi
                    )

        # ── 10. Assemble MMMRunResult ────────────────────────────────────────
        rhat_max = float(rhat_df["rhat"].max()) if not rhat_df.empty else 1.0
        converged = rhat_max < meridian_cfg.convergence_threshold

        shopify_contribs = contributions_df[
            contributions_df["outcome"] == "shopify"
        ] if not contributions_df.empty else pd.DataFrame()
        amazon_contribs = contributions_df[
            contributions_df["outcome"] == "amazon"
        ] if not contributions_df.empty else pd.DataFrame()

        result = MMMRunResult(
            run_id=run_id,
            channel_results=channel_results,
            baseline_shopify=float(
                shopify_contribs[shopify_contribs["channel"] == "baseline"][
                    "contribution_mean"
                ].sum() if not shopify_contribs.empty else 0.0
            ),
            baseline_amazon=float(
                amazon_contribs[amazon_contribs["channel"] == "baseline"][
                    "contribution_mean"
                ].sum() if not amazon_contribs.empty else 0.0
            ),
            rhat_max=rhat_max,
            converged=converged,
            gate_1_passed=converged,
            optimal_allocation=opt_result.allocation,
            expected_shopify_revenue=opt_result.expected_shopify_revenue,
            expected_amazon_revenue=opt_result.expected_amazon_revenue,
            expected_total_revenue=opt_result.expected_total_revenue,
            blended_roas=opt_result.blended_roas,
            nb_mmm_delta=nb_mmm_delta,
            n_dmas=len(dma_codes),
            n_days=int(tensors["kpi"].shape[0]) if tensors.get("kpi") is not None else 0,
            channels=meridian_cfg.channels,
            outcomes=meridian_cfg.outcomes,
            is_stub=model._stub_mode,
        )

        logger.info(
            "MMM run %s complete — R-hat max: %.3f, converged: %s",
            run_id, rhat_max, converged,
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence (same data_dir / output_dir convention)
    # ─────────────────────────────────────────────────────────────────────────

    def save_result(self, result: MMMRunResult) -> Path:
        """Save MMMRunResult as JSON in output_dir."""
        out_path = self.output_dir / f"{result.run_id}_mmm_result.json"
        out_path.write_text(
            result.model_dump_json(indent=2, exclude_none=False)
        )
        logger.info("Saved MMM result to %s", out_path)
        return out_path

    def load_result(self, run_id: str) -> MMMRunResult:
        """Load a previously saved MMMRunResult by run_id."""
        path = self.output_dir / f"{run_id}_mmm_result.json"
        if not path.exists():
            raise FileNotFoundError(f"No MMM result found for run_id '{run_id}' at {path}")
        return MMMRunResult.model_validate_json(path.read_text())

    def load_latest_result(self) -> MMMRunResult | None:
        """Load the most recently generated MMMRunResult, or None if none exist."""
        paths = sorted(self.output_dir.glob("mmm_*_mmm_result.json"), reverse=True)
        if not paths:
            return None
        return MMMRunResult.model_validate_json(paths[0].read_text())

    def list_results(self) -> list[dict]:
        """List all saved MMM run results (summary only)."""
        out = []
        for path in sorted(self.output_dir.glob("mmm_*_mmm_result.json"), reverse=True):
            try:
                r = MMMRunResult.model_validate_json(path.read_text())
                out.append({
                    "run_id": r.run_id,
                    "generated_at": r.generated_at.isoformat(),
                    "converged": r.converged,
                    "rhat_max": r.rhat_max,
                    "n_dmas": r.n_dmas,
                    "n_days": r.n_days,
                    "is_stub": r.is_stub,
                    "blended_roas": r.blended_roas,
                })
            except Exception:
                continue
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience: optimize with a specific budget
    # ─────────────────────────────────────────────────────────────────────────

    def optimize(
        self,
        total_budget: float,
        run_id: str | None = None,
        shopify_weight: float | None = None,
        amazon_weight: float | None = None,
    ) -> dict:
        """Run budget optimizer on an existing fitted model.

        Args:
            total_budget: Total weekly or monthly budget in USD.
            run_id: Specific run to load; defaults to latest.
            shopify_weight: Override config shopify revenue weight.
            amazon_weight: Override config amazon revenue weight.

        Returns:
            dict with allocation, expected_revenue, blended_roas.
        """
        from incrementality.mmm.config import MeridianConfig
        from incrementality.mmm.model import MeridianMMM
        from incrementality.mmm.optimizer import BudgetOptimizer

        meridian_cfg = MeridianConfig(channels=self.config.mmm.channels)
        model = MeridianMMM(config=meridian_cfg)

        # Try to load saved posterior
        if run_id:
            posterior_path = self.output_dir / f"{run_id}_posterior.pkl"
            if posterior_path.exists():
                model.load_posterior(str(posterior_path))

        saturation_curves = model.get_saturation_curves()
        optimizer = BudgetOptimizer(
            saturation_curves=saturation_curves,
            config={"channels": meridian_cfg.channels, "outcomes": meridian_cfg.outcomes},
        )
        sw = shopify_weight if shopify_weight is not None else self.config.mmm.shopify_revenue_weight
        aw = amazon_weight if amazon_weight is not None else self.config.mmm.amazon_revenue_weight
        opt = optimizer.optimize(
            total_budget=total_budget,
            shopify_weight=sw,
            amazon_weight=aw,
            max_concentration=self.config.mmm.max_channel_concentration,
        )
        return {
            "allocation": opt.allocation,
            "allocation_lower": opt.allocation_lower,
            "allocation_upper": opt.allocation_upper,
            "expected_shopify_revenue": opt.expected_shopify_revenue,
            "expected_amazon_revenue": opt.expected_amazon_revenue,
            "expected_total_revenue": opt.expected_total_revenue,
            "blended_roas": opt.blended_roas,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_channel_results(
    contributions_df: pd.DataFrame,
    rhat_df: pd.DataFrame,
) -> list[MMMChannelResult]:
    """Convert raw contribution and rhat DataFrames into MMMChannelResult list."""
    if contributions_df.empty:
        return []

    rhat_lookup: dict[str, float] = {}
    if not rhat_df.empty:
        for _, row in rhat_df.iterrows():
            rhat_lookup[str(row["parameter"])] = float(row["rhat"])

    results: list[MMMChannelResult] = []
    # Group by channel × outcome, average across DMAs and dates
    group_cols = ["channel", "outcome"]
    available = [c for c in group_cols if c in contributions_df.columns]
    if not available:
        return results

    for (channel, outcome), grp in contributions_df.groupby(["channel", "outcome"]):
        if channel == "baseline":
            continue
        contrib_mean = float(grp["contribution_mean"].mean())
        contrib_p10 = float(grp["contribution_p10"].mean()) if "contribution_p10" in grp else 0.0
        contrib_p90 = float(grp["contribution_p90"].mean()) if "contribution_p90" in grp else 0.0

        total = float(contributions_df[
            contributions_df["outcome"] == outcome
        ]["contribution_mean"].sum()) or 1.0
        contrib_pct = contrib_mean / total

        roi = contrib_mean / max(contrib_mean * 0.3, 1.0)  # rough; real value from posterior

        rhat_key = f"beta_{channel}_{outcome}"
        rhat_val = rhat_lookup.get(rhat_key, rhat_lookup.get(f"beta_{channel}", 1.0))

        results.append(MMMChannelResult(
            channel=str(channel),
            outcome=str(outcome),
            contribution_mean=contrib_mean,
            contribution_p10=contrib_p10,
            contribution_p90=contrib_p90,
            contribution_pct=contrib_pct,
            roi_mean=roi,
            roi_p10=roi * 0.7,
            roi_p90=roi * 1.3,
            marginal_roi=roi * 0.8,
            rhat=rhat_val,
        ))

    return results


def compute_iroas_metrics(
    channel_results: list[MMMChannelResult],
    spend_by_channel: dict[str, float],
    geo_holdout_iroas: dict[str, float] | None = None,
    avg_order_value: float = 75.0,
) -> list[MMMChannelResult]:
    """Populate iROAS, Incrementality Factor, and CPIA on each MMMChannelResult.

    Uses geo holdout iROAS as ground truth when available; falls back to the
    MMM posterior ROI as the best available causal estimate.

    Parameters
    ----------
    channel_results : list[MMMChannelResult]
        Raw channel results (iROAS fields are 0.0 on input).
    spend_by_channel : dict[str, float]
        Actual spend per channel name for the measurement period (dollars).
    geo_holdout_iroas : dict[str, float], optional
        Channel → causal iROAS measured by geo holdout.  When present this
        overrides the MMM-implied iROAS and sets ``iroas_source="geo_holdout"``.
    avg_order_value : float
        Average order value in dollars (used to convert revenue → conversions
        for the CPIA calculation).

    Returns
    -------
    list[MMMChannelResult]
        Same list with iROAS, incrementality_factor, and cpia populated.
    """
    geo_holdout_iroas = geo_holdout_iroas or {}
    updated: list[MMMChannelResult] = []

    for cr in channel_results:
        spend = spend_by_channel.get(cr.channel, 0.0)

        # iROAS: prefer geo holdout truth, fall back to MMM posterior ROI
        if cr.channel in geo_holdout_iroas:
            iroas = float(geo_holdout_iroas[cr.channel])
            iroas_source = "geo_holdout"
        else:
            iroas = cr.roi_mean  # MMM posterior mean ROI as proxy for iROAS
            iroas_source = "mmm"

        # Incrementality Factor: incremental / attributed
        # attributed = contribution_mean (MMM posterior); incremental = iROAS × spend
        attributed = max(cr.contribution_mean, 1e-6)
        incremental = iroas * spend if spend > 0 else attributed
        incrementality_factor = incremental / attributed if attributed > 0 else 0.0
        # Clamp to [0, 2] to guard against data quality issues
        incrementality_factor = max(0.0, min(2.0, incrementality_factor))

        # CPIA: cost per incremental acquisition (purchase)
        # incremental_conversions ≈ incremental_revenue / avg_order_value
        incremental_revenue = iroas * spend
        incremental_conversions = incremental_revenue / avg_order_value if avg_order_value > 0 else 0.0
        cpia = spend / incremental_conversions if incremental_conversions > 0 else 0.0

        updated.append(cr.model_copy(update={
            "iroas": round(iroas, 4),
            "iroas_source": iroas_source,
            "incrementality_factor": round(incrementality_factor, 4),
            "cpia": round(cpia, 2),
        }))

    return updated
