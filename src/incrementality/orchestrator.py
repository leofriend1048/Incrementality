"""Test orchestrator -- the main engine that coordinates test lifecycle.

Manages the full workflow:
1. Design: Pull historical data, run optimizer, produce test design
2. Execute: Apply holdout targeting, monitor test progress
3. Analyze: Run causal inference, compute iROAS, generate reports

Now uses the production-grade pipeline:
- Ensemble estimator (ASCM + BSTS + DiD) as primary
- Full validation suite (AA test, placebos, estimator agreement)
- Spillover-adjusted estimates
- Post-treatment observation windows

Stores test state as JSON files for persistence between runs.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from incrementality.analysis.estimators import (
    difference_in_differences,
    run_ensemble,
)
from incrementality.analysis.iroas import compute_iroas
from incrementality.analysis.validation import run_full_validation
from incrementality.config import Config
from incrementality.connectors.amazon import AmazonConnector
from incrementality.connectors.facebook import FacebookConnector
from incrementality.connectors.shopify import ShopifyConnector
from incrementality.connectors.youtube import YouTubeConnector
from incrementality.design.optimizer import auto_design_test, compute_dma_historical_metrics
from incrementality.design.spillover import compute_spillover_risk, adjust_for_spillover
from incrementality.models import (
    AdChannel,
    IncrementalityResult,
    MeasurementScope,
    TestDesign,
    TestReport,
    TestScope,
    TestStatus,
)
from incrementality.report_pdf import save_report_html, save_report_pdf
from incrementality.reporting import (
    generate_report,
    print_report,
    save_report_csv,
    save_report_json,
)

logger = logging.getLogger(__name__)


class TestOrchestrator:
    """Orchestrates the full incrementality test lifecycle."""

    def __init__(self, config: Config):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.output_dir = Path(config.output_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize connectors
        self._shopify = (
            ShopifyConnector(config.shopify) if config.shopify else None
        )
        self._amazon = (
            AmazonConnector(config.amazon) if config.amazon else None
        )
        self._facebook = (
            FacebookConnector(config.facebook) if config.facebook else None
        )
        self._youtube = (
            YouTubeConnector(config.youtube) if config.youtube else None
        )

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def pull_historical_data(
        self,
        lookback_weeks: int = 12,
        end_date: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Pull historical data from all configured connectors."""
        end = end_date or date.today()
        start = end - timedelta(weeks=lookback_weeks)
        data = {}

        if self._shopify:
            logger.info(f"Pulling Shopify data: {start} to {end}")
            data["shopify"] = self._shopify.get_daily_revenue_by_dma(start, end)
        else:
            data["shopify"] = pd.DataFrame()

        if self._amazon:
            logger.info(f"Pulling Amazon data: {start} to {end}")
            data["amazon"] = self._amazon.get_daily_revenue_by_dma(start, end)
        else:
            data["amazon"] = pd.DataFrame()

        if self._facebook:
            logger.info(f"Pulling Facebook spend data: {start} to {end}")
            data["facebook"] = self._facebook.fetch_spend_by_dma(start, end)
        else:
            data["facebook"] = pd.DataFrame()

        if self._youtube:
            logger.info(f"Pulling YouTube spend data: {start} to {end}")
            data["youtube"] = self._youtube.fetch_spend_by_dma(start, end)
        else:
            data["youtube"] = pd.DataFrame()

        # Cache to disk
        for name, df in data.items():
            if not df.empty:
                cache_path = self.data_dir / f"historical_{name}.parquet"
                df.to_parquet(cache_path, index=False)
                logger.info(f"Cached {name} data: {len(df)} rows -> {cache_path}")

        return data

    def load_cached_data(self) -> dict[str, pd.DataFrame]:
        """Load previously cached historical data."""
        data = {}
        for name in ["shopify", "amazon", "facebook", "youtube"]:
            cache_path = self.data_dir / f"historical_{name}.parquet"
            if cache_path.exists():
                data[name] = pd.read_parquet(cache_path)
                logger.info(f"Loaded cached {name} data: {len(data[name])} rows")
            else:
                data[name] = pd.DataFrame()
        return data

    def load_data_from_csv(self, csv_dir: str | Path) -> dict[str, pd.DataFrame]:
        """Load data from CSV files for testing without API access."""
        csv_dir = Path(csv_dir)
        data = {}
        for name in ["shopify", "amazon", "facebook", "youtube"]:
            path = csv_dir / f"{name}_daily.csv"
            if path.exists():
                data[name] = pd.read_csv(path, parse_dates=["date"])
                logger.info(f"Loaded {name} CSV: {len(data[name])} rows")
            else:
                data[name] = pd.DataFrame()
        return data

    # ------------------------------------------------------------------
    # Test design
    # ------------------------------------------------------------------

    def design_test(
        self,
        ad_channel: AdChannel,
        test_scope: TestScope = TestScope.CHANNEL,
        measurement_scope: MeasurementScope = MeasurementScope.SHOPIFY_AND_AMAZON,
        campaign_ids: list[str] | None = None,
        test_name: str = "Incrementality Test",
        target_mde: float | None = None,
        data: dict[str, pd.DataFrame] | None = None,
        lookback_weeks: int = 12,
        run_simulation: bool = True,
    ) -> TestDesign:
        """Design an incrementality test."""
        if data is None:
            try:
                data = self.pull_historical_data(lookback_weeks)
            except Exception:
                logger.info("API pull failed, trying cached data")
                data = self.load_cached_data()

        design = auto_design_test(
            shopify_daily=data.get("shopify"),
            amazon_daily=data.get("amazon"),
            facebook_daily=data.get("facebook"),
            youtube_daily=data.get("youtube"),
            ad_channel=ad_channel,
            test_scope=test_scope,
            measurement_scope=measurement_scope,
            campaign_ids=campaign_ids,
            config=self.config.statistical,
            test_name=test_name,
            target_mde=target_mde,
            run_simulation=run_simulation,
        )

        self._save_design(design)
        return design

    # ------------------------------------------------------------------
    # Test analysis (production pipeline)
    # ------------------------------------------------------------------

    def analyze_test(
        self,
        design: TestDesign,
        pre_data: pd.DataFrame | None = None,
        post_data: pd.DataFrame | None = None,
        ad_spend_data: pd.DataFrame | None = None,
    ) -> TestReport:
        """Analyze a completed test and generate the report.

        Production pipeline:
        1. Run ensemble estimator (ASCM + BSTS + DiD)
        2. Run full validation suite
        3. Compute iROAS using ensemble result
        4. Assess spillover risk
        5. Generate comprehensive report with trust score
        """
        treatment_dmas = design.treatment_cell.dma_codes
        holdout_dmas = design.holdout_cell.dma_codes

        # Determine date ranges
        test_start = design.recommended_start_date
        test_end = design.recommended_end_date
        pre_start = test_start - timedelta(weeks=design.lookback_weeks)
        pre_end = test_start - timedelta(days=1)
        test_duration_days = (test_end - test_start).days

        # Pull data if not provided
        if pre_data is None or post_data is None:
            all_data = self._pull_test_data(
                pre_start, test_end, design.measurement_scope,
            )
            if pre_data is None:
                pre_data = all_data[
                    (pd.to_datetime(all_data["date"]).dt.date >= pre_start)
                    & (pd.to_datetime(all_data["date"]).dt.date <= pre_end)
                ]
            if post_data is None:
                post_data = all_data[
                    (pd.to_datetime(all_data["date"]).dt.date >= test_start)
                    & (pd.to_datetime(all_data["date"]).dt.date <= test_end)
                ]

        if ad_spend_data is None:
            ad_spend_data = self._pull_ad_spend(
                test_start, test_end, design.ad_channel, design.campaign_ids,
            )

        # --- Step 1: Run ensemble estimator ---
        logger.info("Running ensemble causal inference (ASCM + BSTS + DiD)...")
        revenue_col = "revenue"
        try:
            ensemble_result, estimator_results, estimator_weights = run_ensemble(
                pre_data, post_data, treatment_dmas, holdout_dmas,
                revenue_col=revenue_col,
            )
            primary_result = ensemble_result
        except Exception as e:
            logger.warning(f"Ensemble failed: {e}. Falling back to DiD.")
            primary_result = difference_in_differences(
                pre_data, post_data, treatment_dmas, holdout_dmas,
                revenue_col=revenue_col,
            )
            estimator_results = {"did": primary_result}
            estimator_weights = {"did": 1.0}
            ensemble_result = primary_result

        # --- Step 2: Run full validation suite ---
        logger.info("Running validation suite...")
        try:
            validation = run_full_validation(
                pre_data, post_data,
                treatment_dmas, holdout_dmas,
                ensemble_result, estimator_results,
                revenue_col=revenue_col,
            )
            logger.info(
                f"Validation: trust_score={validation.trust_score:.0f}/100, "
                f"trustworthy={validation.is_trustworthy}, "
                f"blockers={len(validation.blockers)}"
            )
            if validation.blockers:
                for blocker in validation.blockers:
                    logger.warning(f"BLOCKER: {blocker}")
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            validation = None

        # --- Step 3: Spillover assessment ---
        spillover = compute_spillover_risk(treatment_dmas, holdout_dmas)
        if spillover["risk_score"] > 0.1:
            logger.info(
                f"Spillover risk: {spillover['risk_score']:.0%}. "
                f"Adjusting effect estimate."
            )
            adjusted_lift = adjust_for_spillover(
                primary_result.absolute_lift, spillover,
            )
            logger.info(
                f"Spillover adjustment: {primary_result.absolute_lift:.4f} -> "
                f"{adjusted_lift:.4f}"
            )

        # --- Step 4: Compute iROAS using ensemble result ---
        logger.info("Computing incremental ROAS...")
        iroas = compute_iroas(
            pre_data, post_data, ad_spend_data,
            treatment_dmas, holdout_dmas,
            design.measurement_scope, test_duration_days,
            primary_result=primary_result,
        )

        # --- Step 5: Cross-platform incrementality ---
        shopify_inc = None
        amazon_inc = None
        if design.measurement_scope == MeasurementScope.SHOPIFY_AND_AMAZON:
            if "shopify_revenue" in post_data.columns:
                try:
                    shopify_inc = difference_in_differences(
                        pre_data, post_data, treatment_dmas, holdout_dmas,
                        revenue_col="shopify_revenue",
                    )
                except Exception:
                    pass
            if "amazon_revenue" in post_data.columns:
                try:
                    amazon_inc = difference_in_differences(
                        pre_data, post_data, treatment_dmas, holdout_dmas,
                        revenue_col="amazon_revenue",
                    )
                except Exception:
                    pass

        # Revenue totals
        treatment_rev = post_data[
            post_data["dma_code"].isin(treatment_dmas)
        ][revenue_col].sum()
        holdout_rev = post_data[
            post_data["dma_code"].isin(holdout_dmas)
        ][revenue_col].sum()

        # --- Step 6: Generate report ---
        report = generate_report(
            design=design,
            incrementality=primary_result,
            iroas=iroas,
            treatment_total_revenue=float(treatment_rev),
            holdout_total_revenue=float(holdout_rev),
            test_duration_days=test_duration_days,
            shopify_incrementality=shopify_inc,
            amazon_incrementality=amazon_inc,
        )

        # Attach validation and ensemble details
        report.validation = validation
        report.estimator_results = estimator_results
        report.estimator_weights = estimator_weights

        # Print and save
        print_report(report)
        save_report_json(report, self.output_dir)
        save_report_csv(report, self.output_dir)
        html_path = save_report_html(report, self.output_dir)
        logger.info(f"HTML report: {html_path}")
        try:
            pdf_path = save_report_pdf(report, self.output_dir)
            logger.info(f"PDF report: {pdf_path}")
        except Exception as e:
            logger.debug(f"PDF generation skipped: {e}")

        # Update design status
        design.status = TestStatus.ANALYZED
        self._save_design(design)

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pull_test_data(
        self,
        start: date,
        end: date,
        measurement_scope: MeasurementScope,
    ) -> pd.DataFrame:
        """Pull revenue data for the test period."""
        frames = []

        if measurement_scope in (
            MeasurementScope.SHOPIFY_ONLY,
            MeasurementScope.SHOPIFY_AND_AMAZON,
        ):
            if self._shopify:
                shopify = self._shopify.get_daily_revenue_by_dma(start, end)
                shopify = shopify.rename(columns={"revenue": "shopify_revenue"})
                frames.append(shopify)

        if measurement_scope in (
            MeasurementScope.AMAZON_ONLY,
            MeasurementScope.SHOPIFY_AND_AMAZON,
        ):
            if self._amazon:
                amazon = self._amazon.get_daily_revenue_by_dma(start, end)
                amazon = amazon.rename(columns={"revenue": "amazon_revenue"})
                frames.append(amazon)

        if not frames:
            raise ValueError("No revenue data sources available")

        result = frames[0]
        for df in frames[1:]:
            result = result.merge(
                df[["date", "dma_code", "amazon_revenue"]],
                on=["date", "dma_code"],
                how="outer",
            )

        result = result.fillna(0)

        rev_cols = [c for c in result.columns if c.endswith("_revenue")]
        if rev_cols:
            result["revenue"] = result[rev_cols].sum(axis=1)
        elif "revenue" not in result.columns:
            result["revenue"] = 0

        return result

    def _pull_ad_spend(
        self,
        start: date,
        end: date,
        channel: AdChannel,
        campaign_ids: list[str] | None,
    ) -> pd.DataFrame:
        """Pull ad spend data for the specified channel."""
        if channel == AdChannel.FACEBOOK and self._facebook:
            return self._facebook.fetch_spend_by_dma(start, end, campaign_ids)
        elif channel == AdChannel.YOUTUBE and self._youtube:
            return self._youtube.fetch_spend_by_dma(start, end, campaign_ids)
        else:
            logger.warning(f"No connector for {channel.value}")
            return pd.DataFrame(columns=["date", "dma_code", "spend"])

    def _save_design(self, design: TestDesign) -> None:
        """Persist test design to disk."""
        path = self.data_dir / f"{design.test_id}_design.json"
        with open(path, "w") as f:
            json.dump(design.model_dump(mode="json"), f, indent=2, default=str)
        logger.info(f"Design saved to {path}")

    def load_design(self, test_id: str) -> TestDesign:
        """Load a saved test design."""
        path = self.data_dir / f"{test_id}_design.json"
        with open(path) as f:
            raw = json.load(f)
        return TestDesign(**raw)

    def list_tests(self) -> list[dict]:
        """List all saved test designs."""
        tests = []
        for path in self.data_dir.glob("*_design.json"):
            with open(path) as f:
                raw = json.load(f)
            tests.append({
                "test_id": raw["test_id"],
                "name": raw["name"],
                "status": raw["status"],
                "channel": raw["ad_channel"],
                "duration_weeks": raw["duration_weeks"],
            })
        return tests
