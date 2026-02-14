"""Test orchestrator — the main engine that coordinates test lifecycle.

Manages the full workflow:
1. Design: Pull historical data, run optimizer, produce test design
2. Execute: Apply holdout targeting, monitor test progress
3. Analyze: Run causal inference, compute iROAS, generate reports

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
    run_all_estimators,
)
from incrementality.analysis.iroas import compute_iroas
from incrementality.config import Config
from incrementality.connectors.amazon import AmazonConnector
from incrementality.connectors.facebook import FacebookConnector
from incrementality.connectors.shopify import ShopifyConnector
from incrementality.connectors.youtube import YouTubeConnector
from incrementality.design.optimizer import auto_design_test, compute_dma_historical_metrics
from incrementality.models import (
    AdChannel,
    IncrementalityResult,
    MeasurementScope,
    TestDesign,
    TestReport,
    TestScope,
    TestStatus,
)
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
        """Pull historical data from all configured connectors.

        Returns dict with keys: shopify, amazon, facebook, youtube
        Each value is a DataFrame with daily DMA-level data.
        """
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
        """Load data from CSV files for testing without API access.

        Expected files:
            shopify_daily.csv: date, dma_code, revenue, orders
            amazon_daily.csv: date, dma_code, revenue, orders
            facebook_daily.csv: date, dma_code, spend, impressions, clicks
            youtube_daily.csv: date, dma_code, spend, impressions, views
        """
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
    ) -> TestDesign:
        """Design an incrementality test.

        If data is not provided, pulls from APIs or cache.
        """
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
        )

        # Save design
        self._save_design(design)
        return design

    # ------------------------------------------------------------------
    # Test analysis
    # ------------------------------------------------------------------

    def analyze_test(
        self,
        design: TestDesign,
        pre_data: pd.DataFrame | None = None,
        post_data: pd.DataFrame | None = None,
        ad_spend_data: pd.DataFrame | None = None,
    ) -> TestReport:
        """Analyze a completed test and generate the report.

        Args:
            design: The test design
            pre_data: Pre-test period data (if None, uses lookback before test start)
            post_data: Test period data (if None, pulls from APIs)
            ad_spend_data: Ad spend during test (if None, pulls from APIs)
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

        # Run estimators
        logger.info("Running causal inference estimators...")
        revenue_col = "revenue"
        all_results = run_all_estimators(
            pre_data, post_data, treatment_dmas, holdout_dmas,
            revenue_col=revenue_col,
        )

        # Use DiD as primary estimator
        primary_result = all_results.get(
            "difference_in_differences",
            next(iter(all_results.values())),
        )

        # Compute iROAS
        logger.info("Computing incremental ROAS...")
        iroas = compute_iroas(
            pre_data, post_data, ad_spend_data,
            treatment_dmas, holdout_dmas,
            design.measurement_scope, test_duration_days,
        )

        # Cross-platform incrementality
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

        # Generate report
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

        # Print and save
        print_report(report)
        save_report_json(report, self.output_dir)
        save_report_csv(report, self.output_dir)

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

        # Merge on date + dma_code
        result = frames[0]
        for df in frames[1:]:
            result = result.merge(
                df[["date", "dma_code", "amazon_revenue"]],
                on=["date", "dma_code"],
                how="outer",
            )

        result = result.fillna(0)

        # Total revenue
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
