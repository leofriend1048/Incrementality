"""Core data models for the incrementality testing platform."""

from __future__ import annotations

import enum
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class Platform(str, enum.Enum):
    SHOPIFY = "shopify"
    AMAZON = "amazon"


class AdChannel(str, enum.Enum):
    FACEBOOK = "facebook"
    YOUTUBE = "youtube"


class CellType(str, enum.Enum):
    TREATMENT = "treatment"
    HOLDOUT = "holdout"


class TestStatus(str, enum.Enum):
    DRAFT = "draft"
    DESIGNED = "designed"
    RUNNING = "running"
    COMPLETED = "completed"
    ANALYZED = "analyzed"


class TestScope(str, enum.Enum):
    """What level the holdout applies to."""
    CHANNEL = "channel"  # Hold out an entire ad channel (e.g., all Facebook ads)
    CAMPAIGN = "campaign"  # Hold out specific campaigns within a channel


class MeasurementScope(str, enum.Enum):
    """What revenue to measure."""
    SHOPIFY_ONLY = "shopify_only"
    AMAZON_ONLY = "amazon_only"
    SHOPIFY_AND_AMAZON = "shopify_and_amazon"  # Cross-platform halo effects


# ---------------------------------------------------------------------------
# DMA models
# ---------------------------------------------------------------------------

class DMAProfile(BaseModel):
    """A Designated Market Area with its attributes."""
    dma_code: str
    name: str
    state: str
    population: int
    tv_households: int
    region: str  # Northeast, Southeast, Midwest, West, Southwest


class DMAHistoricalMetrics(BaseModel):
    """Historical performance metrics for a DMA, used for matching."""
    dma_code: str
    period_start: date
    period_end: date
    # Revenue
    shopify_revenue: float = 0.0
    amazon_revenue: float = 0.0
    total_revenue: float = 0.0
    # Orders
    shopify_orders: int = 0
    amazon_orders: int = 0
    total_orders: int = 0
    # Ad spend by channel
    facebook_spend: float = 0.0
    youtube_spend: float = 0.0
    total_ad_spend: float = 0.0
    # Derived
    revenue_per_capita: float = 0.0
    orders_per_capita: float = 0.0
    aov: float = 0.0  # Average order value
    # Trend
    revenue_trend: float = 0.0  # Week-over-week growth rate
    revenue_volatility: float = 0.0  # Coefficient of variation


# ---------------------------------------------------------------------------
# Test design models
# ---------------------------------------------------------------------------

class TestCell(BaseModel):
    """A group of DMAs assigned to treatment or holdout."""
    cell_type: CellType
    dma_codes: list[str]
    total_population: int = 0
    historical_revenue: float = 0.0
    historical_orders: int = 0


class PowerAnalysisResult(BaseModel):
    """Results from statistical power analysis."""
    minimum_detectable_effect: float  # As a proportion (e.g., 0.10 = 10% lift)
    recommended_duration_weeks: int
    required_sample_size_per_cell: int  # Number of DMAs per cell
    statistical_power: float  # Achieved power (target 0.80)
    significance_level: float  # Alpha (typically 0.05)
    baseline_variance: float
    baseline_mean_revenue: float
    effect_size_cohen_d: float
    # Simulation-based fields (new)
    simulated_power: float = 0.0  # Power from actual simulations
    simulated_false_positive_rate: float = 0.0  # Empirical Type I error
    num_simulations: int = 0  # How many sims were run
    power_score: float = 0.0  # 0-100 composite score (Haus targets 85-90)


class TestDesign(BaseModel):
    """Complete test design specification."""
    test_id: str
    name: str
    description: str = ""
    # Scope
    test_scope: TestScope
    ad_channel: AdChannel
    campaign_ids: list[str] = Field(default_factory=list)  # For campaign-level tests
    measurement_scope: MeasurementScope
    # Timing
    lookback_weeks: int = 12  # Historical data used for design
    recommended_start_date: date | None = None
    recommended_end_date: date | None = None
    duration_weeks: int = 4
    # Cells
    treatment_cell: TestCell
    holdout_cell: TestCell
    num_treatment_dmas: int = 0
    num_holdout_dmas: int = 0
    # Statistical
    power_analysis: PowerAnalysisResult | None = None
    balance_score: float = 0.0  # 0-1, higher = better balance between cells
    # Targeting deployment state — saved before deploy so we can revert
    original_targeting: dict = Field(default_factory=dict)
    deployed_campaign_ids: list[str] = Field(default_factory=list)
    deployed_at: datetime | None = None
    reverted_at: datetime | None = None
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: TestStatus = TestStatus.DESIGNED


# ---------------------------------------------------------------------------
# Test results models
# ---------------------------------------------------------------------------

class DailyMetric(BaseModel):
    """Daily metric observation for a DMA during a test."""
    date: date
    dma_code: str
    cell_type: CellType
    shopify_revenue: float = 0.0
    amazon_revenue: float = 0.0
    total_revenue: float = 0.0
    shopify_orders: int = 0
    amazon_orders: int = 0
    total_orders: int = 0
    ad_spend: float = 0.0


class IncrementalityResult(BaseModel):
    """Core incrementality measurement results."""
    # Lift
    absolute_lift: float  # Treatment mean - holdout mean (per DMA per day)
    relative_lift: float  # (Treatment - Holdout) / Holdout as proportion
    lift_lower_ci: float  # 95% CI lower bound
    lift_upper_ci: float  # 95% CI upper bound
    # Statistical significance
    p_value: float
    is_significant: bool  # p < alpha
    confidence_level: float  # 1 - alpha
    # Effect size
    cohen_d: float
    # Method used
    method: str  # "ascm", "bsts", "did", "synthetic_control", "ensemble"
    # Model quality metrics (new)
    l2_imbalance: float = 0.0  # Pre-period fit quality for SC methods
    pre_period_r_squared: float = 0.0  # R² of pre-period fit
    # Lift likelihood — Bayesian-style: P(true lift > 0)
    lift_likelihood: float = 0.0


class IncrementalROAS(BaseModel):
    """Incremental Return on Ad Spend calculation."""
    incremental_revenue: float  # Total estimated incremental revenue
    total_ad_spend: float  # Ad spend in treatment group during test
    iroas: float  # incremental_revenue / total_ad_spend
    iroas_lower_ci: float
    iroas_upper_ci: float
    # Breakdowns
    shopify_incremental_revenue: float = 0.0
    amazon_incremental_revenue: float = 0.0
    shopify_iroas: float = 0.0
    amazon_iroas: float = 0.0
    # Incrementality Factor & CPIA (populated when conversion data available)
    attributed_conversions: float = 0.0  # Platform-reported conversions
    incremental_conversions: float = 0.0  # Experiment-measured incremental conversions
    incrementality_factor: float = 0.0  # incremental / attributed (1.0 = fully incremental)
    cpia: float = 0.0  # Cost Per Incremental Acquisition: spend / incremental_conversions


class TestReport(BaseModel):
    """Complete incrementality test report."""
    test_id: str
    test_name: str
    # Design recap
    ad_channel: AdChannel
    test_scope: TestScope
    measurement_scope: MeasurementScope
    campaign_ids: list[str] = Field(default_factory=list)
    duration_weeks: int
    num_treatment_dmas: int
    num_holdout_dmas: int
    test_start: date
    test_end: date
    # Results
    incrementality: IncrementalityResult
    iroas: IncrementalROAS
    # Context
    treatment_total_revenue: float
    holdout_total_revenue: float
    treatment_avg_daily_revenue: float
    holdout_avg_daily_revenue: float
    organic_baseline_revenue: float  # What would've happened without ads
    # Cross-platform (when measurement_scope includes both)
    shopify_incrementality: IncrementalityResult | None = None
    amazon_incrementality: IncrementalityResult | None = None
    # Validation (new)
    validation: "ValidationReport | None" = None
    # Ensemble details (new)
    estimator_results: dict[str, "IncrementalityResult"] = Field(default_factory=dict)
    estimator_weights: dict[str, float] = Field(default_factory=dict)
    # Winsorized comparison (new)
    winsorized_comparison: dict = Field(default_factory=dict)
    # Anomaly detection (new)
    anomaly_warnings: list[str] = Field(default_factory=list)
    anomaly_blockers: list[str] = Field(default_factory=list)
    # Recommendations
    recommendations: list[str] = Field(default_factory=list)
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class PlaceboTestResult(BaseModel):
    """Result of a single placebo test."""
    placebo_type: str  # "in_time" or "in_space"
    target_dma: str = ""  # For in-space placebos
    placebo_date: date | None = None  # For in-time placebos
    estimated_effect: float
    p_value: float
    is_false_positive: bool  # Did the placebo falsely detect an effect?


class ValidationReport(BaseModel):
    """Comprehensive validation of test results.

    This is the safety net. Every field here protects you from
    making an 8-figure decision on bad data.
    """
    # Pre-treatment fit
    l2_imbalance: float  # Lower = better. Target < 0.10
    pre_period_r_squared: float  # Target > 0.90
    pre_period_mape: float = 0.0  # Mean absolute percentage error in pre-period
    # Placebo tests
    num_placebo_tests: int = 0
    placebo_pass_rate: float = 0.0  # Fraction of placebos that correctly found no effect
    false_positive_rate: float = 0.0  # Should be ~5% at alpha=0.05
    placebo_results: list[PlaceboTestResult] = Field(default_factory=list)
    # AA test (pre-period only — should find NO effect)
    aa_test_p_value: float = 0.0  # Should be > 0.05
    aa_test_passed: bool = False
    # Estimator agreement
    estimator_agreement: float = 0.0  # Do all methods agree? 0-1 scale
    # Overall
    is_trustworthy: bool = False  # Final verdict: safe to make decisions on?
    trust_score: float = 0.0  # 0-100 composite
    warnings: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)  # Hard stops


class FeasibilityResult(BaseModel):
    """Pre-test feasibility assessment. Gates whether a test should run."""
    is_feasible: bool
    power_score: float  # 0-100, need >= 85 to proceed
    estimated_mde: float
    estimated_duration_weeks: int
    min_holdout_dmas: int
    estimated_opportunity_cost: float  # $ lost from holdout
    reasons: list[str] = Field(default_factory=list)  # Why feasible or not
    recommendations: list[str] = Field(default_factory=list)
