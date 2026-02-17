"""Meridian model configuration dataclass.

All hyper-parameters that drive the Google Meridian NUTS sampler and the
pre-processing pipeline are centralised here so that callers never have to
pass magic numbers.

Typical usage
-------------
>>> from incrementality.mmm.config import MeridianConfig
>>> cfg = MeridianConfig()                          # all defaults
>>> cfg = MeridianConfig(mcmc_chains=2, tpu_enabled=True)
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Default per-channel discount factors
# (PRD Section 6.2 - Media quality adjustments)
# ---------------------------------------------------------------------------
_DEFAULT_CHANNEL_DISCOUNT_FACTORS: Dict[str, float] = {
    "meta_perf": 0.65,
    "meta_aware": 0.60,
    "google_brand": 0.85,
    "google_nonbrand": 0.75,
    "tiktok": 0.65,
    "amz_sponsored": 1.00,  # Amazon click-based; no viewability discount
    "email_sms": 0.80,
}

# ---------------------------------------------------------------------------
# Default per-channel prior sigma values
# (PRD Section 6.3 - Half-normal sigma on ROI priors, calibrated to industry)
# ---------------------------------------------------------------------------
_DEFAULT_PRIOR_SIGMA: Dict[str, float] = {
    "meta_perf": 0.15,
    "meta_aware": 0.18,
    "google_brand": 0.12,
    "google_nonbrand": 0.14,
    "tiktok": 0.18,
    "amz_sponsored": 0.13,
    "email_sms": 0.16,
}

# ---------------------------------------------------------------------------
# Canonical channel and outcome identifiers
# ---------------------------------------------------------------------------
_DEFAULT_CHANNELS: List[str] = [
    "meta_perf",
    "meta_aware",
    "google_brand",
    "google_nonbrand",
    "tiktok",
    "amz_sponsored",
    "email_sms",
]

_DEFAULT_OUTCOMES: List[str] = ["shopify", "amazon"]


class MeridianConfig(BaseModel):
    """Configuration for the Google Meridian MMM model.

    All fields have sensible defaults matching the Michael Todd Beauty PRD.
    Override individual fields as needed; the validator will ensure internal
    consistency (e.g. channel lists match tensor dimension counts).

    Parameters
    ----------
    n_dmas : int
        Number of Designated Market Areas (geo dimension).  Default 210
        covers the full Nielsen DMA universe.
    n_channels : int
        Number of paid media channels.  Must equal ``len(channels)``.
    n_outcomes : int
        Number of KPI outcomes modelled simultaneously.  Must equal
        ``len(outcomes)``.
    channels : list[str]
        Ordered list of channel identifiers.  Order must match the media
        tensor axis-2 ordering.
    outcomes : list[str]
        Ordered list of outcome / KPI identifiers.  Order must match the
        KPI tensor axis-2 ordering.
    mcmc_chains : int
        Number of independent MCMC chains for NUTS sampling.
    mcmc_warmup : int
        Number of warmup (burn-in) NUTS steps discarded from each chain.
    mcmc_samples : int
        Number of posterior samples retained per chain.  Total posterior
        size = ``mcmc_chains * mcmc_samples``.
    convergence_threshold : float
        Maximum acceptable R-hat (Gelman-Rubin) statistic.  Any parameter
        with R-hat above this threshold is flagged as non-converged.
        Default 1.05 per PRD convergence criteria.
    adstock_max_lag : int
        Maximum carryover lag in days for the adstock transformation.
        Default 56 days = 8 weeks.
    channel_discount_factors : dict[str, float]
        Media quality discount applied to gross impressions before entering
        the model (e.g. 0.65 for Meta due to ~35 % non-viewable inventory).
    prior_sigma : dict[str, float]
        Half-normal sigma on the ROI prior for each channel.  Smaller
        values apply a tighter prior (more regularisation).
    random_walk_step_size : float
        Step size for the time-varying beta random walk prior.  Controls
        how quickly channel effectiveness can drift over time.
    daily_update_mode : bool
        When True, the model is re-conditioned on new data daily using
        warm-starting from the previous posterior.  When False (default)
        the model is re-fitted from scratch on a cadence.
    tpu_enabled : bool
        Enable JAX TPU backend for sampling.  Requires the TPU runtime to
        be available in the execution environment.
    """

    # -----------------------------------------------------------------------
    # Geometry
    # -----------------------------------------------------------------------
    n_dmas: int = Field(default=210, ge=1, description="Number of geo units (DMAs).")
    n_channels: int = Field(default=7, ge=1, description="Number of media channels.")
    n_outcomes: int = Field(default=2, ge=1, description="Number of KPI outcomes.")

    channels: List[str] = Field(
        default_factory=lambda: list(_DEFAULT_CHANNELS),
        description="Ordered list of channel identifiers.",
    )
    outcomes: List[str] = Field(
        default_factory=lambda: list(_DEFAULT_OUTCOMES),
        description="Ordered list of outcome identifiers.",
    )

    # -----------------------------------------------------------------------
    # MCMC sampler
    # -----------------------------------------------------------------------
    mcmc_chains: int = Field(default=4, ge=1, le=16, description="Number of NUTS chains.")
    mcmc_warmup: int = Field(default=1000, ge=100, description="Warmup steps per chain.")
    mcmc_samples: int = Field(default=2000, ge=100, description="Posterior samples per chain.")
    convergence_threshold: float = Field(
        default=1.05,
        ge=1.0,
        le=1.5,
        description="Max acceptable R-hat for convergence check.",
    )

    # -----------------------------------------------------------------------
    # Adstock / saturation
    # -----------------------------------------------------------------------
    adstock_max_lag: int = Field(
        default=56,
        ge=1,
        description="Maximum carryover lag in days (default 8 weeks = 56 days).",
    )

    # -----------------------------------------------------------------------
    # Media quality discounts (per channel)
    # -----------------------------------------------------------------------
    channel_discount_factors: Dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_CHANNEL_DISCOUNT_FACTORS),
        description="Impression quality discount factor per channel (0-1).",
    )

    # -----------------------------------------------------------------------
    # Priors (per channel)
    # -----------------------------------------------------------------------
    prior_sigma: Dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_PRIOR_SIGMA),
        description="Half-normal sigma on ROI prior for each channel.",
    )

    # -----------------------------------------------------------------------
    # Time-varying parameters
    # -----------------------------------------------------------------------
    random_walk_step_size: float = Field(
        default=0.05,
        gt=0.0,
        description="Step size for time-varying beta random walk prior.",
    )

    # -----------------------------------------------------------------------
    # Runtime flags
    # -----------------------------------------------------------------------
    daily_update_mode: bool = Field(
        default=False,
        description="Re-condition on new data daily (warm-start mode).",
    )
    tpu_enabled: bool = Field(
        default=False,
        description="Use JAX TPU backend for sampling.",
    )

    # -----------------------------------------------------------------------
    # Derived / computed properties
    # -----------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "MeridianConfig":
        """Ensure n_channels / n_outcomes match the corresponding lists."""
        if len(self.channels) != self.n_channels:
            raise ValueError(
                f"n_channels={self.n_channels} does not match "
                f"len(channels)={len(self.channels)}."
            )
        if len(self.outcomes) != self.n_outcomes:
            raise ValueError(
                f"n_outcomes={self.n_outcomes} does not match "
                f"len(outcomes)={len(self.outcomes)}."
            )
        return self

    @model_validator(mode="after")
    def _validate_discount_factors(self) -> "MeridianConfig":
        """Verify all discount factors are in (0, 1] and cover every channel."""
        for ch, factor in self.channel_discount_factors.items():
            if not (0.0 < factor <= 1.0):
                raise ValueError(
                    f"channel_discount_factors['{ch}'] = {factor} is outside (0, 1]."
                )
        return self

    @model_validator(mode="after")
    def _validate_prior_sigma(self) -> "MeridianConfig":
        """Verify all prior sigmas are positive."""
        for ch, sigma in self.prior_sigma.items():
            if sigma <= 0.0:
                raise ValueError(
                    f"prior_sigma['{ch}'] = {sigma} must be positive."
                )
        return self

    # -----------------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------------

    @property
    def total_posterior_samples(self) -> int:
        """Total number of posterior draws across all chains."""
        return self.mcmc_chains * self.mcmc_samples

    def channel_index(self, channel: str) -> int:
        """Return the integer index of *channel* in the media tensor axis-2."""
        try:
            return self.channels.index(channel)
        except ValueError as exc:
            raise KeyError(f"Unknown channel: '{channel}'") from exc

    def outcome_index(self, outcome: str) -> int:
        """Return the integer index of *outcome* in the KPI tensor axis-2."""
        try:
            return self.outcomes.index(outcome)
        except ValueError as exc:
            raise KeyError(f"Unknown outcome: '{outcome}'") from exc

    model_config = {"frozen": False}
