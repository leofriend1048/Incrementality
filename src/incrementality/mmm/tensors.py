"""Tensor preparation utilities for the Google Meridian MMM engine.

All functions produce numpy arrays in the shapes expected by the Meridian
library:

  KPI tensor          : float32  [T, G, K]   T=time, G=geos, K=outcomes
  Media tensor        : float32  [T, G, C]   C=channels
  Extra-features      : float32  [T, G, F]   F=extra feature dims
  Population weights  : float32  [G]

Adstock / saturation transformations are applied *before* passing tensors
to the Meridian model so that the probabilistic graph operates on the
already-transformed media signals.

Example
-------
>>> from incrementality.mmm.tensors import MeridianTensorBuilder
>>> builder = MeridianTensorBuilder(config)
>>> kpi    = builder.build_kpi_tensor(shopify_df, amazon_df, dma_list)
>>> media  = builder.build_media_tensor(spend_dfs, dma_list, channel_list)
>>> extra  = builder.build_extra_features_tensor(promo_df, trends_df, bsr_df)
>>> pop    = builder.build_population_weights(dma_df)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from incrementality.mmm.config import MeridianConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stand-alone transformation functions
# ---------------------------------------------------------------------------


def apply_geometric_adstock(
    series: np.ndarray,
    decay_rate: float,
    max_lag: int,
) -> np.ndarray:
    """Apply geometric (exponential) adstock to a 1-D spend series.

    Each observation at time *t* carries forward a geometrically decaying
    contribution to all future periods up to *max_lag* days.

    Parameters
    ----------
    series : np.ndarray
        1-D array of spend / impression values, ordered chronologically.
    decay_rate : float
        Decay factor ∈ (0, 1).  A value of 0.5 means spend retains 50 % of
        its effect after one period.
    max_lag : int
        Maximum number of periods to carry the effect forward.

    Returns
    -------
    np.ndarray
        Adstocked series of the same length as *series*.

    Raises
    ------
    ValueError
        If *decay_rate* is not in (0, 1) or *max_lag* is non-positive.
    """
    if not (0.0 < decay_rate < 1.0):
        raise ValueError(f"decay_rate must be in (0, 1), got {decay_rate}.")
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}.")

    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    result = np.zeros(n, dtype=np.float64)

    # Build weight kernel: w[l] = decay_rate^l
    lags = np.arange(max_lag + 1)
    weights = decay_rate ** lags  # shape (max_lag+1,)

    for t in range(n):
        # The kernel reaches back from t to max(0, t-max_lag)
        for lag in range(min(max_lag + 1, t + 1)):
            result[t] += weights[lag] * series[t - lag]

    logger.debug(
        "Geometric adstock applied: decay_rate=%.3f, max_lag=%d, series_len=%d",
        decay_rate,
        max_lag,
        n,
    )
    return result.astype(np.float32)


def apply_weibull_adstock(
    series: np.ndarray,
    alpha: float,
    lam: float,
    k: float,
    max_lag: int,
) -> np.ndarray:
    """Apply Weibull CDF-based adstock to a 1-D spend series.

    The Weibull kernel models non-monotone carryover effects (e.g. a peak
    response 2-3 days after ad exposure followed by decay).

    Parameters
    ----------
    series : np.ndarray
        1-D array of spend / impression values, ordered chronologically.
    alpha : float
        Peak weight scale, controls overall carryover magnitude.
    lam : float
        Weibull scale parameter (λ > 0).  Larger values shift the peak
        response further into the future.
    k : float
        Weibull shape parameter (k > 0).  k < 1 → monotone decay,
        k = 1 → exponential, k > 1 → peaked response.
    max_lag : int
        Maximum number of periods to carry the effect forward.

    Returns
    -------
    np.ndarray
        Adstocked series of the same length as *series*.
    """
    if lam <= 0.0:
        raise ValueError(f"lam (Weibull scale) must be > 0, got {lam}.")
    if k <= 0.0:
        raise ValueError(f"k (Weibull shape) must be > 0, got {k}.")
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}.")

    series = np.asarray(series, dtype=np.float64)
    n = len(series)

    # Build Weibull CDF-derived weight kernel
    lags = np.arange(1, max_lag + 2, dtype=np.float64)  # 1 … max_lag+1
    cdf = 1.0 - np.exp(-((lags / lam) ** k))
    # Lag 0 weight = 0 (no instantaneous carryover beyond the period itself)
    weights = np.diff(np.concatenate([[0.0], cdf]))  # shape (max_lag+1,)
    weights = alpha * weights  # apply scale

    result = np.zeros(n, dtype=np.float64)
    for t in range(n):
        for lag in range(min(max_lag + 1, t + 1)):
            result[t] += weights[lag] * series[t - lag]

    # Add the unshifted series (immediate effect component)
    result += series

    logger.debug(
        "Weibull adstock applied: alpha=%.3f, lam=%.3f, k=%.3f, max_lag=%d",
        alpha,
        lam,
        k,
        max_lag,
    )
    return result.astype(np.float32)


def apply_hill_saturation(
    series: np.ndarray,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    """Apply Hill (S-curve) saturation transformation to a spend series.

    The Hill function is:  f(x) = x^alpha / (x^alpha + gamma^alpha)

    Values near zero receive a near-zero response; the curve inflects around
    *gamma* and asymptotes to 1 for very large inputs.

    Parameters
    ----------
    series : np.ndarray
        1-D (or N-D) array of media values (post-adstock).  Non-negative.
    alpha : float
        Hill exponent (shape).  Controls the steepness of the S-curve.
        alpha > 1 → S-shaped; alpha = 1 → concave; alpha < 1 → convex.
    gamma : float
        Half-saturation constant.  The spend level at which the response
        is 50 % of the maximum.  Must be positive.

    Returns
    -------
    np.ndarray
        Saturated series, same shape as *series*, values in [0, 1).

    Raises
    ------
    ValueError
        If *alpha* or *gamma* are non-positive.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha (Hill exponent) must be > 0, got {alpha}.")
    if gamma <= 0.0:
        raise ValueError(f"gamma (half-saturation) must be > 0, got {gamma}.")

    series = np.asarray(series, dtype=np.float64)
    x_alpha = np.power(np.clip(series, 0.0, None), alpha)
    g_alpha = gamma ** alpha
    saturated = x_alpha / (x_alpha + g_alpha)

    logger.debug(
        "Hill saturation applied: alpha=%.3f, gamma=%.3f, input_range=[%.2f, %.2f]",
        alpha,
        gamma,
        float(series.min()),
        float(series.max()),
    )
    return saturated.astype(np.float32)


# ---------------------------------------------------------------------------
# MeridianTensorBuilder
# ---------------------------------------------------------------------------


class MeridianTensorBuilder:
    """Assemble Meridian-ready numpy tensors from raw pandas DataFrames.

    The builder is stateless; all methods can be called in any order.  Each
    method validates its inputs, logs a summary, and returns a float32 numpy
    array in the exact shape expected by the Meridian InputData object.

    Parameters
    ----------
    config : MeridianConfig
        Model configuration specifying channel names, outcome names, and
        adstock parameters.

    Example
    -------
    >>> builder = MeridianTensorBuilder(config)
    >>> kpi    = builder.build_kpi_tensor(shopify_df, amazon_df, dma_list)
    >>> media  = builder.build_media_tensor(spend_dfs, dma_list, channel_list)
    >>> extra  = builder.build_extra_features_tensor(promo_df, trends_df, bsr_df)
    >>> pop    = builder.build_population_weights(dma_df)
    """

    def __init__(self, config: MeridianConfig) -> None:
        self.config = config
        logger.info(
            "MeridianTensorBuilder initialised with %d channels, %d outcomes.",
            config.n_channels,
            config.n_outcomes,
        )

    # ------------------------------------------------------------------
    # KPI tensor
    # ------------------------------------------------------------------

    def build_kpi_tensor(
        self,
        shopify_df: pd.DataFrame,
        amazon_df: pd.DataFrame,
        dma_list: List[str],
    ) -> np.ndarray:
        """Assemble the KPI (outcome) tensor.

        Parameters
        ----------
        shopify_df : pd.DataFrame
            Long-format DataFrame with columns ``["date", "dma", "revenue"]``.
            One row per (date, DMA) combination.
        amazon_df : pd.DataFrame
            Same structure as *shopify_df* but for Amazon revenue.
        dma_list : list[str]
            Ordered list of DMA codes.  Defines axis-1 ordering in the tensor.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``[T, G, 2]`` where T = number of unique
            dates, G = ``len(dma_list)``, and axis-2 = [shopify, amazon].
        """
        logger.info("Building KPI tensor for %d DMAs.", len(dma_list))

        shopify_pivot = self._pivot_revenue(shopify_df, dma_list, label="shopify")
        amazon_pivot = self._pivot_revenue(amazon_df, dma_list, label="amazon")

        # Align date indices
        common_dates = shopify_pivot.index.intersection(amazon_pivot.index).sort_values()
        shopify_aligned = shopify_pivot.loc[common_dates]
        amazon_aligned = amazon_pivot.loc[common_dates]

        T = len(common_dates)
        G = len(dma_list)

        kpi = np.stack(
            [shopify_aligned.values, amazon_aligned.values],
            axis=-1,
        )  # [T, G, 2]

        assert kpi.shape == (T, G, 2), f"KPI tensor shape mismatch: {kpi.shape}"
        logger.info("KPI tensor shape: %s", kpi.shape)
        return kpi.astype(np.float32)

    def _pivot_revenue(
        self,
        df: pd.DataFrame,
        dma_list: List[str],
        label: str,
    ) -> pd.DataFrame:
        """Pivot a long-format revenue DataFrame to wide (dates × DMAs)."""
        required = {"date", "dma", "revenue"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{label} DataFrame missing columns: {missing}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        pivot = df.pivot_table(
            index="date",
            columns="dma",
            values="revenue",
            aggfunc="sum",
        )
        # Reindex to ensure all DMAs are present (fill missing with 0)
        pivot = pivot.reindex(columns=dma_list, fill_value=0.0)
        return pivot

    # ------------------------------------------------------------------
    # Media tensor
    # ------------------------------------------------------------------

    def build_media_tensor(
        self,
        spend_dfs: Dict[str, pd.DataFrame],
        dma_list: List[str],
        channel_list: List[str],
    ) -> np.ndarray:
        """Assemble the media (spend) tensor with discount factors applied.

        Parameters
        ----------
        spend_dfs : dict[str, pd.DataFrame]
            Mapping ``channel_name → DataFrame``.  Each DataFrame must have
            columns ``["date", "dma", "spend"]``.
        dma_list : list[str]
            Ordered DMA codes; defines axis-1 of the output tensor.
        channel_list : list[str]
            Ordered channel names; defines axis-2 of the output tensor.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``[T, G, C]`` where C = ``len(channel_list)``.
            Spend values have the per-channel quality discount factor applied.
        """
        logger.info(
            "Building media tensor for %d channels, %d DMAs.",
            len(channel_list),
            len(dma_list),
        )

        channel_arrays: List[np.ndarray] = []
        reference_dates: Optional[pd.DatetimeIndex] = None

        for channel in channel_list:
            if channel not in spend_dfs:
                logger.warning(
                    "Channel '%s' not in spend_dfs; filling with zeros.", channel
                )
                if reference_dates is None:
                    raise ValueError(
                        "Cannot infer date range: at least one channel must be "
                        "present in spend_dfs."
                    )
                T = len(reference_dates)
                G = len(dma_list)
                channel_arrays.append(np.zeros((T, G), dtype=np.float64))
                continue

            df = spend_dfs[channel].copy()
            df["date"] = pd.to_datetime(df["date"])

            pivot = df.pivot_table(
                index="date",
                columns="dma",
                values="spend",
                aggfunc="sum",
            ).reindex(columns=dma_list, fill_value=0.0)

            if reference_dates is None:
                reference_dates = pivot.index.sort_values()

            pivot = pivot.reindex(reference_dates, fill_value=0.0)

            # Apply quality discount factor
            discount = self.config.channel_discount_factors.get(channel, 1.0)
            if discount != 1.0:
                logger.debug(
                    "Applying discount factor %.2f to channel '%s'.", discount, channel
                )
            pivot = pivot * discount

            channel_arrays.append(pivot.values)

        media = np.stack(channel_arrays, axis=-1)  # [T, G, C]
        logger.info("Media tensor shape: %s", media.shape)
        return media.astype(np.float32)

    # ------------------------------------------------------------------
    # Extra features tensor
    # ------------------------------------------------------------------

    def build_extra_features_tensor(
        self,
        promo_df: pd.DataFrame,
        trends_df: pd.DataFrame,
        bsr_df: pd.DataFrame,
    ) -> np.ndarray:
        """Assemble the extra features (covariates) tensor.

        Extra features are non-media time-varying signals that help the model
        explain revenue variance unrelated to paid media:

          - *promo_df*   : promotional flags / discount depths
          - *trends_df*  : Google Trends index for key product search terms
          - *bsr_df*     : Amazon Best Seller Rank (organic demand proxy)

        Each input DataFrame must have a ``"date"`` column plus one or more
        numeric feature columns.  They are joined on date and broadcast
        across all DMAs (features are assumed national).

        Parameters
        ----------
        promo_df : pd.DataFrame
            Columns: ``["date", ...]`` where ``...`` are numeric promo features.
        trends_df : pd.DataFrame
            Columns: ``["date", ...]`` with Google Trends index values.
        bsr_df : pd.DataFrame
            Columns: ``["date", ...]`` with BSR or organic rank values.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``[T, G, F]`` where F is the total number
            of feature columns across all three input DataFrames.
        """
        logger.info("Building extra features tensor.")

        frames = []
        for label, df in [("promo", promo_df), ("trends", trends_df), ("bsr", bsr_df)]:
            if "date" not in df.columns:
                raise ValueError(f"{label}_df must contain a 'date' column.")
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            frames.append(df)

        # Outer join on date so we do not silently drop periods
        combined = frames[0].join(frames[1], how="outer", rsuffix="_trends")
        combined = combined.join(frames[2], how="outer", rsuffix="_bsr")
        combined = combined.sort_index().fillna(0.0)

        T = len(combined)
        F = combined.shape[1]

        # Broadcast national features to all DMAs: [T, F] → [T, G, F]
        G = self.config.n_dmas
        national = combined.values  # [T, F]
        extra = np.broadcast_to(national[:, np.newaxis, :], (T, G, F)).copy()

        logger.info(
            "Extra features tensor shape: %s (%d feature columns).", extra.shape, F
        )
        return extra.astype(np.float32)

    # ------------------------------------------------------------------
    # Population weights
    # ------------------------------------------------------------------

    def build_population_weights(
        self,
        dma_df: pd.DataFrame,
    ) -> np.ndarray:
        """Compute normalised population weights for each DMA.

        These weights are passed to Meridian's geo-weighted likelihood so
        that large-population DMAs have proportionally greater influence on
        the national-level posterior.

        Parameters
        ----------
        dma_df : pd.DataFrame
            DataFrame with at minimum columns ``["dma", "population"]``.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``[G]`` with values summing to 1.0.
        """
        required = {"dma", "population"}
        missing = required - set(dma_df.columns)
        if missing:
            raise ValueError(f"dma_df missing columns: {missing}")

        dma_df = dma_df.copy().set_index("dma")
        populations = dma_df.loc[
            [d for d in self.config.channels if d in dma_df.index], "population"
        ] if False else dma_df["population"]  # keep all rows; filter externally

        pop_array = populations.values.astype(np.float64)
        total = pop_array.sum()
        if total == 0.0:
            raise ValueError("Total population is zero; cannot compute weights.")

        weights = pop_array / total
        logger.info(
            "Population weights computed for %d DMAs; sum=%.6f.", len(weights), weights.sum()
        )
        return weights.astype(np.float32)

    # ------------------------------------------------------------------
    # Convenience: full pipeline
    # ------------------------------------------------------------------

    def build_all(
        self,
        shopify_df: pd.DataFrame,
        amazon_df: pd.DataFrame,
        spend_dfs: Dict[str, pd.DataFrame],
        promo_df: pd.DataFrame,
        trends_df: pd.DataFrame,
        bsr_df: pd.DataFrame,
        dma_df: pd.DataFrame,
        dma_list: List[str],
    ) -> Dict[str, np.ndarray]:
        """Build and return all tensors in a single call.

        Returns
        -------
        dict with keys: "kpi", "media", "extra_features", "population"
        """
        return {
            "kpi": self.build_kpi_tensor(shopify_df, amazon_df, dma_list),
            "media": self.build_media_tensor(spend_dfs, dma_list, self.config.channels),
            "extra_features": self.build_extra_features_tensor(
                promo_df, trends_df, bsr_df
            ),
            "population": self.build_population_weights(dma_df),
        }
