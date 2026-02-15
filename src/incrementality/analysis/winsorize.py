"""Winsorized analysis for outlier-robust incrementality estimation.

Winsorization replaces extreme values at specified percentiles with the
cutoff values rather than removing them entirely. This preserves sample
size while limiting the influence of outliers on causal estimates.

Haus runs results through both winsorized AND non-winsorized analysis
and compares. If results diverge substantially, it flags the test as
potentially driven by a few extreme observations.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def winsorize_series(
    values: np.ndarray,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> np.ndarray:
    """Winsorize a 1-D array at specified percentiles.

    Values below `lower_pct` are clipped to the lower cutoff; values above
    `upper_pct` are clipped to the upper cutoff. Unlike trimming, this
    preserves the array length.

    Args:
        values: 1-D numeric array.
        lower_pct: Lower percentile (0-1). Default 1st percentile.
        upper_pct: Upper percentile (0-1). Default 99th percentile.

    Returns:
        Winsorized copy of the array.
    """
    if len(values) == 0:
        return values.copy()

    lower_cutoff = np.percentile(values, lower_pct * 100)
    upper_cutoff = np.percentile(values, upper_pct * 100)

    clipped = np.clip(values, lower_cutoff, upper_cutoff)

    n_lower = int(np.sum(values < lower_cutoff))
    n_upper = int(np.sum(values > upper_cutoff))

    if n_lower > 0 or n_upper > 0:
        logger.debug(
            f"Winsorized {n_lower} low + {n_upper} high values "
            f"(cutoffs: [{lower_cutoff:.2f}, {upper_cutoff:.2f}])"
        )

    return clipped


def winsorize_panel(
    df: pd.DataFrame,
    revenue_col: str = "revenue",
    dma_col: str = "dma_code",
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """Winsorize revenue data at the DMA-day level.

    Computes per-DMA mean revenue, then winsorizes the per-DMA means
    to prevent extreme DMAs from dominating the analysis. The daily
    values within each DMA are scaled proportionally.

    Args:
        df: Panel data with [date, dma_code, revenue, ...].
        revenue_col: Column to winsorize.
        dma_col: DMA identifier column.
        lower_pct: Lower percentile cutoff.
        upper_pct: Upper percentile cutoff.

    Returns:
        DataFrame with winsorized revenue column.
    """
    result = df.copy()

    if revenue_col not in result.columns or len(result) == 0:
        return result

    # Per-DMA mean revenue
    dma_means = result.groupby(dma_col)[revenue_col].mean()

    if len(dma_means) < 5:
        # Too few DMAs to winsorize meaningfully
        return result

    # Winsorize the DMA-level means
    original_means = dma_means.values.copy()
    winsorized_means = winsorize_series(original_means, lower_pct, upper_pct)

    # Compute scaling factor per DMA, capping to prevent extreme amplification
    scale_map = {}
    for i, dma in enumerate(dma_means.index):
        if original_means[i] > 0:
            scale = winsorized_means[i] / original_means[i]
            # Cap scaling factor to prevent extreme amplification for
            # small-mean DMAs (e.g., mean=$1 winsorized to $100 = 100x)
            scale_map[dma] = max(0.1, min(scale, 10.0))
        else:
            scale_map[dma] = 1.0

    # Apply scaling
    result[revenue_col] = result.apply(
        lambda row: row[revenue_col] * scale_map.get(row[dma_col], 1.0),
        axis=1,
    )

    n_adjusted = sum(1 for s in scale_map.values() if abs(s - 1.0) > 1e-8)
    if n_adjusted > 0:
        logger.info(
            f"Winsorized {n_adjusted}/{len(dma_means)} DMAs "
            f"at [{lower_pct:.0%}, {upper_pct:.0%}] percentiles"
        )

    return result


def compare_winsorized_results(
    raw_lift: float,
    winsorized_lift: float,
    raw_p: float,
    winsorized_p: float,
    divergence_threshold: float = 0.30,
) -> dict:
    """Compare winsorized vs non-winsorized results.

    Returns a diagnostic dict with:
    - divergence: relative difference between the two lift estimates
    - is_outlier_driven: True if results diverge substantially
    - recommendation: human-readable assessment

    Args:
        raw_lift: Relative lift from non-winsorized analysis.
        winsorized_lift: Relative lift from winsorized analysis.
        raw_p: p-value from non-winsorized analysis.
        winsorized_p: p-value from winsorized analysis.
        divergence_threshold: Max acceptable relative divergence (default 30%).
    """
    # Relative divergence between the two estimates
    avg_lift = (abs(raw_lift) + abs(winsorized_lift)) / 2
    if avg_lift > 0.001:
        divergence = abs(raw_lift - winsorized_lift) / avg_lift
    else:
        divergence = 0.0

    is_outlier_driven = divergence > divergence_threshold

    # Check if significance flips
    sig_flip = (raw_p < 0.05) != (winsorized_p < 0.05)

    if is_outlier_driven and sig_flip:
        recommendation = (
            f"CAUTION: Winsorized and raw results diverge by {divergence:.0%} "
            f"and significance FLIPS. The result may be driven by extreme "
            f"outlier DMAs. Investigate outlier DMAs before making decisions."
        )
    elif is_outlier_driven:
        recommendation = (
            f"Moderate divergence ({divergence:.0%}) between winsorized and raw "
            f"results. Some outlier DMAs are influencing the estimate. Both "
            f"analyses agree on significance, so the direction is reliable."
        )
    else:
        recommendation = (
            f"Results are robust to winsorization (divergence: {divergence:.0%}). "
            f"Outliers are not driving the result."
        )

    return {
        "raw_lift": raw_lift,
        "winsorized_lift": winsorized_lift,
        "raw_p": raw_p,
        "winsorized_p": winsorized_p,
        "divergence": divergence,
        "is_outlier_driven": is_outlier_driven,
        "significance_flips": sig_flip,
        "recommendation": recommendation,
    }
