"""DMA matching and cell assignment for geo holdout tests.

Implements covariate-balanced matching to create treatment and holdout groups
that are statistically equivalent on observable characteristics. This is
critical for causal inference — unbalanced groups lead to biased estimates.

Methods:
1. Mahalanobis distance matching with optimal assignment
2. Re-randomization with balance constraints
3. Stratified randomization by region

Balance is checked via standardized mean differences (SMD) across covariates.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from incrementality.models import CellType, DMAHistoricalMetrics, TestCell

logger = logging.getLogger(__name__)

# Covariates used for matching (column names in the metrics DataFrame)
_MATCHING_COVARIATES = [
    "total_revenue",
    "total_orders",
    "revenue_per_capita",
    "total_ad_spend",
    "revenue_trend",
    "revenue_volatility",
]


def prepare_matching_data(
    dma_metrics: list[DMAHistoricalMetrics],
    dma_populations: dict[str, int],
) -> pd.DataFrame:
    """Prepare DMA metrics into a DataFrame suitable for matching.

    Normalizes covariates and adds population data.
    """
    records = []
    for m in dma_metrics:
        pop = dma_populations.get(m.dma_code, 1)
        records.append({
            "dma_code": m.dma_code,
            "total_revenue": m.total_revenue,
            "total_orders": m.total_orders,
            "revenue_per_capita": m.total_revenue / pop if pop > 0 else 0,
            "total_ad_spend": m.total_ad_spend,
            "revenue_trend": m.revenue_trend,
            "revenue_volatility": m.revenue_volatility,
            "population": pop,
            "shopify_revenue": m.shopify_revenue,
            "amazon_revenue": m.amazon_revenue,
        })
    return pd.DataFrame(records)


def compute_balance_score(
    treatment_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    covariates: list[str] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute balance between treatment and holdout groups.

    Uses Standardized Mean Difference (SMD) for each covariate.
    SMD < 0.10 is considered well-balanced (Rosenbaum & Rubin, 1985).
    SMD < 0.25 is acceptable for geo tests.

    Returns:
        (overall_score, {covariate: smd})
        overall_score: 1.0 = perfect balance, 0.0 = terrible balance
    """
    covariates = covariates or _MATCHING_COVARIATES
    smds = {}

    for cov in covariates:
        if cov not in treatment_df.columns or cov not in holdout_df.columns:
            continue
        t_mean = treatment_df[cov].mean()
        h_mean = holdout_df[cov].mean()
        t_var = treatment_df[cov].var()
        h_var = holdout_df[cov].var()
        pooled_sd = math.sqrt((t_var + h_var) / 2)
        if pooled_sd > 0:
            smd = abs(t_mean - h_mean) / pooled_sd
        else:
            smd = 0.0
        smds[cov] = smd

    if not smds:
        return 0.0, {}

    # Overall score: transform max SMD to 0-1 scale
    max_smd = max(smds.values())
    # Score: 1.0 if max_smd=0, decays toward 0 as max_smd increases
    score = max(0.0, 1.0 - max_smd / 0.5)

    return score, smds


def match_dmas_mahalanobis(
    df: pd.DataFrame,
    n_holdout: int,
    covariates: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Assign DMAs to treatment/holdout using Mahalanobis distance pairing.

    Strategy:
    1. Compute pairwise Mahalanobis distances between DMAs
    2. Use optimal pairing to create matched pairs
    3. Within each pair, randomly assign one to treatment, one to holdout
    4. For remaining unpaired DMAs (if odd count), assign to treatment

    This ensures treatment and holdout are balanced on covariates.
    """
    covariates = covariates or _MATCHING_COVARIATES
    available_covs = [c for c in covariates if c in df.columns]
    if not available_covs:
        raise ValueError("No matching covariates found in data")

    X = df[available_covs].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_total = len(df)
    n_treatment = n_total - n_holdout

    if n_holdout >= n_total:
        raise ValueError(
            f"Cannot assign {n_holdout} holdout DMAs from {n_total} total"
        )

    # Compute Mahalanobis distance matrix
    # Use covariance of scaled data (should be ~identity but accounts for correlations)
    cov_matrix = np.cov(X_scaled.T)
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix]])
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_matrix)

    dist_matrix = cdist(X_scaled, X_scaled, metric="mahalanobis", VI=cov_inv)

    # Greedy pair matching
    paired_treatment = []
    paired_holdout = []
    used = set()

    # Sort all pairs by distance
    n = len(dist_matrix)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((dist_matrix[i, j], i, j))
    pairs.sort(key=lambda x: x[0])

    n_pairs = min(n_holdout, n_treatment)
    for _, i, j in pairs:
        if len(paired_holdout) >= n_pairs:
            break
        if i in used or j in used:
            continue
        # Randomly assign within pair
        if np.random.random() < 0.5:
            paired_holdout.append(i)
            paired_treatment.append(j)
        else:
            paired_treatment.append(i)
            paired_holdout.append(j)
        used.add(i)
        used.add(j)

    # Remaining DMAs go to treatment (larger group)
    for idx in range(n):
        if idx not in used:
            paired_treatment.append(idx)

    treatment_codes = df.iloc[paired_treatment]["dma_code"].tolist()
    holdout_codes = df.iloc[paired_holdout]["dma_code"].tolist()

    return treatment_codes, holdout_codes


def match_dmas_rerandomization(
    df: pd.DataFrame,
    n_holdout: int,
    n_iterations: int = 10000,
    balance_threshold: float = 0.25,
    covariates: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Assign DMAs using re-randomization with balance constraints.

    Strategy:
    1. Repeatedly randomize assignment
    2. Check balance (SMD) after each randomization
    3. Accept the assignment with the best balance score
    4. Reject assignments where max SMD > threshold

    This is the gold standard for small-sample experiments (Morgan & Rubin, 2012).
    """
    covariates = covariates or _MATCHING_COVARIATES
    available_covs = [c for c in covariates if c in df.columns]

    n_total = len(df)
    best_score = -1.0
    best_treatment = None
    best_holdout = None
    indices = np.arange(n_total)

    # Track best balanced (passes threshold) and best overall (fallback)
    best_balanced_score = -1.0
    best_balanced_treatment = None
    best_balanced_holdout = None

    for _ in range(n_iterations):
        np.random.shuffle(indices)
        holdout_idx = indices[:n_holdout]
        treatment_idx = indices[n_holdout:]

        holdout_df = df.iloc[holdout_idx]
        treatment_df = df.iloc[treatment_idx]

        score, smds = compute_balance_score(treatment_df, holdout_df, available_covs)

        # Reject assignments where any SMD exceeds the threshold
        # (Morgan & Rubin 2012: only accept balanced randomizations)
        if smds and max(smds.values()) > balance_threshold:
            # Track as fallback in case no assignment passes the threshold
            if score > best_score:
                best_score = score
                best_treatment = treatment_df["dma_code"].tolist()
                best_holdout = holdout_df["dma_code"].tolist()
            continue

        # This assignment passes the balance threshold — track separately
        if score > best_balanced_score:
            best_balanced_score = score
            best_balanced_treatment = treatment_df["dma_code"].tolist()
            best_balanced_holdout = holdout_df["dma_code"].tolist()

    # Prefer balanced assignments; fall back to best overall if none passed
    if best_balanced_treatment is not None:
        best_treatment = best_balanced_treatment
        best_holdout = best_balanced_holdout
        best_score = best_balanced_score
    elif best_treatment is not None:
        logger.warning(
            f"No assignment passed balance threshold ({balance_threshold}) "
            f"in {n_iterations} iterations. Using best available (score={best_score:.3f})."
        )

    if best_treatment is None or best_holdout is None:
        raise ValueError("Failed to find a balanced assignment")

    logger.info(f"Best balance score: {best_score:.3f} after re-randomization")
    return best_treatment, best_holdout


def match_dmas_stratified(
    df: pd.DataFrame,
    n_holdout: int,
    strata_col: str = "region",
    dma_regions: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Assign DMAs using stratified randomization by region.

    Ensures proportional representation of each region in both cells.
    Simpler than Mahalanobis but guarantees geographic balance.
    """
    if strata_col not in df.columns and dma_regions:
        df = df.copy()
        df[strata_col] = df["dma_code"].map(dma_regions)

    if strata_col not in df.columns:
        # Fall back to simple random
        return match_dmas_rerandomization(df, n_holdout)

    n_total = len(df)
    holdout_fraction = n_holdout / n_total

    treatment_codes = []
    holdout_codes = []

    for stratum, group in df.groupby(strata_col):
        n_stratum = len(group)
        n_holdout_stratum = max(1, round(n_stratum * holdout_fraction))
        n_holdout_stratum = min(n_holdout_stratum, n_stratum - 1)

        shuffled = group.sample(frac=1.0)
        holdout_codes.extend(shuffled.iloc[:n_holdout_stratum]["dma_code"].tolist())
        treatment_codes.extend(shuffled.iloc[n_holdout_stratum:]["dma_code"].tolist())

    return treatment_codes, holdout_codes


def build_test_cells(
    df: pd.DataFrame,
    treatment_codes: list[str],
    holdout_codes: list[str],
) -> tuple[TestCell, TestCell]:
    """Build TestCell objects from assignment results."""
    t_df = df[df["dma_code"].isin(treatment_codes)]
    h_df = df[df["dma_code"].isin(holdout_codes)]

    treatment_cell = TestCell(
        cell_type=CellType.TREATMENT,
        dma_codes=treatment_codes,
        total_population=int(t_df["population"].sum()) if "population" in t_df else 0,
        historical_revenue=float(t_df["total_revenue"].sum()),
        historical_orders=int(t_df["total_orders"].sum()),
    )

    holdout_cell = TestCell(
        cell_type=CellType.HOLDOUT,
        dma_codes=holdout_codes,
        total_population=int(h_df["population"].sum()) if "population" in h_df else 0,
        historical_revenue=float(h_df["total_revenue"].sum()),
        historical_orders=int(h_df["total_orders"].sum()),
    )

    return treatment_cell, holdout_cell


def validate_balance(
    df: pd.DataFrame,
    treatment_codes: list[str],
    holdout_codes: list[str],
    tolerance: float = 0.25,
) -> tuple[bool, float, dict[str, float]]:
    """Validate that the assignment is adequately balanced.

    Returns (is_balanced, overall_score, covariate_smds).
    """
    t_df = df[df["dma_code"].isin(treatment_codes)]
    h_df = df[df["dma_code"].isin(holdout_codes)]
    score, smds = compute_balance_score(t_df, h_df)

    is_balanced = all(smd <= tolerance for smd in smds.values())
    return is_balanced, score, smds
