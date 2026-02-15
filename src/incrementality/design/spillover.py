"""Geographic spillover mitigation for geo holdout tests.

Spillover = when the treatment effect "leaks" from treatment DMAs to
adjacent holdout DMAs. This biases the estimate toward zero (you
underestimate the true effect) because your holdout isn't truly clean.

Haus uses proprietary GPS-derived "Commuting Zones" to handle this.
We use polygon-based DMA boundary adjacency (via GeoPandas) as primary,
with centroid-distance fallback:

1. Build adjacency from actual DMA boundary polygons (GeoJSON)
2. Fall back to centroid-distance computation for all 210 DMAs
3. Identify treatment DMAs that border holdout DMAs
4. Option A: Exclude border DMAs from analysis (buffer zone)
5. Option B: Flag border pairs and adjust standard errors
6. Score the design by the fraction of "clean" holdout DMAs

This is a critical concern for brands with physical retail, regional
delivery, and any product where customers near DMA borders might be
exposed to ads in the adjacent DMA.
"""

from __future__ import annotations

import logging

import numpy as np

from incrementality.design.dma_boundaries import (
    get_adjacency,
    compute_adjacency_from_centroids,
    _ALL_DMA_CENTROIDS,
)

logger = logging.getLogger(__name__)


# =====================================================================
# DMA adjacency — loaded from polygon boundaries or computed on the fly
# =====================================================================

_ADJACENCY_CACHE: dict[str, list[str]] | None = None


def _get_dma_adjacency() -> dict[str, list[str]]:
    """Get the DMA adjacency map, using the best available source.

    Priority:
    1. Polygon-based adjacency from cached GeoJSON computation
    2. Centroid-distance-based adjacency for all 210 DMAs
    """
    global _ADJACENCY_CACHE

    if _ADJACENCY_CACHE is not None:
        return _ADJACENCY_CACHE

    # Try loading polygon-based adjacency
    adjacency = get_adjacency()
    if adjacency is not None and len(adjacency) > 50:
        logger.info(f"Using polygon-based DMA adjacency ({len(adjacency)} DMAs)")
        _ADJACENCY_CACHE = adjacency
        return _ADJACENCY_CACHE

    # Fall back to centroid-distance computation
    logger.info("Computing DMA adjacency from centroids (polygon boundaries not available)")
    _ADJACENCY_CACHE = compute_adjacency_from_centroids(max_distance_miles=175.0)
    logger.info(f"Computed centroid-based adjacency for {len(_ADJACENCY_CACHE)} DMAs")
    return _ADJACENCY_CACHE


def get_adjacent_dmas(dma_code: str) -> list[str]:
    """Get DMAs adjacent to a given DMA."""
    adjacency = _get_dma_adjacency()
    return adjacency.get(dma_code, [])


def compute_spillover_risk(
    treatment_dmas: list[str],
    holdout_dmas: list[str],
) -> dict:
    """Assess spillover risk between treatment and holdout groups.

    Returns a dict with:
    - border_pairs: list of (treatment_dma, holdout_dma) that share borders
    - risk_score: 0.0 (no risk) to 1.0 (maximum risk)
    - clean_holdout_fraction: fraction of holdout DMAs NOT bordering treatment
    - recommendations: list of suggested actions
    """
    treatment_set = set(treatment_dmas)
    holdout_set = set(holdout_dmas)

    border_pairs = []
    contaminated_holdout = set()
    contaminated_treatment = set()

    for t_dma in treatment_dmas:
        neighbors = get_adjacent_dmas(t_dma)
        for neighbor in neighbors:
            if neighbor in holdout_set:
                border_pairs.append((t_dma, neighbor))
                contaminated_holdout.add(neighbor)
                contaminated_treatment.add(t_dma)

    n_contaminated = len(contaminated_holdout)
    n_holdout = len(holdout_dmas)
    clean_fraction = 1.0 - (n_contaminated / n_holdout) if n_holdout > 0 else 1.0

    # Risk score: higher is worse
    risk_score = n_contaminated / n_holdout if n_holdout > 0 else 0.0

    recommendations = []
    if risk_score > 0.3:
        recommendations.append(
            f"HIGH SPILLOVER RISK: {n_contaminated}/{n_holdout} holdout DMAs "
            f"border treatment DMAs. Consider applying geographic buffer."
        )
    elif risk_score > 0.1:
        recommendations.append(
            f"MODERATE SPILLOVER RISK: {n_contaminated}/{n_holdout} holdout DMAs "
            f"border treatment DMAs. Results may underestimate true effect."
        )
    elif n_contaminated > 0:
        recommendations.append(
            f"LOW SPILLOVER RISK: {n_contaminated}/{n_holdout} holdout DMAs "
            f"border treatment DMAs. Impact likely minimal."
        )

    return {
        "border_pairs": border_pairs,
        "contaminated_holdout_dmas": list(contaminated_holdout),
        "contaminated_treatment_dmas": list(contaminated_treatment),
        "risk_score": risk_score,
        "clean_holdout_fraction": clean_fraction,
        "recommendations": recommendations,
    }


def apply_geographic_buffer(
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    min_holdout: int = 5,
) -> tuple[list[str], list[str], list[str]]:
    """Apply geographic buffer by removing border DMAs from analysis.

    DMAs that border between treatment and holdout are moved to a
    "buffer zone" and excluded from both groups during analysis.

    Returns: (filtered_treatment, filtered_holdout, buffer_dmas)

    The buffer DMAs are still part of the test execution (treatment DMAs
    in the buffer still get ads, holdout DMAs still don't), but they are
    excluded from the statistical analysis to reduce bias from spillover.
    """
    spillover = compute_spillover_risk(treatment_dmas, holdout_dmas)
    contaminated_holdout = set(spillover["contaminated_holdout_dmas"])
    contaminated_treatment = set(spillover["contaminated_treatment_dmas"])

    buffer_dmas = []
    filtered_holdout = holdout_dmas.copy()
    filtered_treatment = treatment_dmas.copy()

    if spillover["risk_score"] > 0.0:
        # Remove contaminated holdout DMAs
        candidate_holdout = [d for d in holdout_dmas if d not in contaminated_holdout]
        # Also remove contaminated treatment DMAs (border DMAs on both sides)
        candidate_treatment = [d for d in treatment_dmas if d not in contaminated_treatment]

        if len(candidate_holdout) >= min_holdout:
            buffer_dmas = list(contaminated_holdout | contaminated_treatment)
            filtered_holdout = candidate_holdout
            filtered_treatment = candidate_treatment
            logger.info(
                f"Geographic buffer: removed {len(contaminated_holdout)} holdout + "
                f"{len(contaminated_treatment)} treatment border DMAs from analysis "
                f"({len(filtered_holdout)} holdout, {len(filtered_treatment)} treatment remain)"
            )
        else:
            # Not enough holdout DMAs left after buffering
            contamination_count = {}
            for h_dma in contaminated_holdout:
                count = sum(
                    1 for t_dma in treatment_dmas
                    if h_dma in get_adjacent_dmas(t_dma)
                )
                contamination_count[h_dma] = count

            sorted_contaminated = sorted(
                contamination_count.items(), key=lambda x: -x[1]
            )
            n_can_remove = len(holdout_dmas) - min_holdout
            if n_can_remove > 0:
                to_remove = [d for d, _ in sorted_contaminated[:n_can_remove]]
                buffer_dmas = to_remove
                filtered_holdout = [d for d in holdout_dmas if d not in to_remove]
                logger.info(
                    f"Partial geographic buffer: removed {len(to_remove)} of "
                    f"{len(contaminated_holdout)} contaminated holdout DMAs"
                )
            else:
                logger.warning(
                    f"Cannot apply geographic buffer: would leave fewer than "
                    f"{min_holdout} holdout DMAs. Proceeding without buffer."
                )

    return filtered_treatment, filtered_holdout, buffer_dmas


def adjust_for_spillover(
    estimated_effect: float,
    spillover_risk: dict,
) -> float:
    """Adjust treatment effect estimate for potential spillover attenuation.

    When holdout DMAs are contaminated by treatment spillover, the measured
    effect is biased toward zero (holdout revenue is higher than it would
    be without spillover, making the gap look smaller).

    This is a simple scaling adjustment:
        adjusted_effect = measured_effect / (1 - contamination_rate * decay)

    where decay represents how much of the treatment effect spills over
    (typically 10-30% for adjacent DMAs based on the literature).
    """
    clean_fraction = spillover_risk["clean_holdout_fraction"]

    if clean_fraction >= 1.0:
        return estimated_effect

    # Assume 20% of treatment effect spills to adjacent holdout DMAs
    spillover_decay = 0.20
    contamination_rate = 1.0 - clean_fraction
    adjustment = 1.0 / (1.0 - contamination_rate * spillover_decay)

    adjusted = estimated_effect * adjustment
    logger.info(
        f"Spillover adjustment: {estimated_effect:.4f} -> {adjusted:.4f} "
        f"(contamination={contamination_rate:.0%}, decay={spillover_decay:.0%})"
    )

    return adjusted
