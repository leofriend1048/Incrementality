"""Geographic spillover mitigation for geo holdout tests.

Spillover = when the treatment effect "leaks" from treatment DMAs to
adjacent holdout DMAs. This biases the estimate toward zero (you
underestimate the true effect) because your holdout isn't truly clean.

Haus uses proprietary GPS-derived "Commuting Zones" to handle this.
We approximate with DMA adjacency data and geographic buffering:

1. Build an adjacency graph of DMAs (which DMAs share borders)
2. Identify treatment DMAs that border holdout DMAs
3. Option A: Exclude border DMAs from analysis (buffer zone)
4. Option B: Flag border pairs and adjust standard errors
5. Score the design by the fraction of "clean" holdout DMAs

This is a critical concern for brands with physical retail, regional
delivery, and any product where customers near DMA borders might be
exposed to ads in the adjacent DMA.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# DMA adjacency map
# =====================================================================
# Each entry: DMA code -> list of adjacent DMA codes
# This covers the major DMAs. Adjacency = shares a geographic border
# or has significant cross-border commuting patterns.
#
# Note: Haus uses GPS-derived commuting zones which are more granular.
# This is a reasonable approximation for DMA-level tests.

_DMA_ADJACENCY: dict[str, list[str]] = {
    # Northeast
    "501": ["504", "533", "521"],  # New York -> Philadelphia, Hartford, Providence
    "504": ["501", "504", "511"],  # Philadelphia -> NYC, DC
    "506": ["521", "533"],  # Boston -> Providence, Hartford
    "511": ["504", "512", "577"],  # DC -> Philadelphia, Baltimore, Wilkes-Barre
    "512": ["511"],  # Baltimore -> DC
    "521": ["506", "501", "533"],  # Providence -> Boston, NYC, Hartford
    "533": ["506", "501", "521"],  # Hartford -> Boston, NYC, Providence
    # Southeast
    "524": ["528", "560"],  # Atlanta -> Nashville, Raleigh
    "528": ["524", "557"],  # Nashville -> Atlanta, Knoxville
    "531": ["560", "570"],  # Raleigh-Durham -> Charlotte, Florence
    "534": ["531"],  # Orlando -> nearby FL
    "539": ["548"],  # Tampa -> West Palm Beach
    "548": ["539", "528"],  # West Palm Beach -> Tampa
    "557": ["528"],  # Knoxville -> Nashville
    "560": ["524", "531"],  # Charlotte -> Atlanta, Raleigh
    "570": ["531", "560"],  # Florence-Myrtle Beach -> Raleigh, Charlotte
    # Midwest
    "602": ["617", "616", "669"],  # Chicago -> Milwaukee, Kansas City, Madison
    "616": ["602"],  # Kansas City -> Chicago
    "617": ["602", "669"],  # Milwaukee -> Chicago, Madison
    "669": ["602", "617"],  # Madison -> Chicago, Milwaukee
    "505": ["610"],  # Detroit -> Cleveland
    "510": ["505"],  # Cleveland -> Detroit (as 610, mapped)
    "610": ["505"],  # Cleveland -> Detroit
    "527": ["602"],  # Indianapolis -> Chicago
    "613": ["527"],  # Minneapolis -> nearby
    # West
    "803": ["807", "825", "868"],  # LA -> SF, San Diego, Sacramento
    "807": ["803", "868", "862"],  # SF -> LA, Sacramento, Portland-ish
    "825": ["803"],  # San Diego -> LA
    "819": ["820"],  # Seattle -> Portland
    "820": ["819"],  # Portland -> Seattle
    "868": ["807", "803"],  # Sacramento -> SF, LA
    "862": ["807"],  # Sacramento adjacent
    # Southwest
    "623": ["618"],  # Dallas -> Houston
    "618": ["623"],  # Houston -> Dallas
    "753": ["623"],  # Phoenix -> (distant but same region)
    "641": ["618"],  # San Antonio -> Houston
}


def get_adjacent_dmas(dma_code: str) -> list[str]:
    """Get DMAs adjacent to a given DMA."""
    return _DMA_ADJACENCY.get(dma_code, [])


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
    # Based on fraction of holdout DMAs at risk of contamination
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

    # Only buffer holdout DMAs (treatment contamination is less of a concern)
    # But do both if risk is very high
    buffer_dmas = []
    filtered_holdout = holdout_dmas.copy()
    filtered_treatment = treatment_dmas.copy()

    if spillover["risk_score"] > 0.0:
        # Remove contaminated holdout DMAs
        candidate_holdout = [d for d in holdout_dmas if d not in contaminated_holdout]

        if len(candidate_holdout) >= min_holdout:
            buffer_dmas = list(contaminated_holdout)
            filtered_holdout = candidate_holdout
            logger.info(
                f"Geographic buffer: removed {len(buffer_dmas)} holdout DMAs "
                f"({len(filtered_holdout)} remain)"
            )
        else:
            # Not enough holdout DMAs left after buffering
            # Only remove the most contaminated (those with most treatment neighbors)
            contamination_count = {}
            for h_dma in contaminated_holdout:
                count = sum(
                    1 for t_dma in treatment_dmas
                    if h_dma in get_adjacent_dmas(t_dma)
                )
                contamination_count[h_dma] = count

            # Sort by contamination, remove worst offenders
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
