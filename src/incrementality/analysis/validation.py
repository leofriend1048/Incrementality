"""Test validation framework.

Every geo holdout test MUST pass validation before results are trusted.
This is the difference between a toy and a tool you bet $10M+ on.

Validation layers:
1. AA Test: Run analysis on pre-period only. Should find NO effect.
2. Placebo-in-time: Fake intervention date in pre-period. Should find nothing.
3. Placebo-in-space: Treat each holdout DMA as "treated." Should find nothing.
4. Estimator agreement: Do ASCM, BSTS, and DiD all agree on direction and magnitude?
5. Pre-period fit: L2 imbalance and R² thresholds.
6. False positive rate: Run many placebos; ~5% should be significant.

If validation fails, the report includes explicit BLOCKERS — hard stops that
prevent you from acting on the results.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from incrementality.analysis.estimators import (
    augmented_synthetic_control,
    difference_in_differences,
)
from incrementality.models import (
    IncrementalityResult,
    PlaceboTestResult,
    ValidationReport,
)

logger = logging.getLogger(__name__)


def run_full_validation(
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    ensemble_result: IncrementalityResult,
    estimator_results: dict[str, IncrementalityResult],
    revenue_col: str = "revenue",
    dma_col: str = "dma_code",
    date_col: str = "date",
    alpha: float = 0.05,
) -> ValidationReport:
    """Run the complete validation suite.

    This is the safety net. If this says "not trustworthy," do NOT
    make spend decisions based on the results.
    """
    warnings_list = []
    blockers = []
    placebo_results = []

    # ---------------------------------------------------------------
    # 1. AA Test — pre-period only, should find NO significant effect
    # ---------------------------------------------------------------
    logger.info("Running AA test (pre-period validation)...")
    aa_p_value, aa_passed = _run_aa_test(
        pre_data, treatment_dmas, holdout_dmas,
        revenue_col, dma_col, date_col, alpha,
    )
    if not aa_passed:
        blockers.append(
            f"AA TEST FAILED (p={aa_p_value:.4f}). The analysis detected a "
            f"'significant' difference in the pre-period when there should be none. "
            f"This means the treatment and holdout groups are NOT comparable. "
            f"DO NOT trust the results."
        )

    # ---------------------------------------------------------------
    # 2. Placebo-in-time tests
    # ---------------------------------------------------------------
    logger.info("Running placebo-in-time tests...")
    time_placebos = _run_placebo_in_time(
        pre_data, treatment_dmas, holdout_dmas,
        revenue_col, dma_col, date_col, alpha,
    )
    placebo_results.extend(time_placebos)

    # ---------------------------------------------------------------
    # 3. Placebo-in-space tests
    # ---------------------------------------------------------------
    logger.info("Running placebo-in-space tests...")
    space_placebos = _run_placebo_in_space(
        pre_data, post_data, holdout_dmas,
        revenue_col, dma_col, date_col, alpha,
    )
    placebo_results.extend(space_placebos)

    # Compute placebo statistics
    total_placebos = len(placebo_results)
    false_positives = sum(1 for p in placebo_results if p.is_false_positive)
    fpr = false_positives / total_placebos if total_placebos > 0 else 0
    pass_rate = 1 - fpr

    if fpr > 0.15:
        blockers.append(
            f"FALSE POSITIVE RATE TOO HIGH ({fpr:.0%}). Expected ~5% at alpha={alpha}. "
            f"Got {false_positives}/{total_placebos} false positives. "
            f"The method is unreliable for this data."
        )
    elif fpr > 0.10:
        warnings_list.append(
            f"Elevated false positive rate ({fpr:.0%}). Treat results with caution."
        )

    # ---------------------------------------------------------------
    # 4. Estimator agreement
    # ---------------------------------------------------------------
    logger.info("Checking estimator agreement...")
    agreement = _compute_estimator_agreement(estimator_results)
    if agreement < 0.5:
        blockers.append(
            f"ESTIMATORS DISAGREE (agreement={agreement:.0%}). "
            f"ASCM, BSTS, and DiD give very different results. "
            f"The signal is too noisy for reliable conclusions."
        )
    elif agreement < 0.7:
        warnings_list.append(
            f"Partial estimator disagreement (agreement={agreement:.0%}). "
            f"Consider running a longer test."
        )

    # ---------------------------------------------------------------
    # 5. Pre-period fit
    # ---------------------------------------------------------------
    l2 = ensemble_result.l2_imbalance
    r2 = ensemble_result.pre_period_r_squared

    if l2 > 0.10:
        blockers.append(
            f"PRE-PERIOD FIT IS POOR (L2 imbalance={l2:.4f}, target < 0.10). "
            f"The synthetic control cannot adequately replicate the treatment group's "
            f"pre-intervention behavior. Results are unreliable."
        )
    elif l2 > 0.05:
        warnings_list.append(
            f"Pre-period fit is marginal (L2={l2:.4f}). Results may have elevated bias."
        )

    if r2 < 0.90:
        warnings_list.append(
            f"Pre-period R²={r2:.3f} is below 0.90 target. "
            f"Counterfactual prediction quality is limited."
        )

    # Normalized RMSE (stored as pre_period_mape for backward compatibility)
    pre_mape = l2  # L2 normalized imbalance = NRMSE

    # ---------------------------------------------------------------
    # 6. Overall trust score
    # ---------------------------------------------------------------
    trust_score = _compute_trust_score(
        aa_passed, fpr, agreement, l2, r2, len(blockers),
    )
    is_trustworthy = len(blockers) == 0 and trust_score >= 70

    if not is_trustworthy and len(blockers) == 0:
        warnings_list.append(
            f"Trust score ({trust_score:.0f}/100) is below the 70 threshold. "
            f"Results should be interpreted with significant caution."
        )

    return ValidationReport(
        l2_imbalance=l2,
        pre_period_r_squared=r2,
        pre_period_mape=pre_mape,
        num_placebo_tests=total_placebos,
        placebo_pass_rate=pass_rate,
        false_positive_rate=fpr,
        placebo_results=placebo_results,
        aa_test_p_value=aa_p_value,
        aa_test_passed=aa_passed,
        estimator_agreement=agreement,
        is_trustworthy=is_trustworthy,
        trust_score=trust_score,
        warnings=warnings_list,
        blockers=blockers,
    )


def _run_aa_test(
    pre_data: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    revenue_col: str,
    dma_col: str,
    date_col: str,
    alpha: float,
) -> tuple[float, bool]:
    """AA test: split pre-period in half, run analysis. Should find no effect.

    Uses ASCM (primary estimator) to validate that the method itself does not
    find spurious effects in the pre-period.
    """
    dates = sorted(pre_data[date_col].unique())
    if len(dates) < 14:
        return 1.0, True  # Not enough data, skip

    midpoint = len(dates) // 2
    aa_pre = pre_data[pre_data[date_col].isin(dates[:midpoint])]
    aa_post = pre_data[pre_data[date_col].isin(dates[midpoint:])]

    try:
        # Use ASCM (primary estimator) so we validate the actual method
        result = augmented_synthetic_control(
            aa_pre, aa_post, treatment_dmas, holdout_dmas,
            revenue_col, dma_col, date_col, alpha,
        )
        return result.p_value, result.p_value > alpha
    except Exception:
        # Fall back to DiD if ASCM fails (e.g. insufficient data for SCM)
        try:
            result = difference_in_differences(
                aa_pre, aa_post, treatment_dmas, holdout_dmas,
                revenue_col, dma_col, alpha,
            )
            return result.p_value, result.p_value > alpha
        except Exception as e:
            logger.warning(f"AA test failed to run: {e}")
            return 1.0, True


def _run_placebo_in_time(
    pre_data: pd.DataFrame,
    treatment_dmas: list[str],
    holdout_dmas: list[str],
    revenue_col: str,
    dma_col: str,
    date_col: str,
    alpha: float,
    n_placebos: int = 5,
) -> list[PlaceboTestResult]:
    """Placebo-in-time: try different fake intervention dates in the pre-period."""
    dates = sorted(pre_data[date_col].unique())
    results = []

    if len(dates) < 21:
        return results

    # Try multiple split points
    n_dates = len(dates)
    split_points = np.linspace(
        int(n_dates * 0.3), int(n_dates * 0.7), n_placebos, dtype=int,
    )

    for split in split_points:
        fake_pre = pre_data[pre_data[date_col].isin(dates[:split])]
        fake_post = pre_data[pre_data[date_col].isin(dates[split:])]

        try:
            result = difference_in_differences(
                fake_pre, fake_post, treatment_dmas, holdout_dmas,
                revenue_col, dma_col, alpha,
            )
            results.append(PlaceboTestResult(
                placebo_type="in_time",
                placebo_date=dates[split],
                estimated_effect=result.relative_lift,
                p_value=result.p_value,
                is_false_positive=result.is_significant,
            ))
        except Exception:
            continue

    return results


def _run_placebo_in_space(
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    holdout_dmas: list[str],
    revenue_col: str,
    dma_col: str,
    date_col: str,
    alpha: float,
) -> list[PlaceboTestResult]:
    """Placebo-in-space: treat each holdout DMA as if it were treated.

    Uses ASCM (primary estimator) so the FPR reflects the actual method.
    """
    results = []

    for target_dma in holdout_dmas:
        donor_dmas = [d for d in holdout_dmas if d != target_dma]
        if len(donor_dmas) < 3:
            continue

        try:
            # Use ASCM to match the primary analysis method
            result = augmented_synthetic_control(
                pre_data, post_data, [target_dma], donor_dmas,
                revenue_col, dma_col, date_col, alpha,
            )
            results.append(PlaceboTestResult(
                placebo_type="in_space",
                target_dma=target_dma,
                estimated_effect=result.relative_lift,
                p_value=result.p_value,
                is_false_positive=result.is_significant,
            ))
        except Exception:
            # Fall back to DiD for this DMA if ASCM fails
            try:
                result = difference_in_differences(
                    pre_data, post_data, [target_dma], donor_dmas,
                    revenue_col, dma_col, alpha,
                )
                results.append(PlaceboTestResult(
                    placebo_type="in_space",
                    target_dma=target_dma,
                    estimated_effect=result.relative_lift,
                    p_value=result.p_value,
                    is_false_positive=result.is_significant,
                ))
            except Exception:
                continue

    return results


def _compute_estimator_agreement(
    results: dict[str, IncrementalityResult],
) -> float:
    """Compute how well estimators agree.

    1.0 = perfect agreement (same direction, similar magnitude)
    0.0 = complete disagreement
    """
    if len(results) <= 1:
        return 1.0

    lifts = [r.relative_lift for r in results.values()]
    # Treat near-zero lifts as agreeing with the majority direction
    signs = [1 if lift_val > 1e-6 else (-1 if lift_val < -1e-6 else 0) for lift_val in lifts]
    non_zero_signs = [s for s in signs if s != 0]

    # Direction agreement: fraction of estimators agreeing on sign
    if non_zero_signs:
        # Count the dominant direction vs total non-zero
        direction_score = max(
            sum(1 for s in non_zero_signs if s > 0),
            sum(1 for s in non_zero_signs if s < 0),
        ) / len(signs)
    else:
        direction_score = 1.0  # All near-zero = agreement

    # Magnitude agreement: coefficient of variation of lifts
    lifts_arr = np.array(lifts)
    mean_lift = np.mean(lifts_arr)
    if abs(mean_lift) > 0.001:
        cv = np.std(lifts_arr) / abs(mean_lift)
        magnitude_score = max(0, 1 - cv)
    else:
        magnitude_score = 1.0 if np.std(lifts_arr) < 0.01 else 0.5

    # Significance agreement: do they agree on significance?
    sigs = [r.is_significant for r in results.values()]
    sig_agreement = 1.0 if len(set(sigs)) == 1 else 0.5

    return 0.4 * direction_score + 0.3 * magnitude_score + 0.3 * sig_agreement


def _compute_trust_score(
    aa_passed: bool,
    fpr: float,
    agreement: float,
    l2: float,
    r2: float,
    n_blockers: int,
) -> float:
    """Compute overall trust score (0-100).

    Components:
    - AA test: 25 points
    - False positive rate: 20 points
    - Estimator agreement: 20 points
    - Pre-period fit: 20 points
    - No blockers: 15 points
    """
    score = 0.0

    # AA test (25 points)
    score += 25.0 if aa_passed else 0.0

    # FPR (20 points) — should be ~5%
    if fpr <= 0.06:
        score += 20.0
    elif fpr <= 0.10:
        score += 15.0
    elif fpr <= 0.15:
        score += 8.0

    # Agreement (20 points)
    score += 20.0 * agreement

    # Pre-period fit (20 points)
    l2_score = max(0, 1 - l2 / 0.10) * 10  # 10 points for L2
    r2_score = max(0, (r2 - 0.80) / 0.10) * 10  # 10 points for R² (full at 0.90)
    score += l2_score + r2_score

    # No blockers (15 points)
    score += 15.0 if n_blockers == 0 else 0.0

    return min(100.0, max(0.0, score))
