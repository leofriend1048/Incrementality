"""Budget optimizer for Michael Todd Beauty MMM platform.

Uses CVXPY (with scipy fallback) to solve a constrained revenue-maximization
problem over the saturation curves estimated by the Meridian model.  Channel
response functions are parametrized as Hill curves fitted to the posterior
mean saturation points.

Typical usage
-------------
>>> from incrementality.mmm.optimizer import BudgetOptimizer
>>> curves = {
...     "meta_perf": {
...         "shopify": (spend_pts, rev_pts),
...         "amazon":  (spend_pts, rev_pts),
...     },
...     ...
... }
>>> opt = BudgetOptimizer(saturation_curves=curves, config={})
>>> result = opt.optimize(total_budget=500_000)
>>> print(result.allocation)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import curve_fit, minimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import CVXPY; fall back gracefully
# ---------------------------------------------------------------------------
try:
    import cvxpy as cp  # type: ignore

    _CVXPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CVXPY_AVAILABLE = False
    logger.warning(
        "cvxpy is not installed.  BudgetOptimizer will fall back to "
        "scipy.optimize.minimize (L-BFGS-B).  Install cvxpy for exact "
        "convex-program guarantees: pip install cvxpy"
    )

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
SpendRevPair = Tuple[np.ndarray, np.ndarray]
SaturationCurves = Dict[str, Dict[str, SpendRevPair]]  # channel → outcome → (x, y)


# ---------------------------------------------------------------------------
# Hill curve helpers
# ---------------------------------------------------------------------------

def _hill(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """Hill saturation function: f(x) = x^alpha / (x^alpha + gamma^alpha).

    Parameters
    ----------
    x : array_like
        Spend values (must be non-negative).
    alpha : float
        Shape parameter (> 0).  Controls the curvature / slope.
    gamma : float
        Half-saturation point (spend where f = 0.5).

    Returns
    -------
    np.ndarray
        Saturation values in [0, 1].
    """
    xa = np.power(np.maximum(x, 0.0), alpha)
    ga = np.power(gamma, alpha)
    return xa / (xa + ga)


def _fit_hill(spend_pts: np.ndarray, rev_pts: np.ndarray) -> Tuple[float, float, float]:
    """Fit a scaled Hill curve to (spend, revenue) data points.

    Returns (scale, alpha, gamma) where revenue ≈ scale * Hill(spend; alpha, gamma).

    Falls back to safe defaults if curve_fit fails.
    """
    spend_pts = np.asarray(spend_pts, dtype=float)
    rev_pts = np.asarray(rev_pts, dtype=float)

    if spend_pts.size == 0 or rev_pts.size == 0:
        return 1.0, 1.0, 1.0

    # Scale parameter: peak revenue in the provided data
    scale_init = float(np.max(rev_pts)) if np.max(rev_pts) > 0 else 1.0
    gamma_init = float(np.median(spend_pts)) if np.median(spend_pts) > 0 else 1.0

    def _scaled_hill(x: np.ndarray, scale: float, alpha: float, gamma: float) -> np.ndarray:
        return scale * _hill(x, alpha, gamma)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _scaled_hill,
                spend_pts,
                rev_pts,
                p0=[scale_init, 1.0, gamma_init],
                bounds=([0.0, 0.1, 1e-6], [np.inf, 5.0, np.inf]),
                maxfev=10_000,
            )
        scale, alpha, gamma = float(popt[0]), float(popt[1]), float(popt[2])
    except RuntimeError:
        logger.debug("Hill curve_fit failed; using linear approximation defaults.")
        # Linear approximation: revenue ≈ (max_rev / max_spend) * spend
        max_spend = float(np.max(spend_pts)) if np.max(spend_pts) > 0 else 1.0
        scale = scale_init
        alpha = 1.0
        gamma = max_spend  # half-saturation at the max spend point
    return scale, alpha, gamma


class _HillCurve:
    """Fitted Hill saturation curve for a single (channel, outcome) pair."""

    __slots__ = ("scale", "alpha", "gamma", "_spend_max", "_rev_at_max")

    def __init__(self, spend_pts: np.ndarray, rev_pts: np.ndarray) -> None:
        self.scale, self.alpha, self.gamma = _fit_hill(spend_pts, rev_pts)
        self._spend_max = float(np.max(spend_pts)) if len(spend_pts) > 0 else 1e6
        self._rev_at_max = float(np.max(rev_pts)) if len(rev_pts) > 0 else 0.0

    def revenue(self, spend: float) -> float:
        """Expected revenue at *spend*."""
        return float(self.scale * _hill(np.array([spend]), self.alpha, self.gamma)[0])

    def marginal_roi(self, spend: float, delta: float = 1.0) -> float:
        """Marginal revenue per incremental dollar at *spend*."""
        if spend < 0:
            spend = 0.0
        r1 = self.revenue(spend + delta / 2)
        r0 = self.revenue(max(0.0, spend - delta / 2))
        return (r1 - r0) / delta

    # -----------------------------------------------------------------------
    # CVXPY-compatible piecewise-linear approximation
    # -----------------------------------------------------------------------
    def pwl_breakpoints(self, n_segments: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x_breaks, y_breaks) for a piecewise-linear approximation."""
        x_max = max(self._spend_max * 1.5, 1.0)
        xs = np.linspace(0.0, x_max, n_segments + 1)
        ys = np.array([self.revenue(x) for x in xs])
        return xs, ys


# ---------------------------------------------------------------------------
# Pydantic result model
# ---------------------------------------------------------------------------

class OptimizationResult(BaseModel):
    """Output from BudgetOptimizer.optimize()."""

    allocation: Dict[str, float] = Field(
        description="Optimal spend per channel (dollars)."
    )
    expected_shopify_revenue: float = Field(
        description="Expected Shopify revenue at the optimal allocation."
    )
    expected_amazon_revenue: float = Field(
        description="Expected Amazon revenue at the optimal allocation."
    )
    expected_total_revenue: float = Field(
        description="Blended expected total revenue."
    )
    blended_roas: float = Field(
        description="Expected total revenue divided by total spend."
    )
    allocation_lower: Dict[str, float] = Field(
        description="Lower bound on channel spend (−1 posterior SD).",
        default_factory=dict,
    )
    allocation_upper: Dict[str, float] = Field(
        description="Upper bound on channel spend (+1 posterior SD).",
        default_factory=dict,
    )
    solver_status: str = Field(
        default="optimal",
        description="Solver status string (e.g. 'optimal', 'scipy-L-BFGS-B').",
    )
    total_budget: float = Field(description="Total budget constraint used.")
    shopify_weight: float = Field(default=0.7)
    amazon_weight: float = Field(default=0.3)


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------

class BudgetOptimizer:
    """Constrained budget optimizer for Michael Todd Beauty channels.

    Maximises the blended expected revenue across Shopify and Amazon outcomes
    subject to a total budget cap, per-channel concentration limits, and
    optional floor (minimum) constraints.

    Parameters
    ----------
    saturation_curves : dict
        Mapping ``{channel: {"shopify": (spend_pts, rev_pts),
                              "amazon":  (spend_pts, rev_pts)}}``.
    config : dict
        Optional configuration overrides (unused keys are silently ignored).
    """

    _DEFAULT_N_PWL_SEGMENTS = 25  # Piecewise-linear segments for CVXPY

    def __init__(self, saturation_curves: SaturationCurves, config: Dict[str, Any]) -> None:
        self.config = config
        self.channels: List[str] = sorted(saturation_curves.keys())
        self._curves: Dict[str, Dict[str, _HillCurve]] = {}

        for channel, outcomes in saturation_curves.items():
            self._curves[channel] = {}
            for outcome, (spend_pts, rev_pts) in outcomes.items():
                sp = np.asarray(spend_pts, dtype=float)
                rp = np.asarray(rev_pts, dtype=float)
                self._curves[channel][outcome] = _HillCurve(sp, rp)

        logger.info(
            "BudgetOptimizer initialised: %d channels, CVXPY=%s",
            len(self.channels),
            _CVXPY_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Revenue helpers
    # ------------------------------------------------------------------

    def _channel_revenue(
        self, channel: str, spend: float, outcome: str
    ) -> float:
        curve = self._curves[channel].get(outcome)
        if curve is None:
            return 0.0
        return curve.revenue(spend)

    def _total_blended_revenue(
        self,
        allocation: Dict[str, float],
        shopify_weight: float,
        amazon_weight: float,
    ) -> Tuple[float, float, float]:
        """Return (shopify_rev, amazon_rev, blended_rev)."""
        shopify_rev = sum(
            self._channel_revenue(ch, alloc, "shopify")
            for ch, alloc in allocation.items()
        )
        amazon_rev = sum(
            self._channel_revenue(ch, alloc, "amazon")
            for ch, alloc in allocation.items()
        )
        blended = shopify_weight * shopify_rev + amazon_weight * amazon_rev
        return shopify_rev, amazon_rev, blended

    # ------------------------------------------------------------------
    # CVXPY optimisation
    # ------------------------------------------------------------------

    def _optimize_cvxpy(
        self,
        total_budget: float,
        shopify_weight: float,
        amazon_weight: float,
        min_floors: Dict[str, float],
        max_concentration: float,
    ) -> Tuple[Dict[str, float], str]:
        """Solve via CVXPY piecewise-linear concave program."""
        n = len(self.channels)
        x = cp.Variable(n, nonneg=True, name="spend")

        # Build piecewise-linear revenue functions for each channel/outcome
        objective_terms = []
        for i, ch in enumerate(self.channels):
            for outcome, weight in [("shopify", shopify_weight), ("amazon", amazon_weight)]:
                curve = self._curves[ch].get(outcome)
                if curve is None:
                    continue
                xs_bp, ys_bp = curve.pwl_breakpoints(self._DEFAULT_N_PWL_SEGMENTS)
                # cp.piecewise is unavailable; use cp.sum of piecewise-linear pieces
                # Implement as upper envelope of linear segments (concave PWL)
                pwl_var = cp.Variable(name=f"rev_{ch}_{outcome}")
                # Each segment: ys_bp[k] + slope_k * (x_i - xs_bp[k]) >= pwl_var  (concave)
                # Equivalently we maximise the minimum of all affine pieces
                # For concave PWL: revenue = min over all upper tangent lines
                # ⟹ Correct formulation: maximise v_i s.t. v_i ≤ each affine piece
                slopes = np.diff(ys_bp) / np.maximum(np.diff(xs_bp), 1e-9)
                constraints_pwl = []
                for k in range(len(xs_bp) - 1):
                    # Affine piece: y = ys_bp[k] + slopes[k] * (x - xs_bp[k])
                    constraints_pwl.append(
                        pwl_var <= ys_bp[k] + slopes[k] * (x[i] - xs_bp[k])
                    )
                objective_terms.append((weight, pwl_var, constraints_pwl))

        # Assemble the objective and constraints
        obj_sum = cp.Constant(0)
        all_constraints: List[Any] = []
        revenue_vars = []
        for weight, var, cstr in objective_terms:
            obj_sum = obj_sum + weight * var
            all_constraints.extend(cstr)
            revenue_vars.append(var)

        # Budget constraints
        all_constraints.append(cp.sum(x) <= total_budget)
        for i, ch in enumerate(self.channels):
            # Concentration cap
            all_constraints.append(x[i] <= max_concentration * total_budget)
            # Floor
            floor = min_floors.get(ch, 0.0)
            if floor > 0:
                all_constraints.append(x[i] >= floor)

        problem = cp.Problem(cp.Maximize(obj_sum), all_constraints)

        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            try:
                problem.solve(solver=cp.SCS, verbose=False)
            except Exception as exc:
                logger.warning("CVXPY solve failed: %s; falling back to scipy.", exc)
                return self._optimize_scipy(
                    total_budget, shopify_weight, amazon_weight,
                    min_floors, max_concentration,
                )

        if problem.status not in ("optimal", "optimal_inaccurate"):
            logger.warning(
                "CVXPY status=%s; falling back to scipy.", problem.status
            )
            return self._optimize_scipy(
                total_budget, shopify_weight, amazon_weight,
                min_floors, max_concentration,
            )

        allocation = {
            ch: float(np.maximum(x.value[i], 0.0))
            for i, ch in enumerate(self.channels)
        }
        return allocation, f"cvxpy-{problem.status}"

    # ------------------------------------------------------------------
    # Scipy fallback optimisation
    # ------------------------------------------------------------------

    def _optimize_scipy(
        self,
        total_budget: float,
        shopify_weight: float,
        amazon_weight: float,
        min_floors: Dict[str, float],
        max_concentration: float,
    ) -> Tuple[Dict[str, float], str]:
        """Gradient-based fallback using scipy L-BFGS-B."""
        n = len(self.channels)
        max_per_channel = max_concentration * total_budget

        def neg_revenue(x_vec: np.ndarray) -> float:
            allocation = dict(zip(self.channels, x_vec))
            _, _, blended = self._total_blended_revenue(
                allocation, shopify_weight, amazon_weight
            )
            return -blended

        def neg_revenue_grad(x_vec: np.ndarray) -> np.ndarray:
            grad = np.zeros(n)
            for i, ch in enumerate(self.channels):
                delta = max(x_vec[i] * 0.001, 1.0)
                r_plus = -neg_revenue(
                    np.concatenate([x_vec[:i], [x_vec[i] + delta], x_vec[i + 1:]])
                )
                r_minus = -neg_revenue(
                    np.concatenate([x_vec[:i], [max(0.0, x_vec[i] - delta)], x_vec[i + 1:]])
                )
                grad[i] = -(r_plus - r_minus) / (2 * delta)
            return grad

        # Bounds: [floor, max_concentration * budget]
        bounds = [
            (min_floors.get(ch, 0.0), max_per_channel)
            for ch in self.channels
        ]
        # Initial point: proportional to historical spend or equal share
        x0 = np.full(n, total_budget / n)
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

        constraints = [
            {"type": "ineq", "fun": lambda x: total_budget - np.sum(x)},
        ]

        result = minimize(
            neg_revenue,
            x0,
            jac=neg_revenue_grad,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        # Project onto budget constraint if violated
        x_opt = np.maximum(result.x, 0.0)
        if np.sum(x_opt) > total_budget:
            x_opt = x_opt / np.sum(x_opt) * total_budget

        allocation = dict(zip(self.channels, x_opt.tolist()))
        return allocation, f"scipy-L-BFGS-B-{result.message}"

    # ------------------------------------------------------------------
    # Causal iROAS calibration
    # ------------------------------------------------------------------

    def calibrate_from_iroas(
        self,
        channel: str,
        iroas_causal: float,
        current_spend: float,
        outcome: str = "shopify",
    ) -> bool:
        """Anchor the Hill curve scale for *channel* to a causal iROAS estimate.

        Re-scales the Hill curve so that the marginal ROI at *current_spend*
        equals *iroas_causal* — grounding the optimizer in geo holdout
        ground truth rather than purely Bayesian priors.

        When geo holdout iROAS data is available this should be called for
        each validated channel before running ``optimize()``.

        Parameters
        ----------
        channel : str
            Channel name (must already be in saturation_curves).
        iroas_causal : float
            Causal incremental ROAS measured by geo holdout test.
        current_spend : float
            Current weekly spend on this channel (dollars).
        outcome : str
            Outcome to calibrate ("shopify" | "amazon").

        Returns
        -------
        bool
            True if the calibration updated the curve, False if the channel
            or outcome was not found.
        """
        curve = self._curves.get(channel, {}).get(outcome)
        if curve is None:
            logger.warning(
                "calibrate_from_iroas: channel=%s outcome=%s not found; skipping.",
                channel, outcome,
            )
            return False

        if current_spend <= 0 or iroas_causal <= 0:
            logger.debug(
                "calibrate_from_iroas: channel=%s skipped (spend=%.0f, iroas=%.3f).",
                channel, current_spend, iroas_causal,
            )
            return False

        # Current marginal ROI at current_spend (uses Δ = 1 % of spend or $100)
        delta = max(current_spend * 0.01, 100.0)
        current_mroi = curve.marginal_roi(current_spend, delta)
        if current_mroi < 1e-9:
            logger.debug(
                "calibrate_from_iroas: channel=%s current_mroi≈0; skipping.", channel
            )
            return False

        # Scale the Hill curve scale parameter so that mroi(current_spend) = iroas_causal
        adjustment = iroas_causal / current_mroi
        new_scale = curve.scale * adjustment
        curve.scale = max(new_scale, 1e-6)  # guard against negative scale
        logger.info(
            "Calibrated %s/%s: scale %.4f → %.4f (iROAS=%.3f, adjustment=%.3f)",
            channel, outcome, curve.scale / adjustment, curve.scale,
            iroas_causal, adjustment,
        )
        return True

    def calibrate_all_from_iroas(
        self,
        iroas_by_channel: Dict[str, float],
        spend_by_channel: Dict[str, float],
        outcomes: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """Batch calibration: call ``calibrate_from_iroas`` for each channel.

        Parameters
        ----------
        iroas_by_channel : dict
            Channel → causal iROAS (from geo holdout results).
        spend_by_channel : dict
            Channel → current spend (dollars).
        outcomes : list[str], optional
            Outcomes to calibrate.  Defaults to ``["shopify", "amazon"]``.

        Returns
        -------
        dict
            Channel → bool indicating whether calibration succeeded.
        """
        outcomes = outcomes or ["shopify", "amazon"]
        results: Dict[str, bool] = {}
        for ch, iroas in iroas_by_channel.items():
            spend = spend_by_channel.get(ch, 0.0)
            ok = any(
                self.calibrate_from_iroas(ch, iroas, spend, outcome=out)
                for out in outcomes
            )
            results[ch] = ok
        calibrated = [c for c, ok in results.items() if ok]
        logger.info(
            "calibrate_all_from_iroas: %d/%d channels calibrated: %s",
            len(calibrated), len(results), calibrated,
        )
        return results

    # ------------------------------------------------------------------
    # Posterior uncertainty: ±1 SD allocation bounds
    # ------------------------------------------------------------------

    def _compute_allocation_bounds(
        self,
        allocation: Dict[str, float],
        total_budget: float,
        posterior_sigma: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute ±1 posterior SD bounds on optimal allocation.

        When *posterior_sigma* is provided (a dict mapping channel →
        fractional standard deviation of the Hill-curve scale parameter)
        the method re-optimises the budget with the scale perturbed by
        ±1 SD and uses the resulting allocations as the bounds.  This
        correctly propagates posterior uncertainty through the non-linear
        Hill saturation function.

        When *posterior_sigma* is not provided the method falls back to a
        simple ±15 % perturbation on the allocation itself — a first-order
        approximation that avoids an extra CVXPY solve.

        Parameters
        ----------
        allocation : dict
            Point-optimal spend per channel.
        total_budget : float
            Total budget constraint.
        posterior_sigma : dict, optional
            Channel → fractional SD of the Hill-curve scale parameter
            (e.g. ``{"meta_perf": 0.12, "google_brand": 0.09}``).
            Typically obtained from the Meridian posterior.

        Returns
        -------
        tuple[dict, dict]
            ``(lower, upper)`` allocation dicts.
        """
        if not posterior_sigma:
            # Fallback: flat ±15 % on allocation
            sigma_frac = 0.15
            lower = {ch: max(allocation.get(ch, 0.0) * (1 - sigma_frac), 0.0)
                     for ch in self.channels}
            upper = {ch: min(allocation.get(ch, 0.0) * (1 + sigma_frac), total_budget)
                     for ch in self.channels}
            return lower, upper

        # Posterior-based: re-optimise with scale perturbed by ±1 SD
        lower: Dict[str, float] = {}
        upper: Dict[str, float] = {}

        # Save original scales
        original_scales: Dict[str, Dict[str, float]] = {
            ch: {out: curve.scale for out, curve in outcomes.items()}
            for ch, outcomes in self._curves.items()
        }

        try:
            # --- Lower bound: pessimistic (−1 SD on all scales) ---
            for ch in self.channels:
                sigma = posterior_sigma.get(ch, 0.15)
                for out, curve in self._curves.get(ch, {}).items():
                    curve.scale = max(original_scales[ch][out] * (1 - sigma), 1e-6)

            alloc_lo, _ = (
                self._optimize_cvxpy(total_budget, 0.7, 0.3, {}, 0.55)
                if _CVXPY_AVAILABLE
                else self._optimize_scipy(total_budget, 0.7, 0.3, {}, 0.55)
            )

            # --- Upper bound: optimistic (+1 SD on all scales) ---
            for ch in self.channels:
                sigma = posterior_sigma.get(ch, 0.15)
                for out, curve in self._curves.get(ch, {}).items():
                    curve.scale = original_scales[ch][out] * (1 + sigma)

            alloc_hi, _ = (
                self._optimize_cvxpy(total_budget, 0.7, 0.3, {}, 0.55)
                if _CVXPY_AVAILABLE
                else self._optimize_scipy(total_budget, 0.7, 0.3, {}, 0.55)
            )

            for ch in self.channels:
                lower[ch] = max(alloc_lo.get(ch, 0.0), 0.0)
                upper[ch] = min(alloc_hi.get(ch, total_budget), total_budget)

        except Exception as exc:
            logger.warning(
                "Posterior-based allocation bounds failed (%s); using ±15 %% fallback.", exc
            )
            for ch in self.channels:
                spend = allocation.get(ch, 0.0)
                lower[ch] = max(spend * 0.85, 0.0)
                upper[ch] = min(spend * 1.15, total_budget)
        finally:
            # Restore original scales
            for ch in self.channels:
                for out, curve in self._curves.get(ch, {}).items():
                    curve.scale = original_scales.get(ch, {}).get(out, curve.scale)

        return lower, upper

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        total_budget: float,
        shopify_weight: float = 0.7,
        amazon_weight: float = 0.3,
        min_floors: Optional[Dict[str, float]] = None,
        max_concentration: float = 0.55,
        posterior_sigma: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """Solve the budget allocation problem.

        Parameters
        ----------
        total_budget : float
            Total media spend budget (dollars).
        shopify_weight : float
            Weight applied to Shopify revenue in the blended objective.
        amazon_weight : float
            Weight applied to Amazon revenue in the blended objective.
        min_floors : dict, optional
            Per-channel minimum spend floors (dollars).  Defaults to zero
            for all channels.
        max_concentration : float
            Maximum fraction of total budget that can go to any single
            channel (default 0.55 = 55 %).
        posterior_sigma : dict, optional
            Channel → fractional SD of the Hill curve scale parameter.
            When provided, allocation CIs are computed via re-optimisation
            rather than the flat ±15 % fallback.

        Returns
        -------
        OptimizationResult
            Optimal allocation plus revenue forecasts and uncertainty bounds.
        """
        if abs(shopify_weight + amazon_weight - 1.0) > 1e-6:
            raise ValueError(
                f"shopify_weight + amazon_weight must equal 1.0, "
                f"got {shopify_weight + amazon_weight:.4f}"
            )
        if total_budget <= 0:
            raise ValueError(f"total_budget must be positive, got {total_budget}")

        floors = min_floors or {}

        # Validate floors do not exceed budget
        total_floors = sum(floors.get(ch, 0.0) for ch in self.channels)
        if total_floors > total_budget:
            raise ValueError(
                f"Sum of min_floors ({total_floors:.0f}) exceeds total_budget "
                f"({total_budget:.0f})."
            )

        # Solve
        if _CVXPY_AVAILABLE:
            allocation, status = self._optimize_cvxpy(
                total_budget, shopify_weight, amazon_weight,
                floors, max_concentration,
            )
        else:
            allocation, status = self._optimize_scipy(
                total_budget, shopify_weight, amazon_weight,
                floors, max_concentration,
            )

        # Compute revenue at the optimal allocation
        shopify_rev, amazon_rev, _ = self._total_blended_revenue(
            allocation, shopify_weight, amazon_weight
        )
        blended_rev = shopify_weight * shopify_rev + amazon_weight * amazon_rev
        total_spend = sum(allocation.values())
        blended_roas = blended_rev / total_spend if total_spend > 0 else 0.0

        # Uncertainty bounds (posterior-based when sigma info is available)
        alloc_lower, alloc_upper = self._compute_allocation_bounds(
            allocation, total_budget, posterior_sigma=posterior_sigma
        )

        logger.info(
            "Optimization complete: budget=%.0f, blended_rev=%.0f, "
            "blended_roas=%.2f, status=%s",
            total_budget, blended_rev, blended_roas, status,
        )

        return OptimizationResult(
            allocation=allocation,
            expected_shopify_revenue=shopify_rev,
            expected_amazon_revenue=amazon_rev,
            expected_total_revenue=blended_rev,
            blended_roas=blended_roas,
            allocation_lower=alloc_lower,
            allocation_upper=alloc_upper,
            solver_status=status,
            total_budget=total_budget,
            shopify_weight=shopify_weight,
            amazon_weight=amazon_weight,
        )

    def get_marginal_rois(self, current_allocation: Dict[str, float]) -> Dict[str, float]:
        """Compute the marginal ROI for each channel at the current allocation.

        Marginal ROI = ΔRevenue / ΔSpend (blended across Shopify + Amazon).

        Parameters
        ----------
        current_allocation : dict
            Current spend per channel (dollars).

        Returns
        -------
        dict
            Channel → marginal ROI (blended).
        """
        marginal_rois: Dict[str, float] = {}
        shopify_weight = 0.7
        amazon_weight = 0.3

        for ch in self.channels:
            spend = current_allocation.get(ch, 0.0)
            delta = max(spend * 0.01, 100.0)  # 1 % or $100, whichever is larger

            mroi_shopify = 0.0
            mroi_amazon = 0.0

            if curve_sh := self._curves[ch].get("shopify"):
                mroi_shopify = curve_sh.marginal_roi(spend, delta)
            if curve_amz := self._curves[ch].get("amazon"):
                mroi_amazon = curve_amz.marginal_roi(spend, delta)

            marginal_rois[ch] = shopify_weight * mroi_shopify + amazon_weight * mroi_amazon

        return marginal_rois

    def get_scenario(
        self,
        base_allocation: Dict[str, float],
        budget_delta_pct: float,
    ) -> OptimizationResult:
        """Re-optimise at a scaled budget.

        Parameters
        ----------
        base_allocation : dict
            The reference allocation whose total spend defines the base budget.
        budget_delta_pct : float
            Percentage change relative to the base budget (e.g. +10 means
            10 % increase, -20 means 20 % cut).

        Returns
        -------
        OptimizationResult
            Optimal allocation at the new budget level.
        """
        base_budget = sum(base_allocation.values())
        new_budget = base_budget * (1 + budget_delta_pct / 100.0)
        if new_budget <= 0:
            raise ValueError(
                f"Resulting budget ({new_budget:.0f}) must be positive."
            )
        return self.optimize(total_budget=new_budget)
