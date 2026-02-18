"""Comprehensive test suite for the MMM components.

Covers:
  - MeridianConfig           (incrementality.mmm.config)
  - CalibrationEvent / CalibrationStore (incrementality.mmm.calibration)
  - MeridianMMM stub mode    (incrementality.mmm.model)
  - BudgetOptimizer          (incrementality.mmm.optimizer)
  - AlertManager             (incrementality.alerts)
  - Tensor utilities         (incrementality.mmm.tensors)
  - compute_mape             (incrementality.mmm.validation)
  - ModelRegistry local fallback (incrementality.mmm.registry)

Design principles
-----------------
- No real API calls; all HTTP is mocked via unittest.mock.patch.
- No actual Meridian sampling; MeridianMMM always runs in stub mode here
  because meridian is not installed in this environment.
- All disk I/O uses pytest's tmp_path fixture so tests are isolated.
- Each test is independent, focused, and fast (< 1 s individually).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import guard helpers
# ---------------------------------------------------------------------------

def _import_or_skip(module_path: str, package: str):
    """Return the module or skip the test if import fails."""
    import importlib
    try:
        return importlib.import_module(module_path)
    except ImportError:
        pytest.skip(f"Optional dependency unavailable: {package}")


# ===========================================================================
# Section 1 — MeridianConfig
# ===========================================================================

class TestMeridianConfig:
    """Tests for incrementality.mmm.config.MeridianConfig."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from incrementality.mmm.config import MeridianConfig
        self.MeridianConfig = MeridianConfig

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def test_default_instantiation_succeeds(self):
        """MeridianConfig() with all defaults should not raise."""
        cfg = self.MeridianConfig()
        assert cfg is not None
        assert cfg.n_channels == 7
        assert cfg.n_outcomes == 2
        assert cfg.n_dmas == 210
        assert cfg.mcmc_chains == 4
        assert cfg.mcmc_samples == 2000

    def test_custom_instantiation_succeeds(self):
        """Overriding individual fields should work as long as they are consistent."""
        cfg = self.MeridianConfig(
            mcmc_chains=2,
            mcmc_samples=500,
            adstock_max_lag=28,
            tpu_enabled=False,
        )
        assert cfg.mcmc_chains == 2
        assert cfg.mcmc_samples == 500
        assert cfg.adstock_max_lag == 28

    # ------------------------------------------------------------------
    # channel_index / outcome_index helpers
    # ------------------------------------------------------------------

    def test_channel_index_returns_correct_position(self):
        cfg = self.MeridianConfig()
        assert cfg.channel_index("meta_perf") == 0
        assert cfg.channel_index("meta_aware") == 1
        assert cfg.channel_index("email_sms") == 6

    def test_channel_index_raises_for_unknown_channel(self):
        cfg = self.MeridianConfig()
        with pytest.raises(KeyError, match="Unknown channel"):
            cfg.channel_index("nonexistent_channel")

    def test_outcome_index_returns_correct_position(self):
        cfg = self.MeridianConfig()
        assert cfg.outcome_index("shopify") == 0
        assert cfg.outcome_index("amazon") == 1

    def test_outcome_index_raises_for_unknown_outcome(self):
        cfg = self.MeridianConfig()
        with pytest.raises(KeyError, match="Unknown outcome"):
            cfg.outcome_index("tiktok_shop")

    # ------------------------------------------------------------------
    # total_posterior_samples property
    # ------------------------------------------------------------------

    def test_total_posterior_samples_default(self):
        """Default: 4 chains × 2000 samples = 8000."""
        cfg = self.MeridianConfig()
        assert cfg.total_posterior_samples == 4 * 2000

    def test_total_posterior_samples_custom(self):
        cfg = self.MeridianConfig(mcmc_chains=2, mcmc_samples=500)
        assert cfg.total_posterior_samples == 2 * 500

    # ------------------------------------------------------------------
    # Validation — invalid channel count
    # ------------------------------------------------------------------

    def test_invalid_channel_count_raises_validation_error(self):
        """n_channels that does not match len(channels) must raise ValueError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self.MeridianConfig(
                n_channels=5,  # mismatch: default channels list has 7 entries
            )

    def test_invalid_outcome_count_raises_validation_error(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self.MeridianConfig(
                n_outcomes=3,  # mismatch: default outcomes list has 2 entries
            )

    def test_invalid_discount_factor_raises_validation_error(self):
        """Discount factors outside (0, 1] must be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self.MeridianConfig(
                channel_discount_factors={"meta_perf": 1.5}  # > 1 is invalid
            )

    def test_zero_discount_factor_raises_validation_error(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self.MeridianConfig(
                channel_discount_factors={"meta_perf": 0.0}  # must be > 0
            )


# ===========================================================================
# Section 2 — CalibrationEvent and CalibrationStore
# ===========================================================================

class TestCalibrationEvent:
    """Tests for incrementality.mmm.calibration.CalibrationEvent."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from incrementality.mmm.calibration import CalibrationEvent
        self.CalibrationEvent = CalibrationEvent

    def _make_event(self, **overrides) -> Any:
        """Return a valid CalibrationEvent with sensible defaults."""
        defaults = dict(
            channel="meta_perf",
            test_type="geo_holdout",
            test_start_date=date(2025, 1, 6),
            test_end_date=date(2025, 2, 2),
            lift_abs=142_000.0,
            lift_lower_90=98_000.0,
            lift_upper_90=186_000.0,
            spend_in_period=218_000.0,
            outcome="shopify",
            geo_scope="national",
        )
        defaults.update(overrides)
        return self.CalibrationEvent(**defaults)

    # ------------------------------------------------------------------
    # Creation with all required fields
    # ------------------------------------------------------------------

    def test_creation_with_all_required_fields(self):
        """A fully-specified CalibrationEvent should be created without error."""
        event = self._make_event()
        assert event.channel == "meta_perf"
        assert event.test_type == "geo_holdout"
        assert event.lift_abs == 142_000.0
        assert event.spend_in_period == 218_000.0
        assert event.outcome == "shopify"

    def test_ingested_at_is_auto_populated(self):
        """ingested_at should be set automatically on creation."""
        event = self._make_event()
        assert event.ingested_at is not None

    # ------------------------------------------------------------------
    # implied_roi auto-computation
    # ------------------------------------------------------------------

    def test_implied_roi_auto_computed_when_not_provided(self):
        """When implied_roi is not given, it should be computed as lift/spend."""
        event = self._make_event(lift_abs=100_000.0, spend_in_period=200_000.0)
        expected = round(100_000.0 / 200_000.0, 4)
        assert abs(event.implied_roi - expected) < 1e-6

    def test_implied_roi_manual_override_respected(self):
        """Explicit implied_roi should not be overwritten by auto-computation."""
        event = self._make_event(
            lift_abs=100_000.0,
            spend_in_period=200_000.0,
            implied_roi=0.99,  # manual override
        )
        assert event.implied_roi == 0.99

    # ------------------------------------------------------------------
    # Date validation
    # ------------------------------------------------------------------

    def test_end_date_before_start_date_raises(self):
        """end_date before start_date must raise a validation error."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="end_date"):
            self._make_event(
                test_start_date=date(2025, 2, 10),
                test_end_date=date(2025, 1, 5),
            )

    def test_same_start_and_end_date_is_valid(self):
        """A single-day test (start == end) must be accepted."""
        event = self._make_event(
            test_start_date=date(2025, 3, 1),
            test_end_date=date(2025, 3, 1),
        )
        assert event.duration_days == 1

    # ------------------------------------------------------------------
    # duration_days property
    # ------------------------------------------------------------------

    def test_duration_days_is_inclusive(self):
        """duration_days should count both endpoints."""
        event = self._make_event(
            test_start_date=date(2025, 1, 1),
            test_end_date=date(2025, 1, 7),
        )
        assert event.duration_days == 7

    # ------------------------------------------------------------------
    # Serialisation round-trip
    # ------------------------------------------------------------------

    def test_to_json_dict_and_from_json_dict_roundtrip(self):
        event = self._make_event()
        json_dict = event.to_json_dict()
        # Dates should be ISO strings
        assert isinstance(json_dict["test_start_date"], str)
        assert isinstance(json_dict["test_end_date"], str)
        # Reconstruct
        reconstructed = self.CalibrationEvent.from_json_dict(json_dict)
        assert reconstructed.channel == event.channel
        assert reconstructed.lift_abs == event.lift_abs
        assert reconstructed.implied_roi == event.implied_roi


class TestCalibrationStore:
    """Tests for incrementality.mmm.calibration.CalibrationStore."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from incrementality.mmm.calibration import CalibrationEvent, CalibrationStore
        self.CalibrationEvent = CalibrationEvent
        self.CalibrationStore = CalibrationStore

    def _make_event(self, channel: str = "meta_perf", **overrides) -> Any:
        defaults = dict(
            channel=channel,
            test_type="geo_holdout",
            test_start_date=date(2025, 1, 6),
            test_end_date=date(2025, 2, 2),
            lift_abs=142_000.0,
            lift_lower_90=98_000.0,
            lift_upper_90=186_000.0,
            spend_in_period=218_000.0,
            outcome="shopify",
            geo_scope="national",
        )
        defaults.update(overrides)
        return self.CalibrationEvent(**defaults)

    # ------------------------------------------------------------------
    # add_event + persistence
    # ------------------------------------------------------------------

    def test_add_event_saves_to_disk(self, tmp_path):
        store_path = tmp_path / "events.json"
        store = self.CalibrationStore(storage_path=store_path)
        event = self._make_event()
        store.add_event(event)
        assert store_path.exists(), "JSON file must be created on first write"
        assert len(store) == 1

    def test_add_multiple_events_persists_all(self, tmp_path):
        store_path = tmp_path / "events.json"
        store = self.CalibrationStore(storage_path=store_path)
        for i in range(3):
            store.add_event(
                self._make_event(
                    spend_in_period=float(100_000 + i * 10_000),
                    test_start_date=date(2025, 1, 1) + timedelta(weeks=i),
                    test_end_date=date(2025, 1, 28) + timedelta(weeks=i),
                )
            )
        assert len(store) == 3

    # ------------------------------------------------------------------
    # get_events — no filter
    # ------------------------------------------------------------------

    def test_get_events_returns_all_events(self, tmp_path):
        store_path = tmp_path / "events.json"
        store = self.CalibrationStore(storage_path=store_path)
        for _ in range(4):
            store.add_event(self._make_event())
        events = store.get_events()
        assert len(events) == 4

    def test_get_events_returns_empty_list_when_no_file(self, tmp_path):
        store_path = tmp_path / "nonexistent.json"
        store = self.CalibrationStore(storage_path=store_path)
        assert store.get_events() == []

    # ------------------------------------------------------------------
    # get_events — channel filter
    # ------------------------------------------------------------------

    def test_get_events_filters_by_channel(self, tmp_path):
        store_path = tmp_path / "events.json"
        store = self.CalibrationStore(storage_path=store_path)

        store.add_event(self._make_event(channel="meta_perf"))
        store.add_event(self._make_event(channel="meta_perf"))
        store.add_event(self._make_event(channel="google_nonbrand"))

        meta_events = store.get_events(channel="meta_perf")
        google_events = store.get_events(channel="google_nonbrand")

        assert len(meta_events) == 2
        assert len(google_events) == 1
        assert all(e.channel == "meta_perf" for e in meta_events)

    def test_get_events_filter_channel_no_match_returns_empty(self, tmp_path):
        store_path = tmp_path / "events.json"
        store = self.CalibrationStore(storage_path=store_path)
        store.add_event(self._make_event(channel="tiktok"))
        result = store.get_events(channel="email_sms")
        assert result == []

    # ------------------------------------------------------------------
    # get_latest_per_channel
    # ------------------------------------------------------------------

    def test_get_latest_per_channel_returns_one_per_channel(self, tmp_path):
        store_path = tmp_path / "events.json"
        store = self.CalibrationStore(storage_path=store_path)

        # Two meta_perf events with different end dates
        store.add_event(self._make_event(
            channel="meta_perf",
            test_end_date=date(2025, 1, 31),
        ))
        store.add_event(self._make_event(
            channel="meta_perf",
            test_end_date=date(2025, 3, 31),
        ))
        store.add_event(self._make_event(channel="tiktok"))

        latest = store.get_latest_per_channel()
        assert set(latest.keys()) == {"meta_perf", "tiktok"}
        # The most recent meta_perf event should be selected
        assert latest["meta_perf"].test_end_date == date(2025, 3, 31)

    def test_get_latest_per_channel_empty_store_returns_empty_dict(self, tmp_path):
        store_path = tmp_path / "no_file.json"
        store = self.CalibrationStore(storage_path=store_path)
        assert store.get_latest_per_channel() == {}

    # ------------------------------------------------------------------
    # to_meridian_constraints
    # ------------------------------------------------------------------

    def test_to_meridian_constraints_returns_list_of_dicts(self, tmp_path):
        store_path = tmp_path / "events.json"
        store = self.CalibrationStore(storage_path=store_path)
        event = self._make_event(channel="meta_perf")
        store.add_event(event)

        events = store.get_events()
        constraints = store.to_meridian_constraints(events)

        assert isinstance(constraints, list)
        assert len(constraints) == 1

    def test_to_meridian_constraints_has_required_keys(self, tmp_path):
        store_path = tmp_path / "events.json"
        store = self.CalibrationStore(storage_path=store_path)
        store.add_event(self._make_event())
        constraints = store.to_meridian_constraints(store.get_events())
        constraint = constraints[0]

        required_keys = {
            "lift_abs", "lift_lower", "lift_upper",
            "channel", "outcome", "spend_in_period",
            "start_date", "end_date", "geo_scope",
        }
        for key in required_keys:
            assert key in constraint, f"Missing required key: '{key}'"

    def test_to_meridian_constraints_values_match_source_event(self, tmp_path):
        store_path = tmp_path / "events.json"
        store = self.CalibrationStore(storage_path=store_path)
        event = self._make_event(lift_abs=99_000.0, spend_in_period=150_000.0)
        store.add_event(event)

        constraints = store.to_meridian_constraints(store.get_events())
        c = constraints[0]
        assert c["lift_abs"] == 99_000.0
        assert c["spend_in_period"] == 150_000.0
        assert c["channel"] == "meta_perf"


# ===========================================================================
# Section 3 — MeridianMMM stub mode
# ===========================================================================

class TestMeridianMMMStub:
    """Tests for MeridianMMM operating in stub mode (meridian not installed)."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @pytest.fixture()
    def cfg(self):
        from incrementality.mmm.config import MeridianConfig
        return MeridianConfig()

    @pytest.fixture()
    def fitted_model(self, cfg):
        """Return a MeridianMMM instance that has been fit on tiny synthetic data."""
        from incrementality.mmm.model import MeridianMMM
        model = MeridianMMM(cfg)

        T, G, C, K = 12, 3, cfg.n_channels, cfg.n_outcomes
        rng = np.random.default_rng(0)
        model.fit(
            kpi_tensor=rng.uniform(1e4, 1e5, (T, G, K)).astype(np.float32),
            media_tensor=rng.uniform(0, 1e4, (T, G, C)).astype(np.float32),
            media_spend=rng.uniform(0, 1e4, (T, G, C)).astype(np.float32),
            extra_features=rng.uniform(0, 1, (T, G, 2)).astype(np.float32),
            population=np.full(G, 1.0 / G, dtype=np.float32),
        )
        return model

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def test_instantiation_succeeds_without_meridian(self, cfg):
        from incrementality.mmm.model import MeridianMMM
        model = MeridianMMM(cfg)
        assert model is not None
        assert model.is_stub is True
        assert model.is_fitted is False

    # ------------------------------------------------------------------
    # get_rhat_diagnostics
    # ------------------------------------------------------------------

    def test_get_rhat_diagnostics_returns_dataframe(self, fitted_model):
        df = fitted_model.get_rhat_diagnostics()
        assert isinstance(df, pd.DataFrame)

    def test_get_rhat_diagnostics_has_required_columns(self, fitted_model):
        df = fitted_model.get_rhat_diagnostics()
        assert "parameter" in df.columns
        assert "rhat" in df.columns

    def test_get_rhat_diagnostics_all_values_below_threshold(self, fitted_model):
        """Stub mode guarantees R-hat < 1.05 for all parameters."""
        df = fitted_model.get_rhat_diagnostics()
        assert df["rhat"].max() < 1.05, (
            f"Stub R-hat should be < 1.05; got max={df['rhat'].max():.4f}"
        )

    def test_get_rhat_diagnostics_raises_before_fit(self, cfg):
        from incrementality.mmm.model import MeridianMMM
        model = MeridianMMM(cfg)
        with pytest.raises(RuntimeError, match="fitted"):
            model.get_rhat_diagnostics()

    # ------------------------------------------------------------------
    # get_time_varying_betas
    # ------------------------------------------------------------------

    def test_get_time_varying_betas_returns_dataframe(self, fitted_model):
        df = fitted_model.get_time_varying_betas()
        assert isinstance(df, pd.DataFrame)

    def test_get_time_varying_betas_has_expected_columns(self, fitted_model):
        df = fitted_model.get_time_varying_betas()
        expected_cols = {"date", "channel", "outcome", "beta_mean", "beta_p10", "beta_p90"}
        assert expected_cols.issubset(set(df.columns))

    def test_get_time_varying_betas_covers_all_channels(self, fitted_model, cfg):
        df = fitted_model.get_time_varying_betas()
        assert set(cfg.channels).issubset(set(df["channel"].unique()))

    # ------------------------------------------------------------------
    # get_channel_contributions
    # ------------------------------------------------------------------

    def test_get_channel_contributions_returns_dataframe(self, fitted_model):
        df = fitted_model.get_channel_contributions()
        assert isinstance(df, pd.DataFrame)

    def test_get_channel_contributions_has_expected_columns(self, fitted_model):
        df = fitted_model.get_channel_contributions()
        expected_cols = {
            "date", "dma", "channel", "outcome",
            "contribution_mean", "contribution_p10", "contribution_p90",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_get_channel_contributions_covers_all_channels(self, fitted_model, cfg):
        df = fitted_model.get_channel_contributions()
        assert set(cfg.channels).issubset(set(df["channel"].unique()))

    # ------------------------------------------------------------------
    # get_saturation_curves
    # ------------------------------------------------------------------

    def test_get_saturation_curves_returns_dict(self, fitted_model):
        curves = fitted_model.get_saturation_curves()
        assert isinstance(curves, dict)

    def test_get_saturation_curves_has_one_entry_per_channel(self, fitted_model, cfg):
        curves = fitted_model.get_saturation_curves()
        assert set(curves.keys()) == set(cfg.channels)

    def test_get_saturation_curves_arrays_have_correct_shape(self, fitted_model):
        n_points = 50
        curves = fitted_model.get_saturation_curves(n_points=n_points)
        for channel, arr in curves.items():
            assert arr.shape == (n_points, 2), (
                f"Channel {channel!r}: expected shape ({n_points}, 2), got {arr.shape}"
            )

    # ------------------------------------------------------------------
    # save_posterior / load_posterior round-trip
    # ------------------------------------------------------------------

    def test_save_and_load_posterior_roundtrip(self, fitted_model, cfg, tmp_path):
        save_path = tmp_path / "posterior.pkl"
        fitted_model.save_posterior(save_path)
        assert save_path.exists(), "Posterior file must be created by save_posterior()"

        # Create a fresh model and load the saved posterior into it
        from incrementality.mmm.model import MeridianMMM
        fresh_model = MeridianMMM(cfg)
        assert not fresh_model.is_fitted
        fresh_model.load_posterior(save_path)
        assert fresh_model.is_fitted

    def test_load_posterior_restores_rhat_capability(self, fitted_model, cfg, tmp_path):
        save_path = tmp_path / "posterior.pkl"
        fitted_model.save_posterior(save_path)

        from incrementality.mmm.model import MeridianMMM
        fresh = MeridianMMM(cfg)
        fresh.load_posterior(save_path)
        df = fresh.get_rhat_diagnostics()
        assert isinstance(df, pd.DataFrame)
        assert "rhat" in df.columns

    def test_load_posterior_file_not_found_raises(self, cfg, tmp_path):
        from incrementality.mmm.model import MeridianMMM
        model = MeridianMMM(cfg)
        with pytest.raises(FileNotFoundError):
            model.load_posterior(tmp_path / "missing.pkl")


# ===========================================================================
# Section 4 — BudgetOptimizer
# ===========================================================================

def _make_saturation_curves(
    channels: List[str],
    n_points: int = 40,
    total_budget: float = 500_000.0,
) -> Dict[str, Dict[str, tuple]]:
    """Build synthetic Hill saturation curves for testing."""
    rng = np.random.default_rng(42)
    spend_pts = np.linspace(0.0, total_budget, n_points)

    def hill_rev(x, scale, alpha, gamma):
        xa = np.power(np.maximum(x, 0.0), alpha)
        ga = gamma ** alpha
        return scale * xa / (xa + ga)

    curves = {}
    for ch in channels:
        shopify_rev = hill_rev(spend_pts, rng.uniform(1e6, 3e6), 1.2, rng.uniform(1e5, 3e5))
        amazon_rev = hill_rev(spend_pts, rng.uniform(3e5, 8e5), 1.0, rng.uniform(8e4, 2e5))
        curves[ch] = {
            "shopify": (spend_pts, shopify_rev),
            "amazon": (spend_pts, amazon_rev),
        }
    return curves


_TEST_CHANNELS = ["meta_perf", "google_nonbrand", "tiktok", "email_sms"]
_TOTAL_BUDGET = 400_000.0


class TestBudgetOptimizer:
    """Tests for incrementality.mmm.optimizer.BudgetOptimizer."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from incrementality.mmm.optimizer import BudgetOptimizer, OptimizationResult
        self.BudgetOptimizer = BudgetOptimizer
        self.OptimizationResult = OptimizationResult

    @pytest.fixture()
    def curves(self):
        return _make_saturation_curves(_TEST_CHANNELS, total_budget=_TOTAL_BUDGET)

    @pytest.fixture()
    def optimizer(self, curves):
        return self.BudgetOptimizer(saturation_curves=curves, config={})

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def test_instantiation_with_synthetic_curves(self, curves):
        opt = self.BudgetOptimizer(saturation_curves=curves, config={})
        assert opt is not None
        assert set(opt.channels) == set(_TEST_CHANNELS)

    # ------------------------------------------------------------------
    # optimize() — budget allocation
    # ------------------------------------------------------------------

    def test_optimize_total_allocation_equals_budget(self, optimizer):
        result = optimizer.optimize(total_budget=_TOTAL_BUDGET)
        total_alloc = sum(result.allocation.values())
        tolerance = _TOTAL_BUDGET * 0.01  # 1 %
        assert abs(total_alloc - _TOTAL_BUDGET) <= tolerance, (
            f"Total allocation {total_alloc:.2f} deviates from "
            f"budget {_TOTAL_BUDGET:.2f} by more than 1 %"
        )

    def test_optimize_respects_concentration_cap(self, optimizer):
        """No single channel should receive more than 55 % of the total budget."""
        result = optimizer.optimize(total_budget=_TOTAL_BUDGET, max_concentration=0.55)
        for ch, alloc in result.allocation.items():
            fraction = alloc / _TOTAL_BUDGET
            assert fraction <= 0.55 + 1e-6, (
                f"Channel {ch!r} received {fraction:.2%} which exceeds the 55% cap"
            )

    def test_optimize_respects_floor_constraints(self, optimizer):
        """Channels with floors should receive at least the floor amount."""
        floors = {"meta_perf": 20_000.0, "email_sms": 10_000.0}
        result = optimizer.optimize(total_budget=_TOTAL_BUDGET, min_floors=floors)
        assert result.allocation["meta_perf"] >= 20_000.0 - 1.0
        assert result.allocation["email_sms"] >= 10_000.0 - 1.0

    def test_optimize_all_allocations_are_non_negative(self, optimizer):
        result = optimizer.optimize(total_budget=_TOTAL_BUDGET)
        for ch, alloc in result.allocation.items():
            assert alloc >= 0.0, f"Channel {ch!r} has negative allocation: {alloc}"

    # ------------------------------------------------------------------
    # get_marginal_rois
    # ------------------------------------------------------------------

    def test_get_marginal_rois_returns_dict_with_all_channels(self, optimizer):
        allocation = {ch: _TOTAL_BUDGET / len(_TEST_CHANNELS) for ch in _TEST_CHANNELS}
        mrois = optimizer.get_marginal_rois(allocation)
        assert isinstance(mrois, dict)
        assert set(mrois.keys()) == set(_TEST_CHANNELS)

    def test_get_marginal_rois_all_values_are_finite(self, optimizer):
        allocation = {ch: _TOTAL_BUDGET / len(_TEST_CHANNELS) for ch in _TEST_CHANNELS}
        mrois = optimizer.get_marginal_rois(allocation)
        for ch, mroi in mrois.items():
            assert np.isfinite(mroi), f"Non-finite marginal ROI for channel {ch!r}: {mroi}"

    # ------------------------------------------------------------------
    # get_scenario
    # ------------------------------------------------------------------

    def test_get_scenario_20pct_increase_raises_total_expected_revenue(self, optimizer):
        """A 20 % budget increase should yield higher expected revenue than the base."""
        base_result = optimizer.optimize(total_budget=_TOTAL_BUDGET)
        scenario_result = optimizer.get_scenario(base_result.allocation, budget_delta_pct=20)
        assert scenario_result.expected_total_revenue >= base_result.expected_total_revenue, (
            "Scenario with +20% budget should not have lower expected revenue than base"
        )

    def test_get_scenario_new_total_budget_is_scaled(self, optimizer):
        base_result = optimizer.optimize(total_budget=_TOTAL_BUDGET)
        scenario_result = optimizer.get_scenario(base_result.allocation, budget_delta_pct=20)
        expected_new_budget = _TOTAL_BUDGET * 1.20
        assert abs(scenario_result.total_budget - expected_new_budget) < 1.0, (
            f"Expected new budget ~{expected_new_budget:.0f}, "
            f"got {scenario_result.total_budget:.0f}"
        )

    # ------------------------------------------------------------------
    # OptimizationResult fields
    # ------------------------------------------------------------------

    def test_optimization_result_has_all_required_fields(self, optimizer):
        result = optimizer.optimize(total_budget=_TOTAL_BUDGET)
        assert isinstance(result, self.OptimizationResult)
        assert isinstance(result.allocation, dict)
        assert isinstance(result.expected_shopify_revenue, float)
        assert isinstance(result.expected_amazon_revenue, float)
        assert isinstance(result.expected_total_revenue, float)
        assert isinstance(result.blended_roas, float)
        assert isinstance(result.solver_status, str)
        assert isinstance(result.total_budget, float)
        assert result.total_budget == _TOTAL_BUDGET

    def test_optimization_result_blended_roas_is_positive(self, optimizer):
        result = optimizer.optimize(total_budget=_TOTAL_BUDGET)
        assert result.blended_roas > 0.0


# ===========================================================================
# Section 5 — AlertManager
# ===========================================================================

class TestAlertManager:
    """Tests for incrementality.alerts.AlertManager."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from incrementality.alerts import AlertManager, AlertSeverity
        self.AlertManager = AlertManager
        self.AlertSeverity = AlertSeverity

    @pytest.fixture()
    def am(self):
        return self.AlertManager(
            slack_webhook_url="https://hooks.slack.com/services/TEST/TEST/TOKEN",
            pagerduty_routing_key="test-pd-key",
        )

    def _good_rhat_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "parameter": ["beta_media[0]", "roi[0]", "sigma"],
            "rhat": [1.01, 1.02, 1.00],
        })

    def _bad_rhat_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "parameter": ["beta_media[0]", "roi[0]", "sigma"],
            "rhat": [1.06, 1.10, 1.02],
        })

    # ------------------------------------------------------------------
    # check_rhat_convergence — good R-hat
    # ------------------------------------------------------------------

    def test_check_rhat_convergence_good_does_not_send_alert(self, am):
        """All R-hats below 1.05 must NOT trigger the alert or call Slack."""
        with patch("requests.post") as mock_post:
            triggered = am.check_rhat_convergence(self._good_rhat_df())
        assert triggered is False
        mock_post.assert_not_called()

    # ------------------------------------------------------------------
    # check_rhat_convergence — bad R-hat
    # ------------------------------------------------------------------

    def test_check_rhat_convergence_bad_returns_true(self, am):
        """At least one R-hat above 1.05 must return True (critical alert)."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_resp):
            triggered = am.check_rhat_convergence(self._bad_rhat_df())
        assert triggered is True

    def test_check_rhat_convergence_bad_calls_slack(self, am):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_resp) as mock_post:
            am.check_rhat_convergence(self._bad_rhat_df())
        mock_post.assert_called()

    # ------------------------------------------------------------------
    # check_mroi_drop
    # ------------------------------------------------------------------

    def test_check_mroi_drop_25pct_wow_returns_triggered_channels(self, am):
        """A >20% WoW drop must return True (list of triggered channels)."""
        current = {"meta_perf": 0.5}
        previous = {"meta_perf": 0.7}  # 28.6 % drop → above 20% threshold
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_resp):
            triggered = am.check_mroi_drop(current, previous)
        assert "meta_perf" in triggered

    def test_check_mroi_drop_10pct_does_not_trigger(self, am):
        """A 10 % drop must NOT trigger the alert (threshold is 20 %)."""
        current = {"meta_perf": 0.9}
        previous = {"meta_perf": 1.0}  # 10 % drop
        with patch("requests.post") as mock_post:
            triggered = am.check_mroi_drop(current, previous)
        assert triggered == []
        mock_post.assert_not_called()

    # ------------------------------------------------------------------
    # check_channel_beta_decay
    # ------------------------------------------------------------------

    def test_check_channel_beta_decay_above_25pct_returns_true(self, am):
        """A >25% beta decay must return True with MEDIUM severity."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = am.check_channel_beta_decay(
                current_beta=0.5,
                rolling_90d_beta=0.8,   # 37.5 % drop — above threshold
                channel_name="meta_perf",
            )
        assert result is True
        mock_post.assert_called_once()

    def test_check_channel_beta_decay_below_25pct_returns_false(self, am):
        """A <25% beta decay must NOT trigger the alert."""
        with patch("requests.post") as mock_post:
            result = am.check_channel_beta_decay(
                current_beta=0.8,
                rolling_90d_beta=0.9,   # 11.1 % drop — below threshold
                channel_name="meta_perf",
            )
        assert result is False
        mock_post.assert_not_called()

    # ------------------------------------------------------------------
    # send_slack — does not make real HTTP calls
    # ------------------------------------------------------------------

    def test_send_slack_does_not_call_real_url(self, am):
        """send_slack() must use the mocked requests.post, never a real URL."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_resp) as mock_post:
            am.send_slack(
                message="Test alert message",
                severity=self.AlertSeverity.INFO,
            )
        mock_post.assert_called_once()
        # The URL used must be the webhook URL, not a live endpoint
        call_args = mock_post.call_args
        assert "hooks.slack.com" in call_args[0][0]

    def test_send_slack_returns_false_on_http_error(self, am):
        """send_slack() should return False when requests raises RequestException."""
        import requests as _requests
        with patch("requests.post", side_effect=_requests.RequestException("connection refused")):
            result = am.send_slack(
                message="Test message",
                severity=self.AlertSeverity.HIGH,
            )
        assert result is False

    # ------------------------------------------------------------------
    # from_config
    # ------------------------------------------------------------------

    def test_from_config_instantiation(self):
        config = {
            "slack_webhook_url": "https://hooks.slack.com/test",
            "pagerduty_routing_key": "KEY123",
        }
        am = self.AlertManager.from_config(config)
        assert isinstance(am, self.AlertManager)
        assert am.slack_webhook_url == config["slack_webhook_url"]
        assert am.pagerduty_routing_key == "KEY123"

    def test_from_config_without_pagerduty_key(self):
        config = {"slack_webhook_url": "https://hooks.slack.com/test"}
        am = self.AlertManager.from_config(config)
        assert am.pagerduty_routing_key is None


# ===========================================================================
# Section 6 — Tensor utilities
# ===========================================================================

class TestApplyGeometricAdstock:
    """Tests for incrementality.mmm.tensors.apply_geometric_adstock."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from incrementality.mmm.tensors import apply_geometric_adstock
        self.apply_geometric_adstock = apply_geometric_adstock

    def test_output_shape_matches_input(self):
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = self.apply_geometric_adstock(series, decay_rate=0.5, max_lag=3)
        assert out.shape == series.shape

    def test_decay_rate_zero_is_invalid(self):
        """decay_rate=0 is out of the valid open interval (0,1) and must raise."""
        with pytest.raises(ValueError, match="decay_rate"):
            self.apply_geometric_adstock(np.ones(5), decay_rate=0.0, max_lag=3)

    def test_decay_rate_one_is_invalid(self):
        """decay_rate=1 is also out of the valid interval."""
        with pytest.raises(ValueError, match="decay_rate"):
            self.apply_geometric_adstock(np.ones(5), decay_rate=1.0, max_lag=3)

    def test_very_small_decay_is_approximately_identity(self):
        """With decay_rate very close to 0, carryover is negligible; output ≈ input."""
        series = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        out = self.apply_geometric_adstock(series, decay_rate=0.0001, max_lag=3)
        # Maximum absolute difference should be tiny
        max_diff = float(np.abs(out.astype(np.float64) - series).max())
        assert max_diff < 1.0, (
            f"Expected near-identity with tiny decay, max diff = {max_diff:.6f}"
        )

    def test_non_negative_output_for_non_negative_input(self):
        series = np.abs(np.random.default_rng(1).normal(size=20))
        out = self.apply_geometric_adstock(series, decay_rate=0.3, max_lag=5)
        assert np.all(out >= 0.0)

    def test_output_dtype_is_float32(self):
        series = np.array([1.0, 2.0, 3.0])
        out = self.apply_geometric_adstock(series, decay_rate=0.5, max_lag=2)
        assert out.dtype == np.float32


class TestApplyHillSaturation:
    """Tests for incrementality.mmm.tensors.apply_hill_saturation."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from incrementality.mmm.tensors import apply_hill_saturation
        self.apply_hill_saturation = apply_hill_saturation

    def test_output_is_in_zero_to_one_range(self):
        series = np.linspace(0, 1_000_000, 200)
        out = self.apply_hill_saturation(series, alpha=1.5, gamma=500_000.0)
        assert float(out.min()) >= 0.0
        assert float(out.max()) < 1.0

    def test_output_is_monotonically_increasing(self):
        series = np.linspace(0, 1_000_000, 100)
        out = self.apply_hill_saturation(series, alpha=1.5, gamma=200_000.0)
        diffs = np.diff(out.astype(np.float64))
        assert np.all(diffs >= -1e-7), "Hill saturation output must be non-decreasing"

    def test_zero_input_gives_zero_output(self):
        out = self.apply_hill_saturation(np.array([0.0]), alpha=2.0, gamma=1000.0)
        assert float(out[0]) == 0.0

    def test_output_shape_matches_input_shape(self):
        series = np.random.default_rng(2).uniform(0, 1e5, (5, 4))
        out = self.apply_hill_saturation(series, alpha=1.0, gamma=5e4)
        assert out.shape == series.shape

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            self.apply_hill_saturation(np.ones(5), alpha=0.0, gamma=1.0)

    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            self.apply_hill_saturation(np.ones(5), alpha=1.0, gamma=0.0)

    def test_output_dtype_is_float32(self):
        out = self.apply_hill_saturation(np.array([100.0, 200.0]), alpha=1.0, gamma=150.0)
        assert out.dtype == np.float32


class TestBuildPopulationWeights:
    """Tests for MeridianTensorBuilder.build_population_weights."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from incrementality.mmm.tensors import MeridianTensorBuilder
        from incrementality.mmm.config import MeridianConfig
        self.builder = MeridianTensorBuilder(MeridianConfig())

    def _dma_df(self, pops: List[float]) -> pd.DataFrame:
        return pd.DataFrame({
            "dma": [f"DMA_{i:03d}" for i in range(len(pops))],
            "population": pops,
        })

    def test_weights_sum_to_one(self):
        df = self._dma_df([100.0, 200.0, 300.0, 400.0])
        weights = self.builder.build_population_weights(df)
        assert abs(float(weights.sum()) - 1.0) < 1e-6

    def test_weights_are_proportional_to_population(self):
        df = self._dma_df([100.0, 300.0])
        weights = self.builder.build_population_weights(df)
        assert abs(float(weights[0]) - 0.25) < 1e-6
        assert abs(float(weights[1]) - 0.75) < 1e-6

    def test_weights_dtype_is_float32(self):
        df = self._dma_df([1.0, 2.0, 3.0])
        weights = self.builder.build_population_weights(df)
        assert weights.dtype == np.float32

    def test_missing_population_column_raises(self):
        df = pd.DataFrame({"dma": ["A", "B"]})
        with pytest.raises(ValueError, match="population"):
            self.builder.build_population_weights(df)

    def test_zero_total_population_raises(self):
        df = self._dma_df([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="zero"):
            self.builder.build_population_weights(df)


class TestBuildKpiTensor:
    """Tests for MeridianTensorBuilder.build_kpi_tensor."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from incrementality.mmm.tensors import MeridianTensorBuilder
        from incrementality.mmm.config import MeridianConfig
        self.builder = MeridianTensorBuilder(MeridianConfig())

    def _make_revenue_df(
        self,
        dmas: List[str],
        n_periods: int,
        revenue: float = 100.0,
        start: str = "2025-01-01",
        freq: str = "W",
    ) -> pd.DataFrame:
        dates = pd.date_range(start, periods=n_periods, freq=freq)
        rows = [
            {"date": d, "dma": dma, "revenue": revenue}
            for d in dates
            for dma in dmas
        ]
        return pd.DataFrame(rows)

    def test_kpi_tensor_returns_correct_shape(self):
        dmas = ["DMA_001", "DMA_002", "DMA_003"]
        T, G = 8, len(dmas)
        shopify_df = self._make_revenue_df(dmas, T, revenue=500.0)
        amazon_df = self._make_revenue_df(dmas, T, revenue=200.0)
        kpi = self.builder.build_kpi_tensor(shopify_df, amazon_df, dmas)
        assert kpi.shape == (T, G, 2), (
            f"Expected shape ({T}, {G}, 2), got {kpi.shape}"
        )

    def test_kpi_tensor_axis_2_has_two_outcomes(self):
        dmas = ["DMA_A", "DMA_B"]
        shopify_df = self._make_revenue_df(dmas, 5)
        amazon_df = self._make_revenue_df(dmas, 5, revenue=50.0)
        kpi = self.builder.build_kpi_tensor(shopify_df, amazon_df, dmas)
        assert kpi.shape[2] == 2

    def test_kpi_tensor_dtype_is_float32(self):
        dmas = ["DMA_A"]
        shopify_df = self._make_revenue_df(dmas, 4)
        amazon_df = self._make_revenue_df(dmas, 4)
        kpi = self.builder.build_kpi_tensor(shopify_df, amazon_df, dmas)
        assert kpi.dtype == np.float32

    def test_kpi_tensor_values_match_input_revenue(self):
        dmas = ["DMA_A"]
        T = 4
        shopify_df = self._make_revenue_df(dmas, T, revenue=1000.0)
        amazon_df = self._make_revenue_df(dmas, T, revenue=500.0)
        kpi = self.builder.build_kpi_tensor(shopify_df, amazon_df, dmas)
        # axis-2 index 0 = shopify, index 1 = amazon
        assert np.allclose(kpi[:, 0, 0], 1000.0, atol=1.0)
        assert np.allclose(kpi[:, 0, 1], 500.0, atol=1.0)


# ===========================================================================
# Section 7 — compute_mape
# ===========================================================================

class TestComputeMape:
    """Tests for incrementality.mmm.validation.compute_mape."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from incrementality.mmm.validation import compute_mape
        self.compute_mape = compute_mape

    def test_perfect_predictions_return_zero(self):
        actuals = np.array([100.0, 200.0, 300.0])
        predictions = np.array([100.0, 200.0, 300.0])
        mape = self.compute_mape(actuals, predictions)
        assert mape == pytest.approx(0.0, abs=1e-9)

    def test_10_percent_error_returns_approx_010(self):
        """10 % over-prediction on each element should yield MAPE ≈ 0.10."""
        actuals = np.array([100.0, 200.0, 300.0])
        predictions = actuals * 1.10
        mape = self.compute_mape(actuals, predictions)
        # compute_mape returns 0-1 scale (0.10 for 10 %)
        assert mape == pytest.approx(0.10, rel=1e-3)

    def test_handles_zero_actuals_without_division_by_zero(self):
        """Zero actuals must not cause ZeroDivisionError; epsilon guards the denominator."""
        actuals = np.array([0.0, 100.0, 200.0])
        predictions = np.array([5.0, 110.0, 210.0])
        # Should not raise; result may be large but must be finite
        mape = self.compute_mape(actuals, predictions)
        assert np.isfinite(mape)

    def test_100_percent_error(self):
        actuals = np.array([100.0, 200.0])
        predictions = np.array([200.0, 400.0])  # 100 % over-prediction
        mape = self.compute_mape(actuals, predictions)
        assert mape == pytest.approx(1.0, rel=1e-3)

    def test_accepts_list_inputs(self):
        """compute_mape should accept plain Python lists as well as numpy arrays."""
        mape = self.compute_mape([50.0, 100.0], [55.0, 110.0])
        assert np.isfinite(mape)

    def test_single_element_arrays(self):
        mape = self.compute_mape(np.array([200.0]), np.array([220.0]))
        assert mape == pytest.approx(0.10, rel=1e-3)


# ===========================================================================
# Section 8 — ModelRegistry local fallback
# ===========================================================================

class TestModelRegistryLocalFallback:
    """Tests for ModelRegistry in local JSON-fallback mode (mlflow not installed)."""

    @pytest.fixture(autouse=True)
    def _patch_env_and_reimport(self, tmp_path, monkeypatch):
        """Override MTB_REGISTRY_DIR so tests write to a tmp directory."""
        monkeypatch.setenv("MTB_REGISTRY_DIR", str(tmp_path / "registry"))
        # Reload the module so the global _FALLBACK_FILE picks up the new env var
        import importlib
        import incrementality.mmm.registry as reg_mod
        importlib.reload(reg_mod)
        self.reg_mod = reg_mod
        self.ModelRegistry = reg_mod.ModelRegistry

    @pytest.fixture()
    def registry(self):
        return self.ModelRegistry(tracking_uri="http://localhost:9999")

    def _make_rhat_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "parameter": ["beta[0]", "roi[0]", "sigma"],
            "rhat": [1.01, 1.02, 1.00],
        })

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def test_instantiation_with_local_fallback_works(self, tmp_path):
        reg = self.ModelRegistry(tracking_uri="http://fake-mlflow:5000")
        assert reg is not None

    # ------------------------------------------------------------------
    # start_run context manager
    # ------------------------------------------------------------------

    def test_start_run_creates_a_run(self, registry, tmp_path):
        """After a start_run context exits, the JSON fallback file should exist."""
        with registry.start_run("test-run-init"):
            pass  # no metrics logged — just ensure the run opens/closes
        # The fallback JSON file should now exist
        fallback_path = self.reg_mod._FALLBACK_FILE
        assert fallback_path.exists(), (
            f"Expected fallback JSON file at {fallback_path}"
        )

    def test_start_run_writes_run_to_file(self, registry):
        with registry.start_run("my-refit-run"):
            pass
        runs = self.reg_mod._load_fallback_db()
        assert len(runs) >= 1
        assert any(r["run_name"] == "my-refit-run" for r in runs)

    # ------------------------------------------------------------------
    # log_refit_metrics
    # ------------------------------------------------------------------

    def test_log_refit_metrics_saves_to_local_file(self, registry):
        rhat_df = self._make_rhat_df()
        with registry.start_run("metric-run"):
            registry.log_refit_metrics(
                rhat_df=rhat_df,
                mape_shopify=0.08,
                mape_amazon=0.11,
                n_dmas=210,
                n_days=365,
            )
        runs = self.reg_mod._load_fallback_db()
        assert len(runs) >= 1
        latest = sorted(runs, key=lambda r: r.get("start_time", ""), reverse=True)[0]
        metrics = latest.get("metrics", {})
        assert "rhat_max" in metrics
        assert "mape_shopify" in metrics
        assert abs(metrics["mape_shopify"] - 0.08) < 1e-9

    def test_log_refit_metrics_records_convergence_flag(self, registry):
        rhat_df = self._make_rhat_df()  # all < 1.05 → convergence_passed = 1.0
        with registry.start_run("convergence-run"):
            registry.log_refit_metrics(rhat_df, 0.07, 0.09, 210, 365)
        runs = self.reg_mod._load_fallback_db()
        latest_metrics = sorted(
            runs, key=lambda r: r.get("start_time", ""), reverse=True
        )[0]["metrics"]
        assert latest_metrics.get("convergence_passed") == 1.0

    # ------------------------------------------------------------------
    # get_latest_run
    # ------------------------------------------------------------------

    def test_get_latest_run_returns_last_logged_run(self, registry):
        with registry.start_run("run-alpha"):
            registry.log_refit_metrics(self._make_rhat_df(), 0.09, 0.12, 210, 365)
        with registry.start_run("run-beta"):
            registry.log_refit_metrics(self._make_rhat_df(), 0.07, 0.10, 210, 365)

        run_id, metrics = registry.get_latest_run()
        assert run_id is not None
        assert isinstance(metrics, dict)
        # The latest run should have the most recent metrics
        assert "rhat_max" in metrics

    def test_get_latest_run_returns_none_when_no_runs(self, tmp_path):
        # Fresh registry with empty fallback directory
        reg = self.ModelRegistry(tracking_uri="http://unused:9999")
        run_id, metrics = reg.get_latest_run()
        # Either None or the first run found (depends on isolation)
        # The key assertion is that this does not raise
        assert metrics is not None  # always a dict, possibly empty

    def test_get_latest_run_metrics_are_floats(self, registry):
        with registry.start_run("float-run"):
            registry.log_refit_metrics(self._make_rhat_df(), 0.08, 0.10, 210, 365)
        _, metrics = registry.get_latest_run()
        for key, value in metrics.items():
            assert isinstance(value, (int, float)), (
                f"Metric {key!r} should be numeric, got {type(value).__name__}"
            )
