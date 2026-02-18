"""Calibration events management for the Meridian MMM engine.

Calibration events encode the results of geo holdout and conversion lift tests
into constraints that Meridian's ROI calibration mechanism can consume.  This
makes the Bayesian posterior update incorporate real-world lift measurements
rather than relying solely on observational data.

PRD Reference: Section 7.4 — Lift Test Calibration Feed

Typical usage
-------------
>>> from incrementality.mmm.calibration import CalibrationEvent, CalibrationStore
>>> store = CalibrationStore("data/calibration_events.json")
>>> event = CalibrationEvent(
...     channel="meta_perf",
...     test_type="geo_holdout",
...     test_start_date=date(2025, 1, 6),
...     test_end_date=date(2025, 2, 2),
...     lift_abs=142_000.0,
...     lift_lower_90=98_000.0,
...     lift_upper_90=186_000.0,
...     spend_in_period=218_000.0,
...     implied_roi=0.65,
...     outcome="shopify",
...     geo_scope="national",
... )
>>> store.add_event(event)
>>> constraints = store.to_meridian_constraints(store.get_events(channel="meta_perf"))
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CalibrationEvent
# ---------------------------------------------------------------------------


class CalibrationEvent(BaseModel):
    """A single lift-test calibration result from PRD Section 7.4.

    Each event encodes one completed geo holdout or conversion lift test.
    The *lift_abs* value (with its 90 % credible interval) becomes a
    direct constraint inside Meridian's ROI calibration module.

    Parameters
    ----------
    channel : str
        The media channel tested (must match a channel in ``MeridianConfig.channels``).
    test_type : str
        Type of test.  One of ``"geo_holdout"``, ``"conversion_lift"``,
        ``"matched_market"``, ``"ghost_ad"``.
    test_start_date : date
        First day the test was live (inclusive).
    test_end_date : date
        Last day the test was live (inclusive).
    lift_abs : float
        Point estimate of incremental revenue (or orders) attributable to the
        channel during the test period.  Positive = channel drove uplift.
    lift_lower_90 : float
        Lower bound of the 90 % confidence / credible interval on *lift_abs*.
    lift_upper_90 : float
        Upper bound of the 90 % confidence / credible interval on *lift_abs*.
    spend_in_period : float
        Total channel spend (USD) during the test window.  Used to derive
        implied ROI.  Must be > 0.
    implied_roi : float
        Incremental revenue per dollar of spend = ``lift_abs / spend_in_period``.
        Computed automatically when not supplied; manual overrides accepted.
    outcome : str
        KPI the lift was measured against: ``"shopify"`` or ``"amazon"``.
    geo_scope : str
        Description of the geo coverage for this test, e.g. ``"national"``,
        ``"southeast"``, a comma-separated DMA list, etc.
    notes : str, optional
        Free-text annotation (test methodology notes, data quality flags, etc.).
    ingested_at : datetime
        UTC timestamp when this event was ingested into the store.
        Auto-populated on creation.
    """

    channel: str = Field(description="Media channel identifier.")
    test_type: str = Field(
        description=(
            "Test methodology: 'geo_holdout', 'conversion_lift', "
            "'matched_market', or 'ghost_ad'."
        )
    )
    test_start_date: date = Field(description="Test start date (inclusive).")
    test_end_date: date = Field(description="Test end date (inclusive).")
    lift_abs: float = Field(description="Incremental revenue point estimate (USD).")
    lift_lower_90: float = Field(
        description="Lower 90 % CI bound on incremental revenue."
    )
    lift_upper_90: float = Field(
        description="Upper 90 % CI bound on incremental revenue."
    )
    spend_in_period: float = Field(
        description="Total channel spend (USD) during test window.", gt=0.0
    )
    implied_roi: float = Field(
        default=0.0,
        description="Incremental revenue per dollar of spend (auto-computed if 0).",
    )
    outcome: str = Field(
        default="shopify",
        description="KPI the lift was measured on: 'shopify' or 'amazon'.",
    )
    geo_scope: str = Field(
        default="national",
        description="Geographic coverage of the test (description or DMA codes).",
    )
    notes: str = Field(
        default="",
        description="Free-text annotation for this event.",
    )
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of ingestion (auto-populated).",
    )

    # -----------------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------------

    @field_validator("test_type")
    @classmethod
    def _validate_test_type(cls, v: str) -> str:
        allowed = {"geo_holdout", "conversion_lift", "matched_market", "ghost_ad"}
        if v not in allowed:
            raise ValueError(
                f"test_type must be one of {sorted(allowed)}, got '{v}'."
            )
        return v

    @field_validator("outcome")
    @classmethod
    def _validate_outcome(cls, v: str) -> str:
        allowed = {"shopify", "amazon", "combined"}
        if v not in allowed:
            raise ValueError(
                f"outcome must be one of {sorted(allowed)}, got '{v}'."
            )
        return v

    @model_validator(mode="after")
    def _validate_dates(self) -> "CalibrationEvent":
        if self.test_end_date < self.test_start_date:
            raise ValueError(
                f"test_end_date ({self.test_end_date}) must not precede "
                f"test_start_date ({self.test_start_date})."
            )
        return self

    @model_validator(mode="after")
    def _validate_ci(self) -> "CalibrationEvent":
        if self.lift_lower_90 > self.lift_upper_90:
            raise ValueError(
                f"lift_lower_90 ({self.lift_lower_90}) must be <= "
                f"lift_upper_90 ({self.lift_upper_90})."
            )
        return self

    @model_validator(mode="after")
    def _compute_implied_roi(self) -> "CalibrationEvent":
        """Auto-compute implied_roi from lift_abs / spend_in_period if not set."""
        if self.implied_roi == 0.0 and self.spend_in_period > 0.0:
            # Use object.__setattr__ because pydantic models may be frozen
            object.__setattr__(
                self,
                "implied_roi",
                round(self.lift_abs / self.spend_in_period, 4),
            )
        return self

    # -----------------------------------------------------------------------
    # Serialisation helpers
    # -----------------------------------------------------------------------

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict with JSON-serialisable types (dates as ISO strings)."""
        data = self.model_dump()
        data["test_start_date"] = data["test_start_date"].isoformat()
        data["test_end_date"] = data["test_end_date"].isoformat()
        data["ingested_at"] = data["ingested_at"].isoformat()
        return data

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "CalibrationEvent":
        """Deserialise from a dict produced by :meth:`to_json_dict`."""
        data = dict(data)
        data["test_start_date"] = date.fromisoformat(data["test_start_date"])
        data["test_end_date"] = date.fromisoformat(data["test_end_date"])
        data["ingested_at"] = datetime.fromisoformat(data["ingested_at"])
        return cls(**data)

    @property
    def duration_days(self) -> int:
        """Number of days the test ran (inclusive of both endpoints)."""
        return (self.test_end_date - self.test_start_date).days + 1

    model_config = {"frozen": False}


# ---------------------------------------------------------------------------
# CalibrationStore
# ---------------------------------------------------------------------------


class CalibrationStore:
    """Persistent store for :class:`CalibrationEvent` objects.

    Events are stored as a JSON-lines file (one JSON object per line) at
    *storage_path*.  The file is created on first write.  All read operations
    parse the entire file; this is acceptable given the expected volume of a
    few hundred events per year.

    Parameters
    ----------
    storage_path : str or Path
        Path to the JSON-lines storage file.
        Default: ``"data/calibration_events.json"``.
    """

    DEFAULT_STORAGE_PATH = "data/calibration_events.json"

    def __init__(
        self,
        storage_path: Union[str, Path] = DEFAULT_STORAGE_PATH,
    ) -> None:
        self.storage_path = Path(storage_path)
        logger.info("CalibrationStore initialised at %s.", self.storage_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_event(self, event: CalibrationEvent) -> None:
        """Append a calibration event to the store.

        Parameters
        ----------
        event : CalibrationEvent
            The event to persist.  The ``ingested_at`` timestamp is set to
            ``datetime.utcnow()`` at validation time if not explicitly provided.
        """
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        existing = self._load_all_raw()
        existing.append(event.to_json_dict())

        with open(self.storage_path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, ensure_ascii=False)

        logger.info(
            "CalibrationEvent added: channel=%s, test_type=%s, dates=%s → %s.",
            event.channel,
            event.test_type,
            event.test_start_date,
            event.test_end_date,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_events(
        self,
        channel: Optional[str] = None,
        after_date: Optional[date] = None,
    ) -> List[CalibrationEvent]:
        """Retrieve stored calibration events with optional filtering.

        Parameters
        ----------
        channel : str, optional
            If provided, only events for this channel are returned.
        after_date : date, optional
            If provided, only events with ``test_end_date >= after_date``
            are returned.

        Returns
        -------
        list[CalibrationEvent]
            Matching events sorted by ``test_start_date`` ascending.
        """
        events = [CalibrationEvent.from_json_dict(d) for d in self._load_all_raw()]

        if channel is not None:
            events = [e for e in events if e.channel == channel]

        if after_date is not None:
            events = [e for e in events if e.test_end_date >= after_date]

        events.sort(key=lambda e: e.test_start_date)
        logger.debug(
            "get_events(channel=%s, after_date=%s) → %d events.",
            channel,
            after_date,
            len(events),
        )
        return events

    def get_latest_per_channel(self) -> Dict[str, CalibrationEvent]:
        """Return the most recent calibration event for each channel.

        "Most recent" is defined as the event with the latest
        ``test_end_date``.  If multiple events share the same latest
        ``test_end_date``, the one with the latest ``ingested_at`` wins.

        Returns
        -------
        dict[str, CalibrationEvent]
            Mapping ``channel_name → CalibrationEvent``.
        """
        all_events = self.get_events()
        latest: Dict[str, CalibrationEvent] = {}

        for event in all_events:
            existing = latest.get(event.channel)
            if existing is None:
                latest[event.channel] = event
            elif event.test_end_date > existing.test_end_date:
                latest[event.channel] = event
            elif (
                event.test_end_date == existing.test_end_date
                and event.ingested_at > existing.ingested_at
            ):
                latest[event.channel] = event

        logger.debug(
            "get_latest_per_channel() → %d channels with events.", len(latest)
        )
        return latest

    # ------------------------------------------------------------------
    # Meridian constraint export
    # ------------------------------------------------------------------

    def to_meridian_constraints(
        self,
        events: List[CalibrationEvent],
    ) -> List[Dict[str, Any]]:
        """Convert a list of CalibrationEvents to Meridian ROI constraint dicts.

        Each dict in the returned list can be passed directly to
        ``ModelSpec.add_roi_calibration(**constraint)`` in the Meridian API.

        The Meridian ROI calibration format requires:
          - ``lift_abs``         : point estimate of incremental revenue
          - ``lift_lower``       : lower bound of the credible interval
          - ``lift_upper``       : upper bound of the credible interval
          - ``channel``          : channel index or name
          - ``outcome``          : outcome index or name
          - ``spend_in_period``  : spend during the test window (USD)
          - ``start_date``       : test start (ISO string)
          - ``end_date``         : test end (ISO string)
          - ``geo_scope``        : geographic scope descriptor

        Parameters
        ----------
        events : list[CalibrationEvent]
            The events to convert.  Typically obtained from
            :meth:`get_events` or :meth:`get_latest_per_channel`.

        Returns
        -------
        list[dict]
            List of constraint dicts, one per event, ready for Meridian.
        """
        constraints: List[Dict[str, Any]] = []

        for event in events:
            constraint = {
                "lift_abs": event.lift_abs,
                "lift_lower": event.lift_lower_90,
                "lift_upper": event.lift_upper_90,
                "channel": event.channel,
                "outcome": event.outcome,
                "spend_in_period": event.spend_in_period,
                "start_date": event.test_start_date.isoformat(),
                "end_date": event.test_end_date.isoformat(),
                "geo_scope": event.geo_scope,
                "implied_roi": event.implied_roi,
                "test_type": event.test_type,
                "notes": event.notes,
            }
            constraints.append(constraint)

        logger.info(
            "to_meridian_constraints: converted %d events → %d constraints.",
            len(events),
            len(constraints),
        )
        return constraints

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def load_from_list(self, events: List[CalibrationEvent]) -> None:
        """Bulk-load a list of events, replacing any existing store contents.

        Parameters
        ----------
        events : list[CalibrationEvent]
            Events to write.  Existing store contents are overwritten.
        """
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        raw = [e.to_json_dict() for e in events]
        with open(self.storage_path, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2, ensure_ascii=False)
        logger.info("CalibrationStore bulk-loaded %d events.", len(events))

    def clear(self) -> None:
        """Delete all events from the store (writes an empty JSON array)."""
        if self.storage_path.exists():
            with open(self.storage_path, "w", encoding="utf-8") as fh:
                json.dump([], fh)
            logger.warning("CalibrationStore cleared at %s.", self.storage_path)

    def __len__(self) -> int:
        """Return the total number of stored events."""
        return len(self._load_all_raw())

    def __repr__(self) -> str:
        return (
            f"CalibrationStore(storage_path={str(self.storage_path)!r}, "
            f"n_events={len(self)})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_all_raw(self) -> List[Dict[str, Any]]:
        """Load all raw JSON records from disk.  Returns [] if file absent."""
        if not self.storage_path.exists():
            return []
        with open(self.storage_path, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                logger.error(
                    "Failed to parse calibration store at %s: %s",
                    self.storage_path,
                    exc,
                )
                return []
        if not isinstance(data, list):
            logger.error(
                "Calibration store at %s has unexpected format (not a JSON array).",
                self.storage_path,
            )
            return []
        return data
