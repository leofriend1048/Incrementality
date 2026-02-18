"""Slack and PagerDuty alerting for the Michael Todd Beauty MMM platform.

Implements every alert rule from PRD Section 9.3 and formats all Slack
messages as rich Block Kit payloads with colour-coded severity attachments.

Typical usage
-------------
>>> from incrementality.alerts import AlertManager, AlertSeverity
>>> am = AlertManager(
...     slack_webhook_url="https://hooks.slack.com/services/XXX/YYY/ZZZ",
...     pagerduty_routing_key="R01234ABCDEF",
... )
>>> am.check_rhat_convergence(rhat_df)
>>> am.check_posterior_mape(mape_7d=0.135)
"""

from __future__ import annotations

import enum
import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------


class AlertSeverity(str, enum.Enum):
    """Alert severity levels (maps to PagerDuty severity and Slack colour)."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    INFO = "info"


# ---------------------------------------------------------------------------
# Colour / emoji maps
# ---------------------------------------------------------------------------

_SEVERITY_COLOR: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: "#FF0000",  # red
    AlertSeverity.HIGH: "#FF8800",      # orange
    AlertSeverity.MEDIUM: "#FFCC00",    # yellow
    AlertSeverity.INFO: "#36A64F",      # green
}

_SEVERITY_EMOJI: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: ":rotating_light:",
    AlertSeverity.HIGH: ":warning:",
    AlertSeverity.MEDIUM: ":large_yellow_circle:",
    AlertSeverity.INFO: ":information_source:",
}

# PagerDuty severity labels expected by the Events API v2
_PD_SEVERITY_MAP: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: "critical",
    AlertSeverity.HIGH: "error",
    AlertSeverity.MEDIUM: "warning",
    AlertSeverity.INFO: "info",
}

# PagerDuty Events API v2 endpoint
_PD_EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"

# Default HTTP timeout (seconds)
_HTTP_TIMEOUT = 10


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class AlertManager:
    """Centralised alert dispatcher for Slack and PagerDuty.

    Parameters
    ----------
    slack_webhook_url : str
        Incoming Webhook URL for the target Slack workspace / channel.
    pagerduty_routing_key : str, optional
        PagerDuty integration routing key (Events API v2).  When absent
        PagerDuty alerts are silently skipped with a log warning.
    """

    def __init__(
        self,
        slack_webhook_url: str,
        pagerduty_routing_key: Optional[str] = None,
    ) -> None:
        self.slack_webhook_url = slack_webhook_url
        self.pagerduty_routing_key = pagerduty_routing_key

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "AlertManager":
        """Construct an AlertManager from a configuration dictionary.

        Expected keys
        -------------
        slack_webhook_url : str
        pagerduty_routing_key : str (optional)

        Example
        -------
        >>> am = AlertManager.from_config({
        ...     "slack_webhook_url": "https://hooks.slack.com/...",
        ...     "pagerduty_routing_key": "RXXXXXXX",
        ... })
        """
        return cls(
            slack_webhook_url=config_dict["slack_webhook_url"],
            pagerduty_routing_key=config_dict.get("pagerduty_routing_key"),
        )

    # ------------------------------------------------------------------
    # Low-level send helpers
    # ------------------------------------------------------------------

    def send_slack(
        self,
        message: str,
        severity: AlertSeverity,
        channel: str = "#mmm-alerts",
        fields: Optional[List[Dict[str, str]]] = None,
        title: Optional[str] = None,
    ) -> bool:
        """Send a formatted Block Kit message to Slack.

        Parameters
        ----------
        message : str
            Primary alert body text.
        severity : AlertSeverity
            Determines the colour bar and header emoji.
        channel : str
            Slack channel override (default ``#mmm-alerts``).
        fields : list of dict, optional
            Additional structured fields shown as ``{title: value}`` pairs.
        title : str, optional
            Override the block title.  Defaults to the severity label.

        Returns
        -------
        bool
            True if Slack accepted the message (2xx response).
        """
        emoji = _SEVERITY_EMOJI[severity]
        color = _SEVERITY_COLOR[severity]
        header_text = title or f"{emoji} MMM Alert — {severity.value.upper()}"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Build Block Kit payload
        blocks: List[Dict[str, Any]] = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": header_text, "emoji": True},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": message},
            },
        ]

        # Optional structured fields section
        if fields:
            field_elements = [
                {
                    "type": "mrkdwn",
                    "text": f"*{f['title']}*\n{f['value']}",
                }
                for f in fields
            ]
            # Slack limits fields to 10 per section; chunk if needed
            for i in range(0, len(field_elements), 10):
                blocks.append(
                    {
                        "type": "section",
                        "fields": field_elements[i : i + 10],
                    }
                )

        blocks.append(
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"_Sent at {ts} | Michael Todd Beauty MMM_"}
                ],
            }
        )

        payload: Dict[str, Any] = {
            "channel": channel,
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks,
                    "fallback": f"[{severity.value.upper()}] {message[:200]}",
                }
            ],
        }

        try:
            resp = requests.post(
                self.slack_webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            logger.info("Slack alert sent: severity=%s, channel=%s", severity.value, channel)
            return True
        except requests.RequestException as exc:
            logger.error("Failed to send Slack alert: %s", exc)
            return False

    def send_pagerduty(
        self,
        title: str,
        body: str,
        severity: AlertSeverity,
        dedup_key: Optional[str] = None,
        custom_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Trigger a PagerDuty incident via the Events API v2.

        Parameters
        ----------
        title : str
            Short incident summary (visible in PagerDuty and mobile push).
        body : str
            Detailed description of the incident.
        severity : AlertSeverity
            Maps to PagerDuty severity (critical, error, warning, info).
        dedup_key : str, optional
            Deduplication key to prevent duplicate incidents.
        custom_details : dict, optional
            Arbitrary key/value pairs attached to the PagerDuty alert.

        Returns
        -------
        bool
            True if PagerDuty accepted the event (2xx response).
        """
        if not self.pagerduty_routing_key:
            logger.warning(
                "PagerDuty routing key not configured; skipping PD alert: %s", title
            )
            return False

        pd_payload: Dict[str, Any] = {
            "routing_key": self.pagerduty_routing_key,
            "event_action": "trigger",
            "dedup_key": dedup_key or f"mtb-mmm-{title[:50]}",
            "payload": {
                "summary": title,
                "severity": _PD_SEVERITY_MAP[severity],
                "source": "MTB-Meridian-MMM",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "custom_details": {
                    "body": body,
                    **(custom_details or {}),
                },
            },
        }

        try:
            resp = requests.post(
                _PD_EVENTS_URL,
                data=json.dumps(pd_payload),
                headers={"Content-Type": "application/json"},
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            logger.info(
                "PagerDuty alert triggered: severity=%s, title=%s", severity.value, title
            )
            return True
        except requests.RequestException as exc:
            logger.error("Failed to send PagerDuty alert: %s", exc)
            return False

    # ------------------------------------------------------------------
    # PRD Section 9.3 alert rules
    # ------------------------------------------------------------------

    def check_mroi_drop(
        self,
        current_mroi: Dict[str, float],
        previous_mroi: Dict[str, float],
        threshold_pct: float = 20.0,
        channel: str = "#mmm-alerts",
    ) -> List[str]:
        """Alert when marginal ROI drops > 20 % week-over-week.

        PRD Section 9.3 — mROI drops >20% WoW → HIGH.

        Parameters
        ----------
        current_mroi : dict
            Channel → current week marginal ROI.
        previous_mroi : dict
            Channel → previous week marginal ROI.
        threshold_pct : float
            Drop percentage that triggers the alert (default 20 %).

        Returns
        -------
        list of str
            Names of channels that triggered the alert.
        """
        triggered: List[str] = []
        for ch, curr in current_mroi.items():
            prev = previous_mroi.get(ch)
            if prev is None or prev == 0:
                continue
            drop_pct = (prev - curr) / abs(prev) * 100
            if drop_pct > threshold_pct:
                triggered.append(ch)
                msg = (
                    f"*Marginal ROI drop detected for `{ch}`.*\n"
                    f"Previous mROI: `{prev:.3f}` → Current mROI: `{curr:.3f}` "
                    f"(drop of `{drop_pct:.1f}%`, threshold `{threshold_pct:.0f}%`)."
                )
                self.send_slack(
                    message=msg,
                    severity=AlertSeverity.HIGH,
                    channel=channel,
                    title=f":warning: mROI Drop — {ch}",
                    fields=[
                        {"title": "Channel", "value": ch},
                        {"title": "Previous mROI", "value": f"{prev:.4f}"},
                        {"title": "Current mROI", "value": f"{curr:.4f}"},
                        {"title": "Drop %", "value": f"{drop_pct:.1f}%"},
                    ],
                )
                logger.warning("mROI drop alert: channel=%s, drop=%.1f%%", ch, drop_pct)
        return triggered

    def check_nb_mmm_delta(
        self,
        nb_attribution: Dict[str, float],
        mmm_attribution: Dict[str, float],
        threshold_pct: float = 40.0,
        channel: str = "#mmm-alerts",
    ) -> List[str]:
        """Alert when NorthBeam vs. MMM attribution delta exceeds 40 %.

        PRD Section 9.3 — NB vs. MMM delta >40% → HIGH.

        Parameters
        ----------
        nb_attribution : dict
            Channel → NorthBeam attributed revenue.
        mmm_attribution : dict
            Channel → MMM attributed revenue.

        Returns
        -------
        list of str
            Channels where the delta exceeded the threshold.
        """
        triggered: List[str] = []
        for ch in set(nb_attribution) | set(mmm_attribution):
            nb_val = nb_attribution.get(ch, 0.0)
            mmm_val = mmm_attribution.get(ch, 0.0)
            baseline = max(abs(nb_val), abs(mmm_val), 1.0)
            delta_pct = abs(nb_val - mmm_val) / baseline * 100
            if delta_pct > threshold_pct:
                triggered.append(ch)
                msg = (
                    f"*NorthBeam vs. MMM attribution gap for `{ch}`.*\n"
                    f"NB: `${nb_val:,.0f}` | MMM: `${mmm_val:,.0f}` "
                    f"— delta `{delta_pct:.1f}%` exceeds `{threshold_pct:.0f}%` threshold."
                )
                self.send_slack(
                    message=msg,
                    severity=AlertSeverity.HIGH,
                    channel=channel,
                    title=f":warning: NB vs. MMM Delta — {ch}",
                    fields=[
                        {"title": "Channel", "value": ch},
                        {"title": "NorthBeam ($)", "value": f"${nb_val:,.0f}"},
                        {"title": "MMM ($)", "value": f"${mmm_val:,.0f}"},
                        {"title": "Delta %", "value": f"{delta_pct:.1f}%"},
                    ],
                )
        return triggered

    def check_posterior_mape(
        self,
        mape_7d: float,
        threshold_pct: float = 12.0,
        channel: str = "#mmm-alerts",
        trigger_refit: bool = True,
    ) -> bool:
        """Alert when 7-day rolling posterior MAPE exceeds 12 %.

        PRD Section 9.3 — MAPE >12% → HIGH + trigger refit.

        Parameters
        ----------
        mape_7d : float
            7-day rolling mean absolute percentage error (0–1 scale).
        threshold_pct : float
            MAPE threshold (default 12 %, expressed as a fraction 0.12).

        Returns
        -------
        bool
            True if the alert was triggered.
        """
        # Accept either 0-100 or 0-1 scale
        mape_val = mape_7d if mape_7d > 1 else mape_7d * 100
        threshold_val = threshold_pct if threshold_pct > 1 else threshold_pct * 100

        if mape_val > threshold_val:
            refit_note = "\n>:arrows_counterclockwise: *Model refit triggered automatically.*" if trigger_refit else ""
            msg = (
                f"*Posterior predictive MAPE is elevated.*\n"
                f"7-day rolling MAPE: `{mape_val:.1f}%` (threshold `{threshold_val:.0f}%`)."
                f"{refit_note}"
            )
            self.send_slack(
                message=msg,
                severity=AlertSeverity.HIGH,
                channel=channel,
                title=":warning: High Posterior MAPE",
                fields=[
                    {"title": "7d MAPE", "value": f"{mape_val:.2f}%"},
                    {"title": "Threshold", "value": f"{threshold_val:.0f}%"},
                    {"title": "Action", "value": "Refit triggered" if trigger_refit else "Manual review"},
                ],
            )
            logger.warning("Posterior MAPE alert: mape=%.2f%%, threshold=%.0f%%", mape_val, threshold_val)
            return True
        return False

    def check_rhat_convergence(
        self,
        rhat_df: pd.DataFrame,
        threshold: float = 1.05,
        channel: str = "#mmm-alerts",
    ) -> bool:
        """Alert when any R-hat statistic exceeds 1.05.

        PRD Section 9.3 — R-hat > 1.05 → CRITICAL + block dashboard.

        Parameters
        ----------
        rhat_df : pd.DataFrame
            DataFrame with columns ``["parameter", "rhat"]`` (or similar).
        threshold : float
            R-hat threshold (default 1.05 per PRD).

        Returns
        -------
        bool
            True if the alert was triggered (convergence failure).
        """
        rhat_col = "rhat" if "rhat" in rhat_df.columns else rhat_df.columns[-1]
        param_col = "parameter" if "parameter" in rhat_df.columns else rhat_df.columns[0]

        failed = rhat_df[rhat_df[rhat_col] > threshold]
        if failed.empty:
            return False

        n_failed = len(failed)
        max_rhat = float(failed[rhat_col].max())
        worst_param = str(failed.loc[failed[rhat_col].idxmax(), param_col])

        msg = (
            f"*MCMC convergence failure detected.*\n"
            f"`{n_failed}` parameter(s) have R-hat > `{threshold:.2f}`.\n"
            f"Worst offender: `{worst_param}` (R-hat = `{max_rhat:.4f}`).\n\n"
            f":no_entry: *Dashboard has been blocked pending refit.*"
        )

        failed[[param_col, rhat_col]].head(10).to_dict("records")
        fields = [
            {"title": "# Failed params", "value": str(n_failed)},
            {"title": "Max R-hat", "value": f"{max_rhat:.4f}"},
            {"title": "Threshold", "value": str(threshold)},
            {"title": "Worst param", "value": worst_param},
        ]

        self.send_slack(
            message=msg,
            severity=AlertSeverity.CRITICAL,
            channel=channel,
            title=":rotating_light: MCMC Convergence Failure",
            fields=fields,
        )
        self.send_pagerduty(
            title="MTB MMM — MCMC Convergence Failure",
            body=(
                f"{n_failed} parameters exceed R-hat threshold {threshold}. "
                f"Worst: {worst_param} = {max_rhat:.4f}. Dashboard blocked."
            ),
            severity=AlertSeverity.CRITICAL,
            dedup_key="mtb-mmm-rhat-convergence",
            custom_details={
                "n_failed": n_failed,
                "max_rhat": max_rhat,
                "worst_param": worst_param,
                "threshold": threshold,
            },
        )
        logger.critical(
            "R-hat convergence alert: %d params > %.2f, worst=%s (%.4f)",
            n_failed, threshold, worst_param, max_rhat,
        )
        return True

    def check_amazon_revenue_drop(
        self,
        current_week_rev: float,
        prior_week_rev: float,
        threshold_pct: float = 15.0,
        channel: str = "#mmm-alerts",
    ) -> bool:
        """Alert when Amazon revenue drops > 15 % week-over-week.

        PRD Section 9.3 — Amazon revenue drop >15% WoW → HIGH.

        Returns
        -------
        bool
            True if the alert was triggered.
        """
        if prior_week_rev <= 0:
            return False

        drop_pct = (prior_week_rev - current_week_rev) / prior_week_rev * 100
        if drop_pct > threshold_pct:
            msg = (
                f"*Amazon revenue WoW decline detected.*\n"
                f"Prior week: `${prior_week_rev:,.0f}` → Current week: `${current_week_rev:,.0f}` "
                f"(drop of `{drop_pct:.1f}%`, threshold `{threshold_pct:.0f}%`)."
            )
            self.send_slack(
                message=msg,
                severity=AlertSeverity.HIGH,
                channel=channel,
                title=":warning: Amazon Revenue Drop",
                fields=[
                    {"title": "Prior Week Revenue", "value": f"${prior_week_rev:,.0f}"},
                    {"title": "Current Week Revenue", "value": f"${current_week_rev:,.0f}"},
                    {"title": "WoW Drop", "value": f"{drop_pct:.1f}%"},
                    {"title": "Threshold", "value": f"{threshold_pct:.0f}%"},
                ],
            )
            logger.warning(
                "Amazon revenue drop alert: prior=%.0f, current=%.0f, drop=%.1f%%",
                prior_week_rev, current_week_rev, drop_pct,
            )
            return True
        return False

    def check_channel_beta_decay(
        self,
        current_beta: float,
        rolling_90d_beta: float,
        channel_name: str,
        threshold_pct: float = 25.0,
        slack_channel: str = "#mmm-alerts",
    ) -> bool:
        """Alert when a channel's beta drops > 25 % vs. its 90-day rolling average.

        PRD Section 9.3 — beta drop >25% vs. 90d rolling → MEDIUM.

        Returns
        -------
        bool
            True if the alert was triggered.
        """
        if rolling_90d_beta == 0:
            return False

        drop_pct = (rolling_90d_beta - current_beta) / abs(rolling_90d_beta) * 100
        if drop_pct > threshold_pct:
            msg = (
                f"*Channel effectiveness (beta) decay detected for `{channel_name}`.*\n"
                f"90-day avg beta: `{rolling_90d_beta:.4f}` → Current beta: `{current_beta:.4f}` "
                f"(drop of `{drop_pct:.1f}%`, threshold `{threshold_pct:.0f}%`).\n"
                f"Consider reviewing creative fatigue or audience saturation."
            )
            self.send_slack(
                message=msg,
                severity=AlertSeverity.MEDIUM,
                channel=slack_channel,
                title=f":large_yellow_circle: Beta Decay — {channel_name}",
                fields=[
                    {"title": "Channel", "value": channel_name},
                    {"title": "90d Rolling Beta", "value": f"{rolling_90d_beta:.4f}"},
                    {"title": "Current Beta", "value": f"{current_beta:.4f}"},
                    {"title": "Decay %", "value": f"{drop_pct:.1f}%"},
                ],
            )
            logger.warning(
                "Beta decay alert: channel=%s, rolling=%.4f, current=%.4f, drop=%.1f%%",
                channel_name, rolling_90d_beta, current_beta, drop_pct,
            )
            return True
        return False

    def check_pipeline_failure(
        self,
        task_name: str,
        error: Exception,
        slack_channel: str = "#mmm-alerts",
    ) -> None:
        """Alert on any pipeline failure — CRITICAL + PagerDuty.

        PRD Section 9.3 — any pipeline failure → CRITICAL + PagerDuty.

        Parameters
        ----------
        task_name : str
            Name of the failed pipeline task / step.
        error : Exception
            The caught exception.
        """
        tb = traceback.format_exc()
        error_str = str(error)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        msg = (
            f"*Pipeline failure in `{task_name}`.*\n"
            f"Error: `{error_str[:300]}`\n\n"
            f"```\n{tb[:800]}\n```"
        )
        self.send_slack(
            message=msg,
            severity=AlertSeverity.CRITICAL,
            channel=slack_channel,
            title=f":rotating_light: Pipeline Failure — {task_name}",
            fields=[
                {"title": "Task", "value": task_name},
                {"title": "Error Type", "value": type(error).__name__},
                {"title": "Time (UTC)", "value": ts},
            ],
        )
        self.send_pagerduty(
            title=f"MTB MMM Pipeline Failure — {task_name}",
            body=f"Task: {task_name}\nError: {error_str}\n\nTraceback:\n{tb[:1000]}",
            severity=AlertSeverity.CRITICAL,
            dedup_key=f"mtb-mmm-pipeline-{task_name}",
            custom_details={
                "task_name": task_name,
                "error_type": type(error).__name__,
                "error_message": error_str[:500],
            },
        )
        logger.critical(
            "Pipeline failure alert: task=%s, error=%s", task_name, error_str[:200]
        )

    def notify_calibration_event(
        self,
        event: Any,
        slack_channel: str = "#mmm-alerts",
    ) -> None:
        """Notify the team when a new calibration event is added.

        PRD Section 9.3 — new calibration event → INFO.

        Parameters
        ----------
        event : any
            CalibrationEvent object or dict describing the event.
        """
        if hasattr(event, "model_dump"):
            ev_dict = event.model_dump()
        elif isinstance(event, dict):
            ev_dict = event
        else:
            ev_dict = vars(event)

        event_id = ev_dict.get("event_id", "unknown")
        channel_name = ev_dict.get("channel", "unknown")
        lift_estimate = ev_dict.get("lift_estimate", "N/A")
        date_str = str(ev_dict.get("start_date", "N/A"))

        msg = (
            f"*New calibration event registered.*\n"
            f"Event `{event_id}` for channel `{channel_name}` has been added to the "
            f"calibration store.  The model will incorporate this lift estimate on the "
            f"next refit."
        )
        fields = [
            {"title": "Event ID", "value": str(event_id)},
            {"title": "Channel", "value": str(channel_name)},
            {"title": "Lift Estimate", "value": str(lift_estimate)},
            {"title": "Start Date", "value": date_str},
        ]
        self.send_slack(
            message=msg,
            severity=AlertSeverity.INFO,
            channel=slack_channel,
            title=":information_source: New Calibration Event",
            fields=fields,
        )
        logger.info(
            "Calibration event notification: id=%s, channel=%s", event_id, channel_name
        )
