"""LIFT — Command-line interface for geo incrementality testing.

Usage:
    incrementality design --channel facebook --name "FB Q1 Test"
    incrementality design --channel youtube --scope campaign --campaigns "123,456"
    incrementality analyze --test-id test_abc123
    incrementality list
    incrementality demo
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*Pyarrow.*")
_warnings.filterwarnings("ignore", category=FutureWarning)

import click  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from incrementality.config import Config, StatisticalConfig  # noqa: E402
from incrementality.models import AdChannel, MeasurementScope, TestScope  # noqa: E402
from incrementality.ui import (  # noqa: E402
    banner,
    branded_table,
    console,
    done,
    fail,
    info,
    kv,
    lift_value,
    money,
    score_bar,
    section,
    spacer,
    step,
    warning,
)

logger = logging.getLogger("incrementality")


def _setup_logging(verbose: bool) -> None:
    if verbose:
        level = logging.DEBUG
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    else:
        # In normal mode, suppress all logs — the branded UI handles output
        level = logging.WARNING
        fmt = "[%(levelname)s] %(message)s"

    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")

    # Suppress noisy third-party loggers regardless of mode
    for noisy in [
        "tensorflow", "absl", "h5py", "urllib3",
        "google.auth", "google.api_core",
    ]:
        logging.getLogger(noisy).setLevel(logging.ERROR)

    # Suppress TensorFlow C++ logs and scipy warnings
    import os
    import warnings
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*deprecated.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*did not converge.*")


# ── Main Group ────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.option("--config", "config_path", type=click.Path(), default=None,
              help="Path to config YAML file")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, verbose: bool) -> None:
    """LIFT — Geo Incrementality Platform

    Design, run, and analyze geo holdout tests to measure
    true incremental ROAS across Shopify and Amazon.
    """
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    if not config_path:
        # Auto-discover config.yaml in current directory
        for candidate in ["config.yaml", "config.yml"]:
            if Path(candidate).exists():
                config_path = candidate
                break
    if config_path and Path(config_path).exists():
        ctx.obj["config"] = Config.from_yaml(config_path)
    else:
        ctx.obj["config"] = Config()

    if ctx.invoked_subcommand is None:
        banner()
        console.print("  [muted]Run[/muted] [accent]incrementality --help[/accent] [muted]for usage.[/muted]")
        spacer()


# ── Design Command ────────────────────────────────────────────────────────────

@cli.command()
@click.option("--channel", type=click.Choice([
    "facebook", "youtube", "tiktok", "tiktok_shop",
    "pinterest", "applovin", "tatari",
]), required=True, help="Ad channel to test")
@click.option("--scope", type=click.Choice(["channel", "campaign"]),
              default="channel", help="Test scope")
@click.option("--measure", type=click.Choice([
    "shopify_only", "amazon_only", "shopify_and_amazon"
]), default="shopify_and_amazon", help="Revenue to measure")
@click.option("--campaigns", default=None,
              help="Comma-separated campaign IDs (for campaign scope)")
@click.option("--name", default="Incrementality Test",
              help="Test name")
@click.option("--target-mde", type=float, default=None,
              help="Target minimum detectable effect (e.g., 0.10 for 10%%)")
@click.option("--data-dir", type=click.Path(), default=None,
              help="Directory with CSV data files")
@click.option("--lookback-weeks", type=int, default=12,
              help="Weeks of historical data to use")
@click.pass_context
def design(
    ctx: click.Context,
    channel: str,
    scope: str,
    measure: str,
    campaigns: str | None,
    name: str,
    target_mde: float | None,
    data_dir: str | None,
    lookback_weeks: int,
) -> None:
    """Design an optimal incrementality test."""
    from incrementality.orchestrator import TestOrchestrator

    banner()

    config = ctx.obj["config"]
    orchestrator = TestOrchestrator(config)

    ad_channel = AdChannel(channel)
    test_scope = TestScope(scope)
    measurement_scope = MeasurementScope(measure)
    campaign_ids = campaigns.split(",") if campaigns else None

    # Configuration summary
    section("Test Configuration")
    kv("Channel", f"[accent]{channel.title()}[/accent]")
    kv("Scope", scope.replace("_", " ").title())
    kv("Measuring", measure.replace("_", " ").title())
    kv("Lookback", f"{lookback_weeks} weeks")
    if campaign_ids:
        kv("Campaigns", ", ".join(campaign_ids))
    spacer()

    # Load data
    data = None
    if data_dir:
        with step("Loading historical data from CSV"):
            data = orchestrator.load_data_from_csv(data_dir)
    elif not data_dir:
        # Pull from APIs with per-connector progress
        with step("Pulling historical data from APIs"):
            try:
                data = orchestrator.pull_historical_data(
                    lookback_weeks,
                    ad_channel=ad_channel,
                    campaign_ids=campaign_ids,
                )
            except Exception as e:
                warning(f"API pull failed: {e}")
                info("Trying cached data...")
                data = orchestrator.load_cached_data()

        # Show what we got and validate
        if data:
            pull_errors = getattr(orchestrator, "pull_errors", {})
            for src, df in data.items():
                if not df.empty:
                    n_dmas = df["dma_code"].nunique() if "dma_code" in df.columns else 0
                    done(f"{src.title()}: [accent]{len(df)}[/accent] rows, "
                         f"[accent]{n_dmas}[/accent] DMAs")
                else:
                    error_detail = pull_errors.get(src, "")
                    if error_detail:
                        warning(f"{src.title()}: no data — {error_detail}")
                    else:
                        warning(f"{src.title()}: no data (check credentials/config)")

            # Validate the selected channel has data
            channel_key = channel  # facebook or youtube
            if channel_key in data and data[channel_key].empty:
                fail(f"No {channel.title()} spend data returned. "
                     f"Cannot design a {channel.title()} test without spend data.")
                info("Check: (1) API credentials, (2) ad account ID, (3) geo targeting")
                spacer()
                sys.exit(1)

            if data.get("shopify", pd.DataFrame()).empty:
                fail("No Shopify revenue data. Cannot design test.")
                info("Check: (1) access token, (2) shop domain, (3) that orders exist")
                spacer()
                sys.exit(1)
        spacer()

    try:
        with step("Running automatic test design"):
            test_design = orchestrator.design_test(
                ad_channel=ad_channel,
                test_scope=test_scope,
                measurement_scope=measurement_scope,
                campaign_ids=campaign_ids,
                test_name=name,
                target_mde=target_mde,
                data=data,
                lookback_weeks=lookback_weeks,
            )
    except ValueError as e:
        spacer()
        fail(f"Test design failed: {e}")
        info("Try: (1) increase --lookback-weeks, (2) check data quality, "
             "(3) run with --verbose for details")
        spacer()
        sys.exit(1)

    spacer()
    _print_design(test_design)


# ── Analyze Command ───────────────────────────────────────────────────────────

@cli.command()
@click.option("--test-id", required=True, help="Test ID to analyze")
@click.option("--data-dir", type=click.Path(), default=None,
              help="Directory with test period CSV data")
@click.option("--attributed-conversions", type=float, default=0.0,
              help="Platform-reported conversions (from Ads Manager) for IF/CPIA")
@click.pass_context
def analyze(ctx: click.Context, test_id: str, data_dir: str | None,
            attributed_conversions: float) -> None:
    """Analyze a completed test and generate the incrementality report."""
    from incrementality.orchestrator import TestOrchestrator

    banner()

    config = ctx.obj["config"]
    orchestrator = TestOrchestrator(config)

    # Load test design with friendly error
    try:
        with step("Loading test design"):
            test_design = orchestrator.load_design(test_id)
        done(f"Loaded [accent]{test_design.name}[/accent]")
    except FileNotFoundError:
        fail(f"Test [accent]{test_id}[/accent] not found.")
        available = orchestrator.list_tests()
        if available:
            info("Available tests:")
            for t in available:
                info(f"  {t['test_id']}: {t['name']} ({t['status']})")
        else:
            info("No tests found. Run [accent]incrementality design[/accent] first.")
        spacer()
        sys.exit(1)

    pre_data = None
    post_data = None
    ad_spend = None

    if data_dir:
        csv_dir = Path(data_dir)
        with step("Loading test period data"):
            pre_path = csv_dir / "pre_period.csv"
            post_path = csv_dir / "post_period.csv"
            spend_path = csv_dir / "ad_spend.csv"
            if pre_path.exists():
                pre_data = pd.read_csv(pre_path, parse_dates=["date"])
            else:
                warning(f"pre_period.csv not found in {data_dir}")
            if post_path.exists():
                post_data = pd.read_csv(post_path, parse_dates=["date"])
            else:
                warning(f"post_period.csv not found in {data_dir}")
            if spend_path.exists():
                ad_spend = pd.read_csv(spend_path, parse_dates=["date"])
            else:
                warning(f"ad_spend.csv not found in {data_dir} — iROAS will be $0")

    try:
        with step("Running causal inference analysis"):
            orchestrator.analyze_test(
                test_design, pre_data, post_data, ad_spend,
                attributed_conversions=attributed_conversions,
            )
    except Exception as e:
        fail(f"Analysis failed: {e}")
        info("Run with --verbose for detailed error info.")
        spacer()
        sys.exit(1)

    spacer()
    report_path = Path(config.output_dir) / f"{test_id}_report.html"
    done(f"JSON + CSV saved to [accent]{config.output_dir}[/accent]")
    done(f"HTML report: [accent]{report_path}[/accent]")
    info("Tip: run [accent]incrementality open {test_id}[/accent] to view the report")
    spacer()


# ── Execute Command ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--test-id", required=True, help="Test ID to deploy")
@click.option("--campaigns", default=None,
              help="Comma-separated campaign IDs (overrides test design)")
@click.option("--update", is_flag=True,
              help="Re-scan for new ad sets and apply holdout exclusions")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def execute(
    ctx: click.Context,
    test_id: str,
    campaigns: str | None,
    update: bool,
    yes: bool,
) -> None:
    """Deploy holdout DMA exclusions to the ad platform.

    Automatically excludes holdout DMAs from ad delivery by updating
    geo-targeting on campaigns via the Facebook/Google Ads API.

    For channel-level tests: applies to ALL active campaigns.
    For campaign-level tests: applies to the specified campaigns.

    Use --update on a running test to catch new campaigns/ad sets
    created after the initial deployment.

    Original targeting is saved so it can be reverted with:
        incrementality revert --test-id <test-id>
    """
    from incrementality.orchestrator import TestOrchestrator

    banner()

    config = ctx.obj["config"]
    orchestrator = TestOrchestrator(config)

    # Load test design
    try:
        with step("Loading test design"):
            test_design = orchestrator.load_design(test_id)
        done(f"Loaded [accent]{test_design.name}[/accent]")
    except FileNotFoundError:
        fail(f"Test [accent]{test_id}[/accent] not found.")
        spacer()
        sys.exit(1)

    # --- Update mode: re-scan for new ad sets on a running test ---
    if update:
        if test_design.status != "running":
            fail("--update only works on running tests.")
            info("Deploy first with [accent]incrementality execute "
                 f"--test-id {test_id}[/accent]")
            spacer()
            sys.exit(1)

        holdout_dmas = test_design.holdout_cell.dma_codes
        n_tracked = len(test_design.original_targeting)

        section("Holdout Update")
        kv("Channel", f"[accent]{test_design.ad_channel.value.title()}[/accent]")
        kv("Holdout DMAs", f"[heading]{len(holdout_dmas)}[/heading]")
        kv("Currently tracked", f"{n_tracked} ad sets")
        spacer()

        info("Scanning for new campaigns/ad sets missing holdout exclusions...")
        spacer()

        if not yes:
            if not click.confirm(click.style(
                "    Apply holdout exclusions to any new ad sets?", bold=True,
            )):
                info("Aborted.")
                spacer()
                return

        try:
            with step("Scanning and updating new ad sets"):
                test_design, n_new = orchestrator.update_holdout(test_design)
        except Exception as e:
            fail(f"Update failed: {e}")
            spacer()
            sys.exit(1)

        spacer()
        if n_new > 0:
            done(f"Applied holdout exclusions to [accent]{n_new}[/accent] new ad sets")
        else:
            done("No new ad sets found — all are already excluded")
        kv("Total tracked", f"{len(test_design.original_targeting)} ad sets")
        spacer()
        return

    # --- Normal deploy mode ---
    if test_design.status == "running":
        warning("This test is already deployed!")
        info("To catch new ad sets, run:")
        info(f"  [accent]incrementality execute --test-id {test_id} --update[/accent]")
        spacer()
        info("To fully re-deploy, revert first:")
        info(f"  [accent]incrementality revert --test-id {test_id}[/accent]")
        spacer()
        sys.exit(1)

    campaign_ids = campaigns.split(",") if campaigns else None
    holdout_dmas = test_design.holdout_cell.dma_codes

    # Show what will happen
    section("Deployment Plan")
    kv("Channel", f"[accent]{test_design.ad_channel.value.title()}[/accent]")
    kv("Holdout DMAs", f"[heading]{len(holdout_dmas)}[/heading]")
    kv("Scope", "All active campaigns" if campaign_ids is None
       else f"Campaigns: {', '.join(campaign_ids)}")
    kv("Duration", f"{test_design.duration_weeks} weeks")
    if test_design.recommended_start_date:
        kv("Start", str(test_design.recommended_start_date))
        kv("End", str(test_design.recommended_end_date))
    spacer()

    warning("This will modify live ad targeting!")
    info(f"  {len(holdout_dmas)} DMAs will be EXCLUDED from "
         f"{test_design.ad_channel.value.title()} ad delivery.")
    info("  Original targeting is saved and can be reverted with:")
    info(f"    [accent]incrementality revert --test-id {test_id}[/accent]")
    spacer()

    if not yes:
        if not click.confirm(click.style("    Proceed with deployment?", bold=True)):
            info("Aborted.")
            spacer()
            return

    try:
        with step("Deploying holdout exclusions"):
            orchestrator.execute_test(test_design, campaign_ids)
    except Exception as e:
        fail(f"Deployment failed: {e}")
        info("No changes were made (or partially applied). Check the ad platform.")
        spacer()
        sys.exit(1)

    spacer()
    done("Holdout deployed successfully!")
    kv("Status", "[ok]● RUNNING[/ok]")
    kv("Deployed at", str(test_design.deployed_at))
    spacer()
    info("When the test ends, revert targeting with:")
    info(f"  [accent]incrementality revert --test-id {test_id}[/accent]")
    spacer()
    info("If you create new campaigns during the test, run:")
    info(f"  [accent]incrementality execute --test-id {test_id} --update[/accent]")
    spacer()


# ── Revert Command ───────────────────────────────────────────────────────────

@cli.command()
@click.option("--test-id", required=True, help="Test ID to revert")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def revert(ctx: click.Context, test_id: str, yes: bool) -> None:
    """Revert holdout DMA exclusions, restoring original ad targeting.

    Run this after the test period ends to restore campaigns to their
    pre-test geo-targeting state.
    """
    from incrementality.orchestrator import TestOrchestrator

    banner()

    config = ctx.obj["config"]
    orchestrator = TestOrchestrator(config)

    try:
        with step("Loading test design"):
            test_design = orchestrator.load_design(test_id)
        done(f"Loaded [accent]{test_design.name}[/accent]")
    except FileNotFoundError:
        fail(f"Test [accent]{test_id}[/accent] not found.")
        spacer()
        sys.exit(1)

    if not test_design.original_targeting:
        fail("No saved targeting state found.")
        info("Was this test deployed with [accent]incrementality execute[/accent]?")
        spacer()
        sys.exit(1)

    if test_design.reverted_at:
        warning(f"This test was already reverted at {test_design.reverted_at}")
        if not click.confirm(click.style("    Revert again?", bold=True)):
            return

    section("Revert Plan")
    kv("Channel", f"[accent]{test_design.ad_channel.value.title()}[/accent]")
    kv("Deployed at", str(test_design.deployed_at))
    if test_design.ad_channel.value == "facebook":
        kv("Ad sets to restore", str(len(test_design.original_targeting)))
    else:
        n_criteria = sum(
            len(v) for v in test_design.original_targeting.values()
            if isinstance(v, list)
        )
        kv("Exclusions to remove", str(n_criteria))
    spacer()

    if not yes:
        if not click.confirm(click.style("    Proceed with revert?", bold=True)):
            info("Aborted.")
            spacer()
            return

    try:
        with step("Reverting to original targeting"):
            orchestrator.revert_test(test_design)
    except Exception as e:
        fail(f"Revert failed: {e}")
        info("Some changes may not have been reverted. Check the ad platform manually.")
        spacer()
        sys.exit(1)

    spacer()
    done("Targeting reverted successfully!")
    kv("Status", "[ok]● COMPLETED[/ok]")
    kv("Reverted at", str(test_design.reverted_at))
    spacer()
    info("You can now analyze the test results:")
    info(f"  [accent]incrementality analyze --test-id {test_id}[/accent]")
    spacer()


# ── List Command ──────────────────────────────────────────────────────────────

@cli.command("list")
@click.pass_context
def list_tests(ctx: click.Context) -> None:
    """List all saved tests."""
    from incrementality.orchestrator import TestOrchestrator

    banner()

    config = ctx.obj["config"]
    orchestrator = TestOrchestrator(config)
    tests = orchestrator.list_tests()

    if not tests:
        info("No tests found. Run [accent]incrementality design[/accent] to create one.")
        spacer()
        return

    section("Tests")

    table = branded_table("", show_header=True)
    table.add_column("Test ID", style="accent")
    table.add_column("Name")
    table.add_column("Status", justify="center")
    table.add_column("Channel", justify="center")
    table.add_column("Duration", justify="right")

    for t in tests:
        status = t["status"]
        if status == "analyzed":
            status_display = "[ok]● analyzed[/ok]"
        elif status == "running":
            status_display = "[accent]● running[/accent]"
        elif status == "designed":
            status_display = "[muted]● designed[/muted]"
        else:
            status_display = f"[muted]● {status}[/muted]"

        table.add_row(
            t["test_id"],
            t["name"],
            status_display,
            t["channel"].title(),
            f"{t['duration_weeks']}w",
        )
    console.print(table)
    spacer()


# ── Demo Command ──────────────────────────────────────────────────────────────

@cli.command()
@click.option("--channel", type=click.Choice(["facebook", "youtube"]),
              default="facebook", help="Channel to simulate")
@click.option("--measure", type=click.Choice([
    "shopify_only", "amazon_only", "shopify_and_amazon"
]), default="shopify_and_amazon", help="Revenue to measure")
@click.option("--true-lift", type=float, default=0.12,
              help="True lift to simulate (e.g., 0.12 for 12%%)")
@click.option("--amazon-halo", type=float, default=0.05,
              help="Amazon halo effect to simulate")
@click.option("--num-dmas", type=int, default=80,
              help="Number of DMAs with data")
@click.option("--weeks", type=int, default=8,
              help="Weeks of historical data")
def demo(
    channel: str,
    measure: str,
    true_lift: float,
    amazon_halo: float,
    num_dmas: int,
    weeks: int,
) -> None:
    """Run a full demo with synthetic data.

    Generates realistic data, designs a test, simulates the test period,
    and produces the full incrementality report.
    """
    from incrementality.orchestrator import TestOrchestrator

    banner()
    section("Demo Mode")
    kv("True Shopify lift", f"[ok]{true_lift:.0%}[/ok]")
    kv("Amazon halo", f"[ok]{amazon_halo:.0%}[/ok]")
    kv("Channel", f"[accent]{channel.title()}[/accent]")
    kv("DMAs", str(num_dmas))
    kv("History", f"{weeks} weeks")
    spacer()

    # Step 1: Generate data
    with step(f"Generating synthetic data ({num_dmas} DMAs × {weeks}w)"):
        data = _generate_synthetic_data(num_dmas, weeks, channel)

    config = Config(
        statistical=StatisticalConfig(
            min_dmas_per_cell=5,
            target_holdout_fraction=0.25,
        ),
    )
    orchestrator = TestOrchestrator(config)

    ad_channel = AdChannel(channel)
    measurement_scope = MeasurementScope(measure)

    # Step 2: Design
    with step("Running automatic test design"):
        test_design = orchestrator.design_test(
            ad_channel=ad_channel,
            measurement_scope=measurement_scope,
            test_name=f"Demo: {channel.title()} Incrementality",
            data=data,
            lookback_weeks=weeks,
            run_simulation=False,
        )

    spacer()
    _print_design(test_design)

    # Step 3: Simulate
    with step("Simulating test period with known lift"):
        pre_data, post_data, spend_data = _simulate_test_period(
            test_design, data, true_lift, amazon_halo, channel,
        )

    # Step 4: Analyze
    with step("Running causal inference (ASCM + BSTS + DiD)"):
        report = orchestrator.analyze_test(
            test_design, pre_data, post_data, spend_data,
        )

    spacer()
    section("Ground Truth Comparison")
    kv("True lift (injected)", f"[ok]{true_lift:.1%}[/ok]")
    kv("Measured lift", lift_value(
        report.incrementality.relative_lift,
        report.incrementality.lift_lower_ci,
        report.incrementality.lift_upper_ci,
        report.incrementality.is_significant,
    ))
    error_pct = abs(report.incrementality.relative_lift - true_lift) / true_lift if true_lift != 0 else 0.0
    kv("Estimation error", f"[muted]{error_pct:.1%}[/muted]")
    spacer()

    done(f"HTML report: [accent]{config.output_dir}/{test_design.test_id}_report.html[/accent]")
    spacer()


# ── Design Printer ────────────────────────────────────────────────────────────

def _print_design(design) -> None:
    """Print a branded test design summary."""
    section("Test Design")

    kv("Test ID", f"[accent]{design.test_id}[/accent]")
    kv("Name", design.name)
    kv("Channel", f"[accent]{design.ad_channel.value.title()}[/accent]")
    kv("Scope", design.test_scope.value.replace("_", " ").title())
    kv("Measurement", design.measurement_scope.value.replace("_", " ").title())
    if design.campaign_ids:
        kv("Campaigns", ", ".join(design.campaign_ids))
    spacer()

    kv("Treatment DMAs", f"[heading]{design.num_treatment_dmas}[/heading]")
    kv("Holdout DMAs", f"[heading]{design.num_holdout_dmas}[/heading]")
    kv("Duration", f"[heading]{design.duration_weeks} weeks[/heading]")
    if design.recommended_start_date:
        kv("Start date", str(design.recommended_start_date))
        kv("End date", str(design.recommended_end_date))
    kv("Balance score", f"{design.balance_score:.3f}")
    spacer()

    # Power analysis
    if design.power_analysis:
        pa = design.power_analysis
        section("Power Analysis")
        kv("Min Detectable Effect", f"[heading]{pa.minimum_detectable_effect:.1%}[/heading]")

        # Power bar
        power_bar = score_bar(pa.statistical_power * 100, 100, 20)
        kv("Statistical Power", f"{power_bar}  {pa.statistical_power:.0%}")

        if pa.simulated_power > 0:
            sim_bar = score_bar(pa.simulated_power * 100, 100, 20)
            kv("Simulated Power", f"{sim_bar}  {pa.simulated_power:.0%}")
            kv("Simulated FPR", f"{pa.simulated_false_positive_rate:.1%}")
            kv("Simulations", str(pa.num_simulations))

        kv("Significance Level", f"{pa.significance_level:.0%}")
        kv("Cohen's d", f"{pa.effect_size_cohen_d:.3f}")

        if pa.power_score > 0:
            ps_bar = score_bar(pa.power_score, 100, 20)
            kv("Power Score", f"{ps_bar}  [heading]{pa.power_score:.0f}/100[/heading]")
        spacer()

    # Feasibility
    if design.feasibility:
        feas = design.feasibility
        section("Feasibility")
        if feas.is_feasible:
            done(f"Test is [ok]FEASIBLE[/ok] (power score: {feas.power_score:.0f}/100)")
        else:
            fail(f"Test is [warn]NOT FEASIBLE[/warn] (power score: {feas.power_score:.0f}/100)")
        kv("Opportunity cost", f"${feas.estimated_opportunity_cost:,.0f}")
        for reason in feas.reasons:
            if reason.startswith("BLOCK:"):
                console.print(f"    [warn]{reason}[/warn]")
            elif reason.startswith("WARNING:"):
                console.print(f"    [yellow]{reason}[/yellow]")
            elif reason.startswith("PASS:"):
                console.print(f"    [ok]{reason}[/ok]")
        if feas.recommendations:
            for rec in feas.recommendations:
                info(f"  {rec}")
        spacer()

    # DMA assignments
    section("DMA Assignments")
    console.print(f"    [label]Treatment ({design.num_treatment_dmas}):[/label]")
    _print_dma_list(design.treatment_cell.dma_codes, indent=6)
    spacer()
    console.print(f"    [label]Holdout ({design.num_holdout_dmas}):[/label]")
    _print_dma_list(design.holdout_cell.dma_codes, indent=6)
    spacer()


def _print_dma_list(codes: list[str], indent: int = 6, max_show: int = 20) -> None:
    """Print a compact DMA code list."""
    pad = " " * indent
    show = codes[:max_show]
    line = ", ".join(show)
    console.print(f"{pad}[muted]{line}[/muted]")
    if len(codes) > max_show:
        console.print(f"{pad}[muted]... and {len(codes) - max_show} more[/muted]")


# ── Synthetic Data Generator ──────────────────────────────────────────────────

def _generate_synthetic_data(
    num_dmas: int,
    weeks: int,
    channel: str,
) -> dict[str, pd.DataFrame]:
    """Generate realistic synthetic DMA-level daily data."""
    from incrementality.dma import get_all_dmas

    rng = np.random.default_rng(seed=42)
    all_dmas = get_all_dmas()[:num_dmas]

    end_date = date.today()
    start_date = end_date - timedelta(weeks=weeks)
    dates = pd.date_range(start_date, end_date, freq="D")

    records_shopify = []
    records_amazon = []
    records_ad = []

    for dma in all_dmas:
        pop_factor = dma.population / 1_000_000
        base_shopify = 500 * pop_factor + rng.normal(0, 50 * pop_factor)
        base_amazon = 200 * pop_factor + rng.normal(0, 30 * pop_factor)
        base_spend = 100 * pop_factor + rng.normal(0, 10 * pop_factor)

        for d in dates:
            dow_effect = 1.0 + 0.15 * (d.dayofweek in [5, 6])
            seasonal = 1.0 + 0.05 * np.sin(2 * np.pi * d.dayofyear / 365)
            noise_s = rng.normal(1.0, 0.15)
            noise_a = rng.normal(1.0, 0.20)
            noise_spend = rng.normal(1.0, 0.10)

            shopify_rev = max(0, base_shopify * dow_effect * seasonal * noise_s)
            amazon_rev = max(0, base_amazon * dow_effect * seasonal * noise_a)
            spend = max(0, base_spend * dow_effect * noise_spend)

            shopify_orders = max(1, int(shopify_rev / (rng.normal(65, 15))))
            amazon_orders = max(1, int(amazon_rev / (rng.normal(45, 10))))

            records_shopify.append({
                "date": d.date(),
                "dma_code": dma.dma_code,
                "revenue": round(shopify_rev, 2),
                "orders": shopify_orders,
            })
            records_amazon.append({
                "date": d.date(),
                "dma_code": dma.dma_code,
                "revenue": round(amazon_rev, 2),
                "orders": amazon_orders,
            })
            records_ad.append({
                "date": d.date(),
                "dma_code": dma.dma_code,
                "spend": round(spend, 2),
                "impressions": max(0, int(spend * rng.normal(100, 20))),
                "clicks": max(0, int(spend * rng.normal(2, 0.5))),
            })

    return {
        "shopify": pd.DataFrame(records_shopify),
        "amazon": pd.DataFrame(records_amazon),
        "facebook" if channel == "facebook" else "youtube": pd.DataFrame(records_ad),
    }


# ── Test Period Simulator ─────────────────────────────────────────────────────

def _simulate_test_period(
    design,
    historical_data: dict[str, pd.DataFrame],
    true_lift: float,
    amazon_halo: float,
    channel: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate test period data with a known treatment effect."""
    rng = np.random.default_rng(seed=123)
    treatment_dmas = set(design.treatment_cell.dma_codes)
    holdout_dmas = set(design.holdout_cell.dma_codes)

    shopify = historical_data.get("shopify", pd.DataFrame())
    amazon = historical_data.get("amazon", pd.DataFrame())
    ad_key = "facebook" if channel == "facebook" else "youtube"
    ad_data = historical_data.get(ad_key, pd.DataFrame())

    if shopify.empty:
        raise ValueError("No Shopify data for simulation")

    shopify["date"] = pd.to_datetime(shopify["date"])
    amazon["date"] = pd.to_datetime(amazon["date"]) if not amazon.empty else amazon
    ad_data["date"] = pd.to_datetime(ad_data["date"]) if not ad_data.empty else ad_data

    min_date = shopify["date"].min()
    max_date = shopify["date"].max()
    total_days = (max_date - min_date).days

    # Split available data: ~60% pre / ~40% test
    test_days = min(total_days * 2 // 5, total_days - 14)
    test_days = max(test_days, 14)

    test_end = max_date
    test_start = test_end - timedelta(days=test_days)
    pre_start = min_date

    pre_mask = (shopify["date"] >= pd.Timestamp(pre_start)) & (shopify["date"] < pd.Timestamp(test_start))
    post_mask = (shopify["date"] >= pd.Timestamp(test_start)) & (shopify["date"] <= pd.Timestamp(test_end))

    pre_shopify = shopify[pre_mask].copy()
    post_shopify = shopify[post_mask].copy()

    # Apply treatment effect
    treatment_mask = post_shopify["dma_code"].isin(treatment_dmas)
    post_shopify.loc[treatment_mask, "revenue"] *= (1 + true_lift)
    post_shopify.loc[treatment_mask, "orders"] = (
        post_shopify.loc[treatment_mask, "orders"] * (1 + true_lift * 0.8)
    ).astype(int)

    pre_data = pre_shopify.rename(columns={"revenue": "shopify_revenue", "orders": "shopify_orders"})
    post_data = post_shopify.rename(columns={"revenue": "shopify_revenue", "orders": "shopify_orders"})

    if not amazon.empty:
        pre_amazon = amazon[
            (amazon["date"] >= pd.Timestamp(pre_start)) & (amazon["date"] < pd.Timestamp(test_start))
        ].copy()
        post_amazon = amazon[
            (amazon["date"] >= pd.Timestamp(test_start)) & (amazon["date"] <= pd.Timestamp(test_end))
        ].copy()

        amazon_treatment_mask = post_amazon["dma_code"].isin(treatment_dmas)
        post_amazon.loc[amazon_treatment_mask, "revenue"] *= (1 + amazon_halo)

        pre_data = pre_data.merge(
            pre_amazon[["date", "dma_code", "revenue", "orders"]].rename(
                columns={"revenue": "amazon_revenue", "orders": "amazon_orders"}
            ),
            on=["date", "dma_code"], how="outer",
        )
        post_data = post_data.merge(
            post_amazon[["date", "dma_code", "revenue", "orders"]].rename(
                columns={"revenue": "amazon_revenue", "orders": "amazon_orders"}
            ),
            on=["date", "dma_code"], how="outer",
        )

    pre_data = pre_data.fillna(0)
    post_data = post_data.fillna(0)

    for df in [pre_data, post_data]:
        rev_cols = [c for c in df.columns if c.endswith("_revenue")]
        df["revenue"] = df[rev_cols].sum(axis=1)

    # Ad spend (holdout gets zero)
    spend_data = pd.DataFrame()
    if not ad_data.empty:
        spend_mask = (ad_data["date"] >= pd.Timestamp(test_start)) & (ad_data["date"] <= pd.Timestamp(test_end))
        spend_data = ad_data[spend_mask].copy()
        holdout_spend_mask = spend_data["dma_code"].isin(holdout_dmas)
        spend_data.loc[holdout_spend_mask, "spend"] = 0
    else:
        records = []
        dates = pd.date_range(test_start, test_end, freq="D")
        for dma in treatment_dmas:
            for d in dates:
                records.append({"date": d, "dma_code": dma, "spend": rng.normal(100, 20)})
        for dma in holdout_dmas:
            for d in dates:
                records.append({"date": d, "dma_code": dma, "spend": 0})
        spend_data = pd.DataFrame(records)

    return pre_data, post_data, spend_data


# ── Setup Command ─────────────────────────────────────────────────────────────

@cli.command()
@click.option("--output", "output_path", default="config.yaml",
              help="Where to save the config file")
def setup(output_path: str) -> None:
    """Interactive setup wizard to configure credentials and test parameters."""
    from incrementality.config import (
        AmazonConfig,
        Config,
        FacebookConfig,
        ShopifyConfig,
        StatisticalConfig,
        YouTubeConfig,
    )

    banner()
    section("Setup Wizard")
    console.print("    [muted]This wizard will walk you through connecting your[/muted]")
    console.print("    [muted]platforms and configuring your first incrementality test.[/muted]")
    spacer()

    shopify_cfg = None
    amazon_cfg = None
    facebook_cfg = None
    youtube_cfg = None

    # ── Shopify (required) ────────────────────────────────────────────
    section("Shopify (required)")
    console.print("    [muted]Your Shopify store is the primary revenue source.[/muted]")
    console.print("    [muted]You need a custom app with read_orders scope.[/muted]")
    spacer()

    shop_domain = click.prompt(
        click.style("    Shop domain", bold=True),
        type=str,
        default="your-store.myshopify.com",
    )
    access_token = click.prompt(
        click.style("    Access token", bold=True),
        type=str,
        hide_input=True,
    )
    api_version = click.prompt(
        click.style("    API version", bold=True),
        type=str,
        default="2025-01",
    )

    shopify_cfg = ShopifyConfig(
        shop_domain=shop_domain,
        access_token=access_token,
        api_version=api_version,
    )
    done(f"Shopify: [accent]{shop_domain}[/accent]")
    spacer()

    # ── Amazon (optional) ─────────────────────────────────────────────
    section("Amazon (optional)")
    console.print("    [muted]Add Amazon to measure cross-platform halo effects.[/muted]")
    spacer()

    if click.confirm(click.style("    Connect Amazon?", bold=True), default=False):
        marketplace_id = click.prompt(
            click.style("    Marketplace ID", bold=True),
            default="ATVPDKIKX0DER",
        )
        seller_id = click.prompt(click.style("    Seller ID", bold=True))
        refresh_token = click.prompt(
            click.style("    Refresh token", bold=True), hide_input=True,
        )
        client_id = click.prompt(click.style("    Client ID", bold=True))
        client_secret = click.prompt(
            click.style("    Client secret", bold=True), hide_input=True,
        )

        amazon_cfg = AmazonConfig(
            marketplace_id=marketplace_id,
            seller_id=seller_id,
            refresh_token=refresh_token,
            client_id=client_id,
            client_secret=client_secret,
        )
        done(f"Amazon: [accent]{marketplace_id}[/accent]")
    else:
        info("Skipping Amazon — you can add it later in config.yaml")
    spacer()

    # ── Ad Channels ───────────────────────────────────────────────────
    section("Ad Channels")
    console.print("    [muted]Connect at least one ad channel to test.[/muted]")
    spacer()

    # Facebook
    if click.confirm(click.style("    Connect Facebook Ads?", bold=True), default=True):
        fb_app_id = click.prompt(click.style("    App ID", bold=True))
        fb_app_secret = click.prompt(
            click.style("    App secret", bold=True), hide_input=True,
        )
        fb_access_token = click.prompt(
            click.style("    Access token", bold=True), hide_input=True,
        )
        fb_ad_account = click.prompt(
            click.style("    Ad account ID", bold=True),
            default="act_XXXXXXXXX",
        )

        facebook_cfg = FacebookConfig(
            app_id=fb_app_id,
            app_secret=fb_app_secret,
            access_token=fb_access_token,
            ad_account_id=fb_ad_account,
        )
        done(f"Facebook: [accent]{fb_ad_account}[/accent]")
    else:
        info("Skipping Facebook Ads")
    spacer()

    # YouTube / Google Ads
    if click.confirm(click.style("    Connect YouTube / Google Ads?", bold=True), default=False):
        yt_client_id = click.prompt(click.style("    OAuth client ID", bold=True))
        yt_client_secret = click.prompt(
            click.style("    OAuth client secret", bold=True), hide_input=True,
        )
        yt_refresh_token = click.prompt(
            click.style("    Refresh token", bold=True), hide_input=True,
        )
        yt_customer_id = click.prompt(
            click.style("    Google Ads customer ID", bold=True),
        )
        yt_dev_token = click.prompt(
            click.style("    Developer token", bold=True),
            default="",
        )

        youtube_cfg = YouTubeConfig(
            client_id=yt_client_id,
            client_secret=yt_client_secret,
            refresh_token=yt_refresh_token,
            customer_id=yt_customer_id,
            developer_token=yt_dev_token,
        )
        done(f"YouTube: [accent]{yt_customer_id}[/accent]")
    else:
        info("Skipping YouTube / Google Ads")
    spacer()

    # ── Build & Save Config ───────────────────────────────────────────
    config = Config(
        shopify=shopify_cfg,
        amazon=amazon_cfg,
        facebook=facebook_cfg,
        youtube=youtube_cfg,
        statistical=StatisticalConfig(),
    )

    config.to_yaml(output_path)

    section("Configuration Saved")
    done(f"Config written to [accent]{output_path}[/accent]")
    spacer()

    # Summary
    platforms = ["Shopify"]
    if amazon_cfg:
        platforms.append("Amazon")
    channels = []
    if facebook_cfg:
        channels.append("Facebook")
    if youtube_cfg:
        channels.append("YouTube")

    kv("Revenue platforms", ", ".join(platforms))
    kv("Ad channels", ", ".join(channels) if channels else "[warn]None configured[/warn]")
    spacer()

    # Next steps
    section("Next Steps")
    console.print("    [accent]1.[/accent]  Design your first test:")
    channel_hint = channels[0].lower() if channels else "facebook"
    measure_hint = "shopify_and_amazon" if amazon_cfg else "shopify_only"
    console.print(
        f"       [muted]incrementality --config {output_path} design "
        f"--channel {channel_hint} --measure {measure_hint}[/muted]"
    )
    spacer()
    console.print("    [accent]2.[/accent]  Or run a demo with synthetic data:")
    console.print("       [muted]incrementality demo[/muted]")
    spacer()


# ── Open Command ─────────────────────────────────────────────────────────────

@cli.command("open")
@click.argument("test_id")
@click.pass_context
def open_report(ctx: click.Context, test_id: str) -> None:
    """Open a test report in the default browser."""
    import webbrowser

    config = ctx.obj["config"]
    report_path = Path(config.output_dir) / f"{test_id}_report.html"

    if not report_path.exists():
        fail(f"Report not found: [accent]{report_path}[/accent]")
        # Try to find any reports
        output_dir = Path(config.output_dir)
        if output_dir.exists():
            reports = sorted(output_dir.glob("*_report.html"))
            if reports:
                info("Available reports:")
                for r in reports:
                    info(f"  {r.stem.replace('_report', '')}")
        spacer()
        sys.exit(1)

    webbrowser.open(f"file://{report_path.absolute()}")
    done(f"Opened [accent]{report_path}[/accent] in browser")
    spacer()


# ── Boundaries Commands ───────────────────────────────────────────────────────

@cli.group()
def boundaries() -> None:
    """Manage DMA boundary data for spillover analysis."""
    pass


@boundaries.command("compute")
@click.option("--input", "geojson_path", required=True,
              type=click.Path(exists=True),
              help="Path to DMA boundaries GeoJSON/TopoJSON file")
@click.option("--dma-field", default=None,
              help="Column name containing DMA codes (auto-detected if omitted)")
@click.option("--buffer-miles", type=float, default=1.0,
              help="Buffer distance in miles for adjacency detection")
@click.option("--output-dir", type=click.Path(), default="./data",
              help="Directory to save computed adjacency")
def boundaries_compute(
    geojson_path: str,
    dma_field: str | None,
    buffer_miles: float,
    output_dir: str,
) -> None:
    """Compute DMA adjacency from a GeoJSON boundary file."""
    from incrementality.design.dma_boundaries import compute_adjacency_from_geojson

    banner()
    section("Compute Polygon Adjacency")
    kv("Input", f"[accent]{geojson_path}[/accent]")
    kv("Buffer", f"{buffer_miles} miles")
    kv("Output", f"{output_dir}/dma_adjacency.json")
    spacer()

    try:
        with step("Computing adjacency from polygon boundaries"):
            adjacency = compute_adjacency_from_geojson(
                geojson_path,
                dma_code_field=dma_field,
                buffer_miles=buffer_miles,
                cache_dir=output_dir,
            )
        n_edges = sum(len(v) for v in adjacency.values()) // 2
        done(f"Computed adjacency for {len(adjacency)} DMAs with {n_edges} border pairs")
        spacer()

        # Sample output
        sample_dmas = list(adjacency.keys())[:5]
        for dma in sample_dmas:
            neighbors = adjacency[dma]
            kv(f"DMA {dma}", f"{len(neighbors)} neighbors → {neighbors[:5]}")
        if len(adjacency) > 5:
            info(f"... and {len(adjacency) - 5} more DMAs")
        spacer()

    except ImportError:
        fail("GeoPandas is required for polygon-based adjacency.")
        console.print("    [muted]Install with:[/muted] [accent]pip install 'incrementality[geo]'[/accent]")
        spacer()
    except Exception as e:
        fail(str(e))
        spacer()


@boundaries.command("centroid")
@click.option("--max-distance", type=float, default=175.0,
              help="Maximum centroid distance in miles for adjacency")
@click.option("--output-dir", type=click.Path(), default="./data",
              help="Directory to save computed adjacency")
def boundaries_centroid(max_distance: float, output_dir: str) -> None:
    """Compute DMA adjacency from centroid distances (no GeoJSON needed)."""
    import json as json_mod
    from incrementality.design.dma_boundaries import compute_adjacency_from_centroids

    banner()
    section("Compute Centroid Adjacency")
    kv("Max distance", f"{max_distance} miles")
    kv("Output", f"{output_dir}/dma_adjacency.json")
    spacer()

    with step("Computing adjacency from DMA centroids"):
        adjacency = compute_adjacency_from_centroids(max_distance_miles=max_distance)

    n_edges = sum(len(v) for v in adjacency.values()) // 2

    out_path = Path(output_dir) / "dma_adjacency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json_mod.dump(adjacency, f, indent=2)

    done(f"Computed adjacency for {len(adjacency)} DMAs with {n_edges} border pairs")
    done(f"Saved to [accent]{out_path}[/accent]")
    spacer()

    # Stats
    neighbor_counts = [len(v) for v in adjacency.values()]
    section("Statistics")
    kv("Avg neighbors / DMA", f"{np.mean(neighbor_counts):.1f}")
    kv("Max neighbors", str(max(neighbor_counts)))
    kv("Isolated DMAs", str(sum(1 for c in neighbor_counts if c == 0)))
    spacer()


# ── MMM Command Group ─────────────────────────────────────────────────────────

@cli.group()
def mmm() -> None:
    """Marketing Mix Model (Meridian) commands.

    \b
    Fit a Google Meridian Bayesian MMM on 3+ years of DMA-level data,
    optimize channel budget allocation, and reconcile against Northbeam MTA.

    \b
    Examples:
        incrementality mmm fit
        incrementality mmm fit --lookback-weeks 156 --budget 350000
        incrementality mmm optimize --budget 350000
        incrementality mmm results
        incrementality mmm calibrate --channel meta_perf --lift 42000
        incrementality mmm compare --run-id mmm_20260217_081234_abc123
    """
    pass


@mmm.command("fit")
@click.option("--lookback-weeks", type=int, default=156,
              help="Weeks of history to train on (~3 years = 156)")
@click.option("--budget", type=float, default=None,
              help="Monthly budget for optimizer (USD). Defaults to config value.")
@click.option("--shopify-weight", type=float, default=None,
              help="Weight on Shopify revenue in optimizer objective (0–1)")
@click.option("--amazon-weight", type=float, default=None,
              help="Weight on Amazon revenue in optimizer objective (0–1)")
@click.option("--use-cache", is_flag=True, default=False,
              help="Use cached parquet data instead of pulling from APIs")
@click.pass_context
def mmm_fit(
    ctx: click.Context,
    lookback_weeks: int,
    budget: float | None,
    shopify_weight: float | None,
    amazon_weight: float | None,
    use_cache: bool,
) -> None:
    """Fit a Meridian MMM on DMA-level spend + revenue data."""
    from incrementality.mmm_orchestrator import MMMOrchestrator

    banner()
    config = ctx.obj["config"]
    orch = MMMOrchestrator(config)

    section("MMM Configuration")
    kv("Lookback", f"{lookback_weeks} weeks (~{lookback_weeks // 52:.1f} years)")
    kv("Channels", ", ".join(config.mmm.channels))
    kv("Outcomes", "Shopify + Amazon (joint bivariate)")
    kv("MCMC", f"{config.mmm.mcmc_chains} chains × {config.mmm.mcmc_samples} samples")
    kv("Engine", "Google Meridian (NUTS sampler)")
    spacer()

    # Data
    with step("Pulling historical data"):
        data = orch.pull_historical_data(lookback_weeks=lookback_weeks) \
            if not use_cache else orch.load_cached_data()

    pull_errors = getattr(orch, "pull_errors", {})
    for src, df in data.items():
        if not df.empty:
            done(f"{src.title()}: {len(df):,} rows")
        elif src in pull_errors:
            warning(f"{src.title()}: no data — {pull_errors[src]}")
        else:
            info(f"{src.title()}: no connector configured (will use synthetic prior)")
    spacer()

    # Fit
    with step("Fitting Meridian model (NUTS sampling)"):
        result = orch.fit(
            lookback_weeks=lookback_weeks,
            data=data,
        )

    # Save
    out_path = orch.save_result(result)

    # Results summary
    section("Model Results")
    kv("Run ID", f"[accent]{result.run_id}[/accent]")
    kv("DMAs", str(result.n_dmas))
    kv("Training days", str(result.n_days))
    kv("R-hat max", f"{result.rhat_max:.4f}")
    kv("Converged", "[ok]Yes[/ok]" if result.converged else "[bad]No — review diagnostics[/bad]")
    kv("Engine mode", "[muted](stub)[/muted]" if result.is_stub else "[ok]Meridian[/ok]")
    spacer()

    if result.channel_results:
        section("Channel Contributions")
        rows = []
        for cr in sorted(result.channel_results, key=lambda x: x.contribution_mean, reverse=True):
            rows.append({
                "Channel": cr.channel,
                "Outcome": cr.outcome,
                "Contribution": money(cr.contribution_mean),
                "Share %": f"{cr.contribution_pct * 100:.1f}%",
                "ROI": f"{cr.roi_mean:.2f}x",
                "R-hat": f"{cr.rhat:.3f}",
            })
        branded_table(rows, title="Channel × Outcome Contributions")
        spacer()

    section("Budget Optimizer")
    kv("Total budget", money(budget or 350_000))
    if result.optimal_allocation:
        for ch, alloc in sorted(result.optimal_allocation.items(), key=lambda x: x[1], reverse=True):
            kv(f"  {ch}", money(alloc))
    kv("Expected Shopify rev", money(result.expected_shopify_revenue))
    kv("Expected Amazon rev", money(result.expected_amazon_revenue))
    kv("Blended ROAS", f"{result.blended_roas:.2f}x")
    spacer()

    if result.is_stub:
        info("Running in stub mode — install 'meridian' for real Bayesian inference.")
        info("  pip install meridian")
        spacer()

    done(f"Result saved to [accent]{out_path}[/accent]")
    spacer()


@mmm.command("optimize")
@click.option("--budget", type=float, required=True,
              help="Total budget to optimize across channels (USD)")
@click.option("--shopify-weight", type=float, default=None,
              help="Revenue weight for Shopify (0–1); remainder goes to Amazon")
@click.option("--run-id", type=str, default=None,
              help="Specific MMM run ID to use; defaults to latest")
@click.pass_context
def mmm_optimize(
    ctx: click.Context,
    budget: float,
    shopify_weight: float | None,
    run_id: str | None,
) -> None:
    """Run budget optimizer on a previously fitted MMM."""
    from incrementality.mmm_orchestrator import MMMOrchestrator

    banner()
    config = ctx.obj["config"]
    orch = MMMOrchestrator(config)

    sw = shopify_weight if shopify_weight is not None else config.mmm.shopify_revenue_weight
    aw = 1.0 - sw

    section("Budget Optimization")
    kv("Total budget", money(budget))
    kv("Shopify weight", f"{sw:.0%}")
    kv("Amazon weight", f"{aw:.0%}")
    spacer()

    with step("Running optimizer"):
        alloc = orch.optimize(
            total_budget=budget,
            run_id=run_id,
            shopify_weight=sw,
            amazon_weight=aw,
        )

    section("Recommended Allocation")
    for ch, spend in sorted(alloc["allocation"].items(), key=lambda x: x[1], reverse=True):
        lower = alloc["allocation_lower"].get(ch, spend * 0.85)
        upper = alloc["allocation_upper"].get(ch, spend * 1.15)
        kv(f"  {ch}", f"{money(lower)}–{money(upper)}  (mid: {money(spend)})")
    spacer()
    kv("Expected Shopify rev", money(alloc["expected_shopify_revenue"]))
    kv("Expected Amazon rev", money(alloc["expected_amazon_revenue"]))
    kv("Expected total rev", money(alloc["expected_total_revenue"]))
    kv("Blended ROAS", f"{alloc['blended_roas']:.2f}x")
    spacer()


@mmm.command("results")
@click.option("--run-id", type=str, default=None,
              help="Show details for a specific run ID; omit for latest")
@click.pass_context
def mmm_results(ctx: click.Context, run_id: str | None) -> None:
    """Show saved MMM run results."""
    from incrementality.mmm_orchestrator import MMMOrchestrator

    banner()
    config = ctx.obj["config"]
    orch = MMMOrchestrator(config)

    if run_id:
        try:
            result = orch.load_result(run_id)
        except FileNotFoundError:
            fail(f"No MMM result found for run_id '{run_id}'")
            spacer()
            return
    else:
        result = orch.load_latest_result()
        if result is None:
            info("No MMM results found. Run [accent]incrementality mmm fit[/accent] first.")
            spacer()
            return

    section("MMM Run Summary")
    kv("Run ID", f"[accent]{result.run_id}[/accent]")
    kv("Generated", result.generated_at.strftime("%Y-%m-%d %H:%M UTC"))
    kv("DMAs", str(result.n_dmas))
    kv("Training days", str(result.n_days))
    kv("R-hat max", f"{result.rhat_max:.4f}")
    kv("Converged", "[ok]Yes[/ok]" if result.converged else "[bad]No[/bad]")
    kv("Mode", "[muted]stub[/muted]" if result.is_stub else "[ok]Meridian[/ok]")
    spacer()

    if result.channel_results:
        rows = []
        for cr in sorted(result.channel_results, key=lambda x: x.contribution_mean, reverse=True):
            rows.append({
                "Channel": cr.channel,
                "Outcome": cr.outcome,
                "Contribution": money(cr.contribution_mean),
                "Share": f"{cr.contribution_pct * 100:.1f}%",
                "ROI": f"{cr.roi_mean:.2f}x",
                "mROI": f"{cr.marginal_roi:.2f}x",
                "R-hat": f"{cr.rhat:.3f}",
            })
        branded_table(rows, title="Channel × Outcome Contributions")
        spacer()

    if result.nb_mmm_delta:
        section("Northbeam vs MMM Delta")
        for ch, delta in sorted(result.nb_mmm_delta.items(), key=lambda x: abs(x[1]), reverse=True):
            sign = "+" if delta >= 0 else ""
            color = "ok" if abs(delta) < 0.20 else ("bad" if abs(delta) > 0.40 else "warn")
            kv(f"  {ch}", f"[{color}]{sign}{delta * 100:.1f}%[/{color}]")
        spacer()


@mmm.command("calibrate")
@click.option("--channel", required=True,
              help="Channel tested (e.g. meta_perf, google_brand, tiktok)")
@click.option("--test-type", type=click.Choice([
    "geo_holdout", "meta_conversion_lift", "amazon_ab", "ghost_bidding"
]), required=True, help="Type of incrementality test")
@click.option("--start-date", required=True, help="Test start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="Test end date (YYYY-MM-DD)")
@click.option("--lift", "lift_abs", type=float, required=True,
              help="Absolute revenue lift ($) attributed to channel")
@click.option("--lift-lower", type=float, default=None,
              help="90%% CI lower bound for lift")
@click.option("--lift-upper", type=float, default=None,
              help="90%% CI upper bound for lift")
@click.option("--spend", "spend_in_period", type=float, required=True,
              help="Total channel spend during the test ($)")
@click.option("--outcome", type=click.Choice(["shopify", "amazon", "combined"]),
              default="shopify", help="Revenue outcome measured")
@click.option("--geo-scope", default="NATIONAL",
              help="Comma-separated DMA codes tested, or NATIONAL")
@click.option("--notes", default="", help="Test design notes")
@click.pass_context
def mmm_calibrate(
    ctx: click.Context,
    channel: str,
    test_type: str,
    start_date: str,
    end_date: str,
    lift_abs: float,
    lift_lower: float | None,
    lift_upper: float | None,
    spend_in_period: float,
    outcome: str,
    geo_scope: str,
    notes: str,
) -> None:
    """Ingest a geo holdout or conversion lift result as a Meridian calibration event.

    This constrains the posterior for the tested channel so the model's
    channel contribution estimate stays within the experimentally measured
    lift confidence interval — always more accurate than Northbeam priors.
    """
    from incrementality.mmm.calibration import CalibrationEvent, CalibrationStore
    from datetime import date as date_type

    banner()
    config = ctx.obj["config"]
    data_dir = Path(config.data_dir)

    section("New Calibration Event")
    kv("Channel", f"[accent]{channel}[/accent]")
    kv("Test type", test_type)
    kv("Period", f"{start_date} → {end_date}")
    kv("Lift ($)", money(lift_abs))
    kv("Spend ($)", money(spend_in_period))
    kv("Implied ROI", f"{lift_abs / spend_in_period:.2f}x")
    spacer()

    event = CalibrationEvent(
        channel=channel,
        test_type=test_type,
        test_start_date=date_type.fromisoformat(start_date),
        test_end_date=date_type.fromisoformat(end_date),
        lift_abs=lift_abs,
        lift_lower_90=lift_lower if lift_lower is not None else lift_abs * 0.70,
        lift_upper_90=lift_upper if lift_upper is not None else lift_abs * 1.30,
        spend_in_period=spend_in_period,
        outcome=outcome,
        geo_scope=geo_scope,
        notes=notes,
    )

    store = CalibrationStore(
        storage_path=str(data_dir / "calibration_events.json")
    )
    store.add_event(event)

    done(f"Calibration event saved — implied ROI: {event.implied_roi:.2f}x")
    info("Re-run [accent]incrementality mmm fit[/accent] to apply this calibration constraint.")
    spacer()


@mmm.command("compare")
@click.option("--run-id", type=str, default=None,
              help="MMM run ID to compare against incrementality tests; defaults to latest")
@click.pass_context
def mmm_compare(ctx: click.Context, run_id: str | None) -> None:
    """Compare MMM channel ROI estimates against incrementality test results.

    Side-by-side: MMM posterior ROI vs. iROAS from geo holdout tests,
    highlighting where the two methods agree or diverge.
    """
    from incrementality.mmm_orchestrator import MMMOrchestrator
    from incrementality.orchestrator import TestOrchestrator

    banner()
    config = ctx.obj["config"]
    mmm_orch = MMMOrchestrator(config)
    test_orch = TestOrchestrator(config)

    # Load MMM result
    mmm_result = mmm_orch.load_result(run_id) if run_id else mmm_orch.load_latest_result()
    if mmm_result is None:
        info("No MMM results found. Run [accent]incrementality mmm fit[/accent] first.")
        spacer()
        return

    # Load incrementality test results
    tests = test_orch.list_tests()
    analyzed = [t for t in tests if t.get("status") == "analyzed"]

    section("MMM vs Geo Holdout Comparison")
    kv("MMM run", f"[accent]{mmm_result.run_id}[/accent]")
    kv("MMM generated", mmm_result.generated_at.strftime("%Y-%m-%d"))
    kv("Analyzed tests", str(len(analyzed)))
    spacer()

    if mmm_result.channel_results:
        rows = []
        # Build a quick lookup: channel → iroas from test results
        test_iroas: dict[str, float] = {}
        for t in analyzed:
            ch = t.get("channel", "").lower()
            iroas = t.get("iroas", 0.0)
            if ch and iroas:
                test_iroas[ch] = iroas

        for cr in sorted(
            [r for r in mmm_result.channel_results if r.outcome == "shopify"],
            key=lambda x: x.roi_mean, reverse=True
        ):
            test_roi = test_iroas.get(cr.channel, None)
            delta_str = "—"
            if test_roi and cr.roi_mean > 0:
                delta = (cr.roi_mean - test_roi) / test_roi
                sign = "+" if delta >= 0 else ""
                color = "ok" if abs(delta) < 0.25 else "bad"
                delta_str = f"[{color}]{sign}{delta * 100:.0f}%[/{color}]"

            rows.append({
                "Channel": cr.channel,
                "MMM ROI": f"{cr.roi_mean:.2f}x",
                "MMM 80% CI": f"[{cr.roi_p10:.2f}–{cr.roi_p90:.2f}]",
                "Geo Holdout iROAS": f"{test_roi:.2f}x" if test_roi else "no test",
                "Delta": delta_str,
            })
        branded_table(rows, title="Shopify ROI: MMM vs Geo Holdout")
    spacer()

    if not analyzed:
        info("No analyzed geo holdout tests found for comparison.")
        info("Run [accent]incrementality analyze --test-id <id>[/accent] to add test data.")
    spacer()


if __name__ == "__main__":
    cli()
