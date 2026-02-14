"""LIFT — Command-line interface for geo incrementality testing.

Usage:
    incrementality design --channel facebook --name "FB Q1 Test"
    incrementality design --channel youtube --scope campaign --campaigns "123,456"
    incrementality analyze --test-id test_abc123
    incrementality list
    incrementality demo
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import click
import numpy as np
import pandas as pd

from incrementality.config import Config, StatisticalConfig
from incrementality.models import AdChannel, MeasurementScope, TestScope
from incrementality.ui import (
    BRAND,
    VERSION,
    banner,
    branded_table,
    card,
    console,
    done,
    fail,
    info,
    kv,
    iroas_value,
    lift_value,
    money,
    result_card,
    score_bar,
    section,
    sig_badge,
    spacer,
    step,
    warning,
)

logger = logging.getLogger("incrementality")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


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
    if config_path:
        ctx.obj["config"] = Config.from_yaml(config_path)
    else:
        ctx.obj["config"] = Config()

    if ctx.invoked_subcommand is None:
        banner()
        console.print("  [muted]Run[/muted] [accent]incrementality --help[/accent] [muted]for usage.[/muted]")
        spacer()


# ── Design Command ────────────────────────────────────────────────────────────

@cli.command()
@click.option("--channel", type=click.Choice(["facebook", "youtube"]),
              required=True, help="Ad channel to test")
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

    spacer()
    _print_design(test_design)


# ── Analyze Command ───────────────────────────────────────────────────────────

@cli.command()
@click.option("--test-id", required=True, help="Test ID to analyze")
@click.option("--data-dir", type=click.Path(), default=None,
              help="Directory with test period CSV data")
@click.pass_context
def analyze(ctx: click.Context, test_id: str, data_dir: str | None) -> None:
    """Analyze a completed test and generate the incrementality report."""
    from incrementality.orchestrator import TestOrchestrator

    banner()

    config = ctx.obj["config"]
    orchestrator = TestOrchestrator(config)

    with step("Loading test design"):
        test_design = orchestrator.load_design(test_id)
    done(f"Loaded [accent]{test_design.name}[/accent]")

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
            if post_path.exists():
                post_data = pd.read_csv(post_path, parse_dates=["date"])
            if spend_path.exists():
                ad_spend = pd.read_csv(spend_path, parse_dates=["date"])

    with step("Running causal inference analysis"):
        report = orchestrator.analyze_test(
            test_design, pre_data, post_data, ad_spend,
        )

    spacer()
    done(f"Report saved to [accent]{config.output_dir}[/accent]")
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
    error_pct = abs(report.incrementality.relative_lift - true_lift) / true_lift
    kv("Estimation error", f"[muted]{error_pct:.1%}[/muted]")
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
                "impressions": int(spend * rng.normal(100, 20)),
                "clicks": int(spend * rng.normal(2, 0.5)),
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


if __name__ == "__main__":
    cli()
