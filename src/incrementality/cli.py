"""Command-line interface for the incrementality testing platform.

Usage:
    incrementality design --channel facebook --name "FB Q1 Test"
    incrementality design --channel youtube --scope campaign --campaigns "123,456"
    incrementality analyze --test-id test_abc123
    incrementality list
    incrementality report --test-id test_abc123
    incrementality demo  # Run with synthetic data
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
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from incrementality.config import Config, StatisticalConfig
from incrementality.models import AdChannel, MeasurementScope, TestScope

console = Console()
logger = logging.getLogger("incrementality")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--config", "config_path", type=click.Path(), default=None,
              help="Path to config YAML file")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, verbose: bool) -> None:
    """Incrementality Testing Platform

    Design, run, and analyze geo holdout tests to measure true incremental
    ROAS across Shopify and Amazon for your ad channels.
    """
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    if config_path:
        ctx.obj["config"] = Config.from_yaml(config_path)
    else:
        ctx.obj["config"] = Config()


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
              help="Target minimum detectable effect (e.g., 0.10 for 10%)")
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
    """Design an optimal incrementality test.

    Analyzes historical data and produces a statistically sound test design
    with matched treatment/holdout DMA cells.
    """
    from incrementality.orchestrator import TestOrchestrator

    config = ctx.obj["config"]
    orchestrator = TestOrchestrator(config)

    ad_channel = AdChannel(channel)
    test_scope = TestScope(scope)
    measurement_scope = MeasurementScope(measure)
    campaign_ids = campaigns.split(",") if campaigns else None

    # Load data
    data = None
    if data_dir:
        data = orchestrator.load_data_from_csv(data_dir)

    console.print(Panel(
        f"Channel: [bold]{channel}[/bold]\n"
        f"Scope: {scope}\n"
        f"Measuring: {measure.replace('_', ' ')}\n"
        f"Lookback: {lookback_weeks} weeks",
        title="Designing Incrementality Test",
        border_style="blue",
    ))

    design = orchestrator.design_test(
        ad_channel=ad_channel,
        test_scope=test_scope,
        measurement_scope=measurement_scope,
        campaign_ids=campaign_ids,
        test_name=name,
        target_mde=target_mde,
        data=data,
        lookback_weeks=lookback_weeks,
    )

    _print_design(design)


@cli.command()
@click.option("--test-id", required=True, help="Test ID to analyze")
@click.option("--data-dir", type=click.Path(), default=None,
              help="Directory with test period CSV data")
@click.pass_context
def analyze(ctx: click.Context, test_id: str, data_dir: str | None) -> None:
    """Analyze a completed test and generate the incrementality report."""
    from incrementality.orchestrator import TestOrchestrator

    config = ctx.obj["config"]
    orchestrator = TestOrchestrator(config)

    design = orchestrator.load_design(test_id)
    console.print(f"Loaded test design: [bold]{design.name}[/bold]")

    pre_data = None
    post_data = None
    ad_spend = None

    if data_dir:
        csv_dir = Path(data_dir)
        pre_path = csv_dir / "pre_period.csv"
        post_path = csv_dir / "post_period.csv"
        spend_path = csv_dir / "ad_spend.csv"
        if pre_path.exists():
            pre_data = pd.read_csv(pre_path, parse_dates=["date"])
        if post_path.exists():
            post_data = pd.read_csv(post_path, parse_dates=["date"])
        if spend_path.exists():
            ad_spend = pd.read_csv(spend_path, parse_dates=["date"])

    report = orchestrator.analyze_test(
        design, pre_data, post_data, ad_spend,
    )
    console.print(f"\nReport saved to: [bold]{config.output_dir}[/bold]")


@cli.command("list")
@click.pass_context
def list_tests(ctx: click.Context) -> None:
    """List all tests."""
    from incrementality.orchestrator import TestOrchestrator

    config = ctx.obj["config"]
    orchestrator = TestOrchestrator(config)
    tests = orchestrator.list_tests()

    if not tests:
        console.print("No tests found.")
        return

    table = Table(title="Tests")
    table.add_column("Test ID", style="bold")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Channel")
    table.add_column("Duration")
    for t in tests:
        table.add_row(
            t["test_id"],
            t["name"],
            t["status"],
            t["channel"],
            f"{t['duration_weeks']} weeks",
        )
    console.print(table)


@cli.command()
@click.option("--channel", type=click.Choice(["facebook", "youtube"]),
              default="facebook", help="Channel to simulate")
@click.option("--measure", type=click.Choice([
    "shopify_only", "amazon_only", "shopify_and_amazon"
]), default="shopify_and_amazon", help="Revenue to measure")
@click.option("--true-lift", type=float, default=0.12,
              help="True lift to simulate (e.g., 0.12 for 12%)")
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

    Generates realistic synthetic data, designs a test, simulates the
    test period, and produces the incrementality report.
    """
    from incrementality.orchestrator import TestOrchestrator

    console.print(Panel(
        f"True Shopify lift: [bold]{true_lift:.0%}[/bold]\n"
        f"Amazon halo: [bold]{amazon_halo:.0%}[/bold]\n"
        f"Channel: [bold]{channel}[/bold]\n"
        f"DMAs: {num_dmas} | History: {weeks} weeks",
        title="Demo: Synthetic Incrementality Test",
        border_style="green",
    ))

    # Generate synthetic data
    console.print("\n[dim]Generating synthetic historical data...[/dim]")
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

    # Design
    console.print("[dim]Running automatic test design...[/dim]\n")
    test_design = orchestrator.design_test(
        ad_channel=ad_channel,
        measurement_scope=measurement_scope,
        test_name=f"Demo: {channel.title()} Incrementality",
        data=data,
        lookback_weeks=weeks,
        run_simulation=False,
    )
    _print_design(test_design)

    # Simulate test period
    console.print("\n[dim]Simulating test period with known lift...[/dim]")
    pre_data, post_data, spend_data = _simulate_test_period(
        test_design, data, true_lift, amazon_halo, channel,
    )

    # Analyze
    console.print("[dim]Running analysis...[/dim]\n")
    report = orchestrator.analyze_test(
        test_design, pre_data, post_data, spend_data,
    )


def _print_design(design) -> None:
    """Print test design summary."""
    table = Table(title="Test Design", show_header=False, border_style="dim")
    table.add_column("", style="bold")
    table.add_column("")
    table.add_row("Test ID", design.test_id)
    table.add_row("Name", design.name)
    table.add_row("Channel", design.ad_channel.value.title())
    table.add_row("Scope", design.test_scope.value.title())
    table.add_row("Measurement", design.measurement_scope.value.replace("_", " ").title())
    table.add_row("Treatment DMAs", str(design.num_treatment_dmas))
    table.add_row("Holdout DMAs", str(design.num_holdout_dmas))
    table.add_row("Duration", f"{design.duration_weeks} weeks")
    if design.recommended_start_date:
        table.add_row("Start Date", str(design.recommended_start_date))
        table.add_row("End Date", str(design.recommended_end_date))
    table.add_row("Balance Score", f"{design.balance_score:.3f}")

    if design.power_analysis:
        pa = design.power_analysis
        table.add_row("", "")
        table.add_row("Min Detectable Effect", f"{pa.minimum_detectable_effect:.1%}")
        table.add_row("Statistical Power", f"{pa.statistical_power:.1%}")
        if pa.simulated_power > 0:
            table.add_row("Simulated Power", f"{pa.simulated_power:.1%}")
            table.add_row("Simulated FPR", f"{pa.simulated_false_positive_rate:.1%}")
            table.add_row("Num Simulations", str(pa.num_simulations))
        table.add_row("Significance Level", f"{pa.significance_level:.0%}")
        table.add_row("Cohen's d", f"{pa.effect_size_cohen_d:.3f}")
        if pa.power_score > 0:
            score_style = "green" if pa.power_score >= 85 else ("yellow" if pa.power_score >= 70 else "red")
            table.add_row("Power Score", f"[{score_style}]{pa.power_score:.0f}/100[/{score_style}]")

    console.print(table)

    # Print DMA assignments
    console.print(f"\n[bold]Treatment DMAs ({design.num_treatment_dmas}):[/bold]")
    console.print(", ".join(design.treatment_cell.dma_codes[:20]))
    if len(design.treatment_cell.dma_codes) > 20:
        console.print(f"  ... and {len(design.treatment_cell.dma_codes) - 20} more")

    console.print(f"\n[bold]Holdout DMAs ({design.num_holdout_dmas}):[/bold]")
    console.print(", ".join(design.holdout_cell.dma_codes))


def _generate_synthetic_data(
    num_dmas: int,
    weeks: int,
    channel: str,
) -> dict[str, pd.DataFrame]:
    """Generate realistic synthetic DMA-level daily data."""
    from incrementality.dma import get_all_dmas

    rng = np.random.default_rng(seed=42)
    all_dmas = get_all_dmas()[:num_dmas]
    dma_codes = [d.dma_code for d in all_dmas]
    populations = {d.dma_code: d.population for d in all_dmas}

    end_date = date.today()
    start_date = end_date - timedelta(weeks=weeks)
    dates = pd.date_range(start_date, end_date, freq="D")

    records_shopify = []
    records_amazon = []
    records_ad = []

    for dma in all_dmas:
        pop_factor = dma.population / 1_000_000  # Scale by pop in millions
        base_shopify = 500 * pop_factor + rng.normal(0, 50 * pop_factor)
        base_amazon = 200 * pop_factor + rng.normal(0, 30 * pop_factor)
        base_spend = 100 * pop_factor + rng.normal(0, 10 * pop_factor)

        for d in dates:
            # Day of week effect
            dow_effect = 1.0 + 0.15 * (d.dayofweek in [5, 6])  # Weekend boost
            # Seasonal trend
            seasonal = 1.0 + 0.05 * np.sin(2 * np.pi * d.dayofyear / 365)
            # Noise
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

    shopify_df = pd.DataFrame(records_shopify)
    amazon_df = pd.DataFrame(records_amazon)
    ad_df = pd.DataFrame(records_ad)

    return {
        "shopify": shopify_df,
        "amazon": amazon_df,
        "facebook" if channel == "facebook" else "youtube": ad_df,
    }


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
    all_dmas = treatment_dmas | holdout_dmas

    # Use last few weeks of historical as pre-period
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

    # Split available data: ~60% pre-period, ~40% test period
    # Ensure at least 14 days for each period
    test_days = min(total_days * 2 // 5, total_days - 14)
    test_days = max(test_days, 14)
    pre_days = total_days - test_days

    test_end = max_date
    test_start = test_end - timedelta(days=test_days)
    pre_start = min_date

    # Pre-period data
    pre_mask = (shopify["date"] >= pd.Timestamp(pre_start)) & (shopify["date"] < pd.Timestamp(test_start))
    post_mask = (shopify["date"] >= pd.Timestamp(test_start)) & (shopify["date"] <= pd.Timestamp(test_end))

    pre_shopify = shopify[pre_mask].copy()
    post_shopify = shopify[post_mask].copy()

    # Apply treatment effect to treatment DMAs in post period
    treatment_mask = post_shopify["dma_code"].isin(treatment_dmas)
    post_shopify.loc[treatment_mask, "revenue"] *= (1 + true_lift)
    post_shopify.loc[treatment_mask, "orders"] = (
        post_shopify.loc[treatment_mask, "orders"] * (1 + true_lift * 0.8)
    ).astype(int)

    # Build combined pre/post with both platforms
    pre_data = pre_shopify.rename(columns={"revenue": "shopify_revenue", "orders": "shopify_orders"})
    post_data = post_shopify.rename(columns={"revenue": "shopify_revenue", "orders": "shopify_orders"})

    if not amazon.empty:
        pre_amazon = amazon[(amazon["date"] >= pd.Timestamp(pre_start)) & (amazon["date"] < pd.Timestamp(test_start))].copy()
        post_amazon = amazon[(amazon["date"] >= pd.Timestamp(test_start)) & (amazon["date"] <= pd.Timestamp(test_end))].copy()

        # Amazon halo effect
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

    # Combined revenue
    for df in [pre_data, post_data]:
        rev_cols = [c for c in df.columns if c.endswith("_revenue")]
        df["revenue"] = df[rev_cols].sum(axis=1)

    # Ad spend data (holdout gets zero)
    spend_data = pd.DataFrame()
    if not ad_data.empty:
        spend_mask = (ad_data["date"] >= pd.Timestamp(test_start)) & (ad_data["date"] <= pd.Timestamp(test_end))
        spend_data = ad_data[spend_mask].copy()
        # Zero out spend in holdout DMAs
        holdout_spend_mask = spend_data["dma_code"].isin(holdout_dmas)
        spend_data.loc[holdout_spend_mask, "spend"] = 0
    else:
        # Generate synthetic spend
        records = []
        dates = pd.date_range(test_start, test_end, freq="D")
        for dma in treatment_dmas:
            for d in dates:
                records.append({
                    "date": d,
                    "dma_code": dma,
                    "spend": rng.normal(100, 20),
                })
        for dma in holdout_dmas:
            for d in dates:
                records.append({
                    "date": d,
                    "dma_code": dma,
                    "spend": 0,
                })
        spend_data = pd.DataFrame(records)

    return pre_data, post_data, spend_data


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
    """Compute DMA adjacency from a GeoJSON boundary file.

    Downloads a DMA boundaries GeoJSON and computes which DMAs share
    borders using polygon intersection. Results are cached for use
    by the spillover analysis.

    Example:
        incrementality boundaries compute --input dma_boundaries.geojson
    """
    from incrementality.design.dma_boundaries import compute_adjacency_from_geojson

    console.print(Panel(
        f"Input: [bold]{geojson_path}[/bold]\n"
        f"Buffer: {buffer_miles} miles\n"
        f"Output: {output_dir}/dma_adjacency.json",
        title="Computing DMA Adjacency",
        border_style="blue",
    ))

    try:
        adjacency = compute_adjacency_from_geojson(
            geojson_path,
            dma_code_field=dma_field,
            buffer_miles=buffer_miles,
            cache_dir=output_dir,
        )
        n_edges = sum(len(v) for v in adjacency.values()) // 2
        console.print(
            f"\n[green]Computed adjacency for {len(adjacency)} DMAs "
            f"with {n_edges} border pairs[/green]"
        )
        # Show sample
        sample_dmas = list(adjacency.keys())[:5]
        for dma in sample_dmas:
            neighbors = adjacency[dma]
            console.print(f"  {dma}: {len(neighbors)} neighbors -> {neighbors[:5]}")
        if len(adjacency) > 5:
            console.print(f"  ... and {len(adjacency) - 5} more DMAs")
    except ImportError:
        console.print(
            "[red]GeoPandas is required for polygon-based adjacency.[/red]\n"
            "Install with: pip install 'incrementality[geo]'"
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@boundaries.command("centroid")
@click.option("--max-distance", type=float, default=175.0,
              help="Maximum centroid distance in miles for adjacency")
@click.option("--output-dir", type=click.Path(), default="./data",
              help="Directory to save computed adjacency")
def boundaries_centroid(max_distance: float, output_dir: str) -> None:
    """Compute DMA adjacency from centroid distances (no GeoJSON needed).

    Uses approximate centroid locations for all 210 DMAs and considers
    DMAs within the specified distance as adjacent. This is a reasonable
    fallback when polygon boundary data is not available.
    """
    import json as json_mod
    from incrementality.design.dma_boundaries import compute_adjacency_from_centroids

    console.print(Panel(
        f"Max distance: {max_distance} miles\n"
        f"Output: {output_dir}/dma_adjacency.json",
        title="Computing Centroid-Based DMA Adjacency",
        border_style="blue",
    ))

    adjacency = compute_adjacency_from_centroids(max_distance_miles=max_distance)
    n_edges = sum(len(v) for v in adjacency.values()) // 2

    # Save
    out_path = Path(output_dir) / "dma_adjacency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json_mod.dump(adjacency, f, indent=2)

    console.print(
        f"\n[green]Computed adjacency for {len(adjacency)} DMAs "
        f"with {n_edges} border pairs[/green]"
    )
    console.print(f"Saved to: {out_path}")

    # Stats
    neighbor_counts = [len(v) for v in adjacency.values()]
    console.print(
        f"\nAvg neighbors per DMA: {np.mean(neighbor_counts):.1f}\n"
        f"Max neighbors: {max(neighbor_counts)}\n"
        f"DMAs with no neighbors: {sum(1 for c in neighbor_counts if c == 0)}"
    )


if __name__ == "__main__":
    cli()
