# Incrementality

A production-grade geo holdout testing platform that measures the true causal impact of advertising spend for e-commerce brands running ads on Facebook and YouTube, with revenue tracked across Shopify and Amazon.

## The Problem

Ad platforms (Facebook, Google) systematically over-report conversions by 20-80% through last-click attribution and modeled conversions. Marketers have no reliable way to know which portion of attributed revenue would have happened organically. After Apple's ATT, this problem has only gotten worse. Platform-reported ROAS cannot be trusted for budget allocation decisions.

## The Solution

Incrementality runs controlled experiments at the geographic level (DMA-based geo holdout tests) to isolate the true causal effect of advertising. It automates the full lifecycle: **design**, **deploy**, **monitor**, **analyze**, and **revert** -- producing metrics like incremental ROAS (iROAS), Incrementality Factor, and spend response curves backed by rigorous causal inference.

## Key Features

- **Automatic test design** -- Analyzes historical data, determines optimal holdout size, balances DMAs via covariate matching, and gates on statistical feasibility before running
- **Ensemble causal inference** -- Three independent estimators (ASCM, BSTS, DiD) combined with adaptive weighting for robust results
- **Comprehensive validation** -- AA tests, placebo-in-time, placebo-in-space, estimator agreement, and pre-period fit checks produce a trust score (0-100)
- **Spend response curves** -- Hill saturation model estimates optimal spend levels and marginal ROAS
- **Multi-platform** -- Shopify + Amazon (revenue sources), Facebook + YouTube (ad platforms)
- **Multiple interfaces** -- CLI, Streamlit dashboard, and Python API
- **Production-hardened** -- Retry with exponential backoff, timeouts, graceful degradation, and comprehensive error handling

## Key Metrics

| Metric | What It Tells You |
|--------|-------------------|
| **iROAS** | Incremental revenue per dollar of ad spend |
| **Incrementality Factor** | How much the ad platform over/under-reports (IF=0.5 means 2x over-reporting) |
| **CPIA** | True cost per incremental acquisition |
| **Spend Response Curve** | Optimal spend level based on diminishing returns |

## Supported Platforms

| Revenue Sources | Ad Platforms |
|-----------------|-------------|
| Shopify (Admin API) | Facebook / Meta (Marketing API v22.0) |
| Amazon (SP-API with RDT) | YouTube / Google Ads |

Measurement scope: Shopify-only, Amazon-only, or cross-platform (Shopify + Amazon).

## Architecture

```
src/incrementality/
├── connectors/           # API integrations (Shopify, Amazon, Facebook, YouTube)
├── design/               # Test design pipeline (optimizer, power analysis, matching, spillover)
├── analysis/             # Causal inference (ASCM, BSTS, DiD, validation, iROAS, spend curves)
├── dashboard/            # Streamlit web UI
├── orchestrator.py       # Test lifecycle coordinator
├── cli.py                # Command-line interface
├── models.py             # Pydantic data models
├── config.py             # Configuration loader
├── dma.py                # 210 US DMA definitions
├── report_pdf.py         # HTML/PDF report generation
├── reporting.py          # JSON/CSV export
└── ui.py                 # Branded CLI output
```

## Installation

**Requires Python >= 3.10**

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install with all optional dependencies
pip install -e ".[all]"

# Or minimal (CLI + analysis only)
pip install -e .

# Pick specific connectors
pip install -e ".[shopify,facebook,dashboard]"
```

## Configuration

Create a `config.yaml` in the project root:

```bash
cp config/example_config.yaml config.yaml
```

Fill in your API credentials for the platforms you use:

```yaml
shopify:
  shop_domain: "my-store.myshopify.com"
  access_token: "shpat_xxxxx"
  api_version: "2025-01"

facebook:
  app_id: "123456"
  app_secret: "secret"
  access_token: "token"
  ad_account_id: "act_123456"

youtube:
  client_id: "xxx.apps.googleusercontent.com"
  client_secret: "secret"
  refresh_token: "refresh"
  customer_id: "1234567890"
  developer_token: "dev_token"

amazon:
  marketplace_id: "ATVPDKIKX0DER"
  seller_id: "seller_id"
  refresh_token: "token"
  client_id: "amzn_client_id"
  client_secret: "amzn_secret"

statistical:
  significance_level: 0.05
  target_power: 0.90
  min_test_duration_weeks: 2
  max_test_duration_weeks: 12
  default_lookback_weeks: 12
  min_dmas_per_cell: 5
  max_holdout_fraction: 0.50
  target_holdout_fraction: 0.25
  balance_tolerance: 0.10

data_dir: ./data
output_dir: ./output
```

## Usage

### CLI

```bash
# Design a test
incrementality design --channel facebook --name "Q1 Test"

# Deploy holdout targeting to ad platform
incrementality execute --test-id test_abc123

# Monitor for new ad sets during test
incrementality execute --test-id test_abc123 --update

# Analyze completed test
incrementality analyze --test-id test_abc123

# Restore original ad targeting
incrementality revert --test-id test_abc123

# List all tests
incrementality list

# Run demo with sample data
incrementality demo
```

### Dashboard

```bash
streamlit run src/incrementality/dashboard/app.py
```

Opens at `http://localhost:8501` with pages for configuration, test design, test management, analysis reports, and spend optimization.

### Python API

```python
from incrementality.config import Config
from incrementality.orchestrator import TestOrchestrator
from incrementality.models import AdChannel, MeasurementScope

config = Config.from_yaml("config.yaml")
orchestrator = TestOrchestrator(config)

# Design
design = orchestrator.design_test(
    test_name="Q1 Test",
    ad_channel=AdChannel.FACEBOOK,
    measurement_scope=MeasurementScope.SHOPIFY_AND_AMAZON,
)

# Execute
orchestrator.execute_test(design)

# Analyze
report = orchestrator.analyze_test(design)
print(f"iROAS: {report.iroas.iroas:.2f}x")
print(f"Trust Score: {report.validation.trust_score}/100")
```

## How It Works

### 1. Design

The platform pulls 12 weeks of historical revenue and ad spend data, estimates within-DMA variance (adjusted for autocorrelation), runs power analysis, and uses Morgan & Rubin re-randomization (10,000 candidates) to assign DMAs into balanced treatment and holdout cells. A feasibility gate blocks underpowered tests before deployment.

### 2. Execute

Holdout exclusions are deployed to ad platforms via their APIs (Facebook Graph API, Google Ads API). The platform monitors for newly created ad sets that may bypass exclusions.

### 3. Analyze

Three causal estimators run independently:

| Estimator | Method | Strength |
|-----------|--------|----------|
| **ASCM** (primary) | Augmented Synthetic Control | Doubly robust, no parallel trends assumption |
| **BSTS** (secondary) | Bayesian Structural Time Series | Full posterior distribution, automatic covariate selection |
| **DiD** (tertiary) | Difference-in-Differences | Simple, transparent baseline |

The ensemble combines estimates with adaptive weighting based on pre-period fit quality. Results are only reported as significant when all three methods agree.

Five validation checks produce a composite trust score:

| Check | What it tests |
|-------|--------------|
| AA Test | No pre-existing differences between cells |
| Placebo-in-Time | No false effects at random pre-period dates |
| Placebo-in-Space | False positive rate near expected 5% |
| Estimator Agreement | Direction, magnitude, and significance consensus |
| Pre-Period Fit | Synthetic control tracks holdout accurately |

### 4. Report

Outputs include:
- **iROAS** -- incremental revenue per dollar of ad spend
- **Incrementality Factor** -- fraction of platform-attributed conversions that are truly incremental
- **CPIA** -- true cost per incremental acquisition
- **Spend response curve** -- Hill saturation model with optimal spend recommendation
- **Trust score** -- 0-100 composite reliability metric

Reports are generated as HTML, PDF, JSON, and CSV.

## Statistical Defaults

| Parameter | Default |
|-----------|---------|
| Significance level | 0.05 |
| Target power | 0.90 |
| Test duration | 2-12 weeks |
| Historical lookback | 12 weeks |
| Min DMAs per cell | 5 |
| Target holdout fraction | 25% |
| Balance tolerance (SMD) | 0.10 |

## Optional Dependencies

| Group | Packages | Purpose |
|-------|----------|---------|
| `shopify` | shopifyapi | Shopify Admin API connector |
| `facebook` | facebook-business | Meta Graph API connector |
| `google` | google-ads | Google Ads API connector |
| `causalimpact` | tfcausalimpact | TensorFlow-based BSTS estimator |
| `geo` | geopandas, shapely | DMA polygon adjacency for spillover detection |
| `pdf` | weasyprint | PDF report generation |
| `dashboard` | streamlit, plotly | Web UI |
| `all` | All of the above | Full installation |
| `dev` | pytest, mypy, ruff | Development and testing |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

## Documentation

See [docs/whitepaper.md](docs/whitepaper.md) for a comprehensive technical whitepaper covering methodology, statistical framework, and platform architecture.

## License

Proprietary. All rights reserved.
