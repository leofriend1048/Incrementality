# Incrementality: Measuring True Advertising Impact with Geo Holdout Testing

**A Production Platform for Causal Inference in E-Commerce Advertising**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Problem: Why Attribution Is Broken](#2-the-problem-why-attribution-is-broken)
3. [The Solution: Geo Holdout Testing](#3-the-solution-geo-holdout-testing)
4. [Platform Overview](#4-platform-overview)
5. [Test Design Pipeline](#5-test-design-pipeline)
6. [Causal Inference Methodology](#6-causal-inference-methodology)
7. [Validation Framework](#7-validation-framework)
8. [Key Metrics: iROAS, Incrementality Factor, and CPIA](#8-key-metrics-iroas-incrementality-factor-and-cpia)
9. [Spend Response Curves](#9-spend-response-curves)
10. [Supported Platforms](#10-supported-platforms)
11. [Feasibility Gating](#11-feasibility-gating)
12. [Technical Architecture](#12-technical-architecture)
13. [Glossary](#13-glossary)

---

## 1. Executive Summary

### For Growth Marketers

You're spending money on Facebook and YouTube ads. Your ad platforms say they're working. But how much of that revenue would have happened anyway — without the ads?

**Incrementality** answers that question. It runs a controlled experiment across geographic markets: some markets see your ads (treatment), others don't (holdout). By comparing revenue between the two groups, you measure the *true* incremental impact of your advertising — not what the ad platform claims, but what actually happened.

The platform handles everything: designing the test, deploying the holdout, analyzing results, and telling you exactly how much each dollar of ad spend is actually worth.

**Key outputs:**
- **Incremental ROAS (iROAS):** How much *true* revenue each ad dollar generates
- **Incrementality Factor:** How much the ad platform is over- or under-counting
- **Optimal Spend:** Where your marginal dollar stops being profitable
- **Statistical Confidence:** Whether you can trust the results enough to act on them

### For Data Scientists

Incrementality implements a production-grade geo holdout testing framework built on three causal inference estimators — Augmented Synthetic Control Method (ASCM), Bayesian Structural Time Series (BSTS), and Difference-in-Differences (DiD) — combined in a weighted ensemble. The platform automates the full experimental lifecycle: variance estimation, power analysis (analytical + GeoLift-style simulation), re-randomization matching (Morgan & Rubin, 2012), spillover risk assessment, and a comprehensive validation suite (AA tests, placebo-in-time, placebo-in-space, conformal inference).

The system targets the same statistical rigor as Haus and GeoLift while adding end-to-end automation for Shopify/Amazon brands running Facebook and YouTube ads. It enforces pre-test feasibility gating (power score ≥ 70/100, MDE ≤ 30%) and post-test trust scoring (≥ 70/100) before results are considered actionable.

---

## 2. The Problem: Why Attribution Is Broken

### The Marketer's View

Ad platforms like Facebook and Google report conversions using attribution models — last-click, view-through, or their own machine learning models. These models have a fundamental problem: they take credit for conversions that would have happened anyway.

Consider a customer who:
1. Sees your Facebook ad on Monday
2. Searches your brand on Google on Tuesday
3. Buys from your website on Wednesday

Facebook claims credit. Google claims credit. Both report a conversion. But maybe this customer was already going to buy — they'd seen your product on a friend's Instagram, or they were a repeat customer. The ad didn't *cause* the purchase; it just *preceded* it.

This isn't a minor issue. Studies consistently show that ad platforms over-report incrementality by 20–80%. For a brand spending $1M/month on ads, that could mean $200K–$800K/month in wasted spend that's attributed as "working."

### The Data Scientist's View

Platform-reported attribution suffers from several well-documented biases:

- **Selection bias:** Users who see ads are systematically different from those who don't (ad platforms target high-intent users)
- **Activity bias:** Ad exposure correlates with purchasing intent, not because ads cause purchases, but because active users both see more ads and buy more
- **Cross-platform double-counting:** Each platform independently claims credit for the same conversion
- **Post-iOS 14 signal loss:** Apple's ATT framework has degraded deterministic attribution, making modeled conversions less reliable

The only rigorous way to measure causal advertising impact is through **randomized controlled experiments**. Geo holdout tests are the gold standard for advertising incrementality because they randomize at the market level, avoiding the individual-level consent issues that plague user-level experiments.

---

## 3. The Solution: Geo Holdout Testing

### How It Works (The Simple Version)

1. **Divide markets into two groups:** Treatment markets keep seeing your ads as usual. Holdout markets have your ads turned off.
2. **Wait 2–12 weeks:** Both groups generate revenue. You measure the difference.
3. **The difference is your true incremental impact:** Revenue that *only* happened because of the ads.

### How It Works (The Technical Version)

A geo holdout test is a cluster-randomized controlled experiment where:

- **Unit of randomization:** Designated Market Areas (DMAs) — 210 non-overlapping geographic regions covering the entire US
- **Treatment assignment:** DMAs are assigned to treatment (ads on) or holdout (ads off) using covariate-balanced re-randomization
- **Outcome variable:** Daily revenue per DMA (Shopify, Amazon, or both)
- **Causal identification:** Difference-in-differences framework with pre-period matching, augmented by synthetic control methods
- **Inference:** Conformal permutation tests for finite-sample validity (no asymptotic assumptions)

The key assumption is the **Stable Unit Treatment Value Assumption (SUTVA):** a DMA's outcome depends only on its own treatment assignment, not on neighboring DMAs' assignments. We assess and mitigate violations of this assumption through spillover risk analysis and geographic buffering.

---

## 4. Platform Overview

### End-to-End Workflow

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐
│  DESIGN │───>│ EXECUTE │───>│   RUN   │───>│ REVERT  │───>│ ANALYZE  │
│         │    │         │    │         │    │         │    │          │
│ Auto-   │    │ Deploy  │    │ 2-12    │    │ Restore │    │ Ensemble │
│ design  │    │ holdout │    │ weeks   │    │ original│    │ causal   │
│ optimal │    │ to ad   │    │         │    │ ad      │    │ analysis │
│ test    │    │ platform│    │         │    │ targeting│   │ + report │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └──────────┘
```

### What the Platform Automates

| Phase | What It Does | Why It Matters |
|-------|-------------|----------------|
| **Design** | Analyzes 12 weeks of historical data, estimates variance, determines optimal holdout size, matches DMAs into balanced groups, runs power analysis, checks feasibility | Without proper design, tests fail silently — you wait 8 weeks and learn nothing |
| **Execute** | Deploys DMA exclusions to Facebook/YouTube ad sets programmatically | Manual exclusion across hundreds of ad sets is error-prone |
| **Monitor** | Detects and excludes new ad sets created during the test | New campaigns launched mid-test would contaminate the holdout |
| **Revert** | Restores original targeting after the test completes | Clean rollback with saved state ensures no permanent changes |
| **Analyze** | Runs three causal estimators, validates results, computes iROAS, generates reports | Raw data without proper causal inference leads to wrong conclusions |

---

## 5. Test Design Pipeline

### Step 1: Historical Variance Estimation

**For Marketers:** The platform looks at your last 12 weeks of revenue data across all 210 US markets to understand how much revenue naturally fluctuates. Markets with volatile revenue need larger tests to detect real effects.

**For Data Scientists:** We decompose total variance into between-DMA variance (persistent cross-sectional differences) and within-DMA variance (temporal fluctuations) using a random-effects model. The within-DMA variance, adjusted for autocorrelation, drives the MDE calculation:

$$\sigma^2_{DiD} = \frac{\sigma^2_w}{T_{eff}}$$

where $T_{eff} = T \cdot \frac{1 - \rho}{1 + \rho}$ and $\rho$ is the lag-1 autocorrelation of weekly DMA revenue.

### Step 2: Optimal Holdout Sizing

**For Marketers:** The platform determines how many markets to hold out. More holdout markets = more statistical power (ability to detect smaller effects), but also more lost revenue during the test. The platform finds the sweet spot — typically around 25% of markets.

**For Data Scientists:** We optimize the holdout size by maximizing a scoring function that balances MDE minimization against opportunity cost:

$$\text{score} = 0.8 \cdot \text{mde\_score} - 0.1 \cdot \text{holdout\_penalty} - 0.1 \cdot \text{target\_distance}$$

where:
- `mde_score = max(0, 1 - MDE / target_MDE)` — how close MDE is to the target
- `holdout_penalty = n_holdout / n_total` — fraction of DMAs held out
- `target_distance = |n_holdout - target| / n_total` — deviation from the target holdout fraction (default 25%)

MDE is evaluated at the maximum test duration (default 12 weeks) with 80% power and 5% significance level.

### Step 3: DMA Matching

**For Marketers:** Not all markets are equal — New York generates far more revenue than Fargo. The platform ensures that the treatment and holdout groups are balanced on revenue, orders, ad spend, growth trends, and volatility. This prevents the results from being skewed by one group having systematically larger or faster-growing markets.

**For Data Scientists:** We implement Morgan & Rubin's (2012) re-randomization design:

1. Generate 10,000 random treatment-holdout assignments
2. For each assignment, compute the Standardized Mean Difference (SMD) across covariates:
   - Total revenue, total orders, revenue per capita
   - Total ad spend, revenue trend (week-over-week), revenue volatility (CV)
3. Accept only assignments where $\max(\text{SMD}) \leq 0.25$
4. Select the assignment with the best composite balance score

**Balance Validation:** Post-matching, we verify that all covariate SMDs are below 0.10 (well-balanced). If re-randomization fails to find an acceptable assignment, we fall back to greedy Mahalanobis distance matching.

### Step 4: Spillover Risk Assessment

**For Marketers:** If a holdout market borders a treatment market, customers near the border might drive to the treatment market to buy, contaminating results. The platform identifies these border pairs and can exclude them from the analysis.

**For Data Scientists:** We compute an adjacency graph of treatment-holdout DMA pairs using polygon intersection (primary) or centroid distance with a 175-mile threshold (fallback). The spillover risk score is:

$$\text{risk} = \frac{\text{number of holdout DMAs bordering treatment DMAs}}{\text{total holdout DMAs}}$$

**Mitigation strategies:**
- **Geographic buffer:** Exclude border DMAs from the statistical analysis (they remain in the experimental design for execution purposes)
- **Spillover adjustment:** Scale the estimated effect by $\frac{1}{1 - \text{contamination} \times \text{decay}}$, assuming a 20% decay to adjacent DMAs

### Step 5: Power Analysis

**For Marketers:** Power analysis tells you whether the test will actually work — whether you'll be able to detect a real effect if one exists. A test with low power is a waste of time and money. The platform requires a power score of at least 70/100 before greenlighting a test.

**For Data Scientists:** We provide two layers of power analysis:

**Analytical (fast):** Standard two-sample formula with autocorrelation adjustment:

$$\text{MDE} = \frac{(z_{\alpha/2} + z_\beta) \cdot \text{SE}}{\bar{y}}$$

where $\text{SE} = \sqrt{\sigma^2_{DiD}/n_T + \sigma^2_{DiD}/n_H}$

**Simulation-based (production - GeoLift-style):** Monte Carlo permutation power analysis:
1. Use historical data as ground truth
2. For each of 200 simulations:
   - Pick a random date to split into fake pre/post periods
   - Inject a known treatment effect (e.g., +15%) into post-period treatment DMAs
   - Run the DiD estimator
   - Record whether the effect was detected ($p < \alpha$)
3. Empirical power = fraction of simulations detecting the true effect
4. Empirical FPR = fraction of null simulations (effect = 0) falsely rejecting $H_0$

**Power Score (0–100):**

| Component | Max Points | Criteria |
|-----------|-----------|----------|
| Simulated power | 40 | ≥80% power |
| FPR calibration | 20 | ~5% at $\alpha = 0.05$ |
| MDE | 25 | ≤15% is ideal |
| Sample size | 15 | ≥20 holdout DMAs |

---

## 6. Causal Inference Methodology

### The Three Estimators

The platform runs three independent causal estimators and combines them in a weighted ensemble. This approach provides robustness: if one estimator's assumptions are violated, the others can compensate.

### 6.1 Augmented Synthetic Control Method (ASCM) — Primary

**For Marketers:** ASCM creates a "synthetic" version of each holdout market by combining treatment markets in just the right proportions. If the synthetic version closely tracks the real holdout before the test, the gap during the test is your causal effect.

**For Data Scientists:**

ASCM (Ben-Michael, Feller, Rothstein, JASA 2021) combines the synthetic control method with an outcome model for double robustness:

1. **SCM step:** Find weights $w$ that minimize pre-period imbalance:
   $$\min_w \sum_{t \in \text{pre}} \left( Y_{0t} - \sum_j w_j Y_{jt} \right)^2 \quad \text{s.t.} \quad w_j \geq 0, \; \sum_j w_j = 1$$

2. **Augmentation step:** Fit ridge regression on the SCM residuals to de-bias:
   $$\hat{\tau}_t = Y_{0t} - \hat{Y}_{0t}^{SCM} - \hat{\mu}(X_{0t}) + \sum_j w_j \hat{\mu}(X_{jt})$$

3. **Inference:** Conformal permutation test (Chernozhukov, Wuthrich, Zhu 2022):
   - Pool pre-period residuals and post-period gaps
   - Permute 2,000 times to build a null distribution
   - $p$-value = $\frac{\text{count}(|\text{perm stat}| \geq |\text{observed stat}|) + 1}{n_{\text{perm}} + 1}$
   - Confidence intervals via test inversion

**Why ASCM is the primary estimator:**
- Does not assume parallel trends (unlike DiD)
- Double robustness: consistent if either SCM weights or outcome model is correct
- Conformal inference provides finite-sample valid p-values (no large-sample assumptions)
- Validated by in-space placebo tests

### 6.2 Bayesian Structural Time Series (BSTS) — Secondary

**For Marketers:** BSTS builds a forecasting model of what holdout revenue *would have been* if the test hadn't happened, using treatment markets as predictors. The gap between forecast and actual is your causal effect. It also provides a probability of the effect being positive.

**For Data Scientists:**

Our BSTS implementation follows Brodersen et al. (2015) — the methodology behind Google's CausalImpact:

1. **State-space model:** Local linear trend + regression on covariates (treatment DMAs)
2. **Spike-and-slab priors:** Automatic covariate selection (top 10 by correlation)
3. **Posterior inference:** 5,000 MCMC draws from the posterior predictive distribution
4. **Credible intervals:** Full Bayesian 95% credible intervals (not confidence intervals — they have a direct probabilistic interpretation)

**Implementation:** Uses `tfcausalimpact` (TensorFlow Probability) when available, with a `statsmodels` Unobserved Components fallback.

### 6.3 Difference-in-Differences (DiD) — Tertiary

**For Marketers:** DiD compares the before/after change in holdout markets to the before/after change in treatment markets. If treatment markets grew more (or shrank less) than holdout markets after the test started, the difference is attributed to the ads.

**For Data Scientists:**

The standard DiD estimator:

$$\hat{\tau} = (\bar{Y}_{T,\text{post}} - \bar{Y}_{T,\text{pre}}) - (\bar{Y}_{H,\text{post}} - \bar{Y}_{H,\text{pre}})$$

**Inference:** DMA-level clustered bootstrap (2,000 iterations) to account for within-DMA serial correlation.

**Limitations:** DiD assumes parallel trends — that treatment and holdout markets would have evolved identically absent the intervention. This assumption frequently fails for geo data due to market-specific shocks, seasonality differences, and heterogeneous trends. For this reason, DiD receives the lowest ensemble weight.

### 6.4 Ensemble

**For Marketers:** Rather than betting everything on one method, the platform combines all three estimates. The final answer is a weighted average that emphasizes the most reliable estimator for your specific data. Results are only considered significant if all three methods agree.

**For Data Scientists:**

The ensemble combines estimators with adaptive weights based on pre-period fit quality:

| Estimator | Base Weight | Typical Range |
|-----------|------------|---------------|
| ASCM | 2.0 | 40–60% |
| BSTS | 1.5 | 25–40% |
| DiD | 0.5 | 10–20% |

Weights are further adjusted by each estimator's pre-period L2 imbalance and $R^2$. The final ensemble uses:

- **Point estimate:** Weighted average
- **Confidence interval:** Conservative — widest bounds across all estimators
- **P-value:** Conservative — maximum across estimators
- **Significance:** Requires unanimity — all estimators must agree on significance

---

## 7. Validation Framework

### Why Validation Matters

**For Marketers:** A test result is only useful if you can trust it. The platform runs five separate validation checks to make sure the methodology is working correctly on your specific data. Think of it like a pre-flight checklist — you wouldn't fly a plane without one, and you shouldn't bet your ad budget on unvalidated results.

**For Data Scientists:** Every result is subjected to the following validation battery. A composite trust score (0–100) is computed, with a minimum threshold of 70 for actionable results.

### Validation Layers

#### 1. AA Test (Pre-Period Placebo)

Split the pre-period in half and run ASCM as if the midpoint were the intervention. Since no intervention occurred, finding a significant effect indicates methodology failure.

- **Pass:** $p > 0.05$ (no spurious effect detected)
- **Weight in trust score:** 25 points

#### 2. Placebo-in-Time

Test 5 random fake intervention dates within the pre-period. Each should yield a non-significant result. This validates that the methodology doesn't generate false positives at arbitrary time points.

#### 3. Placebo-in-Space

Treat each holdout DMA as "treated" and construct a synthetic control from remaining holdout DMAs. Under the null hypothesis, no effect should be detected. The empirical false positive rate should be approximately $\alpha = 0.05$.

- **Blocker if FPR > 15%** (methodology is unreliable on this data)
- **Warning if FPR > 10%**
- **Weight in trust score:** 20 points

#### 4. Estimator Agreement

All three estimators should agree on:
- **Direction:** Same sign of the lift estimate
- **Magnitude:** Coefficient of variation across estimates < threshold
- **Significance:** Same conclusion about statistical significance

Agreement score = $0.4 \times \text{direction} + 0.3 \times \text{magnitude} + 0.3 \times \text{significance}$

- **Weight in trust score:** 20 points

#### 5. Pre-Period Fit

The synthetic control must closely track the actual holdout during the pre-period. Poor fit indicates the control is unreliable.

- **L2 imbalance:** < 0.10 (target < 0.05)
- **$R^2$:** > 0.90
- **Weight in trust score:** 20 points

### Trust Score

| Score | Interpretation |
|-------|---------------|
| 90–100 | Excellent — high confidence in results |
| 70–89 | Good — results are actionable |
| 50–69 | Marginal — proceed with caution |
| < 50 | Unreliable — do not act on these results |

---

## 8. Key Metrics: iROAS, Incrementality Factor, and CPIA

### Incremental ROAS (iROAS)

**For Marketers:** iROAS is the *true* return on ad spend — not what the ad platform reports, but what a controlled experiment reveals. If your iROAS is 3.0, every $1 you spend on ads generates $3 in revenue that *would not have happened otherwise*.

**For Data Scientists:**

$$\text{iROAS} = \frac{\text{Incremental Revenue}}{\text{Total Ad Spend}}$$

Incremental revenue is computed by scaling the per-DMA-per-day lift estimate to the full treatment group and test duration:

$$\text{Incremental Revenue} = \hat{\tau}_{\text{daily}} \times n_{\text{treatment}} \times T_{\text{days}}$$

where $\hat{\tau}_{\text{daily}}$ is the ensemble estimate of the daily per-DMA treatment effect.

### Incrementality Factor (IF)

**For Marketers:** The Incrementality Factor tells you how much the ad platform is over-counting. If Facebook reports 10,000 conversions but your IF is 0.6, only 6,000 of those conversions were truly incremental — the other 4,000 would have happened without ads.

$$\text{IF} = \frac{\text{Incremental Conversions (from test)}}{\text{Attributed Conversions (from ad platform)}}$$

| IF Value | Meaning |
|----------|---------|
| > 1.0 | Platform *under*-reports — ads work better than claimed |
| = 1.0 | Platform reporting is accurate |
| 0.5–1.0 | Platform over-reports by up to 2x |
| < 0.5 | Platform over-reports by more than 2x |
| = 0.0 | Ads have zero incremental impact |

### CPIA (Cost Per Incremental Acquisition)

**For Marketers:** CPIA is your true customer acquisition cost — what it actually costs to acquire one customer who wouldn't have bought without ads.

$$\text{CPIA} = \frac{\text{Total Ad Spend}}{\text{Incremental Conversions}}$$

CPIA is typically much higher than the CPA your ad platform reports, because it only counts customers the ads actually brought in. Use CPIA for cross-channel comparison to find which channels deliver the cheapest truly incremental customers.

---

## 9. Spend Response Curves

### What It Tells You

**For Marketers:** The spend response curve shows how your incremental revenue changes as you spend more (or less) on ads. It answers the critical question: *should I spend more, less, or the same?*

At low spend levels, each additional dollar generates strong returns. As you spend more, returns diminish — you saturate the market. The optimal spend level is where the marginal return equals $1 (spending more costs more than it generates).

### The Model

**For Data Scientists:** We fit a Hill saturation function to the relationship between DMA-level ad spend and incremental revenue:

$$R(s) = R_{\max} \cdot \frac{s^\alpha}{K^\alpha + s^\alpha}$$

**Parameters:**
- $R_{\max}$: Maximum achievable incremental revenue (saturation ceiling)
- $K$: Half-saturation spend (spend level at which you achieve 50% of max)
- $\alpha$: Shape parameter ($< 1$: strongly concave, $= 1$: standard Michaelis-Menten)

### Calibration

The Hill curve is calibrated to the causal iROAS from the primary analysis. Without calibration, the curve reflects correlational patterns; with calibration, it reflects causal relationships.

### Outputs

| Metric | Description |
|--------|-------------|
| **Current iROAS** | Return at current spend level |
| **Marginal ROAS** | Return on the *next* dollar spent |
| **Optimal Spend** | Spend level where marginal ROAS = target (default $1) |
| **95% CI** | Bootstrap confidence intervals on optimal spend |
| **Recommendation** | "Increase spend" / "Decrease spend" / "Maintain current" |

---

## 10. Supported Platforms

### Revenue Sources (Measurement)

| Platform | Integration | Data Pulled | Key Details |
|----------|------------|-------------|-------------|
| **Shopify** | Admin API (v2025-01) | Orders with shipping zip codes | Zip-to-DMA mapping; filters paid/partially refunded |
| **Amazon** | SP-API with RDT | Orders with shipping addresses | Restricted Data Token for PII access; supports all US marketplaces |

### Ad Platforms (Treatment)

| Platform | Integration | Capabilities | Key Details |
|----------|------------|--------------|-------------|
| **Facebook** | Marketing API (Graph v22.0) | Deploy holdout, DMA-level spend reporting, revert | Targets at ad set level; supports DMA exclusions via `geo_markets` |
| **YouTube** | Google Ads API | Deploy holdout, DMA-level spend reporting, revert | Targets video campaigns; native DMA (Metro) support |

### Measurement Scope

Tests can measure incremental impact on:
- **Shopify only:** Direct-to-consumer revenue
- **Amazon only:** Marketplace revenue
- **Shopify + Amazon:** Cross-platform total (captures halo effects where ads on one platform drive sales on another)

---

## 11. Feasibility Gating

### Pre-Test Gate

Before running a test, the platform checks whether it's likely to produce actionable results. Running an underpowered test wastes time and ad spend.

| Check | Threshold | Action |
|-------|-----------|--------|
| Holdout DMAs | ≥ 5 | BLOCK if fewer |
| Treatment DMAs | ≥ 5 | BLOCK if fewer |
| Historical data | ≥ 4 weeks | BLOCK if less |
| MDE | ≤ 30% | BLOCK if higher |
| MDE | ≤ 20% | WARNING if higher |
| Simulated power | ≥ 50% | BLOCK if lower |
| Simulated power | ≥ 70% | WARNING if lower |
| Power score | ≥ 50 | BLOCK if lower |
| Power score | ≥ 70 | WARNING if lower |
| False positive rate | ≤ 15% | BLOCK if higher |

### What MDE Means in Practice

| MDE | Interpretation |
|-----|---------------|
| ≤ 10% | Excellent — can detect subtle effects |
| 10–15% | Good — suitable for most channels |
| 15–20% | Acceptable — will catch medium-to-large effects |
| 20–30% | Marginal — only catches large effects |
| > 30% | Not feasible — the test won't tell you anything useful |

**For Marketers:** MDE is the smallest real effect the test can reliably detect. If your MDE is 20%, and your ads only generate a 10% lift, the test will likely miss it — you'll conclude "no effect" when there actually is one.

---

## 12. Technical Architecture

### System Components

```
incrementality/
├── connectors/          # Data ingestion from APIs
│   ├── shopify.py       # Shopify Admin API
│   ├── amazon.py        # Amazon SP-API + RDT
│   ├── facebook.py      # Meta Marketing API
│   ├── youtube.py       # Google Ads API
│   └── geo.py           # Zip-to-DMA mapping (210 DMAs)
│
├── design/              # Pre-test experiment design
│   ├── optimizer.py     # Auto-design pipeline
│   ├── power_analysis.py # Analytical + simulation power
│   ├── matching.py      # DMA covariate balancing
│   └── spillover.py     # Geographic spillover mitigation
│
├── analysis/            # Post-test causal inference
│   ├── estimators.py    # ASCM, BSTS, DiD, Ensemble
│   ├── iroas.py         # Incremental ROAS + IF + CPIA
│   ├── validation.py    # AA tests, placebos, trust score
│   ├── spend_response.py # Hill saturation curves
│   ├── anomaly.py       # Pre-analysis data quality
│   └── winsorize.py     # Outlier robustness
│
├── orchestrator.py      # Test lifecycle management
├── cli.py               # Command-line interface
├── models.py            # Data models (Pydantic)
├── dma.py               # 210 US DMA definitions
├── reporting.py         # Report generation
└── dashboard/           # Streamlit web UI
```

### Interfaces

**CLI:**
```bash
incrementality design --channel facebook --measure shopify_and_amazon
incrementality execute --test-id test_abc123
incrementality analyze --test-id test_abc123
incrementality revert --test-id test_abc123
```

**Streamlit Dashboard:** Interactive web UI for test design, monitoring, and analysis with visualizations.

### Statistical Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| Significance level ($\alpha$) | 0.05 | Two-sided type I error rate |
| Target power ($1 - \beta$) | 0.80 | Probability of detecting a true effect |
| Min test duration | 2 weeks | Minimum experiment length |
| Max test duration | 12 weeks | Maximum experiment length |
| Lookback period | 12 weeks | Historical data for variance estimation |
| Min DMAs per cell | 5 | Minimum treatment or holdout DMAs |
| Max holdout fraction | 50% | Upper bound on holdout size |
| Target holdout fraction | 25% | Preferred holdout size |
| Balance tolerance (SMD) | 0.10 | Maximum standardized mean difference |

---

## 13. Glossary

| Term | Definition |
|------|-----------|
| **ASCM** | Augmented Synthetic Control Method — a doubly robust causal estimator that combines synthetic control weights with an outcome model |
| **BSTS** | Bayesian Structural Time Series — a Bayesian forecasting method used for counterfactual estimation |
| **CPIA** | Cost Per Incremental Acquisition — total ad spend divided by truly incremental conversions |
| **DiD** | Difference-in-Differences — a causal estimator that compares pre/post changes between treatment and control groups |
| **DMA** | Designated Market Area — one of 210 non-overlapping geographic regions in the US defined by Nielsen |
| **FPR** | False Positive Rate — the probability of detecting a non-existent effect (should be ~5% at $\alpha = 0.05$) |
| **Holdout** | The group of DMAs where ads are turned off during the test |
| **IF** | Incrementality Factor — ratio of measured incremental conversions to platform-attributed conversions |
| **iROAS** | Incremental Return on Ad Spend — true causal revenue generated per dollar of ad spend |
| **MDE** | Minimum Detectable Effect — the smallest effect size a test can reliably detect |
| **Power** | The probability of detecting a true effect (1 - Type II error rate) |
| **Re-randomization** | A design technique that repeatedly randomizes treatment assignment until covariate balance is achieved |
| **SMD** | Standardized Mean Difference — a measure of covariate balance between treatment and holdout groups |
| **Spillover** | Contamination of holdout DMAs due to geographic proximity to treatment DMAs |
| **SUTVA** | Stable Unit Treatment Value Assumption — one unit's outcome depends only on its own treatment, not others' |
| **Treatment** | The group of DMAs where ads continue running as usual during the test |
| **Trust Score** | A composite 0–100 score reflecting the reliability of test results |

---

## References

- Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The Augmented Synthetic Control Method. *Journal of the American Statistical Association*, 116(536), 1789–1803.
- Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., & Scott, S. L. (2015). Inferring causal impact using Bayesian structural time-series models. *Annals of Applied Statistics*, 9(1), 247–274.
- Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2022). An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls. *Journal of the American Statistical Association*, 116(536), 1849–1864.
- Morgan, K. L., & Rubin, D. B. (2012). Rerandomization to improve covariate balance in experiments. *Annals of Statistics*, 40(2), 1263–1282.
- Meta Open Source. GeoLift: Geo-based experimental design and analysis. [github.com/facebookincubator/GeoLift](https://github.com/facebookincubator/GeoLift)

---

*Built for brands making 8-figure ad spend decisions. Because "Facebook said it worked" isn't good enough.*
