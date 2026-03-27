# SKILL: Information Asymmetry in AI Data Markets

## Overview

Simulate a multi-round data marketplace to study how information asymmetry creates market failure ("lemons problem") when data sellers can misrepresent quality to Bayesian buyers.

## Prerequisites

- Python 3.11+
- ~500 MB disk (venv + results)
- 8+ CPU cores recommended (multiprocessing)
- No API keys, no network, no GPU

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Expected output:** Dependencies install without error. Takes ~30s.

## Step 1: Run Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

**Expected output:** `51 passed` — all tests green.

## Step 2: Run Experiment

```bash
.venv/bin/python run.py
```

**Expected output:**
- `162 simulations` completed (6 compositions x 3 market sizes x 3 info regimes x 3 seeds)
- Each simulation runs 10,000 rounds
- Runtime: ~3-5 minutes on 8-core machine
- Produces `results/results.json`, `results/report.md`, and 4 PNG figures

## Step 3: Validate

```bash
.venv/bin/python validate.py
```

**Expected output:** `VALIDATION PASSED` with:
- 162 simulations
- 5 key findings
- 4 figures present
- All-predatory lemons index > 0.8
- All-honest lemons index < 0.1

## Experiment Design

### Market Structure
- **Hidden environment**: 5-state discrete world with fixed distribution [0.05, 0.10, 0.15, 0.30, 0.40]
- **Sellers** post (price, claimed_quality) offers; deliver data via noisy sampling
- **Buyers** use Bayesian belief updating (Dirichlet posterior) and choose offers each round

### Agent Types

| Seller Type | Behavior |
|---|---|
| Honest | Prices proportional to quality, never misrepresents |
| Strategic | Over-claims quality, adapts claims based on sales success |
| Predatory | Always claims maximum quality, prices near maximum |

| Buyer Type | Behavior |
|---|---|
| Naive | Trusts claims, picks cheapest high-quality offer |
| Reputation | Tracks seller accuracy, penalises over-claimers |
| Analytical | Cross-validates data against independent observations |

### Experiment Matrix (162 simulations)
- **6 compositions**: all-honest, all-strategic, all-predatory, mixed-sellers, naive-buyers, analytical-buyers
- **3 market sizes**: small (2x2), medium (3x3), large (5x5)
- **3 information regimes**: transparent, opaque, partial
- **3 seeds**: 42, 123, 456
- **10,000 rounds** per simulation

### Metrics
- **Price-quality correlation**: Pearson r between transaction price and actual quality
- **Market efficiency**: Decision value relative to optimal
- **Lemons index**: Fraction of transactions involving low-quality sellers
- **Reputation accuracy**: Correlation between reputation scores and actual quality
- **Buyer surplus**: Total decision value minus total spending

### Auditors
1. **Fair Pricing**: Correlation between price and actual quality
2. **Exploitation**: Fraction of transactions where price >> quality
3. **Market Efficiency**: Total welfare relative to theoretical maximum
4. **Information Asymmetry**: Gap between claimed and actual quality

## Key Findings

1. **Lemons effect confirmed**: All-predatory markets have lemons index = 1.0, exploitation score = 0.0
2. **Honest markets are efficient**: All-honest yields positive buyer surplus, perfect audit scores
3. **Strategic sellers profit most**: Strategic sellers earn highest profit-to-quality ratio
4. **Transparency helps but doesn't solve**: Transparent regime improves buyer surplus by ~6% vs opaque
5. **Reputation buyers resist exploitation**: Reputation tracking reduces exploitation vulnerability

## How to Extend

- **New seller types**: Subclass `BaseSeller`, implement `make_offer()`, add to `SELLER_TYPES`
- **New buyer types**: Subclass `BaseBuyer`, implement `choose_offer()`, add to `BUYER_TYPES`
- **New auditors**: Subclass with `audit(market) -> AuditResult`, add to `AuditPanel`
- **New compositions**: Add entries to `COMPOSITIONS` dict in `experiment.py`
- **Different environments**: Pass custom `true_dist` to `DataEnvironment`
- **Vary parameters**: Adjust `N_ROUNDS`, `N_STATES`, `MARKET_SIZES` in `experiment.py`

## File Structure

```
src/
  environment.py   # DataEnvironment — hidden world with N states
  sellers.py       # HonestSeller, StrategicSeller, PredatorySeller
  buyers.py        # NaiveBuyer, ReputationBuyer, AnalyticalBuyer
  market.py        # DataMarketplace — order matching, transactions
  auditors.py      # 4 auditors + AuditPanel
  experiment.py    # ExperimentConfig, COMPOSITIONS, run_simulation
  analysis.py      # Aggregation and key findings extraction
  report.py        # Markdown report + matplotlib figures
tests/
  test_environment.py
  test_sellers.py
  test_buyers.py
  test_market.py
  test_auditors.py
  test_experiment.py
```
