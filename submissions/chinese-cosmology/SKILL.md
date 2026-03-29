---
name: chinese-cosmology-analysis
description: >
  Computational consistency analysis of Chinese metaphysical systems (BaZi, Zi Wei
  Dou Shu, Wu Xing). Studies cross-system agreement using information theory and
  multi-agent evaluation across 263,000 birth charts spanning a 60-year cycle.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Cross-System Consistency in Chinese Computational Cosmology

This skill analyzes whether three independently-evolved Chinese metaphysical systems — BaZi (八字), Zi Wei Dou Shu (紫微斗数), and Wu Xing Dynamics (五行) — converge on similar assessments when given the same birth datetime. A multi-agent evaluation panel measures cross-system agreement using Pearson correlation, mutual information, domain agreement rates, and Wu Xing predictiveness across ~263,000 birth charts spanning one full 甲子 cycle (1984–2044).

## Prerequisites

- Requires **Python 3.10+**.
- The analysis itself is pure computation with embedded lookup tables (no runtime API/data calls), but initial dependency installation requires package-index access unless dependencies are already present.
- Expected runtime: **~10–20 minutes** on a single CPU (no GPU required).
- All commands must be run from the **submission directory** (`submissions/chinese-cosmology/`).

## Step 1: Environment Setup

Create a virtual environment and install pinned dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Expected: `Successfully installed numpy-... scipy-... matplotlib-... pytest-...`

Verify imports:

```bash
.venv/bin/python -c "import numpy, scipy, matplotlib; print('All imports OK')"
```

Expected: `All imports OK`

## Step 2: Run Unit Tests

Verify all system agent and evaluator modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with `X passed` and exit code 0. Tests cover the calendar engine, BaZi agent, Zi Wei agent, Wu Xing dynamics, evaluator panel, experiment runner, and statistical analysis.

## Step 3: Run the Analysis

Execute the full analysis across 263,000 birth charts:

```bash
.venv/bin/python run.py
```

Expected: Script prints progress per year range, for example:

```
[1/5] Generating birth chart datetimes (1984-2044)...
  262,980 datetimes generated
[2/5] Running BaZi + Zi Wei + Wu Xing agents on 262,980 charts...
  1/262,980 (0%) — 1984
  4,383/262,980 (2%) — 1984
  ...
[3/5] Evaluating cross-system consistency...
  5 domains evaluated
[4/5] Generating report and figures...
  5 figures saved to results/figures/
[5/5] Saving results to results/
```

Script exits with code 0. The following files are created:
- `results/results.json` — all chart records with agent scores
- `results/report.md` — markdown report with correlation tables and key findings
- `results/statistical_tests.json` — Pearson r, mutual information, domain agreement, temporal patterns
- `results/figures/cross_system_correlation.png`
- `results/figures/domain_agreement.png`
- `results/figures/mutual_information.png`
- `results/figures/temporal_patterns.png`
- `results/figures/wuxing_predictiveness.png`

For quick smoke checks (small sample, optional no-figure mode):

```bash
.venv/bin/python run.py --start-year 2000 --end-year 2001 --max-charts 120 --skip-figures --output-dir results_smoke
```

## Step 4: Validate Results

Check that results are complete and statistically consistent:

```bash
.venv/bin/python validate.py
```

Expected output includes:
- `Charts expected: 262,980` and `Records found: 262,980`
- `All 3 systems present in all records: OK`
- `All domain scores in [0, 1]: OK`
- `Correlation summary (BaZi–ZiWei career): ...`
- `95% CI: [...]`, `p-value: ...`, `Bonferroni p: ...`
- `Validation passed.`

To validate a non-default output directory:

```bash
python validate.py --results-file results_smoke/results.json
```

## Step 5: Review the Report

Read the generated markdown report:

```bash
cat results/report.md
```

Review the cross-system correlation table (Pearson r per domain with 95% CI and p-values), Bonferroni-corrected p-values, domain agreement rates, mutual information in nats, Wu Xing predictiveness (R²), and temporal pattern analysis across 天干 and 地支 cycles.

## How to Extend

- **Add a metaphysical system:** Create a new agent module in `src/` implementing the `BaseSystemAgent` interface with an `analyze(datetime) -> dict` method that returns `domain_scores` for the 5 life domains (career, wealth, relationships, health, overall).
- **Add an evaluation metric:** Subclass `BaseEvaluator` in `src/evaluators.py` and add the instance to `EvaluatorPanel` — the panel aggregation and report generation pick it up automatically.
- **Add life domains:** Extend the `DOMAINS` list in `src/evaluators.py` and add corresponding scoring logic in all three system agents (`src/bazi.py`, `src/ziwei.py`, `src/wuxing.py`).
- **Change the time period:** Modify `START_YEAR` and `END_YEAR` in `src/experiment.py` — the 节气 solar term table covers 1984–2044; extending beyond 2044 requires adding rows to `SOLAR_TERMS` in `src/tables.py`.
- **Add Western astrology comparison:** Implement a Western zodiac agent in `src/western_astro.py` following the same `analyze(datetime) -> dict` interface — the evaluator panel will automatically include it in cross-system consistency analysis.
