# Replication Package

## Is the Idiosyncratic Volatility Premium a Survivorship Artifact? Quality, Disorder, and Market Temperature in the Cross-Section of Equity Returns

**Ethan Wuang** · Independent Researcher, Lake Forest, Illinois · June 2026

---

### Abstract

The apparent positive relationship between idiosyncratic volatility and returns in large-cap equity panels is a survivorship artifact. In a survivorship-corrected full-universe panel of 15,522 firms, including 11,178 subsequently delisted companies, the disorder premium collapses from FM t = +4.80 to FM t = +0.02, and a synthetic stress test shows the +13.4%/yr quintile premium is fully eliminated at empirically observed delisting rates of 0.5–1.0%/month among high-disorder survivors. Prior positive-entropy findings, including Fu (2009) and Ormos and Zibriczky (2014), reflect the same class of sample-construction artifact. What survives correction is a quality premium: the gross profit margin stability channel strengthens to FM t = +3.46, clearing the Harvey-Liu-Zhu threshold. A Gibbs free energy decomposition (ΔG = ΔH − TΔS) motivates a secondary, directional finding: the entropy loading co-varies positively with market temperature (HAC t = +2.70) while the stability loading does not, consistent with the equation's functional form; this pattern is concentrated post-2009 and awaits monthly-frequency full-universe replication.

**Keywords:** Cross-sectional returns, idiosyncratic volatility, survivorship bias, quality premium, gross profit margin stability, market temperature, Fama-MacBeth, thermodynamic analogy

**JEL:** G12, G14, C52, C58

---

### Data Sources (NOT included in this repo)

| Dataset | Source | Access |
|---|---|---|
| Sharadar SF1 (fundamentals, full history incl. delisted) | Nasdaq Data Link / Quandl | Paid subscription — `SHARADAR/SF1`, `SHARADAR/TICKERS`, `SHARADAR/SP500` via `nasdaqdatalink` Python package |
| S&P 500 monthly prices (current constituents, survivorship-biased) | Yahoo Finance | Free via `yfinance` package |
| Fama-French factors and 25 portfolios | Ken French Data Library | Free download from `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/` (fetched automatically by `scripts/step1_french_data.py`) |

Raw data files are **not redistributed** in this repository. The `data/` directory is empty; scripts will populate it on first run.

---

### Environment Setup

Python 3.13 was used for all computations.

```bash
pip install -r requirements.txt
```

For the Sharadar pipeline (`scripts/step4_sharadar_panel.py`), you need a Nasdaq Data Link API key with an active Sharadar subscription:

```bash
export NASDAQ_DATA_LINK_API_KEY="your_key_here"
```

The script reads this from the environment; it will raise a clear error if the variable is not set.

---

### Script Execution Order

All scripts in `scripts/` must be run **from the `scripts/` directory** (they resolve `../data/` relative to their own location). Robustness scripts in `robustness/` must be run **from the `robustness/` directory**.

#### Step 1 — French factor data and market temperature T
```bash
cd scripts && python step1_french_data.py
```
Downloads FF5 monthly/daily factors, 25 Size×B/M portfolio returns, and builds the `T` (market temperature) series from 12-month realized variance. Saves to `data/`.

#### Step 2 — S&P 500 price panel (survivorship-biased, current constituents)
```bash
cd scripts && python step2_sp500_prices.py
```
Downloads monthly adjusted-close prices for ~500 current S&P 500 names via yfinance (1988–2023). **Survivorship bias acknowledged**: delisted firms are excluded. Saves `data/stock_prices_monthly.parquet`.

#### Step 3 — Stock-level ΔH, ΔS, T, ΔG variables
```bash
cd scripts && python step3_stock_variables.py
```
Builds price-based proxies:
- **ΔH** = −1 × rolling 60-month std of monthly stock return (return stability)
- **ΔS** = rolling 36-month std of FF3 residuals (idiosyncratic disorder)
- **T** = 12-month realized market variance, normalized (mean 0.04, std 0.02)
- **ΔG** = ΔH_z − T × ΔS_z, cross-sectionally z-scored

Saves `data/variables_stock_monthly.parquet`. This file is then copied to `data/variables_monthly.parquet` (the name the analysis scripts expect):

```bash
cp data/variables_stock_monthly.parquet data/variables_monthly.parquet
```

#### Step 4 — Sharadar SF1 panel and accounting ΔH_GPM (requires Sharadar key)
```bash
cd scripts && python step4_sharadar_panel.py
```
Downloads Sharadar SF1 (all US domestic common stocks, ARY dimension), builds point-in-time monthly fundamentals (GPM, ROE, EPS growth), and computes accounting-based ΔH_GPM (rolling 60-month std of gross profit margin). Merges with the existing panel. Saves `data/merged_with_accounting.parquet`.

This script downloads ~850 MB of data on first run and caches it locally. Subsequent runs skip the download.

#### Step 5 — Fama-MacBeth regressions → Table 2
```bash
cd scripts && python step5_fama_macbeth.py
```
Runs Models A (ΔG only), B (ΔH + ΔS), and C (ΔH + T·ΔS) with and without FF5+momentum controls. Uses Newey-West standard errors (6 lags). Saves `outputs/tables/table2_fama_macbeth.{csv,tex}`.

#### Step 6 — Constraint validity (AIC/BIC/Vuong) → Table 3
```bash
cd scripts && python step6_constraint_validity.py
```
Tests whether imposing the Gibbs structure (Model C) is a valid restriction of Model B using AIC, BIC, and the Vuong (1989) non-nested model comparison test. Saves `outputs/tables/table3_constraint_validity.{csv,tex}`.

#### Step 7 — Survivorship-free quarterly panel → Tables 6 and 7 (partial)
```bash
cd robustness && python R18_sf1_quarterly_survfree.py
```
Builds the full-universe quarterly panel (15,522 tickers, 11,178 delisted) from Sharadar SF1 ARQ dimension. Computes quarterly ΔS (12-quarter rolling FF3 residual iVol), reuses accounting ΔH_GPM from Step 4, runs FM regressions and cluster-robust Wald tests. Saves `results/survivorship_free/R18_sf1_quarterly_results.txt` and `data/merged_sf1_quarterly_survfree.parquet`.

#### Step 8 — Synthetic delisting stress test → Table 6 (Section 4.8)
```bash
cd robustness && python R19_delisting_bias_bound.py
```
Counterfactual stress test: sweeps monthly delisting rates (0%–5%) applied to the most-distressed survivors in the primary S&P 500 panel, using Shumway (1997) delisting returns (−30% NYSE, −55% NASDAQ, −40% blended). Shows the disorder premium collapses at ~0.5–1.0%/month, within empirically observed rates. Saves `results/survivorship_free/R19_delisting_bias_bound.txt`.

#### Step 9 — ΔH window robustness and full-universe FM table → Tables 5 and 7
```bash
cd robustness && python R22_v19_battery.py
```
This is an omnibus revision script that runs four tasks:
- **TASK 2** (lines ~60–150): Full FM table (Panels A/B/C) for the R18 quarterly survivorship-free panel → **Table 7**
- **TASK 3** (lines ~153–197): ΔH window sensitivity sweep (24/36/48/60/72 months) with corrected PIT forward-fill → **Table 5**. Note: the original per-window t(ΔS) discrepancy (t=0.92 vs t=4.80 across windows) was traced to an implementation bug in an earlier script (sparse forward-fill of annual GPM filings); R22 fixes this. The corrected result is FM t(ΔS) = 3.85–4.66 across all five windows.
- **TASK 1**: Pre-2009 high-T vs low-T β_ΔS structural-break test (robustness for Section 4.3)
- **TASK 4**: 1,000-draw block bootstrap distribution of the T·ΔS Wald statistic

Saves `results/revision/R22_v19_battery.txt`.

---

### Script → Table/Section Map

| Paper Table | Script | Output file |
|---|---|---|
| Table 2 — FM Regressions (S&P 500 panel) | `scripts/step5_fama_macbeth.py` | `outputs/tables/table2_fama_macbeth.csv` |
| Table 3 — Constraint Validity (AIC/BIC/Vuong) | `scripts/step6_constraint_validity.py` | `outputs/tables/table3_constraint_validity.csv` |
| Table 5 — ΔH Window Robustness | `robustness/R22_v19_battery.py` (TASK 3) | `results/revision/R22_v19_battery.txt` |
| Table 6 — Survivorship Correction Summary | `robustness/R18_sf1_quarterly_survfree.py` + `R19_delisting_bias_bound.py` | `results/survivorship_free/R18_sf1_quarterly_results.txt`, `R19_delisting_bias_bound.txt` |
| Table 7 — Full-Universe FM Regressions | `robustness/R22_v19_battery.py` (TASK 2) | `results/revision/R22_v19_battery.txt` |
| §4.4 HAC asymmetric prediction (β_ΔS ~ T, HAC t = +2.70) | `robustness/R20_section44_hac.py` | `results/revision/R20_section44_hac.txt` |
| Market temperature T construction | `scripts/step1_french_data.py` | `data/market_temperature.parquet` |
| SF1 panel (ΔH_GPM, accounting variables) | `scripts/step4_sharadar_panel.py` | `data/merged_with_accounting.parquet` |

---

### Flags and Incomplete Coverage

The following are explicitly flagged because a replicator should know before running:

1. **Sharadar API key**: `step4_sharadar_panel.py`, `R18_sf1_quarterly_survfree.py`, and `R19_delisting_bias_bound.py` all require the Sharadar SF1 data to already be on disk (from Step 4). The API key must be set as `NASDAQ_DATA_LINK_API_KEY` before running Step 4. Steps 5 and 6 (Tables 2 and 3) do NOT require Sharadar; they run on the yfinance/French data only.

2. **SEP (price) entitlement gap**: The paper notes that the Sharadar SEP (daily prices for all tickers including delisted) was not accessible under the author's subscription. R18 therefore uses SF1 quarterly filing prices instead of monthly SEP prices. This means the survivorship-free ΔS in R18 is at quarterly (not monthly) frequency, and is coarser than the monthly AHXZ iVol measure used in the primary analysis. The paper is explicit about this limitation.

3. **Table 5 corrected numbers vs. any earlier version**: Earlier drafts of Table 5 showed FM t(ΔS) = 0.92 for the 24-month ΔH window, which appeared to show fragility. R22_v19_battery.py (TASK 3) traced this to a sparse forward-fill bug in the original window-sweep code. The correct FM t(ΔS) is stable at 3.85–4.66 across all five windows. The corrected table is what appears in the final paper.

4. **`variables_monthly.parquet` naming**: The analysis scripts (`step5_fama_macbeth.py`, `step6_constraint_validity.py`) read from `data/variables_monthly.parquet`. Step 3 produces `data/variables_stock_monthly.parquet`. You must copy/rename one to the other before running Steps 5–6 (see Step 3 instructions above). The `scripts/run_stock_analysis.py` helper handles this swap automatically if you want to run all analysis modules in sequence.

5. **T·ΔS FM non-identification**: In a cross-sectional Fama-MacBeth setting, T is constant within each month (the same scalar for all stocks), making T·ΔS and ΔS perfectly collinear in every cross-sectional OLS. This means the FM estimator cannot identify the T·ΔS interaction. The paper addresses this with the pooled-panel two-way-clustered interaction (t = +2.49) and the cluster-robust Wald test (p = 0.013). See Section 4.3 of the paper and `results/revision/SUMMARY.txt` for the full explanation.

6. **Comprehensive validation and other robustness scripts**: `robustness/R17_comprehensive_validation.py` (89 KB, ~1,500 lines) runs a battery of 20+ validation checks. The final paper does not cite specific table numbers for all of these but the outputs are in `results/revision/`. `R20_section44_hac.py` specifically addresses the HAC robustness of the §4.4 asymmetric prediction (the "make-or-break" check for the T-scaling result). Neither of these is required to reproduce Tables 2, 3, 5, 6, or 7.

---

### Repository Layout

```
.
├── scripts/
│   ├── step1_french_data.py        # Ken French → factors_monthly, market_temperature
│   ├── step2_sp500_prices.py       # yfinance → stock_prices_monthly
│   ├── step3_stock_variables.py    # price-based ΔH/ΔS/T/ΔG → variables_stock_monthly
│   ├── step4_sharadar_panel.py     # Sharadar SF1 → merged_with_accounting (needs API key)
│   ├── step5_fama_macbeth.py       # Table 2
│   ├── step6_constraint_validity.py# Table 3
│   ├── run_stock_analysis.py       # helper: runs modules 02-10 on the stock panel
│   └── utils.py                    # shared: fama_macbeth(), dm_test(), vuong_test(), etc.
├── robustness/
│   ├── R18_sf1_quarterly_survfree.py  # Tables 6+7 (survivorship-free quarterly FM)
│   ├── R19_delisting_bias_bound.py    # Table 6 (synthetic stress test, Section 4.8)
│   └── R22_v19_battery.py             # Tables 5+7 (ΔH window sweep, full-universe FM)
├── data/                           # populated by scripts; NOT in version control
├── outputs/
│   ├── tables/                     # CSV and LaTeX tables (populated by scripts)
│   └── figures/                    # PNG figures (populated by 09_plots.py)
├── results/
│   ├── survivorship_free/          # R18 + R19 output text files
│   └── revision/                   # R20–R24 output text files
├── requirements.txt
└── .gitignore
```

---

### Citation

If you use this code, please cite:

> Wuang, E. (2026). Is the Idiosyncratic Volatility Premium a Survivorship Artifact? Quality, Disorder, and Market Temperature in the Cross-Section of Equity Returns. Working paper.
