# Replication Package

## Thermodynamic Non-Equilibrium and the Cross-Section of Equity Returns: A Gibbs Free Energy Decomposition

**Ethan Wuang** · Independent Researcher · July 2026

---

### Abstract

The apparent positive relationship between idiosyncratic volatility and returns in large-cap equity panels is a survivorship artifact. In a survivorship-corrected full-universe panel of 12,449 firms (8,937, or 72%, subsequently delisted), constructed from 15,522 SF1-covered tickers, the disorder premium collapses from FM t = +4.80 to FM t = +0.02, and a synthetic stress test shows the +13.4%/yr quintile premium is fully eliminated at empirically observed delisting rates of 0.5–1.0%/month among high-disorder survivors. Prior positive-entropy findings, including Fu (2009) and Ormos and Zibriczky (2014), reflect the same class of sample-construction artifact. What survives correction is a quality premium: the gross profit margin stability channel strengthens to FM t = +3.46, clearing the Harvey-Liu-Zhu threshold. A Gibbs free energy decomposition (ΔG = ΔH − TΔS) motivates a secondary, directional finding: the entropy loading co-varies positively with market temperature (HAC t = +2.70) while the stability loading does not, consistent with the equation's functional form; this pattern is concentrated post-2009 and awaits monthly-frequency full-universe replication.

**Keywords:** Cross-sectional returns, idiosyncratic volatility, survivorship bias, quality premium, gross profit margin stability, market temperature, Fama-MacBeth, thermodynamic analogy

**JEL:** G12, G14, C52, C58

---

### Data Sources (NOT included in this repo)

| Dataset | Source | Access |
|---|---|---|
| Sharadar SF1 (fundamentals, full history incl. delisted) | Nasdaq Data Link | Paid subscription — `SHARADAR/SF1`, `SHARADAR/TICKERS`, `SHARADAR/SP500` via the `nasdaqdatalink` Python package (https://data.nasdaq.com/databases/SF1) |
| S&P 500 monthly prices (current constituents, survivorship-biased) | Yahoo Finance | Free via the `yfinance` package |
| Fama-French factors and 25 portfolios | Ken French Data Library | Free — https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html (fetched automatically by `code/project/data_pipeline.py`) |
| q-factor model returns | global-q.org | Free — https://global-q.org/factors.html |
| AQR factor data (QMJ et al.) | AQR Data Library | Free — https://www.aqr.com/Insights/Datasets |

Raw data files are **not redistributed** in this repository; the `derived-data/` directory contains only cross-sectional aggregates (regression coefficients and portfolio-mean returns). Scripts populate a local `data/` directory on first run.

**DATEKEY convention.** SF1 rows are point-in-time: `datekey` is the SEC filing (first-availability) date and `calendardate` the normalized fiscal period end. The monthly S&P 500 panel merges fundamentals point-in-time on `datekey` (a filing is usable only from its `datekey` forward, via `merge_asof`). The quarterly full-universe panel snaps each `calendardate` to its calendar quarter and keeps the last filing per ticker-quarter; quarterly returns come from the split-adjusted SF1 `price` field between consecutive quarters plus a dividend yield approximated from TTM `dps`/4.

---

### Environment Setup

Python 3.13 was used for all computations.

```bash
pip install -r code/requirements.txt
```

For the Sharadar pipeline (`code/project/sharadar_pipeline.py`), you need a Nasdaq Data Link API key with an active Sharadar subscription:

```bash
export NASDAQ_DATA_LINK_API_KEY="your_key_here"
```

The code reads the key from the environment only; no key appears in this repository.

---

### Two reproduction paths

1. **Aggregate-only (no subscription):** every time-series regression, portfolio L/S statistic, and figure can be recomputed directly from `derived-data/` — monthly and quarterly Fama-MacBeth loading series (β_ΔH,t, β_ΔS,t, with T), quintile/long-short portfolio return series for both panels, the 576-specification robustness master table, and the Table 8 survival-conditioning summary.
2. **Full rebuild:** follow the script order below (requires the Sharadar subscription; a few hours, dominated by the SF1 download and rolling-residual iVol estimation).

---

### Script Execution Order

Scripts resolve `data/`, `outputs/`, and `results/` relative to the repository root (pipeline scripts in `code/project/`) or their own directory (robustness scripts in `code/robustness/` expect `../data`-style layout of the private working copy — run them from `code/robustness/`).

#### Step 1 — French factor data and market temperature T
```bash
python code/project/data_pipeline.py
```
Downloads FF5 monthly/daily factors, 25 Size×B/M portfolio returns, and builds the `T` (market temperature) series from 12-month (252-trading-day) realized variance of the FF daily market return. Saves to `data/`.

#### Step 2 — S&P 500 price panel (survivorship-biased, current constituents)
```bash
python code/project/00b_stock_data.py
```
Downloads monthly adjusted-close prices for ~500 current S&P 500 names via yfinance. **Survivorship bias acknowledged**: delisted firms are excluded; this panel exists precisely so the paper can correct it. Saves `data/stock_prices_monthly.parquet`.

#### Step 3 — Stock-level ΔH, ΔS, T, ΔG variables
```bash
python code/project/01b_stock_variables.py
```
Builds price-based proxies:
- **ΔH** = −1 × rolling 60-month std of monthly stock return (return stability)
- **ΔS** = rolling 36-month std of FF3 residuals (idiosyncratic disorder)
- **T** = 12-month realized market variance, normalized
- **ΔG** = ΔH_z − T × ΔS_z, cross-sectionally z-scored

Saves `data/variables_stock_monthly.parquet`. `code/project/run_stock_analysis.py` swaps this in as `data/variables_monthly.parquet` (the name modules 02–10 expect) and runs the full analysis sequence.

#### Step 4 — Sharadar SF1 panel and accounting ΔH_GPM (requires Sharadar key)
```bash
python code/project/sharadar_pipeline.py
```
Downloads Sharadar SF1 (all US domestic common stocks), builds point-in-time monthly fundamentals (GPM, ROE, EPS growth) merged on `datekey`, and computes accounting-based ΔH_GPM (rolling 60-month std of gross profit margin). Saves `data/merged_with_accounting.parquet` and `data/monthly_fundamentals.parquet`. Downloads ~850 MB on first run and caches locally.

#### Step 5 — Fama-MacBeth regressions → Table 2
```bash
python code/project/04_fama_macbeth.py
```
Models A (ΔG only), B (ΔH + ΔS), and C (ΔH + T·ΔS), with and without FF5+momentum controls, Newey-West standard errors. Saves `outputs/tables/table2_fama_macbeth.{csv,tex}` (pushed copy: `results/paper_tables/`).

#### Step 6 — Structural tests (AIC/BIC/Vuong) → Table 3
```bash
python code/project/05_constraint_validity.py
```
Tests whether imposing the Gibbs structure (Model C) is a valid restriction of Model B (AIC, BIC, Vuong 1989). Saves `outputs/tables/table3_constraint_validity.{csv,tex}`.

#### Step 7 — Survivorship-free quarterly panel → Tables 6 and 7 (partial)
```bash
cd code/robustness && python R18_sf1_quarterly_survfree.py
```
Builds the full-universe quarterly panel (12,449 analysis-panel firms, 8,937 delisted; built from 15,522 SF1-covered tickers before the valid-return/ΔS filter) from Sharadar SF1 ARQ dimension. Computes quarterly ΔS (12-quarter rolling FF3 residual iVol), reuses accounting ΔH_GPM from Step 4, runs FM regressions and cluster-robust Wald tests. Saves `results/survivorship_free/R18_sf1_quarterly_results.txt` and `data/merged_sf1_quarterly_survfree.parquet`. The stage-by-stage ticker counts are reproduced by `code/robustness/A1_panel_count_reconciliation.py`.

#### Step 8 — Synthetic delisting stress test → Table 6 (Section 4.8)
```bash
cd code/robustness && python R19_delisting_bias_bound.py
```
Counterfactual stress test: sweeps monthly delisting rates (0%–5%) applied to the most-distressed survivors in the primary S&P 500 panel, using terminal delisting returns of −30% for NYSE/AMEX performance delistings (Shumway, 1997), −55% for Nasdaq (Shumway and Warther, 1999), and a −40% blend for the primary sweep. Shows the disorder premium collapses at ~0.5–1.0%/month, within empirically observed rates. Saves `results/survivorship_free/R19_delisting_bias_bound.txt`.

#### Step 9 — ΔH window robustness and full-universe FM table → Tables 5 and 7
```bash
cd code/robustness && python R22_v19_battery.py
```
Omnibus revision script, four tasks:
- **TASK 2**: Full FM table (Panels A/B/C) for the R18 quarterly survivorship-free panel → **Table 7**
- **TASK 3**: ΔH window sensitivity sweep (24/36/48/60/72 months) with corrected PIT forward-fill → **Table 5**. The original per-window t(ΔS) discrepancy (t = 0.92 vs t = 4.80 across windows) was traced to a sparse forward-fill bug in an earlier script; R22 fixes this. Corrected FM t(ΔS) = 3.85–4.66 across all five windows.
- **TASK 1**: Pre-2009 high-T vs low-T β_ΔS structural-break test
- **TASK 4**: 1,000-draw block bootstrap of the T·ΔS Wald statistic

Saves `results/revision/R22_v19_battery.txt`.

#### Step 10 — Survival-conditioning ladder → Table 8
```bash
cd code/robustness && python R25_post_review_experiments.py
```
Imposes k-year continuous-survival conditioning (k ∈ {0, 5, 10, 15, 20, 25, 27}) on the R18 full-universe panel, mimicking the Ormos–Zibriczky (2014) "available for the whole period" design, and re-runs the ΔS quintile sort and Model B FM inside each conditioned panel (experiment E1; the script also runs the V-series verification items). The k=0 row reproduces R18 exactly; the entropy premium rises monotonically to FM t(ΔS) = +3.23 at k=27. Saves `results/revision/R25_post_review.txt`; summary table in `derived-data/e1_survival_conditioning_summary.csv`.

---

### Script → Table/Section Map

| Paper Table | Script | Output file |
|---|---|---|
| Table 1 — Summary Statistics | `code/project/02_summary_statistics.py` | `results/paper_tables/table0_panel{A,B,C}_*.csv` |
| Table 2 — Fama-MacBeth Cross-Sectional Regressions (Primary: Accounting-Based ΔH) | `code/project/04_fama_macbeth.py` | `results/paper_tables/table2_fama_macbeth.csv` |
| Table 3 — T·ΔS Structural Tests | `code/project/05_constraint_validity.py` | `results/paper_tables/table3_constraint_validity.csv` |
| Table 4 — Markov Regime-Conditional Factor Loadings (Price-Based) | `code/project/06_regime_analysis.py` | `results/paper_tables/table4_regime_analysis.csv` |
| Table 5 — Channel Robustness to ΔH Window Length | `code/robustness/R22_v19_battery.py` (TASK 3) | `results/revision/R22_v19_battery.txt` |
| Table 6 — Survivorship Correction, Key Results (R18 Full-Universe SF1 Panel) | `code/robustness/R18_sf1_quarterly_survfree.py` + `R19_delisting_bias_bound.py` | `results/survivorship_free/R18_sf1_quarterly_results.txt`, `R19_delisting_bias_bound.txt` |
| Table 7 — Full-Universe FM Regressions, R18 Survivorship-Corrected Quarterly | `code/robustness/R22_v19_battery.py` (TASK 2) | `results/revision/R22_v19_battery.txt` |
| Table 8 — Entropy Premium as a Function of Required Survival Length | `code/robustness/R25_post_review_experiments.py` (E1) | `results/revision/R25_post_review.txt`, `derived-data/e1_survival_conditioning_summary.csv` |
| §4.4 HAC asymmetric prediction (β_ΔS ~ T) | `code/robustness/R20_section44_hac.py` | `results/revision/R20_section44_hac.txt` |
| Panel-count reconciliation (§3.1) | `code/robustness/A1_panel_count_reconciliation.py` | `results/revision/A1_panel_count_reconciliation.txt` |
| Market temperature T construction | `code/project/data_pipeline.py` | `data/market_temperature.parquet` |
| SF1 panel (ΔH_GPM, accounting variables) | `code/project/sharadar_pipeline.py` | `data/merged_with_accounting.parquet` |

**Label disambiguation.** The paper's Section 4.8 uses paper-internal experiment labels (R18/R19/R20). Paper-R20 — the survival-conditioning demonstration behind **Table 8** — is implemented in the script `R25_post_review_experiments.py` (experiment E1). The separately named script `R20_section44_hac.py` is a different analysis: the paper's **Section 4.4** HAC test. Script filenames follow the repository's chronological R-numbering, not the paper's labels.

---

### Flags and Incomplete Coverage

1. **Sharadar API key**: `sharadar_pipeline.py` requires `NASDAQ_DATA_LINK_API_KEY`; the robustness scripts require the SF1 data to already be on disk (from Step 4). Steps 5–6 (Tables 2–3) run on the yfinance/French data only, no subscription needed.

2. **SEP (price) entitlement gap**: Sharadar SEP (daily prices incl. delisted tickers) was not accessible under the author's subscription — verified by a live entitlement check logged in `results/revision/R25_post_review.txt` (E2). R18 therefore uses SF1 quarterly filing prices, so the survivorship-free ΔS is quarterly, coarser than the monthly AHXZ iVol measure in the primary analysis. The paper is explicit about this limitation.

3. **Table 5 corrected numbers vs. earlier drafts**: earlier drafts showed FM t(ΔS) = 0.92 for the 24-month window; R22 traced this to a forward-fill bug. Corrected t(ΔS) = 3.85–4.66 across all windows; the corrected table is what appears in the paper.

4. **`variables_monthly.parquet` naming**: modules 02–10 read `data/variables_monthly.parquet`; Step 3 writes `data/variables_stock_monthly.parquet`. `code/project/run_stock_analysis.py` handles the swap automatically.

5. **T·ΔS FM non-identification**: T is constant within a month, so T·ΔS and ΔS are perfectly collinear in every cross-sectional OLS — the FM estimator cannot identify the interaction. The paper uses the pooled two-way-clustered interaction (S&P 500 t = +2.49; full universe t = +2.49, `results/revision/R25_post_review.txt` V2) and the cluster-robust Wald test instead. See `results/revision/SUMMARY.txt`.

6. **Robustness battery**: `code/robustness/R01–R11` form the 576-specification battery (master summary: `derived-data/master_robustness_table.csv`); R12–R17 are referee-response rounds (R17 alone runs 20+ validation checks); logs are in `results/robustness_battery/`. None of these is required to reproduce Tables 1–8.

---

### Repository Layout

```
.
├── code/
│   ├── project/                    # data pipeline + Tables 1-4
│   │   ├── data_pipeline.py        # Step 1: Ken French → factors, market temperature T
│   │   ├── 00b_stock_data.py       # Step 2: yfinance → S&P 500 monthly prices
│   │   ├── 01b_stock_variables.py  # Step 3: price-based ΔH/ΔS/T/ΔG
│   │   ├── sharadar_pipeline.py    # Step 4: Sharadar SF1 → accounting panel (needs API key)
│   │   ├── 02..10_*.py             # summary stats, sorts, FM, validity, regimes, OOS, plots, tables
│   │   ├── run_stock_analysis.py   # helper: runs modules 02-10 on the stock panel
│   │   └── utils.py
│   ├── robustness/                 # R01-R25 batteries + diagnostics
│   │   ├── R01..R11_*.py           # 576-spec robustness battery
│   │   ├── R12..R17_*.py           # referee-response / comprehensive validation
│   │   ├── R18_sf1_quarterly_survfree.py   # Step 7: Tables 6+7 (survivorship-free panel)
│   │   ├── R19_delisting_bias_bound.py     # Step 8: Table 6 (synthetic stress test)
│   │   ├── R20..R24_*.py           # revision batteries (R20 = §4.4 HAC test)
│   │   ├── R25_post_review_experiments.py  # Step 10: Table 8 (survival conditioning) + V-series
│   │   ├── A1_panel_count_reconciliation.py# §3.1 ticker-count reconciliation
│   │   └── DIAG_*.py               # one-variable-at-a-time decomposition ladder
│   ├── generate_derived_data.py    # rebuilds derived-data/ from the private raw panels
│   ├── requirements.txt
│   └── README.md                   # detailed run order
├── derived-data/                   # AGGREGATES ONLY (no vendor rows)
│   ├── fm_loadings_monthly_sp500.csv
│   ├── fm_loadings_quarterly_fulluniverse.csv
│   ├── portfolio_returns_monthly_sp500.csv
│   ├── portfolio_returns_quarterly_fulluniverse.csv
│   ├── e1_survival_conditioning_summary.csv   # Table 8
│   ├── master_robustness_table.csv            # 576-spec battery summary
│   └── decomposition_ladder/                  # DIAG writeups
├── results/
│   ├── survivorship_free/          # R18, R19, DIAG logs
│   ├── revision/                   # R20-R25 + A1 logs
│   ├── robustness_battery/         # R01-R17 outputs (576-spec battery)
│   └── paper_tables/               # Tables 1-4 CSVs
├── LICENSE                         # MIT (code only)
├── README.md
└── .gitignore
```

---

### License

Code is released under the MIT License (see `LICENSE`). The license covers the code in this repository only; access to and use of the underlying datasets are governed by the respective providers' terms (Nasdaq Data Link/Sharadar, Yahoo Finance, Ken French Data Library, global-q.org, AQR).

---

### Citation

If you use this code, please cite:

> Wuang, E. (2026). Thermodynamic Non-Equilibrium and the Cross-Section of Equity Returns: A Gibbs Free Energy Decomposition. Working paper.
