# Replication Package

## Idiosyncratic Volatility and Survival Conditioning: A Matched-Convention Comparison of Biased and Delisting-Inclusive US Equity Panels

**Ethan Wuang** · Independent Researcher · July 2026

---

### Abstract

For a long-horizon idiosyncratic volatility measure (36-month rolling Fama-French three-factor residual dispersion), we compare a current S&P 500 monthly panel against a delisting-inclusive full-universe SF1 quarterly panel (12,449 firms, 72% later delisted; 10,699 in the regression sample). Under a matched statistical convention the two panels differ significantly (t = −4.11 pooled, −5.19 Fama-MacBeth-family). Holding return frequency and then measurement window fixed in turn isolates survivorship correction as the dominant remaining driver of a three-step decomposition (FM t: +4.70 → +4.39 → +3.65). The corrected panel's premium is an imprecise near-zero, not a confirmed zero (FM t = +0.02, 95% CI ≈ ±3.8%/yr; equivalence tests against ±2–3%/yr bounds fail); an instrumental-variable correction for measurement error gives a similarly imprecise estimate (t = −0.04) rather than a hidden premium. A survival-length ladder raises the estimate to t = +3.23 among 27-year continuous survivors, but firms merely listed early, with no survival requirement, already recover 90–93% of that movement, so the ladder is predominantly a birth-cohort effect, not independent evidence for survivorship. A second, unrelated characteristic, gross profit margin stability, strengthens under the identical correction (FM t = +3.46) and is more broadly robust. We conclude the two panels are reliably different, the corrected panel alone cannot rule out a meaningful premium, and survival conditioning is a plausible but not cleanly isolated contributor to the difference.

**Keywords:** Cross-sectional returns, idiosyncratic volatility, survivorship bias, quality premium, gross profit margin stability, market variance, Fama-MacBeth, delisting bias

**JEL:** G12, G14, C52, C58

**A note on terminology.** Earlier drafts of this project (and some script/variable names still in this repository) used a Gibbs free-energy analogy — ΔG = ΔH − TΔS, with ΔH standing in for margin stability, ΔS for idiosyncratic volatility, and T for market variance. The submitted manuscript retains this mapping only as a one-paragraph hypothesis-generating device in the Introduction and drops the thermodynamic labels from Section 3 onward in favor of plain names: **STAB** (margin stability, was ΔH), **IVOL** (idiosyncratic volatility, was ΔS), **MV** (market variance, was T). Script and variable names in `code/` predate that renaming in places — `ΔH`/`ΔS`/`T` in a script or output file correspond to `STAB`/`IVOL`/`MV` in the current manuscript.

---

### ⚠ Repository sync status — read before citing table numbers

This repository is synced through the pre-print round corresponding to CITATION.cff's `v49-plos-submission` tag. The submitted manuscript is a later revision (internally, V54) that added a full round of response to external review after this repository was last synced, including three new analyses **not yet present as scripts in this repository**:

1. **Left-truncation-only ladder** — tests whether the survival-conditioning ladder (Table 11) is driven by continuous survival specifically or by birth-cohort/early-listing selection. Finding: firms merely listed early recover 90–93% of the ladder's movement with no survival requirement at all — this materially changed how the manuscript characterizes that evidence.
2. **Known-premium validation on the biased panel** — re-runs the size/momentum/value/profitability recovery check on the S&P 500 panel at matched quarterly spacing, to separate a generic quarterly-construction artifact from a corrected-panel-specific instrument-power problem.
3. **Split-sample instrumental-variable correction** — a Shanken (1992)-style errors-in-variables correction for IVOL's measurement noise, sharper than the informal disattenuation bound used earlier.

The manuscript's title, abstract, and several sections were revised in light of these results (in particular, the title no longer asserts the premium "is a survivorship artifact," and the survival-ladder discussion now explicitly flags the birth-cohort confound). **Before the archival Zenodo release is cut, this repository's `code/robustness/` should be synced with the three scripts implementing the above** (working titles: `REV4_E1_left_truncation_ladder.py`, `REV4_E2_known_premium_biased_panel.py`, `REV4_E3_eiv_correction.py`), and the table-numbering map below should be re-checked against the actual submitted PDF/DOCX, not assumed from this file. Manuscript table numbers have shifted across every major drafting round; treat the map below as a script-to-topic index, not a guaranteed table-number lookup.

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

1. **Aggregate-only (no subscription):** every time-series regression, portfolio L/S statistic, and figure can be recomputed directly from `derived-data/` — monthly and quarterly Fama-MacBeth loading series (β_STAB,t, β_IVOL,t, with MV), quintile/long-short portfolio return series for both panels, the full specification-battery master table, and the survival-conditioning summary.
2. **Full rebuild:** follow the script order below (requires the Sharadar subscription; a few hours, dominated by the SF1 download and rolling-residual IVOL estimation).

---

### Script Execution Order

Scripts resolve `data/`, `outputs/`, and `results/` relative to the repository root (pipeline scripts in `code/project/`) or their own directory (robustness scripts in `code/robustness/` expect `../data`-style layout of the private working copy — run them from `code/robustness/`).

#### Step 1 — French factor data and market variance (MV)
```bash
python code/project/data_pipeline.py
```
Downloads FF5 monthly/daily factors, 25 Size×B/M portfolio returns, and builds the `MV` (market variance) series from 12-month (252-trading-day) realized variance of the FF daily market return. Saves to `data/`.

#### Step 2 — S&P 500 price panel (survivorship-biased, current constituents)
```bash
python code/project/00b_stock_data.py
```
Downloads monthly adjusted-close prices for ~500 current S&P 500 names via yfinance. **Survivorship bias acknowledged**: delisted firms are excluded; this panel exists precisely so it can be compared against the corrected one. Saves `data/stock_prices_monthly.parquet`.

#### Step 3 — Stock-level STAB, IVOL, MV variables (price-based construction)
```bash
python code/project/01b_stock_variables.py
```
Builds price-based proxies:
- **STAB** = −1 × rolling 60-month std of monthly stock return (return stability)
- **IVOL** = rolling 36-month std of FF3 residuals (idiosyncratic volatility)
- **MV** = 12-month realized market variance, normalized
- Composite = STAB_z − MV × IVOL_z, cross-sectionally z-scored

Saves `data/variables_stock_monthly.parquet`. `code/project/run_stock_analysis.py` swaps this in as `data/variables_monthly.parquet` (the name modules 02–10 expect) and runs the full analysis sequence.

#### Step 4 — Sharadar SF1 panel and accounting-based STAB (requires Sharadar key)
```bash
python code/project/sharadar_pipeline.py
```
Downloads Sharadar SF1 (all US domestic common stocks), builds point-in-time monthly fundamentals (GPM, ROE, EPS growth) merged on `datekey`, and computes accounting-based STAB (rolling 60-month std of gross profit margin). Saves `data/merged_with_accounting.parquet` and `data/monthly_fundamentals.parquet`. Downloads ~850 MB on first run and caches locally.

#### Step 5 — Fama-MacBeth regressions (primary accounting-based table)
```bash
python code/project/04_fama_macbeth.py
```
Models A (composite only), B (STAB + IVOL), and C (STAB + MV·IVOL), with and without FF5+momentum controls, Newey-West standard errors.

#### Step 6 — Structural tests (AIC/BIC/Vuong)
```bash
python code/project/05_constraint_validity.py
```
Tests whether imposing the composite structure (Model C) is a valid restriction of Model B (AIC, BIC, Vuong 1989). This test's one structural prediction is withdrawn in the manuscript (Section 4.3) on the evidence of the placebo battery (Step 9 below) — the script and its output remain in the repository for transparency, but the result is not advanced as a finding.

#### Step 7 — Survivorship-free quarterly panel (headline correction)
```bash
cd code/robustness && python R18_sf1_quarterly_survfree.py
```
Builds the full-universe quarterly panel (12,449 analysis-panel firms, 8,937 delisted; built from 15,522 SF1-covered tickers before the valid-return/IVOL filter) from Sharadar SF1 ARQ dimension. Computes quarterly IVOL (12-quarter rolling FF3 residual volatility), reuses accounting STAB from Step 4, runs FM regressions and cluster-robust Wald tests. Saves `results/survivorship_free/R18_sf1_quarterly_results.txt` and `data/merged_sf1_quarterly_survfree.parquet`. Stage-by-stage ticker counts are reproduced by `code/robustness/A1_panel_count_reconciliation.py`.

#### Step 8 — Synthetic delisting stress test
```bash
cd code/robustness && python R19_delisting_bias_bound.py
```
Counterfactual stress test: sweeps monthly delisting rates (0%–5%) applied to the most-distressed survivors in the primary S&P 500 panel, using terminal delisting returns of −30% for NYSE/AMEX performance delistings (Shumway, 1997), −55% for Nasdaq (Shumway and Warther, 1999), and a −40% blend for the primary sweep. The premium is already statistically indistinguishable from zero at δ = 0.5%/month and turns significantly negative by δ = 1.0%/month — both within empirically observed delisting-rate ranges. Saves `results/survivorship_free/R19_delisting_bias_bound.txt`.

#### Step 9 — Placebo battery and window robustness
```bash
cd code/robustness && python R22_v19_battery.py
```
Omnibus revision script: full FM table for the R18 quarterly panel, STAB window sensitivity sweep (24/36/48/60/72 months), pre/post-2009 structural-break test, and a 1,000-draw block bootstrap of the variance-scaling Wald statistic. See `E1_dispersion_normalized_placebos.py` and `D1_placebo_characteristic_test.py` for the placebo-characteristic tests (size, book-to-market, momentum, beta) that show variance-conditioning of cross-sectional loadings is generic rather than distinctive to IVOL.

#### Step 10 — Survival-conditioning ladder
```bash
cd code/robustness && python E3_survival_ladder_placebos.py
```
Imposes k-year continuous-survival conditioning (k ∈ {0, 5, 10, 15, 20, 25, 27}) on the R18 full-universe panel, mimicking the Ormos-Zibriczky (2014) "available for the whole period" design, and re-runs the IVOL quintile sort and Model B FM inside each conditioned panel. The k=0 row reproduces R18 exactly; FM t(IVOL) rises monotonically to +3.23 at k=27. **See the repository-sync note above**: the manuscript's current interpretation of this result depends on a follow-up left-truncation-only test not yet in this repository.

---

### Script → Topic Map

This is an index by **topic**, not a verified table-number lookup — see the sync-status note above. Cross-check exact "Table N" labels against the submitted manuscript before citing them.

| Topic | Script(s) | Output |
|---|---|---|
| Summary statistics | `code/project/02_summary_statistics.py` | `results/paper_tables/table0_panel{A,B,C}_*.csv` |
| Primary Fama-MacBeth regressions (accounting-based) | `code/project/04_fama_macbeth.py` | `results/paper_tables/table2_fama_macbeth.csv` |
| Composite-structure validity (AIC/BIC/Vuong) | `code/project/05_constraint_validity.py` | `results/paper_tables/table3_constraint_validity.csv` |
| Placebo-characteristic second-step test (variance-scaling genericity) | `code/robustness/D1_placebo_characteristic_test.py`, `E1_dispersion_normalized_placebos.py` | `results/revision/D1_placebo_characteristic_test.txt`, `E1_dispersion_normalized_placebos.txt` |
| Markov regime-conditional factor loadings | `code/project/06_regime_analysis.py` | `results/paper_tables/table4_regime_analysis.csv` |
| STAB window-length robustness (monthly panel) | `code/robustness/R22_v19_battery.py` (TASK 3) | `results/revision/R22_v19_battery.txt` |
| Full-universe FM regressions, R18 corrected quarterly panel | `code/robustness/R18_sf1_quarterly_survfree.py`, `R22_v19_battery.py` (TASK 2) | `results/survivorship_free/R18_sf1_quarterly_results.txt`, `results/revision/R22_v19_battery.txt` |
| Synthetic delisting stress test | `code/robustness/R19_delisting_bias_bound.py` | `results/survivorship_free/R19_delisting_bias_bound.txt` |
| Survival-conditioning ladder | `code/robustness/E3_survival_ladder_placebos.py` | `results/revision/E3_survival_ladder_placebos.txt`, `derived-data/e1_survival_conditioning_summary.csv` |
| Entropy check vs. Ormos-Zibriczky (2014) construction | `code/robustness/M9_oz_entropy_ladder.py`, `M9b_oz_entropy_fixedgrid.py` | `results/revision/M9b_oz_entropy_fixedgrid.txt` |
| Constant-measurement test (t(IVOL) by estimator × frequency) | `code/robustness/M1_sp500_quarterly_ds.py`, `M1b_estimator_reconciliation.py` | `results/revision/M1_sp500_quarterly.txt`, `M1b_estimator_reconciliation.txt` |
| Size-orthogonalized test (survivorship-free by size cut) | `code/robustness/M2_size_orthogonalized.py` | `results/revision/M2_size_orthogonalized.txt` |
| Out-of-sample tests (both panels) | `code/robustness/R17_comprehensive_validation.py`, `OOS_R18_provenance.py` | `results/robustness_battery/R17_ceiling_tests.txt`, `results/revision/OOS_R18_provenance.txt` |
| Asymmetric temperature/variance prediction, HAC-corrected | `code/robustness/R20_section44_hac.py` | `results/revision/R20_section44_hac.txt` |
| Full specification battery (500+ tested variants) | `code/robustness/R01–R17_*.py` (master build: `master_robustness_table.py`) | `derived-data/master_robustness_table.csv` |
| Panel-count reconciliation | `code/robustness/A1_panel_count_reconciliation.py` | `results/revision/A1_panel_count_reconciliation.txt` |
| Cross-panel difference test (matched-convention, stacked + FM-family) | `code/robustness/R25_cross_panel_diff.py`, `D4_crosspanel_table.py` | `results/revision/R25_cross_panel_diff.txt`, `D4_crosspanel_table.txt` |
| Reliability-stratified estimation / disattenuation bound | `code/robustness/R26_build_reliability.py`, `R26_reliability_stratified.py`, `D2_reliability_x_size.py` | `results/revision/R26_reliability_stratified.txt`, `D2_reliability_x_size.txt` |
| Financial-firm exclusion (SIC 6000–6999) | `code/robustness/R27_financial_exclusion.py`, `D5_financials_table.py` | `results/revision/R27_financial_exclusion.txt`, `D5_financials_table.txt` |
| Size × survival interaction | `code/robustness/R28_size_survival_2x2.py`, `D1a_2x2_cell.py`, `D1b_triple_interaction.py` | `results/revision/R28_size_survival_2x2.txt`, `D1a_2x2_cells.txt`, `D1b_triple_interaction.txt` |
| Terminal-return sensitivity / SF1 delisting-coverage cross-check | `code/robustness/H1_terminal_return_cell.py`, `H2_coverage_crosscheck.py`, `I1_s6_coverage_table.py` | `results/revision/I1_s6_coverage_table.txt` |
| Independent price validation (yfinance/WIKI cross-check) | `code/robustness/G1_yfinance_wiki_crosscheck.py` | `results/revision/G1_yfinance_wiki_crosscheck.txt` |
| Known-premium validation (gross profitability, B/M, size, momentum) | `code/robustness/SPEC_G2_review2_experiments.py` *(not yet synced — see note above)* | — |
| Reliability ladder by size tercile | `code/robustness/SPEC_T6_reliability_ladder.py` *(not yet synced — see note above)* | — |
| Left-truncation-only ladder / biased-panel known-premium / EIV correction | `REV4_E1/E2/E3_*.py` *(not yet synced — see note above)* | — |

---

### Flags and Incomplete Coverage

1. **Sharadar API key**: `sharadar_pipeline.py` requires `NASDAQ_DATA_LINK_API_KEY`; the robustness scripts require the SF1 data to already be on disk (from Step 4). Steps 5–6 run on the yfinance/French data only, no subscription needed.

2. **SEP (price) entitlement gap**: Sharadar SEP (daily prices incl. delisted tickers) was not accessible under the author's subscription. R18 therefore uses SF1 quarterly filing prices, so the survivorship-free IVOL is quarterly, coarser than the monthly measure in the primary S&P 500 analysis. The manuscript's harmonization decomposition (Section 4.6) separates the effect of this frequency change from the effect of survivorship correction itself.

3. **The survival-conditioning ladder is not clean survivorship evidence.** As of the manuscript's final revision, a dedicated test shows that firms merely listed early in the sample (no continuous-survival requirement) already recover 90–93% of the ladder's t(IVOL) movement from the unconditional null to the 27-year-survivor rung. The ladder is predominantly a birth-cohort/left-truncation effect, not clean confirmation of the survivorship mechanism specifically. This finding is not yet reflected in this repository's scripts (see sync-status note above).

4. **IVOL–momentum/size instrument-power caveat.** A known-premium validation shows the corrected panel recovers gross profitability and book-to-market but not size or momentum. A follow-up check on the biased S&P 500 panel at matched quarterly spacing shows momentum's failure is generic to the quarterly construction (fails in both panels), but size's failure is specific to the corrected panel — an unresolved instrument-power concern for the corrected panel's IVOL null. Not yet reflected in this repository's scripts.

5. **T·IVOL FM non-identification**: MV is constant within a month, so MV·IVOL and IVOL are perfectly collinear in every cross-sectional OLS — the FM estimator cannot identify the interaction. The manuscript uses the pooled two-way-clustered interaction and the cluster-robust Wald test instead.

6. **Robustness battery**: `code/robustness/R01–R17` form the large specification battery (master summary: `derived-data/master_robustness_table.csv`); R12–R17 are referee-response rounds. None of these is required to reproduce the primary tables.

---

### Repository Layout

```
.
├── code/
│   ├── project/                    # data pipeline + primary tables
│   │   ├── data_pipeline.py        # Step 1: Ken French → factors, market variance MV
│   │   ├── 00b_stock_data.py       # Step 2: yfinance → S&P 500 monthly prices
│   │   ├── 01b_stock_variables.py  # Step 3: price-based STAB/IVOL/MV
│   │   ├── sharadar_pipeline.py    # Step 4: Sharadar SF1 → accounting panel (needs API key)
│   │   ├── 02..10_*.py             # summary stats, sorts, FM, validity, regimes, OOS, plots, tables
│   │   ├── run_stock_analysis.py   # helper: runs modules 02-10 on the stock panel
│   │   └── utils.py
│   ├── robustness/                 # R01-I1 batteries + diagnostics (D/E/F/G/H/I referee-response rounds)
│   │   ├── R01..R11_*.py           # specification-battery scripts
│   │   ├── R12..R17_*.py           # referee-response / comprehensive validation
│   │   ├── R18_sf1_quarterly_survfree.py   # Step 7: survivorship-free panel
│   │   ├── R19_delisting_bias_bound.py     # Step 8: synthetic stress test
│   │   ├── R20..R28_*.py           # revision batteries (R20 = §4.4 HAC test)
│   │   ├── E3_survival_ladder_placebos.py  # Step 10: survival conditioning + placebo ladder
│   │   ├── A1_panel_count_reconciliation.py# §3.1 ticker-count reconciliation
│   │   ├── D1..D6_*.py              # referee-response cells (survival x size, reliability, placebos)
│   │   ├── E1..E7_*.py              # referee-response battery (placebo/dispersion normalization)
│   │   ├── F1..F5c_*.py             # referee-response battery (excluded-ticker audit, EDGAR checks)
│   │   ├── G1_yfinance_wiki_crosscheck.py  # independent price cross-check (WIKI/PRICES)
│   │   ├── H1_terminal_return_cell.py      # terminal-return sensitivity grid on R18
│   │   ├── H2_coverage_crosscheck.py       # SF1 coverage cross-check (catastrophic delistings)
│   │   ├── I1_s6_coverage_table.py         # S6 Supporting Information coverage table
│   │   └── DIAG_*.py               # one-variable-at-a-time decomposition ladder
│   ├── generate_derived_data.py    # rebuilds derived-data/ from the private raw panels
│   ├── requirements.txt
│   └── README.md                   # detailed run order
├── derived-data/                   # AGGREGATES ONLY (no vendor rows)
│   ├── fm_loadings_monthly_sp500.csv
│   ├── fm_loadings_quarterly_fulluniverse.csv
│   ├── portfolio_returns_monthly_sp500.csv
│   ├── portfolio_returns_quarterly_fulluniverse.csv
│   ├── e1_survival_conditioning_summary.csv   # survival ladder
│   ├── master_robustness_table.csv            # specification-battery summary
│   └── decomposition_ladder/                  # DIAG writeups
├── results/
│   ├── survivorship_free/          # R18, R19, DIAG logs
│   ├── revision/                   # R20-R28 + A1 logs
│   ├── robustness_battery/         # R01-R17 outputs
│   └── paper_tables/               # primary-table CSVs
├── LICENSE                         # MIT (code only)
├── README.md
└── .gitignore
```

---

### License

**Code license:** the code in this repository (everything under `code/`) is released under the MIT License (see `LICENSE`). This license covers the code only.

**Data license — separate and more restrictive:** the raw Sharadar SF1 fundamentals data used to construct the panels is **not included and not redistributed** in this repository or in the archived Zenodo deposit. SF1 is a licensed, paid Nasdaq Data Link product; access requires a subscription obtained independently by the researcher on the same terms as any other subscriber (see Data Sources above). The `derived-data/` directory contains only cross-sectional aggregates derived from SF1 (regression coefficients, portfolio-mean returns, and summary panels) — no firm-level SF1 fields (prices, fundamentals, tickers-with-financials) are redistributed anywhere in this repository or deposit. Yahoo Finance, Ken French Data Library, global-q.org, and AQR data are used under their respective free-access terms; see Data Sources above.

---

### Citation

Machine-readable citation metadata is provided in `CITATION.cff` and `.zenodo.json`. If you use this code, please cite:

> Wuang, E. (2026). Idiosyncratic Volatility and Survival Conditioning: A Matched-Convention Comparison of Biased and Delisting-Inclusive US Equity Panels. Working paper. Replication package: https://github.com/789wethan-wq/Gibbs-Replication
