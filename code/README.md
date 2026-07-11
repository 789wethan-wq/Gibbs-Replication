# Code — build and run order

All scripts are plain Python (see `requirements.txt`). Scripts expect to be run
with the repository root as working directory and write to `data/`, `outputs/`,
and `results/` (created as needed). Raw-data builds require
`NASDAQ_DATA_LINK_API_KEY` in the environment and a Sharadar SF1 subscription.

## 1. Data build (`project/`)

| order | script | what it does |
|---|---|---|
| 1 | `project/data_pipeline.py` | Downloads Ken French factors/portfolios (free); builds market temperature T from FF daily market returns. |
| 2 | `project/sharadar_pipeline.py` | Downloads SHARADAR/SF1 + TICKERS via Nasdaq Data Link; builds the S&P 500 monthly panel with point-in-time (`datekey`) fundamental merges (`merged_with_accounting.parquet`, `monthly_fundamentals.parquet`). |
| 3 | `project/00b_stock_data.py`, `project/01b_stock_variables.py` | Stock-level variable construction (ΔH, ΔS, T; z-scores). |
| 4 | `project/run_stock_analysis.py` | Re-runs modules 02–10 on the stock-level panel: summary stats, portfolio sorts, Fama–MacBeth, constraint validity, regime analysis, OOS tests, robustness, plots, paper tables (`outputs/`). |

## 2. Robustness batteries (`robustness/`)

Run in numeric order; each is standalone and writes its own log.

- `R01`–`R11`: the 576-specification robustness battery on the S&P 500 monthly
  panel (variable construction, sample sensitivity, statistical methods, factor
  controls, Vuong stress, regime sensitivity, confounds, microstructure,
  economic significance, multiple testing, bootstrap). Master summary:
  `master_robustness_table.csv`.
- `R12`–`R17`: referee-response and comprehensive-validation rounds.
- `R18_sf1_quarterly_survfree.py`: **the survivorship-free full-universe
  quarterly rebuild** (15,522-ticker universe → 12,449-ticker analysis panel,
  8,937 delisted). Writes `merged_sf1_quarterly_survfree.parquet` and
  `results/survivorship_free/R18_sf1_quarterly_results.txt`.
- `DIAG_survivorship.py`, `DIAG_channels.py`, `DIAG_channel_verification.py`:
  the controlled one-variable-at-a-time decomposition ladder
  (frequency → breadth → survivorship).
- `R19_delisting_bias_bound.py`: synthetic Shumway delisting stress test.
- `R20_section44_hac.py` – `R24_v22_battery.py`: revision-round batteries
  (HAC corrections, pooled two-way-clustered interaction, SE robustness).
- `R25_post_review_experiments.py`: post-review experiments — the
  survival-conditioning (Ormos–Zibriczky) exhibit E1, the SEP entitlement gate
  E2, the OOS year audit V1, full-universe pooled interaction V2, and the
  V3–V5 verification items.
- `A1_panel_count_reconciliation.py`: prints the ticker count at every pipeline
  stage (15,522 / 12,449 / 8,937 reconciliation).

## 3. Derived-data regeneration

`generate_derived_data.py` (this directory) rebuilds everything in
`derived-data/` from the private raw panels. It writes aggregates only.
