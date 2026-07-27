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
- `A_runs_plos.py`, `A3_clustered.py`: PLOS-round robustness (QMJ control,
  microcap screens, FU-interaction asymmetry test).
- `MC_LOCK.py`, `T2_LOCK.py`, `T10_RECON.py`, `PanelC_fix.py`,
  `PanelC_verify_ab.py`, `VERIFY_presub.py`: pre-submission reproducibility
  locks — confirm Table 2/10/Panel C are exactly regenerable from the locked
  panel and flag the Panel C FM min-norm rank-deficiency artifact.
- `OOS_R18_provenance.py`: traces the §4.6 out-of-sample exhibit back to its
  exact source panel and script.
- `M1_sp500_quarterly_ds.py`, `M1b_estimator_reconciliation.py`,
  `M2_size_orthogonalized.py`: constant-measurement / size-orthogonalized
  survivorship tests (R4 gate items).
- `M9_oz_entropy_ladder.py`, `M9b_oz_entropy_fixedgrid.py`,
  `M9_M2_entropy_size_orthogonalized.py`: the Ormos–Zibriczky entropy-ladder
  reproduction (§5.2) under fixed-grid and size-orthogonalized binning.
- `R21_dh_degeneracy_audit.py`, `R22_horizon_normalization.py`,
  `R23_r19_delta_calibration.py`, `R3_confirm_2_5.py`, `R3_date_fe.py`:
  later-round revision batteries distinct from the same-numbered `R21`–`R23`
  batteries above (ΔH degeneracy check, horizon normalization, R19 δ
  calibration, Table 2.5 confirmation, date fixed effects).
- `D1_placebo_characteristic_test.py`, `D1a_2x2_cell.py`,
  `D1b_triple_interaction.py`: survival × size 2×2 and placebo-characteristic
  tests on the corrected R18 panel.
- `D2_size_decile_reliability.py`, `D2_corrected_split_half.py`,
  `D2_reliability_x_size.py`, `D3_cell.py`, `D3_turnover_costs_risk.py`,
  `D4_crosspanel_table.py`, `D4_lagged_cap_rerun.py`, `D5_financials_table.py`,
  `D6_mirror_image_dH.py`: reliability-by-size, turnover/cost, cross-panel,
  lagged-capitalization, and mirror-image-ΔH referee-response cells (V34,
  D1–D7).
- `E1_manuscript_edit.py`, `E1_E3_build_placebos.py`,
  `E1_dispersion_normalized_placebos.py`, `E2_value_weighted_dS.py`,
  `E3_survival_ladder_placebos.py`, `E4_temperature_audit.py`,
  `E5_cap_timing_grid_dH.py`, `E6_filing_vs_listing_survivorship.py`,
  `E7_monotone_hazard_R19.py`: placebo-characteristic and dispersion-
  normalization battery (V38, E1–E8; E8 is a documentation note, not a
  script — see `results/revision/E8_price_source_confirmation.txt`).
- `F1_excluded_ticker_audit.py`, `F1_supplement_rawsd.py`,
  `F2_cap_threshold_universe.py`, `F3_expanding_window_T.py`,
  `F4_dH_filing_density.py`, `F5_successor_symbol_validation.py`,
  `F5b_full_population_edgar.py`, `F5c_edgar_sensitivity_rerun.py`,
  `F_facts.py`: excluded-ticker waterfall, cap-threshold universe,
  expanding-window T, ΔH filing density, and SEC EDGAR successor-symbol
  validation (V39, F1–F5c; F5b scales F5's 100-firm sample to the full
  4,593-firm population, F5c reruns the FM collapse excluding EDGAR-confirmed
  M&A firms).
- `G1_yfinance_wiki_crosscheck.py`: independent price cross-check against
  Nasdaq Data Link's WIKI/PRICES table (CRSP is not accessible in this
  environment; Stooq is bot-gated).
- `H1_terminal_return_cell.py`: terminal-return sensitivity grid applied
  directly to the R18 panel (Shumway-convention δ ∈ {0, −10%, −30%, −55%,
  −100%}, uniform and EDGAR-confirmed-acquisition-conditional variants) — the
  return-denominated analogue of R19's hazard-rate bound (V46, H1).
- `H2_coverage_crosscheck.py`: resolves whether Enron/WorldCom/Lehman
  Brothers (and 18 further well-known 1995–2023 catastrophic delistings) are
  genuinely absent from SF1 or a ticker-mapping artifact (V46, H2).
- `I1_s6_coverage_table.py`: generates the S6 Supporting Information table —
  membership of all 22 names checked in H2 read directly from the R18
  analysis panel object, with the design-matrix hash cross-checked against
  H1's verified anchor (V49, I1).
- `build_SI.py`, `build_table_4_4a_object.py`: Supporting Information and
  Table 4.4a document assembly.

## 3. Derived-data regeneration

`generate_derived_data.py` (this directory) rebuilds everything in
`derived-data/` from the private raw panels. It writes aggregates only.
