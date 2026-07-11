# A Large-Cap Anomaly That Does Not Generalize: Universe Breadth, Survivorship, and the Fragility of the Idiosyncratic-Volatility Premium

Public code and derived-data release accompanying the manuscript (submitted to
PLOS ONE, 2026). Independent research; contact the author via the address on
the manuscript.

## What is in this repository

| Directory | Contents |
|---|---|
| `code/` | All analysis code: the data-build pipeline (`code/project/`) and the full robustness/revision battery R01–R25 plus diagnostic scripts (`code/robustness/`). Run order in `code/README.md`. |
| `derived-data/` | **Aggregate** series sufficient to reproduce every figure and headline test without a data subscription: monthly and quarterly Fama–MacBeth loading series (β_ΔH,t, β_ΔS,t, with T and cross-section sizes), quintile-portfolio and long-short return series for both panels, the survival-conditioning summary (E1), the one-variable-at-a-time decomposition ladder, and the 576-specification robustness master table. |
| `results/` | Full text logs of every robustness and revision battery: the 576-spec battery output (`results/robustness_battery/`), the survivorship-free rebuild and delisting-bound logs (`results/survivorship_free/`), and the revision-round batteries R20–R25 including the post-review experiment log (`results/revision/R25_post_review.txt`). |

## Data access (raw data are NOT redistributed)

Raw security-level data are licensed and cannot be redistributed here. The
`derived-data/` directory contains only cross-sectional aggregates (regression
coefficients and portfolio-mean returns); no vendor rows appear anywhere in
this repository.

To rebuild the raw panels you need:

1. **Sharadar Core US Fundamentals, via Nasdaq Data Link** (paid subscription):
   https://data.nasdaq.com/databases/SF1
   - Tables used: **SHARADAR/SF1** (fundamentals, `ARQ` dimension — as-reported
     quarterly; fields incl. `price`, `dps`, `marketcap`, gross-profit inputs)
     and **SHARADAR/TICKERS** (universe metadata: `permaticker`, `category`,
     `exchange`, `isdelisted`, `currency`).
   - **DATEKEY convention:** SF1 rows are point-in-time: `datekey` is the SEC
     filing (first-availability) date and `calendardate` the normalized fiscal
     period end. The monthly S&P 500 panel merges fundamentals point-in-time on
     `datekey` (a filing is usable only from its `datekey` forward, via
     `merge_asof`). The quarterly full-universe panel (R18) snaps each
     `calendardate` to its calendar quarter and keeps the **last filing per
     ticker-quarter**. Quarterly returns are computed from the split-adjusted
     SF1 `price` field between consecutive calendar quarters, plus a dividend
     yield approximated from TTM `dps`/4.
   - The SEP price table (daily/monthly prices) was **not** part of the
     entitlement used for this paper; this is why the full-universe
     survivorship-free panel is quarterly (see manuscript §3 and the E2 gate in
     `results/revision/R25_post_review.txt`).
   - Set the API key via environment variable: `export NASDAQ_DATA_LINK_API_KEY=...`
     (the code never contains a key).
2. **Fama–French factors and portfolios** — Ken French Data Library (free):
   https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
   (5 factors 2x3 monthly + daily, momentum, 25 size/BM portfolios).
3. **q-factor model returns** — global-q.org (free):
   https://global-q.org/factors.html (q5 factors, monthly).
4. **AQR factor data** (QMJ et al.) — AQR Data Library (free):
   https://www.aqr.com/Insights/Datasets

## Reproduction paths

- **Aggregate-only (no subscription):** every time-series regression, portfolio
  L/S statistic, and figure in the paper can be recomputed directly from
  `derived-data/` (see column headers in each CSV).
- **Full rebuild:** follow `code/README.md` (requires the Sharadar
  subscription; total build time is a few hours, dominated by the SF1 download
  and rolling-residual idiosyncratic-volatility estimation).

## License note on derived data

Derived aggregates are released under CC-BY-4.0; code under MIT. The raw
Sharadar tables remain subject to Nasdaq Data Link's license and are not
included in any form.
