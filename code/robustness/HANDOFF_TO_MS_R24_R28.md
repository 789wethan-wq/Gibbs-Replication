# Handoff to MS/chat instance — R24–R28 Round 3 results (v2, post-adjudication)

Executor: [CODE]. All five runs plus the four documentation queries are
complete. **This version incorporates four corrections requested after
review of the first pass** — see the "ADJUDICATIONS" section immediately
below before reading anything else; it changes how R25 and R28 should be
written up, and resolves (not sidesteps) the min-cross-section-floor
question.

Full run logs (all under `results/revision/`):
- `R24_coldproc_verification.txt`
- `R25_cross_panel_diff.txt`
- `R26_reliability_stratified.txt`
- `R27_financial_exclusion.txt`
- `R28_size_survival_2x2.txt`
- `DOC1_markov_filtered_vs_smoothed.txt`, `DOC2_delisting_reason_split.txt`,
  `DOC3_yfinance_stooq_crosscheck.txt`, `DOC4_oos_scope.txt`

---

## ADJUDICATIONS (read first)

### 1. Floor=4 audit — the manuscript's headline numbers are NOT affected

Audited every script in `robustness/` and `project/` that runs a
Fama-MacBeth regression on the full-universe quarterly panel
(`merged_sf1_quarterly_survfree.parquet`) — 30+ scripts, listed below.
**Every pre-existing script uses `min_cs=20`** (either as the function's
own default or an explicit argument). The `min_cs=4` floor never appears in
any pre-existing script for the full-universe panel; it was an error I
introduced only in the FIRST DRAFTS of two NEW scripts written this session
(`R25_cross_panel_diff.py`, `R27_financial_exclusion.py`), caught before
finalizing, and both are now fixed to `min_cs=20`.

**Direct confirmation of the headline source:** `results/survivorship_free/
R18_sf1_quarterly_results.txt`, generated **2026-07-07 08:46** (17 days
before this session started) by `robustness/R18_sf1_quarterly_survfree.py`
line 248 — `fama_macbeth_nw(pf, "ret_next", ["delta_h_z","delta_s_z"])`,
called with no override, so it runs under the function's own default
`min_cs=20` (line 78 of that same file) — reports `β_ΔS = +0.00009, t=+0.02`
verbatim. This is an exact match to the manuscript's headline and it was
never computed under floor=4. **No re-verification of any existing
manuscript number is required; only my own two new scripts needed the fix
that was already applied before I reported R25/R27's final numbers.**

Scripts confirmed on `min_cs=20` (canonical, full-universe quarterly FM):
`R18_sf1_quarterly_survfree.py` (the panel-construction/headline source),
`R22_v19_battery.py`, `A3_clustered.py`, `A_runs_plos.py`,
`D2_corrected_split_half.py`, `D2_size_decile_reliability.py`,
`D4_lagged_cap_rerun.py`, `DIAG_channels.py`, `DIAG_channel_verification.py`,
`DIAG_survivorship.py`, `M2_size_orthogonalized.py`, `M9_oz_entropy_ladder.py`,
`M9b_oz_entropy_fixedgrid.py`, `M9_M2_entropy_size_orthogonalized.py`,
`OOS_R18_provenance.py`, `R20_section44_hac.py`, `R21_dh_degeneracy_audit.py`,
`R22_horizon_normalization.py`, `R23_v21_battery.py`,
`R23_r19_delta_calibration.py`, `R25_post_review_experiments.py`,
`T10_RECON.py`, `VERIFY_presub.py`, `D1_placebo_characteristic_test.py`
(explicit `min_cs=20` on its quarterly call). `R21_revision_battery.py` uses
an even stricter `min_cs=200` for an unrelated annual-grouping robustness
check — not a laxer floor, and not the FU-quarterly headline spec.

### 2. R25's t-ordering (FM-consistent |t|=5.19 > pooled-OLS |t|=4.11) — explained, not asserted

Computed `corr(β_SP,t, β_FU,t)` across the 111 common quarters, using the
already-existing per-quarter coefficient series from R25(1)/R25(4):
**corr = +0.42**. Var(β_SP)=0.001385, Var(β_FU)=0.002176,
Var(β_FU − β_SP)=0.002087 — i.e. differencing the two paired series removes
their *shared* quarter-to-quarter variation, leaving Var(diff) at **58.6%**
of what it would be if the two series were independent (Var(SP)+Var(FU) =
0.003562). The FM-consistent test is a matched-pairs design (same quarter,
two panels) and benefits from exactly this variance cancellation — the
classic reason a paired test out-powers an unpaired one when the paired
quantities are positively correlated. The pooled-OLS stacked regression does
not exploit this pairing structure (it estimates from individual firm-quarter
observations with two-way-clustered SEs, not from pre-averaged,
quarter-matched coefficient series), so it doesn't get the same variance
reduction. Both are legitimate; the ordering is explained by pairing power,
not an error in either.

### 3. R28 does NOT close the survivorship-vs-index-membership label question — retracted overclaim

Retracting the R28 write-up's earlier claim that the result "directly informs"
the title decision. R28 varies an *imposed continuous-listing requirement*
(k=27y) against a *size cut*, both within the corrected panel. That cleanly
establishes survival-conditioning as the dominant channel relative to size —
genuine and useful. But Major 6's objection is about the **comparison arm**:
462 *current* S&P constituents is endpoint index membership (selection on
size, liquidity, profitability, and committee judgment), which is a different
and additional selection mechanism beyond "did this firm keep trading for 27
years." R28 never touches the comparison panel or simulates committee-level
index-inclusion criteria, so it cannot decompose survivorship from
index-membership selection. **The §4.8 concession stands as written, and the
title decision should rest on that concession, not on R28.** R28's genuine
contribution is narrower and still worth stating: within the corrected panel,
survivorship dominates size as an explanation for where the premium appears.

### 4. Filtered Markov probabilities — report as primary, not a footnote to smoothed

Reversing the framing: report **60.5%** (filtered, look-ahead-free) as the
number that belongs in Table 1/Table 5, with 61.4% (smoothed, the
originally-reported figure) noted as what full-sample conditioning gives.
96.3% month-classification agreement (13/347 months differ). H3 is
unsupported under either classification, so no downstream conclusion moves —
but the look-ahead was real and unstated, and smoothed-as-primary is the
wrong default in a paper this scrupulous elsewhere.

**Also re-verified fresh today (not relying on the 2026-07-11 check):** SEP
entitlement — live query, `SHARADAR/SEP` for AAPL returns 0 rows; `SHARADAR/
SF1` control returns 128 rows (key is working, SEP specifically is not
entitled). Confirms doc query 3's disposition is not stale.

---

## R24 — Code-path integrity check: CLEARED

Re-estimated both arms (SP500, full-universe) x both specs (baseline,
date-FE) from cold in **separate OS processes** (confirmed distinct PIDs),
each with its own independently-defined estimator function (not imported
from a shared module). Both arms reproduce the manuscript's reported
t-statistics exactly:
- baseline: t=+2.4881 (SP) / t=+2.4921 (FU), design shapes (118014,4) vs
  (392557,4), distinct SHA-1 hashes, corr(resid) on 38,917 alignable
  (ticker,quarter) observations = **-0.035** (essentially zero).
- date-FE: t=+2.4566 (SP) / t=+2.5375 (FU), same pattern.

**Verdict: the coincidence is real, not a code-path artifact.**

**Footnote text (drafted, ready to place):**
> The S&P 500 and full-universe arms of this test were re-estimated from cold
> in separate Python processes with independently constructed design
> matrices (shapes 118,014×4 and 392,557×4; distinct SHA-1 hashes) to rule
> out a shared cache as the source of the close agreement between t=+2.4881
> and t=+2.4921 (and, for the date-FE specification, t=+2.46 and t=+2.54).
> Both values reproduced exactly under this cold re-estimation; residual
> correlation on the 38,917 alignable (ticker, quarter) observations common
> to both panels was -0.035, indicating the agreement is a numerical
> coincidence rather than a code-path artifact.

---

## R25 — Formal cross-panel coefficient difference test: SIGNIFICANT

Annualized (linear, ×12/×4; both regressors are within-period z-scored so
this is "return spread per 1 SD of ΔS per year"): SP500 +6.22%/yr, FU
+0.035%/yr (both computed under the corrected min_cs=20 floor — see
Adjudication 1).

**Two independent estimator families, both significant, both robust to
block bootstrap** (see Adjudication 2 for why their magnitudes differ):
- Pooled-OLS stacked panel (`ret ~ ΔS + D_FU + D_FU×ΔS + ΔH`, two-way
  firm×quarter clustered, harmonized to quarterly for both panels):
  D_FU×ΔS coef=-0.01731, SE=0.00421, **t=-4.11, p<0.0001**,
  95% CI=(-0.0256, -0.0091). Block bootstrap (500 reps, resampling whole
  quarters): 95% CI=(-0.0259, -0.0080).
- FM-consistent (same estimator family as the manuscript's own "+4.70→+0.02"
  headline: per-quarter cross-sectional coefficient series, NW-averaged):
  mean(β_FU - β_SP)=-0.01490/quarter, **t=-5.19, p<0.0001**,
  95% CI=(-0.0205, -0.0093). corr(β_SP,t, β_FU,t) across quarters = +0.42.

Harmonizing SP500 to quarterly costs relatively little: FM t(ΔS) goes
+4.70 (monthly, T=335) → +4.39 (quarterly, T=111).

**Verdict: Major Weakness 1 dissolves.** Suggested language:
> The cross-panel difference in the entropy premium is statistically
> significant: stacking both panels at a harmonized quarterly frequency and
> testing β_FU − β_SP via a two-way-clustered interaction gives t=-4.11
> (95% CI: -0.0256 to -0.0091; block-bootstrap CI: -0.0259 to -0.0080); the
> same test run in the manuscript's own Fama-MacBeth estimator family — which
> exploits the quarter-by-quarter pairing of the two panels' coefficient
> estimates (corr=+0.42 across quarters) and is correspondingly more
> powerful — gives t=-5.19 (95% CI: -0.0205 to -0.0093). Both confirm the
> "+4.70 → +0.02" contrast reflects a real, not merely descriptive,
> difference.

---

## R26 — Reliability-stratified estimation: COLLAPSE STANDS (bounded attenuation)

Per-firm reliability is unusably noisy (SB-corrected range -408 to +0.94,
mean -0.74; individual firms don't have enough independent observations).
Pivoted to the D2-validated group-pooled design, selecting groups by
**measured** reliability (reported per group), not by assumed size rank:

- Highest-reliability tercile (measured reliability=0.530, cap≥$1.38B):
  **t(ΔS)=+0.33**, coef=+0.0014, N=126,331, avg 1,138 firms/qtr (T=111
  quarters), L/S=+1.4%/yr (t=0.36, T=112 quarters).
- Highest-reliability decile (measured reliability=0.533, cap≥$10.0B):
  **t(ΔS)=+1.06**, coef=+0.0045, N=36,170, avg 326 firms/qtr (T=111
  quarters), L/S=+4.1%/yr (t=1.05, T=112 quarters).
- EIV correction (full unconditioned panel, N=392,557, overall pooled
  reliability=0.4835): raw coef=+0.000087, SE=0.004818, t=+0.018 →
  EIV-adjusted coef=+0.000181, SE=0.009965; **t unchanged (+0.018)** by
  construction under simple scalar attenuation-correction.

**Verdict: premium stays statistically indistinguishable from zero even at
reliability≈0.53 with delisted firms retained** — report both the tercile
(t=0.33) and decile (t=1.06) numbers rather than only the flatter one; the
decile figure does rise noticeably (0.02→1.06) even though it stays far from
significant, and that non-monotonicity is worth stating plainly rather than
smoothing over (same instinct as the note on R28's non-monotonic size effect
within the survival-conditioned stratum).

---

## R27 — Financial-firm exclusion: FINDINGS ROBUST

Financial-firm share (SIC 6000-6999) of matched tickers: SP500 19.3%
(88/457 firms), full-universe 22.2% (2,374/10,699 firms) — both below the
"24.7%" figure quoted in the spec; worth reconciling the denominator if that
number is cited verbatim elsewhere (likely a raw-universe vs.
regression-matched-sample difference).

|            | SP500 t(ΔH) | SP500 t(ΔS) | FU t(ΔH) | FU t(ΔS) |
|---|---|---|---|---|
| With financials (N=118,014 / 392,557) | +2.606 | +4.695 | +3.461 | +0.018 |
| Ex financials (N=94,544 / 299,631)    | +2.776 | +4.723 | +3.660 | -0.254 |

Both headline findings hold after excluding financials — SP t(ΔS)
essentially unchanged; FU t(ΔH) strengthens (+3.46→+3.66); FU t(ΔS) stays
insignificant, sign-flipping from +0.02 to -0.25 but nowhere near
significance either way.

---

## R28 — Size × survival 2×2: supports survivorship-over-size WITHIN the corrected panel (see Adjudication 3 for scope limits)

|                        | No survival requirement | Survival-conditioned (k=27y) |
|---|---|---|
| **Full breadth**       | t(ΔS)=+0.018, t(ΔH)=+3.458, L/S=-1.02%/yr, N=392,557, avg 3,505/qtr (T=111), medCap=$401M | t(ΔS)=+3.229, t(ΔH)=-0.044, L/S=+10.97%/yr, N=36,402, avg 328/qtr (T=111), medCap=$4,082M |
| **Large-cap (top-500, lagged cap)** | t(ΔS)=+1.027, t(ΔH)=+1.897, L/S=+3.78%/yr, N=52,310, avg 467/qtr (T=111), medCap=$14,349M | t(ΔS)=+2.316, t(ΔH)=+2.547, L/S=+6.26%/yr, N=16,961, avg 153/qtr (T=111), medCap=$17,417M |

(Two cells — full-breadth × {no-survival, k=27} — reproduce
`R25_post_review_experiments.py` E1's k=0/k=27 rows exactly, a built-in
consistency check. An early draft had a bug where the top-500 rank was
recomputed after subsetting to the small k=27 universe, making the cut
vacuous — fixed by computing the top-500 flag once against the full-universe
cross-section, then intersecting with the survival filter; final cell (2,2)
has 16,961 obs, distinct from cell (1,2)'s 36,402.)

**Decomposition (t(ΔS) only):**
- Survival effect, within full-breadth: **+3.211** (+0.02→+3.23)
- Survival effect, within large-cap: **+1.289** (+1.03→+2.32)
- Size effect, within no-survival-requirement: **+1.009** (+0.02→+1.03)
- Size effect, within survival-conditioned: **-0.912** (+3.23→+2.32 — further
  restricting an already-survived sample to large-cap firms *weakens* the
  signal)

**Verdict, narrowly scoped:** within the corrected panel, survival
conditioning moves t(ΔS) sharply in both size strata (+3.21, +1.29); size
moves it comparatively little, and with inconsistent sign, in both survival
strata (+1.01, -0.91). This is a clean survivorship-over-size result for
Majors 1/3. **It does not bear on Major 6 or the title decision** (see
Adjudication 3) — that rests on the §4.8 concession about index-membership
selection in the comparison arm, which R28 does not test.

---

## Documentation queries

1. **Table 5 Markov regime — report filtered as primary (see Adjudication
   4).** Filtered (look-ahead-free): 60.5% high-T months (210/347). Smoothed
   (originally reported, conditions on full sample): 61.4% (213/347). 96.3%
   month agreement, 13/347 differ. H3 unsupported either way.

2. **Delisting reason split.** SF1 has no delisting-reason code. Two
   imperfect proxies on all 8,937 analysis-panel delistings:
   (a) `relatedtickers` populated (successor-symbol link): 51.4% (4,593/8,937).
   (b) terminal-quarter return: 11.0% end at ≤-50% (failure-like, 980/8,937),
   72.0% end at >-10% (flat/positive, M&A/reorg-like, 6,434/8,937), 17.0%
   ambiguous (1,523/8,937). The two proxies disagree on a meaningful fraction
   (cross-tab in the log: 726 tickers are relatedtickers=Y AND
   failure-like-return simultaneously) — report as a range/caveat, not a
   single clean percentage.

3. **yfinance cross-check — BLOCKED, reported not worked around; SEP
   re-verified as unavailable TODAY (not a stale 2026-07-11 finding).** Live
   query just now: `SHARADAR/SEP` for AAPL returns 0 rows; `SHARADAR/SF1`
   control returns 128 rows (key works, SEP specifically excluded from this
   subscription). Stooq's free CSV endpoint returns a JavaScript
   proof-of-work anti-bot challenge for every ticker tested (5-name probe,
   systematic, remaining 45 of the drawn random-50 sample not probed as
   redundant); `pandas_datareader`'s Stooq backend has been removed
   (`NotImplementedError`). No JS-challenge circumvention attempted. **No
   free independently-sourced second price feed is reachable from this
   environment; this must be disclosed as unverified, or a paid/keyed source
   obtained if the cross-check is a hard publication requirement.**

4. **OOS scope.** Confirmed: the 576-specification robustness battery (S2
   Table, §4.7) is a full-sample sensitivity grid, not a sequential/
   expanding-window search. Drafted one-sentence §4.6 addition:
   > The 576-specification robustness battery underlying this
   > multiple-testing disclosure (Section 4.7, S2 Table) was constructed and
   > evaluated using the full sample; consequently, the expanding-window
   > out-of-sample test above is out-of-sample with respect to the estimated
   > parameters at each step, but not with respect to model selection, since
   > the functional form being tested out-of-sample was chosen after seeing
   > the full-sample robustness grid.

---

## Scripts added (all under `robustness/`)

`R24_coldproc_sp500.py`, `R24_coldproc_fulluniv.py`,
`R24_coldproc_datefe_sp500.py`, `R24_coldproc_datefe_fulluniv.py`,
`R24_compare.py`, `R25_cross_panel_diff.py`, `R26_build_reliability.py`,
`R26_reliability_stratified.py`, `R27_financial_exclusion.py`,
`R28_size_survival_2x2.py`, `DOC1_markov_filtered_vs_smoothed.py`,
`DOC2_delisting_reason_split.py`, `DOC3_yfinance_stooq_crosscheck.py`.
Intermediate data: `data/R26_split_half_obs.parquet`,
`data/R26_firm_reliability.parquet` (the latter documents the failed
per-firm approach; not used downstream).
