"""REV4_E1_left_truncation_ladder.py — decisive experiment from the fourth
external review (score 4/10): separates BIRTH-COHORT (left-truncation) from
SURVIVAL in the R18 reliability ladder's k=27 rung (t(IVOL)=+3.23).

The k=27 rung (D1a_2x2_cell.py "FB_surv", SPEC_T6_reliability_ladder.py k=27)
requires a >=108-consecutive-quarter run within the panel's 114-quarter span
(1995Q3q_ord=102 .. 2023Q4 q_ord=215) -- since only a 6-quarter window of run
starts can satisfy that, it is effectively "listed within the panel's first
~2 years AND present almost continuously through 2023Q3/Q4," i.e. survival
conditioning entangled with birth-cohort (old, large-at-listing firms).

This script builds the COUNTERFACTUAL cohort: firms first observed within the
panel's first 4 quarters (q_ord <= 105, i.e. listed/first-SF1-filed by
~1996Q3), with NO requirement that they survive to the end -- deaths allowed,
using every quarter they actually have (same as the unconditional FB_noSurv
panel, just restricted to this early-listed cohort). N=312 tickers, closest
comparable size to the k=27 rung's 336. A robustness alternate widens the
window to the first 8 quarters (N=500).

If t(IVOL) in this no-survival-required early cohort already accounts for
most of the 0.02->3.23 movement, the ladder's "survival resurrects the
premium" claim is actually a birth-cohort artifact, not a survivorship
effect. If it does not, survival is doing real, separable work.

Same FM Model B (dH_gpm + dS, quarterly, NW-4, min-cross-section 20) spec as
D1a_2x2_cell.py / SPEC_T6_reliability_ladder.py -- exact convention match.

Outputs: results/revision/REV4_E1_left_truncation_ladder.txt
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision"
os.makedirs(OUT, exist_ok=True)

LOG = []
def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    LOG.append(line)


def cs_winsorize_zscore(df, col, date_col="q", pct=0.01):
    def _wz(x):
        x = x.dropna()
        if len(x) < 5:
            return pd.Series(np.nan, index=x.index)
        lo, hi = x.quantile(pct), x.quantile(1 - pct)
        xc = x.clip(lo, hi)
        std = xc.std()
        if std < 1e-10:
            return pd.Series(np.nan, index=x.index)
        return (xc - xc.mean()) / std
    return df.groupby(date_col)[col].transform(_wz)


def fama_macbeth_nw(panel, y_col, x_cols, date_col="q", lags=4, min_cs=20):
    coefs = []
    for d, grp in panel.groupby(date_col):
        sub = grp[[y_col] + x_cols].dropna()
        if len(sub) < max(min_cs, len(x_cols) + 2):
            continue
        X = sm.add_constant(sub[x_cols], has_constant="add")
        res = sm.OLS(sub[y_col], X).fit()
        coefs.append(res.params[x_cols].rename(d))
    if not coefs:
        return {}, pd.DataFrame()
    cdf = pd.DataFrame(coefs)
    out = {}
    for col in x_cols:
        s = cdf[col].dropna()
        n = len(s)
        mean_ = s.mean()
        gamma0 = (s ** 2).mean() - mean_ ** 2
        var = gamma0
        for l in range(1, min(lags + 1, n)):
            g = ((s.iloc[l:].values - mean_) * (s.iloc[:-l].values - mean_)).mean()
            var += 2 * (1 - l / (lags + 1)) * g
        se = np.sqrt(max(var, 1e-30) / n)
        out[col] = dict(coef=mean_, se=se, t=mean_ / se, n=n)
    return out, cdf


say("=" * 96)
say("REV4 E1 — LEFT-TRUNCATION-ONLY COHORT vs. THE k=27 SURVIVAL LADDER RUNG")
say("=" * 96)

panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
say(f"\nFull panel: N={len(panel):,}  tickers={panel['ticker'].nunique():,}  "
    f"q_ord range=[{panel['q_ord'].min()},{panel['q_ord'].max()}]  "
    f"({panel['q'].min()}..{panel['q'].max()}, {panel['q'].nunique()} quarters)")

first_q = panel.groupby("ticker")["q_ord"].min()
last_q = panel.groupby("ticker")["q_ord"].max()
qmin, qmax = panel["q_ord"].min(), panel["q_ord"].max()

say(f"\nFirst-quarter-in-panel distribution: min={first_q.min()} max={first_q.max()}")
say(f"Tickers with first_q == qmin (panel's very first quarter): {(first_q==qmin).sum():,}")
say(f"Tickers surviving to qmax (last quarter): {(last_q==qmax).sum():,}")

# ── reference: the k=27 survival ladder rung (reproduced, same spec as D1a) ──
say("\n" + "-" * 96)
say("REFERENCE: k=27 survival ladder rung (run-length >= 108 consecutive quarters)")
say("-" * 96)
panel_sorted = panel.sort_values(["ticker", "q_ord"]).reset_index(drop=True)
new_run = (panel_sorted["q_ord"] - panel_sorted.groupby("ticker")["q_ord"].shift(1)) != 1
panel_sorted["run_id"] = new_run.groupby(panel_sorted["ticker"]).cumsum()
panel_sorted["run_len_q"] = panel_sorted.groupby(["ticker", "run_id"])["q_ord"].transform("size")
surv27 = panel_sorted[panel_sorted["run_len_q"] >= 108].copy()
surv27["ds_z_r"] = cs_winsorize_zscore(surv27, "delta_s")
surv27["dh_z_r"] = cs_winsorize_zscore(surv27, "dH_gpm")
pf_surv = surv27.dropna(subset=["ret_next", "ds_z_r", "dh_z_r"])
fm_surv, _ = fama_macbeth_nw(pf_surv, "ret_next", ["dh_z_r", "ds_z_r"])
r_surv = fm_surv.get("ds_z_r", dict(coef=np.nan, se=np.nan, t=np.nan, n=0))
say(f"  N={len(pf_surv):,}  tickers={pf_surv['ticker'].nunique():,}  Tq={r_surv['n']}  "
    f"t(IVOL)={r_surv['t']:+.4f}  beta={r_surv['coef']:+.6f}  "
    f"(manuscript/D1a reference: +3.23, 336 tickers)")

# ── unconditional reference (FB_noSurv) ──────────────────────────────────────
say("\n" + "-" * 96)
say("REFERENCE: unconditional full panel (no cohort/survival restriction)")
say("-" * 96)
panel["ds_z_r"] = cs_winsorize_zscore(panel, "delta_s")
panel["dh_z_r"] = cs_winsorize_zscore(panel, "dH_gpm")
pf_uncond = panel.dropna(subset=["ret_next", "ds_z_r", "dh_z_r"])
fm_u, _ = fama_macbeth_nw(pf_uncond, "ret_next", ["dh_z_r", "ds_z_r"])
r_u = fm_u.get("ds_z_r", dict(coef=np.nan, se=np.nan, t=np.nan, n=0))
say(f"  N={len(pf_uncond):,}  tickers={pf_uncond['ticker'].nunique():,}  Tq={r_u['n']}  "
    f"t(IVOL)={r_u['t']:+.4f}  beta={r_u['coef']:+.6f}  (manuscript reference: +0.02)")

# ── EXPERIMENT: left-truncation-only cohorts, no survival requirement ────────
say("\n" + "=" * 96)
say("EXPERIMENT: left-truncation-only cohorts (early-listed, deaths allowed)")
say("=" * 96)

results = []
for window_label, w in [("first 4 quarters (q_ord<=qmin+3)", 3),
                          ("first 8 quarters (q_ord<=qmin+7)", 7)]:
    cohort_tickers = first_q[first_q <= qmin + w].index
    say(f"\nCohort: {window_label}")
    say(f"  N tickers in cohort: {len(cohort_tickers):,}")
    n_survive_to_end = (last_q.loc[cohort_tickers] == qmax).sum()
    say(f"  Of these, {n_survive_to_end:,} ({n_survive_to_end/len(cohort_tickers):.1%}) "
        f"also happen to survive to the panel's last quarter (NOT conditioned on here)")

    cell = panel[panel["ticker"].isin(cohort_tickers)].copy()
    # re-standardize within the restricted cohort per quarter, same convention as SPEC_T6
    cell["ds_z_c"] = cs_winsorize_zscore(cell, "delta_s")
    cell["dh_z_c"] = cs_winsorize_zscore(cell, "dH_gpm")
    pf = cell.dropna(subset=["ret_next", "ds_z_c", "dh_z_c"])
    fm, _ = fama_macbeth_nw(pf, "ret_next", ["dh_z_c", "ds_z_c"])
    r = fm.get("ds_z_c", dict(coef=np.nan, se=np.nan, t=np.nan, n=0))
    say(f"  Model B FM (dH + dS), all quarters this cohort actually has (deaths allowed):")
    say(f"    N={len(pf):,}  tickers={pf['ticker'].nunique():,}  Tq={r['n']}  "
        f"t(IVOL)={r['t']:+.4f}  beta(IVOL)={r['coef']:+.6f}  SE={r['se']:.6f}")
    results.append(dict(window=window_label, n_cohort=len(cohort_tickers),
                         n_survive_to_end=int(n_survive_to_end),
                         n_obs=len(pf), n_tickers_used=pf['ticker'].nunique(),
                         Tq=r['n'], t_ivol=r['t'], beta_ivol=r['coef'], se_ivol=r['se']))

say("\n" + "=" * 96)
say("DECOMPOSITION SUMMARY")
say("=" * 96)
say(f"  Unconditional (full panel):                 t(IVOL) = {r_u['t']:+.4f}   (N tickers={pf_uncond['ticker'].nunique():,})")
for res in results:
    frac = np.nan
    if (r_surv['t'] - r_u['t']) != 0:
        frac = (res['t_ivol'] - r_u['t']) / (r_surv['t'] - r_u['t'])
    say(f"  Left-truncation only, {res['window']:32}: t(IVOL) = {res['t_ivol']:+.4f}   "
        f"(N tickers={res['n_tickers_used']:,})  "
        f"-> {frac*100:+.0f}% of the +0.02->+3.23 ladder movement" if np.isfinite(frac) else "")
say(f"  k=27 survival ladder rung (108q run, deaths NOT allowed):  t(IVOL) = {r_surv['t']:+.4f}   "
    f"(N tickers={pf_surv['ticker'].nunique():,})")

say("\nInterpretation: if the left-truncation-only t(IVOL) values above already")
say("recover most of the movement from +0.02 to +3.23 WITHOUT requiring survival to")
say("the end of the sample, the ladder's resurrection is at least partly a birth-cohort")
say("(old/large-at-listing) artifact rather than a survivorship effect specifically.")
say("If the left-truncation-only cohorts stay close to the unconditional +0.02 and the")
say("full jump to +3.23 requires the additional 108-quarter run-length (survival)")
say("restriction, survival conditioning is doing the separable work the paper claims.")

with open(f"{OUT}/REV4_E1_left_truncation_ladder.txt", "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\n[written] {OUT}/REV4_E1_left_truncation_ladder.txt")
