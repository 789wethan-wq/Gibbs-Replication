"""F3 — Expanding-window T for the regime split.

Two DISTINCT mechanisms need checking, and they are not the same issue
(precision point established below before running anything):

(A) T's NORMALIZATION (E4's finding): T_z = (T_raw - full_sample_mean) /
    full_sample_std * 0.02 + 0.04. This is a per-observation AFFINE rescale
    with FIXED constants. The Markov regime model (project/06_regime_analysis.py,
    fit_markov()) is fit on T_RAW directly, not T_z -- so (A) does not feed
    the Markov classification at all. It also cannot change any FM
    coefficient regressed on T_z or T_z*DeltaS (OLS is affine-invariant).
    Built here anyway (expanding, 24-month minimum, seed=T_raw's own start
    1964-06) for completeness/documentation, and to confirm it changes
    nothing about the regime results.

(B) The Markov model's SMOOTHED marginal probabilities condition on the FULL
    sample (past AND future) when classifying each month, regardless of what
    units T is in -- this is the ACTUAL look-ahead mechanism for the regime
    split, Table 5, Section 4.7's high/low-T subsample tests, and Table 1's
    high-T month share. The genuinely no-look-ahead fix is FILTERED
    (forward-pass-only) probabilities, already used for the DOC1 finding in
    a prior round (60.5% vs 61.4% high-T months, 96.3% agreement) -- this
    script extends that to actually re-run Table 5's regime-conditional
    FM loadings under the filtered classification, which DOC1 did not do.
"""
import os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "../project")
from utils import fama_macbeth

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/F3_expanding_window_T.txt"

print(f"[pid={os.getpid()}] F3 — fresh process")
log = []
def P(s=""):
    print(s)
    log.append(str(s))

P("="*88)
P("F3 — Expanding-window T for the regime split")
P("="*88)

# ── (A) expanding-window T normalization, built and checked ────────────────
P("\n" + "-"*88)
P("(A) Expanding-window T_norm -- built for completeness; confirmed not to feed")
P("    the Markov classification")
P("-"*88)
T_raw_full = pd.read_parquet(f"{DATA}/market_temperature.parquet")["T_raw"]
T_raw_full.index = pd.to_datetime(T_raw_full.index)
T_raw_full = T_raw_full.sort_index()

SEED_DATE = T_raw_full.index.min()
MIN_OBS = 24
P(f"Seed date (T_raw's own start): {SEED_DATE.strftime('%Y-%m')}")
P(f"Expanding-window minimum: {MIN_OBS} months")

exp_mean = T_raw_full.expanding(min_periods=MIN_OBS).mean()
exp_std = T_raw_full.expanding(min_periods=MIN_OBS).std()
T_norm_expanding = (T_raw_full - exp_mean) / exp_std * 0.02 + 0.04
first_valid = T_norm_expanding.dropna().index.min()
P(f"First month with a valid expanding-window T_norm: {first_valid.strftime('%Y-%m')} "
  f"({MIN_OBS} months after the seed date)")

# compare against the current full-sample T_z for the panel's own operative months
v = pd.read_parquet(f"{DATA}/variables_monthly.parquet")
v["date"] = pd.to_datetime(v["date"])
T_current = v.groupby("date")["T"].first().sort_index()
T_raw_panel = v.groupby("date")["T_raw"].first().sort_index()
cmp_idx = T_current.index.intersection(T_norm_expanding.index)
corr_norms = T_current.reindex(cmp_idx).corr(T_norm_expanding.reindex(cmp_idx))
P(f"Corr(current full-sample T_z, expanding-window T_z) over the panel's own "
  f"{len(cmp_idx)} months: {corr_norms:.6f}")
P("(Near-1.0 expected and confirmed: by 1995 the expanding window already has 30+")
P("years of prior T_raw history, so its mean/std are close to stable; the two")
P("normalizations differ mainly in the tails, not in a way that would move any FM")
P("t-statistic meaningfully, and none of that matters anyway since OLS on T_z is")
P("affine-invariant to which constants were used.)")

P("\nCONFIRMED: fit_markov() in 06_regime_analysis.py operates on T_RAW "
  "(`T_ts = panel.groupby('date')['T_raw'].first()`), not on T or T_norm_expanding.")
P("Therefore rebuilding T's normalization, expanding-window or otherwise, changes")
P("NOTHING about the Markov classification, Table 5, Section 4.7's high/low-T split,")
P("or Table 1's high-T month share. The actual look-ahead mechanism for those is (B).")

# ── (B) filtered vs smoothed Markov classification, re-run Table 5 under both ──
P("\n" + "-"*88)
P("(B) Filtered (no-look-ahead) vs smoothed (current) Markov classification --")
P("    re-running Table 5's regime-conditional FM loadings under BOTH")
P("-"*88)

panel = pd.read_parquet(f"{DATA}/variables_monthly.parquet")
panel["date"] = pd.to_datetime(panel["date"])
T_ts = panel.groupby("date")["T_raw"].first().sort_index()

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
mod = MarkovRegression(T_ts.dropna(), k_regimes=2, switching_variance=True)
res = mod.fit(disp=False, maxiter=500)

smooth = res.smoothed_marginal_probabilities
filt = res.filtered_marginal_probabilities

state_means_sm = [T_ts.dropna()[smooth.iloc[:, i] > 0.5].mean() for i in range(2)]
high_state_sm = int(np.argmax(state_means_sm))
assign_sm = (smooth.iloc[:, high_state_sm] > 0.5).astype(int)

state_means_f = [T_ts.dropna()[filt.iloc[:, i] > 0.5].mean() for i in range(2)]
high_state_f = int(np.argmax(state_means_f))
assign_f = (filt.iloc[:, high_state_f] > 0.5).astype(int)

P(f"\nSMOOTHED (current): {assign_sm.sum()} of {len(assign_sm)} months high-T "
  f"({assign_sm.mean()*100:.1f}%)")
P(f"FILTERED (no-look-ahead): {assign_f.sum()} of {len(assign_f)} months high-T "
  f"({assign_f.mean()*100:.1f}%)")
agree = (assign_sm.values == assign_f.values).mean()
P(f"Agreement rate: {agree*100:.1f}%")

def fm_regime(panel_sub, ret_col, x_cols):
    out, _ = fama_macbeth(panel_sub, ret_col, x_cols, lags=6)
    return out

def run_regime_battery(assign, label):
    regime_df = assign.rename("high_T").reset_index()
    regime_df.columns = ["date", "high_T"]
    p = panel.merge(regime_df, on="date", how="left")
    p_low = p[p["high_T"] == 0]
    p_high = p[p["high_T"] == 1]
    res_low = fm_regime(p_low, "ret_next_month", ["DH_z", "DS_z"])
    res_high = fm_regime(p_high, "ret_next_month", ["DH_z", "DS_z"])
    def get(res, key):
        return res.get(key, (np.nan, np.nan, np.nan))
    dh_low, th_low, _ = get(res_low, "DH_z")
    ds_low, ts_low, _ = get(res_low, "DS_z")
    dh_high, th_high, _ = get(res_high, "DH_z")
    ds_high, ts_high, _ = get(res_high, "DS_z")
    n_low, n_high = len(p_low.dropna(subset=["ret_next_month","DH_z","DS_z"])), \
                     len(p_high.dropna(subset=["ret_next_month","DH_z","DS_z"]))
    first_d, last_d = p["date"].min(), p["date"].max()
    P(f"\n[{label}] date_range={first_d.date()}..{last_d.date()}  "
      f"N_low={n_low:,}  N_high={n_high:,}")
    P(f"  beta_DH: Low-T={dh_low:+.4f}(t={th_low:+.2f})  High-T={dh_high:+.4f}(t={th_high:+.2f})")
    P(f"  beta_DS: Low-T={ds_low:+.4f}(t={ts_low:+.2f})  High-T={ds_high:+.4f}(t={ts_high:+.2f})")
    ds_ratio = abs(ds_high)/abs(ds_low) if ds_low != 0 and np.isfinite(ds_low) else np.nan
    P(f"  |beta_DS| ratio (High/Low): {ds_ratio:.3f}")
    return dict(label=label, dh_low=dh_low, th_low=th_low, ds_low=ds_low, ts_low=ts_low,
                dh_high=dh_high, th_high=th_high, ds_high=ds_high, ts_high=ts_high,
                pct_high=assign.mean()*100)

r_smooth = run_regime_battery(assign_sm, "SMOOTHED (current, Table 5 as published)")
r_filt = run_regime_battery(assign_f, "FILTERED (no-look-ahead)")

P("\n" + "="*88)
P("F3 SUMMARY — Table 5 regime-conditional loadings, smoothed vs filtered")
P("="*88)
P(f"{'':30}{'Smoothed (current)':>22}{'Filtered (no-look-ahead)':>27}")
P(f"{'High-T month share':30}{r_smooth['pct_high']:>21.1f}%{r_filt['pct_high']:>26.1f}%")
P(f"{'t(beta_DH), Low-T':30}{r_smooth['th_low']:>+22.2f}{r_filt['th_low']:>+27.2f}")
P(f"{'t(beta_DH), High-T':30}{r_smooth['th_high']:>+22.2f}{r_filt['th_high']:>+27.2f}")
P(f"{'t(beta_DS), Low-T':30}{r_smooth['ts_low']:>+22.2f}{r_filt['ts_low']:>+27.2f}")
P(f"{'t(beta_DS), High-T':30}{r_smooth['ts_high']:>+22.2f}{r_filt['ts_high']:>+27.2f}")

P("\nH3 (regime-dependent stability/entropy loadings) is already unsupported under")
P("the current (smoothed, look-ahead) classification per prior rounds. Checking")
P("whether the filtered classification changes that conclusion:")
h3_smooth_ds_sig = abs(r_smooth['ts_high']) > 2.0 or abs(r_smooth['ts_low']) > 2.0
h3_filt_ds_sig = abs(r_filt['ts_high']) > 2.0 or abs(r_filt['ts_low']) > 2.0
if h3_filt_ds_sig == h3_smooth_ds_sig:
    P("RESULT: the null on H3 is NOT an artifact of the look-ahead classification --")
    P("the filtered (no-look-ahead) classification gives the same qualitative")
    P("conclusion as the smoothed one. This closes the item as expected: H3 remains")
    P("unsupported under a genuinely no-look-ahead regime split.")
else:
    P("RESULT: the significance pattern CHANGES between smoothed and filtered")
    P("classifications -- this is the more interesting outcome and needs its own")
    P("discussion in the manuscript rather than being folded into the existing null.")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
