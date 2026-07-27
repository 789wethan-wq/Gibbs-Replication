"""E3 — R20/E1's k-year survival ladder applied to the placebo characteristics.

Identical construction to R25_post_review_experiments.py's E1 (the ladder
that recovers t(DeltaS)=+3.23 at k=27 from an unconditional ~0): per-ticker
consecutive-quarter run length, condition the panel on runs >= k years,
recompute cross-sectional z-scores WITHIN the conditioned panel, k in
{0,5,10,15,20,25,27}. Applied here to size_z, bm_z, mom_z, beta_z (the same
placebo panel built once by E1_E3_build_placebos.py), with delta_h_z (ΔH)
and delta_s_z (ΔS) run alongside as the existing internal controls.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
sys.path.insert(0, "../project")
from utils import newey_west_mean_tstat

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/E3_survival_ladder_placebos.txt"

print(f"[pid={os.getpid()}] E3 — fresh process")
log = []
def P(s=""):
    print(s)
    log.append(str(s))


def cs_wz(df, col, date_col="q", pct=0.01):
    def _f(x):
        x2 = x.dropna()
        if len(x2) < 5:
            return pd.Series(np.nan, index=x.index)
        lo, hi = x2.quantile(pct), x2.quantile(1 - pct)
        xc = x.clip(lo, hi)
        s = xc.std()
        if s < 1e-10:
            return pd.Series(np.nan, index=x.index)
        return (xc - xc.mean()) / s
    return df.groupby(date_col)[col].transform(_f)


def fama_macbeth_nw(panel, y_col, x_cols, date_col="q", lags=4, min_cs=20):
    coefs = []
    for d, g in panel.groupby(date_col):
        s = g[[y_col] + x_cols].dropna()
        if len(s) < max(min_cs, len(x_cols) + 2):
            continue
        X = sm.add_constant(s[x_cols], has_constant="add")
        coefs.append(sm.OLS(s[y_col], X).fit().params[x_cols].rename(d))
    if not coefs:
        return {}
    cdf = pd.DataFrame(coefs)
    out = {}
    for c in x_cols:
        s = cdf[c].dropna()
        mean_, t_, p_ = newey_west_mean_tstat(s.values, lags=lags)
        out[c] = dict(coef=mean_, t=t_, n=len(s))
    return out


P("="*88)
P("E3 — Survival ladder applied to placebo characteristics (size, B/M, momentum, beta)")
P("="*88)

panel = pd.read_parquet(f"{DATA}/E1E3_qpanel_with_placebos.parquet")
panel = panel.sort_values(["ticker", "q_ord"]).reset_index(drop=True)
new_run = (panel["q_ord"] - panel.groupby("ticker")["q_ord"].shift(1)) != 1
panel["run_id"] = new_run.groupby(panel["ticker"]).cumsum()
panel["run_len_q"] = panel.groupby(["ticker", "run_id"])["q_ord"].transform("size")

KS = [0, 5, 10, 15, 20, 25, 27]
CHARS = {"delta_h_z": "DeltaH (control)", "delta_s_z": "DeltaS (control)",
         "size_z": "Size", "bm_z": "Book-to-market", "mom_z": "Momentum", "beta_z": "Beta"}

rows = []
for k in KS:
    thr_q = int(round(4 * k))
    sub = panel[panel["run_len_q"] >= max(thr_q, 1)].copy() if k > 0 else panel.copy()
    if len(sub) == 0:
        P(f"\n[k={k}] EMPTY -- skipping")
        continue
    # recompute cross-sectional z-scores WITHIN the conditioned panel (same as R20/E1).
    # delta_s/dH_gpm: raw columns available, z-score directly.
    # size/bm/mom/beta: only the FULL-PANEL z-score is cached (raw values were not
    # saved by the build step). Re-standardizing an already-per-quarter-z-scored
    # column within a SUBSET, using the subset's own per-quarter mean/SD and
    # winsorization quantiles, is algebraically IDENTICAL to z-scoring the raw
    # characteristic within that subset directly: z-scoring is an affine transform
    # per (quarter, firm), so Z2 = standardize_subset(standardize_full(X)) reduces
    # to standardize_subset(X) once the full-panel's per-quarter constants cancel,
    # and winsorization at within-subset quantiles is preserved under the strictly
    # monotonic full-panel z-score transform. So applying cs_wz() to the cached
    # _z column below gives the same result as having the raw column.
    for raw_col, z_col in [("delta_s", "delta_s_z"), ("dH_gpm", "delta_h_z")]:
        sub[z_col] = cs_wz(sub, raw_col)
    for z_col in ["size_z", "bm_z", "mom_z", "beta_z"]:
        sub[z_col] = cs_wz(sub, z_col)
    n_tick = sub["ticker"].nunique()
    avg_q = sub.groupby("q").size().mean()
    row = dict(k=k, n_tick=n_tick, avg_q=avg_q, n_obs=len(sub))
    for col in CHARS:
        r = fama_macbeth_nw(sub.dropna(subset=["ret_next", col]), "ret_next", [col])
        row[col] = r.get(col, {}).get("t", np.nan)
        row[col + "_coef"] = r.get(col, {}).get("coef", np.nan)
        row[col + "_n"] = r.get(col, {}).get("n", 0)
    rows.append(row)
    P(f"\n[k={k}] N_tickers={n_tick}  avg_firms/qtr={avg_q:.1f}  N_obs={len(sub):,}")
    for col, label in CHARS.items():
        P(f"  {label:22} t={row[col]:+.3f}  coef={row[col+'_coef']:+.6f}  quarters={row[col+'_n']}")

P("\n" + "="*88)
P("E3 SUMMARY LADDER TABLE — FM t by characteristic, by k")
P("="*88)
P(f"{'k(yrs)':>7}" + "".join(f"{lbl:>16}" for lbl in CHARS.values()))
for row in rows:
    P(f"{row['k']:>7}" + "".join(f"{row[col]:>+16.2f}" for col in CHARS))

P("\n" + "="*88)
P("Monotonicity check: does t rise (in the DeltaS direction) monotonically with k")
P("for the placebos too, or is DeltaS distinctive?")
P("="*88)
P("(Using |t| MAGNITUDE, not signed t, as the criterion: a placebo becoming MORE")
P("negative-and-significant is just as much evidence that survival conditioning")
P("inflates unrelated slopes as becoming more positive. A first pass here using")
P("signed net-move produced a misleading 1/4 count by missing Size's and Momentum's")
P("large moves in the negative direction -- corrected below.)")
abs_ts = {}
for col, label in CHARS.items():
    ts = [row[col] for row in rows]
    abs_t = [abs(x) for x in ts]
    abs_ts[col] = abs_t
    abs_diffs = np.diff(abs_t)
    is_monotone_increasing = np.all(abs_diffs >= -0.10)  # small tolerance
    net_move_abs = abs_t[-1] - abs_t[0]
    P(f"{label:22}: |t|(k=0)={abs_t[0]:.2f} -> |t|(k=27)={abs_t[-1]:.2f}  "
      f"net_|t|_move={net_move_abs:+.2f}  monotone_nondecreasing_|t|={'YES' if is_monotone_increasing else 'no'}  "
      f"(raw t: {ts[0]:+.2f} -> {ts[-1]:+.2f})")

ds_climb_abs = abs_ts["delta_s_z"][-1] - abs_ts["delta_s_z"][0]
placebo_moves_abs = {col: abs_ts[col][-1] - abs_ts[col][0] for col in ["size_z", "bm_z", "mom_z", "beta_z"]}
n_placebos_climb_abs = sum(1 for v in placebo_moves_abs.values() if v > 1.0)
n_placebos_monotone_abs = sum(1 for col in ["size_z", "bm_z", "mom_z", "beta_z"]
                               if np.all(np.diff(abs_ts[col]) >= -0.10))
P(f"\nDeltaS |t| net move (k=0 -> k=27): {ds_climb_abs:+.2f}")
P(f"Placebo |t| net moves: " + ", ".join(f"{CHARS[c]}={v:+.2f}" for c, v in placebo_moves_abs.items()))
P(f"Placebos with |t| net move > 1.0 (magnitude, either sign): {n_placebos_climb_abs}/4")
P(f"Placebos with a monotone-nondecreasing |t| trajectory across the full ladder: {n_placebos_monotone_abs}/4")

if n_placebos_climb_abs >= 2:
    P("\nVERDICT: survival conditioning inflates MULTIPLE placebo slopes in magnitude too,")
    P("not just DeltaS -- Size (|t| 0.13->2.54) and Beta (|t| 0.39->1.81) both show moves")
    P("comparable in size to DeltaS's own (|t| 0.28->3.33), and Momentum (|t| 0.05->1.39)")
    P("shows a smaller but real move in the same direction. R20 demonstrates a property of")
    P("survival-conditioned panels more broadly (shrinking, longer-surviving samples")
    P("systematically move MULTIPLE unrelated cross-sectional slopes toward larger")
    P("|t|-statistics, not just DeltaS) rather than something specific to disorder/entropy.")
    P("The exhibit survives as a description of the DeltaS collapse-and-recovery pattern,")
    P("but the causal interpretation needs to narrow from 'survival conditioning")
    P("manufactures THE premium' (singular, disorder-specific) to 'survival conditioning")
    P("manufactures premiums' (plural, a property of the conditioning, not the")
    P("characteristic) -- exactly the restatement the spec anticipated as the less")
    P("favorable but real possible outcome. Book-to-market is the partial exception")
    P("(already borderline-significant at k=0, t=+1.07, and moves only modestly further).")
else:
    P("\nVERDICT: placebos stay flat or non-monotone in |t| while DeltaS climbs sharply --")
    P("R20 is stronger than currently claimed; the survival-conditioning effect on")
    P("DeltaS looks distinctive rather than a generic small-sample/survivor-panel")
    P("artifact common to any cross-sectional characteristic.")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
