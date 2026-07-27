"""E1 — Dispersion-normalized placebo characteristics.

Extends D1.2's dispersion normalization (previously applied to DeltaS only)
to all four placebo characteristics (size, book-to-market, momentum, beta),
in both panels, plus the pairwise slope-difference test (DeltaS-normalized
vs each placebo-normalized) so the comparison is a formal test rather than a
threshold eyeball. Uses the cached placebo panels built once by
E1_E3_build_placebos.py (data/E1E3_{sp,qpanel}_with_placebos.parquet) --
same expensive rolling-beta construction reused for E3.

Per spec: report the four cells whatever they show; the normalization window
and dispersion measure are NOT tuned to separate DeltaS from the placebos --
both use the identical construction already implemented for DeltaS in D1.2
(sigma_cs,t+1 = cross-sectional SD of the priced return in the SAME
first-step cross-section).
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/E1_dispersion_normalized_placebos.txt"

print(f"[pid={os.getpid()}] E1 — fresh process")
log = []
def P(s=""):
    print(s)
    log.append(str(s))


def ar1(s):
    s = pd.Series(s).dropna().values
    if len(s) < 3:
        return np.nan
    return np.corrcoef(s[:-1], s[1:])[0, 1]


def bic_nw_lag(resid, max_lag=12):
    r = pd.Series(resid).dropna()
    n = len(r)
    best_p, best_bic = 0, np.inf
    for p in range(0, min(max_lag, n // 4) + 1):
        try:
            if p == 0:
                rss = float(((r - r.mean())**2).sum()); k = 1
            else:
                mfit = sm.tsa.AutoReg(r, lags=p, old_names=False).fit()
                rss = float((mfit.resid**2).sum()); k = p + 1
            bic = n * np.log(rss / n) + k * np.log(n)
            if bic < best_bic:
                best_bic, best_p = bic, p
        except Exception:
            pass
    return best_p


def reg_report(y, x, tag):
    y = np.asarray(y).ravel()
    x = np.asarray(x).reshape(len(y), -1)
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    p = bic_nw_lag(ols.resid)
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(p, 1)})
    n = int(ols.nobs)
    rho = ar1(y)
    P(f"  {tag}")
    P(f"    slope={ols.params[1]:+.5f}  R2={ols.rsquared:.3f}  n={n}  AR1(y)={rho:+.2f}")
    P(f"    HAC  t={hac.tvalues[1]:+.2f} (p={hac.pvalues[1]:.3f})  [NW lag={max(p,1)} by BIC]")
    return dict(hac_t=hac.tvalues[1], hac_p=hac.pvalues[1], slope=ols.params[1], n=n)


def step1_betas_disp(panel, ycol, dh, ds, datecol, Tcol, min_cs):
    rec = []
    for d, g in panel.groupby(datecol):
        sub = g[[ycol, dh, ds, Tcol]].dropna()
        if len(sub) < min_cs:
            continue
        X = sm.add_constant(sub[[dh, ds]], has_constant="add")
        r = sm.OLS(sub[ycol], X).fit()
        rec.append((d, r.params[dh], r.params[ds], sub[Tcol].iloc[0], sub[ycol].std()))
    return pd.DataFrame(rec, columns=["date", "b_dH", "b_ds", "T", "sigma_cs"]).set_index("date")


def stacked_diff_test(b_ref, b_other, T, tag):
    """Slope-difference test: b_ref (DeltaS-normalized) vs b_other (placebo-normalized)."""
    df = pd.DataFrame({"bref": b_ref, "both": b_other, "T": T}).dropna()
    stk = pd.concat([
        pd.DataFrame({"beta": df["bref"], "T": df["T"], "D_S": 0.0, "per": df.index}),
        pd.DataFrame({"beta": df["both"], "T": df["T"], "D_S": 1.0, "per": df.index}),
    ], ignore_index=True)
    stk["TxD"] = stk["T"] * stk["D_S"]
    X = sm.add_constant(stk[["T", "D_S", "TxD"]])
    grp = pd.Categorical(stk["per"].astype(str)).codes
    res = sm.OLS(stk["beta"], X).fit(cov_type="cluster", cov_kwds={"groups": grp})
    d_, td, pdv = res.params["TxD"], res.tvalues["TxD"], res.pvalues["TxD"]
    P(f"  {tag:42} d(TxD)={d_:+.5f}  t={td:+.2f}  p={pdv:.4f}  [clusters={len(set(grp))}, obs={len(stk)}]")
    return d_, td, pdv


P("="*88)
P("E1 — Dispersion-normalized placebo characteristics (size, B/M, momentum, beta)")
P("="*88)

sp = pd.read_parquet(f"{DATA}/E1E3_sp_with_placebos.parquet")
q_panel = pd.read_parquet(f"{DATA}/E1E3_qpanel_with_placebos.parquet")
P(f"Loaded cached placebo panels: SP {len(sp):,} rows ({sp['date'].min()}..{sp['date'].max()}), "
  f"R18 {len(q_panel):,} rows ({q_panel['q'].min()}..{q_panel['q'].max()})")

PLACEBOS = ["size_z", "bm_z", "mom_z", "beta_z"]
PLACEBO_LABELS = {"size_z": "Size (log mktcap)", "bm_z": "Book-to-market",
                   "mom_z": "Momentum (12-1)", "beta_z": "Market beta"}

P("\n" + "="*88)
P("Raw and dispersion-normalized beta_char,t ~ T_t, both panels, all four placebos")
P("="*88)

results = {}
for ph in PLACEBOS:
    P(f"\n--- {PLACEBO_LABELS[ph]} ---")
    P("-- S&P 500 monthly --")
    b_sp = step1_betas_disp(sp, "ret_next_month", "dH_gpm_z", ph, "date", "T", min_cs=10)
    b_sp[f"b_norm"] = b_sp["b_ds"] / b_sp["sigma_cs"]
    r_sp_raw = reg_report(b_sp["b_ds"].values, b_sp[["T"]].values, "raw")
    r_sp_norm = reg_report(b_sp["b_norm"].values, b_sp[["T"]].values, "dispersion-normalized")

    P("-- R18 full-universe quarterly --")
    sub = q_panel.dropna(subset=["ret_next", "delta_h_z", ph, "T"])
    b_q = step1_betas_disp(sub, "ret_next", "delta_h_z", ph, "q", "T", min_cs=20)
    b_q["b_norm"] = b_q["b_ds"] / b_q["sigma_cs"]
    r_q_raw = reg_report(b_q["b_ds"].values, b_q[["T"]].values, "raw")
    r_q_norm = reg_report(b_q["b_norm"].values, b_q[["T"]].values, "dispersion-normalized")

    results[ph] = dict(b_sp=b_sp, b_q=b_q, sp_raw=r_sp_raw, sp_norm=r_sp_norm,
                        q_raw=r_q_raw, q_norm=r_q_norm)

P("\n" + "="*88)
P("SUMMARY TABLE — raw vs dispersion-normalized HAC t, both panels")
P("="*88)
P(f"{'Characteristic':22}{'SP raw t':>10}{'SP norm t':>11}{'FU raw t':>10}{'FU norm t':>11}")
# recompute DeltaS row for comparison (as in D1.2)
b_sp_ds = step1_betas_disp(sp, "ret_next_month", "dH_gpm_z", "DS_z", "date", "T", min_cs=10)
b_sp_ds["b_norm"] = b_sp_ds["b_ds"] / b_sp_ds["sigma_cs"]
sub_ds = q_panel.dropna(subset=["ret_next", "delta_h_z", "delta_s_z", "T"])
b_q_ds = step1_betas_disp(sub_ds, "ret_next", "delta_h_z", "delta_s_z", "q", "T", min_cs=20)
b_q_ds["b_norm"] = b_q_ds["b_ds"] / b_q_ds["sigma_cs"]
ds_sp_raw_t = reg_report(b_sp_ds["b_ds"].values, b_sp_ds[["T"]].values, "(silent)")["hac_t"] if False else None

import contextlib, io as _io
def quiet_reg(y, x):
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        old_log = log.copy()
        r = reg_report(y, x, "(suppressed)")
        del log[len(old_log):]
    return r

ds_sp_raw = quiet_reg(b_sp_ds["b_ds"].values, b_sp_ds[["T"]].values)
ds_sp_norm = quiet_reg(b_sp_ds["b_norm"].values, b_sp_ds[["T"]].values)
ds_q_raw = quiet_reg(b_q_ds["b_ds"].values, b_q_ds[["T"]].values)
ds_q_norm = quiet_reg(b_q_ds["b_norm"].values, b_q_ds[["T"]].values)
P(f"{'DeltaS (reference)':22}{ds_sp_raw['hac_t']:>+10.2f}{ds_sp_norm['hac_t']:>+11.2f}"
  f"{ds_q_raw['hac_t']:>+10.2f}{ds_q_norm['hac_t']:>+11.2f}")
for ph in PLACEBOS:
    r = results[ph]
    P(f"{PLACEBO_LABELS[ph]:22}{r['sp_raw']['hac_t']:>+10.2f}{r['sp_norm']['hac_t']:>+11.2f}"
      f"{r['q_raw']['hac_t']:>+10.2f}{r['q_norm']['hac_t']:>+11.2f}")

n_placebo_sig_norm = sum(1 for ph in PLACEBOS
                          if abs(results[ph]["sp_norm"]["hac_t"]) > 2.0 or abs(results[ph]["q_norm"]["hac_t"]) > 2.0)
P(f"\nPlacebos significant on T AFTER normalization (either panel, |t|>2.0): {n_placebo_sig_norm}/4")
P(f"DeltaS significant on T after normalization: SP {'YES' if abs(ds_sp_norm['hac_t'])>2.0 else 'no'} "
  f"(t={ds_sp_norm['hac_t']:+.2f}), FU {'YES' if abs(ds_q_norm['hac_t'])>2.0 else 'no'} (t={ds_q_norm['hac_t']:+.2f})")

P("\n" + "="*88)
P("Pairwise slope-difference test: DeltaS-normalized vs each placebo-normalized")
P("="*88)
for ph in PLACEBOS:
    r = results[ph]
    P(f"\n--- DeltaS-norm vs {PLACEBO_LABELS[ph]}-norm ---")
    stacked_diff_test(b_sp_ds["b_norm"], r["b_sp"]["b_norm"], b_sp_ds["T"], "S&P500 (monthly)")
    # align FU on quarter index
    common_fu = b_q_ds.index.intersection(r["b_q"].index)
    stacked_diff_test(b_q_ds.loc[common_fu, "b_norm"], r["b_q"].loc[common_fu, "b_norm"],
                       b_q_ds.loc[common_fu, "T"], "R18 full-universe (quarterly)")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
