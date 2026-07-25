"""D1a_2x2_cell.py — ONE cell of the survival x top-500-size 2x2, run as its
own standalone process (invoked separately per cell via CLI arg, per the V34
ground rule: fresh process per estimate, no cached panels/design matrices
across cells).

Usage: python3 D1a_2x2_cell.py {FB_noSurv|FB_surv|LC_noSurv|LC_surv}

Corrected-panel defaults: R18 full-universe SF1 quarterly panel, delisted
retained, no survival requirement unless the cell name says otherwise, size
cuts on LAGGED (t-1) capitalization, FM with Newey-West lag 4, within-quarter
cross-sectional z-scoring. (FF5+UMD control note: the underlying R18 panel
carries delta_h_z/delta_s_z as the only regressors in the manuscript's
Table 6/7/8/12 2x2 spec -- no FF5+UMD term is used in THAT spec, and D1a's
job is to reproduce it exactly, so none is added here. FF5+UMD is added in
D1b, which is a different, newly-specified regression.)
"""
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT_DIR = "../results/revision"

CELL = sys.argv[1] if len(sys.argv) > 1 else "FB_noSurv"
assert CELL in ("FB_noSurv", "FB_surv", "LC_noSurv", "LC_surv")
K = 27
THR_Q = K * 4


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
    for col in x_cols:
        s = cdf[col].dropna()
        n = len(s)
        mean_ = s.mean()
        gamma0 = (s**2).mean() - mean_**2
        var = gamma0
        for l in range(1, min(lags + 1, n)):
            g_ = ((s.iloc[l:].values - mean_) * (s.iloc[:-l].values - mean_)).mean()
            var += 2 * (1 - l / (lags + 1)) * g_
        se = np.sqrt(max(var, 1e-30) / n)
        out[col] = dict(coef=float(mean_), se=float(se), t=float(mean_ / se), n_quarters=int(n))
    return out


print(f"[pid={os.getpid()}] D1a cell={CELL} — fresh process, no shared state")

# ── fresh load, every cell rebuilds from parquet on disk ────────────────────
panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
panel = panel.sort_values(["ticker", "q_ord"]).reset_index(drop=True)

new_run = (panel["q_ord"] - panel.groupby("ticker")["q_ord"].shift(1)) != 1
panel["run_id"] = new_run.groupby(panel["ticker"]).cumsum()
panel["run_len_q"] = panel.groupby(["ticker", "run_id"])["q_ord"].transform("size")

sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                       columns=["ticker", "dimension", "calendardate", "marketcap"])
mc = sf1[sf1["dimension"] == "ARQ"].copy()
mc["calendardate"] = pd.to_datetime(mc["calendardate"], errors="coerce")
mc = mc.dropna(subset=["calendardate", "marketcap"])
mc = mc[mc["marketcap"] > 0]
mc["q"] = mc["calendardate"].dt.to_period("Q")
mc = mc.sort_values(["ticker", "calendardate"]).drop_duplicates(["ticker", "q"], keep="last")[["ticker", "q", "marketcap"]]
mc = mc.sort_values(["ticker", "q"])
mc["marketcap_lag1"] = mc.groupby("ticker")["marketcap"].shift(1)
panel = panel.merge(mc, on=["ticker", "q"], how="left")

# top-500 rank computed once against the FULL-universe cross-section (a prior
# bug recomputed this rank AFTER subsetting to the small survival-conditioned
# universe, making the cut vacuous -- fixed here by construction, rank is
# always against the unrestricted full panel)
mc_ranked = panel.dropna(subset=["marketcap_lag1"]).copy()
mc_ranked["mc_rank"] = mc_ranked.groupby("q")["marketcap_lag1"].rank(ascending=False, method="first")
panel = panel.merge(mc_ranked[["ticker", "q", "mc_rank"]], on=["ticker", "q"], how="left")
panel["top500_flag"] = panel["mc_rank"] <= 500

if CELL.startswith("FB"):
    d = panel.copy()
else:
    d = panel[panel["top500_flag"] == True].copy()
if CELL.endswith("surv"):
    d = d[d["run_len_q"] >= THR_Q].copy()

d["ds_z"] = cs_wz(d, "delta_s")
d["dh_z"] = cs_wz(d, "dH_gpm")
pf = d.dropna(subset=["ret_next", "ds_z", "dh_z"])

fm = fama_macbeth_nw(pf, "ret_next", ["dh_z", "ds_z"])

first_q = str(pf["q"].min())
last_q = str(pf["q"].max())
avg_firms_q = pf.groupby("q").size().mean()
n_tickers = pf["ticker"].nunique()

X = sm.add_constant(pf[["dh_z", "ds_z"]]).values
design_hash = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()

print(f"cell={CELL}  N={len(pf):,}  tickers={n_tickers:,}  avg firms/qtr={avg_firms_q:.1f}  "
      f"quarters(t_dS)={fm['ds_z']['n_quarters']}  date_range={first_q}..{last_q}")
print(f"design shape={X.shape}  SHA-1={design_hash}  pid={os.getpid()}")
print(f"t(dS)={fm['ds_z']['t']:+.4f}  coef(dS)={fm['ds_z']['coef']:+.6f}  SE(dS)={fm['ds_z']['se']:.6f}")
print(f"t(dH)={fm['dh_z']['t']:+.4f}  coef(dH)={fm['dh_z']['coef']:+.6f}  SE(dH)={fm['dh_z']['se']:.6f}")

out = dict(cell=CELL, pid=os.getpid(), N=int(len(pf)), n_tickers=int(n_tickers),
           avg_firms_q=float(avg_firms_q), first_q=first_q, last_q=last_q,
           design_shape=list(X.shape), design_hash=design_hash,
           t_ds=fm["ds_z"]["t"], coef_ds=fm["ds_z"]["coef"], se_ds=fm["ds_z"]["se"], nq_ds=fm["ds_z"]["n_quarters"],
           t_dh=fm["dh_z"]["t"], coef_dh=fm["dh_z"]["coef"], se_dh=fm["dh_z"]["se"], nq_dh=fm["dh_z"]["n_quarters"])
os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/D1a_cell_{CELL}.json", "w") as f:
    json.dump(out, f)
print(f"wrote {OUT_DIR}/D1a_cell_{CELL}.json")
