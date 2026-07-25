"""D3_cell.py — ONE arm of the +1.03-vs-+0.58 reconciliation, run as its own
process. Crosses cap timing (lagged t-1 vs contemporaneous) against two rig
variants:

  rig=2x2   : D1a_2x2_cell.py's construction -- top-500 rank computed against
              the FULL cap-available cross-section each quarter (before any
              return/DS/DH dropna), then ds_z/dh_z RECOMPUTED (re-standardized)
              WITHIN the resulting top-500 subset.
  rig=table12 : D4_lagged_cap_rerun.py's construction (the Table 12 size-ladder
              script) -- same rank-then-subset order, same re-standardization
              within the subset. Included in full even though inspection
              suggests it is structurally identical to rig=2x2, because the
              point of this run is to VERIFY that, not assert it from reading
              the code.

Usage: python3 D3_cell.py {lag|contemp} {2x2|table12}
"""
import os
import sys
import hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT_DIR = "../results/revision"

CAP_ARG = sys.argv[1] if len(sys.argv) > 1 else "lag"
RIG_ARG = sys.argv[2] if len(sys.argv) > 2 else "2x2"
assert CAP_ARG in ("lag", "contemp")
assert RIG_ARG in ("2x2", "table12")

print(f"[pid={os.getpid()}] D3 arm cap={CAP_ARG} rig={RIG_ARG} — fresh process")


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


panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")

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

capcol = "marketcap_lag1" if CAP_ARG == "lag" else "marketcap"

if RIG_ARG == "2x2":
    # D1a_2x2_cell.py's exact construction: rank against the full cap-available
    # cross-section (fixed top500_flag column), then merge back onto full panel.
    mc_ranked = panel.dropna(subset=[capcol]).copy()
    mc_ranked["mc_rank"] = mc_ranked.groupby("q")[capcol].rank(ascending=False, method="first")
    panel2 = panel.merge(mc_ranked[["ticker", "q", "mc_rank"]], on=["ticker", "q"], how="left")
    panel2["top500_flag"] = panel2["mc_rank"] <= 500
    d = panel2[panel2["top500_flag"] == True].copy()
else:
    # D4_lagged_cap_rerun.py's run_arm() construction: dropna(capcol) first,
    # rank within that subset, then take mc_rank<=500 directly (no merge-back
    # onto the unrestricted panel -- rows without capcol are excluded from the
    # ranking universe from the start, not merged in as NaN-ranked).
    d = panel.dropna(subset=[capcol]).copy()
    d["mc_rank"] = d.groupby("q")[capcol].rank(ascending=False, method="first")
    d = d[d["mc_rank"] <= 500].copy()

d["ds_z"] = cs_wz(d, "delta_s")
d["dh_z"] = cs_wz(d, "dH_gpm")
pf = d.dropna(subset=["ret_next", "ds_z", "dh_z"])

fm = fama_macbeth_nw(pf, "ret_next", ["dh_z", "ds_z"])

first_q, last_q = str(pf["q"].min()), str(pf["q"].max())
avg_firms_q = pf.groupby("q").size().mean()
X = sm.add_constant(pf[["dh_z", "ds_z"]]).values
design_hash = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()

print(f"cap={CAP_ARG} rig={RIG_ARG}  N={len(pf):,}  avg firms/qtr={avg_firms_q:.1f}  "
      f"date_range={first_q}..{last_q}")
print(f"design shape={X.shape}  SHA-1={design_hash}  pid={os.getpid()}")
print(f"t(dS)={fm['ds_z']['t']:+.4f}  coef(dS)={fm['ds_z']['coef']:+.6f}  SE(dS)={fm['ds_z']['se']:.6f}  "
      f"quarters={fm['ds_z']['n_quarters']}")
print(f"t(dH)={fm['dh_z']['t']:+.4f}  coef(dH)={fm['dh_z']['coef']:+.6f}  SE(dH)={fm['dh_z']['se']:.6f}")

import json
out = dict(cap=CAP_ARG, rig=RIG_ARG, pid=os.getpid(), N=int(len(pf)), avg_firms_q=float(avg_firms_q),
           first_q=first_q, last_q=last_q, design_shape=list(X.shape), design_hash=design_hash,
           t_ds=fm["ds_z"]["t"], coef_ds=fm["ds_z"]["coef"], se_ds=fm["ds_z"]["se"],
           t_dh=fm["dh_z"]["t"], coef_dh=fm["dh_z"]["coef"], se_dh=fm["dh_z"]["se"])
os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/D3_cell_{CAP_ARG}_{RIG_ARG}.json", "w") as f:
    json.dump(out, f)
print(f"wrote {OUT_DIR}/D3_cell_{CAP_ARG}_{RIG_ARG}.json")
