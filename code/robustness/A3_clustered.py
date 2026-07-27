#!/usr/bin/env python3
"""A3 re-run with DATE-CLUSTERED standard errors.

The stacked second-step design places two observations per period (beta_dH,t and
beta_dS,t) from the SAME cross-section -> within-period correlated. HAC/NW on the
stacked series ignores that; clustering by date (period) is the correct SE.
Model:  beta_channel,t = a + b*T_t + c*D_S + d*(T_t*D_S) + e ,  cluster by period.
d = differential temperature-sensitivity of the entropy slope vs the enthalpy slope.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robustness_utils import winsorize_cs, zscore_cs
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def fmnw(panel, y, xs, date="q", min_cs=20):
    coefs = []
    for d, g in panel.groupby(date):
        s = g[[y] + xs].dropna()
        if len(s) < max(min_cs, len(xs) + 2):
            continue
        X = sm.add_constant(s[xs], has_constant="add")
        coefs.append(sm.OLS(s[y], X).fit().params[xs].rename(d))
    return pd.DataFrame(coefs)

def a3_clustered(betas, Tser, tag, dh, ds):
    b = betas.copy()
    T = Tser.reindex(b.index)
    df = pd.DataFrame({"bDH": b[dh], "bDS": b[ds], "T": T, "per": b.index}).dropna()
    stk = pd.concat([
        pd.DataFrame({"beta": df["bDH"], "T": df["T"], "D_S": 0.0, "per": df["per"]}),
        pd.DataFrame({"beta": df["bDS"], "T": df["T"], "D_S": 1.0, "per": df["per"]}),
    ], ignore_index=True)
    stk["TxD"] = stk["T"] * stk["D_S"]
    X = sm.add_constant(stk[["T", "D_S", "TxD"]])
    grp = pd.Categorical(stk["per"].astype(str)).codes
    res = sm.OLS(stk["beta"], X).fit(cov_type="cluster", cov_kwds={"groups": grp})
    d_, td, pd_ = res.params["TxD"], res.tvalues["TxD"], res.pvalues["TxD"]
    print(f"{tag:26} d={d_:+.4f}  t(d)={td:+.2f}  p(d)={pd_:.4f}   "
          f"[clusters={len(set(grp))}, obs={len(stk)}]")
    return d_, td, pd_

# Full-universe quarterly
fu = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
pf = fu.dropna(subset=["ret_next", "delta_s_z", "delta_h_z"])
bfu = fmnw(pf, "ret_next", ["delta_h_z", "delta_s_z"], date="q")
Tq = pf.groupby("q")["T"].first()

# S&P500 monthly
sp = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
sp["date"] = pd.to_datetime(sp["date"])
sp["dH_gpm_z"] = sp.groupby("date")["dH_gpm"].transform(lambda x: zscore_cs(winsorize_cs(x)))
spb = sp.dropna(subset=["ret_next_month", "dH_gpm_z", "DS_z"])
bsp = fmnw(spb, "ret_next_month", ["dH_gpm_z", "DS_z"], date="date")
Tm = spb.groupby("date")["T"].first()

print("A3 — SLOPE-DIFFERENCE TEST, DATE-CLUSTERED SE")
print("-" * 68)
a3_clustered(bfu, Tq, "FULL-UNIVERSE (quarterly)", "delta_h_z", "delta_s_z")
a3_clustered(bsp, Tm, "S&P500 (monthly)", "dH_gpm_z", "DS_z")
