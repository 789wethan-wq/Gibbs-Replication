"""PanelC_fix.py — Re-estimate Table 2 Panel C control rows on the IDENTIFIED
pooled two-way-clustered (firm x date) interaction spec, S&P 500 panel.

Replaces the rank-deficient within-month FM encompassing loop (V1). Controls
reuse the exact R16 constructions: SYY mispricing composite, SYY x T, HXZ
q-factors (ME,IA,ROE), MAX proxy (|monthly ret|). FF5+UMD factor returns added
as time controls. DIAGNOSTIC — writes no manuscript text.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"

def cs_wz(df, col, date_col="date", pct=0.01):
    def _f(x):
        x2 = x.dropna()
        if len(x2) < 5: return pd.Series(np.nan, index=x.index)
        lo, hi = x2.quantile(pct), x2.quantile(1 - pct)
        xc = x.clip(lo, hi); s = xc.std()
        if s < 1e-10: return pd.Series(np.nan, index=x.index)
        return (xc - xc.mean()) / s
    return df.groupby(date_col)[col].transform(_f)

def cluster_vcov(X, resid, groups):
    n_, k_ = X.shape; inv = np.linalg.pinv(X.T @ X); B = np.zeros((k_, k_))
    for g in np.unique(groups):
        mm = groups == g; B += X[mm].T @ np.outer(resid[mm], resid[mm]) @ X[mm]
    G = len(np.unique(groups)); sc = (G / (G - 1)) * ((n_ - 1) / (n_ - k_))
    return sc * inv @ B @ inv

def twoway_vcov(X, resid, gd, gf):
    inter = pd.Categorical(pd.Series(gd).astype(str) + "_" + pd.Series(gf).astype(str)).codes
    return cluster_vcov(X, resid, gd) + cluster_vcov(X, resid, gf) - cluster_vcov(X, resid, inter)

def pooled_2way(df, xcols, ycol="ret_next_month"):
    """Pooled OLS, two-way (firm x date) clustered. Returns dict col->(coef,t,p)."""
    d = df.dropna(subset=xcols + [ycol]).copy()
    n = len(d)
    X = np.column_stack([np.ones(n)] + [d[c].values.astype(float) for c in xcols])
    y = d[ycol].values.astype(float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    gd = pd.Categorical(d["date"].astype(str)).codes
    gf = pd.Categorical(d["stock_id"].astype(str)).codes
    V = twoway_vcov(X, r, gd, gf)
    se = np.sqrt(np.diag(V)); t = b / se
    names = ["const"] + xcols
    out = {nm: (b[i], t[i], 1 - chi2.cdf(t[i]**2, 1)) for i, nm in enumerate(names)}
    out["_N"] = n; out["_G_date"] = d["date"].nunique(); out["_G_firm"] = d["stock_id"].nunique()
    return out

# ── base panel ──────────────────────────────────────────────────────────────
m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
m["date"] = pd.to_datetime(m["date"])
m["dH_gpm_z"] = cs_wz(m, "dH_gpm")            # exactly as R16 load_base()
# TxDS already in panel (= T * DS_z, verified). Rename for clarity.
m["TxDS_z"] = m["TxDS"]
FF = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]
fac = pd.read_parquet(f"{DATA}/factors_monthly.parquet"); fac.index = pd.to_datetime(fac.index)
m = m.merge(fac[FF].reset_index().rename(columns={"index": "date"}), on="date", how="left")

# SYY mispricing composite + interaction
mip = pd.read_parquet(f"{DATA}/mispricing_monthly.parquet")
mip["date"] = pd.to_datetime(mip["date"])
m = m.merge(mip[["date", "stock_id", "mispricing_raw"]], on=["date", "stock_id"], how="left")
m["mispricing_z"] = cs_wz(m, "mispricing_raw")
m["mip_T"] = m["mispricing_z"] * m["T"]

# HXZ q-factors (returns, time-varying) merged on date
q = pd.read_parquet(f"{DATA}/hxz_q5_monthly.parquet")
q["date"] = (pd.to_datetime(q["year"].astype(str) + "-" + q["month"].astype(str).str.zfill(2) + "-01")
             + pd.offsets.MonthEnd(0))
for c in ["R_ME", "R_IA", "R_ROE"]: q[c] = q[c] / 100.0
m = m.merge(q[["date", "R_ME", "R_IA", "R_ROE"]], on="date", how="left")

# MAX proxy = |monthly ret|, z-scored within month (R16.4)
m["max_proxy"] = m["ret"].abs()
m["max_proxy_z"] = cs_wz(m, "max_proxy")

BASE = ["dH_gpm_z", "DS_z", "TxDS_z"]

def line(tag, res, extra=()):
    dS = res["DS_z"]; Tx = res["TxDS_z"]
    s = (f"[PanelC-fix | {tag} | pooled 2-way-clustered firm×date]\n"
         f"  coef_TxdS={Tx[0]:+.6f}  t_TxdS={Tx[1]:+.4f}  p={Tx[2]:.4f}   "
         f"coef_dS={dS[0]:+.6f}  t_dS={dS[1]:+.4f}")
    for nm, key in extra:
        c = res[key]; s += f"   {nm}: coef={c[0]:+.6f} t={c[1]:+.4f}"
    s += f"\n  N={res['_N']:,}  months={res['_G_date']}  firms={res['_G_firm']}"
    print(s)

print("="*74)
# Row 0 — confirm V2 (no FF5): identified baseline
r_v2 = pooled_2way(m, BASE)
line("0. V2 confirm (ΔH+ΔS+T·ΔS, NO FF5)", r_v2)

# Row 1 — Baseline + FF5+UMD
r1 = pooled_2way(m, BASE + FF)
line("1. Baseline + FF5+UMD", r1,
     extra=[("Mkt","Mkt_RF"), ("HML","HML"), ("Mom","Mom")])

# Row 2 — + SYY mispricing composite
r2 = pooled_2way(m, BASE + FF + ["mispricing_z"])
line("2. + SYY mispricing_z", r2, extra=[("SYY","mispricing_z")])

# Row 3 — + SYY mispricing × T
r3 = pooled_2way(m, BASE + FF + ["mispricing_z", "mip_T"])
line("3. + SYY mispricing_z + SYY×T", r3,
     extra=[("SYY","mispricing_z"), ("SYY×T","mip_T")])

# Row 4 — + HXZ q-factors (ME, IA, ROE)
r4 = pooled_2way(m, BASE + FF + ["R_ME", "R_IA", "R_ROE"])
line("4. + HXZ q-factors (ME,IA,ROE)", r4,
     extra=[("q_ME","R_ME"), ("q_IA","R_IA"), ("q_ROE","R_ROE")])

# Row 5 — + MAX proxy (|monthly ret|)
r5 = pooled_2way(m, BASE + FF + ["max_proxy_z"])
line("5. + MAX proxy |ret|", r5, extra=[("MAX","max_proxy_z")])
print("="*74)
