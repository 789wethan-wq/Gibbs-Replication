"""R3 items 2 & 5 — confirmations (no manuscript text).

Item 2: Table 5b window sweep (w=60) sample vs Table 7 Model-B sample. The sweep
uses proportional min_periods=max(8,w/2)=30 at w=60; the primary R18 build uses
fixed min_periods=24. Confirm subset relationship and that the excluded rows are
EXACTLY the 24–29-obs-GPM-history rows, with no other construction difference.

Item 5: Table 3 Panel A three Wald p (price 0.013 / accounting GPM 0.017 /
accounting ROE 0.024). Confirm all three are DATE-clustered (same convention),
distinct from the double-cluster values.
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

# ══════════════════════════════════════════════════════════════════════════
# ITEM 2 — window sweep (w=60, mp=30) vs Table 7 primary (w=60, mp=24)
# ══════════════════════════════════════════════════════════════════════════
print("#"*78); print("# R3-2  Table 5b sweep-60 vs Table 7 Model-B: sample difference"); print("#"*78)

q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
# Table 7 Model-B primary sample: the SAVED panel (built with mp=24), full channel
t7 = q.dropna(subset=["ret_next", "delta_h_z", "delta_s_z", "T_delta_s"])
set_t7_saved = set(zip(t7["ticker"], t7["q"].astype(str)))
print(f"  [saved panel] Table 7 Model-B N = {len(t7):,}  (unique keys {len(set_t7_saved):,})")

# Rebuild ΔH_GPM at w=60 for BOTH min_periods rules from the SAME source, so the
# only thing that changes between the two is min_periods.
mf = pd.read_parquet(f"{DATA}/monthly_fundamentals.parquet")
mf["date"] = pd.to_datetime(mf["date"]); mf = mf.sort_values(["stock_id", "date"])
base = q[["ticker", "q", "ret_next", "delta_s", "delta_s_z", "T"]].copy()

def build_sample(mp, tag):
    dh = -mf.groupby("stock_id")["gpm"].transform(lambda x: x.rolling(60, min_periods=mp).std())
    cnt = mf.groupby("stock_id")["gpm"].transform(lambda x: x.rolling(60, min_periods=1).count())
    tmp = mf[["stock_id", "date"]].assign(dH_gpm=dh.values, gpm_cnt=cnt.values)
    tmp["q"] = tmp["date"].dt.to_period("Q")
    gpmq = (tmp.dropna(subset=["dH_gpm"]).sort_values(["stock_id", "date"])
              .drop_duplicates(["stock_id", "q"], keep="last")
              .rename(columns={"stock_id": "ticker"})[["ticker", "q", "dH_gpm", "gpm_cnt"]])
    p = base.merge(gpmq, on=["ticker", "q"], how="inner")
    p["delta_h_z"] = cs_wz(p, "dH_gpm", "q")
    ps = p.dropna(subset=["ret_next", "delta_h_z", "delta_s_z"]).copy()
    ps["key"] = list(zip(ps["ticker"], ps["q"].astype(str)))
    print(f"  [rebuild mp={mp:>2}] {tag:14} N = {len(ps):,}")
    return ps

primary = build_sample(24, "primary/T7")     # should reproduce the saved panel
sweep   = build_sample(30, "sweep-60")

set_primary = set(primary["key"]); set_sweep = set(sweep["key"])
symdiff = set_primary ^ set_sweep
excluded = set_primary - set_sweep            # in primary, dropped by sweep
extra    = set_sweep - set_primary            # in sweep, not in primary (should be 0)
subset   = set_sweep.issubset(set_primary)

print(f"\n  rebuild-primary(mp24) N={len(set_primary):,}  vs saved-Table7 N={len(set_t7_saved):,}  "
      f"match={set_primary == set_t7_saved} (|Δ|={len(set_primary ^ set_t7_saved):,})")
print(f"  sweep_N={len(set_sweep):,}  t7_N={len(set_primary):,}  symdiff={len(symdiff):,}  "
      f"subset(sweep⊂primary)={subset}  extra(sweep∖primary)={len(extra):,}")

# mechanism: excluded rows must have GPM history count in [24,29]
exc_df = primary[primary["key"].isin(excluded)]
cnt_min, cnt_max = exc_df["gpm_cnt"].min(), exc_df["gpm_cnt"].max()
in_range = ((exc_df["gpm_cnt"] >= 24) & (exc_df["gpm_cnt"] <= 29)).mean()
print(f"  excluded rows: {len(excluded):,}  gpm_cnt range=[{cnt_min:.0f},{cnt_max:.0f}]  "
      f"share in [24,29]={in_range:.4%}")
print(f"  mechanism_confirmed={subset and len(extra)==0 and in_range==1.0 and cnt_min>=24 and cnt_max<=29}")

# ══════════════════════════════════════════════════════════════════════════
# ITEM 5 — Table 3 Panel A three Wald p, clustering convention
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "#"*78); print("# R3-5  Table 3 Panel A: clustering convention of the 3 Wald p-values"); print("#"*78)

def cluster_vcov(X, resid, codes):
    n_, k_ = X.shape; inv = np.linalg.pinv(X.T @ X)
    G = int(codes.max()) + 1; Xr = X * resid[:, None]
    S = np.zeros((G, k_)); np.add.at(S, codes, Xr); B = S.T @ S
    sc = (G / (G - 1)) * ((n_ - 1) / (n_ - k_))
    return sc * inv @ B @ inv

def twoway(X, resid, gd, gf):
    inter = pd.Categorical(pd.Series(gd).astype(str) + "_" + pd.Series(gf).astype(str)).codes
    return cluster_vcov(X, resid, gd) + cluster_vcov(X, resid, gf) - cluster_vcov(X, resid, inter)

m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
m["date"] = pd.to_datetime(m["date"])
m["dH_gpm_z"] = cs_wz(m, "dH_gpm")
m["dH_acc_z"] = cs_wz(m, "dH_accounting")   # accounting-ROE stability, z-scored within month
# DH_z (price-based) already z-scored in panel

def wald_txds(dh_col, tag, claimed):
    sub = m.dropna(subset=[dh_col, "DS_z", "TxDS", "ret_next_month"]).copy()
    n = len(sub)
    X = np.column_stack([np.ones(n), sub[dh_col], sub["DS_z"], sub["TxDS"]])
    y = sub["ret_next_month"].values
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    gd = pd.Categorical(sub["date"].astype(str)).codes
    gf = pd.Categorical(sub["stock_id"].astype(str)).codes
    out = {}
    for lab, V in [("date", cluster_vcov(X, r, gd)), ("double", twoway(X, r, gd, gf))]:
        t = b[3] / np.sqrt(V[3, 3]); out[lab] = (t, 1 - chi2.cdf(t**2, 1))
    dt, dp = out["date"]; ct, cp = out["double"]
    print(f"  {tag:16} N={n:,}  date-cluster: t={dt:+.4f} p={dp:.4f}  |  "
          f"double-cluster: t={ct:+.4f} p={cp:.4f}   (Table3 PanelA claims p={claimed})")
    return dp

p_price = wald_txds("DH_z",     "price-based ΔH", 0.013)
p_gpm   = wald_txds("dH_gpm_z", "accounting GPM", 0.017)
p_roe   = wald_txds("dH_acc_z", "accounting ROE", 0.024)
print(f"\n  date-clustered p: price={p_price:.4f} (claim .013), GPM={p_gpm:.4f} (claim .017), "
      f"ROE={p_roe:.4f} (claim .024)")
match = all(abs(a - b) < 0.001 for a, b in [(p_price, 0.013), (p_gpm, 0.017), (p_roe, 0.024)])
print(f"  all three reproduced as DATE-clustered within .001: {match}")
