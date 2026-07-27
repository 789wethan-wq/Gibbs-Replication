"""F_facts.py — Block 3 F3/F4/F5 numeric facts.

F3: GPM winsorization at ±1 (site: project/sharadar_pipeline.py:202, clip(-1,1)).
    Raw gpm = gp/revenue from SF1 ARY, pre-winsorization. Share clipped upper
    (>+1) and lower (<-1) separately, + 5-number summary, in both panels.
F4: AR(1) of monthly returns in the S&P panel.
F5: Mean/median distinct annual GPM filings underlying ΔH at each window
    (24/36/48/60/72 months), both panels — from monthly_fundamentals gpm.
"""
import os
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"


def five_num(s):
    s = s.dropna()
    return (s.min(), s.quantile(0.25), s.median(), s.quantile(0.75), s.max())


# ── panels & ticker sets ──────────────────────────────────────────────────────
m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")          # monthly / S&P
q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")   # quarterly / full-univ
sp_tickers = set(m["stock_id"].unique())
fu_tickers = set(q["ticker"].unique())

# ══ F3 ════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("[F3] GPM winsorization at ±1 (clip site sharadar_pipeline.py:202)")
print("=" * 72)
sf = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet")
sf = sf[sf["dimension"] == "ARY"].copy()
sf["datekey"] = pd.to_datetime(sf["datekey"], errors="coerce")
sf = sf.dropna(subset=["datekey", "gp", "revenue"])
sf = sf[sf["datekey"] >= "1985-01-01"]
sf["gpm_raw"] = sf["gp"] / sf["revenue"].replace(0, np.nan)
sf = sf.dropna(subset=["gpm_raw"])
sf["gpm_raw"] = sf["gpm_raw"].replace([np.inf, -np.inf], np.nan)
sf = sf.dropna(subset=["gpm_raw"])

for label, tset in [("MONTHLY / S&P panel", sp_tickers),
                    ("QUARTERLY / full-universe panel", fu_tickers)]:
    d = sf[sf["ticker"].isin(tset)]
    n = len(d)
    up = (d["gpm_raw"] > 1).mean() * 100
    lo = (d["gpm_raw"] < -1).mean() * 100
    mn, q1, md, q3, mx = five_num(d["gpm_raw"])
    print(f"  {label}: N_annual_obs={n:,}")
    print(f"    clipped upper (>+1): {up:.3f}%   clipped lower (<-1): {lo:.3f}%")
    print(f"    pre-winsor gpm 5-num: min={mn:+.3f} Q1={q1:+.3f} med={md:+.3f} "
          f"Q3={q3:+.3f} max={mx:+.3f}")

# ══ F4 ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("[F4] AR(1) of monthly returns — S&P panel (merged_with_accounting)")
print("=" * 72)
m["date"] = pd.to_datetime(m["date"])
mkt = m.groupby("date")["ret"].mean().sort_index()
ar1_mkt = mkt.autocorr(lag=1)
# firm-level AR(1), median across firms with ≥36 monthly obs
firm_ar = []
for sid, g in m.sort_values("date").groupby("stock_id"):
    r = g["ret"].dropna()
    if len(r) >= 36:
        a = r.autocorr(lag=1)
        if pd.notna(a):
            firm_ar.append(a)
firm_ar = pd.Series(firm_ar)
print(f"  aggregate (cross-sectional mean return) AR(1) = {ar1_mkt:+.3f}  (T={len(mkt)})")
print(f"  firm-level AR(1): median={firm_ar.median():+.3f}, mean={firm_ar.mean():+.3f} "
      f"(n_firms={len(firm_ar)}, ≥36 obs)")
print(f"  context: Panel B Monte Carlo note references a 0.809 ratio")

# ══ F5 ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("[F5] Distinct annual GPM filings underlying ΔH by window (both panels)")
print("=" * 72)
mf = pd.read_parquet(f"{DATA}/monthly_fundamentals.parquet")
mf["date"] = pd.to_datetime(mf["date"])
mf = mf.dropna(subset=["gpm"]).sort_values(["stock_id", "date"]).reset_index(drop=True)

# Forward-filled annual gpm is a step function. Assign a segment id that increments
# at each new filing value; distinct annual filings in a trailing w-month window =
# seg_id[t] - seg_id[t-(w-1)] + 1 (exact, since seg_id is non-decreasing per stock).
mf["newfile"] = (mf["gpm"] != mf.groupby("stock_id")["gpm"].shift(1)).astype(int)
mf["seg_id"] = mf.groupby("stock_id")["newfile"].cumsum()

windows = [24, 36, 48, 60, 72]
for label, tset in [("MONTHLY / S&P panel", sp_tickers),
                    ("QUARTERLY / full-universe panel", fu_tickers)]:
    sub = mf[mf["stock_id"].isin(tset)].copy()
    print(f"  {label}:")
    gseg = sub.groupby("stock_id")["seg_id"]
    for w in windows:
        distinct = sub["seg_id"] - gseg.shift(w - 1) + 1  # NaN until w obs of same stock
        vals = distinct.dropna()
        print(f"    w={w:>2}m: mean distinct filings={vals.mean():.2f}, "
              f"median={vals.median():.0f}  (n_obs={len(vals):,})")
