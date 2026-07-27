"""E4 — Temperature normalization: date-boundary audit.

Traces T's construction through the ACTUAL saved pipeline outputs (not
config constants): daily market-return series -> T_raw (252-trading-day
realized variance) -> T (normalized) -> the monthly stock panel -> the FM
estimation sample. Reports exact date boundaries and observation counts at
every stage, and states plainly whether the normalization is an expanding
window (as hypothesized in the spec) or something else (found below: it is
NOT an expanding window).
"""
import os
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/E4_temperature_audit.txt"

log = []
def P(s=""):
    print(s)
    log.append(str(s))

print(f"[pid={os.getpid()}] E4 — fresh process")

P("="*88)
P("E4 — Temperature normalization date-boundary audit")
P("="*88)

# ── (1) daily series building the 12-month realized variance ───────────────
P("\n" + "-"*88)
P("(1) Daily series for the 12-month (252-trading-day) realized variance")
P("-"*88)
daily = pd.read_parquet(f"{DATA}/sp500_daily.parquet")
daily.index = pd.to_datetime(daily.index)
P(f"sp500_daily.parquet (built from Ken French FF5-daily Mkt_RF+RF, per data_pipeline.py):")
P(f"  first date = {daily.index.min().date()}   last date = {daily.index.max().date()}   N = {len(daily):,} trading days")

logret = daily["log_ret"]
rv = logret.pow(2).rolling(252).sum()
first_rv_valid = rv.dropna().index.min()
P(f"  rolling(252).sum() of squared daily log returns: first NON-NULL value at {first_rv_valid.date()} "
  f"(252 trading days after the daily series starts, i.e. burn-in of {252} obs, "
  f"~{(first_rv_valid - daily.index.min()).days} calendar days)")

T_raw_full = pd.read_parquet(f"{DATA}/market_temperature.parquet")["T_raw"]
T_raw_full.index = pd.to_datetime(T_raw_full.index)
P(f"\nmarket_temperature.parquet (T_raw, resample('ME').last() of the above):")
P(f"  first month = {T_raw_full.index.min().strftime('%Y-%m')}   last month = {T_raw_full.index.max().strftime('%Y-%m')}   "
  f"N = {len(T_raw_full)} months")
P(f"  -> T_raw (and therefore T, however normalized) is defined from "
  f"{T_raw_full.index.min().strftime('%Y-%m')}, which predates the equity panel's 1995 start by "
  f"{(pd.Timestamp('1995-01-01') - T_raw_full.index.min()).days // 365} years.")

# ── (2) the normalization step itself ───────────────────────────────────────
P("\n" + "-"*88)
P("(2) T normalization -- what the code actually does (project/01b_stock_variables.py)")
P("-"*88)
prices = pd.read_parquet(f"{DATA}/stock_prices_monthly.parquet")
prices.index = pd.to_datetime(prices.index)
factors = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
factors.index = pd.to_datetime(factors.index)
monthly_idx = prices.index.intersection(factors.index).intersection(T_raw_full.index).sort_values()
T_raw_m = T_raw_full.reindex(monthly_idx)

P(f"prices.index (stock_prices_monthly.parquet): {prices.index.min().strftime('%Y-%m')} .. "
  f"{prices.index.max().strftime('%Y-%m')}  (N={len(prices)})")
P(f"factors.index (factors_monthly.parquet):     {factors.index.min().strftime('%Y-%m')} .. "
  f"{factors.index.max().strftime('%Y-%m')}  (N={len(factors)})")
P(f"monthly_idx = intersection(prices, factors, T_raw): {monthly_idx.min().strftime('%Y-%m')} .. "
  f"{monthly_idx.max().strftime('%Y-%m')}  (N={len(monthly_idx)})")
P(f"\nCODE (verbatim, project/01b_stock_variables.py):")
P(f'    T_norm = (T_raw_m - T_raw_m.mean()) / T_raw_m.std() * 0.02 + 0.04')
P(f"\nThis is a FULL-SAMPLE rescale: T_raw_m.mean() and T_raw_m.std() are each a SINGLE")
P(f"scalar computed ONCE over the ENTIRE monthly_idx span above "
  f"({monthly_idx.min().strftime('%Y-%m')}..{monthly_idx.max().strftime('%Y-%m')}, N={len(monthly_idx)} months),")
P(f"applied identically to every month. THIS IS NOT AN EXPANDING WINDOW. There is no")
P(f"24-month minimum, no expanding accumulation, and no month-by-month recomputation")
P(f"anywhere in this normalization step -- confirmed by reading the code, not inferred.")
P(f"mean(T_raw_m) = {T_raw_m.mean():.6f}   std(T_raw_m) = {T_raw_m.std():.6f}")
P(f"\nConsequence: EVERY month's normalized T value (not just the early ones) uses")
P(f"information from the full sample, including months AFTER that observation --")
P(f"this is look-ahead for the entire series, not merely an early-sample burn-in")
P(f"issue as an expanding-window design would produce. This is a DIFFERENT and MORE")
P(f"SEVERE issue than the one hypothesized in the spec (which assumed an expanding")
P(f"window with a binding 24-month minimum -- no such construction exists in this")
P(f"codebase for T; searched project/*.py and robustness/*.py for 'expanding' in the")
P(f"context of T and found none).")
P(f"\nNote for calibration: because T only enters as a STANDARDIZED (near-linear)")
P(f"rescaling of T_raw applied UNIFORMLY across the sample, and the FM regressions of")
P(f"interest condition on T contemporaneously within a given month (not on the")
P(f"normalization constants themselves), the practical channel for bias is mainly")
P(f"through the T-interaction/Markov-regime classification analyses (Table 4/5) that")
P(f"use T's LEVEL to split or weight the sample -- exactly the look-ahead concern the")
P(f"filtered-vs-smoothed Markov probability finding (prior round, DOC1) already")
P(f"flagged from a different angle. A linear rescale by fixed constants does not, by")
P(f"itself, change any FM t-statistic that regresses returns on T_z or T*ΔS (since")
P(f"OLS is invariant to a fixed affine transform of a regressor) -- but it DOES matter")
P(f"for any split/threshold defined on T's level (regime classification, high/low-T")
P(f"subsample tests), which use the full-sample mean/std as their reference point.")

# ── (3) exact accounting for the panel's dropped observations ──────────────
P("\n" + "-"*88)
P("(3) Panel date range and the exact accounting for dropped observations")
P("-"*88)
varm = pd.read_parquet(f"{DATA}/variables_monthly.parquet")
varm["date"] = pd.to_datetime(varm["date"])
vm_months = sorted(varm["date"].unique())
P(f"variables_monthly.parquet: {vm_months[0].strftime('%Y-%m')} .. {vm_months[-1].strftime('%Y-%m')}  "
  f"(N={len(vm_months)} distinct months)")

m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
m["date"] = pd.to_datetime(m["date"])
ma_months = sorted(m["date"].unique())
P(f"merged_with_accounting.parquet (headline panel): {ma_months[0].strftime('%Y-%m')} .. "
  f"{ma_months[-1].strftime('%Y-%m')}  (N={len(ma_months)} distinct months)")

# what the primary FM rig (T2_LOCK/MC_LOCK convention) actually uses
def cs_wz(df, col, date_col="date", pct=0.01):
    def _f(x):
        x2 = x.dropna()
        if len(x2) < 5: return pd.Series(np.nan, index=x.index)
        lo, hi = x2.quantile(pct), x2.quantile(1 - pct)
        xc = x.clip(lo, hi); s = xc.std()
        if s < 1e-10: return pd.Series(np.nan, index=x.index)
        return (xc - xc.mean()) / s
    return df.groupby(date_col)[col].transform(_f)

m["dH_gpm_z"] = cs_wz(m, "dH_gpm")
prim = m.dropna(subset=["ret_next_month", "dH_gpm_z", "DS_z"])
fm_months = sorted(prim["date"].unique())
P(f"FM estimation sample (dropna on ret_next_month, dH_gpm_z, DS_z, primary rig): "
  f"{fm_months[0].strftime('%Y-%m')} .. {fm_months[-1].strftime('%Y-%m')}  (N={len(fm_months)} months)")

dropped_from_varm = set(vm_months) - set(fm_months)
dropped_from_ma = set(ma_months) - set(fm_months)
P(f"\nMonths in variables_monthly.parquet but NOT in the FM sample: {len(dropped_from_varm)}")
if dropped_from_varm:
    P(f"  {sorted(pd.Timestamp(d).strftime('%Y-%m') for d in dropped_from_varm)}")
P(f"Months in merged_with_accounting.parquet but NOT in the FM sample: {len(dropped_from_ma)}")
if dropped_from_ma:
    P(f"  {sorted(pd.Timestamp(d).strftime('%Y-%m') for d in dropped_from_ma)}")

P(f"\nReconciling the spec's '347 months in the panel, 335 used' claim against what")
P(f"is actually on disk:")
P(f"  variables_monthly.parquet has {len(vm_months)} months -- {'MATCHES' if len(vm_months)==347 else 'DOES NOT MATCH'} the cited 347")
P(f"  merged_with_accounting.parquet has {len(ma_months)} months -- {'MATCHES' if len(ma_months)==347 else 'DOES NOT MATCH'} the cited 347")
P(f"  FM sample has {len(fm_months)} months -- {'MATCHES' if len(fm_months)==335 else 'DOES NOT MATCH'} the cited 335")

# Check where dH_gpm (accounting) itself first becomes available -- this is
# likely the actual binding constraint for the 335 vs full-panel-months gap,
# since dH_gpm is an ACCOUNTING variable merged on later via sharadar_pipeline.py,
# not part of variables_monthly.parquet's own price-based DH burn-in.
dhg_avail = m.dropna(subset=["dH_gpm"])
dhg_months = sorted(dhg_avail["date"].unique())
P(f"\ndH_gpm (accounting ΔH, merged separately via sharadar_pipeline.py) is non-null "
  f"in merged_with_accounting.parquet from {dhg_months[0].strftime('%Y-%m')} "
  f"(N={len(dhg_months)} months with any non-null dH_gpm)")
missing_dhg_months = set(ma_months) - set(dhg_months)
P(f"Months in merged_with_accounting.parquet with NO non-null dH_gpm anywhere: "
  f"{len(missing_dhg_months)}")
if missing_dhg_months:
    P(f"  {sorted(pd.Timestamp(d).strftime('%Y-%m') for d in missing_dhg_months)}")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
