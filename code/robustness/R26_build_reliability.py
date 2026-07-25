"""R26 step 0 — rebuild the per-observation split-half ΔS records (same
construction as D2_corrected_split_half.py) and additionally compute a
PER-FIRM reliability score (not a per-size-decile score): for each ticker,
correlate its own time series of (ds_odd, ds_even) split-half estimates
across all its full-12-quarter-window observations, Spearman-Brown correct.

This lets R26 stratify directly on measured reliability (a firm-level
property) rather than on size (a proxy for it, per the R26 spec).

Saves:
  data/R26_split_half_obs.parquet       (ticker, q, ds_odd, ds_even, marketcap)
  data/R26_firm_reliability.parquet     (ticker, n_obs, corr, reliability)
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"

print("Rebuilding quarterly return series (same construction as D2)...")
tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"]
uni = sf1t[
    sf1t["category"].str.contains("Domestic Common", na=False) &
    sf1t["exchange"].isin(["NYSE", "NASDAQ", "NYSEARCA", "BATS", "NYSEMKT"]) &
    (sf1t["currency"] == "USD")
].copy()
uni_set = set(uni["ticker"])

sf1p = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                        columns=["ticker", "dimension", "calendardate", "datekey", "price", "dps", "marketcap"])
arqp = sf1p[(sf1p["dimension"] == "ARQ") & sf1p["ticker"].isin(uni_set)].copy()
arqp["calendardate"] = pd.to_datetime(arqp["calendardate"], errors="coerce")
arqp = arqp.dropna(subset=["calendardate", "price"])
arqp = arqp[arqp["price"] > 0]
arqp["q"] = arqp["calendardate"].dt.to_period("Q")
arqp = arqp.sort_values(["ticker", "calendardate"]).drop_duplicates(["ticker", "q"], keep="last")
arqp = arqp.sort_values(["ticker", "q"])
arqp["q_ord"] = arqp["q"].apply(lambda p: p.ordinal)
arqp["price_prev"] = arqp.groupby("ticker")["price"].shift(1)
arqp["gap"] = arqp["q_ord"] - arqp.groupby("ticker")["q_ord"].shift(1)
ret_px = arqp["price"] / arqp["price_prev"] - 1.0
div_q = (arqp["dps"].fillna(0) / 4.0) / arqp["price_prev"]
arqp["ret_full"] = np.where(arqp["gap"] == 1, ret_px + div_q, np.nan)
arqp["ret_full"] = arqp.groupby("q")["ret_full"].transform(
    lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))

fac = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
fac.index = pd.to_datetime(fac.index)
facq = fac.copy()
facq["q"] = facq.index.to_period("Q")
def cmpd(s): return (1 + s).prod() - 1
ffq = facq.groupby("q").agg({"Mkt_RF": cmpd, "SMB": cmpd, "HML": cmpd, "RF": cmpd}).reset_index()
arqp = arqp.merge(ffq, on="q", how="left")
arqp["exret_full"] = arqp["ret_full"] - arqp["RF"]
arqp = arqp.dropna(subset=["ret_full"])
cnt = arqp.groupby("ticker")["ret_full"].transform("size")
arqp = arqp[cnt >= 8].sort_values(["ticker", "q_ord"]).reset_index(drop=True)
print(f"Quarterly return obs (>=8/ticker): {len(arqp):,} | tickers: {arqp['ticker'].nunique():,}")

FACTORS = ["Mkt_RF", "SMB", "HML"]
def half_ivol(sub):
    if len(sub) < 5:
        return np.nan
    y = sub["exret_full"].values
    X = sm.add_constant(sub[FACTORS].values, has_constant="add")
    if np.isnan(X).any() or np.isnan(y).any():
        return np.nan
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return resid.std(ddof=1)

print("Computing split-half ΔS on full 12-quarter windows...")
records = []
tickers_all = arqp["ticker"].unique()
for ti, tkr in enumerate(tickers_all):
    if ti % 3000 == 0:
        print(f"  {ti}/{len(tickers_all)}...")
    g = arqp[arqp["ticker"] == tkr].sort_values("q_ord").reset_index(drop=True)
    n = len(g)
    for i in range(8, n + 1):
        lo = max(0, i - 12)
        window = g.iloc[lo:i]
        if len(window) != 12:
            continue
        odd_half = window.iloc[1::2]
        even_half = window.iloc[0::2]
        ds_odd = half_ivol(odd_half)
        ds_even = half_ivol(even_half)
        if np.isnan(ds_odd) or np.isnan(ds_even):
            continue
        records.append({"ticker": tkr, "q": g.iloc[i - 1]["q"], "ds_odd": ds_odd, "ds_even": ds_even,
                         "marketcap": g.iloc[i - 1]["marketcap"]})

obs_df = pd.DataFrame(records)
print(f"Full-12-quarter-window observations: {len(obs_df):,}")
obs_df.to_parquet(f"{DATA}/R26_split_half_obs.parquet")

# ── per-firm reliability: correlate ds_odd/ds_even ACROSS TIME within ticker ──
print("\nComputing PER-FIRM reliability (within-ticker corr of ds_odd, ds_even over time)...")
rows = []
for tkr, g in obs_df.groupby("ticker"):
    n = len(g)
    if n < 8:   # need a minimally stable within-firm correlation
        continue
    r = g["ds_odd"].corr(g["ds_even"])
    if not np.isfinite(r):
        continue
    sb = 2 * r / (1 + r) if (1 + r) != 0 else np.nan
    rows.append({"ticker": tkr, "n_obs": n, "corr": r, "reliability": sb})
rel_df = pd.DataFrame(rows)
print(f"Firms with a computable per-firm reliability (>=8 own obs): {len(rel_df):,}")
print(rel_df["reliability"].describe())
rel_df.to_parquet(f"{DATA}/R26_firm_reliability.parquet")
print("\nSaved data/R26_split_half_obs.parquet and data/R26_firm_reliability.parquet")
