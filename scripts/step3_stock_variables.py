"""01b_stock_variables.py — Build ΔH, ΔS, T, ΔG at the individual stock level.

Uses price-based proxies (defensible given yfinance fundamentals only go back ~4yr):
  ΔH = -1 × rolling 60m std of monthly stock return   (return stability = enthalpy)
  ΔS = rolling 36m std of FF3 residuals                (idiosyncratic disorder = entropy)
  T  = 12m realized market variance (normalized)
  ΔG = ΔH_z - T × ΔS_z,  cross-sectionally z-scored each month

Identical construction logic to the portfolio-level test, now applied stock-by-stock.
"""
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import winsorize

DATA  = "../data"
START = "1995-01-01"   # allow 60m DH + 36m DS burn-in from 1988 download
END   = "2023-12-31"
DH_WINDOW = 60   # months for return stability
DS_WINDOW = 36   # months for FF3 residual vol


def rolling_ff3_resid_std(excess_ret, ff3, window=36):
    """Rolling FF3 residual std for one stock. Returns monthly series."""
    y  = excess_ret.values
    Xv = ff3.values
    n  = len(y)
    out = np.full(n, np.nan)
    for i in range(window, n + 1):
        ys = y[i - window:i]
        Xs = Xv[i - window:i]
        if np.isnan(ys).any() or np.isnan(Xs).any():
            continue
        Xs1 = np.column_stack([np.ones(window), Xs])
        try:
            beta, *_ = np.linalg.lstsq(Xs1, ys, rcond=None)
        except Exception:
            continue
        resid = ys - Xs1 @ beta
        out[i - 1] = resid.std(ddof=1)
    return pd.Series(out, index=excess_ret.index)


def main():
    print("=== STOCK-LEVEL VARIABLE CONSTRUCTION ===\n")

    prices  = pd.read_parquet(f"{DATA}/stock_prices_monthly.parquet")
    factors = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
    T_raw   = pd.read_parquet(f"{DATA}/market_temperature.parquet")["T_raw"]

    prices.index  = pd.to_datetime(prices.index)
    factors.index = pd.to_datetime(factors.index)
    T_raw.index   = pd.to_datetime(T_raw.index)

    # Common monthly index
    monthly_idx = prices.index.intersection(factors.index).intersection(T_raw.index)
    monthly_idx = monthly_idx.sort_values()

    prices  = prices.reindex(monthly_idx)
    factors = factors.reindex(monthly_idx)
    T_raw_m = T_raw.reindex(monthly_idx)

    rf  = factors["RF"]
    ff3 = factors[["Mkt_RF", "SMB", "HML"]]

    # Normalize temperature
    T_norm = (T_raw_m - T_raw_m.mean()) / T_raw_m.std() * 0.02 + 0.04

    tickers = prices.columns.tolist()
    print(f"Processing {len(tickers)} stocks × {len(monthly_idx)} months...\n")

    # Pre-compute ΔH (rolling 60m return std) and ΔS (rolling 36m FF3 resid std) for all stocks
    # ΔH: straightforward rolling std on the price return series
    print("Computing ΔH (return stability)...")
    ret_all  = prices.pct_change()
    DH_all   = -1.0 * ret_all.rolling(DH_WINDOW, min_periods=DH_WINDOW).std()

    # ΔS: per-stock FF3 rolling residual std (vectorised loop over stocks)
    print("Computing ΔS (idiosyncratic vol) — this takes ~2-3 min...")
    excess_all = ret_all.sub(rf, axis=0)
    DS_dict = {}
    for i, tkr in enumerate(tickers):
        if i % 100 == 0:
            print(f"  {i}/{len(tickers)}...")
        ex = excess_all[tkr].dropna()
        if len(ex) < DS_WINDOW + 5:
            continue
        ex = ex.reindex(monthly_idx)
        DS_dict[tkr] = rolling_ff3_resid_std(ex, ff3, window=DS_WINDOW)
    DS_all = pd.DataFrame(DS_dict)

    # ── Build long panel ─────────────────────────────────────────────────────
    print("\nBuilding long panel...")
    ret_next = ret_all.shift(-1)

    DH_stack      = DH_all.stack(future_stack=True).rename("DH")
    DS_stack      = DS_all.reindex(monthly_idx).stack(future_stack=True).rename("DS_raw")
    ret_stack     = ret_all.stack(future_stack=True).rename("ret")
    ret_next_stack = ret_next.stack(future_stack=True).rename("ret_next_month")

    panel = pd.concat([DH_stack, DS_stack, ret_stack, ret_next_stack], axis=1)
    panel = panel.reset_index()
    panel.columns = ["date", "stock_id", "DH", "DS_raw", "ret", "ret_next_month"]

    panel = panel.merge(
        T_norm.rename("T").reset_index().rename(columns={"index":"date"}), on="date", how="left"
    )
    panel = panel.merge(
        T_raw_m.rename("T_raw").reset_index().rename(columns={"index":"date"}), on="date", how="left"
    )

    panel = panel.dropna(subset=["DH", "DS_raw", "T", "ret_next_month"])
    panel = panel[(panel["date"] >= START) & (panel["date"] <= END)].copy()

    # ── Winsorize + cross-sectional z-score ─────────────────────────────────
    def winsorize_zscore(df, src, out_col):
        w  = df.groupby("date")[src].transform(lambda s: winsorize(s))
        m  = w.groupby(df["date"]).transform("mean")
        sd = w.groupby(df["date"]).transform("std")
        df[out_col] = (w - m) / sd

    winsorize_zscore(panel, "DH",     "DH_z")
    winsorize_zscore(panel, "DS_raw", "DS_z")

    # ── ΔG = ΔH_z - T × ΔS_z, z-scored ─────────────────────────────────────
    panel["DG_raw"] = panel["DH_z"] - panel["T"] * panel["DS_z"]
    g = panel.groupby("date")["DG_raw"]
    panel["DG"]   = (panel["DG_raw"] - g.transform("mean")) / g.transform("std")
    panel["TxDS"] = panel["T"] * panel["DS_z"]

    panel.to_parquet(f"{DATA}/variables_stock_monthly.parquet")

    corr      = panel[["DH_z","DS_z"]].corr().iloc[0,1]
    n_stocks  = panel["stock_id"].nunique()
    n_months  = panel["date"].nunique()
    avg_n     = len(panel) / n_months

    print(f"\nVariable construction complete (stock-level, price-based proxies).")
    print(f"  Window:      {panel['date'].min():%Y-%m} to {panel['date'].max():%Y-%m}")
    print(f"  Stocks:      {n_stocks}")
    print(f"  Months:      {n_months}")
    print(f"  Stock-months:{len(panel):,}")
    print(f"  Avg N/month: {avg_n:.0f}")
    print(f"  ΔH range:    [{panel.DH_z.min():.2f}, {panel.DH_z.max():.2f}]")
    print(f"  ΔS range:    [{panel.DS_z.min():.2f}, {panel.DS_z.max():.2f}]")
    print(f"  T range:     [{panel['T'].min():.3f}, {panel['T'].max():.3f}]")
    print(f"  ΔG range:    [{panel.DG.min():.2f}, {panel.DG.max():.2f}]")
    print(f"  Corr(ΔH,ΔS): {corr:.3f}")
    print(f"\n*** Price-based proxies: ΔH=return stability, ΔS=idiosyncratic vol ***")
    print(f"*** Survivorship bias: current S&P 500 constituents only ***")


if __name__ == "__main__":
    main()
