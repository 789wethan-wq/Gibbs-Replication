"""01_variable_construction.py — Build ΔH, ΔS, T, ΔG at the portfolio level.

Per brief's portfolio-level test (25 FF portfolios, Size x B/M):
  ΔH = -1 x rolling 60m std of portfolio monthly return        (stability/enthalpy)
  ΔS = std of FF3 residuals from rolling 36m regression         (disorder/entropy)
  T  = 12m realized variance of market (normalized mean .04, std .02)
  ΔG = ΔH - T x ΔS,  cross-sectionally z-scored across 25 portfolios each month
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import winsorize

DATA = "data"
START, END = "1990-01-01", "2023-12-31"


def rolling_ff3_resid_std(y, X, window=36):
    """Rolling-window FF3 residual std for one portfolio's excess return series."""
    out = pd.Series(index=y.index, dtype=float)
    yv, Xv = y.values, X.values
    for i in range(window, len(y) + 1):
        ys, Xs = yv[i - window:i], Xv[i - window:i]
        if np.isnan(ys).any():
            continue
        Xs1 = np.column_stack([np.ones(window), Xs])
        try:
            beta, *_ = np.linalg.lstsq(Xs1, ys, rcond=None)
        except Exception:
            continue
        resid = ys - Xs1 @ beta
        out.iloc[i - 1] = resid.std(ddof=1)
    return out


def main():
    factors = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
    ff25 = pd.read_parquet(f"{DATA}/ff25_returns.parquet")
    T_raw = pd.read_parquet(f"{DATA}/market_temperature.parquet")["T_raw"]

    # align monthly index
    idx = ff25.index.intersection(factors.index)
    ff25 = ff25.loc[idx]
    factors = factors.loc[idx]
    ports = list(ff25.columns)

    # excess returns
    excess = ff25.sub(factors["RF"], axis=0)
    ff3 = factors[["Mkt_RF", "SMB", "HML"]]

    recs = []
    resid_std = {}
    for p in ports:
        resid_std[p] = rolling_ff3_resid_std(excess[p], ff3, window=36)
    resid_std = pd.DataFrame(resid_std)

    # ΔH: -1 * rolling 60m std of portfolio (total) monthly return
    DH = -1.0 * ff25.rolling(60).std()
    # ΔS: rolling FF3 residual std
    DS = resid_std

    # Temperature: align, normalize to mean 0.04 std 0.02
    T = T_raw.reindex(idx)
    T_norm = (T - T.mean()) / T.std() * 0.02 + 0.04

    # Build long panel
    long = []
    nextret = ff25.shift(-1)  # ret_{t+1}
    for p in ports:
        d = pd.DataFrame({
            "date": idx,
            "stock_id": p,
            "DH": DH[p].values,
            "DS_raw": DS[p].values,
            "ret": ff25[p].values,
            "ret_next_month": nextret[p].values,
            "excess_next": (nextret[p] - factors["RF"].shift(-1)).values,
        })
        long.append(d)
    panel = pd.concat(long, ignore_index=True)
    panel = panel.merge(T_norm.rename("T").reset_index().rename(columns={"index": "date"}),
                        on="date", how="left")
    panel = panel.merge(T.rename("T_raw").reset_index().rename(columns={"index": "date"}),
                        on="date", how="left")

    # Winsorize raw inputs each period, then cross-sectional z-score
    def zscore_period(col, src):
        w = panel.groupby("date")[src].transform(lambda s: winsorize(s))
        m = w.groupby(panel["date"]).transform("mean")
        sd = w.groupby(panel["date"]).transform("std")
        panel[col] = (w - m) / sd

    zscore_period("DH_z", "DH")
    zscore_period("DS_z", "DS_raw")

    # ΔG = ΔH_z - T * ΔS_z, then cross-sectionally z-score
    panel["DG_raw"] = panel["DH_z"] - panel["T"] * panel["DS_z"]
    g = panel.groupby("date")["DG_raw"]
    panel["DG"] = (panel["DG_raw"] - g.transform("mean")) / g.transform("std")
    # T*DS interaction term for constrained model
    panel["TxDS"] = panel["T"] * panel["DS_z"]

    # Restrict analysis window
    panel = panel[(panel["date"] >= START) & (panel["date"] <= END)].copy()
    panel = panel.dropna(subset=["DH_z", "DS_z", "DG", "ret_next_month"])

    panel.to_parquet(f"{DATA}/variables_monthly.parquet")

    corr = panel[["DH_z", "DS_z"]].corr().iloc[0, 1]
    print("Variable construction complete (portfolio-level, 25 FF portfolios).")
    print(f"  Analysis window: {panel['date'].min():%Y-%m} to {panel['date'].max():%Y-%m}, "
          f"{panel['date'].nunique()} months, {len(panel)} portfolio-months")
    print(f"  ΔH_z range: [{panel.DH_z.min():.2f}, {panel.DH_z.max():.2f}]  "
          f"ΔS_z range: [{panel.DS_z.min():.2f}, {panel.DS_z.max():.2f}]")
    print(f"  T range: [{panel['T'].min():.3f}, {panel['T'].max():.3f}]  "
          f"ΔG range: [{panel.DG.min():.2f}, {panel.DG.max():.2f}]")
    print(f"  Corr(ΔH, ΔS): {corr:.3f}")


if __name__ == "__main__":
    main()
