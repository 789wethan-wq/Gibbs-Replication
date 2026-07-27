"""T2_LOCK.py — Block 1b/1c: lag-provenance ladder for Table 2 Model B (accounting).

Runs the PATCHED primary rig (project/utils.fama_macbeth, min cross-section cut =
len(xcols)+2) on the accounting Model B (dH_gpm_z + DS_z), primary cut, and prints
the plain-FM Newey-West t-stats at lags 0/4/5/6 for BOTH ΔH and ΔS.

For provenance it also prints the SAME ladder on the R17/R3-3 rig (min cut = 15),
so any lag-0 divergence is attributable to the sample-definition diff, not relabeling.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "../project"))
DATA = os.path.join(ROOT, "../data")

from utils import newey_west_mean_tstat  # the patched primary rig's NW routine
import statsmodels.api as sm


def cs_wz(df, col, date_col="date", pct=0.01):
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


def coef_series(panel, ret_col, xcols, min_cut):
    """Cross-sectional OLS each date; return DataFrame of coef series + (N, T)."""
    dates = sorted(panel["date"].unique())
    rows = []
    n_used = 0
    for d in dates:
        sub = panel[panel["date"] == d].dropna(subset=[ret_col] + xcols)
        if len(sub) < min_cut:
            continue
        X = sm.add_constant(sub[xcols])
        try:
            res = sm.OLS(sub[ret_col], X).fit()
        except Exception:
            continue
        rows.append(pd.Series(res.params, name=d))
        n_used += len(sub)
    cdf = pd.DataFrame(rows)
    return cdf, n_used, len(cdf)


def ladder(tag, panel, ret_col, xcols, min_cut):
    cdf, N, T = coef_series(panel, ret_col, xcols, min_cut)
    print(f"\n[T2-LOCK] {tag}  (min-obs cut={min_cut})")
    for lag in [0, 4, 5, 6]:
        _, t_dh, _ = newey_west_mean_tstat(cdf[xcols[0]].values, lags=lag)
        _, t_ds, _ = newey_west_mean_tstat(cdf[xcols[1]].values, lags=lag)
        print(f"  lag={lag}: t(dH)={t_dh:+.2f}, t(dS)={t_ds:+.2f}")
    print(f"  N={N:,}, T={T}, min-obs cut={min_cut}")
    return cdf


def main():
    m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
    m["date"] = pd.to_datetime(m["date"])
    m["dH_gpm_z"] = cs_wz(m, "dH_gpm")

    xcols = ["dH_gpm_z", "DS_z"]
    # Primary cut = the patched 04_fama_macbeth.py sample: dropna on ret+xcols.
    prim = m.dropna(subset=["ret_next_month"] + xcols).copy()

    print("="*72)
    print("BLOCK 1b/1c — Table 2 Model B (accounting dH_gpm_z + DS_z), primary cut")
    print("="*72)

    # PATCHED PRIMARY RIG: utils.fama_macbeth min cut = len(xcols)+2 = 4
    ladder("patched primary rig (utils.fama_macbeth, cut=len+2)",
           prim, "ret_next_month", xcols, min_cut=len(xcols) + 2)

    # R17/R3-3 rig: min cut = 15 (fm_nw uses max(15, len+2))
    ladder("R17/R3-3 verification rig (fm_nw, cut=15)",
           prim, "ret_next_month", xcols, min_cut=15)

    print("\nReference (R3-3 lag sweep, cut=15): t(dH)=+2.45@lag0, +2.70@lag5, +2.77@lag6")


if __name__ == "__main__":
    main()
