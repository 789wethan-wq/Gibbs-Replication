"""MC_LOCK.py — Block 1: Model C lock on the patched committed rig.

Model C = dH_gpm_z + TxDS (no raw ΔS), accounting construction, primary cut,
run through the SAME patched rig as Model B (project/utils.fama_macbeth, min
cross-section cut = len(xcols)+2). Prints lag 0/4/5/6 ladder for both terms,
N, T, and the pooled design-matrix rank + condition number.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "../project"))
DATA = os.path.join(ROOT, "../data")

from utils import newey_west_mean_tstat
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
    dates = sorted(panel["date"].unique())
    rows, n_used = [], 0
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
    return pd.DataFrame(rows), n_used, len(rows)


def main():
    m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
    m["date"] = pd.to_datetime(m["date"])
    m["dH_gpm_z"] = cs_wz(m, "dH_gpm")

    xcols = ["dH_gpm_z", "TxDS"]
    prim = m.dropna(subset=["ret_next_month"] + xcols).copy()

    print("=" * 72)
    print("BLOCK 1 — Model C (accounting dH_gpm_z + T·ΔS, no raw ΔS), primary cut")
    print("=" * 72)

    cut = len(xcols) + 2  # patched primary rig, matches Model B re-anchor
    cdf, N, T = coef_series(prim, "ret_next_month", xcols, min_cut=cut)
    for lag in [0, 4, 5, 6]:
        m_dh, t_dh, _ = newey_west_mean_tstat(cdf["dH_gpm_z"].values, lags=lag)
        m_tx, t_tx, _ = newey_west_mean_tstat(cdf["TxDS"].values, lags=lag)
        tag = "lag=0" if lag == 0 else f"lag={lag}"
        print(f"[MC-LOCK] {tag}: t(dH)={t_dh:+.2f}, t(TxdS)={t_tx:+.2f}, "
              f"coef_dH={m_dh:+.5f}, coef_TxdS={m_tx:+.4f}")

    # Pooled design-matrix rank + condition number: [const, dH_gpm_z, TxDS]
    Xp = sm.add_constant(prim[xcols]).values
    rank = np.linalg.matrix_rank(Xp)
    sv = np.linalg.svd(Xp, compute_uv=False)
    cond = sv[0] / sv[-1]
    print(f"[MC-LOCK] N={N:,}, T={T}, rank={rank}/{Xp.shape[1]}, "
          f"cond={cond:.1f}  (full rank = {rank == Xp.shape[1]})")
    print("[MC-LOCK] MANUSCRIPT Table 2 Panel A Model C: dH=+0.0010 (+2.61), "
          "T·ΔS=+0.121 (+4.59)")


if __name__ == "__main__":
    main()
