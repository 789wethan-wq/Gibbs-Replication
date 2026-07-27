"""R3_date_fe.py — R3 item 1: add date fixed effects to the IDENTIFIED pooled
two-way-clustered interaction spec, both panels.

    r ~ ΔS_z + T·ΔS_z + ΔH_z + C(date),  SE two-way clustered (firm × date)

Date FE absorbed by within-date demeaning (Frisch–Waugh–Lovell): the slope
coefficients are numerically identical to including an explicit date-dummy block,
but the design stays small. FF5+UMD dropped — date FE absorb all date-level
variation (factor returns, T's main effect) and make them redundant/collinear.

T·ΔS_z stays identified: ΔS_z is z-scored WITHIN month (mean 0 per date), so its
cross-firm variation survives the within-date transform, and T·ΔS_z = T_t·ΔS_z
varies cross-sectionally within each date. DIAGNOSTIC — no manuscript text.
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

def _meat(X, resid, codes):
    """Cluster meat B = sum_g (X_g' r_g)(X_g' r_g)', vectorized via grouped sums.
    codes = integer group labels 0..G-1."""
    G = int(codes.max()) + 1
    Xr = X * resid[:, None]
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, codes, Xr)
    return S.T @ S, G

def cluster_vcov(X, resid, codes, n_params):
    """Cluster-robust vcov with finite-sample scale. n_params = effective # estimated
    params (incl. absorbed date FE)."""
    n_, k_ = X.shape; inv = np.linalg.pinv(X.T @ X)
    B, G = _meat(X, resid, codes)
    sc = (G / (G - 1)) * ((n_ - 1) / (n_ - n_params))
    return sc * inv @ B @ inv

def twoway_vcov(X, resid, gd, gf, n_params):
    # firm×date interaction: in a balanced panel each (firm,date) is unique, so this
    # meat reduces to X'diag(r^2)X; the grouped-sum form handles it either way.
    inter = pd.Categorical(pd.Series(gd).astype(str) + "_" + pd.Series(gf).astype(str)).codes
    return (cluster_vcov(X, resid, gd, n_params)
            + cluster_vcov(X, resid, gf, n_params)
            - cluster_vcov(X, resid, inter, n_params))

def date_fe_2way(df, xcols, ycol, datecol, firmcol):
    """Within-date-demeaned OLS (absorbs date FE), two-way (firm×date) clustered."""
    d = df.dropna(subset=xcols + [ycol]).copy()
    n = len(d)
    n_dates = d[datecol].nunique()
    # within-date demeaning (FWL) — absorbs C(date), incl. T main effect & factor returns
    Xd = np.column_stack([
        (d[c] - d.groupby(datecol)[c].transform("mean")).values.astype(float)
        for c in xcols])
    yd = (d[ycol] - d.groupby(datecol)[ycol].transform("mean")).values.astype(float)
    # rank check of the within-transformed design
    rank = np.linalg.matrix_rank(Xd)
    cond = np.linalg.cond(Xd)
    corr = np.corrcoef(Xd[:, xcols.index("__DS__")], Xd[:, xcols.index("__TxDS__")])[0, 1]
    b, *_ = np.linalg.lstsq(Xd, yd, rcond=None); r = yd - Xd @ b
    gd = pd.Categorical(d[datecol].astype(str)).codes
    gf = pd.Categorical(d[firmcol].astype(str)).codes
    n_params = Xd.shape[1] + n_dates            # regressors + absorbed date dummies
    V = twoway_vcov(Xd, r, gd, gf, n_params)
    se = np.sqrt(np.diag(V)); t = b / se
    p = 1 - chi2.cdf(t**2, 1)
    return dict(coef=b, t=t, p=p, N=n, n_dates=n_dates, n_firms=d[firmcol].nunique(),
                rank=rank, ncol=Xd.shape[1], cond=cond, corr=corr)

def report(tag, res, names):
    i_dS = names.index("__DS__"); i_Tx = names.index("__TxDS__"); i_dH = names.index("__dH__")
    print(f"[R3-1 | {tag} | date-FE] "
          f"coef_TxdS={res['coef'][i_Tx]:+.6f}, t={res['t'][i_Tx]:+.4f}, p={res['p'][i_Tx]:.4f}, "
          f"coef_dS={res['coef'][i_dS]:+.6f}, t_dS={res['t'][i_dS]:+.4f}, N={res['N']:,}")
    print(f"        coef_dH={res['coef'][i_dH]:+.6f}, t_dH={res['t'][i_dH]:+.4f}  |  "
          f"dates={res['n_dates']}, firms={res['n_firms']}")
    print(f"        within-design rank={res['rank']}/{res['ncol']}, cond={res['cond']:.2f}, "
          f"corr(ΔS,T·ΔS)_within={res['corr']:+.4f}")

# ── S&P 500 monthly panel ────────────────────────────────────────────────────
print("="*78)
m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
m["date"] = pd.to_datetime(m["date"])
m["__dH__"] = cs_wz(m, "dH_gpm")     # exactly as R16/PanelC load_base()
m["__DS__"] = m["DS_z"]
m["__TxDS__"] = m["TxDS"]            # = T * DS_z (verified)
cols = ["__DS__", "__TxDS__", "__dH__"]
res_sp = date_fe_2way(m, cols, "ret_next_month", "date", "stock_id")
report("SP500", res_sp, cols)

# ── R18 full-universe quarterly panel ────────────────────────────────────────
print("-"*78)
q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
q["__DS__"] = q["delta_s_z"]
q["__TxDS__"] = q["T_delta_s"]
q["__dH__"] = q["delta_h_z"]
res_fu = date_fe_2way(q, cols, "ret_next", "q", "ticker")
report("FullU", res_fu, cols)
print("="*78)
