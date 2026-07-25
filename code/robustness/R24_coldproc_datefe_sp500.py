"""R24 supplementary check — date-FE row (+2.46 SP / +2.54 FU), SP500 arm ONLY,
run as a separate process from R24_coldproc_datefe_fulluniv.py."""
import os, json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import chi2

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/R24_datefe_sp500_cold.json"


def cs_winsorize_zscore(df, col, date_col="date", pct=0.01):
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


def cluster_meat(X, resid, codes):
    G = int(codes.max()) + 1
    Xr = X * resid[:, None]
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, codes, Xr)
    return S.T @ S, G


def cluster_vcov(X, resid, codes, n_params):
    n_, k_ = X.shape
    inv = np.linalg.pinv(X.T @ X)
    B, G = cluster_meat(X, resid, codes)
    sc = (G / (G - 1)) * ((n_ - 1) / (n_ - n_params))
    return sc * inv @ B @ inv


def twoway_vcov(X, resid, gd, gf, n_params):
    inter = pd.Categorical(pd.Series(gd).astype(str) + "_" + pd.Series(gf).astype(str)).codes
    return (cluster_vcov(X, resid, gd, n_params)
            + cluster_vcov(X, resid, gf, n_params)
            - cluster_vcov(X, resid, inter, n_params))


def date_fe_2way(df, xcols, ycol, datecol, firmcol):
    d = df.dropna(subset=xcols + [ycol]).copy()
    n = len(d)
    n_dates = d[datecol].nunique()
    Xd = np.column_stack([
        (d[c] - d.groupby(datecol)[c].transform("mean")).values.astype(float)
        for c in xcols])
    yd = (d[ycol] - d.groupby(datecol)[ycol].transform("mean")).values.astype(float)
    b, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    r = yd - Xd @ b
    gd = pd.Categorical(d[datecol].astype(str)).codes
    gf = pd.Categorical(d[firmcol].astype(str)).codes
    n_params = Xd.shape[1] + n_dates
    V = twoway_vcov(Xd, r, gd, gf, n_params)
    se = np.sqrt(np.diag(V))
    t = b / se
    p = 1 - chi2.cdf(t**2, 1)
    return dict(X=Xd, resid=r, b=b, se=se, t=t, p=p, d=d, n_dates=n_dates)


if __name__ == "__main__":
    print(f"[pid={os.getpid()}] R24 date-FE SP500 arm — cold process")
    m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
    m["date"] = pd.to_datetime(m["date"])
    m["__dH__"] = cs_winsorize_zscore(m, "dH_gpm")
    m["__DS__"] = m["DS_z"]
    m["__TxDS__"] = m["TxDS"]
    cols = ["__DS__", "__TxDS__", "__dH__"]
    res = date_fe_2way(m, cols, "ret_next_month", "date", "stock_id")
    i_tx = cols.index("__TxDS__")
    X, resid, d = res["X"], res["resid"], res["d"]
    design_hash = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()
    print(f"design(within-date) shape={X.shape} N={len(d):,} dates={res['n_dates']} firms={d['stock_id'].nunique()}")
    print(f"coef_TxDS={res['b'][i_tx]:+.6f} t={res['t'][i_tx]:+.6f} p={res['p'][i_tx]:.6f}")
    print(f"design SHA-1={design_hash}  pid={os.getpid()}  id(X)={id(X)}")
    out = dict(panel="SP500-dateFE", pid=os.getpid(), shape=list(X.shape), N=int(len(d)),
               coef_TxDS=float(res['b'][i_tx]), t_TxDS=float(res['t'][i_tx]), p_TxDS=float(res['p'][i_tx]),
               design_hash=design_hash)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
