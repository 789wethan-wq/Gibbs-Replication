"""R24 supplementary check — date-FE row (+2.46 SP / +2.54 FU), full-universe
arm ONLY, run as a separate process from R24_coldproc_datefe_sp500.py."""
import os, json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import chi2

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/R24_datefe_fulluniv_cold.json"


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
    print(f"[pid={os.getpid()}] R24 date-FE full-universe arm — cold process")
    q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
    q["__DS__"] = q["delta_s_z"]
    q["__TxDS__"] = q["T_delta_s"]
    q["__dH__"] = q["delta_h_z"]
    cols = ["__DS__", "__TxDS__", "__dH__"]
    res = date_fe_2way(q, cols, "ret_next", "q", "ticker")
    i_tx = cols.index("__TxDS__")
    X, resid, d = res["X"], res["resid"], res["d"]
    design_hash = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()
    print(f"design(within-date) shape={X.shape} N={len(d):,} dates={res['n_dates']} firms={d['ticker'].nunique()}")
    print(f"coef_TxDS={res['b'][i_tx]:+.6f} t={res['t'][i_tx]:+.6f} p={res['p'][i_tx]:.6f}")
    print(f"design SHA-1={design_hash}  pid={os.getpid()}  id(X)={id(X)}")
    out = dict(panel="FullUniv-dateFE", pid=os.getpid(), shape=list(X.shape), N=int(len(d)),
               coef_TxDS=float(res['b'][i_tx]), t_TxDS=float(res['t'][i_tx]), p_TxDS=float(res['p'][i_tx]),
               design_hash=design_hash)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
