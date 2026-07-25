"""R24_coldproc_sp500.py — R24 code-path integrity check, SP500 arm ONLY.

Run as a fully separate `python` invocation from the full-universe arm
(R24_coldproc_fulluniv.py): no shared process, no import of a common module
that could hold cached state, no pickled intermediates. Every array below is
built from the raw parquet on disk in THIS process.

Spec:  ret_next_month ~ 1 + dH_gpm_z + DS_z + T*DS_z
       SE: two-way cluster (firm x date), Wald test on the T*DS_z coefficient.

Writes results/revision/R24_sp500_cold.json for the cross-process comparison.
"""
import os, sys, json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import chi2

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/R24_sp500_cold.json"


def cs_winsorize_zscore(df, col, date_col="date", pct=0.01):
    """Cross-sectional winsorize (1%/99%) then z-score, within each date."""
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


def cluster_vcov(X, resid, codes):
    n_, k_ = X.shape
    inv = np.linalg.pinv(X.T @ X)
    B, G = cluster_meat(X, resid, codes)
    sc = (G / (G - 1)) * ((n_ - 1) / (n_ - k_))
    return sc * inv @ B @ inv


def twoway_vcov(X, resid, g_date, g_firm):
    inter = pd.Categorical(pd.Series(g_date).astype(str) + "_" + pd.Series(g_firm).astype(str)).codes
    return (cluster_vcov(X, resid, g_date)
            + cluster_vcov(X, resid, g_firm)
            - cluster_vcov(X, resid, inter))


def estimate_pooled_twoway(df, xcols, ycol, datecol, firmcol):
    """The single estimation function used by BOTH arms (pasted verbatim into
    both cold scripts — not imported from a shared module)."""
    d = df.dropna(subset=xcols + [ycol]).copy()
    X = np.column_stack([np.ones(len(d))] + [d[c].values.astype(float) for c in xcols])
    y = d[ycol].values.astype(float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    g_date = pd.Categorical(d[datecol].astype(str)).codes
    g_firm = pd.Categorical(d[firmcol].astype(str)).codes
    V = twoway_vcov(X, resid, g_date, g_firm)
    se = np.sqrt(np.diag(V))
    t = b / se
    p = 1 - chi2.cdf(t**2, 1)
    return dict(X=X, y=y, b=b, resid=resid, se=se, t=t, p=p, d=d,
                g_date=g_date, g_firm=g_firm)


if __name__ == "__main__":
    print(f"[pid={os.getpid()}] R24 SP500 arm — cold process, no shared state")
    m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
    m["date"] = pd.to_datetime(m["date"])
    m["dH_gpm_z"] = cs_winsorize_zscore(m, "dH_gpm")
    xcols = ["dH_gpm_z", "DS_z", "TxDS"]

    # === CALL SITE 1 (SP500) ===
    res = estimate_pooled_twoway(m, xcols, "ret_next_month", "date", "stock_id")

    X, resid, b, se, t, p, d = res["X"], res["resid"], res["b"], res["se"], res["t"], res["p"], res["d"]
    i_tx = 1 + xcols.index("TxDS")  # +1 for intercept col

    n_dates = d["date"].nunique()
    n_firms = d["stock_id"].nunique()
    design_hash = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()
    resid_hash = hashlib.sha1(np.ascontiguousarray(resid).tobytes()).hexdigest()

    print(f"design shape = {X.shape}, N = {len(d):,}")
    print(f"clusters: date={n_dates}, firm={n_firms}")
    print(f"coef_TxDS = {b[i_tx]:+.6f}  SE = {se[i_tx]:.6f}  t = {t[i_tx]:+.6f}  Wald p = {p[i_tx]:.6f}")
    print(f"design matrix SHA-1 = {design_hash}")
    print(f"residual SHA-1      = {resid_hash}")
    print(f"process id = {os.getpid()}, python object id of X = {id(X)}")

    out = dict(
        panel="SP500", pid=os.getpid(), shape=list(X.shape), N=int(len(d)),
        n_dates=int(n_dates), n_firms=int(n_firms),
        coef_TxDS=float(b[i_tx]), se_TxDS=float(se[i_tx]), t_TxDS=float(t[i_tx]), p_TxDS=float(p[i_tx]),
        design_hash=design_hash, resid_hash=resid_hash,
        resid_key=list(zip(d["stock_id"].astype(str).tolist(), d["date"].dt.to_period("Q").astype(str).tolist())),
        resid_vals=resid.tolist(),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {OUT}")
