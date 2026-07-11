"""Regenerate everything in /derived-data from the raw panels.

Requires the private data/ directory (built by code/project/sharadar_pipeline.py
and code/project/data_pipeline.py from a Nasdaq Data Link Sharadar subscription
and the public factor libraries). All outputs are cross-sectional AGGREGATES
(regression coefficients, portfolio-mean returns) — no vendor rows are written.

Run from the repository root of the private working copy:
    python3 code/generate_derived_data.py
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

DATA = "data"                      # private raw-data dir (not distributed)
OUT = "public_release/derived-data" if os.path.isdir("public_release") else "derived-data"
os.makedirs(OUT, exist_ok=True)

def cs_wz(df, col, datecol, pct=0.01):
    def _w(x):
        x = x.dropna()
        if len(x) < 5: return pd.Series(np.nan, index=x.index)
        xc = x.clip(x.quantile(pct), x.quantile(1-pct)); sd = xc.std()
        if sd < 1e-10: return pd.Series(np.nan, index=x.index)
        return (xc-xc.mean())/sd
    return df.groupby(datecol)[col].transform(_w)

def fm_loadings(panel, ycol, dh, ds, datecol, Tcol, min_cs):
    rec = []
    for d, g in panel.groupby(datecol):
        sub = g[[ycol, dh, ds, Tcol]].dropna()
        if len(sub) < min_cs: continue
        X = sm.add_constant(sub[[dh, ds]], has_constant="add")
        r = sm.OLS(sub[ycol], X).fit()
        rec.append((d, r.params[dh], r.params[ds], sub[Tcol].iloc[0], len(sub)))
    return pd.DataFrame(rec, columns=[datecol, "beta_dH", "beta_dS", "T", "n_stocks"])

def quintile_series(panel, sortcol, ycol, datecol, prefix):
    d = panel.dropna(subset=[sortcol, ycol]).copy()
    d["qd"] = d.groupby(datecol)[sortcol].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
        if x.nunique() >= 5 else np.nan)
    d = d.dropna(subset=["qd"])
    qr = d.groupby([datecol, "qd"])[ycol].mean().unstack("qd")
    qr.columns = [f"{prefix}_Q{int(c)+1}" for c in qr.columns]
    qr[f"{prefix}_LS"] = qr[f"{prefix}_Q5"] - qr[f"{prefix}_Q1"]
    return qr

# ── monthly S&P 500 panel ────────────────────────────────────────────────────
m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
m["date"] = pd.to_datetime(m["date"])
m["dH_gpm_z"] = cs_wz(m, "dH_gpm", "date")

fm_m = fm_loadings(m, "ret_next_month", "dH_gpm_z", "DS_z", "date", "T", min_cs=10)
fm_m.to_csv(f"{OUT}/fm_loadings_monthly_sp500.csv", index=False)
print(f"fm_loadings_monthly_sp500.csv: {len(fm_m)} months")

pm = pd.concat([quintile_series(m, "DG",   "ret_next_month", "date", "DG"),
                quintile_series(m, "DS_z", "ret_next_month", "date", "DS")], axis=1)
pm.index.name = "date"
pm.to_csv(f"{OUT}/portfolio_returns_monthly_sp500.csv")
print(f"portfolio_returns_monthly_sp500.csv: {len(pm)} months")

# ── quarterly full-universe (survivorship-free) panel ───────────────────────
q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
fm_q = fm_loadings(q, "ret_next", "delta_h_z", "delta_s_z", "q", "T", min_cs=20)
fm_q["q"] = fm_q["q"].astype(str)
fm_q.to_csv(f"{OUT}/fm_loadings_quarterly_fulluniverse.csv", index=False)
print(f"fm_loadings_quarterly_fulluniverse.csv: {len(fm_q)} quarters")

pq = pd.concat([quintile_series(q, "delta_g",   "ret_next", "q", "DG"),
                quintile_series(q, "delta_s_z", "ret_next", "q", "DS")], axis=1)
pq.index = pq.index.astype(str); pq.index.name = "quarter"
pq.to_csv(f"{OUT}/portfolio_returns_quarterly_fulluniverse.csv")
print(f"portfolio_returns_quarterly_fulluniverse.csv: {len(pq)} quarters")

# ── E1 survival-conditioning summary (from results/revision/R25_post_review.txt) ─
e1 = pd.DataFrame({
    "k_years":            [0, 5, 10, 15, 20, 25, 27],
    "ls_ann_return":      [-0.0102, 0.0197, 0.0494, 0.0599, 0.0719, 0.0918, 0.1097],
    "ls_t_simple":        [-0.2004, 0.3949, 1.023, 1.281, 1.577, 2.139, 3.154],
    "ls_t_nw4":           [-0.1807, 0.3498, 0.911, 1.165, 1.471, 1.992, 3.076],
    "fm_t_dS":            [0.01812, 0.602, 0.9396, 1.152, 1.421, 1.858, 3.229],
    "fm_t_dH":            [3.458, 3.377, 3.0, 2.6, 1.922, 1.271, -0.04384],
    "n_tickers":          [12449, 6777, 3958, 2627, 1678, 1047, 336],
    "avg_stocks_qtr":     [3723, 3340, 2606, 2054, 1482, 983.2, 331.1],
    "n_obs":              [420712, 377441, 294452, 230058, 165982, 110124, 37084],
    "median_marketcap_usd_m": [401.1, 466.4, 574, 753.9, 959.1, 1362, 4082],
})
e1.to_csv(f"{OUT}/e1_survival_conditioning_summary.csv", index=False)
print("e1_survival_conditioning_summary.csv: 7 rows")
print("Done.")
