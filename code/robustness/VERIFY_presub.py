"""VERIFY_presub.py — Pre-submission diagnostics V1–V4. READ-ONLY / DIAGNOSTIC.
Reproduces the exact estimators the manuscript uses; changes no manuscript text.
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, norm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
np.set_printoptions(suppress=True)

def cs_wz(df, col, date_col="date", pct=0.01):
    def _f(x):
        x2 = x.dropna()
        if len(x2) < 5: return pd.Series(np.nan, index=x.index)
        lo, hi = x2.quantile(pct), x2.quantile(1 - pct)
        xc = x.clip(lo, hi); s = xc.std()
        if s < 1e-10: return pd.Series(np.nan, index=x.index)
        return (xc - xc.mean()) / s
    return df.groupby(date_col)[col].transform(_f)

def nw_tstat(series, lags):
    s = pd.Series(series).dropna().values; n = len(s); m = s.mean()
    g0 = ((s - m) ** 2).mean(); v = g0
    for l in range(1, min(lags + 1, n)):
        gl = ((s[l:] - m) * (s[:-l] - m)).mean()
        v += 2 * (1 - l / (lags + 1)) * gl
    se = np.sqrt(max(v, 1e-30) / n)
    return m, m / se, n

def fm(panel, y, xcols, datecol, lags, min_cs):
    coefs = []
    for d, grp in panel.groupby(datecol):
        sub = grp[[y] + xcols].dropna()
        if len(sub) < max(min_cs, len(xcols) + 2): continue
        X = sm.add_constant(sub[xcols], has_constant="add")
        try:
            r = sm.OLS(sub[y], X).fit(); coefs.append(r.params[xcols].rename(d))
        except Exception: pass
    cdf = pd.DataFrame(coefs); out = {}
    for c in xcols: out[c] = nw_tstat(cdf[c].values, lags)
    return out, len(cdf)

def cluster_vcov(X, resid, groups):
    n_, k_ = X.shape; inv = np.linalg.pinv(X.T @ X); B = np.zeros((k_, k_))
    for g in np.unique(groups):
        mm = groups == g; B += X[mm].T @ np.outer(resid[mm], resid[mm]) @ X[mm]
    G = len(np.unique(groups)); sc = (G / (G - 1)) * ((n_ - 1) / (n_ - k_))
    return sc * inv @ B @ inv

def double_cluster(X, resid, g1, g2):
    inter = pd.Categorical(pd.Series(g1).astype(str) + "_" + pd.Series(g2).astype(str)).codes
    return cluster_vcov(X, resid, g1) + cluster_vcov(X, resid, g2) - cluster_vcov(X, resid, inter)

print("#"*72); print("# V1 — Is Table 2 Panel C rank-deficient? (S&P 500 encompassing FM loop)")
print("#"*72)
m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
m["date"] = pd.to_datetime(m["date"])
m["dH_gpm_z"] = cs_wz(m, "dH_gpm")            # exactly as R16.2 load_base()
sub = m.dropna(subset=["dH_gpm_z", "DS_z", "TxDS", "ret_next_month"]).copy()
cols = ["dH_gpm_z", "DS_z", "TxDS"]           # encompassing spec: ΔH, ΔS, T·ΔS
months = sorted(sub["date"].unique())
pick = [months[0], months[len(months)//3], months[len(months)//2], months[-2]]
for d in pick:
    g = sub[sub["date"] == d]
    if len(g) < 15: continue
    X = sm.add_constant(g[cols], has_constant="add").values   # [const, dH, dS, TxDS]
    rank = np.linalg.matrix_rank(X); ncol = X.shape[1]
    corr = np.corrcoef(g["DS_z"], g["TxDS"])[0, 1]
    cond = np.linalg.cond(X)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        res = sm.OLS(g["ret_next_month"], X).fit()     # statsmodels default solver
        fired = [str(w.message) for w in wlist]
    ym = pd.Timestamp(d).strftime("%Y-%m")
    print(f"[V1 | month={ym}] rank={rank}, n_cols={ncol}, corr(dS,TdS)={corr:+.6f}, "
          f"cond={cond:.3e}, solver=statsmodels.OLS(method='{res.model.__dict__.get('_fit_method','pinv')}'→pinv/SVD), "
          f"warning={fired if fired else 'NONE'}")
print("  (statsmodels OLS default cov/solve = Moore-Penrose pinv via SVD; "
      "returns min-norm solution silently on rank deficiency, no rank warning.)")

print("\n" + "#"*72); print("# V2 — Are the two t=+2.49 / p=0.013 pooled results independent?")
print("#"*72)
# (a) S&P 500 pooled two-way clustered interaction (R20 §[3])
mm = m.dropna(subset=["ret_next_month", "dH_gpm_z", "DS_z", "TxDS"]).copy()
Xa = np.column_stack([np.ones(len(mm)), mm["dH_gpm_z"], mm["DS_z"], mm["TxDS"]])
ya = mm["ret_next_month"].values
ba, *_ = np.linalg.lstsq(Xa, ya, rcond=None); ra = ya - Xa @ ba
ga_d = pd.Categorical(mm["date"].astype(str)).codes
ga_f = pd.Categorical(mm["stock_id"]).codes
Va = double_cluster(Xa, ra, ga_d, ga_f)
ta = ba[3] / np.sqrt(Va[3, 3]); pa = 1 - chi2.cdf(ta**2, 1)
# (b) Full-universe pooled two-way clustered interaction (R18 STEP 10)
q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
pf = q.dropna(subset=["ret_next", "delta_h_z", "delta_s_z", "T_delta_s"]).copy()
Xb = np.column_stack([np.ones(len(pf)), pf["delta_h_z"], pf["delta_s_z"], pf["T_delta_s"]])
yb = pf["ret_next"].values
bb, *_ = np.linalg.lstsq(Xb, yb, rcond=None); rb = yb - Xb @ bb
gb_d = pd.Categorical(pf["q"].astype(str)).codes
gb_f = pd.Categorical(pf["ticker"]).codes
Vb = double_cluster(Xb, rb, gb_d, gb_f)
tb = bb[3] / np.sqrt(Vb[3, 3]); pb = 1 - chi2.cdf(tb**2, 1)
print(f"[V2] SP500:       coef={ba[3]:+.6f}  t={ta:.4f}  p={pa:.4f}  N={len(mm):,}  "
      f"months={mm['date'].nunique()}  firms={mm['stock_id'].nunique()}")
print(f"[V2] FullUniverse:coef={bb[3]:+.6f}  t={tb:.4f}  p={pb:.4f}  N={len(pf):,}  "
      f"quarters={pf['q'].nunique()}  firms={pf['ticker'].nunique()}")
print(f"     distinct DataFrames: SP500 id={id(mm)} vs Full id={id(pf)}; "
      f"N differ ({len(mm):,} vs {len(pf):,}); frequency differ (monthly vs quarterly).")

print("\n" + "#"*72); print("# V3 — Does z-scoring leave FM t(ΔH) invariant on MATCHED N?")
print("#"*72)
# 60-month ΔH window = primary dH_gpm (already 60m rolling std upstream). Matched sample:
mv3 = m.dropna(subset=["ret_next_month", "dH_gpm", "dH_gpm_z", "DS_z", "DS_raw"]).copy()
# raw: dH_gpm + DS_raw ; standardized: dH_gpm_z + DS_z  — IDENTICAL rows
o_std, n_std = fm(mv3, "ret_next_month", ["dH_gpm_z", "DS_z"], "date", lags=6, min_cs=15)
o_raw, n_raw = fm(mv3, "ret_next_month", ["dH_gpm", "DS_raw"], "date", lags=6, min_cs=15)
# also raw-dH but keep DS z-scored (isolate ΔH rescaling only)
o_mix, _ = fm(mv3, "ret_next_month", ["dH_gpm", "DS_z"], "date", lags=6, min_cs=15)
print(f"[V3] matched N={len(mv3):,}  months={mv3['date'].nunique()}")
print(f"     standardized (dH_gpm_z,DS_z): t(dH)={o_std['dH_gpm_z'][1]:+.4f}  t(dS)={o_std['DS_z'][1]:+.4f}")
print(f"     raw         (dH_gpm ,DS_raw): t(dH)={o_raw['dH_gpm'][1]:+.4f}  t(dS)={o_raw['DS_raw'][1]:+.4f}")
print(f"     raw dH only (dH_gpm ,DS_z  ): t(dH)={o_mix['dH_gpm'][1]:+.4f}  t(dS)={o_mix['DS_z'][1]:+.4f}")

print("\n" + "#"*72); print("# V4 — Window sweep (24/36/48/60/72m ΔH) on R18 full-universe panel")
print("#"*72)
# Rebuild ΔH_GPM at each rolling-std window from monthly_fundamentals, remerge to R18 panel.
mf = pd.read_parquet(f"{DATA}/monthly_fundamentals.parquet")
mf["date"] = pd.to_datetime(mf["date"]); mf = mf.sort_values(["stock_id", "date"])
base = q[["ticker", "q", "ret_next", "delta_s", "delta_s_z", "T"]].copy()
print(f"     {'window':>7}{'constr':>7}{'N':>9}{'t(dH)':>8}{'t(dS)':>8}")
for w in [24, 36, 48, 60, 72]:
    mp = max(8, w // 2)
    dh = -mf.groupby("stock_id")["gpm"].transform(lambda x: x.rolling(w, min_periods=mp).std())
    tmp = mf[["stock_id", "date"]].assign(dH_gpm=dh.values)
    tmp["q"] = tmp["date"].dt.to_period("Q")
    gpmq = (tmp.dropna(subset=["dH_gpm"]).sort_values(["stock_id", "date"])
              .drop_duplicates(["stock_id", "q"], keep="last")
              .rename(columns={"stock_id": "ticker"})[["ticker", "q", "dH_gpm"]])
    p = base.merge(gpmq, on=["ticker", "q"], how="inner")
    p["delta_h_z"] = cs_wz(p, "dH_gpm", "q")
    # standardized construction
    ps = p.dropna(subset=["ret_next", "delta_h_z", "delta_s_z"])
    os_, ns = fm(ps, "ret_next", ["delta_h_z", "delta_s_z"], "q", lags=4, min_cs=20)
    print(f"[V4] {str(w)+'mo':>7}{'std':>7}{ns and len(ps):>9,}{os_['delta_h_z'][1]:>+8.2f}{os_['delta_s_z'][1]:>+8.2f}")
    # raw construction
    pr = p.dropna(subset=["ret_next", "dH_gpm", "delta_s"])
    or_, nr = fm(pr, "ret_next", ["dH_gpm", "delta_s"], "q", lags=4, min_cs=20)
    print(f"[V4] {str(w)+'mo':>7}{'raw':>7}{len(pr):>9,}{or_['dH_gpm'][1]:>+8.2f}{or_['delta_s'][1]:>+8.2f}")
