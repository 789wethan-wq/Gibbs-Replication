"""PanelC_verify_ab.py — (a) confirm the double-cluster Wald runs on an IDENTIFIED
pooled design matrix (full rank, T·ΔS not collinear once pooled across months);
(b) confirm the 'β_ΔS survives HXZ, t=4.97' claim is Model B + HXZ (no interaction),
hence identified and valid. DIAGNOSTIC — no manuscript text.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
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

def cluster_vcov(X, resid, groups):
    # Vectorized (O(n)) equivalent of the per-group outer-product sum: for
    # contiguous integer cluster codes, B = sum_g (X_g' r_g)(X_g' r_g)'.
    n_, k_ = X.shape; inv = np.linalg.pinv(X.T @ X)
    codes = np.asarray(groups)
    if codes.dtype.kind not in "iu":
        codes = pd.Categorical(codes).codes
    G = int(codes.max()) + 1
    Xr = X * resid[:, None]
    S = np.zeros((G, k_)); np.add.at(S, codes, Xr); B = S.T @ S
    sc = (G / (G - 1)) * ((n_ - 1) / (n_ - k_))
    return sc * inv @ B @ inv

def twoway(X, resid, gd, gf):
    inter = pd.Categorical(pd.Series(gd).astype(str) + "_" + pd.Series(gf).astype(str)).codes
    return cluster_vcov(X, resid, gd) + cluster_vcov(X, resid, gf) - cluster_vcov(X, resid, inter)

# ══════════════════════════════════════════════════════════════════════════
print("#"*74); print("# (a) Is the double-cluster Wald design matrix IDENTIFIED (pooled)?")
print("#"*74)

def rank_report(tag, X, dscol, txcol):
    r = np.linalg.matrix_rank(X); c = np.linalg.cond(X)
    corr = np.corrcoef(X[:, dscol], X[:, txcol])[0, 1]
    print(f"[{tag}] POOLED design [1,ΔH,ΔS,T·ΔS]: rank={r}/{X.shape[1]}  "
          f"cond={c:.2f}  corr(ΔS,T·ΔS)_pooled={corr:+.4f}")
    return r

# S&P 500
m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet"); m["date"] = pd.to_datetime(m["date"])
m["dH_gpm_z"] = cs_wz(m, "dH_gpm")
sp = m.dropna(subset=["ret_next_month", "dH_gpm_z", "DS_z", "TxDS"]).copy()
Xsp = np.column_stack([np.ones(len(sp)), sp["dH_gpm_z"], sp["DS_z"], sp["TxDS"]])
rank_report("S&P500", Xsp, 2, 3)
# per-month unique-T count to show WHY pooling identifies it
nT = sp.groupby("date")["T"].first().nunique()
print(f"         distinct monthly T values across sample = {nT} (T varies across months "
      f"⇒ ΔS and T·ΔS de-collinearised once pooled)")
ysp = sp["ret_next_month"].values
bsp, *_ = np.linalg.lstsq(Xsp, ysp, rcond=None); rsp = ysp - Xsp @ bsp
gd = pd.Categorical(sp["date"].astype(str)).codes; gf = pd.Categorical(sp["stock_id"].astype(str)).codes
for lab, V in [("date-cluster", cluster_vcov(Xsp, rsp, gd)),
               ("double-cluster(firm×date)", twoway(Xsp, rsp, gd, gf))]:
    t = bsp[3] / np.sqrt(V[3, 3]); p = 1 - chi2.cdf(t**2, 1)
    print(f"         S&P Wald T·ΔS [{lab:26}] coef={bsp[3]:+.5f}  t={t:+.4f}  p={p:.4f}")

# Full universe
q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
fu = q.dropna(subset=["ret_next", "delta_h_z", "delta_s_z", "T_delta_s"]).copy()
Xfu = np.column_stack([np.ones(len(fu)), fu["delta_h_z"], fu["delta_s_z"], fu["T_delta_s"]])
rank_report("FullUniv", Xfu, 2, 3)
yfu = fu["ret_next"].values
bfu, *_ = np.linalg.lstsq(Xfu, yfu, rcond=None); rfu = yfu - Xfu @ bfu
gdq = pd.Categorical(fu["q"].astype(str)).codes; gfq = pd.Categorical(fu["ticker"].astype(str)).codes
for lab, V in [("date-cluster", cluster_vcov(Xfu, rfu, gdq)),
               ("double-cluster(firm×date)", twoway(Xfu, rfu, gdq, gfq))]:
    t = bfu[3] / np.sqrt(V[3, 3]); p = 1 - chi2.cdf(t**2, 1)
    print(f"         Full Wald T·ΔS [{lab:26}] coef={bfu[3]:+.5f}  t={t:+.4f}  p={p:.4f}")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "#"*74)
print("# (b) 'β_ΔS survives HXZ, t=4.97' — Model B + HXZ, NO interaction term")
print("#"*74)
qf = pd.read_parquet(f"{DATA}/hxz_q5_monthly.parquet")
qf["date"] = (pd.to_datetime(qf["year"].astype(str) + "-" + qf["month"].astype(str).str.zfill(2) + "-01")
              + pd.offsets.MonthEnd(0))
for c in ["R_ME", "R_IA", "R_ROE"]: qf[c] = qf[c] / 100.0
mb = m.merge(qf[["date", "R_ME", "R_IA", "R_ROE"]], on="date", how="left")
sub = mb.dropna(subset=["dH_gpm_z", "DS_z", "R_ME", "R_IA", "R_ROE", "ret_next_month"]).copy()
cols = ["dH_gpm_z", "DS_z", "R_ME", "R_IA", "R_ROE"]   # Model B + HXZ, NO T·ΔS
X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in cols])
print(f"  spec = ΔH_z + ΔS_z + q(ME,IA,ROE)  [no T·ΔS]  rank={np.linalg.matrix_rank(X)}/{X.shape[1]}  "
      f"cond={np.linalg.cond(X):.2f}  N={len(sub):,}")
y = sub["ret_next_month"].values
b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
gd = pd.Categorical(sub["date"].astype(str)).codes; gf = pd.Categorical(sub["stock_id"].astype(str)).codes
for lab, V in [("date-cluster", cluster_vcov(X, r, gd)),
               ("double-cluster(firm×date)", twoway(X, r, gd, gf))]:
    se = np.sqrt(np.diag(V)); t = b / se
    print(f"  [{lab:26}] β_ΔH={b[1]:+.5f} t={t[1]:+.3f}   β_ΔS={b[2]:+.5f} t={t[2]:+.3f}   "
          f"(q t: ME={t[3]:+.2f} IA={t[4]:+.2f} ROE={t[5]:+.2f})")
print("  NB: no T·ΔS in this spec ⇒ ΔS is separately identified (full rank); the 4.97")
print("      figure is the DATE-clustered Model B+HXZ ΔS level on the S&P 500 panel.")
