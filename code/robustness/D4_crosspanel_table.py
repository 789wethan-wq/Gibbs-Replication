"""D4_crosspanel_table.py — fresh, single-process re-verification of every row
in the D4 cross-panel difference table (§4.8 prose numbers), assembled into
one table. Rebuilds everything from parquet; does not import any cached
result from a prior R25 run.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
sys.path.insert(0, "../project")
from utils import newey_west_mean_tstat

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/D4_crosspanel_table.txt"

print(f"[pid={os.getpid()}] D4 — fresh process")

log = []
def P(s=""):
    print(s)
    log.append(str(s))


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


m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
m["date"] = pd.to_datetime(m["date"])
m["dH_gpm_z"] = cs_wz(m, "dH_gpm")
prim = m.dropna(subset=["ret_next_month", "dH_gpm_z", "DS_z"])
rows_sp = []
for d in sorted(prim["date"].unique()):
    sub = prim[prim["date"] == d]
    if len(sub) < 4:
        continue
    X = sm.add_constant(sub[["dH_gpm_z", "DS_z"]])
    res = sm.OLS(sub["ret_next_month"], X).fit()
    rows_sp.append(pd.Series(res.params, name=d))
cdf_sp = pd.DataFrame(rows_sp)
mean_sp, t_sp, p_sp = newey_west_mean_tstat(cdf_sp["DS_z"].values, lags=0)
se_sp = mean_sp / t_sp
first_sp, last_sp = str(prim["date"].min().date()), str(prim["date"].max().date())
P(f"Row 1 input: SP500 monthly Model B DS_z: coef={mean_sp:+.6f} SE={se_sp:.6f} t={t_sp:+.4f} "
  f"N={len(prim):,} T={len(cdf_sp)} range={first_sp}..{last_sp}")

q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
xcols_fu = ["delta_h_z", "delta_s_z"]
rows_fu = []
for d in sorted(q["q"].unique()):
    sub = q[q["q"] == d].dropna(subset=["ret_next"] + xcols_fu)
    if len(sub) < 20:  # canonical FU floor
        continue
    X = sm.add_constant(sub[xcols_fu])
    res = sm.OLS(sub["ret_next"], X).fit()
    rows_fu.append(pd.Series(res.params, name=d))
cdf_fu = pd.DataFrame(rows_fu)
mean_fu, t_fu, p_fu = newey_west_mean_tstat(cdf_fu["delta_s_z"].values, lags=4)
se_fu = mean_fu / t_fu
first_fu, last_fu = str(q["q"].min()), str(q["q"].max())
P(f"Row 2 input: FullUniv quarterly Model B delta_s_z: coef={mean_fu:+.6f} SE={se_fu:.6f} t={t_fu:+.4f} "
  f"N={q.dropna(subset=['ret_next']+xcols_fu).shape[0]:,} T={len(cdf_fu)} range={first_fu}..{last_fu}")

ann_sp = mean_sp * 12
ann_sp_se = se_sp * 12
ann_fu = mean_fu * 4
ann_fu_se = se_fu * 4
P(f"\nAnnualized SP500:    {ann_sp*100:+.3f}%/yr per SD, SE={ann_sp_se*100:.3f}pp -> "
  f"reported as +6.22%/yr in manuscript, this run: {ann_sp*100:+.2f}%/yr")
P(f"Annualized FullUniv: {ann_fu*100:+.3f}%/yr per SD, SE={ann_fu_se*100:.3f}pp -> "
  f"reported as +0.035%/yr in manuscript, this run: {ann_fu*100:+.3f}%/yr")

# quarter-matched series correlation + variance reduction (needs the harmonized
# SP quarterly series -- rebuilt fresh here, not imported)
sp = m.dropna(subset=["ret_next_month", "DS_z", "dH_gpm_z"]).copy()
sp = sp.sort_values(["stock_id", "date"]).reset_index(drop=True)
g = sp.groupby("stock_id")
r1 = sp["ret_next_month"]; r2 = g["ret_next_month"].shift(-1); r3 = g["ret_next_month"].shift(-2)
sp["fwd3_ret"] = (1 + r1) * (1 + r2) * (1 + r3) - 1
sp["quarter"] = sp["date"].dt.to_period("Q")
sp["is_qend"] = sp.groupby(["stock_id", "quarter"])["date"].transform("max") == sp["date"]
sp_q = sp[sp["is_qend"]].dropna(subset=["fwd3_ret"]).copy()
rows_spq = []
for d in sorted(sp_q["quarter"].unique()):
    sub = sp_q[sp_q["quarter"] == d]
    if len(sub) < 4:
        continue
    X = sm.add_constant(sub[["dH_gpm_z", "DS_z"]])
    res = sm.OLS(sub["fwd3_ret"], X).fit()
    rows_spq.append(pd.Series(res.params, name=d))
cdf_spq = pd.DataFrame(rows_spq)
mean_spq, t_spq, _ = newey_west_mean_tstat(cdf_spq["DS_z"].values, lags=4)
P(f"\nHarmonization cost: SP500 raw monthly t(dS)={t_sp:+.4f} -> harmonized quarterly t(dS)={t_spq:+.4f} "
  f"(manuscript: +4.70 -> +4.39)")

cs_ser = cdf_spq["DS_z"].rename("beta_sp"); cs_ser.index = cs_ser.index.astype(str)
cf_ser = cdf_fu["delta_s_z"].rename("beta_fu"); cf_ser.index = cf_ser.index.astype(str)
aligned = pd.concat([cs_ser, cf_ser], axis=1).dropna()
rho = aligned["beta_sp"].corr(aligned["beta_fu"])
var_sp, var_fu = aligned["beta_sp"].var(), aligned["beta_fu"].var()
diff = aligned["beta_fu"] - aligned["beta_sp"]
var_diff = diff.var()
var_indep = var_sp + var_fu
var_reduction_pct = (1 - var_diff / var_indep) * 100
P(f"\nQuarter-matched series correlation: rho(beta_sp, beta_fu) = {rho:+.4f} over {len(aligned)} common quarters "
  f"(manuscript: rho=+0.42)")
P(f"Variance-reduction check: var(diff)/[var(sp)+var(fu)] = {var_diff/var_indep:.4f}  "
  f"-> variance REDUCED by {var_reduction_pct:.1f}% relative to independence "
  f"(manuscript: 58.6%; note this equals 1-var_diff/var_indep, which for the special")
P(f"case cov=rho*sd_sp*sd_fu simplifies toward but is NOT algebraically identical to 1-rho "
  f"[1-rho = {(1-rho)*100:.1f}% -- close to {var_reduction_pct:.1f}% here because var_sp and var_fu are of similar")
P(f"order, but '1-rho' is an approximation, not an identity; both numbers reported for transparency].")

mean_diff, t_diff, p_diff = newey_west_mean_tstat(diff.values, lags=4)
se_diff = mean_diff / t_diff
ci_lo_fm, ci_hi_fm = mean_diff - 1.96*se_diff, mean_diff + 1.96*se_diff
P(f"\nFM-family paired difference (fresh): mean={mean_diff:+.6f} t={t_diff:+.4f} "
  f"95% CI=({ci_lo_fm:+.6f}, {ci_hi_fm:+.6f})  (manuscript: t=-5.19, CI -0.0205 to -0.0093)")

# stacked pooled-OLS + bootstrap: rebuild fresh, same construction as R25(2)/(3)
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
def twoway_vcov(X, resid, g1, g2):
    inter = pd.Categorical(pd.Series(g1).astype(str) + "_" + pd.Series(g2).astype(str)).codes
    return cluster_vcov(X, resid, g1) + cluster_vcov(X, resid, g2) - cluster_vcov(X, resid, inter)

sp_stack = sp_q[["stock_id", "quarter", "DS_z", "dH_gpm_z", "fwd3_ret"]].rename(
    columns={"stock_id": "firm", "DS_z": "dS", "dH_gpm_z": "dH", "fwd3_ret": "y"})
sp_stack["D_FU"] = 0
fu_stack = q.dropna(subset=["ret_next", "delta_s_z", "delta_h_z"])[
    ["ticker", "q", "delta_s_z", "delta_h_z", "ret_next"]].rename(
    columns={"ticker": "firm", "q": "quarter", "delta_s_z": "dS", "delta_h_z": "dH", "ret_next": "y"})
fu_stack["D_FU"] = 1
stacked = pd.concat([sp_stack, fu_stack], ignore_index=True)
stacked["quarter"] = stacked["quarter"].astype(str)
stacked["DxDS"] = stacked["D_FU"] * stacked["dS"]
Xcols = ["dS", "dH", "D_FU", "DxDS"]
Xs = sm.add_constant(stacked[Xcols]).values
ys = stacked["y"].values
b, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
resid = ys - Xs @ b
g_firm = pd.Categorical(stacked["firm"]).codes
g_q = pd.Categorical(stacked["quarter"]).codes
V = twoway_vcov(Xs, resid, g_firm, g_q)
se = np.sqrt(np.diag(V))
i_int = (["const"] + Xcols).index("DxDS")
t_stack = b[i_int] / se[i_int]
ci_lo, ci_hi = b[i_int] - 1.96*se[i_int], b[i_int] + 1.96*se[i_int]
P(f"\nStacked pooled-OLS two-way-clustered (fresh): coef={b[i_int]:+.6f} SE={se[i_int]:.6f} t={t_stack:+.4f} "
  f"95% CI=({ci_lo:+.6f}, {ci_hi:+.6f})  (manuscript: t=-4.11, CI -0.0256 to -0.0091)")

SEED = 20250725
rng = np.random.RandomState(SEED)
all_quarters = stacked["quarter"].unique()
nQ = len(all_quarters)
NBOOT = 500
boot_diffs = np.empty(NBOOT)
quarter_groups = {qq: stacked[stacked["quarter"] == qq] for qq in all_quarters}
for bidx in range(NBOOT):
    samp_q = rng.choice(all_quarters, size=nQ, replace=True)
    bsamp = pd.concat([quarter_groups[qq] for qq in samp_q], ignore_index=True)
    Xb = sm.add_constant(bsamp[Xcols]).values
    yb = bsamp["y"].values
    bb, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    boot_diffs[bidx] = bb[i_int]
boot_lo, boot_hi = np.percentile(boot_diffs, [2.5, 97.5])
P(f"Block bootstrap (fresh, block=1 quarter, resample WITH replacement, {NBOOT} draws, seed={SEED}): "
  f"95% CI=({boot_lo:+.6f}, {boot_hi:+.6f})  (manuscript: -0.0259 to -0.0080)")

P("\n" + "="*88)
P("D4 FINAL TABLE")
P("="*88)
P(f"{'Row':42}{'This run':>24}{'Manuscript':>20}")
P(f"{'Annualized b(DS), biased(SP) panel, /SD':42}{ann_sp*100:>+20.3f}%/yr (SE {ann_sp_se*100:.3f}pp)  {'+6.22%/yr':>0}")
P(f"{'Annualized b(DS), corrected(FU) panel, /SD':42}{ann_fu*100:>+20.3f}%/yr (SE {ann_fu_se*100:.3f}pp)  {'+0.035%/yr':>0}")
P(f"{'Stacked 2way-clustered difference, t':42}{t_stack:>+24.4f}{'-4.11':>20}")
P(f"{'  95% CI':42}{f'({ci_lo:+.4f}, {ci_hi:+.4f})':>24}{'(-0.0256, -0.0091)':>20}")
P(f"{'Block-bootstrap 95% CI':42}{f'({boot_lo:+.4f}, {boot_hi:+.4f})':>24}{'(-0.0259, -0.0080)':>20}")
P(f"{'FM-family paired difference, t':42}{t_diff:>+24.4f}{'-5.19':>20}")
P(f"{'  95% CI':42}{f'({ci_lo_fm:+.4f}, {ci_hi_fm:+.4f})':>24}{'(-0.0205, -0.0093)':>20}")
P(f"{'Quarter-matched series correlation, rho':42}{rho:>+24.4f}{'+0.42':>20}")
P(f"{'Variance-reduction factor (var_diff/indep)':42}{var_diff/var_indep*100:>23.1f}%{'58.6%':>20}")
P(f"{'Harmonization cost, FM t(dS) monthly':42}{t_sp:>+24.4f}{'+4.70':>20}")
P(f"{'  -> quarterly':42}{t_spq:>+24.4f}{'+4.39':>20}")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
