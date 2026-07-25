"""R25 — Formal test of the cross-panel ΔS coefficient difference.

The manuscript's headline is a comparison of two t-statistics from two
non-nested, different-frequency samples ("+4.70 -> +0.02"). This script:
  (1) annualizes both raw coefficients to a common scale and states the
      convention,
  (2) harmonizes BOTH panels to quarterly frequency, stacks them with a
      panel indicator D_FU and interaction D_FU x DeltaS_z, and estimates the
      pooled difference with two-way (firm x quarter) clustered SEs,
  (3) block-bootstraps the difference (resampling whole quarters) as a
      robustness check on the CI,
  (4) reports the harmonized single-panel (SP500-quarterly) estimate
      alongside the raw monthly one so the reader can see what harmonizing
      costs.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, norm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/R25_cross_panel_diff.txt"

import sys
sys.path.insert(0, "../project")
from utils import newey_west_mean_tstat

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


# ═══════════════════════════════════════════════════════════════════════════
P("="*78)
P("R25 (1) — Annualizing the raw ΔS coefficients")
P("="*78)

m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
m["date"] = pd.to_datetime(m["date"])
m["dH_gpm_z"] = cs_wz(m, "dH_gpm")

xcols = ["dH_gpm_z", "DS_z"]
prim = m.dropna(subset=["ret_next_month"] + xcols).copy()
dates = sorted(prim["date"].unique())
rows = []
for d in dates:
    sub = prim[prim["date"] == d]
    if len(sub) < len(xcols) + 2:
        continue
    X = sm.add_constant(sub[xcols])
    res = sm.OLS(sub["ret_next_month"], X).fit()
    rows.append(pd.Series(res.params, name=d))
cdf_sp = pd.DataFrame(rows)
mean_ds_sp, t_ds_sp, p_ds_sp = newey_west_mean_tstat(cdf_sp["DS_z"].values, lags=0)
P(f"SP500 monthly  Model B: mean coef(DS_z) = {mean_ds_sp:+.6f}/month  NW-0 t = {t_ds_sp:+.4f}  "
  f"(N={len(prim):,}, T={len(cdf_sp)} months)")

q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
xcols_fu = ["delta_h_z", "delta_s_z"]
# NB: full-universe canonical primary rig uses min_cs=20 per quarter (R22_v19_battery.py
# fm()), NOT len(xcols)+2=4 -- verified this exactly reproduces the manuscript's
# +3.46/+0.02 (min_cs=4 gives +1.06/+0.25 instead, a materially different answer
# because it lets thin early-1990s cross-sections into the NW average).
FU_MIN_CS = 20
qdates = sorted(q["q"].unique())
rowsq = []
for d in qdates:
    sub = q[q["q"] == d].dropna(subset=["ret_next"] + xcols_fu)
    if len(sub) < FU_MIN_CS:
        continue
    X = sm.add_constant(sub[xcols_fu])
    res = sm.OLS(sub["ret_next"], X).fit()
    rowsq.append(pd.Series(res.params, name=d))
cdf_fu = pd.DataFrame(rowsq)
mean_ds_fu, t_ds_fu, p_ds_fu = newey_west_mean_tstat(cdf_fu["delta_s_z"].values, lags=4)
P(f"FullUniv quarterly Model B: mean coef(delta_s_z) = {mean_ds_fu:+.6f}/quarter  NW-4 t = {t_ds_fu:+.4f}  "
  f"(N={q.dropna(subset=['ret_next']+xcols_fu).shape[0]:,}, T={len(cdf_fu)} quarters, min_cs={FU_MIN_CS})")

P("")
P("Annualization convention: both DS_z/delta_s_z regressors are already")
P("cross-sectionally z-scored WITHIN period (unit SD), so the FM coefficient")
P("is directly '(average) return spread per 1 SD of ΔS, at the observation")
P("frequency'. At these magnitudes (<1%/period) simple linear scaling and")
P("geometric compounding differ by <2% of the annualized value, so we report")
P("simple linear annualization (x12 monthly, x4 quarterly) as primary, with")
P("the compounding alternative alongside for transparency.")

ann_sp_lin = mean_ds_sp * 12
ann_fu_lin = mean_ds_fu * 4
ann_sp_geo = (1 + mean_ds_sp) ** 12 - 1
ann_fu_geo = (1 + mean_ds_fu) ** 4 - 1
P(f"SP500:    annualized coef (linear x12)   = {ann_sp_lin:+.5f}/yr = {ann_sp_lin*100:+.3f}%/yr   "
  f"(geometric compounding: {ann_sp_geo*100:+.3f}%/yr)")
P(f"FullUniv: annualized coef (linear x4)    = {ann_fu_lin:+.5f}/yr = {ann_fu_lin*100:+.3f}%/yr   "
  f"(geometric compounding: {ann_fu_geo*100:+.3f}%/yr)")
P(f"Annualized coefficient ratio (SP/FU)     = {ann_sp_lin/ann_fu_lin:.2f}x" if ann_fu_lin != 0 else "n/a")
P("NB: annualizing the COEFFICIENT is not the same as annualizing the t-stat;")
P("the t-stat is a function of the coefficient's sampling variance across")
P("periods, not of the coefficient's magnitude, and does not rescale by n.")
P("The t=+4.70 -> +0.02 comparison is a comparison of two t-stats, which is")
P("exactly the reasoning the stacked test below replaces with a single CI on")
P("a difference.")

# ═══════════════════════════════════════════════════════════════════════════
P("")
P("="*78)
P("R25 (4) — Harmonizing SP500 to quarterly frequency, and what it costs")
P("="*78)

sp = m.dropna(subset=["ret_next_month", "DS_z", "dH_gpm_z"]).copy()
sp = sp.sort_values(["stock_id", "date"]).reset_index(drop=True)
g = sp.groupby("stock_id")
# 3-month-forward compounded return starting the month AFTER the snapshot date
r1 = sp["ret_next_month"]
r2 = g["ret_next_month"].shift(-1)
r3 = g["ret_next_month"].shift(-2)
sp["fwd3_ret"] = (1 + r1) * (1 + r2) * (1 + r3) - 1
sp["quarter"] = sp["date"].dt.to_period("Q")
# keep the LAST monthly snapshot of each calendar quarter as that quarter's
# state variable (DS_z, dH_gpm_z), matched to the compounded 3-month FORWARD
# return computed above (approximates a quarterly-rebalance FM design)
sp["is_qend"] = sp.groupby(["stock_id", "quarter"])["date"].transform("max") == sp["date"]
sp_q = sp[sp["is_qend"]].dropna(subset=["fwd3_ret"]).copy()

P(f"SP500 monthly panel (raw):        N={len(sp):,}, {sp['date'].nunique()} months, "
  f"{sp['stock_id'].nunique()} firms")
P(f"SP500 harmonized-to-quarterly:    N={len(sp_q):,}, {sp_q['quarter'].nunique()} quarters, "
  f"{sp_q['stock_id'].nunique()} firms")
P("(row count falls ~1/3 as expected from monthly->quarterly resampling, plus")
P(" additional loss from requiring 3 consecutive forward monthly returns)")

qdates_sp = sorted(sp_q["quarter"].unique())
rowsq_sp = []
for d in qdates_sp:
    sub = sp_q[sp_q["quarter"] == d]
    if len(sub) < len(xcols) + 2:
        continue
    X = sm.add_constant(sub[["dH_gpm_z", "DS_z"]])
    res = sm.OLS(sub["fwd3_ret"], X).fit()
    rowsq_sp.append(pd.Series(res.params, name=d))
cdf_sp_q = pd.DataFrame(rowsq_sp)
mean_ds_spq, t_ds_spq, p_ds_spq = newey_west_mean_tstat(cdf_sp_q["DS_z"].values, lags=4)
P(f"SP500 harmonized-quarterly Model B: mean coef(DS_z) = {mean_ds_spq:+.6f}/quarter  NW-4 t = {t_ds_spq:+.4f}  "
  f"(T={len(cdf_sp_q)} quarters)")
P(f"  vs raw monthly: coef={mean_ds_sp:+.6f}/month t(NW-0)={t_ds_sp:+.4f}")
P("  Harmonization cost: quarterly resampling collapses ~3x fewer independent")
P(f"  cross-sections ({len(cdf_sp_q)} vs {len(cdf_sp)}), which is the main driver of any t-stat change.")

# ═══════════════════════════════════════════════════════════════════════════
P("")
P("="*78)
P("R25 (2) — Stacked panel test: ret ~ DS + D_FU + D_FU*DS + dH, two-way clustered")
P("="*78)

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

P(f"Stacked panel: N={len(stacked):,}  (SP500-quarterly N={len(sp_stack):,}, FullUniv N={len(fu_stack):,})")
P(f"Distinct firms (union of both, by ticker string) = {stacked['firm'].nunique():,}, "
  f"distinct quarters = {stacked['quarter'].nunique()}")

Xcols = ["dS", "dH", "D_FU", "DxDS"]
Xs = sm.add_constant(stacked[Xcols]).values
ys = stacked["y"].values
b, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
resid = ys - Xs @ b
g_firm = pd.Categorical(stacked["firm"]).codes
g_q = pd.Categorical(stacked["quarter"]).codes
V = twoway_vcov(Xs, resid, g_firm, g_q)
se = np.sqrt(np.diag(V))
tstat = b / se
pval = 1 - chi2.cdf(tstat**2, 1)
names = ["const"] + Xcols
i_int = names.index("DxDS")
ci_lo = b[i_int] - 1.96 * se[i_int]
ci_hi = b[i_int] + 1.96 * se[i_int]

P(f"design shape={Xs.shape}, clusters(firm,quarter)=({g_firm.max()+1},{g_q.max()+1})")
for nm, bb, ss, tt, pp in zip(names, b, se, tstat, pval):
    P(f"  {nm:8s} coef={bb:+.6f}  SE={ss:.6f}  t={tt:+.4f}  p={pp:.4f}")
P(f"\nD_FU x DeltaS_z (the panel-difference test): coef={b[i_int]:+.6f}  SE={se[i_int]:.6f}  "
  f"t={tstat[i_int]:+.4f}  p={pval[i_int]:.4f}  95% CI=({ci_lo:+.6f}, {ci_hi:+.6f})")
P(f"Note the sign: DxDS = (beta_FU - beta_SP) at harmonized quarterly frequency;")
P(f"beta_SP(harmonized quarterly, from same stacked regression) = coef(dS) = {b[Xcols.index('dS')+1]:+.6f}")
P(f"beta_FU(harmonized quarterly)                                        = {b[Xcols.index('dS')+1]+b[i_int]:+.6f}")

# ═══════════════════════════════════════════════════════════════════════════
P("")
P("="*78)
P("R25 (2b) — Reconciliation: pooled-OLS vs FM-average weighting")
P("="*78)
P("The stacked test above is POOLED OLS (implicitly N-weighted: quarters with")
P("more firms count more). The manuscript's own '+4.70 -> +0.02' headline is")
P("FAMA-MACBETH (equal-weighted across quarters, then NW-averaged). These are")
P("different estimators and can disagree when cross-section size varies a lot")
P("over time -- which it does here (FU panel firm count grows from a few")
P("hundred to several thousand). Concretely:")
Xfu_only = sm.add_constant(fu_stack[["dS", "dH"]]).values
yfu_only = fu_stack["y"].values
bfu_only, *_ = np.linalg.lstsq(Xfu_only, yfu_only, rcond=None)
P(f"  FU-only, pooled OLS (uncontrolled for FM weighting): coef(dS) = {bfu_only[1]:+.6f}")
P(f"  FU-only, Fama-MacBeth (equal-quarter-weighted, from R25(1))   = {mean_ds_fu:+.6f}")
P("  -> sign flips between the two estimators for the FU panel alone. This is")
P("  a real feature of the data (pooled OLS is dominated by the many-firm")
P("  later quarters; FM equal-weights every quarter regardless of firm count),")
P("  not an error. To keep the difference test in the SAME estimator family as")
P("  the manuscript's headline, we also run a FM-consistent version below:")
P("  align the two already-computed per-quarter coefficient series (SP500")
P("  harmonized-quarterly Model B DS_z, from R25(4); FullUniv Model B")
P("  delta_s_z, from R25(1)) on their common quarters, take the per-quarter")
P("  DIFFERENCE, and NW-test whether its mean is zero.")

cs_ser = cdf_sp_q["DS_z"].rename("beta_sp")
cf_ser = cdf_fu["delta_s_z"].rename("beta_fu")
cs_ser.index = cs_ser.index.astype(str)
cf_ser.index = cf_ser.index.astype(str)
aligned = pd.concat([cs_ser, cf_ser], axis=1).dropna()
aligned["diff"] = aligned["beta_fu"] - aligned["beta_sp"]
mean_diff, t_diff, p_diff = newey_west_mean_tstat(aligned["diff"].values, lags=4)
P(f"  common quarters with both coefficient estimates = {len(aligned)}")
P(f"  FM-consistent mean(beta_FU - beta_SP) = {mean_diff:+.6f}/quarter  NW-4 t = {t_diff:+.4f}  "
  f"p = {p_diff:.4f}")
se_diff = mean_diff / t_diff if t_diff not in (0, np.nan) and not np.isnan(t_diff) else np.nan
P(f"  95% CI (FM-consistent) = ({mean_diff-1.96*se_diff:+.6f}, {mean_diff+1.96*se_diff:+.6f})" if np.isfinite(se_diff) else "  95% CI unavailable")

# ═══════════════════════════════════════════════════════════════════════════
P("")
P("="*78)
P("R25 (3) — Block bootstrap on the pooled-OLS difference (resample whole quarters)")
P("="*78)

rng = np.random.RandomState(20250725)  # fixed seed, stated for reproducibility
all_quarters = stacked["quarter"].unique()
nQ = len(all_quarters)
NBOOT = 500
boot_diffs = np.empty(NBOOT)
quarter_groups = {qq: stacked[stacked["quarter"] == qq] for qq in all_quarters}
for bidx in range(NBOOT):
    samp_q = rng.choice(all_quarters, size=nQ, replace=True)
    parts = [quarter_groups[qq] for qq in samp_q]
    bsamp = pd.concat(parts, ignore_index=True)
    Xb = sm.add_constant(bsamp[Xcols]).values
    yb = bsamp["y"].values
    bb, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    boot_diffs[bidx] = bb[i_int]

boot_lo, boot_hi = np.percentile(boot_diffs, [2.5, 97.5])
P(f"Block bootstrap (resampling whole quarters, {NBOOT} reps, seed=20250725):")
P(f"  mean(boot DxDS)={boot_diffs.mean():+.6f}  SD={boot_diffs.std():.6f}")
P(f"  95% percentile CI = ({boot_lo:+.6f}, {boot_hi:+.6f})")
P(f"  analytic 95% CI   = ({ci_lo:+.6f}, {ci_hi:+.6f})")
P(f"  fraction of boot draws with same sign as point estimate = "
  f"{(np.sign(boot_diffs) == np.sign(b[i_int])).mean():.3f}")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
