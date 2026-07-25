"""D1b_triple_interaction.py — formal pooled interaction test replacing the
eyeballed +3.21-vs-+1.29 comparison in D1a/R28.

r_{i,t+1} ~ DS_z + DH_z + Surv_i + Large_{i,t}
          + DS_z*Surv_i + DS_z*Large_{i,t} + Surv_i*Large_{i,t}
          + DS_z*Surv_i*Large_{i,t} + FF5+UMD(quarterly, pooled)

Surv_i is FIRM-LEVEL (not per-observation): 1 if that ticker's LONGEST
consecutive-quarter run of valid returns, anywhere in its FULL available
history (not truncated to the observation's own date), reaches >=27 years
(108 quarters). This is a DELIBERATE look-ahead: an observation from a
firm's 3rd year of listing is coded Surv_i=1 if that same firm goes on to
survive 27 more years, which the firm-in-year-3 could not have known. This
reproduces the same biased-by-construction logic as the paper's k=27y
survival-conditioning exhibit (D1a/R28), just embedded as a static firm
label instead of a run-membership filter -- the point of this run per the
V34 spec is to test the CLAIM under the SAME bias the manuscript already
uses, not to fix it.

Large_{i,t} is time-varying: 1 if firm i is in the top 500 by market cap
LAGGED ONE QUARTER, ranked against the full-universe cross-section each
quarter (same construction as D1a).

Two-way clustered (firm x quarter) SEs. This is a SINGLE pooled OLS on the
full corrected panel -- not run per-cell -- so "fresh process per estimate"
here means: one process, one regression, one design matrix built once and
hashed once (there is only one estimate to report, not four).
"""
import os
import hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/D1b_triple_interaction.txt"

log = []
def P(s=""):
    print(s)
    log.append(str(s))

print(f"[pid={os.getpid()}] D1b — fresh process")


def cs_wz(df, col, date_col="q", pct=0.01):
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


P("="*78)
P("D1b — Formal pooled triple-interaction test")
P("="*78)

panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
panel = panel.sort_values(["ticker", "q_ord"]).reset_index(drop=True)

# firm-level, FULL-HISTORY consecutive-run length (deliberate look-ahead) --
# note this uses q_ord over the ticker's ENTIRE listed history in the panel,
# not truncated at any observation date
new_run = (panel["q_ord"] - panel.groupby("ticker")["q_ord"].shift(1)) != 1
panel["run_id"] = new_run.groupby(panel["ticker"]).cumsum()
panel["run_len_q"] = panel.groupby(["ticker", "run_id"])["q_ord"].transform("size")
firm_max_run = panel.groupby("ticker")["run_len_q"].max()
K = 27
THR_Q = K * 4
surv_firms = set(firm_max_run[firm_max_run >= THR_Q].index)
panel["Surv"] = panel["ticker"].isin(surv_firms).astype(int)
P(f"\nSurv_i: {len(surv_firms):,} of {panel['ticker'].nunique():,} firms ever reach a "
  f"{K}-year ({THR_Q}-quarter) consecutive run somewhere in their full history "
  f"(firm-level flag applied to ALL of that firm's observations, including pre-run ones).")

# time-varying Large_{i,t}: top-500 by lagged cap vs FULL-universe cross-section
sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                       columns=["ticker", "dimension", "calendardate", "marketcap"])
mc = sf1[sf1["dimension"] == "ARQ"].copy()
mc["calendardate"] = pd.to_datetime(mc["calendardate"], errors="coerce")
mc = mc.dropna(subset=["calendardate", "marketcap"])
mc = mc[mc["marketcap"] > 0]
mc["q"] = mc["calendardate"].dt.to_period("Q")
mc = mc.sort_values(["ticker", "calendardate"]).drop_duplicates(["ticker", "q"], keep="last")[["ticker", "q", "marketcap"]]
mc = mc.sort_values(["ticker", "q"])
mc["marketcap_lag1"] = mc.groupby("ticker")["marketcap"].shift(1)
panel = panel.merge(mc, on=["ticker", "q"], how="left")
mc_ranked = panel.dropna(subset=["marketcap_lag1"]).copy()
mc_ranked["mc_rank"] = mc_ranked.groupby("q")["marketcap_lag1"].rank(ascending=False, method="first")
panel = panel.merge(mc_ranked[["ticker", "q", "mc_rank"]], on=["ticker", "q"], how="left")
panel["Large"] = (panel["mc_rank"] <= 500).astype(int)
P(f"Large_{{i,t}}: {panel['Large'].sum():,} of {len(panel):,} firm-quarter rows flagged "
  f"top-500 by lagged cap (rows with no lagged-cap coverage are coded Large=0, not dropped).")

# within-quarter cross-sectional z-scoring (corrected-panel default)
panel["ds_z"] = cs_wz(panel, "delta_s")
panel["dh_z"] = cs_wz(panel, "dH_gpm")

# FF5+UMD, compounded monthly->quarterly, pooled as date-level controls
fac = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
fac.index = pd.to_datetime(fac.index)
facq = fac.copy()
facq["q"] = facq.index.to_period("Q")
def cmpd(s): return (1 + s).prod() - 1
ffq = facq.groupby("q").agg({c: cmpd for c in ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]}).reset_index()
panel = panel.merge(ffq, on="q", how="left")

panel["DSxSurv"] = panel["ds_z"] * panel["Surv"]
panel["DSxLarge"] = panel["ds_z"] * panel["Large"]
panel["SurvxLarge"] = panel["Surv"] * panel["Large"]
panel["DSxSurvxLarge"] = panel["ds_z"] * panel["Surv"] * panel["Large"]

xcols = ["ds_z", "dh_z", "Surv", "Large", "DSxSurv", "DSxLarge", "SurvxLarge",
         "DSxSurvxLarge", "Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]
pf = panel.dropna(subset=["ret_next"] + xcols).copy()

first_q, last_q = str(pf["q"].min()), str(pf["q"].max())
n_tickers = pf["ticker"].nunique()
avg_firms_q = pf.groupby("q").size().mean()
P(f"\nPooled regression sample: N={len(pf):,}  tickers={n_tickers:,}  "
  f"avg firms/qtr={avg_firms_q:.1f}  quarters={pf['q'].nunique()}  date_range={first_q}..{last_q}")

X = sm.add_constant(pf[xcols]).values
y = pf["ret_next"].values
design_hash = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()
P(f"design shape={X.shape}  SHA-1={design_hash}  pid={os.getpid()}")

b, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ b
g_firm = pd.Categorical(pf["ticker"]).codes
g_q = pd.Categorical(pf["q"].astype(str)).codes
V = twoway_vcov(X, resid, g_firm, g_q)
se = np.sqrt(np.diag(V))
tstat = b / se
pval = 1 - chi2.cdf(tstat**2, 1)
names = ["const"] + xcols

P(f"\nclusters(firm,quarter)=({g_firm.max()+1},{g_q.max()+1})")
P(f"\n{'term':16}{'coef':>12}{'SE':>12}{'t':>9}{'p':>9}")
for nm, bb, ss, tt, pp in zip(names, b, se, tstat, pval):
    P(f"{nm:16}{bb:>+12.6f}{ss:>12.6f}{tt:>+9.3f}{pp:>9.4f}")

P("\n" + "="*78)
P("The four DeltaS terms (quantity of interest):")
P("="*78)
i_ds = names.index("ds_z")
i_dss = names.index("DSxSurv")
i_dsl = names.index("DSxLarge")
i_dssl = names.index("DSxSurvxLarge")
P(f"  DS_z (main, FB x no-Surv baseline slope)         coef={b[i_ds]:+.6f}  SE={se[i_ds]:.6f}  t={tstat[i_ds]:+.4f}  p={pval[i_ds]:.4f}")
P(f"  DS_z x Surv (survival effect on the DS slope)     coef={b[i_dss]:+.6f}  SE={se[i_dss]:.6f}  t={tstat[i_dss]:+.4f}  p={pval[i_dss]:.4f}")
P(f"  DS_z x Large (size effect on the DS slope)        coef={b[i_dsl]:+.6f}  SE={se[i_dsl]:.6f}  t={tstat[i_dsl]:+.4f}  p={pval[i_dsl]:.4f}")
P(f"  DS_z x Surv x Large (TRIPLE, quantity of interest) coef={b[i_dssl]:+.6f}  SE={se[i_dssl]:.6f}  t={tstat[i_dssl]:+.4f}  p={pval[i_dssl]:.4f}")
P("\nInterpretation: the triple interaction tests whether the survival")
P("effect on the DS slope (DS_z x Surv) itself differs between large-cap and")
P("full-breadth firms. A significant, opposite-signed triple term would")
P("confirm the D1a/R28 pattern (survival effect weaker in large-cap: +1.29")
P("vs +3.21 in t-units) formally rather than by eyeballing two t-stats.")

P("\n" + "="*78)
P("Implied cell-specific DS_z slopes (linear combinations, delta-method SE)")
P("vs. D1a's actual per-cell FM coefficients -- QUANTIFYING THE GAP")
P("="*78)


def lincomb(coef_idx_list, sign_list=None):
    if sign_list is None:
        sign_list = [1.0] * len(coef_idx_list)
    w = np.zeros(len(b))
    for idx, s in zip(coef_idx_list, sign_list):
        w[idx] = s
    est = w @ b
    var = w @ V @ w
    return est, np.sqrt(var)


cells_d1b = {
    "FB_noSurv": lincomb([i_ds]),
    "FB_surv": lincomb([i_ds, i_dss]),
    "LC_noSurv": lincomb([i_ds, i_dsl]),
    "LC_surv": lincomb([i_ds, i_dss, i_dsl, i_dssl]),
}
# D1a's actual per-cell FM coefficients (from D1a_2x2_cell.py runs, this session)
d1a_actual = {
    "FB_noSurv": 0.000087,
    "FB_surv": 0.009865,
    "LC_noSurv": 0.004186,
    "LC_surv": 0.006488,
}
P(f"{'cell':12}{'D1b implied coef':>18}{'D1b SE':>10}{'D1b implied t':>15}{'D1a actual coef':>18}{'gap (D1b-D1a)':>15}")
for cell in cells_d1b:
    est, se_ = cells_d1b[cell]
    t_ = est / se_ if se_ > 0 else np.nan
    gap = est - d1a_actual[cell]
    P(f"{cell:12}{est:>+18.6f}{se_:>10.6f}{t_:>+15.3f}{d1a_actual[cell]:>+18.6f}{gap:>+15.6f}")

P("\nGap is expected and reported, not corrected: D1a re-standardizes ds_z/dh_z")
P("WITHIN each conditioned sub-panel (z-scoring recomputed on the cell's own")
P("cross-section each quarter); D1b's ds_z is z-scored ONCE on the full")
P("unconditioned panel and never re-standardized within the Surv/Large")
P("subgroups. The two are testing related but distinct quantities -- D1a asks")
P("'what is the DS slope when ΔS is re-standardized to the conditioned")
P("sample's own cross-section', D1b asks 'what is the DS slope, holding the")
P("full-panel standardization fixed, as a firm crosses into the Surv/Large")
P("subgroups'. Per spec, no attempt is made to force these into agreement.")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
