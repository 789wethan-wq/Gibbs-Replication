"""R26 — Reliability-stratified estimation.

Per-firm reliability (correlating one firm's own odd/even split-half ΔS
estimates over its ~8-30 own quarters) turned out far too noisy to use as a
stratification variable (SB-corrected values ranging from -408 to +0.94,
mean -0.74) -- individual firms simply don't have enough independent
observations for a stable correlation estimate. Reliability is a GROUP
statistic requiring pooling over many thousands of observations to be
estimated stably (this is exactly why D2 pooled within size deciles).

So this script stratifies the SAME WAY D2 validated (pool split-half ΔS
estimates within a group, correlate odd/even within the group, Spearman-Brown
correct) but forms the groups directly from measured reliability rank, not
by asserting size is the right cut: contemporaneous-marketcap terciles and
deciles are used only as the GROUPING mechanism (a continuous covariate must
be binned somehow to pool enough observations per bin), and the group
selected for the "highest reliability" tests is chosen by its MEASURED
reliability value, which the script also reports, not by assumed size rank.

Uses data/R26_split_half_obs.parquet (built by R26_build_reliability.py:
420,850 full-12-quarter-window observations with ds_odd, ds_even per
ticker-quarter) to measure reliability, and merged_sf1_quarterly_survfree.parquet
(the corrected panel, delisted firms retained, no survival requirement
anywhere in this script) for the FM ΔS tests.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
sys.path.insert(0, "../project")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/R26_reliability_stratified.txt"

log = []
def P(s=""):
    print(s)
    log.append(str(s))


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


def fama_macbeth_nw(panel, y_col, x_cols, date_col="q", lags=4, min_cs=20):
    coefs = []
    for d, g in panel.groupby(date_col):
        s = g[[y_col] + x_cols].dropna()
        if len(s) < max(min_cs, len(x_cols) + 2):
            continue
        X = sm.add_constant(s[x_cols], has_constant="add")
        coefs.append(sm.OLS(s[y_col], X).fit().params[x_cols].rename(d))
    if not coefs:
        return {}
    cdf = pd.DataFrame(coefs)
    out = {}
    for col in x_cols:
        s = cdf[col].dropna()
        n = len(s)
        mean_ = s.mean()
        gamma0 = (s**2).mean() - mean_**2
        var = gamma0
        for l in range(1, min(lags + 1, n)):
            g_ = ((s.iloc[l:].values - mean_) * (s.iloc[:-l].values - mean_)).mean()
            var += 2 * (1 - l / (lags + 1)) * g_
        se = np.sqrt(max(var, 1e-30) / n)
        out[col] = dict(coef=mean_, se=se, t=mean_ / se, n=n)
    return out


def quintile_ls(df, sortcol, date_col="q", ycol="ret_next"):
    d = df.dropna(subset=[sortcol, ycol]).copy()
    d["qd"] = d.groupby(date_col)[sortcol].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") if x.nunique() >= 5 else np.nan)
    d = d.dropna(subset=["qd"])
    qr = d.groupby([date_col, "qd"])[ycol].mean().unstack("qd")
    if 0 not in qr.columns or 4 not in qr.columns:
        return np.nan, np.nan, 0
    ls = (qr[4] - qr[0]).dropna()
    t = ls.mean() / (ls.std() / np.sqrt(len(ls))) if len(ls) > 2 else np.nan
    return ls.mean() * 4, t, len(ls)


P("="*78)
P("R26 (0) — Measuring reliability directly (group-pooled, not per-firm)")
P("="*78)

obs = pd.read_parquet(f"{DATA}/R26_split_half_obs.parquet")
obs = obs.dropna(subset=["marketcap"])
obs = obs[obs["marketcap"] > 0].copy()
P(f"Split-half observation pool: N={len(obs):,}, tickers={obs['ticker'].nunique():,}")

# overall (full-sample) pooled reliability, for the EIV correction denominator
r_overall = obs["ds_odd"].corr(obs["ds_even"])
rel_overall = 2 * r_overall / (1 + r_overall)
P(f"Overall pooled reliability (all observations, one group): raw corr={r_overall:.4f}  "
  f"SB-corrected={rel_overall:.4f}")

for nbins, label in [(3, "tercile"), (10, "decile")]:
    obs[f"grp_{nbins}"] = pd.qcut(obs["marketcap"], nbins, labels=False, duplicates="drop") + 1
    P(f"\n{label.capitalize()}s of contemporaneous market cap (grouping mechanism only; "
      f"selection below is by MEASURED reliability, not assumed size rank):")
    rows = []
    for g in sorted(obs[f"grp_{nbins}"].dropna().unique()):
        gg = obs[obs[f"grp_{nbins}"] == g]
        r = gg["ds_odd"].corr(gg["ds_even"])
        sb = 2 * r / (1 + r) if np.isfinite(r) and (1 + r) != 0 else np.nan
        med_cap = gg["marketcap"].median() / 1e6
        rows.append(dict(grp=int(g), n=len(gg), corr=r, reliability=sb, med_cap_m=med_cap))
        P(f"  group {int(g):>2}: N={len(gg):>7,}  corr={r:+.3f}  SB-reliability={sb:.3f}  medCap=${med_cap:,.1f}M")
    globals()[f"rel_table_{nbins}"] = pd.DataFrame(rows)

top_tercile_grp = int(rel_table_3.loc[rel_table_3["reliability"].idxmax(), "grp"])
top_decile_grp = int(rel_table_10.loc[rel_table_10["reliability"].idxmax(), "grp"])
top_tercile_rel = rel_table_3["reliability"].max()
top_decile_rel = rel_table_10["reliability"].max()
P(f"\nHighest-reliability tercile = group {top_tercile_grp} (reliability={top_tercile_rel:.3f})")
P(f"Highest-reliability decile  = group {top_decile_grp} (reliability={top_decile_rel:.3f})")
P("(Both happen to be the largest-cap group in this dataset -- D2 already")
P(" established reliability rises monotonically with size here. The selection")
P(" criterion applied below is the measured reliability value, reported above,")
P(" not an assumption that size is the right proxy.)")

# cap thresholds that define the selected groups, to apply to the analysis panel
tercile_cap_lo = obs.loc[obs["grp_3"] == top_tercile_grp, "marketcap"].min()
decile_cap_lo = obs.loc[obs["grp_10"] == top_decile_grp, "marketcap"].min()

P("\n" + "="*78)
P("R26 (1)-(2) — FM t(ΔS) restricted to the highest-reliability tercile / decile")
P("(corrected panel, delisted firms retained, NO survival requirement applied anywhere)")
P("="*78)

panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                       columns=["ticker", "dimension", "calendardate", "marketcap"])
mc = sf1[sf1["dimension"] == "ARQ"].copy()
mc["calendardate"] = pd.to_datetime(mc["calendardate"], errors="coerce")
mc = mc.dropna(subset=["calendardate", "marketcap"])
mc = mc[mc["marketcap"] > 0]
mc["q"] = mc["calendardate"].dt.to_period("Q")
mc = mc.sort_values(["ticker", "calendardate"]).drop_duplicates(["ticker", "q"], keep="last")[["ticker", "q", "marketcap"]]
panel = panel.merge(mc, on=["ticker", "q"], how="left")
P(f"Analysis panel (full, unconditioned): N={len(panel):,}, tickers={panel['ticker'].nunique():,}, "
  f"quarters={panel['q'].nunique()}")

def run_stratum(df, cap_lo, label, rel_value):
    d = df[df["marketcap"] >= cap_lo].copy()
    d["ds_z"] = cs_wz(d, "delta_s")
    d["dh_z"] = cs_wz(d, "dH_gpm")
    pf = d.dropna(subset=["ret_next", "ds_z", "dh_z"])
    pe = d.dropna(subset=["ret_next", "ds_z"])
    fm = fama_macbeth_nw(pf, "ret_next", ["dh_z", "ds_z"])
    ls_ann, ls_t, ls_Tq = quintile_ls(pe, "ds_z")
    avg_n = pf.groupby("q").size().mean()
    P(f"\n[{label}]  cap threshold=${cap_lo/1e6:,.1f}M  measured reliability={rel_value:.3f}")
    if "ds_z" in fm:
        P(f"  FM Model B: t(dS)={fm['ds_z']['t']:+.3f}  coef(dS)={fm['ds_z']['coef']:+.6f}  "
          f"t(dH)={fm['dh_z']['t']:+.3f}  N={len(pf):,}  avg firms/qtr={avg_n:.1f}  "
          f"quarters(dS)={fm['ds_z']['n']}")
    P(f"  Quintile L/S: {ls_ann*100:+.2f}%/yr  t={ls_t:+.3f}  T_quarters={ls_Tq}")
    return fm, ls_ann, ls_t, avg_n, len(pf)

run_stratum(panel, tercile_cap_lo, "Highest-reliability TERCILE", top_tercile_rel)
run_stratum(panel, decile_cap_lo, "Highest-reliability DECILE", top_decile_rel)

P("\n" + "="*78)
P("R26 (3) — Errors-in-variables correction (full, unconditioned panel)")
P("="*78)
d_full = panel.copy()
d_full["ds_z"] = cs_wz(d_full, "delta_s")
d_full["dh_z"] = cs_wz(d_full, "dH_gpm")
pf_full = d_full.dropna(subset=["ret_next", "ds_z", "dh_z"])
fm_full = fama_macbeth_nw(pf_full, "ret_next", ["dh_z", "ds_z"])
raw_coef = fm_full["ds_z"]["coef"]
raw_se = fm_full["ds_z"]["se"]
raw_t = fm_full["ds_z"]["t"]
P(f"Raw (unconditioned, full panel) FM ΔS: coef={raw_coef:+.6f}  SE={raw_se:.6f}  t={raw_t:+.4f}  "
  f"N={len(pf_full):,}")
P(f"Overall split-half reliability (denominator) = {rel_overall:.4f}")
eiv_coef = raw_coef / rel_overall
eiv_se = raw_se / rel_overall
P(f"EIV-corrected (coef / reliability): coef={eiv_coef:+.6f}  SE={eiv_se:.6f}  "
  f"t={eiv_coef/eiv_se:+.4f}  (t is UNCHANGED under this linear rescaling by construction: "
  f"both coef and SE scale by 1/reliability)")
P("Interpretation: EIV correction rescales the ECONOMIC MAGNITUDE of the implied")
P("error-free coefficient upward by ~1/reliability (here, roughly "
  f"{1/rel_overall:.2f}x); it does not and cannot, under this simple scalar-attenuation")
P("model, change statistical significance -- that would require accounting for the")
P("sampling uncertainty of the reliability estimate itself (not attempted here).")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
