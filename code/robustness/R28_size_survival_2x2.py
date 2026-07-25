"""R28 — Size x survival 2x2. Within the corrected panel (universe and
frequency held fixed), cross NO survival requirement / survival-conditioned
(k=27y, consecutive-quarter run length) against full breadth / large-cap
(top-500 by LAGGED market cap, delisted retained per D4.4). Reports t(ΔS),
t(ΔH), long-short %/yr, N, avg firms/quarter in every cell.

Two cells (full-breadth x {no-survival, k=27}) reproduce
R25_post_review_experiments.py's E1 k=0/k=27 rows exactly, as a consistency
check. Two cells (large-cap x {no-survival, k=27}) are new: large-cap-only
reproduces D4_lagged_cap_rerun.py's top-500-lagged row; large-cap AND
survival-conditioned is the new cell that completes the 2x2.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/R28_size_survival_2x2.txt"

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
P("R28 — Size x survival 2x2 (corrected panel, universe & frequency held fixed)")
P("="*78)

panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
panel = panel.sort_values(["ticker", "q_ord"]).reset_index(drop=True)

# consecutive-quarter run length (identical construction to R25_post_review E1)
new_run = (panel["q_ord"] - panel.groupby("ticker")["q_ord"].shift(1)) != 1
panel["run_id"] = new_run.groupby(panel["ticker"]).cumsum()
panel["run_len_q"] = panel.groupby(["ticker", "run_id"])["q_ord"].transform("size")

# lagged (t-1) market cap, per D4.4
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

K = 27
THR_Q = K * 4
P(f"\nSurvival conditioning: keep observations inside runs of >= {K} years "
  f"({THR_Q} consecutive quarters), per firm.")
P("Large-cap cut: top-500 firms/quarter by LAGGED (t-1) market cap, per D4.4.")
P("Cross-sectional z-scores for ΔH/ΔS are recomputed WITHIN each conditioned")
P("cell (the conditioned/cut panel is treated as its own universe, exactly as")
P("in R25_post_review E1 and D4.4).")


def run_cell(df, label, top500=False):
    # top500 flag here means "df already carries a top500_flag column computed
    # against the FULL-universe cross-section" -- see cell construction below.
    # Recomputing the rank AFTER subsetting to a small survival-conditioned
    # universe (~328 avg firms/qtr) would make a top-500 cut non-binding, which
    # was a real bug caught in the first pass (cell 2,2 came out identical to
    # cell 1,2). So rank is always computed once, on the full panel, upstream.
    d = df.copy()
    if top500:
        d = d[d["top500_flag"]].copy()
    d["ds_z"] = cs_wz(d, "delta_s")
    d["dh_z"] = cs_wz(d, "dH_gpm")
    pf = d.dropna(subset=["ret_next", "ds_z", "dh_z"])
    pe = d.dropna(subset=["ret_next", "ds_z"])
    fm = fama_macbeth_nw(pf, "ret_next", ["dh_z", "ds_z"])
    ls_ann, ls_t, ls_Tq = quintile_ls(pe, "ds_z")
    avg_n = pf.groupby("q").size().mean()
    n_tick = d["ticker"].nunique()
    med_mc = d["marketcap"].median() / 1e6 if "marketcap" in d else np.nan
    t_ds = fm.get("ds_z", {}).get("t", np.nan)
    t_dh = fm.get("dh_z", {}).get("t", np.nan)
    c_ds = fm.get("ds_z", {}).get("coef", np.nan)
    Tq = fm.get("ds_z", {}).get("n", 0)
    P(f"\n[{label}]")
    P(f"  N_obs(FM)={len(pf):,}  N_tickers={n_tick:,}  avg firms/qtr={avg_n:.1f}  "
      f"T_quarters={Tq}  med.cap=${med_mc:,.1f}M")
    P(f"  t(ΔS)={t_ds:+.3f}  coef(ΔS)={c_ds:+.6f}  t(ΔH)={t_dh:+.3f}")
    P(f"  Quintile L/S={ls_ann*100:+.2f}%/yr  t={ls_t:+.3f}  T_quarters(L/S)={ls_Tq}")
    return dict(label=label, N=len(pf), n_tick=n_tick, avg_n=avg_n, Tq=Tq, med_mc=med_mc,
                t_ds=t_ds, c_ds=c_ds, t_dh=t_dh, ls_ann=ls_ann, ls_t=ls_t)


# top-500 rank computed ONCE on the FULL panel's quarterly cross-section, so
# "large-cap" means the same fixed-size-in-the-full-universe cut in every cell
mc_ranked = panel.dropna(subset=["marketcap_lag1"]).copy()
mc_ranked["mc_rank"] = mc_ranked.groupby("q")["marketcap_lag1"].rank(ascending=False, method="first")
panel = panel.merge(mc_ranked[["ticker", "q", "mc_rank"]], on=["ticker", "q"], how="left")
panel["top500_flag"] = panel["mc_rank"] <= 500
P(f"\ntop-500 flag set on {panel['top500_flag'].sum():,} of {len(panel):,} full-panel rows "
  f"(computed against the FULL-universe quarterly cross-section, before any survival filter)")

cells = {}
P("\n" + "-"*78)
P("CELL (1,1): Full breadth, NO survival requirement")
P("-"*78)
cells["FB_noSurv"] = run_cell(panel, "Full breadth x No-survival-requirement", top500=False)

P("\n" + "-"*78)
P(f"CELL (1,2): Full breadth, survival-conditioned k={K}y")
P("-"*78)
sub_surv = panel[panel["run_len_q"] >= THR_Q].copy()
cells["FB_surv"] = run_cell(sub_surv, f"Full breadth x Survival-conditioned(k={K})", top500=False)

P("\n" + "-"*78)
P("CELL (2,1): Large-cap (top-500, lagged cap), NO survival requirement")
P("-"*78)
cells["LC_noSurv"] = run_cell(panel, "Large-cap(top500) x No-survival-requirement", top500=True)

P("\n" + "-"*78)
P(f"CELL (2,2): Large-cap (top-500, lagged cap), survival-conditioned k={K}y")
P("-"*78)
cells["LC_surv"] = run_cell(sub_surv, f"Large-cap(top500) x Survival-conditioned(k={K})", top500=True)
n_lc_in_surv = sub_surv["top500_flag"].sum()
P(f"\n(Diagnostic: of the {len(sub_surv):,} survival-conditioned rows, "
  f"{n_lc_in_surv:,} ({n_lc_in_surv/len(sub_surv)*100:.1f}%) also carry the "
  f"full-universe top-500 flag -- this is the actual binding overlap, not a "
  f"post-hoc-shrunk rank.)")

P("\n" + "="*78)
P("R28 SUMMARY TABLE")
P("="*78)
P(f"{'Cell':30}{'t(dS)':>9}{'t(dH)':>9}{'L/S%/yr':>10}{'N':>10}{'avg n/q':>9}{'medCap$M':>11}")
for k, r in cells.items():
    P(f"{r['label']:30}{r['t_ds']:>+9.3f}{r['t_dh']:>+9.3f}{r['ls_ann']*100:>+10.2f}"
      f"{r['N']:>10,}{r['avg_n']:>9.1f}{r['med_mc']:>11,.1f}")

P("\n" + "="*78)
P("R28 DECOMPOSITION — does survival conditioning move t(ΔS) within BOTH size")
P("strata, and does size move it within BOTH survival strata?")
P("="*78)
d_survival_effect_fullbreadth = cells["FB_surv"]["t_ds"] - cells["FB_noSurv"]["t_ds"]
d_survival_effect_largecap = cells["LC_surv"]["t_ds"] - cells["LC_noSurv"]["t_ds"]
d_size_effect_nosurv = cells["LC_noSurv"]["t_ds"] - cells["FB_noSurv"]["t_ds"]
d_size_effect_surv = cells["LC_surv"]["t_ds"] - cells["FB_surv"]["t_ds"]
P(f"Survival effect on t(ΔS), within full-breadth: {d_survival_effect_fullbreadth:+.3f} "
  f"({cells['FB_noSurv']['t_ds']:+.2f} -> {cells['FB_surv']['t_ds']:+.2f})")
P(f"Survival effect on t(ΔS), within large-cap:    {d_survival_effect_largecap:+.3f} "
  f"({cells['LC_noSurv']['t_ds']:+.2f} -> {cells['LC_surv']['t_ds']:+.2f})")
P(f"Size effect on t(ΔS), within no-survival-req:  {d_size_effect_nosurv:+.3f} "
  f"({cells['FB_noSurv']['t_ds']:+.2f} -> {cells['LC_noSurv']['t_ds']:+.2f})")
P(f"Size effect on t(ΔS), within survival-cond:    {d_size_effect_surv:+.3f} "
  f"({cells['FB_surv']['t_ds']:+.2f} -> {cells['LC_surv']['t_ds']:+.2f})")
P("")
if abs(d_survival_effect_fullbreadth) > abs(d_size_effect_nosurv) and abs(d_survival_effect_largecap) > abs(d_size_effect_surv):
    P("VERDICT: survival conditioning moves t(ΔS) sharply within BOTH size strata; "
      "size moves it comparatively little within BOTH survival strata -> survivorship "
      "attribution is the dominant channel, cleanly separated from size (Majors 1/3/6).")
elif abs(d_size_effect_nosurv) > abs(d_survival_effect_fullbreadth) and abs(d_size_effect_surv) > abs(d_survival_effect_largecap):
    P("VERDICT: size moves t(ΔS) more than survival conditioning within both strata -> "
      "the paper's central attribution needs restating toward size, not survivorship.")
else:
    P("VERDICT: MIXED -- survival dominates in one stratum, size dominates (or is comparable) "
      "in the other. Report both magnitudes; do not force a single dominant-channel claim.")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
