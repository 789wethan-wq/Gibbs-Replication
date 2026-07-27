"""D4.4 rerun — lagged vs. contemporaneous market cap in the Table 10/15
size-cut ladder.

M2_size_orthogonalized.py joins market cap on the SAME quarter q as the
sorted/measured return (calendardate -> q, merged onto the panel by
(ticker,q)). That embeds the quarter's own return in the sort: a firm that
crashed this quarter ends the quarter with a smaller market cap, so it is
mechanically more likely to fall into a smaller size bucket in the very
quarter whose return is being measured, biasing size-return correlations.

This script reruns the exact same size-cut ladder (top quintile / decile /
top-500 / top-150 by cap, full universe baseline) twice: once reproducing
the original contemporaneous-cap construction as a sanity check, and once
with market cap LAGGED ONE QUARTER (last quarter's cap decides this
quarter's bucket), reporting FM t(ΔS) and t(ΔH) side by side.

Outputs: results/revision/D4_lagged_cap_rerun.txt
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"; OUT = "../results/revision"; os.makedirs(OUT, exist_ok=True)
LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

def cs_winsorize_zscore(df, col, date_col="q", pct=0.01):
    def _wz(x):
        x = x.dropna()
        if len(x) < 5: return pd.Series(np.nan, index=x.index)
        lo, hi = x.quantile(pct), x.quantile(1 - pct)
        xc = x.clip(lo, hi); std = xc.std()
        if std < 1e-10: return pd.Series(np.nan, index=x.index)
        return (xc - xc.mean()) / std
    return df.groupby(date_col)[col].transform(_wz)

def fama_macbeth_nw(panel, y_col, x_cols, date_col="q", lags=4, min_cs=20):
    coefs = []
    for d, g in panel.groupby(date_col):
        s = g[[y_col] + x_cols].dropna()
        if len(s) < max(min_cs, len(x_cols) + 2): continue
        X = sm.add_constant(s[x_cols], has_constant="add")
        coefs.append(sm.OLS(s[y_col], X).fit().params[x_cols].rename(d))
    if not coefs: return {}
    cdf = pd.DataFrame(coefs)
    out = {}
    for col in x_cols:
        s = cdf[col].dropna(); n = len(s); mean_ = s.mean()
        gamma0 = (s**2).mean() - mean_**2; var = gamma0
        for l in range(1, min(lags + 1, n)):
            g_ = ((s.iloc[l:].values - mean_) * (s.iloc[:-l].values - mean_)).mean()
            var += 2 * (1 - l / (lags + 1)) * g_
        se = np.sqrt(max(var, 1e-30) / n)
        out[col] = (mean_, mean_ / se, n)
    return out

def quintile_ls(df, sortcol):
    d = df.dropna(subset=[sortcol,"ret_next"]).copy()
    d["qd"] = d.groupby("q")[sortcol].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
        if x.nunique() >= 5 else np.nan)
    d = d.dropna(subset=["qd"])
    qr = d.groupby(["q","qd"])["ret_next"].mean().unstack("qd")
    ls = (qr.get(4) - qr.get(0)).dropna()
    t = ls.mean()/(ls.std()/np.sqrt(len(ls))) if len(ls) > 2 else np.nan
    return ls.mean()*400, t, len(ls)

def run_arm(df, capcol, label, target=None, frac=None):
    d = df.dropna(subset=[capcol]).copy()
    d["mc_rank"] = d.groupby("q")[capcol].rank(ascending=False, method="first")
    if target is not None:
        d = d[d["mc_rank"] <= target].copy()
    else:
        qsize = d.groupby("q")[capcol].transform("size")
        d = d[d["mc_rank"] <= np.ceil(qsize * frac)].copy()
    d["ds_z"] = cs_winsorize_zscore(d, "delta_s")
    d["dh_z"] = cs_winsorize_zscore(d, "dH_gpm")
    pf = d.dropna(subset=["ret_next","ds_z","dh_z"])
    pe = d.dropna(subset=["ret_next","ds_z"])
    rb = fama_macbeth_nw(pf, "ret_next", ["dh_z","ds_z"])
    ls_yr, ls_t, _ = quintile_ls(pe, "ds_z")
    _,t_ds,nq = rb.get("ds_z",(np.nan,np.nan,0)); _,t_dh,_ = rb.get("dh_z",(np.nan,np.nan,0))
    avg_n = d.groupby("q").size().mean()
    med_mc = d[capcol].median()/1e9
    return dict(label=label, N=len(pf), avg_n=avg_n, nq=nq, t_ds=t_ds, t_dh=t_dh,
                ls_yr=ls_yr, ls_t=ls_t, med_mc=med_mc)

say("="*72); say("D4.4 RERUN — LAGGED VS CONTEMPORANEOUS MARKET CAP, SIZE-CUT LADDER"); say("="*72)

panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
say(f"\nR18 panel: N={len(panel):,}  tickers={panel['ticker'].nunique():,}  quarters={panel['q'].nunique()}")

sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                      columns=["ticker","dimension","calendardate","marketcap"])
mc = sf1[sf1["dimension"] == "ARQ"].copy()
mc["calendardate"] = pd.to_datetime(mc["calendardate"], errors="coerce")
mc = mc.dropna(subset=["calendardate","marketcap"])
mc = mc[mc["marketcap"] > 0]
mc["q"] = mc["calendardate"].dt.to_period("Q")
mc = (mc.sort_values(["ticker","calendardate"])
        .drop_duplicates(["ticker","q"], keep="last")[["ticker","q","marketcap"]])
mc = mc.sort_values(["ticker","q"])
mc["marketcap_lag1"] = mc.groupby("ticker")["marketcap"].shift(1)

panel = panel.merge(mc, on=["ticker","q"], how="left")
say(f"Contemporaneous market-cap coverage: {panel['marketcap'].notna().mean():.1%}")
say(f"Lagged (t-1) market-cap coverage:    {panel['marketcap_lag1'].notna().mean():.1%}")

rungs = [("top-quintile", None, 0.20),
         ("top-decile",   None, 0.10),
         ("top-500",      500,  None),
         ("top-150",      150,  None)]

say("\n" + "-"*104)
say(f"{'Cut':14}{'Cap timing':>16}{'N firms/q':>11}{'N obs (FM)':>12}{'Quarters':>10}{'t(ΔS)':>9}{'t(ΔH)':>9}{'L/S %/yr':>11}{'medMC $B':>10}")
say("-"*104)
results = []
for lbl, tgt, fr in rungs:
    for capcol, tag in [("marketcap", "contemporaneous"), ("marketcap_lag1", "lagged (t-1)")]:
        r = run_arm(panel, capcol, lbl, target=tgt, frac=fr)
        say(f"{lbl:14}{tag:>16}{r['avg_n']:>11.0f}{r['N']:>12,}{r['nq']:>10}{r['t_ds']:>+9.2f}{r['t_dh']:>+9.2f}"
            f"{r['ls_yr']:>+11.1f}{r['med_mc']:>10.1f}")
        results.append({"cut": lbl, "cap_timing": tag, **r})

say("-"*90)
say("\nFull universe baseline (no size cut, no cap column needed):")
d = panel.copy()
d["ds_z"] = cs_winsorize_zscore(d, "delta_s"); d["dh_z"] = cs_winsorize_zscore(d, "dH_gpm")
pf = d.dropna(subset=["ret_next","ds_z","dh_z"]); pe = d.dropna(subset=["ret_next","ds_z"])
rb = fama_macbeth_nw(pf, "ret_next", ["dh_z","ds_z"]); lsyr, lst, _ = quintile_ls(pe, "ds_z")
say(f"  full-universe: t(ΔS)={rb['ds_z'][1]:+.2f}  t(ΔH)={rb['dh_z'][1]:+.2f}  L/S={lsyr:+.1f}%/yr (t={lst:+.2f})")

res_df = pd.DataFrame(results)
say("\n" + "#"*72); say("# CONTEMPORANEOUS VS LAGGED, SIDE BY SIDE"); say("#"*72)
for lbl in [r[0] for r in rungs]:
    sub = res_df[res_df["cut"] == lbl]
    contemp = sub[sub["cap_timing"] == "contemporaneous"].iloc[0]
    lagged = sub[sub["cap_timing"] == "lagged (t-1)"].iloc[0]
    delta_t = lagged["t_ds"] - contemp["t_ds"]
    say(f"  {lbl:14} t(ΔS) contemporaneous={contemp['t_ds']:+.2f}  lagged={lagged['t_ds']:+.2f}  "
        f"Δ={delta_t:+.2f}  ({'MATERIAL' if abs(delta_t) > 0.5 else 'similar'})")

with open(f"{OUT}/D4_lagged_cap_rerun.txt", "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\nSaved: {OUT}/D4_lagged_cap_rerun.txt")
