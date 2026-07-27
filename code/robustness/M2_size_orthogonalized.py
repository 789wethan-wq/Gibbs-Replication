"""M2 — Size-orthogonalized survivorship test (R4 gate item)

Within the R18 corrected (survivorship-free, full-universe quarterly) panel,
restrict each quarter to the largest ~500 firms by market cap, KEEPING delisted
large caps in. Re-run Model B FM (ΔH + ΔS) and the ΔS quintile sort. This holds
size composition ~fixed at the S&P scale while retaining survivorship-free
coverage — the complement to M1 (which held survivorship fixed and changed
measurement).

  β_ΔS ≈ 0 among survivorship-free large caps  -> survivorship is the driver.
  β_ΔS significantly positive there            -> large-cap phenomenon, not
                                                   survivorship (headline wrong).

Also runs the Table-8-style SIZE-ONLY ladder: top quintile / decile / top-500 /
top-150 by cap each quarter, NO survival requirement, FM t per rung.

Cross-sectional z-scores are recomputed WITHIN each restricted cross-section
(the standardization must live in the analyzed sample), from the raw iVol /
GPM-stability columns saved by R18.

Output: results/revision/M2_size_orthogonalized.txt
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT  = "../results/revision"
os.makedirs(OUT, exist_ok=True)

LOG = []
def say(*a):
    line = " ".join(str(x) for x in a); print(line); LOG.append(line)

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
    for d, grp in panel.groupby(date_col):
        sub = grp[[y_col] + x_cols].dropna()
        if len(sub) < max(min_cs, len(x_cols) + 2): continue
        X = sm.add_constant(sub[x_cols], has_constant="add")
        coefs.append(sm.OLS(sub[y_col], X).fit().params[x_cols].rename(d))
    if not coefs: return {}
    cdf = pd.DataFrame(coefs); out = {}
    for col in x_cols:
        s = cdf[col].dropna(); n = len(s); mean_ = s.mean()
        var = (s**2).mean() - mean_**2
        for l in range(1, min(lags + 1, n)):
            g = ((s.iloc[l:].values - mean_) * (s.iloc[:-l].values - mean_)).mean()
            var += 2 * (1 - l / (lags + 1)) * g
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

# ═══════════════════════════════════════════════════════════════════════════
say("="*64); say("M2 — SIZE-ORTHOGONALIZED SURVIVORSHIP TEST"); say("="*64)

# ── load R18 corrected panel + merge market cap + delisting flag ─────────────
panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
say(f"\nR18 corrected panel: N={len(panel):,}  tickers={panel['ticker'].nunique():,}  "
    f"quarters={panel['q'].nunique()}")

sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                      columns=["ticker","dimension","calendardate","marketcap"])
mc = sf1[sf1["dimension"] == "ARQ"].copy()
mc["calendardate"] = pd.to_datetime(mc["calendardate"], errors="coerce")
mc = mc.dropna(subset=["calendardate","marketcap"])
mc["q"] = mc["calendardate"].dt.to_period("Q")
mc = (mc.sort_values(["ticker","calendardate"])
        .drop_duplicates(["ticker","q"], keep="last")[["ticker","q","marketcap"]])
panel = panel.merge(mc, on=["ticker","q"], how="left")
say(f"Market-cap coverage in panel: {panel['marketcap'].notna().mean():.1%}")

tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
delisted = set(tk.loc[(tk["table"]=="SF1") & (tk["isdelisted"]=="Y"), "ticker"])
panel["is_delisted"] = panel["ticker"].isin(delisted)

def run_arm(df, label, target=None, frac=None):
    """Restrict each quarter by size cut, re-z-score within cut, run Model B + sort."""
    d = df.dropna(subset=["marketcap"]).copy()
    # rank firms within each quarter by market cap (1 = largest)
    d["mc_rank"] = d.groupby("q")["marketcap"].rank(ascending=False, method="first")
    if target is not None:
        d = d[d["mc_rank"] <= target].copy()
    else:
        qsize = d.groupby("q")["marketcap"].transform("size")
        d = d[d["mc_rank"] <= np.ceil(qsize * frac)].copy()
    # re-standardize within the restricted cross-section
    d["ds_z"] = cs_winsorize_zscore(d, "delta_s")
    d["dh_z"] = cs_winsorize_zscore(d, "dH_gpm")
    pf = d.dropna(subset=["ret_next","ds_z","dh_z"])
    pe = d.dropna(subset=["ret_next","ds_z"])
    rb = fama_macbeth_nw(pf, "ret_next", ["dh_z","ds_z"])
    ls_yr, ls_t, _ = quintile_ls(pe, "ds_z")
    _,t_ds,nq = rb.get("ds_z",(np.nan,np.nan,0)); _,t_dh,_ = rb.get("dh_z",(np.nan,np.nan,0))
    avg_n = d.groupby("q").size().mean()
    del_share_obs = d["is_delisted"].mean()
    del_share_firms = d.loc[d["is_delisted"],"ticker"].nunique()/d["ticker"].nunique()
    med_mc = d["marketcap"].median()/1e9
    return dict(label=label, N=len(pf), avg_n=avg_n, nq=nq, t_ds=t_ds, t_dh=t_dh,
                ls_yr=ls_yr, ls_t=ls_t, del_obs=del_share_obs,
                del_firms=del_share_firms, med_mc=med_mc)

# ── PRIMARY M2 ARM: top-500 by cap, survivorship-free ────────────────────────
say("\n" + "-"*64); say("PRIMARY ARM — top-500 by market cap each quarter (delisted kept in)")
r = run_arm(panel, "top-500", target=500)
say(f"  N={r['N']:,}  avg firms/quarter={r['avg_n']:.0f}  quarters={r['nq']}")
say(f"  delisted share: {r['del_obs']:.1%} of obs, {r['del_firms']:.1%} of firms")
say(f"  median mkt cap: ${r['med_mc']:.1f}B")
say(f"  Model B FM: t(ΔS)={r['t_ds']:+.2f}  t(ΔH)={r['t_dh']:+.2f}")
say(f"  ΔS quintile L/S: {r['ls_yr']:+.1f}%/yr  t={r['ls_t']:+.2f}")
say("")
say(f"[M2] R18 top-500-by-cap: N={r['N']:,}, avg firms/quarter={r['avg_n']:.0f}, "
    f"delisted share={r['del_obs']:.1%}, FM t(ΔS)={r['t_ds']:+.2f}, "
    f"FM t(ΔH)={r['t_dh']:+.2f}, quintile L/S={r['ls_yr']:+.1f}%/yr (t={r['ls_t']:+.2f})")

# ── SIZE-ONLY LADDER (Table-8 format, no survival requirement) ───────────────
say("\n" + "-"*64); say("SIZE-ONLY LADDER — each quarter, by cap, NO survival requirement")
say(f"  {'cut':14} {'Nfirms':>7} {'t(ΔS)':>7} {'t(ΔH)':>7} {'L/S%/yr':>8} {'medMC$B':>8}")
rungs = [("top-quintile", None, 0.20),
         ("top-decile",   None, 0.10),
         ("top-500",      500,  None),
         ("top-150",      150,  None)]
for lbl, tgt, fr in rungs:
    rr = run_arm(panel, lbl, target=tgt, frac=fr)
    say(f"  {lbl:14} {rr['avg_n']:7.0f} {rr['t_ds']:+7.2f} {rr['t_dh']:+7.2f} "
        f"{rr['ls_yr']:+8.1f} {rr['med_mc']:8.1f}")
    say(f"    [M2-ladder | cut={lbl}] N firms={rr['avg_n']:.0f}, "
        f"FM t(ΔS)={rr['t_ds']:+.2f}, FM t(ΔH)={rr['t_dh']:+.2f}, "
        f"median mktcap=${rr['med_mc']:.1f}B")

# ── interpretation anchor: full-universe R18 baseline (no size cut) ──────────
say("\n" + "-"*64); say("BASELINE — full-universe R18 (no size cut), same re-run")
d = panel.copy()
d["ds_z"]=cs_winsorize_zscore(d,"delta_s"); d["dh_z"]=cs_winsorize_zscore(d,"dH_gpm")
pf=d.dropna(subset=["ret_next","ds_z","dh_z"]); pe=d.dropna(subset=["ret_next","ds_z"])
rb=fama_macbeth_nw(pf,"ret_next",["dh_z","ds_z"]); lsyr,lst,_=quintile_ls(pe,"ds_z")
say(f"  full-universe: avg firms/q={d.groupby('q').size().mean():.0f}  "
    f"t(ΔS)={rb['ds_z'][1]:+.2f}  t(ΔH)={rb['dh_z'][1]:+.2f}  "
    f"L/S={lsyr:+.1f}%/yr (t={lst:+.2f})  medMC=${d['marketcap'].median()/1e9:.2f}B")

out_txt = f"{OUT}/M2_size_orthogonalized.txt"
with open(out_txt, "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\nSaved: {out_txt}")
