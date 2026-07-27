"""M9-M2 — Entropy analog of the size-orthogonalized survivorship test.

M2 (M2_size_orthogonalized.py) closed the size-composition alternative for iVol:
within the R18 corrected panel restricted to the ~top-500 by cap (delisted large
caps kept in), FM t(ΔS)=+0.58, flat across size rungs -> the ΔS survival ladder
is NOT a size effect. Before we add an entropy ladder (M9-R win) to the paper,
we must run the SAME test on the faithful entropy measure so a reviewer can't
ask "is the entropy ladder just the size tilt?"

This is M2's design verbatim, with the faithful fixed-grid Shannon entropy H
(M9b: 10-bin common grid over pooled return support, 12q rolling window,
dispersion-capturing, Corr(H,ΔS)=+0.73) in place of ΔS. Model B FM = ΔH_GPM + H;
quintile sort on H; primary top-500 arm + size-only ladder + full-universe
baseline, all identical to M2.

Pre-committed interpretation (per spec):
  t(H) ≈ 0 among survivorship-free large caps, flat across rungs (like M2's
        +0.58) -> entropy ladder is NOT size composition; the O-Z win integrates.
  t(H) significantly positive there -> entropy result is partly size; §5.2 says so.

Output: results/revision/M9_M2_entropy_size_orthogonalized.txt
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"; OUT = "../results/revision"
os.makedirs(OUT, exist_ok=True)
LOG = []
def say(*a):
    line = " ".join(str(x) for x in a); print(line); LOG.append(line)

# ── helpers (identical to M2) ────────────────────────────────────────────────
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

# ── faithful fixed-grid Shannon entropy (identical to M9b) ───────────────────
WINDOW, MIN_OBS = 12, 8
def entropy_ticker_fixed(g, edges):
    g = g.sort_values("q"); r = g["ret"].values
    out = pd.Series(np.nan, index=g.index); n = len(g)
    for i in range(MIN_OBS, n + 1):
        w = r[max(0, i - WINDOW):i]; w = w[~np.isnan(w)]
        if len(w) < MIN_OBS: continue
        wc = np.clip(w, edges[0], edges[-1])
        counts, _ = np.histogram(wc, bins=edges)
        p = counts / counts.sum(); p = p[p > 0]
        out.iloc[i - 1] = float(-(p * np.log(p)).sum())
    return out

say("="*66); say("M9-M2 — ENTROPY SIZE-ORTHOGONALIZED SURVIVORSHIP TEST"); say("="*66)

# ── rebuild R18 quarterly returns + compute H_fix10 (identical to M9b) ───────
say("\nRebuilding R18 quarterly returns and computing fixed-grid entropy H...")
tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"]
uni = sf1t[sf1t["category"].str.contains("Domestic Common", na=False) &
           sf1t["exchange"].isin(["NYSE","NASDAQ","NYSEARCA","BATS","NYSEMKT"]) &
           (sf1t["currency"] == "USD")].copy()
uni_set = set(uni["ticker"])
sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                      columns=["ticker","dimension","calendardate","datekey","price","dps"])
arq = sf1[(sf1["dimension"] == "ARQ") & sf1["ticker"].isin(uni_set)].copy()
arq["calendardate"] = pd.to_datetime(arq["calendardate"], errors="coerce")
arq = arq.dropna(subset=["calendardate","price"]); arq = arq[arq["price"] > 0]
arq["q"] = arq["calendardate"].dt.to_period("Q")
arq = arq.sort_values(["ticker","calendardate"]).drop_duplicates(["ticker","q"], keep="last")
arq = arq.sort_values(["ticker","q"]); arq["q_ord"] = arq["q"].apply(lambda p: p.ordinal)
arq["price_prev"] = arq.groupby("ticker")["price"].shift(1)
arq["gap"] = arq["q_ord"] - arq.groupby("ticker")["q_ord"].shift(1)
ret_px = arq["price"] / arq["price_prev"] - 1.0
div_q = (arq["dps"].fillna(0) / 4.0) / arq["price_prev"]
arq["ret"] = np.where(arq["gap"] == 1, ret_px + div_q, np.nan)
arq = arq.dropna(subset=["ret"])
arq["ret"] = arq.groupby("q")["ret"].transform(lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
cnt = arq.groupby("ticker")["ret"].transform("size"); arq = arq[cnt >= 8]
prices = arq[["ticker","q","ret"]].copy()

lo, hi = prices["ret"].quantile(0.005), prices["ret"].quantile(0.995)
edges = np.linspace(lo, hi, 11)   # 10-bin common grid (M9b faithful default)
prices["H"] = prices.groupby("ticker", group_keys=False).apply(
    lambda g: entropy_ticker_fixed(g, edges)).rename("H")
say(f"  pooled support [{lo:+.4f},{hi:+.4f}], 10-bin grid; "
    f"H coverage {prices['H'].notna().mean():.1%}")

# ── load R18 panel, merge H + market cap + delisting flag (M2 style) ─────────
panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
panel = panel.merge(prices[["ticker","q","H"]], on=["ticker","q"], how="left")
say(f"R18 corrected panel: N={len(panel):,}  tickers={panel['ticker'].nunique():,}  "
    f"quarters={panel['q'].nunique()}  H coverage in panel={panel['H'].notna().mean():.1%}")

mc = sf1[sf1["dimension"] == "ARQ"].copy() if False else None
sf1mc = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                        columns=["ticker","dimension","calendardate","marketcap"])
mc = sf1mc[sf1mc["dimension"] == "ARQ"].copy()
mc["calendardate"] = pd.to_datetime(mc["calendardate"], errors="coerce")
mc = mc.dropna(subset=["calendardate","marketcap"])
mc["q"] = mc["calendardate"].dt.to_period("Q")
mc = (mc.sort_values(["ticker","calendardate"])
        .drop_duplicates(["ticker","q"], keep="last")[["ticker","q","marketcap"]])
panel = panel.merge(mc, on=["ticker","q"], how="left")
say(f"Market-cap coverage in panel: {panel['marketcap'].notna().mean():.1%}")

delisted = set(tk.loc[(tk["table"]=="SF1") & (tk["isdelisted"]=="Y"), "ticker"])
panel["is_delisted"] = panel["ticker"].isin(delisted)

# ── M2's run_arm, but sorting/regressing on H (Model B = ΔH_GPM + H) ─────────
def run_arm(df, label, target=None, frac=None):
    d = df.dropna(subset=["marketcap"]).copy()
    d["mc_rank"] = d.groupby("q")["marketcap"].rank(ascending=False, method="first")
    if target is not None:
        d = d[d["mc_rank"] <= target].copy()
    else:
        qsize = d.groupby("q")["marketcap"].transform("size")
        d = d[d["mc_rank"] <= np.ceil(qsize * frac)].copy()
    d["H_z"]  = cs_winsorize_zscore(d, "H")
    d["dh_z"] = cs_winsorize_zscore(d, "dH_gpm")
    pf = d.dropna(subset=["ret_next","H_z","dh_z"])
    pe = d.dropna(subset=["ret_next","H_z"])
    rb = fama_macbeth_nw(pf, "ret_next", ["dh_z","H_z"])
    ls_yr, ls_t, _ = quintile_ls(pe, "H_z")
    _,t_h,nq = rb.get("H_z",(np.nan,np.nan,0)); _,t_dh,_ = rb.get("dh_z",(np.nan,np.nan,0))
    avg_n = d.groupby("q").size().mean()
    del_share_obs = d["is_delisted"].mean()
    del_share_firms = d.loc[d["is_delisted"],"ticker"].nunique()/d["ticker"].nunique()
    med_mc = d["marketcap"].median()/1e9
    return dict(label=label, N=len(pf), avg_n=avg_n, nq=nq, t_h=t_h, t_dh=t_dh,
                ls_yr=ls_yr, ls_t=ls_t, del_obs=del_share_obs,
                del_firms=del_share_firms, med_mc=med_mc)

# ── PRIMARY ARM: top-500 by cap, survivorship-free ──────────────────────────
say("\n" + "-"*66); say("PRIMARY ARM — top-500 by market cap each quarter (delisted kept in)")
r = run_arm(panel, "top-500", target=500)
say(f"  N={r['N']:,}  avg firms/quarter={r['avg_n']:.0f}  quarters={r['nq']}")
say(f"  delisted share: {r['del_obs']:.1%} of obs, {r['del_firms']:.1%} of firms")
say(f"  median mkt cap: ${r['med_mc']:.1f}B")
say(f"  Model B FM: t(H)={r['t_h']:+.2f}  t(ΔH_GPM)={r['t_dh']:+.2f}")
say(f"  H quintile L/S: {r['ls_yr']:+.1f}%/yr  t={r['ls_t']:+.2f}")
say("")
say(f"[M9-M2] R18 top-500-by-cap, entropy H: FM t(H)={r['t_h']:+.2f}, "
    f"FM t(ΔH_GPM)={r['t_dh']:+.2f}, quintile L/S={r['ls_yr']:+.1f}%/yr (t={r['ls_t']:+.2f}), "
    f"N={r['N']:,}, avg firms/qtr={r['avg_n']:.0f}, delisted share={r['del_obs']:.1%}, "
    f"median cap=${r['med_mc']:.1f}B")

# ── SIZE-ONLY LADDER (Table-10 rungs, no survival requirement) ──────────────
say("\n" + "-"*66); say("SIZE-ONLY LADDER — each quarter by cap, NO survival requirement")
say(f"  {'cut':16} {'Nfirms':>7} {'t(H)':>7} {'t(ΔH)':>7} {'L/S%/yr':>8} {'medMC$B':>8}")
rungs = [("top-quintile", None, 0.20),
         ("top-decile",   None, 0.10),
         ("top-500",      500,  None),
         ("top-150",      150,  None)]
for lbl, tgt, fr in rungs:
    rr = run_arm(panel, lbl, target=tgt, frac=fr)
    say(f"  {lbl:16} {rr['avg_n']:7.0f} {rr['t_h']:+7.2f} {rr['t_dh']:+7.2f} "
        f"{rr['ls_yr']:+8.1f} {rr['med_mc']:8.1f}")
    say(f"    [M9-M2-ladder | cut={lbl}] N firms={rr['avg_n']:.0f}, "
        f"FM t(H)={rr['t_h']:+.2f}, L/S={rr['ls_yr']:+.1f}%/yr, "
        f"median cap=${rr['med_mc']:.1f}B")

# ── full-universe baseline (no size cut), same re-run ───────────────────────
say("\n" + "-"*66); say("BASELINE — full-universe R18 (no size cut), same re-run")
d = panel.copy()
d["H_z"]=cs_winsorize_zscore(d,"H"); d["dh_z"]=cs_winsorize_zscore(d,"dH_gpm")
pf=d.dropna(subset=["ret_next","H_z","dh_z"]); pe=d.dropna(subset=["ret_next","H_z"])
rb=fama_macbeth_nw(pf,"ret_next",["dh_z","H_z"]); lsyr,lst,_=quintile_ls(pe,"H_z")
say(f"  full-universe: avg firms/q={d.groupby('q').size().mean():.0f}  "
    f"t(H)={rb['H_z'][1]:+.2f}  t(ΔH)={rb['dh_z'][1]:+.2f}  "
    f"L/S={lsyr:+.1f}%/yr (t={lst:+.2f})  medMC=${d['marketcap'].median()/1e9:.2f}B")
say(f"    [M9-M2-ladder | cut=full-universe] N firms={d.groupby('q').size().mean():.0f}, "
    f"FM t(H)={rb['H_z'][1]:+.2f}, L/S={lsyr:+.1f}%/yr, "
    f"median cap=${d['marketcap'].median()/1e9:.2f}B")

say("\n[M9-M2-note] H = fixed-grid 10-bin Shannon entropy (M9b faithful measure, "
    "Corr(H,ΔS)=+0.73), 12q window; Model B controls for ΔH_GPM exactly as M2's "
    "Model B controls for it. Size rungs, re-standardization, delisted-inclusion, "
    "and L/S annualization (×400) are byte-identical to M2_size_orthogonalized.py.")

out_txt = f"{OUT}/M9_M2_entropy_size_orthogonalized.txt"
with open(out_txt, "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\nSaved: {out_txt}")
