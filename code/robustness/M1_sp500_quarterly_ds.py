"""M1 — Constant-measurement survivorship test (R4 gate item, HIGHEST PRIORITY)

Rebuilds ΔS on the 462-name S&P 500 panel using the IDENTICAL R18 construction
(12-quarter rolling FF3 residual iVol on filing-date-spaced SF1 quarterly
returns), survivorship held at maximum (all S&P names kept). Re-runs Model B FM
(ΔH + ΔS) and the ΔS quintile sort on this panel.

The point: the primary S&P panel measures ΔS as 36-month monthly iVol; R18
measures it as 12-quarter iVol. If t(ΔS) collapses even here — same firms, same
survivorship, ONLY the measurement changed to quarterly — then the R18 collapse
is (partly) a measurement-frequency artifact and the headline attribution is not
established. If t(ΔS) stays significant, differential measurement error is dead
and survivorship attribution is clean.

This is R18 with the universe restricted to the 462 S&P names. All other
construction choices are byte-identical to R18_sf1_quarterly_survfree.py.

Output: results/revision/M1_sp500_quarterly.txt
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT  = "../results/revision"
os.makedirs(OUT, exist_ok=True)

LOG = []
def say(*a):
    line = " ".join(str(x) for x in a)
    print(line); LOG.append(line)

# ── helpers (identical to R18) ───────────────────────────────────────────────
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
        res = sm.OLS(sub[y_col], X).fit()
        coefs.append(res.params[x_cols].rename(d))
    if not coefs: return {}, pd.DataFrame()
    cdf = pd.DataFrame(coefs)
    out = {}
    for col in x_cols:
        s = cdf[col].dropna(); n = len(s); mean_ = s.mean()
        gamma0 = (s**2).mean() - mean_**2; var = gamma0
        for l in range(1, min(lags + 1, n)):
            g = ((s.iloc[l:].values - mean_) * (s.iloc[:-l].values - mean_)).mean()
            var += 2 * (1 - l / (lags + 1)) * g
        se = np.sqrt(max(var, 1e-30) / n)
        out[col] = (mean_, mean_ / se, n)
    return out, cdf

# ═══════════════════════════════════════════════════════════════════════════
say("="*64); say("M1 — S&P 500 QUARTERLY-ΔS PANEL (constant-measurement test)"); say("="*64)

# ── STEP 1: universe = the 462 S&P names from the primary monthly panel ──────
v = pd.read_parquet(f"{DATA}/variables_monthly.parquet")
uni_set = set(v["stock_id"].unique())
say(f"\nS&P monthly-panel universe: {len(uni_set)} names (survivorship held at max)")

# ── BENCHMARK: primary monthly-panel Model B FM (36m iVol), apples reference ─
# IMPORTANT: use the ACCOUNTING ΔH (dH_gpm_z) — the same channel R18/M1 use and
# the manuscript headline uses. The price-based composite DH_z is ~collinear
# with the iVol ΔS (corr -0.85) and spuriously crushes t(ΔS) in FM to ~1.4;
# dH_gpm_z (corr -0.26) is the correct comparison and gives the headline t~4.7.
def fm_monthly(panel, y, xs, date_col="date", lags=6, min_cs=20):
    coefs=[]
    for d,grp in panel.groupby(date_col):
        sub=grp[[y]+xs].dropna()
        if len(sub)<max(min_cs,len(xs)+2): continue
        X=sm.add_constant(sub[xs],has_constant="add")
        coefs.append(sm.OLS(sub[y],X).fit().params[xs].rename(d))
    cdf=pd.DataFrame(coefs); out={}
    for c in xs:
        s=cdf[c].dropna(); n=len(s); m=s.mean(); var=(s**2).mean()-m**2
        for l in range(1,min(lags+1,n)):
            g=((s.iloc[l:].values-m)*(s.iloc[:-l].values-m)).mean()
            var+=2*(1-l/(lags+1))*g
        out[c]=(m, m/np.sqrt(max(var,1e-30)/n), n)
    return out
_ma = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
_ma["date"] = pd.to_datetime(_ma["date"])
_ma["dH_gpm_z"] = cs_winsorize_zscore(_ma, "dH_gpm", date_col="date")
mb = fm_monthly(_ma, "ret_next_month", ["dH_gpm_z","DS_z"])
say(f"  [benchmark] monthly-panel Model B FM (36m iVol, accounting ΔH, NW-6): "
    f"t(ΔS)={mb['DS_z'][1]:+.2f}  t(ΔH)={mb['dH_gpm_z'][1]:+.2f}  (Tm={mb['DS_z'][2]})")

# ── STEP 2: quarterly split-adjusted prices + returns  (R18 STEP 2 verbatim) ─
say("\nLoading SF1 ARQ price series (identical R18 construction)...")
sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                      columns=["ticker","dimension","calendardate","datekey",
                               "price","dps"])
arq = sf1[(sf1["dimension"] == "ARQ") & sf1["ticker"].isin(uni_set)].copy()
arq["calendardate"] = pd.to_datetime(arq["calendardate"], errors="coerce")
arq = arq.dropna(subset=["calendardate","price"])
arq = arq[arq["price"] > 0]
arq["q"] = arq["calendardate"].dt.to_period("Q")
arq = (arq.sort_values(["ticker","calendardate"])
          .drop_duplicates(["ticker","q"], keep="last"))
arq = arq.sort_values(["ticker","q"])
arq["q_ord"] = arq["q"].apply(lambda p: p.ordinal)
arq["price_prev"] = arq.groupby("ticker")["price"].shift(1)
arq["gap"] = arq["q_ord"] - arq.groupby("ticker")["q_ord"].shift(1)
ret_px = arq["price"] / arq["price_prev"] - 1.0
div_q  = (arq["dps"].fillna(0) / 4.0) / arq["price_prev"]
arq["ret"] = np.where(arq["gap"] == 1, ret_px + div_q, np.nan)
arq = arq.dropna(subset=["ret"])
arq["ret"] = arq.groupby("q")["ret"].transform(
    lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
cnt = arq.groupby("ticker")["ret"].transform("size")
arq = arq[cnt >= 8]
say(f"Quarterly return obs: {len(arq):,} | tickers: {arq['ticker'].nunique():,} "
    f"| quarters: {arq['q'].nunique()}")
prices = arq[["ticker","q","ret"]].copy()

# ── STEP 3: quarterly FF3 factors (R18 verbatim) ────────────────────────────
fac = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
fac.index = pd.to_datetime(fac.index)
facq = fac.copy(); facq["q"] = facq.index.to_period("Q")
def cmpd(s): return (1 + s).prod() - 1
ff = facq.groupby("q").agg({"Mkt_RF":cmpd,"SMB":cmpd,"HML":cmpd,"RF":cmpd}).reset_index()

# ── STEP 4: ΔS — rolling 12-quarter FF3 residual iVol (R18 verbatim) ─────────
say("\nComputing quarterly iVol (12q rolling FF3)...")
px = prices.merge(ff, on="q", how="left")
px["exret"] = px["ret"] - px["RF"]
def ivol_ticker(g, window=12, min_obs=8):
    g = g.sort_values("q")
    out = pd.Series(np.nan, index=g.index)
    Xall = g[["Mkt_RF","SMB","HML"]].values
    yall = g["exret"].values
    n = len(g)
    for i in range(min_obs, n + 1):
        lo = max(0, i - window)
        Xs = Xall[lo:i]; ys = yall[lo:i]
        if len(ys) < min_obs or np.isnan(Xs).any() or np.isnan(ys).any():
            continue
        Xc = np.column_stack([np.ones(len(Xs)), Xs])
        beta, *_ = np.linalg.lstsq(Xc, ys, rcond=None)
        resid = ys - Xc @ beta
        out.iloc[i - 1] = resid.std(ddof=1)
    return out
px["delta_s"] = px.groupby("ticker", group_keys=False).apply(lambda g: ivol_ticker(g))
ds = px[["ticker","q","delta_s"]].dropna()
say(f"ΔS obs: {len(ds):,} | tickers: {ds['ticker'].nunique():,}")

# ── STEP 5: ΔH — GPM stability (R18 verbatim) ────────────────────────────────
mf = pd.read_parquet(f"{DATA}/monthly_fundamentals.parquet")
mf["date"] = pd.to_datetime(mf["date"])
mf = mf.sort_values(["stock_id","date"])
mf["dH_gpm"] = -mf.groupby("stock_id")["gpm"].transform(
    lambda x: x.rolling(60, min_periods=24).std())
mf["q"] = mf["date"].dt.to_period("Q")
gpm = (mf.dropna(subset=["dH_gpm"]).sort_values(["stock_id","date"])
         .drop_duplicates(["stock_id","q"], keep="last")
         .rename(columns={"stock_id":"ticker"})[["ticker","q","dH_gpm"]])

# ── STEP 6: T quarterly (R18 verbatim) ───────────────────────────────────────
vt = pd.read_parquet(f"{DATA}/variables_monthly.parquet")
vt["date"] = pd.to_datetime(vt["date"])
Tm = vt.groupby("date")["T"].first().to_frame("T")
Tm["q"] = Tm.index.to_period("Q")
Tq = Tm.groupby("q")["T"].last().reset_index()

# ── STEP 7: assemble panel (R18 verbatim) ────────────────────────────────────
panel = (prices.merge(ds, on=["ticker","q"], how="inner")
                .merge(gpm, on=["ticker","q"], how="left")
                .merge(Tq, on="q", how="inner"))
panel = panel.sort_values(["ticker","q"])
panel["q_ord"] = panel["q"].apply(lambda p: p.ordinal)
panel["ret_next"] = panel.groupby("ticker")["ret"].shift(-1)
gap_next = panel.groupby("ticker")["q_ord"].shift(-1) - panel["q_ord"]
panel.loc[gap_next != 1, "ret_next"] = np.nan
panel["delta_s_z"] = cs_winsorize_zscore(panel, "delta_s")
panel["delta_h_z"] = cs_winsorize_zscore(panel, "dH_gpm")
panel["T_delta_s"] = panel["T"] * panel["delta_s_z"]

pf = panel.dropna(subset=["ret_next","delta_s_z","delta_h_z"])
pe = panel.dropna(subset=["ret_next","delta_s_z"])
panel.to_parquet(f"{DATA}/M1_sp500_quarterly_panel.parquet")
say(f"\nFull-channel panel: N={len(pf):,}  tickers={pf['ticker'].nunique():,}  "
    f"quarters={pf['q'].nunique()}  avg/q={pf.groupby('q').size().mean():.0f}")
say(f"Date range: {pf['q'].min()} .. {pf['q'].max()}")

# ── STEP 9: Fama-MacBeth Model B (ΔH + ΔS), NW-4 ─────────────────────────────
say("\n" + "-"*64); say("FAMA-MACBETH — Model B (ΔH + ΔS), quarterly NW-4")
rb, _ = fama_macbeth_nw(pf, "ret_next", ["delta_h_z","delta_s_z"])
def shw(tag, d, k):
    m,t,n = d.get(k,(np.nan,np.nan,0)); say(f"  {tag:12} β={m:+.5f}  t={t:+.2f}  (Tq={n})")
shw("β_ΔH", rb, "delta_h_z"); shw("β_ΔS", rb, "delta_s_z")

# ── STEP 12: ΔS quintile sort ────────────────────────────────────────────────
say("\n" + "-"*64); say("ΔS QUINTILE SORT (Q5−Q1 long-short)")
def quintile_ls(df, sortcol, label):
    d = df.dropna(subset=[sortcol,"ret_next"]).copy()
    d["qd"] = d.groupby("q")[sortcol].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
        if x.nunique() >= 5 else np.nan)
    d = d.dropna(subset=["qd"])
    qr = d.groupby(["q","qd"])["ret_next"].mean().unstack("qd")
    means = {int(c): qr[c].mean() for c in qr.columns}
    ls = (qr.get(4) - qr.get(0)).dropna()
    t = ls.mean()/(ls.std()/np.sqrt(len(ls))) if len(ls) > 2 else np.nan
    say(f"  {label}")
    for q in sorted(means): say(f"    Q{q+1}: {means[q]*100:+.2f}%/q ({means[q]*400:+.1f}%/yr)")
    say(f"    L/S (Q5−Q1): {ls.mean()*400:+.1f}%/yr  t={t:+.2f}  (Tq={len(ls)})")
    return ls.mean()*400, t
ls_yr, ls_t = quintile_ls(pe, "delta_s_z", "Sort on ΔS (12q iVol)")

# ── HEADLINE ─────────────────────────────────────────────────────────────────
say("\n" + "="*64)
_, t_ds, nq = rb.get("delta_s_z",(np.nan,np.nan,0))
_, t_dh, _  = rb.get("delta_h_z",(np.nan,np.nan,0))
say(f"[M1] SP500 quarterly-ΔS panel: N={len(pf):,}, T={pf['q'].nunique()}, "
    f"FM t(ΔS)={t_ds:+.2f}, FM t(ΔH)={t_dh:+.2f}, "
    f"quintile L/S={ls_yr:+.1f}%/yr (t={ls_t:+.2f})")
say("="*64)
say("\nReference points:")
say("  Primary S&P monthly panel (36m iVol): ΔS premium significant (headline).")
say("  R18 full-universe quarterly panel:    FM t(ΔS)≈0 (collapse).")
say("  M1 isolates measurement frequency: same 462 firms, max survivorship,")
say("  ONLY monthly->quarterly iVol. t(ΔS) here discriminates the two stories.")

out_txt = f"{OUT}/M1_sp500_quarterly.txt"
with open(out_txt, "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\nSaved: {out_txt}")
