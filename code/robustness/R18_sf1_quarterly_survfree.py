"""R18 — SF1 Quarterly Survivorship-Free Panel  (Brief V16, workaround #4)

The SEP price product is not entitled on this Sharadar key, so a monthly
survivorship-free price panel cannot be pulled. SF1 (fundamentals) IS entitled
and survivorship-free: it carries a split-adjusted `price` field at fiscal
quarter-end for 15,519 US domestic common tickers, 11,175 of them DELISTED.

This script rebuilds the entropy channel (ΔS) and returns at QUARTERLY
frequency on that full delisted-inclusive universe and re-runs the paper's
FM / Wald / asymmetric-prediction battery. It is a coarser robustness
*appendix* — quarterly iVol is not the monthly AHXZ measure — whose purpose
is to answer one question: does the T·ΔS structure and the disorder-premium
sign survive once delisted firms enter the sample?

ΔH (GPM stability) and T are reused from the existing pipeline unchanged
(both already survivorship-free / market-level). No new downloads.

NOTE (added post-hoc): this panel changes THREE things at once vs. the primary
S&P500 panel — survivorship (delisted firms in), universe breadth (S&P500 large
caps -> full US-common universe), and frequency (monthly -> quarterly). A
controlled one-variable-at-a-time decomposition of the FM t(ΔS) collapse
(robustness/DIAG_survivorship.py, DIAG_channels.py -> DIAG_Q1Q3.md /
DIAG_channels.md) attributes ~60% to BREADTH (non-generalization beyond the
S&P500), ~30% to survivorship, ~8% to frequency. The quality channel ΔH and the
asymmetric temperature prediction (β_ΔS~T) survive all three toggles; the
unconditional ΔS premium and the FM T·ΔS level coefficient do not. So the honest
framing of the ΔS result is 'survivorship + non-generalization', not pure
survivorship.

Outputs: results/survivorship_free/R18_sf1_quarterly_results.txt
         data/merged_sf1_quarterly_survfree.parquet
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT  = "../results/survivorship_free"
os.makedirs(OUT, exist_ok=True)

LOG = []
def say(*a):
    line = " ".join(str(x) for x in a)
    print(line); LOG.append(line)

# ── helpers (copied from sharadar_pipeline for standalone use) ───────────────
def cs_winsorize_zscore(df, col, date_col="q", pct=0.01):
    def _wz(x):
        x = x.dropna()
        if len(x) < 5: return pd.Series(np.nan, index=x.index)
        lo, hi = x.quantile(pct), x.quantile(1 - pct)
        xc = x.clip(lo, hi); std = xc.std()
        if std < 1e-10: return pd.Series(np.nan, index=x.index)
        return (xc - xc.mean()) / std
    return df.groupby(date_col)[col].transform(_wz)

def cluster_vcov(X, resid, groups):
    n_, k_ = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    B = np.zeros((k_, k_))
    for g in np.unique(groups):
        m = groups == g
        B += X[m].T @ np.outer(resid[m], resid[m]) @ X[m]
    G = len(np.unique(groups))
    sc = (G / (G - 1)) * ((n_ - 1) / (n_ - k_))
    return sc * xtx_inv @ B @ xtx_inv

def double_cluster_vcov(X, resid, g1, g2):
    """Cameron-Gelbach-Miller two-way: V1 + V2 - V12(intersection)."""
    inter = pd.Categorical(pd.Series(g1).astype(str)+"_"+pd.Series(g2).astype(str)).codes
    return (cluster_vcov(X, resid, g1) + cluster_vcov(X, resid, g2)
            - cluster_vcov(X, resid, inter))

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
say("="*64); say("R18 — SF1 QUARTERLY SURVIVORSHIP-FREE PANEL"); say("="*64)

# ── STEP 1: universe (US domestic common, SF1-covered) ──────────────────────
tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"]
uni = sf1t[
    sf1t["category"].str.contains("Domestic Common", na=False) &
    sf1t["exchange"].isin(["NYSE","NASDAQ","NYSEARCA","BATS","NYSEMKT"]) &
    (sf1t["currency"] == "USD")
].copy()
uni_set = set(uni["ticker"])
delisted_set = set(uni.loc[uni["isdelisted"] == "Y", "ticker"])
say(f"\nUniverse (US dom common, SF1): {len(uni_set):,} tickers "
    f"({len(delisted_set):,} delisted)")

# ── STEP 2: quarterly split-adjusted prices + returns ───────────────────────
say("\nLoading SF1 ARQ price series...")
sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                      columns=["ticker","dimension","calendardate","datekey",
                               "price","dps"])
arq = sf1[(sf1["dimension"] == "ARQ") & sf1["ticker"].isin(uni_set)].copy()
arq["calendardate"] = pd.to_datetime(arq["calendardate"], errors="coerce")
arq = arq.dropna(subset=["calendardate","price"])
arq = arq[arq["price"] > 0]
# snap fiscal quarter-end to calendar quarter; keep last filing per ticker-quarter
arq["q"] = arq["calendardate"].dt.to_period("Q")
arq = (arq.sort_values(["ticker","calendardate"])
          .drop_duplicates(["ticker","q"], keep="last"))

# build clean 1-quarter total returns (price return + approx dividend yield)
arq = arq.sort_values(["ticker","q"])
arq["q_ord"] = arq["q"].apply(lambda p: p.ordinal)
arq["price_prev"] = arq.groupby("ticker")["price"].shift(1)
arq["gap"] = arq["q_ord"] - arq.groupby("ticker")["q_ord"].shift(1)
ret_px = arq["price"] / arq["price_prev"] - 1.0
div_q  = (arq["dps"].fillna(0) / 4.0) / arq["price_prev"]   # dps is TTM → /4
arq["ret"] = np.where(arq["gap"] == 1, ret_px + div_q, np.nan)
arq = arq.dropna(subset=["ret"])
# winsorize returns cross-sectionally per quarter
arq["ret"] = arq.groupby("q")["ret"].transform(
    lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
# min 8 quarterly obs per ticker
cnt = arq.groupby("ticker")["ret"].transform("size")
arq = arq[cnt >= 8]
say(f"Quarterly return obs: {len(arq):,} | tickers: {arq['ticker'].nunique():,} "
    f"| quarters: {arq['q'].nunique()}")
prices = arq[["ticker","q","ret"]].copy()

# ── STEP 3: quarterly FF3 factors ───────────────────────────────────────────
fac = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
fac.index = pd.to_datetime(fac.index)
facq = fac.copy(); facq["q"] = facq.index.to_period("Q")
def cmpd(s): return (1 + s).prod() - 1
ff = facq.groupby("q").agg({"Mkt_RF":cmpd,"SMB":cmpd,"HML":cmpd,"RF":cmpd})
ff = ff.reset_index()

# ── STEP 4: ΔS — rolling 12-quarter FF3 residual iVol ───────────────────────
say("\nComputing quarterly idiosyncratic volatility (12q rolling FF3)...")
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
px["delta_s"] = px.groupby("ticker", group_keys=False).apply(
    lambda g: ivol_ticker(g))
ds = px[["ticker","q","delta_s"]].dropna()
say(f"ΔS obs: {len(ds):,} | tickers: {ds['ticker'].nunique():,}")

# ── STEP 5: ΔH — GPM stability from monthly_fundamentals (survivorship-free) ─
say("\nBuilding ΔH_GPM (rolling 60m GPM std) and sampling to quarter-end...")
mf = pd.read_parquet(f"{DATA}/monthly_fundamentals.parquet")
mf["date"] = pd.to_datetime(mf["date"])
mf = mf.sort_values(["stock_id","date"])
mf["dH_gpm"] = -mf.groupby("stock_id")["gpm"].transform(
    lambda x: x.rolling(60, min_periods=24).std())
mf["q"] = mf["date"].dt.to_period("Q")
gpm = (mf.dropna(subset=["dH_gpm"]).sort_values(["stock_id","date"])
         .drop_duplicates(["stock_id","q"], keep="last")
         .rename(columns={"stock_id":"ticker"})[["ticker","q","dH_gpm"]])
say(f"ΔH_GPM quarterly obs: {len(gpm):,} | tickers: {gpm['ticker'].nunique():,}")

# ── STEP 6: T quarterly (reuse existing monthly normalized T) ────────────────
v = pd.read_parquet(f"{DATA}/variables_monthly.parquet")
v["date"] = pd.to_datetime(v["date"])
Tm = v.groupby("date")["T"].first().to_frame("T")
Tm["q"] = Tm.index.to_period("Q")
Tq = Tm.groupby("q")["T"].last().reset_index()   # quarter-end T

# ── STEP 7: assemble panel + next-quarter return ────────────────────────────
say("\nAssembling panel...")
panel = (prices.merge(ds, on=["ticker","q"], how="inner")
                .merge(gpm, on=["ticker","q"], how="left")
                .merge(Tq, on="q", how="inner"))          # restrict to T window
panel = panel.sort_values(["ticker","q"])
panel["q_ord"] = panel["q"].apply(lambda p: p.ordinal)
panel["ret_next"] = panel.groupby("ticker")["ret"].shift(-1)
gap_next = panel.groupby("ticker")["q_ord"].shift(-1) - panel["q_ord"]
panel.loc[gap_next != 1, "ret_next"] = np.nan

# cross-sectional z-scores within quarter
panel["delta_s_z"] = cs_winsorize_zscore(panel, "delta_s")
panel["delta_h_z"] = cs_winsorize_zscore(panel, "dH_gpm")
panel["T_delta_s"] = panel["T"] * panel["delta_s_z"]
panel["delta_g_raw"] = panel["delta_h_z"] - panel["T_delta_s"]
panel["delta_g"] = cs_winsorize_zscore(panel, "delta_g_raw")
panel.to_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")

# panels for entropy-only vs full-channel
pe = panel.dropna(subset=["ret_next","delta_s_z"])
pf = panel.dropna(subset=["ret_next","delta_s_z","delta_h_z"])
say(f"\nEntropy panel:      N={len(pe):,}  tickers={pe['ticker'].nunique():,}  "
    f"avg/q={pe.groupby('q').size().mean():.0f}")
say(f"Full-channel panel: N={len(pf):,}  tickers={pf['ticker'].nunique():,}  "
    f"avg/q={pf.groupby('q').size().mean():.0f}")
say(f"Date range: {panel['q'].min()} .. {panel['q'].max()}")

# delisting diagnostics
last_q = panel.groupby("ticker")["q"].max()
end_pre = (last_q < pd.Period("2023Q3")).sum()
say(f"Tickers in full panel whose series ends pre-2023Q3: {end_pre:,} "
    f"({end_pre/panel['ticker'].nunique():.0%}) — delisted/dropped mid-sample")
dh_cov = pf["delta_h_z"].notna().mean()
say(f"ΔH_GPM coverage within entropy panel: "
    f"{panel['delta_h_z'].notna().mean():.1%}")

# ── STEP 8: separation diagnostic ───────────────────────────────────────────
corr = pf.groupby("q").apply(lambda x: x["delta_h_z"].corr(x["delta_s_z"])).mean()
say("\n" + "-"*64)
say("SEPARATION DIAGNOSTIC")
say(f"  Corr(ΔH_GPM, ΔS), full universe: {corr:+.3f}   "
    f"[S&P500 sample: -0.259]")

# ── STEP 9: Fama-MacBeth (same specs as primary paper) ──────────────────────
say("\n" + "-"*64); say("FAMA-MACBETH (quarterly, NW-4)")
ra, _ = fama_macbeth_nw(pf, "ret_next", ["delta_g"])
rb, betas_b = fama_macbeth_nw(pf, "ret_next", ["delta_h_z","delta_s_z"])
rc, _ = fama_macbeth_nw(pf, "ret_next", ["delta_h_z","T_delta_s"])
re_, _ = fama_macbeth_nw(pf, "ret_next", ["delta_h_z","delta_s_z","T_delta_s"])
def shw(tag, d, k):
    m,t,n = d.get(k,(np.nan,np.nan,0)); say(f"  {tag:26} β={m:+.5f}  t={t:+.2f}  (Tq={n})")
say("Model A — composite ΔG");        shw("β_ΔG", ra, "delta_g")
say("Model B — decomposed ΔH + ΔS");  shw("β_ΔH", rb, "delta_h_z"); shw("β_ΔS", rb, "delta_s_z")
say("Model C — ΔH + T·ΔS");           shw("β_ΔH", rc, "delta_h_z"); shw("β_(T·ΔS)", rc, "T_delta_s")
say("Encompassing — ΔH + ΔS + T·ΔS"); shw("β_ΔH", re_, "delta_h_z"); shw("β_ΔS", re_, "delta_s_z"); shw("β_(T·ΔS)", re_, "T_delta_s")

# ── STEP 10: cluster-robust Wald on T·ΔS ────────────────────────────────────
say("\n" + "-"*64); say("CLUSTER-ROBUST WALD TEST (T·ΔS = 0)")
w = pf.dropna(subset=["delta_h_z","delta_s_z","T_delta_s","ret_next"]).copy()
Xw = np.column_stack([np.ones(len(w)), w["delta_h_z"], w["delta_s_z"], w["T_delta_s"]])
yw = w["ret_next"].values
bw, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
rw = yw - Xw @ bw
g_date = pd.Categorical(w["q"].astype(str)).codes
g_firm = pd.Categorical(w["ticker"]).codes
for label, V in [("date-cluster", cluster_vcov(Xw, rw, g_date)),
                 ("double-cluster", double_cluster_vcov(Xw, rw, g_date, g_firm))]:
    se = np.sqrt(V[3,3]); t = bw[3]/se; p = 1 - chi2.cdf(t**2, 1)
    say(f"  {label:16} β(T·ΔS)={bw[3]:+.5f}  t={t:+.2f}  Wald p={p:.4f}   "
        f"[S&P500: p=0.017]")

# ── STEP 11: asymmetric temperature prediction ──────────────────────────────
say("\n" + "-"*64); say("ASYMMETRIC TEMPERATURE PREDICTION (§4.4)")
bt = betas_b.join(Tq.set_index("q")["T"], how="inner").dropna()
for chan, col in [("β_ΔH ~ T", "delta_h_z"), ("β_ΔS ~ T", "delta_s_z")]:
    X = sm.add_constant(bt["T"]); r = sm.OLS(bt[col], X).fit(
        cov_type="HAC", cov_kwds={"maxlags":4})
    say(f"  {chan:12} slope={r.params['T']:+.4f}  t={r.tvalues['T']:+.2f}")
say("  (paper: ΔH T-independent t≈-1.08, ΔS T-dependent t≈+2.45)")

# ── STEP 12: quintile sorts (ΔG and pure ΔS/disorder) ───────────────────────
say("\n" + "-"*64); say("QUINTILE SORTS  (Q5−Q1 long-short)")
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
    for q in sorted(means): say(f"    Q{q+1}: {means[q]*100:+.2f}%/q ({means[q]*400:+.1f}%/yr ann)")
    say(f"    L/S (Q5−Q1): {ls.mean()*100:+.2f}%/q ({ls.mean()*400:+.1f}%/yr)  t={t:+.2f}  (Tq={len(ls)})")
    return ls.mean()*4, t
quintile_ls(pf, "delta_g", "Sort on ΔG (composite)")
quintile_ls(pe, "delta_s_z", "Sort on ΔS (disorder / iVol) — AHXZ premium")

# ── save log ────────────────────────────────────────────────────────────────
out_txt = f"{OUT}/R18_sf1_quarterly_results.txt"
with open(out_txt, "w") as f:
    f.write("\n".join(LOG) + "\n")
say("\n" + "="*64); say(f"Saved: {out_txt}")
say(f"Saved: {DATA}/merged_sf1_quarterly_survfree.parquet")
