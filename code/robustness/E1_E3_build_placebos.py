"""D1 — Placebo-characteristic second-step test.  [BLOCKING, run first]

THE CONCERN:
  beta_DS,t (the §4.4 second-step FM loading) is a cross-sectional slope on a
  standardized regressor, i.e. proportional to cov(r_{i,t+1}, DS^z_{i,t}).
  Cross-sectional return dispersion rises with market volatility T. So ANY
  characteristic with a non-zero mean loading can produce a second-step slope
  whose magnitude co-varies with T mechanically, with no conditional pricing
  whatsoever. The paper placebos the T side (reversed/trend/random) but never
  the characteristic side. This script closes that hole.

D1.1 — for size (log mktcap), B/M (1/pb), momentum (12-1 analog), and market
       beta: extract beta_char,t from the SAME first-step regression as ΔH/ΔS
       (ret_next ~ const + dH_z + char_z), same panel/controls/months, then
       regress beta_char,t on T_t with identical HAC/BIC-lag convention as §4.4.
D1.2 — dispersion normalization: (beta_DS,t / sigma_cs,t+1) ~ T_t.
D1.3 — stacked date-clustered slope-difference test (A3 design), raw AND
       dispersion-normalized.
D1.4 — alignment audit: exact timing convention, pasted from source.

Outputs: results/revision/D1_placebo_characteristic_test.txt
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

def cs_wz(df, col, datecol, pct=0.01):
    def _w(x):
        x = x.dropna()
        if len(x) < 5: return pd.Series(np.nan, index=x.index)
        xc = x.clip(x.quantile(pct), x.quantile(1-pct)); sd = xc.std()
        if sd < 1e-10: return pd.Series(np.nan, index=x.index)
        return (xc-xc.mean())/sd
    return df.groupby(datecol)[col].transform(_w)

def ar1(s):
    s = pd.Series(s).dropna().values
    if len(s) < 3: return np.nan
    return np.corrcoef(s[:-1], s[1:])[0,1]

def bic_nw_lag(resid, max_lag=12):
    r = pd.Series(resid).dropna(); n = len(r); best_p, best_bic = 0, np.inf
    for p in range(0, min(max_lag, n//4)+1):
        try:
            if p == 0:
                rss = float(((r-r.mean())**2).sum()); k = 1
            else:
                mfit = sm.tsa.AutoReg(r, lags=p, old_names=False).fit()
                rss = float((mfit.resid**2).sum()); k = p+1
            bic = n*np.log(rss/n) + k*np.log(n)
            if bic < best_bic: best_bic, best_p = bic, p
        except Exception:
            pass
    return best_p

def step1_betas_disp(panel, ycol, dh, ds, datecol, Tcol, min_cs):
    """Monthly/quarterly cross-sectional OLS -> beta_dh, beta_ds series + sigma_cs of y."""
    rec = []
    for d, g in panel.groupby(datecol):
        sub = g[[ycol, dh, ds, Tcol]].dropna()
        if len(sub) < min_cs: continue
        X = sm.add_constant(sub[[dh, ds]], has_constant="add")
        r = sm.OLS(sub[ycol], X).fit()
        rec.append((d, r.params[dh], r.params[ds], sub[Tcol].iloc[0], sub[ycol].std()))
    return pd.DataFrame(rec, columns=["date","b_dH","b_ds","T","sigma_cs"]).set_index("date")

def reg_report(y, x, tag):
    y = np.asarray(y).ravel(); x = np.asarray(x).reshape(len(y), -1)
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    p = bic_nw_lag(ols.resid)
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(p,1)})
    n = int(ols.nobs); rho = ar1(y)
    say(f"  {tag}")
    say(f"    slope={ols.params[1]:+.5f}  R²={ols.rsquared:.3f}  n={n}  AR1(y)={rho:+.2f}")
    say(f"    OLS  t={ols.tvalues[1]:+.2f} (p={ols.pvalues[1]:.3f})")
    say(f"    HAC  t={hac.tvalues[1]:+.2f} (p={hac.pvalues[1]:.3f})  [NW lag={max(p,1)} by BIC]")
    dy = pd.Series(np.asarray(y)).diff().dropna()
    dx = pd.Series(np.asarray(x).ravel()).diff().dropna()
    Xd = sm.add_constant(dx.values)
    fd = sm.OLS(dy.values, Xd).fit(cov_type="HAC", cov_kwds={"maxlags":1})
    say(f"    ΔΔ   t={fd.tvalues[1]:+.2f} (first-difference, stationarity check)")
    return dict(ols_t=ols.tvalues[1], hac_t=hac.tvalues[1], fd_t=fd.tvalues[1],
                slope=ols.params[1], r2=ols.rsquared, n=n)

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

def stacked_diff_test(bDH, bDS, T, tag):
    df = pd.DataFrame({"bDH": bDH, "bDS": bDS, "T": T}).dropna()
    stk = pd.concat([
        pd.DataFrame({"beta": df["bDH"], "T": df["T"], "D_S": 0.0, "per": df.index}),
        pd.DataFrame({"beta": df["bDS"], "T": df["T"], "D_S": 1.0, "per": df.index}),
    ], ignore_index=True)
    stk["TxD"] = stk["T"] * stk["D_S"]
    X = sm.add_constant(stk[["T", "D_S", "TxD"]])
    grp = pd.Categorical(stk["per"].astype(str)).codes
    res = sm.OLS(stk["beta"], X).fit(cov_type="cluster", cov_kwds={"groups": grp})
    d_, td, pdv = res.params["TxD"], res.tvalues["TxD"], res.pvalues["TxD"]
    say(f"  {tag:32} d(TxD)={d_:+.5f}  t={td:+.2f}  p={pdv:.4f}  [clusters={len(set(grp))}, obs={len(stk)}]")
    return d_, td, pdv

say("="*72); say("D1 — PLACEBO-CHARACTERISTIC SECOND-STEP TEST"); say("="*72)

# ═════════════════════════════════════════════════════════════════════════
# BUILD PLACEBO CHARACTERISTICS — SF1-based (size, B/M), price-based (mom, beta)
# ═════════════════════════════════════════════════════════════════════════
say("\n" + "#"*72); say("# BUILDING PLACEBO CHARACTERISTICS"); say("#"*72)

sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                       columns=["ticker","dimension","datekey","calendardate","marketcap","pb"])
arq_f = sf1[sf1["dimension"] == "ARQ"].copy()
arq_f["datekey"] = pd.to_datetime(arq_f["datekey"], errors="coerce").astype("datetime64[ns]")
arq_f = arq_f.dropna(subset=["datekey"]).sort_values(["ticker","datekey"])
arq_f["size_raw"] = np.where(arq_f["marketcap"] > 0, np.log(arq_f["marketcap"]), np.nan)
arq_f["bm_raw"]   = np.where(arq_f["pb"] > 0, 1.0/arq_f["pb"], np.nan)
say(f"SF1 ARQ records for size/BM: {len(arq_f):,}  "
    f"(size non-null {arq_f['size_raw'].notna().mean():.1%}, "
    f"BM non-null {arq_f['bm_raw'].notna().mean():.1%})")

# ---- S&P 500 monthly panel: point-in-time backward-asof size/BM (same
#      convention as project/sharadar_pipeline.py:build_monthly_fundamentals) ----
say("\n-- S&P 500 monthly: size/BM via backward merge_asof on datekey --")
sp_prices = pd.read_parquet(f"{DATA}/stock_prices_monthly.parquet")
sp_prices.index = pd.to_datetime(sp_prices.index).astype("datetime64[ns]")
monthly_dates = sp_prices.index.sort_values()
sp_tickers = [c for c in sp_prices.columns if c != "Date"]

sp_char_panels = []
for tkr in sp_tickers:
    grp = arq_f[arq_f["ticker"] == tkr][["datekey","size_raw","bm_raw"]]
    if len(grp) < 2: continue
    grp2 = grp.rename(columns={"datekey":"date"}).sort_values("date")
    month_df = pd.DataFrame({"date": monthly_dates})
    merged = pd.merge_asof(month_df, grp2, on="date", direction="backward")
    merged["stock_id"] = tkr
    sp_char_panels.append(merged)
sp_char = pd.concat(sp_char_panels, ignore_index=True)
say(f"  size/BM point-in-time panel: {len(sp_char):,} rows, {sp_char['stock_id'].nunique()} tickers")

# ---- S&P momentum (12-1 analog, skip most recent month) and beta (36m rolling CAPM) ----
say("-- S&P 500 monthly: momentum (t-12..t-2) and beta (36m rolling CAPM) --")
factors = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
factors.index = pd.to_datetime(factors.index)
common_idx = sp_prices.index.intersection(factors.index).sort_values()
sp_prices_c = sp_prices.reindex(common_idx)[sp_tickers]
fac_c = factors.reindex(common_idx)
ret_all = sp_prices_c.pct_change()
logret_all = np.log1p(ret_all.clip(lower=-0.99))
mom_all = np.exp(logret_all.shift(2).rolling(11).sum()) - 1.0   # skip t-1, use t-2..t-12

exret_all = ret_all.sub(fac_c["RF"], axis=0)
mkt_rf = fac_c["Mkt_RF"]
def rolling_capm_beta(ex, mkt, window=36, min_p=24):
    out = pd.Series(np.nan, index=ex.index)
    y = ex.values; x = mkt.values; n = len(y)
    for i in range(min_p, n+1):
        lo = max(0, i-window)
        ys = y[lo:i]; xs = x[lo:i]
        mask = ~(np.isnan(ys) | np.isnan(xs))
        if mask.sum() < min_p: continue
        Xc = np.column_stack([np.ones(mask.sum()), xs[mask]])
        beta, *_ = np.linalg.lstsq(Xc, ys[mask], rcond=None)
        out.iloc[i-1] = beta[1]
    return out
beta_dict = {}
for i, tkr in enumerate(sp_tickers):
    if i % 100 == 0: say(f"    beta {i}/{len(sp_tickers)}...")
    beta_dict[tkr] = rolling_capm_beta(exret_all[tkr], mkt_rf)
beta_all = pd.DataFrame(beta_dict)

mom_stack = mom_all.stack(future_stack=True).rename("mom_raw")
beta_stack = beta_all.stack(future_stack=True).rename("beta_raw")
sp_mb = pd.concat([mom_stack, beta_stack], axis=1).reset_index()
sp_mb.columns = ["date","stock_id","mom_raw","beta_raw"]

sp_char = sp_char.merge(sp_mb, on=["date","stock_id"], how="left")
for col in ["size_raw","bm_raw","mom_raw","beta_raw"]:
    sp_char[col.replace("_raw","_z")] = cs_wz(sp_char, col, "date")
say(f"  coverage: size {sp_char['size_z'].notna().mean():.1%}  bm {sp_char['bm_z'].notna().mean():.1%}  "
    f"mom {sp_char['mom_z'].notna().mean():.1%}  beta {sp_char['beta_z'].notna().mean():.1%}")

sp = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
sp["date"] = pd.to_datetime(sp["date"])
sp["dH_gpm_z"] = cs_wz(sp, "dH_gpm", "date")
sp = sp.merge(sp_char[["date","stock_id","size_z","bm_z","mom_z","beta_z"]],
              on=["date","stock_id"], how="left")

# ---- R18 full-universe quarterly panel: same construction, quarterly frequency ----
say("\n-- R18 full-universe quarterly: size/BM via backward-asof -> quarter-end sample --")
tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"]
uni = sf1t[
    sf1t["category"].str.contains("Domestic Common", na=False) &
    sf1t["exchange"].isin(["NYSE","NASDAQ","NYSEARCA","BATS","NYSEMKT"]) &
    (sf1t["currency"] == "USD")
].copy()
uni_set = set(uni["ticker"])

q_panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
q_tickers = set(q_panel["ticker"].unique()) & uni_set
say(f"  tickers needing size/BM: {len(q_tickers):,}")

# point-in-time monthly size/BM (same backward-asof logic), then sample to quarter-end
arq_u = arq_f[arq_f["ticker"].isin(q_tickers)].copy()
monthly_dates_full = pd.date_range("1990-01-31", "2024-01-31", freq="ME").astype("datetime64[ns]")
q_char_panels = []
for i, tkr in enumerate(sorted(q_tickers)):
    if i % 1000 == 0: say(f"    size/BM asof {i}/{len(q_tickers)}...")
    grp = arq_u[arq_u["ticker"] == tkr][["datekey","size_raw","bm_raw"]].sort_values("datekey")
    if len(grp) < 2: continue
    grp2 = grp.rename(columns={"datekey":"date"})
    month_df = pd.DataFrame({"date": monthly_dates_full})
    merged = pd.merge_asof(month_df, grp2, on="date", direction="backward")
    merged["ticker"] = tkr
    q_char_panels.append(merged)
q_char_m = pd.concat(q_char_panels, ignore_index=True)
q_char_m["q"] = q_char_m["date"].dt.to_period("Q")
q_char = (q_char_m.dropna(subset=["size_raw","bm_raw"], how="all")
                   .sort_values(["ticker","date"])
                   .drop_duplicates(["ticker","q"], keep="last")[["ticker","q","size_raw","bm_raw"]])
say(f"  size/BM quarterly obs: {len(q_char):,}")

# momentum + beta from full SF1 ARQ price series (rebuilt, not the already-merged panel,
# so rolling windows aren't broken by rows the entropy/GPM inner-joins dropped)
say("-- R18: rebuilding full quarterly return series for momentum/beta rolling windows --")
sf1p = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                        columns=["ticker","dimension","calendardate","datekey","price","dps"])
arqp = sf1p[(sf1p["dimension"] == "ARQ") & sf1p["ticker"].isin(q_tickers)].copy()
arqp["calendardate"] = pd.to_datetime(arqp["calendardate"], errors="coerce")
arqp = arqp.dropna(subset=["calendardate","price"])
arqp = arqp[arqp["price"] > 0]
arqp["q"] = arqp["calendardate"].dt.to_period("Q")
arqp = arqp.sort_values(["ticker","calendardate"]).drop_duplicates(["ticker","q"], keep="last")
arqp = arqp.sort_values(["ticker","q"])
arqp["q_ord"] = arqp["q"].apply(lambda p: p.ordinal)
arqp["price_prev"] = arqp.groupby("ticker")["price"].shift(1)
arqp["gap"] = arqp["q_ord"] - arqp.groupby("ticker")["q_ord"].shift(1)
ret_px = arqp["price"]/arqp["price_prev"] - 1.0
div_q = (arqp["dps"].fillna(0)/4.0)/arqp["price_prev"]
arqp["ret_full"] = np.where(arqp["gap"] == 1, ret_px + div_q, np.nan)

fac = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
fac.index = pd.to_datetime(fac.index)
facq = fac.copy(); facq["q"] = facq.index.to_period("Q")
def cmpd(s): return (1+s).prod()-1
ffq = facq.groupby("q").agg({"Mkt_RF":cmpd, "RF":cmpd}).reset_index()
arqp = arqp.merge(ffq, on="q", how="left")
arqp["exret_full"] = arqp["ret_full"] - arqp["RF"]

def mom_beta_ticker(g):
    g = g.sort_values("q_ord").reset_index(drop=True)
    logr = np.log1p(g["ret_full"].clip(lower=-0.99))
    mom = np.exp(logr.shift(2).rolling(3).sum()) - 1.0   # skip q-1, use q-2,q-3,q-4
    beta = pd.Series(np.nan, index=g.index)
    y = g["exret_full"].values; x = g["Mkt_RF"].values; n = len(g)
    for i in range(8, n+1):
        lo = max(0, i-12)
        ys = y[lo:i]; xs = x[lo:i]
        mask = ~(np.isnan(ys) | np.isnan(xs))
        if mask.sum() < 8: continue
        Xc = np.column_stack([np.ones(mask.sum()), xs[mask]])
        b, *_ = np.linalg.lstsq(Xc, ys[mask], rcond=None)
        beta.iloc[i-1] = b[1]
    return pd.DataFrame({"ticker": g["ticker"], "q": g["q"], "mom_raw": mom.values, "beta_raw": beta.values})

mb_list = []
tickers_arqp = arqp["ticker"].unique()
for i, tkr in enumerate(tickers_arqp):
    if i % 2000 == 0: say(f"    mom/beta {i}/{len(tickers_arqp)}...")
    mb_list.append(mom_beta_ticker(arqp[arqp["ticker"] == tkr]))
q_mb = pd.concat(mb_list, ignore_index=True)

q_char_full = q_char.merge(q_mb, on=["ticker","q"], how="outer")
for col in ["size_raw","bm_raw","mom_raw","beta_raw"]:
    q_char_full[col.replace("_raw","_z")] = cs_wz(q_char_full, col, "q")
say(f"  R18 placebo coverage: size {q_char_full['size_z'].notna().mean():.1%}  "
    f"bm {q_char_full['bm_z'].notna().mean():.1%}  mom {q_char_full['mom_z'].notna().mean():.1%}  "
    f"beta {q_char_full['beta_z'].notna().mean():.1%}")

q_panel = q_panel.merge(q_char_full[["ticker","q","size_z","bm_z","mom_z","beta_z"]],
                         on=["ticker","q"], how="left")

say(f"\nBoth panels assembled. S&P: {len(sp):,} rows, {sp['stock_id'].nunique()} tickers, "
    f"{sp['date'].nunique()} months.  R18: {len(q_panel):,} rows, {q_panel['ticker'].nunique()} tickers, "
    f"{q_panel['q'].nunique()} quarters.")

# ── save for reuse by E1 (dispersion-normalized placebos) and E3 (survival
# ladder on placebos) -- both need the identical placebo-characteristic
# panels; this is the expensive (rolling-beta, ~12k-ticker) build step, run
# ONCE and cached, exactly the same pattern used for R26's split-half build.
sp.to_parquet(f"{DATA}/E1E3_sp_with_placebos.parquet")
q_panel.to_parquet(f"{DATA}/E1E3_qpanel_with_placebos.parquet")
say(f"\nSaved {DATA}/E1E3_sp_with_placebos.parquet and {DATA}/E1E3_qpanel_with_placebos.parquet")

with open(f"{OUT}/E1_E3_build_placebos.txt", "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"wrote {OUT}/E1_E3_build_placebos.txt")
