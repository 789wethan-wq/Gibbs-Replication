"""R25 — Post-Review Additions for V25 (spec: "R20 Experiment Spec")

E1  Survival-conditioning demonstration (Ormos-Zibriczky exhibit) on the R18
    full-universe quarterly SF1 panel: impose k-year continuous-survival
    conditioning, k in {0,5,10,15,20,25,27}, and re-run the R18 ΔS quintile
    sort + Model B FM inside each conditioned panel.
E2  Data gate only: SEP entitlement checked via live API (curl) — NOT entitled
    (empty datatable for AAPL while SF1/DAILY return rows). Reported as
    "SEP unavailable"; skipped per spec. No yfinance substitute.
V1  OOS year count audit: expanding-window (120m train) OOS L/S annual returns
    2006–2023 on the S&P 500 monthly panel (R17 C12 procedure, verbatim).
V2  Pooled two-way-clustered interaction on the R18 full-universe panel.
V3  R19 draw-scheme: documented from code (in the output text, no re-run).
V4  Corr(GPM level, ΔH_GPM), pooled and mean cross-sectional, both panels.
V5  Firms vs tickers: unique tickers vs permatickers in the R18 panel,
    delisted counts under each definition.

Outputs: results/revision/R25_post_review.txt
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

def nw_mean_t(s, lags=4):
    """NW t-stat of the mean of a time series (same kernel as FM aggregator)."""
    s = pd.Series(s).dropna(); n = len(s)
    if n < 5: return np.nan
    m = s.mean(); var = (s**2).mean() - m**2
    for l in range(1, min(lags + 1, n)):
        g = ((s.iloc[l:].values - m) * (s.iloc[:-l].values - m)).mean()
        var += 2 * (1 - l / (lags + 1)) * g
    return m / np.sqrt(max(var, 1e-30) / n)

def quintile_ls(df, sortcol, date_col="q", ycol="ret_next"):
    """R18-identical quintile sort. Returns (ann_ls, t_simple, t_nw4, Tq)."""
    d = df.dropna(subset=[sortcol, ycol]).copy()
    d["qd"] = d.groupby(date_col)[sortcol].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
        if x.nunique() >= 5 else np.nan)
    d = d.dropna(subset=["qd"])
    qr = d.groupby([date_col, "qd"])[ycol].mean().unstack("qd")
    if 0 not in qr.columns or 4 not in qr.columns:
        return np.nan, np.nan, np.nan, 0
    ls = (qr[4] - qr[0]).dropna()
    t = ls.mean()/(ls.std()/np.sqrt(len(ls))) if len(ls) > 2 else np.nan
    return ls.mean()*4, t, nw_mean_t(ls, lags=4), len(ls)

# ═══════════════════════════════════════════════════════════════════════════
say("="*72)
say("R25 — POST-REVIEW EXPERIMENTS FOR V25  (E1, E2 gate, V1, V2, V4, V5)")
say("="*72)

# ═══ E1 — SURVIVAL-CONDITIONING DEMONSTRATION ═══════════════════════════════
say("\n" + "#"*72)
say("# E1 — Survival-conditioning demonstration (Ormos-Zibriczky exhibit)")
say("#"*72)

panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
panel = panel.sort_values(["ticker", "q_ord"]).reset_index(drop=True)
# every row of the saved panel has a valid return AND valid ΔS (R18 inner merge)
say(f"\nR18 panel loaded: N={len(panel):,} rows, tickers={panel['ticker'].nunique():,}, "
    f"quarters={panel['q'].nunique()} ({panel['q'].min()}..{panel['q'].max()})")

# per-ticker consecutive-quarter run IDs
new_run = (panel["q_ord"] - panel.groupby("ticker")["q_ord"].shift(1)) != 1
panel["run_id"] = new_run.groupby(panel["ticker"]).cumsum()
run_len = panel.groupby(["ticker", "run_id"])["q_ord"].transform("size")
panel["run_len_q"] = run_len

# coverage / span / max-run diagnostics
cov = panel.groupby("ticker").agg(
    coverage_quarters=("q_ord", "size"),
    first_q=("q_ord", "min"), last_q=("q_ord", "max"))
cov["span_years"] = (cov["last_q"] - cov["first_q"]) / 4.0
mr = panel.groupby("ticker")["run_len_q"].max() / 4.0
cov["max_run_years"] = mr
say(f"coverage_quarters: median={cov['coverage_quarters'].median():.0f}, "
    f"mean={cov['coverage_quarters'].mean():.4g}")
say(f"span_years:        median={cov['span_years'].median():.4g}, "
    f"mean={cov['span_years'].mean():.4g}")
say(f"max_run_years:     median={cov['max_run_years'].median():.4g}, "
    f"mean={cov['max_run_years'].mean():.4g}, max={cov['max_run_years'].max():.4g}")

# market cap (SF1 ARQ, snapped to quarter exactly as R18 snapped price)
say("\nLoading SF1 ARQ marketcap for the cap-tilt diagnostic...")
mc = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                     columns=["ticker","dimension","calendardate","marketcap"])
mc = mc[(mc["dimension"] == "ARQ") & mc["ticker"].isin(set(panel["ticker"]))].copy()
mc["calendardate"] = pd.to_datetime(mc["calendardate"], errors="coerce")
mc = mc.dropna(subset=["calendardate","marketcap"])
mc["q"] = mc["calendardate"].dt.to_period("Q")
mc = (mc.sort_values(["ticker","calendardate"])
        .drop_duplicates(["ticker","q"], keep="last"))[["ticker","q","marketcap"]]
panel = panel.merge(mc, on=["ticker","q"], how="left")

KS = [0, 5, 10, 15, 20, 25, 27]
rows = []
for k in KS:
    thr_q = int(round(4 * k))
    # condition the PANEL: keep only observations inside runs >= k years
    sub = panel[panel["run_len_q"] >= max(thr_q, 1)].copy() if k > 0 else panel.copy()
    if len(sub) == 0:
        say(f"\n[k={k}] EMPTY subsample — skipping"); continue
    # recompute cross-sectional z-scores WITHIN the conditioned panel (the
    # conditioned panel is the universe, as in the OZ design); k=0 reproduces R18
    sub["delta_s_z"] = cs_winsorize_zscore(sub, "delta_s")
    sub["delta_h_z"] = cs_winsorize_zscore(sub, "dH_gpm")
    pe = sub.dropna(subset=["ret_next", "delta_s_z"])
    pf = sub.dropna(subset=["ret_next", "delta_s_z", "delta_h_z"])

    ls_ann, ls_t, ls_tnw, ls_Tq = quintile_ls(pe, "delta_s_z")
    fm, _ = fama_macbeth_nw(pf, "ret_next", ["delta_h_z", "delta_s_z"])
    b_dh, t_dh, Tq = fm.get("delta_h_z", (np.nan, np.nan, 0))
    b_ds, t_ds, _  = fm.get("delta_s_z", (np.nan, np.nan, 0))

    n_tick = sub["ticker"].nunique()
    avg_q  = pe.groupby("q").size().mean() if len(pe) else np.nan
    med_mc = sub["marketcap"].median()

    say(f"\n[E1 | k={k} | ΔS quintile sort]")
    say(f"LS_ann = {ls_ann:+.4f}   t_simple = {ls_t:+.4g}   t_NW4 = {ls_tnw:+.4g}   "
        f"T_quarters = {ls_Tq}")
    say(f"[E1 | k={k} | FM Model B]")
    say(f"beta_dS = {b_ds:+.6f}   t_dS = {t_ds:+.4g}   beta_dH = {b_dh:+.6f}   "
        f"t_dH = {t_dh:+.4g}")
    say(f"N_tickers = {n_tick}   avg_stocks_qtr = {avg_q:.4g}   "
        f"N_obs = {len(pe)}   N_obs_fullchan = {len(pf)}   T_quarters = {Tq}")
    say(f"median_marketcap_USDm = {med_mc/1e6:.4g}")

    rows.append(dict(k=k, ls_ann=ls_ann, ls_t=ls_t, ls_tnw=ls_tnw,
                     t_ds=t_ds, t_dh=t_dh, n_tick=n_tick, avg_q=avg_q,
                     n_obs=len(pe), med_mc=med_mc))

    if k == 0:
        say("\n  [RECONCILIATION vs R18]  target: FM t(dS)=+0.02, "
            "L/S=-1.0%/yr (t=-0.20)")
        ok = (abs(t_ds - 0.02) < 0.05) and (abs(ls_ann*100 - (-1.0)) < 0.2) \
             and (abs(ls_t - (-0.20)) < 0.05)
        say(f"  got: t(dS)={t_ds:+.4g}, L/S={ls_ann*100:+.4g}%/yr (t={ls_t:+.4g})  "
            f"-> {'MATCH' if ok else '*** MISMATCH — DIAGNOSE BEFORE USING ***'}")
        if not ok:
            raise SystemExit("E1 reconciliation failed; stopping per spec.")
        full_med_mc = med_mc

# summary table
say("\n" + "-"*72)
say("[E1 | SUMMARY TABLE]  (L/S annualized from quarterly means, ×4)")
say(f"{'k(yrs)':>6} {'L/S ann':>9} {'L/S t':>7} {'L/S tNW4':>8} {'FM t(dS)':>9} "
    f"{'FM t(dH)':>9} {'N_tick':>7} {'avg/qtr':>8} {'N_obs':>8} {'medMC($M)':>10}")
for r in rows:
    say(f"{r['k']:>6} {r['ls_ann']*100:>+8.2f}% {r['ls_t']:>+7.2f} {r['ls_tnw']:>+8.2f} "
        f"{r['t_ds']:>+9.2f} {r['t_dh']:>+9.2f} {r['n_tick']:>7,} {r['avg_q']:>8.0f} "
        f"{r['n_obs']:>8,} {r['med_mc']/1e6:>10.4g}")
r27 = [r for r in rows if r["k"] == 27]
if r27:
    say(f"\nCap tilt: median marketcap k=27 = ${r27[0]['med_mc']/1e6:,.4g}M vs "
        f"full panel = ${full_med_mc/1e6:,.4g}M "
        f"(ratio {r27[0]['med_mc']/full_med_mc:.4g}x)")

# ═══ E2 — MONTHLY FULL-UNIVERSE GATE ════════════════════════════════════════
say("\n" + "#"*72)
say("# E2 — Monthly-frequency full-universe check: DATA GATE")
say("#"*72)
say("""
[E2 | data gate | SEP entitlement]
Checked live via Nasdaq Data Link API (datatables/SHARADAR/SEP), 2026-07-11:
  SEP query for AAPL 2023-01-01..2023-01-10  -> empty datatable (no rows)
  SEP query for ENRNQ (delisted)             -> empty datatable (no rows)
  SF1 query for AAPL (control)               -> rows returned (entitled)
  DAILY query for AAPL (control)             -> rows returned (entitled;
                                                metrics only, no prices)
VERDICT: SEP unavailable on this subscription. E2 skipped per spec.
yfinance NOT substituted (no delisted tickers -> would reintroduce the bias).""")

# ═══ V1 — OOS YEAR COUNT AUDIT ══════════════════════════════════════════════
say("\n" + "#"*72)
say("# V1 — OOS year count audit (expanding-window, S&P 500 monthly panel)")
say("#"*72)

def cs_wz_m(df, col, datecol="date", pct=0.01):
    def _w(x):
        x = x.dropna()
        if len(x) < 5: return pd.Series(np.nan, index=x.index)
        xc = x.clip(x.quantile(pct), x.quantile(1-pct)); sd = xc.std()
        if sd < 1e-10: return pd.Series(np.nan, index=x.index)
        return (xc-xc.mean())/sd
    return df.groupby(datecol)[col].transform(_w)

m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
m["date"] = pd.to_datetime(m["date"])
m["dH_gpm_z"] = cs_wz_m(m, "dH_gpm")
m12 = m.dropna(subset=["dH_gpm_z","DS_z","ret_next_month"]).copy()
m12 = m12.sort_values(["date","stock_id"])
dates12 = sorted(m12["date"].unique())
TRAIN_PERIODS = 120
say(f"\nPanel: {len(m12):,} obs, {len(dates12)} months "
    f"({pd.Timestamp(dates12[0]).date()}..{pd.Timestamp(dates12[-1]).date()}); "
    f"train={TRAIN_PERIODS}m expanding (R17 C12 procedure)")

# monthly cross-sectional betas are invariant to the expanding window end,
# so compute them once (identical to refitting inside the loop, as R17 did)
res_all = {}
for d, g in m12.groupby("date"):
    s = g[["ret_next_month","dH_gpm_z","DS_z"]].dropna()
    if len(s) < 10: continue
    X = sm.add_constant(s[["dH_gpm_z","DS_z"]], has_constant="add")
    try:
        r = sm.OLS(s["ret_next_month"], X).fit()
        res_all[d] = r.params[["dH_gpm_z","DS_z"]]
    except Exception: pass
beta_df = pd.DataFrame(res_all).T.sort_index()

oos_recs = []
for i in range(TRAIN_PERIODS, len(dates12)-1):
    train_b = beta_df[beta_df.index <= dates12[i]]
    if train_b.empty: continue
    b_dh12, b_ds12 = train_b["dH_gpm_z"].mean(), train_b["DS_z"].mean()
    test_d = dates12[i+1]
    test_obs = m12[m12["date"] == test_d].dropna(
        subset=["dH_gpm_z","DS_z","ret_next_month"])
    if len(test_obs) < 5: continue
    pred = b_dh12*test_obs["dH_gpm_z"] + b_ds12*test_obs["DS_z"]
    pred_q = pd.qcut(pred, 5, labels=False, duplicates="drop") + 1
    if pred_q.isna().all(): continue
    top = test_obs["ret_next_month"][pred_q == 5].mean() if (pred_q==5).any() else np.nan
    bot = test_obs["ret_next_month"][pred_q == 1].mean() if (pred_q==1).any() else np.nan
    oos_recs.append({"date": test_d, "ls": top - bot})

oos_df = pd.DataFrame(oos_recs)
oos_df["year"] = pd.to_datetime(oos_df["date"]).dt.year
annual_mean = oos_df.groupby("year")["ls"].mean()
annual_cmp  = oos_df.groupby("year")["ls"].apply(lambda x: (1+x).prod()-1)
n_months    = oos_df.groupby("year").size()

say(f"\n[V1 | expanding-window OOS L/S | annual returns]")
say(f"{'year':>6} {'mean %/mo':>10} {'ann(x12) %':>11} {'compound %':>11} {'n_mo':>5}")
for yr in annual_mean.index:
    say(f"{yr:>6} {annual_mean[yr]*100:>+10.4g} {annual_mean[yr]*1200:>+11.4g} "
        f"{annual_cmp[yr]*100:>+11.4g} {n_months[yr]:>5}")
pos = (annual_mean > 0).sum(); tot = len(annual_mean)
neg_years = sorted(annual_mean.index[annual_mean <= 0].tolist())
say(f"\npositive_years = {pos}/{tot}")
say(f"negative_years = {neg_years}")
say(f"mean_monthly_LS_pct = {oos_df['ls'].mean()*100:+.4g}")
say(f"(compound-basis sign check: negative years = "
    f"{sorted(annual_cmp.index[annual_cmp <= 0].tolist())})")

# ═══ V2 — POOLED INTERACTION, FULL UNIVERSE ═════════════════════════════════
say("\n" + "#"*72)
say("# V2 — Pooled two-way-clustered interaction, R18 full-universe panel")
say("#"*72)
w = panel.dropna(subset=["delta_h_z","delta_s_z","T_delta_s","ret_next"]).copy()
Xw = np.column_stack([np.ones(len(w)), w["delta_h_z"], w["delta_s_z"], w["T_delta_s"]])
yw = w["ret_next"].values
bw, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
rw = yw - Xw @ bw
g_date = pd.Categorical(w["q"].astype(str)).codes
g_firm = pd.Categorical(w["ticker"]).codes
V2v = double_cluster_vcov(Xw, rw, g_date, g_firm)
se = np.sqrt(V2v[3,3]); t2 = bw[3]/se; p2 = 1 - chi2.cdf(t2**2, 1)
say(f"\n[V2 | full universe | pooled r ~ dH_z + dS_z + TxdS_z, 2-way clustered]")
say(f"coef_TxdS = {bw[3]:+.6f}   t = {t2:+.4g}   p = {p2:.4g}   N_obs = {len(w)}")
say(f"coef_dH = {bw[1]:+.6f} (t={bw[1]/np.sqrt(V2v[1,1]):+.4g})   "
    f"coef_dS = {bw[2]:+.6f} (t={bw[2]/np.sqrt(V2v[2,2]):+.4g})")
say(f"[S&P 500 comparator: coef=+0.13535, t=+2.49, p=0.0128 (R20 §[3])]")

# ═══ V3 — R19 DRAW-SCHEME DOCUMENTATION (from code, no re-run) ══════════════
say("\n" + "#"*72)
say("# V3 — R19 draw-scheme documentation (read from R19 code; no re-run)")
say("#"*72)
say("""
[V3 | R19 terminal-return assignment, row by row]
Mechanism (R19_delisting_bias_bound.py, stress()): each month, among the
top-quartile-DS_z (most-distressed) survivors, round(delta*N) firms are drawn
uniformly at random (seeded RNG 20260617) and their ret_next_month is REPLACED
by the single fixed value dr for that row; they are removed thereafter.
No mixture and no midpoint is ever used — one fixed dr per row:
  row 1: delta=0.000, dr n/a      — baseline, no delisting applied.
  row 2: delta=0.005, dr=-0.40    — all delisters get -40% (Shumway-Warther blend).
  row 3: delta=0.010, dr=-0.40    — all delisters get -40% (blend).
  row 4: delta=0.020, dr=-0.40    — all delisters get -40% (blend).
  row 5: delta=0.050, dr=-0.40    — all delisters get -40% (blend).
  row 6: delta=0.020, dr=-0.30    — all delisters get -30% (NYSE-only value).
  row 7: delta=0.020, dr=-0.55    — all delisters get -55% (NASDAQ-only value).
Note: the -40% "blend" is the Shumway-Warther blended estimate, not the
-30/-55 midpoint (which would be -42.5%).""")

# ═══ V4 — GPM LEVEL vs ΔH_GPM CORRELATION ═══════════════════════════════════
say("\n" + "#"*72)
say("# V4 — Corr(GPM level, ΔH_GPM)")
say("#"*72)

def corr_pair(df, a, b, datecol):
    d = df.dropna(subset=[a, b])
    pooled = d[a].corr(d[b])
    cs = d.groupby(datecol).apply(
        lambda x: x[a].corr(x[b]) if len(x) >= 5 else np.nan).dropna()
    return pooled, cs.mean(), len(d)

# S&P 500 monthly panel (paper's primary panel)
p_sp, cs_sp, n_sp = corr_pair(m, "gpm", "dH_gpm", "date")
say(f"\n[V4 | S&P 500 monthly panel]")
say(f"pooled_corr = {p_sp:+.4f}   mean_cs_corr = {cs_sp:+.4f}   N_obs = {n_sp}")

# full-universe monthly fundamentals (ΔH_GPM rebuilt exactly as R18 step 5)
mf = pd.read_parquet(f"{DATA}/monthly_fundamentals.parquet")
mf["date"] = pd.to_datetime(mf["date"])
mf = mf.sort_values(["stock_id","date"])
mf["dH_gpm"] = -mf.groupby("stock_id")["gpm"].transform(
    lambda x: x.rolling(60, min_periods=24).std())
p_fu, cs_fu, n_fu = corr_pair(mf, "gpm", "dH_gpm", "date")
say(f"[V4 | full universe monthly fundamentals]")
say(f"pooled_corr = {p_fu:+.4f}   mean_cs_corr = {cs_fu:+.4f}   N_obs = {n_fu}")

# ═══ V5 — FIRMS vs TICKERS ══════════════════════════════════════════════════
say("\n" + "#"*72)
say("# V5 — Firms vs tickers in the R18 panel (permaticker = SF1 permanent id)")
say("#"*72)
tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"][["permaticker","ticker","isdelisted"]].copy()
panel_ticks = pd.DataFrame({"ticker": sorted(panel["ticker"].unique())})
mp = panel_ticks.merge(sf1t, on="ticker", how="left")
n_t = len(panel_ticks)
ambig = mp.groupby("ticker")["permaticker"].nunique()
n_ambig = int((ambig > 1).sum())
n_unmapped = int(mp["permaticker"].isna().groupby(mp["ticker"]).all().sum())
n_p = mp["permaticker"].nunique()
del_t = mp.groupby("ticker")["isdelisted"].apply(lambda x: (x == "Y").all()).sum()
firm = mp.dropna(subset=["permaticker"]).groupby("permaticker")["isdelisted"]
del_p_all = int(firm.apply(lambda x: (x == "Y").all()).sum())
del_p_any = int(firm.apply(lambda x: (x == "Y").any()).sum())
say(f"\n[V5 | R18 panel identity counts]")
say(f"unique_tickers = {n_t}")
say(f"unique_permatickers = {n_p}   "
    f"(tickers mapping to >1 permaticker: {n_ambig}; unmapped: {n_unmapped})")
say(f"delisted_tickers (isdelisted=Y, all SF1 rows for symbol) = {int(del_t)}  "
    f"({del_t/n_t:.1%} of tickers)")
say(f"delisted_permatickers (all listings delisted) = {del_p_all}  "
    f"({del_p_all/n_p:.1%} of permatickers)")
say(f"permatickers with ANY delisted listing = {del_p_any}")

# ── save ────────────────────────────────────────────────────────────────────
out_txt = f"{OUT}/R25_post_review.txt"
with open(out_txt, "w") as f:
    f.write("\n".join(LOG) + "\n")
say("\n" + "="*72); say(f"Saved: {out_txt}")
