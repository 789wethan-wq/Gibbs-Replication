#!/usr/bin/env python3
"""
A_runs_plos.py — PLOS closing-edit analysis runs A1, A2, A3.

Faithful to the established project methodology:
  * Full-universe quarterly Model B  = fama_macbeth_nw(pf,"ret_next",["delta_h_z","delta_s_z"]), NW-4, min_cs=20
    on data/merged_sf1_quarterly_survfree.parquet  (reproduces headline t(dH)=+3.46).
  * S&P500 monthly Model B           = fama_macbeth(...,["dH_gpm_z","DS_z"]), NW-6
    on data/merged_with_accounting.parquet (462 names).
  * QMJ control + spanning alpha follow R12_reviewer_responses.py Task 1.
  * A3 HAC uses BIC-selected NW lag, replicating R20_section44_hac.py bic_nw_lag.

Reports primary specifications AS RUN. No bias-correction, no cherry-picking.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robustness_utils import winsorize_cs, zscore_cs, fama_macbeth, quintile_sort_ls

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def fama_macbeth_nw(panel, y_col, x_cols, date_col="q", lags=4, min_cs=20, w_col=None):
    """Quarterly FM (R18 convention). If w_col given -> WLS by that weight each cross-section."""
    coefs = []
    for d, grp in panel.groupby(date_col):
        cols = [y_col] + x_cols + ([w_col] if w_col else [])
        sub = grp[cols].dropna()
        if len(sub) < max(min_cs, len(x_cols) + 2):
            continue
        X = sm.add_constant(sub[x_cols], has_constant="add")
        if w_col:
            wgt = sub[w_col].clip(lower=0).values
            if wgt.sum() <= 0:
                continue
            res = sm.WLS(sub[y_col], X, weights=wgt).fit()
        else:
            res = sm.OLS(sub[y_col], X).fit()
        coefs.append(res.params[x_cols].rename(d))
    if not coefs:
        return {}, pd.DataFrame()
    cdf = pd.DataFrame(coefs); out = {}
    for col in x_cols:
        s = cdf[col].dropna(); n = len(s); mean_ = s.mean()
        var = (s**2).mean() - mean_**2
        for l in range(1, min(lags + 1, n)):
            g = ((s.iloc[l:].values - mean_) * (s.iloc[:-l].values - mean_)).mean()
            var += 2 * (1 - l / (lags + 1)) * g
        se = np.sqrt(max(var, 1e-30) / n)
        out[col] = (mean_, mean_ / se, n)
    return out, cdf

def vif_two(a, b):
    """VIF of column a controlling for b (pooled), symmetric two-variable case = 1/(1-r^2)."""
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    r = df["a"].corr(df["b"])
    return 1.0 / max(1e-12, (1.0 - r**2)), r

def bic_nw_lag(resid, max_lag=12):
    r = pd.Series(resid).dropna(); n = len(r); best_p, best_bic = 0, np.inf
    for p in range(0, min(max_lag, n // 4) + 1):
        try:
            if p == 0:
                rss = float(((r - r.mean())**2).sum()); k = 1
            else:
                m = sm.tsa.AutoReg(r, lags=p, old_names=False).fit()
                rss = float((m.resid**2).sum()); k = p + 1
            bic = n * np.log(rss / n) + k * np.log(n)
            if bic < best_bic:
                best_bic, best_p = bic, p
        except Exception:
            pass
    return best_p

def spanning_alpha(ls_ret, factor_df, ff_cols):
    """Regress L/S excess return on factor set; return alpha (per period), t, p, NW lag, n."""
    d = pd.DataFrame({"ls": ls_ret}).join(factor_df[ff_cols], how="inner").dropna()
    y = d["ls"].values
    X = sm.add_constant(d[ff_cols].values)
    lag = max(bic_nw_lag(sm.OLS(y, X).fit().resid), 1)
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return res.params[0], res.tvalues[0], res.pvalues[0], lag, len(d)

# ═════════════════════════════════════════════════════════════════════════════
# LOAD PANELS
# ═════════════════════════════════════════════════════════════════════════════
say("="*72); say("A-RUNS FOR PLOS CLOSING EDIT  (A1 QMJ | A2 microcap | A3 asymmetry)"); say("="*72)

# Full-universe quarterly (ground truth: t(dH)=+3.46)
fu = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
pf = fu.dropna(subset=["ret_next", "delta_s_z", "delta_h_z"]).copy()
say(f"\nFull-universe quarterly: N={len(pf):,}  tickers={pf['ticker'].nunique():,}  "
    f"avg/q={pf.groupby('q').size().mean():.0f}")

# S&P500 monthly (462 names)
sp = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
sp["date"] = pd.to_datetime(sp["date"])
sp["dH_gpm_z"] = sp.groupby("date")["dH_gpm"].transform(lambda x: zscore_cs(winsorize_cs(x)))
sp_b = sp.dropna(subset=["ret_next_month", "dH_gpm_z", "DS_z"]).copy()
say(f"S&P500 monthly:         N={len(sp_b):,}  names={sp_b['stock_id'].nunique():,}  "
    f"months={sp_b['date'].nunique()}")

# factors (monthly) + quarterly compounded
fac = pd.read_parquet(f"{DATA}/factors_monthly.parquet"); fac.index = pd.to_datetime(fac.index)
qmj = pd.read_parquet(f"{DATA}/../robustness/aqr_data/qmj_monthly_us.parquet"); qmj.index = pd.to_datetime(qmj.index)
facq = fac.copy(); facq["q"] = facq.index.to_period("Q")
cmpd = lambda s: (1 + s).prod() - 1
ff_cols_all = [c for c in ["Mkt_RF","SMB","HML","RMW","CMA","Mom"] if c in fac.columns]
faq = facq.groupby("q").agg({c: cmpd for c in ff_cols_all + (["RF"] if "RF" in fac.columns else [])})
qmjq = qmj.copy(); qmjq["q"] = qmjq.index.to_period("Q")
qmjq = qmjq.groupby("q")["QMJ"].apply(lambda s: (1 + s).prod() - 1)

# baselines
bfu, betas_fu = fama_macbeth_nw(pf, "ret_next", ["delta_h_z", "delta_s_z"])
bsp, _ = fama_macbeth(sp_b, "ret_next_month", ["dH_gpm_z", "DS_z"])
# dated monthly Model B coefs (fama_macbeth drops dates; recompute with date index for A3)
_bsp, betas_sp = fama_macbeth_nw(sp_b, "ret_next_month", ["dH_gpm_z", "DS_z"],
                                 date_col="date", lags=6, min_cs=20)
say(f"          (dated monthly Model B check: t(dH)={_bsp['dH_gpm_z'][1]:+.2f})")
say(f"\nBASELINE  FU: t(dH)={bfu['delta_h_z'][1]:+.2f} t(dS)={bfu['delta_s_z'][1]:+.2f}  |  "
    f"SP500: t(dH)={bsp['dH_gpm_z'][1]:+.2f} t(dS)={bsp['DS_z'][1]:+.2f}")

# ═════════════════════════════════════════════════════════════════════════════
# A1 — QMJ / EARNINGS-QUALITY ORTHOGONALITY
# ═════════════════════════════════════════════════════════════════════════════
say("\n" + "="*72); say("A1 — QMJ ORTHOGONALITY OF THE STABILITY CHANNEL"); say("="*72)

# --- (a) FM with QMJ added as cross-sectional control (R12 convention) ---
# FULL UNIVERSE (quarterly): merge quarterly QMJ, add QMJ_z (constant per q -> ~no-op, as expected)
pfq = pf.copy()
pfq["QMJ"] = pfq["q"].map(qmjq.to_dict())
pfq["QMJ_z"] = pfq.groupby("q")["QMJ"].transform(zscore_cs)
o_fu_qmj, _ = fama_macbeth_nw(pfq, "ret_next", ["delta_h_z", "delta_s_z", "QMJ_z"])
# SP500 (monthly)
spq = sp_b.copy()
spq["QMJ"] = spq["date"].map(qmj["QMJ"].to_dict())
spq["QMJ_z"] = spq.groupby("date")["QMJ"].transform(zscore_cs)
o_sp_qmj, _ = fama_macbeth(spq.dropna(subset=["dH_gpm_z","DS_z","QMJ_z","ret_next_month"]),
                           "ret_next_month", ["dH_gpm_z", "DS_z", "QMJ_z"])

say("\n(a) FM t(dH) with QMJ added as cross-sectional control:")
say(f"    Full-universe : t(dH|+QMJ) = {o_fu_qmj.get('delta_h_z',(0,np.nan))[1]:+.2f}   "
    f"(baseline {bfu['delta_h_z'][1]:+.2f})")
say(f"    S&P500 monthly: t(dH|+QMJ) = {o_sp_qmj.get('dH_gpm_z',(0,np.nan))[1]:+.2f}   "
    f"(baseline {bsp['dH_gpm_z'][1]:+.2f})")
say("    NOTE: QMJ is a common factor (constant within each cross-section), so as an FM")
say("    control it is near-vacuous by construction; the substantive test is the spanning")
say("    alpha in (c) below. Reported for completeness / per R12 convention.")

# --- (b) VIF(dH_GPM, QMJ) pooled ---
vfu, rfu = vif_two(pfq["delta_h_z"], pfq["QMJ_z"] if pfq["QMJ_z"].notna().any() else pfq["QMJ"])
# QMJ_z is constant->NaN per q; use raw quarterly QMJ for the correlation diagnostic
vfu, rfu = vif_two(pfq["delta_h_z"], pfq["QMJ"])
vsp, rsp = vif_two(spq["dH_gpm_z"], spq["QMJ"])
say(f"\n(b) VIF(dH_GPM, QMJ)  [pooled, raw QMJ vs firm dH]:")
say(f"    Full-universe : corr={rfu:+.3f}  VIF={vfu:.2f}")
say(f"    S&P500 monthly: corr={rsp:+.3f}  VIF={vsp:.2f}")

# --- (c) SUBSTANTIVE spanning test: dH-sorted L/S alpha on FF5+UMD (+QMJ) ---
say("\n(c) Spanning test — dH-sorted Q5-Q1 L/S regressed on factor models:")
# S&P500 monthly L/S on dH_gpm_z
ls_sp, _ = quintile_sort_ls(sp_b.rename(columns={}), "dH_gpm_z", "ret_next_month", fac)
ls_sp = ls_sp.dropna(); ls_sp.index = pd.to_datetime(ls_sp.index)
rf_sp = fac["RF"].reindex(ls_sp.index).fillna(0) if "RF" in fac else 0
ls_sp_ex = ls_sp - rf_sp
facx = fac.copy(); facx["QMJ"] = qmj["QMJ"]
for cols, tag in [(ff_cols_all, "FF5+UMD"), (ff_cols_all + ["QMJ"], "FF5+UMD+QMJ")]:
    a, t, p, lag, n = spanning_alpha(ls_sp_ex, facx, cols)
    say(f"    SP500  {tag:14}: alpha={a*100:+.3f}%/mo  t={t:+.2f}  p={p:.4f}  [NWlag={lag}, n={n}]")

# Full-universe quarterly L/S on delta_h_z
pf2 = pf.rename(columns={"q": "date"}).copy()
ls_fu = (pf2.assign(_q=pf2.groupby("date")["delta_h_z"].transform(
            lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") if x.nunique() >= 5 else np.nan))
            .dropna(subset=["_q", "ret_next"])
            .groupby(["date", "_q"])["ret_next"].mean().unstack("_q"))
ls_fu = (ls_fu.get(4) - ls_fu.get(0)).dropna()
faq_idx = faq.copy(); faq_idx["QMJ"] = qmjq
faq_idx.index = faq_idx.index  # PeriodIndex Q
ls_fu.index = ls_fu.index  # PeriodIndex Q
for cols, tag in [(ff_cols_all, "FF5+UMD"), (ff_cols_all + ["QMJ"], "FF5+UMD+QMJ")]:
    d = pd.DataFrame({"ls": ls_fu}).join(faq_idx[cols], how="inner").dropna()
    y = d["ls"].values; X = sm.add_constant(d[cols].values)
    lag = max(bic_nw_lag(sm.OLS(y, X).fit().resid), 1)
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    say(f"    FU     {tag:14}: alpha={res.params[0]*100:+.3f}%/q   t={res.tvalues[0]:+.2f}  "
        f"p={res.pvalues[0]:.4f}  [NWlag={lag}, n={len(d)}]")

say("\n(d) SF1 earnings-quality/accruals factor: NOT RUN (not constructed at low cost from the")
say("    stored panels without new accrual-variable extraction). Stated as not run, not fabricated.")

# ═════════════════════════════════════════════════════════════════════════════
# A2 — MICROCAP ROBUSTNESS OF FULL-UNIVERSE t(dH)=+3.46
# ═════════════════════════════════════════════════════════════════════════════
say("\n" + "="*72); say("A2 — MICROCAP ROBUSTNESS OF HEADLINE FULL-UNIVERSE t(dH)"); say("="*72)

# merge price + marketcap (ARQ, formation quarter) and exchange
sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                      columns=["ticker", "dimension", "calendardate", "price", "marketcap"])
arq = sf1[sf1["dimension"] == "ARQ"].copy()
arq["calendardate"] = pd.to_datetime(arq["calendardate"], errors="coerce")
arq = arq.dropna(subset=["calendardate"])
arq["q"] = arq["calendardate"].dt.to_period("Q")
arq = (arq.sort_values(["ticker", "calendardate"])
          .drop_duplicates(["ticker", "q"], keep="last")[["ticker", "q", "price", "marketcap"]])
tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
exch = tk[tk["table"] == "SF1"][["ticker", "exchange"]].drop_duplicates("ticker")

pa = pf.merge(arq, on=["ticker", "q"], how="left").merge(exch, on="ticker", how="left")
say(f"\nMerged price/cap coverage: price {pa['price'].notna().mean():.1%}  "
    f"marketcap {pa['marketcap'].notna().mean():.1%}  exchange {pa['exchange'].notna().mean():.1%}")

def report_screen(panel, tag, wcol=None):
    o, _ = fama_macbeth_nw(panel, "ret_next", ["delta_h_z", "delta_s_z"], w_col=wcol)
    m, t, n = o.get("delta_h_z", (np.nan, np.nan, 0))
    avgn = panel.dropna(subset=["ret_next","delta_h_z","delta_s_z"]).groupby("q").size().mean()
    say(f"    {tag:34}: t(dH)={t:+.2f}  (avgN/q={avgn:.0f}, Tq={n})")
    return t

say("\nModel B t(dH) under microcap screens (full-universe quarterly):")
report_screen(pa, "baseline (all firms)")

# (i) NYSE 20th-pct market-cap breakpoint each quarter
nyse = pa[pa["exchange"] == "NYSE"]
bkpt = nyse.groupby("q")["marketcap"].quantile(0.20).rename("nyse20")
pn = pa.merge(bkpt, on="q", how="left")
pn = pn[pn["marketcap"] >= pn["nyse20"]]
report_screen(pn, "(i) NYSE 20th-pct size breakpoint")

# (ii) price >= $1 at formation
report_screen(pa[pa["price"] >= 1.0], "(ii) price >= $1 at formation")
# (iii) price >= $5
report_screen(pa[pa["price"] >= 5.0], "(iii) price >= $5 at formation")
# (iv) cap-weighted (WLS by formation-quarter marketcap)
report_screen(pa, "(iv) cap-weighted FM (WLS by cap)", wcol="marketcap")

# ═════════════════════════════════════════════════════════════════════════════
# A3 — FORMAL ASYMMETRY SLOPE-DIFFERENCE TEST
# ═════════════════════════════════════════════════════════════════════════════
say("\n" + "="*72); say("A3 — FORMAL ASYMMETRY SLOPE-DIFFERENCE TEST"); say("="*72)
say("  Stack beta_dH,t and beta_dS,t; regress  beta = a + b*T + c*D_S + d*(T*D_S).")
say("  d = differential T-sensitivity of the entropy slope vs the enthalpy slope.")

def a3_test(betas, Tser, panel_tag, dh="delta_h_z", ds="delta_s_z"):
    b = betas.copy()
    b.index.name = "t"
    T = Tser.reindex(b.index)
    df = pd.DataFrame({"bDH": b[dh], "bDS": b[ds], "T": T}).dropna()
    # stack: enthalpy (D_S=0), entropy (D_S=1)
    stk = pd.concat([
        pd.DataFrame({"beta": df["bDH"], "T": df["T"], "D_S": 0.0}),
        pd.DataFrame({"beta": df["bDS"], "T": df["T"], "D_S": 1.0}),
    ], ignore_index=True)
    stk["TxD"] = stk["T"] * stk["D_S"]
    X = sm.add_constant(stk[["T", "D_S", "TxD"]])
    lag = max(bic_nw_lag(sm.OLS(stk["beta"], X).fit().resid), 1)
    res = sm.OLS(stk["beta"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    d_, td, pd_ = res.params["TxD"], res.tvalues["TxD"], res.pvalues["TxD"]
    say(f"\n  {panel_tag}:  n(pairs)={len(df)}  NWlag={lag}")
    say(f"     b(T, enthalpy slope-on-T)   = {res.params['T']:+.4f}  (t={res.tvalues['T']:+.2f})")
    say(f"     d(T*D_S, entropy extra)     = {d_:+.4f}  (t={td:+.2f}, p={pd_:.4f})")
    return d_, td, pd_

# Full-universe: quarterly T per quarter
Tq_fu = pf.groupby("q")["T"].first()
d_fu = a3_test(betas_fu, Tq_fu, "FULL-UNIVERSE (quarterly)")
# S&P500 monthly: T per month
Tm_sp = sp_b.groupby("date")["T"].first()
d_sp = a3_test(betas_sp, Tm_sp, "S&P500 (monthly)", dh="dH_gpm_z", ds="DS_z")

# ═════════════════════════════════════════════════════════════════════════════
say("\n" + "="*72); say("RESULTS BLOCK SUMMARY"); say("="*72)
say(f"A1  FU t(dH|+QMJ FM control) = {o_fu_qmj.get('delta_h_z',(0,np.nan))[1]:+.2f}   "
    f"VIF(dH,QMJ)_FU={vfu:.2f}")
say(f"A2  see screen table above (NYSE-bkpt / price / cap-weighted)")
say(f"A3  FU interaction d: t={d_fu[1]:+.2f} p={d_fu[2]:.4f} | SP500 d: t={d_sp[1]:+.2f} p={d_sp[2]:.4f}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "A_runs_plos_results.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\nSaved: {out}")
