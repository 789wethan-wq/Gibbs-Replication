"""R23 -- R19 stress-test delta calibration (BLOCKING for the R19 claim only).

Table 14 reports delta=0.5%/month taking the disorder L/S from +13.4%/yr to
+0.3%/yr. This script (1) pastes and states unambiguously the R19 stress-loop
definition of delta, (2) reconciles that magnitude analytically against the
code's actual mechanism (not just re-running the code, which is circular),
and (3) recalibrates delta against an empirical performance-delisting rate
for the relevant reference population.

Report as run. No tuning toward the manuscript value.

Outputs: robustness/outputs/R23_r19_delta_calibration_results.txt
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260617)

LOG = []
def say(*a):
    line = " ".join(str(x) for x in a)
    print(line); LOG.append(line)

# ═══════════════════════════════════════════════════════════════════════════
say("=" * 78); say("R23.1 -- DEFINITIONAL AUDIT OF THE R19 STRESS-TEST LOOP"); say("=" * 78)

say("""
VERBATIM (robustness/R19_delisting_bias_bound.py, lines 36-37 and 76-100):

DR_NYSE, DR_NASDAQ, DR_BLEND = -0.30, -0.55, -0.40

def stress(delta, dr):
    \"\"\"Apply per-month delisting at rate `delta` to most-distressed survivors.\"\"\"
    if delta == 0:
        return p.copy()
    d = p.copy().sort_values(["date","stock_id"])
    dead = set()                      # tickers already delisted (removed going fwd)
    out = []
    for dt, g in d.groupby("date"):
        g = g[~g["stock_id"].isin(dead)]
        if g.empty: continue
        g = g.copy()
        # candidates = most distressed (top-quartile disorder) this month
        if g["DS_z"].notna().sum() >= 8:
            thr = g["DS_z"].quantile(0.75)
            cand = g.index[g["DS_z"] >= thr]
        else:
            cand = g.index
        n_del = int(round(delta * len(g)))
        if n_del > 0 and len(cand) > 0:
            n_del = min(n_del, len(cand))
            chosen = RNG.choice(cand, size=n_del, replace=False)
            g.loc[chosen, "ret_next_month"] = dr        # delisting return
            dead.update(g.loc[chosen, "stock_id"].tolist())
        out.append(g)
    return pd.concat(out)

grid used for Table 14: [(0.0, DR_BLEND), (0.005, DR_BLEND), (0.01, DR_BLEND), (0.02, DR_BLEND)]
""")

say("Unambiguous statements:")
say("  (a) delta is a FRACTION OF THE FULL CROSS-SECTION each month (n_del = round(delta *")
say("      len(g)), where len(g) = all surviving firms that month), but the n_del chosen")
say("      firms are drawn ONLY from the candidate pool = top quartile of DS_z (disorder).")
say("      This is definition (b) in the R23 spec: 'a fraction of the full panel drawn")
say("      from the top quartile' -- NOT a per-firm hazard applied within the quartile.")
say("      The per-firm hazard WITHIN the quartile pool is delta / (quartile share of the")
say("      cross-section), i.e. approximately 4*delta if the quartile pool is exactly 25%")
say("      of that month's cross-section (measured empirically below).")
say("  (b) Delisting return substituted: DR_BLEND = -0.40, a blend of -30% (NYSE/AMEX,")
say("      Shumway 1997) and -55% (Nasdaq, Shumway-Warther 1999). Source: manuscript Sec 4.8")
say("      text, matches R19.py DR_NYSE/DR_NASDAQ/DR_BLEND constants exactly.")
say("  (c) A delisted firm's NEXT return is replaced by dr ONE TIME, and the firm is then")
say("      PERMANENTLY REMOVED from the panel in all subsequent months (dead.update(...),")
say("      filtered via `g[~g['stock_id'].isin(dead)]` every iteration going forward).")
say("      It is not merely a single-month return replacement with continued inclusion.")

# ═══════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 78); say("R23.2 -- ANALYTIC RECONCILIATION"); say("=" * 78)

p = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
p["date"] = pd.to_datetime(p["date"])
p = p.dropna(subset=["ret_next_month"]).sort_values(["stock_id","date"])
n_months = p["date"].nunique()
avg_n = p.groupby("date")["stock_id"].nunique().mean()
say(f"\nPrimary S&P500 monthly panel: N={len(p):,}  tickers={p['stock_id'].nunique()}  "
    f"months={n_months}  ({p['date'].min().date()}..{p['date'].max().date()})")
say(f"Average cross-section size per month: {avg_n:.1f}")

# empirical candidate-pool share (top quartile of DS_z each month)
pool_shares = []
for dt, g in p.groupby("date"):
    if g["DS_z"].notna().sum() < 8: continue
    thr = g["DS_z"].quantile(0.75)
    pool_shares.append((g["DS_z"] >= thr).sum() / len(g))
pool_share = np.mean(pool_shares)
say(f"Empirical top-quartile candidate-pool share of cross-section: {pool_share:.4f}")

def fm_t(panel, ycol, xcol, datecol="date", lags=5, min_cs=20):
    coefs = []
    for d, g in panel.groupby(datecol):
        sub = g[[ycol, xcol]].dropna()
        if len(sub) < min_cs: continue
        X = sm.add_constant(sub[[xcol]], has_constant="add")
        coefs.append(sm.OLS(sub[ycol], X).fit().params[xcol])
    s = pd.Series(coefs).dropna(); n = len(s)
    if n < 5: return np.nan, np.nan
    m = s.mean(); g0 = (s**2).mean() - m**2; var = g0
    for l in range(1, min(lags+1, n)):
        var += 2*(1-l/(lags+1))*((s.iloc[l:].values-m)*(s.iloc[:-l].values-m)).mean()
    return m, m/np.sqrt(max(var,1e-30)/n)

def ls_quintile(panel, sortcol, ycol="ret_next_month", datecol="date"):
    d = panel.dropna(subset=[sortcol, ycol]).copy()
    d["q"] = d.groupby(datecol)[sortcol].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
        if x.nunique() >= 5 else np.nan)
    d = d.dropna(subset=["q"])
    qr = d.groupby([datecol,"q"])[ycol].mean().unstack("q")
    if 0 not in qr.columns or 4 not in qr.columns: return np.nan, np.nan, {}
    ls = (qr[4]-qr[0]).dropna()
    t = ls.mean()/(ls.std()/np.sqrt(len(ls)))
    means = {int(c): qr[c].mean() for c in qr.columns}
    return ls.mean()*12, t, means

def stress(delta, dr):
    if delta == 0:
        return p.copy()
    d = p.copy().sort_values(["date","stock_id"])
    dead = set()
    out = []
    for dt, g in d.groupby("date"):
        g = g[~g["stock_id"].isin(dead)]
        if g.empty: continue
        g = g.copy()
        if g["DS_z"].notna().sum() >= 8:
            thr = g["DS_z"].quantile(0.75)
            cand = g.index[g["DS_z"] >= thr]
        else:
            cand = g.index
        n_del = int(round(delta * len(g)))
        if n_del > 0 and len(cand) > 0:
            n_del = min(n_del, len(cand))
            chosen = RNG.choice(cand, size=n_del, replace=False)
            g.loc[chosen, "ret_next_month"] = dr
            dead.update(g.loc[chosen, "stock_id"].tolist())
        out.append(g)
    return pd.concat(out)

DR_BLEND = -0.40
say("\n-- reproducing Table 14 (verbatim mechanism, RNG fixed as in R19.py) --")
say(f"{'δ/mo':>7} {'disorder L/S %/yr':>18} {'FM t(ΔS)':>10}")
grid = [0.0, 0.005, 0.01, 0.02]
tab14 = {}
_, base_means = None, None
for delta in grid:
    ps = stress(delta, DR_BLEND)
    ann, t, means = ls_quintile(ps, "DS_z")
    tab14[delta] = (ann, t, means)
    say(f"{delta:>7.3f} {ann*100:>+18.2f} {t:>+10.2f}")
    if delta == 0.0:
        base_means = means

say(f"\nBaseline (δ=0) Q5/Q1 mean monthly returns: Q1={base_means[0]*100:+.3f}%  "
    f"Q5={base_means[4]*100:+.3f}%")

say("\nImplied per-firm hazard / annual hazard / cohort survival at each δ:")
say(f"{'δ/mo':>7} {'h=δ/pool_share':>16} {'h annual':>10} {'cohort surv @T='+str(n_months)+'mo':>22}")
for delta in grid:
    if delta == 0:
        say(f"{delta:>7.3f} {'--':>16} {'--':>10} {'--':>22}")
        continue
    h = delta / pool_share
    h_ann = 1 - (1-h)**12
    surv = (1-h)**n_months
    say(f"{delta:>7.3f} {h:>16.4f} {h_ann:>10.2%} {surv:>22.2e}")

say("\nAnalytic mechanical-drag reconciliation (linear first-order approximation):")
say("  Candidates are drawn ONLY from DS_z>=P75 (never touches Q1), so the L/S (Q5-Q1)")
say("  drop should be driven almost entirely by the drop in Q5's mean return. Approximate:")
say("    replaced_frac_of_Q5 ≈ delta / (Q5 share of cross-section) = delta / 0.20")
say("    analytic_dQ5/month ≈ replaced_frac_of_Q5 * (organic_Q5_ret - dr)")
say("    analytic_dLS/yr    ≈ 12 * analytic_dQ5/month   (Q1 ≈ unaffected)")
q5_ret = base_means[4]
for delta in grid:
    if delta == 0: continue
    replaced_frac_q5 = delta / 0.20
    analytic_dq5 = replaced_frac_q5 * (q5_ret - DR_BLEND)
    analytic_dls_yr = 12 * analytic_dq5
    actual_ls_ann = tab14[delta][0]
    actual_drop = tab14[0.0][0] - actual_ls_ann
    analytic_drop_pct = analytic_dls_yr * 100
    say(f"  δ={delta:.3f}: analytic drop={analytic_drop_pct:+.1f}pp/yr   "
        f"actual observed drop={actual_drop*100:+.1f}pp/yr   "
        f"ratio(actual/analytic)={ (actual_drop*100)/analytic_drop_pct if analytic_drop_pct!=0 else float('nan'):.2f}")
say("\n  (Discrepancy, if any, reflects compounding attrition -- removed firms are excluded")
say("   from ALL subsequent cross-sections, not just shocked once, and quartile")
say("   re-ranking after removals pulls in previously-Q4 firms -- effects the linear")
say("   first-order approximation does not capture. Reported as computed, not adjusted.)")

# ═══════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 78); say("R23.3 -- EMPIRICAL DELISTING-RATE RECALIBRATION"); say("=" * 78)

say("\nAttempted reference population: S&P 500 constituents, top iVol quartile, 1995-2023.")
say("SEP (point-in-time monthly prices for historical/delisted names) is NOT entitled on")
say("this Sharadar key (established in prior runs), so returns for departed S&P500 names")
say("cannot be pulled directly. The point-in-time membership file data/")
say("sp500_monthly_membership.parquet was checked as an alternative:")

mem = pd.read_parquet(f"{DATA}/sp500_monthly_membership.parquet")
avg_by_year = mem.groupby(pd.to_datetime(mem["date"]).dt.year)["stock_id"].nunique()
say(f"\n  Average member count per year (should be ~500 throughout):")
for yr in [1995,1996,1997,1998,1999,2000,2005,2010,2015,2020,2023]:
    if yr in avg_by_year.index:
        say(f"    {yr}: {avg_by_year[yr]:.1f}")
say(f"\n  CONFIRMED DEFECT: average membership collapses from {avg_by_year.get(1997,np.nan):.0f} in 1997")
say(f"  to {avg_by_year.get(1999,np.nan):.0f} in 1999 and stays in the single digits through 2023.")
say("  data/sp500_monthly_membership.parquet is NOT usable for the requested 1995-2023 window")
say("  (it is a known construction defect in project/sharadar_pipeline.py build_sp500_membership,")
say("  not fixed as part of this run). Stated plainly per the R23.3 instruction.")

say("\nCLOSEST AVAILABLE PROXY (population explicitly redefined, not S&P500 membership):")
say("  NYSE 20th-percentile-or-above market-cap firms in the R18 full-universe quarterly")
say("  panel, 1995-2023 -- the same institutional/large-cap screen used in the A2 robustness")
say("  run (a standard large-cap proxy population when true index membership is unavailable).")

# rebuild marketcap/exchange screen exactly as in A_runs_plos.py A2
q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                      columns=["ticker","dimension","calendardate","price","marketcap"])
arq = sf1[sf1["dimension"] == "ARQ"].copy()
arq["calendardate"] = pd.to_datetime(arq["calendardate"], errors="coerce")
arq = arq.dropna(subset=["calendardate"])
arq["q"] = arq["calendardate"].dt.to_period("Q")
arq = (arq.sort_values(["ticker", "calendardate"])
          .drop_duplicates(["ticker", "q"], keep="last")[["ticker", "q", "price", "marketcap"]])
tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
exch = tk[tk["table"] == "SF1"][["ticker", "exchange"]].drop_duplicates("ticker")
sf1t_full = tk[tk["table"] == "SF1"][["ticker","isdelisted"]].drop_duplicates("ticker")
delisted_set = set(sf1t_full.loc[sf1t_full["isdelisted"] == "Y", "ticker"])

pa = q.merge(arq, on=["ticker", "q"], how="left").merge(exch, on="ticker", how="left")
nyse = pa[pa["exchange"] == "NYSE"]
bkpt = nyse.groupby("q")["marketcap"].quantile(0.20).rename("nyse20")
pa = pa.merge(bkpt, on="q", how="left")
lc = pa[pa["marketcap"] >= pa["nyse20"]].dropna(subset=["marketcap"]).copy()
lc = lc[(lc["q"].dt.year >= 1995) & (lc["q"].dt.year <= 2023)]
say(f"\n  Large-cap (NYSE 20th-pct+) proxy sample: N={len(lc):,}  tickers={lc['ticker'].nunique():,}  "
    f"quarters={lc['q'].nunique()}  1995-2023")

# DS_z quartile within the large-cap subsample each quarter
lc["ds_q_lc"] = lc.groupby("q")["delta_s_z"].transform(
    lambda x: pd.qcut(x, 4, labels=False, duplicates="drop") if x.notna().sum() >= 8 else np.nan)

# firm's true LAST quarter across the FULL panel (real data-coverage end / delisting proxy)
last_q_full = q.groupby("ticker")["q_ord"].max().rename("last_q_ord_full")
lc = lc.merge(last_q_full, on="ticker", how="left")
lc["is_delisted_ticker"] = lc["ticker"].isin(delisted_set)
lc["delisted_within_4q"] = (
    lc["is_delisted_ticker"] &
    (lc["last_q_ord_full"] >= lc["q_ord"] + 1) &
    (lc["last_q_ord_full"] <= lc["q_ord"] + 4)
)

say("\n  Forward 4-quarter (~1yr) data-exit/delisting rate by large-cap DS_z quartile:")
say(f"    {'DS quartile':>12} {'N':>8} {'rate/yr':>9} {'rate/mo (÷12)':>14}")
for qd in sorted(lc["ds_q_lc"].dropna().unique()):
    sub = lc[lc["ds_q_lc"] == qd]
    rate_yr = sub["delisted_within_4q"].mean()
    say(f"    Q{int(qd)+1:>11} {len(sub):>8,} {rate_yr:>8.2%} {rate_yr/12:>13.3%}")

top_q = lc["ds_q_lc"].max()
top_rate = lc.loc[lc["ds_q_lc"] == top_q, "delisted_within_4q"].mean()
say(f"\n  Top-DS_z-quartile large-cap annual data-exit rate: {top_rate:.2%}  "
    f"({top_rate/12:.3%}/month)")
say("  CAVEAT: Sharadar's isdelisted=='Y' flag does not distinguish performance-related")
say("  delisting (bankruptcy, poor-performance) from M&A/acquisition-related exits, so")
say("  this rate is a broad 'coverage exit' rate, not a pure performance-delisting rate")
say("  as in Shumway (1997)'s classification. It plausibly OVERSTATES the pure-performance")
say("  component (M&A exits inflate it) while UNDERSTATING what a true point-in-time")
say("  S&P500-only population would show (large caps are acquired more, delist for")
say("  distress less, than the broader universe this large-cap screen still contains).")

say(f"\nComparison to the delta grid:")
say(f"  δ=0.5%/month implies per-firm hazard WITHIN quartile ≈ {0.005/pool_share:.2%}/month "
    f"({(1-(1-0.005/pool_share)**12):.1%}/yr annualized)")
say(f"  δ=1.0%/month implies per-firm hazard WITHIN quartile ≈ {0.01/pool_share:.2%}/month "
    f"({(1-(1-0.01/pool_share)**12):.1%}/yr annualized)")
say(f"  Empirical large-cap top-DS_z-quartile exit rate (this proxy): {top_rate:.2%}/yr "
    f"({top_rate/12:.3%}/month)")

out_txt = f"{OUT}/R23_r19_delta_calibration_results.txt"
with open(out_txt, "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\nSaved: {out_txt}")
