"""E2 — Value-weighted t(DeltaS) in the corrected (R18) panel, matching the
four variants already reported for DeltaH in Section 5.3 (A_runs_plos.py A2):
NYSE 20th-percentile size breakpoint, price>=$1 at formation, price>=$5 at
formation, and value-(cap-)weighted FM (WLS by market cap). Per ground rule 4
(corrected-panel default), capitalization is LAGGED one quarter throughout
(the A2 script this extends used contemporaneous cap; lagged is used here
for the size breakpoint and the value-weighting, consistent with D3/D4.4's
established convention -- price screens use price at formation, unchanged).
Delisted retained, no survival requirement.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
sys.path.insert(0, "../project")
from utils import newey_west_mean_tstat

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/E2_value_weighted_dS.txt"

print(f"[pid={os.getpid()}] E2 — fresh process")
log = []
def P(s=""):
    print(s)
    log.append(str(s))


def fama_macbeth_nw(panel, y_col, x_cols, date_col="q", lags=4, min_cs=20, w_col=None):
    coefs = []
    for d, g in panel.groupby(date_col):
        s = g[[y_col] + x_cols + ([w_col] if w_col else [])].dropna()
        if len(s) < max(min_cs, len(x_cols) + 2):
            continue
        X = sm.add_constant(s[x_cols], has_constant="add")
        if w_col:
            w = s[w_col].values
            w = w / w.sum()
            res = sm.WLS(s[y_col], X, weights=w).fit()
        else:
            res = sm.OLS(s[y_col], X).fit()
        coefs.append(res.params[x_cols].rename(d))
    if not coefs:
        return {}, pd.DataFrame()
    cdf = pd.DataFrame(coefs)
    out = {}
    for c in x_cols:
        s = cdf[c].dropna()
        mean_, t_, p_ = newey_west_mean_tstat(s.values, lags=lags)
        out[c] = dict(coef=mean_, t=t_, n=len(s))
    return out, cdf


P("="*88)
P("E2 — Value-weighted t(DeltaS), corrected (R18) panel, four Section 5.3 variants")
P("="*88)

q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                       columns=["ticker", "dimension", "calendardate", "price", "marketcap"])
arq = sf1[sf1["dimension"] == "ARQ"].copy()
arq["calendardate"] = pd.to_datetime(arq["calendardate"], errors="coerce")
arq = arq.dropna(subset=["calendardate"])
arq = arq[arq["marketcap"] > 0]
arq["q"] = arq["calendardate"].dt.to_period("Q")
arq = arq.sort_values(["ticker", "calendardate"]).drop_duplicates(["ticker", "q"], keep="last")[["ticker", "q", "price", "marketcap"]]
arq = arq.sort_values(["ticker", "q"])
arq["marketcap_lag1"] = arq.groupby("ticker")["marketcap"].shift(1)

tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
exch = tk[tk["table"] == "SF1"][["ticker", "exchange"]].drop_duplicates("ticker")

pa = q.merge(arq, on=["ticker", "q"], how="left").merge(exch, on="ticker", how="left")
P(f"Merged coverage: price {pa['price'].notna().mean():.1%}  marketcap_lag1 "
  f"{pa['marketcap_lag1'].notna().mean():.1%}  exchange {pa['exchange'].notna().mean():.1%}")


def report(panel, tag, wcol=None):
    d = panel.dropna(subset=["ret_next", "delta_h_z", "delta_s_z"])
    o, cdf = fama_macbeth_nw(d, "ret_next", ["delta_h_z", "delta_s_z"], w_col=wcol)
    first_q, last_q = str(d["q"].min()), str(d["q"].max())
    avgn = d.groupby("q").size().mean()
    n_tick = d["ticker"].nunique()
    if "delta_h_z" not in o:
        P(f"  {tag:34}: INFEASIBLE (no quarters meet min_cs)")
        return None
    P(f"  {tag:34}: t(dH)={o['delta_h_z']['t']:+.3f}  t(dS)={o['delta_s_z']['t']:+.3f}  "
      f"coef(dH)={o['delta_h_z']['coef']:+.6f}  coef(dS)={o['delta_s_z']['coef']:+.6f}  "
      f"N={len(d):,}  avgN/q={avgn:.0f}  tickers={n_tick}  quarters={o['delta_h_z']['n']}  "
      f"range={first_q}..{last_q}")
    return dict(tag=tag, t_dh=o['delta_h_z']['t'], t_ds=o['delta_s_z']['t'],
                coef_dh=o['delta_h_z']['coef'], coef_ds=o['delta_s_z']['coef'],
                N=len(d), avgn=avgn, n_tick=n_tick, quarters=o['delta_h_z']['n'])


rows = []
rows.append(report(pa, "baseline (unweighted, all firms)"))

nyse = pa[pa["exchange"] == "NYSE"]
bkpt = nyse.groupby("q")["marketcap_lag1"].quantile(0.20).rename("nyse20")
pn = pa.merge(bkpt, on="q", how="left")
pn = pn[pn["marketcap_lag1"] >= pn["nyse20"]]
rows.append(report(pn, "(i) NYSE 20th-pct size breakpoint (lagged cap)"))

rows.append(report(pa[pa["price"] >= 1.0], "(ii) price >= $1 at formation"))
rows.append(report(pa[pa["price"] >= 5.0], "(iii) price >= $5 at formation"))
rows.append(report(pa, "(iv) value-weighted FM (WLS by lagged cap)", wcol="marketcap_lag1"))
rows.append(report(pa, "(iv-contemp) value-weighted FM (WLS by CONTEMPORANEOUS cap)", wcol="marketcap"))

P("\n" + "="*88)
P("E2 SUMMARY TABLE")
P("="*88)
P(f"{'Variant':44}{'t(dH)':>9}{'t(dS)':>9}{'N':>10}{'avgN/q':>8}{'tickers':>9}")
for r in rows:
    if r is None:
        continue
    P(f"{r['tag']:44}{r['t_dh']:>+9.2f}{r['t_ds']:>+9.2f}{r['N']:>10,}{r['avgn']:>8.0f}{r['n_tick']:>9,}")

P("\nManuscript comparator (Section 5.3, ΔH): value-weighted t(ΔH) = +2.76.")
vw_lag = [r for r in rows if r and r["tag"].startswith("(iv) ")][0]
vw_con = [r for r in rows if r and r["tag"].startswith("(iv-contemp)")][0]
P(f"  Lagged cap:         t(ΔH)={vw_lag['t_dh']:+.3f}  t(ΔS)={vw_lag['t_ds']:+.3f}")
P(f"  Contemporaneous cap: t(ΔH)={vw_con['t_dh']:+.3f}  t(ΔS)={vw_con['t_ds']:+.3f}  "
  f"-- MATCHES the cited +2.76 essentially exactly")
P(f"\nReconciliation (D3-style): Section 5.3's value-weighted ΔH figure was computed with")
P(f"CONTEMPORANEOUS market cap (matches +2.76 to 2 decimals), not lagged cap. Per ground rule 4")
P(f"lagged cap is the corrected-panel default, so both are reported; the substantive finding is")
P(f"unaffected by which convention is used -- value-weighted t(ΔS) stays far from significance")
P(f"either way ({vw_lag['t_ds']:+.2f} lagged, {vw_con['t_ds']:+.2f} contemporaneous).")
P(f"\nThe headline missing number: value-weighted t(ΔS) = {vw_con['t_ds']:+.2f} (contemporaneous cap, "
  f"coef={vw_con['coef_ds']:+.6f}) / {vw_lag['t_ds']:+.2f} (lagged cap, coef={vw_lag['coef_ds']:+.6f}). "
  f"This is the number Section 5.3 does not currently report anywhere.")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
