"""F2 — Market-cap-threshold universe (separating survivorship from index
membership). At each historical quarter, include every R18-panel firm above
an ABSOLUTE (CPI-adjusted, constant 2023 dollars) capitalization threshold --
$1B, $5B, $10B -- with delisted firms retained and no survival requirement.
Report FM t(DeltaS), t(DeltaH) for each threshold.

Uses LAGGED (t-1) market cap for the threshold cut, per the corrected-panel
default (ground rule 4, prior rounds).

CPI-U (BLS series CUUR0000SA0, annual averages, 1982-84=100) used to convert
the stated 2023-dollar thresholds into period-appropriate nominal thresholds.
These are standard published BLS figures reproduced from general knowledge,
not independently re-verified via a live data pull in this environment --
stated explicitly so the convention is auditable and replaceable if the
author wants to substitute an exact vintage-matched series.
"""
import os
import hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
sys.path.insert(0, "../project")
from utils import newey_west_mean_tstat

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/F2_cap_threshold_universe.txt"

print(f"[pid={os.getpid()}] F2 — fresh process")
log = []
def P(s=""):
    print(s)
    log.append(str(s))

# BLS CPI-U annual averages (1982-84=100), standard published series
CPI = {
    1995: 152.4, 1996: 156.9, 1997: 160.5, 1998: 163.0, 1999: 166.6,
    2000: 172.2, 2001: 177.1, 2002: 179.9, 2003: 184.0, 2004: 188.9,
    2005: 195.3, 2006: 201.6, 2007: 207.3, 2008: 215.3, 2009: 214.6,
    2010: 218.1, 2011: 224.9, 2012: 229.6, 2013: 233.0, 2014: 236.7,
    2015: 237.0, 2016: 240.0, 2017: 245.1, 2018: 251.1, 2019: 255.7,
    2020: 258.8, 2021: 271.0, 2022: 292.7, 2023: 304.7,
}
BASE_YEAR = 2023

P("="*88)
P("F2 — Market-cap-threshold universe (CPI-adjusted, constant 2023 dollars)")
P("="*88)
P(f"CPI base year: {BASE_YEAR} (CPI={CPI[BASE_YEAR]})")
P("Nominal threshold(year) = real_threshold_2023dollars * CPI(year) / CPI(2023)")


def cs_wz(df, col, date_col="q", pct=0.01):
    def _f(x):
        x2 = x.dropna()
        if len(x2) < 5:
            return pd.Series(np.nan, index=x.index)
        lo, hi = x2.quantile(pct), x2.quantile(1 - pct)
        xc = x.clip(lo, hi)
        s = xc.std()
        if s < 1e-10:
            return pd.Series(np.nan, index=x.index)
        return (xc - xc.mean()) / s
    return df.groupby(date_col)[col].transform(_f)


def fama_macbeth_nw(panel, y_col, x_cols, date_col="q", lags=4, min_cs=20):
    coefs = []
    for d, g in panel.groupby(date_col):
        s = g[[y_col] + x_cols].dropna()
        if len(s) < max(min_cs, len(x_cols) + 2):
            continue
        X = sm.add_constant(s[x_cols], has_constant="add")
        coefs.append(sm.OLS(s[y_col], X).fit().params[x_cols].rename(d))
    if not coefs:
        return {}
    cdf = pd.DataFrame(coefs)
    out = {}
    for c in x_cols:
        s = cdf[c].dropna()
        mean_, t_, p_ = newey_west_mean_tstat(s.values, lags=lags)
        se_ = mean_ / t_ if t_ != 0 else np.nan
        out[c] = dict(coef=mean_, se=se_, t=t_, n=len(s))
    return out


panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                       columns=["ticker", "dimension", "calendardate", "marketcap"])
mc = sf1[sf1["dimension"] == "ARQ"].copy()
mc["calendardate"] = pd.to_datetime(mc["calendardate"], errors="coerce")
mc = mc.dropna(subset=["calendardate", "marketcap"])
mc = mc[mc["marketcap"] > 0]
mc["q"] = mc["calendardate"].dt.to_period("Q")
mc = mc.sort_values(["ticker", "calendardate"]).drop_duplicates(["ticker", "q"], keep="last")[["ticker", "q", "marketcap"]]
mc = mc.sort_values(["ticker", "q"])
mc["marketcap_lag1"] = mc.groupby("ticker")["marketcap"].shift(1)
panel = panel.merge(mc, on=["ticker", "q"], how="left")
panel["year"] = panel["q"].apply(lambda p: p.year)
panel["cpi_factor"] = panel["year"].map(CPI) / CPI[BASE_YEAR]

P(f"\nBase panel: N={len(panel):,}  tickers={panel['ticker'].nunique():,}  "
  f"quarters={panel['q'].nunique()}  range={panel['q'].min()}..{panel['q'].max()}")
P(f"Lagged-cap coverage: {panel['marketcap_lag1'].notna().mean():.1%}")

thresholds_2023usd = [1e9, 5e9, 10e9]
results = []
for thr in thresholds_2023usd:
    d = panel.dropna(subset=["marketcap_lag1", "cpi_factor"]).copy()
    d["nominal_threshold"] = thr * d["cpi_factor"]
    d = d[d["marketcap_lag1"] >= d["nominal_threshold"]].copy()
    d["ds_z"] = cs_wz(d, "delta_s")
    d["dh_z"] = cs_wz(d, "dH_gpm")
    pf = d.dropna(subset=["ret_next", "ds_z", "dh_z"])
    fm = fama_macbeth_nw(pf, "ret_next", ["dh_z", "ds_z"])
    if "ds_z" not in fm:
        P(f"\n[${thr/1e9:.0f}B threshold]: INFEASIBLE (no quarters meet min_cs)")
        continue
    avg_n = pf.groupby("q").size().mean()
    n_tick = pf["ticker"].nunique()
    n_tick_delisted = pf.merge(
        pd.read_parquet(f"{DATA}/sharadar_tickers.parquet").query("table=='SF1'")[["ticker", "isdelisted"]],
        on="ticker", how="left")
    delisted_share = (n_tick_delisted.drop_duplicates("ticker")["isdelisted"] == "Y").mean()
    first_q, last_q = str(pf["q"].min()), str(pf["q"].max())
    X = sm.add_constant(pf[["dh_z", "ds_z"]]).values
    dhash = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()[:12]
    P(f"\n[${thr/1e9:.0f}B threshold (2023 dollars, CPI-adjusted per quarter)]")
    P(f"  N={len(pf):,}  tickers={n_tick:,}  avg firms/qtr={avg_n:.1f}  delisted-share={delisted_share*100:.1f}%  "
      f"date_range={first_q}..{last_q}  design_hash={dhash}")
    P(f"  t(dS)={fm['ds_z']['t']:+.4f}  coef(dS)={fm['ds_z']['coef']:+.6f}  SE(dS)={fm['ds_z']['se']:.6f}  "
      f"quarters={fm['ds_z']['n']}")
    P(f"  t(dH)={fm['dh_z']['t']:+.4f}  coef(dH)={fm['dh_z']['coef']:+.6f}  SE(dH)={fm['dh_z']['se']:.6f}")
    results.append(dict(thr=thr, N=len(pf), n_tick=n_tick, avg_n=avg_n, delisted_share=delisted_share,
                         t_ds=fm['ds_z']['t'], coef_ds=fm['ds_z']['coef'], se_ds=fm['ds_z']['se'],
                         t_dh=fm['dh_z']['t'], coef_dh=fm['dh_z']['coef'], se_dh=fm['dh_z']['se'],
                         first_q=first_q, last_q=last_q))

P("\n" + "="*88)
P("F2 SUMMARY TABLE")
P("="*88)
P(f"{'Threshold':12}{'t(dS)':>9}{'t(dH)':>9}{'N':>10}{'tickers':>9}{'avgN/q':>8}{'delisted%':>11}")
for r in results:
    P(f"${r['thr']/1e9:>3.0f}B{'':7}{r['t_ds']:>+9.3f}{r['t_dh']:>+9.3f}{r['N']:>10,}{r['n_tick']:>9,}"
      f"{r['avg_n']:>8.1f}{r['delisted_share']*100:>10.1f}%")

P("\nInterpretation: this universe is conditioned on size ALONE, at every quarter, with")
P("delisted firms retained and no index committee, membership file, or endpoint")
P("conditioning of any kind -- it isolates the size margin from the survivorship/")
P("index-membership margin that Section 4.8 concedes are not separated elsewhere.")
if all(abs(r["t_ds"]) < 2.0 for r in results):
    P("\nFINDING: t(DeltaS) stays near zero / insignificant at every absolute-cap threshold")
    P("tested. This strengthens the survivorship reading: the premium does not appear")
    P("under size conditioning alone, so the S&P 500 comparison panel's premium is not")
    P("simply a large-cap phenomenon recoverable by capitalization screening -- it")
    P("specifically requires the additional (index-membership or continuous-survival)")
    P("selection margin the original panel embeds.")
else:
    sig = [r for r in results if abs(r["t_ds"]) >= 2.0]
    sig_desc = ", ".join(f"${r['thr']/1e9:.0f}B: t={r['t_ds']:+.2f}" for r in sig)
    P(f"\nFINDING: t(DeltaS) is significant at {len(sig)}/{len(results)} threshold(s) tested "
      f"({sig_desc}).")
    P("This means index membership (beyond size conditioning alone) IS doing some of the")
    P("work the paper currently attributes to survival -- absolute-size conditioning")
    P("alone recovers part of the premium. Section 4.8 should be revised to reflect that")
    P("size and survivorship/index-membership are not the same margin, and this exhibit")
    P("shows size alone is not sufficient to explain the full picture.")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
