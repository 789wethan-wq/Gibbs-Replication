"""R27 — Financial-firm exclusion. FM t(ΔH) and t(ΔS), both panels, excluding
SIC 6000-6999 (Finance, Insurance, and Real Estate). Reports N and avg
firms/period, with-financials vs ex-financials, both panels.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
warnings_ = None
sys.path.insert(0, "../project")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/R27_financial_exclusion.txt"

from utils import newey_west_mean_tstat

log = []
def P(s=""):
    print(s)
    log.append(str(s))


def cs_wz(df, col, date_col, pct=0.01):
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


def fm_ladder(panel, ycol, xcols, datecol, lag, min_cs=None):
    """min_cs: canonical primary-rig floor per panel. SP500 monthly rig uses
    len(xcols)+2=4 (T2_LOCK.py 'patched primary rig'); full-universe quarterly
    rig uses 20 (R22_v19_battery.py fm(), reproduces the manuscript's +3.46/+0.02
    exactly -- verified: min_cs=4 gives t(dH)=+1.06/t(dS)=+0.25, min_cs=20 gives
    t(dH)=+3.46/t(dS)=+0.02)."""
    if min_cs is None:
        min_cs = len(xcols) + 2
    dates = sorted(panel[datecol].unique())
    rows = []
    for d in dates:
        sub = panel[panel[datecol] == d].dropna(subset=[ycol] + xcols)
        if len(sub) < max(min_cs, len(xcols) + 2):
            continue
        X = sm.add_constant(sub[xcols])
        res = sm.OLS(sub[ycol], X).fit()
        rows.append(pd.Series(res.params, name=d))
    cdf = pd.DataFrame(rows)
    out = {}
    for c in xcols:
        mean_, t_, p_ = newey_west_mean_tstat(cdf[c].values, lags=lag)
        out[c] = (mean_, t_, p_)
    return out, cdf


tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sic = tk[["ticker", "siccode", "sicsector"]].dropna(subset=["siccode"]).drop_duplicates("ticker")
sic["siccode"] = pd.to_numeric(sic["siccode"], errors="coerce")
sic["is_financial"] = sic["siccode"].between(6000, 6999)
P("="*78)
P("R27 — Financial-firm exclusion (SIC 6000-6999)")
P("="*78)
P(f"Ticker-level SIC coverage: {sic['ticker'].nunique():,} tickers, "
  f"{sic['is_financial'].sum():,} flagged financial ({sic['is_financial'].mean()*100:.1f}%)")
P(f"sicsector breakdown of flagged financial tickers: "
  f"{sic.loc[sic['is_financial'],'sicsector'].value_counts().to_dict()}")

# ── SP500 monthly panel ──────────────────────────────────────────────────────
P("\n" + "-"*78)
P("SP500 monthly panel (Model B: dH_gpm_z + DS_z)")
P("-"*78)
m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
m["date"] = pd.to_datetime(m["date"])
m["dH_gpm_z"] = cs_wz(m, "dH_gpm", "date")
m = m.merge(sic[["ticker", "is_financial"]].rename(columns={"ticker": "stock_id"}), on="stock_id", how="left")
n_unmatched = m["is_financial"].isna().sum()
P(f"SP500 rows with no SIC match: {n_unmatched:,} / {len(m):,} "
  f"({m.loc[m['is_financial'].isna(),'stock_id'].nunique()} tickers)")
xcols = ["dH_gpm_z", "DS_z"]

m_all = m.dropna(subset=["ret_next_month"] + xcols)
res_all, cdf_all = fm_ladder(m_all, "ret_next_month", xcols, "date", lag=0, min_cs=len(xcols) + 2)
P(f"WITH financials   : t(dH)={res_all['dH_gpm_z'][1]:+.3f}  t(dS)={res_all['dS_z'][1] if 'DS_z' not in res_all else res_all['DS_z'][1]:+.3f}  "
  f"N={len(m_all):,}  avg firms/month={len(m_all)/m_all['date'].nunique():.1f}  months={m_all['date'].nunique()}")

m_ex = m[m["is_financial"] != True].dropna(subset=["ret_next_month"] + xcols)
res_ex, cdf_ex = fm_ladder(m_ex, "ret_next_month", xcols, "date", lag=0, min_cs=len(xcols) + 2)
P(f"EX-financials      : t(dH)={res_ex['dH_gpm_z'][1]:+.3f}  t(dS)={res_ex['DS_z'][1]:+.3f}  "
  f"N={len(m_ex):,}  avg firms/month={len(m_ex)/m_ex['date'].nunique():.1f}  months={m_ex['date'].nunique()}")
P(f"Firms excluded: {m_all['stock_id'].nunique() - m_ex['stock_id'].nunique()} of {m_all['stock_id'].nunique()} "
  f"({(1 - m_ex['stock_id'].nunique()/m_all['stock_id'].nunique())*100:.1f}%)")

# ── Full-universe quarterly panel ────────────────────────────────────────────
P("\n" + "-"*78)
P("Full-universe quarterly panel (Model B: delta_h_z + delta_s_z)")
P("-"*78)
q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
q = q.merge(sic[["ticker", "is_financial"]], on="ticker", how="left")
n_unmatched_q = q["is_financial"].isna().sum()
P(f"FullUniv rows with no SIC match: {n_unmatched_q:,} / {len(q):,} "
  f"({q.loc[q['is_financial'].isna(),'ticker'].nunique()} tickers)")
xcols_fu = ["delta_h_z", "delta_s_z"]

q_all = q.dropna(subset=["ret_next"] + xcols_fu)
res_all_q, _ = fm_ladder(q_all, "ret_next", xcols_fu, "q", lag=4, min_cs=20)
P(f"WITH financials   : t(dH)={res_all_q['delta_h_z'][1]:+.3f}  t(dS)={res_all_q['delta_s_z'][1]:+.3f}  "
  f"N={len(q_all):,}  avg firms/quarter={len(q_all)/q_all['q'].nunique():.1f}  quarters={q_all['q'].nunique()}")

q_ex = q[q["is_financial"] != True].dropna(subset=["ret_next"] + xcols_fu)
res_ex_q, _ = fm_ladder(q_ex, "ret_next", xcols_fu, "q", lag=4, min_cs=20)
P(f"EX-financials      : t(dH)={res_ex_q['delta_h_z'][1]:+.3f}  t(dS)={res_ex_q['delta_s_z'][1]:+.3f}  "
  f"N={len(q_ex):,}  avg firms/quarter={len(q_ex)/q_ex['q'].nunique():.1f}  quarters={q_ex['q'].nunique()}")
P(f"Firms excluded: {q_all['ticker'].nunique() - q_ex['ticker'].nunique()} of {q_all['ticker'].nunique()} "
  f"({(1 - q_ex['ticker'].nunique()/q_all['ticker'].nunique())*100:.1f}%)")

P("\nNote: unmatched-SIC rows (ticker not in sharadar_tickers SIC table) are")
P("RETAINED in both 'with' and 'ex' financials rows (only rows explicitly")
P("flagged is_financial==True are dropped in the 'ex' row) -- so the 24.7%")
P("figure in the spec should be checked against the matched-only base rate.")
matched_frac_fin_sp = sic.set_index("ticker").reindex(m_all["stock_id"].unique())["is_financial"].mean()
matched_frac_fin_fu = sic.set_index("ticker").reindex(q_all["ticker"].unique())["is_financial"].mean()
P(f"Financial-firm share of matched SP500 tickers: {matched_frac_fin_sp*100:.1f}%")
P(f"Financial-firm share of matched FullUniv tickers: {matched_frac_fin_fu*100:.1f}%")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
