"""D5_financials_table.py — fresh re-verification of the financials-exclusion
table (R27's logic), with design-matrix hashes and date ranges added per the
V34 ground rules.
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
OUT = "../results/revision/D5_financials_table.txt"

print(f"[pid={os.getpid()}] D5 — fresh process")
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


def fm_ladder(panel, ycol, xcols, datecol, min_cs):
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
        mean_, t_, p_ = newey_west_mean_tstat(cdf[c].values, lags=4 if datecol == "q" else 0)
        out[c] = dict(coef=mean_, t=t_, n_quarters=len(cdf))
    return out, cdf


tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
P("="*88)
P("D5 — Financials exclusion (SIC 6000-6999)")
P("="*88)
P(f"SIC field used: sharadar_tickers.parquet column 'siccode' (SF1 rows), cross-checked "
  f"against 'sicsector' text label.")
sic = tk[["ticker", "siccode", "sicsector"]].dropna(subset=["siccode"]).drop_duplicates("ticker")
sic["siccode"] = pd.to_numeric(sic["siccode"], errors="coerce")
sic["is_financial"] = sic["siccode"].between(6000, 6999)
P(f"Cross-check: {(sic.loc[sic['is_financial'],'sicsector']=='Finance Insurance And Real Estate').mean()*100:.1f}% "
  f"of siccode-flagged financials also carry sicsector='Finance Insurance And Real Estate' (should be ~100%).")

# ── corrected panel (FU quarterly) ──────────────────────────────────────────
q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
q = q.merge(sic[["ticker", "is_financial"]], on="ticker", how="left")
xcols_fu = ["delta_h_z", "delta_s_z"]
q_all = q.dropna(subset=["ret_next"] + xcols_fu)
q_ex = q[q["is_financial"] != True].dropna(subset=["ret_next"] + xcols_fu)

fin_share_fu = sic.set_index("ticker").reindex(q_all["ticker"].unique())["is_financial"].mean()

for label, d in [("Corrected(R18) WITH financials", q_all), ("Corrected(R18) EX financials", q_ex)]:
    fm, cdf = fm_ladder(d, "ret_next", xcols_fu, "q", min_cs=20)
    X = sm.add_constant(d[xcols_fu]).values
    dh = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()[:12]
    P(f"\n[{label}] N={len(d):,} tickers={d['ticker'].nunique():,} avg firms/qtr={len(d)/d['q'].nunique():.1f} "
      f"date_range={d['q'].min()}..{d['q'].max()} design_shape={X.shape} hash={dh}")
    P(f"  t(dH)={fm['delta_h_z']['t']:+.3f}  t(dS)={fm['delta_s_z']['t']:+.3f}  "
      f"quarters={fm['delta_h_z']['n_quarters']}")

# ── monthly (SP500) panel ───────────────────────────────────────────────────
m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
m["date"] = pd.to_datetime(m["date"])
m["dH_gpm_z"] = cs_wz(m, "dH_gpm", "date")
m = m.merge(sic[["ticker", "is_financial"]].rename(columns={"ticker": "stock_id"}), on="stock_id", how="left")
xcols_sp = ["dH_gpm_z", "DS_z"]
m_all = m.dropna(subset=["ret_next_month"] + xcols_sp)
m_ex = m[m["is_financial"] != True].dropna(subset=["ret_next_month"] + xcols_sp)

fin_share_sp = sic.set_index("ticker").reindex(m_all["stock_id"].unique())["is_financial"].mean()

for label, d in [("Monthly(SP500) WITH financials", m_all), ("Monthly(SP500) EX financials", m_ex)]:
    fm, cdf = fm_ladder(d, "ret_next_month", xcols_sp, "date", min_cs=len(xcols_sp) + 2)
    X = sm.add_constant(d[xcols_sp]).values
    dh = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()[:12]
    P(f"\n[{label}] N={len(d):,} tickers={d['stock_id'].nunique():,} avg firms/month={len(d)/d['date'].nunique():.1f} "
      f"date_range={d['date'].min().date()}..{d['date'].max().date()} design_shape={X.shape} hash={dh}")
    P(f"  t(dH)={fm['dH_gpm_z']['t']:+.3f}  t(dS)={fm['DS_z']['t']:+.3f}  quarters/months={fm['dH_gpm_z']['n_quarters']}")

P("\n" + "="*88)
P("D5 FINAL TABLE")
P("="*88)
fm_fu_w, _ = fm_ladder(q_all, "ret_next", xcols_fu, "q", 20)
fm_fu_x, _ = fm_ladder(q_ex, "ret_next", xcols_fu, "q", 20)
fm_sp_w, _ = fm_ladder(m_all, "ret_next_month", xcols_sp, "date", 4)
fm_sp_x, _ = fm_ladder(m_ex, "ret_next_month", xcols_sp, "date", 4)
P(f"{'Panel':22}{'t(dH) incl.':>12}{'t(dH) excl.':>12}{'t(dS) incl.':>12}{'t(dS) excl.':>12}{'N excl.':>10}")
P(f"{'Corrected (R18)':22}{fm_fu_w['delta_h_z']['t']:>+12.2f}{fm_fu_x['delta_h_z']['t']:>+12.2f}"
  f"{fm_fu_w['delta_s_z']['t']:>+12.2f}{fm_fu_x['delta_s_z']['t']:>+12.2f}{len(q_ex):>10,}")
P(f"{'Monthly (S&P 500)':22}{fm_sp_w['dH_gpm_z']['t']:>+12.2f}{fm_sp_x['dH_gpm_z']['t']:>+12.2f}"
  f"{fm_sp_w['DS_z']['t']:>+12.2f}{fm_sp_x['DS_z']['t']:>+12.2f}{len(m_ex):>10,}")

P(f"\nFinancial-firm share of matched tickers, FullUniv: {fin_share_fu*100:.1f}% (manuscript §3.1: 22.2%)")
P(f"Financial-firm share of matched tickers, SP500:     {fin_share_sp*100:.1f}% (manuscript §3.1: 19.3%)")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
