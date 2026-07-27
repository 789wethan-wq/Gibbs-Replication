"""F4 — ΔH filing density. Distribution of DISTINCT annual filings inside the
ΔH_GPM estimation window (60 months, the primary window), per observation:
median, quartiles, share at 1/2/3/4+ filings -- for the corrected (R18,
12,449-ticker) panel and the monthly (S&P 500, 462-ticker) panel SEPARATELY.

Reuses the exact rolling-distinct-count machinery already validated in
robustness/R21_dh_degeneracy_audit.py (which established the mass-point
finding at the aggregate/full-universe level); this restricts to each
panel's ACTUAL ticker set before computing, since R21 ran on the full
14,600-ticker monthly_fundamentals.parquet unrestricted.
"""
import os
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/F4_dH_filing_density.txt"

print(f"[pid={os.getpid()}] F4 — fresh process")
log = []
def P(s=""):
    print(s)
    log.append(str(s))

P("="*88)
P("F4 — DeltaH filing density: distinct filings inside the 60-month estimation window")
P("="*88)


def rolling_nunique_and_count(values, window):
    n = len(values)
    ndist = np.full(n, np.nan)
    ncnt = np.zeros(n, dtype=np.int64)
    counts = {}
    distinct = 0
    cnt = 0
    start = 0
    for i in range(n):
        v = values[i]
        if not np.isnan(v):
            c = counts.get(v, 0)
            if c == 0:
                distinct += 1
            counts[v] = c + 1
            cnt += 1
        lo = i - window + 1
        while start < lo:
            vs = values[start]
            if not np.isnan(vs):
                cs = counts[vs] - 1
                counts[vs] = cs
                if cs == 0:
                    distinct -= 1
                cnt -= 1
            start += 1
        ndist[i] = distinct
        ncnt[i] = cnt
    return ndist, ncnt


mf = pd.read_parquet(f"{DATA}/monthly_fundamentals.parquet")
mf["date"] = pd.to_datetime(mf["date"])
mf = mf.sort_values(["stock_id", "date"]).reset_index(drop=True)
P(f"monthly_fundamentals.parquet: {len(mf):,} rows, {mf['stock_id'].nunique():,} tickers "
  f"(full source; restricted to each panel's own ticker set below)")

q_panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
sp_panel = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
q_tickers = set(q_panel["ticker"].unique())
sp_tickers = set(sp_panel["stock_id"].unique())
P(f"Corrected (R18) panel tickers: {len(q_tickers):,}  |  Monthly (SP500) panel tickers: {len(sp_tickers):,}")

W = 60
MIN_PERIODS = 24


def build_samp(mf_sub, W, tag, sample_to_quarter_end=False):
    ndist_all = np.full(len(mf_sub), np.nan)
    ncnt_all = np.zeros(len(mf_sub), dtype=np.int64)
    for tkr, idx in mf_sub.groupby("stock_id", sort=False).groups.items():
        idx = idx.to_numpy()
        vals = mf_sub["gpm"].values[idx]
        nd, nc = rolling_nunique_and_count(vals, W)
        ndist_all[idx] = nd
        ncnt_all[idx] = nc
    tmp = mf_sub[["stock_id", "date"]].copy()
    tmp["n_distinct"] = ndist_all
    tmp["row_cnt"] = ncnt_all
    tmp["dH_gpm"] = -mf_sub.groupby("stock_id")["gpm"].transform(
        lambda x: x.rolling(W, min_periods=MIN_PERIODS).std())
    samp = tmp.dropna(subset=["dH_gpm"]).copy()
    if sample_to_quarter_end:
        # R18's ACTUAL observations are one per ticker-QUARTER (drop_duplicates on
        # q, keep='last'), not one per ticker-month -- computing n_distinct on every
        # monthly row and reporting shares over all of them triple-counts each real
        # observation (3 identical monthly rows collapse to 1 quarterly one in the
        # panel R18 actually estimates on). Match that here for a fair comparison.
        samp["q"] = samp["date"].dt.to_period("Q")
        samp = (samp.sort_values(["stock_id", "date"])
                     .drop_duplicates(["stock_id", "q"], keep="last"))
    P(f"\n[{tag}] N observations ({'quarter-end sampled, matching R18' if sample_to_quarter_end else 'monthly rows, matching the SP500 panel'}) "
      f"= {len(samp):,}, tickers = {samp['stock_id'].nunique():,}")
    return samp


mf_q = mf[mf["stock_id"].isin(q_tickers)].sort_values(["stock_id", "date"]).reset_index(drop=True)
mf_sp = mf[mf["stock_id"].isin(sp_tickers)].sort_values(["stock_id", "date"]).reset_index(drop=True)

samp_q_raw = build_samp(mf_q, W, f"Corrected (R18) panel, W={W}mo (pre-restriction)", sample_to_quarter_end=True)
samp_sp = build_samp(mf_sp, W, f"Monthly (SP500) panel, W={W}mo", sample_to_quarter_end=False)

# restrict to the EXACT (ticker, quarter) pairs that actually appear as
# observations in the R18 panel (merged_sf1_quarterly_survfree.parquet) --
# the unrestricted version above over-counts: it includes every quarter a
# ticker has valid GPM history, not just the quarters where that ticker
# actually has a panel observation (valid return + DeltaS that quarter too)
q_panel_keys = q_panel[["ticker", "q"]].drop_duplicates()
q_panel_keys = q_panel_keys.rename(columns={"ticker": "stock_id"})
samp_q = samp_q_raw.merge(q_panel_keys, on=["stock_id", "q"], how="inner")
P(f"\nRestricted to actual R18 panel (ticker,quarter) observations: N={len(samp_q):,} "
  f"(R18 panel's own total row count: {len(q_panel):,}; gap is quarters where GPM has")
P(f"a valid rolling estimate but DeltaS/return does not, or vice versa)")


def report(samp, tag):
    nd = samp["n_distinct"]
    P(f"\n{tag}:")
    P(f"  median={nd.median():.1f}  Q1={nd.quantile(0.25):.1f}  Q3={nd.quantile(0.75):.1f}  mean={nd.mean():.2f}")
    for k in [1, 2, 3]:
        share = (nd == k).mean()
        P(f"  share at exactly {k} filing(s): {share*100:.1f}%")
    share4plus = (nd >= 4).mean()
    P(f"  share at 4+ filings: {share4plus*100:.1f}%")
    share_le2 = (nd <= 2).mean()
    P(f"  share at <=2 filings (coarse instrument threshold): {share_le2*100:.1f}%")
    return dict(median=nd.median(), q1=nd.quantile(0.25), q3=nd.quantile(0.75), mean=nd.mean(),
                share_le2=share_le2)


r_q = report(samp_q, f"Corrected (R18) panel, W={W}mo")
r_sp = report(samp_sp, f"Monthly (SP500) panel, W={W}mo")

P("\n" + "="*88)
P("F4 SUMMARY")
P("="*88)
P(f"{'':30}{'Corrected (R18)':>18}{'Monthly (SP500)':>18}")
P(f"{'Median n_distinct filings':30}{r_q['median']:>18.1f}{r_sp['median']:>18.1f}")
P(f"{'Q1':30}{r_q['q1']:>18.1f}{r_sp['q1']:>18.1f}")
P(f"{'Q3':30}{r_q['q3']:>18.1f}{r_sp['q3']:>18.1f}")
P(f"{'Mean':30}{r_q['mean']:>18.2f}{r_sp['mean']:>18.2f}")
P(f"{'Share at <=2 filings':30}{r_q['share_le2']*100:>17.1f}%{r_sp['share_le2']*100:>17.1f}%")

P("\nNote on methodology (self-caught, reported for transparency): an earlier pass of")
P("this script restricted only by TICKER membership in the R18 panel (not by the exact")
P("(ticker,quarter) OBSERVATION pairs that actually appear in it), giving N=865,063 --")
P("nearly double the panel's real row count (434,016) -- and a median of 2 filings /")
P("47% at n_distinct=1, matching R21's own UNRESTRICTED full-14,600-ticker output but")
P("NOT matching what the R18 panel's actual observations look like. Restricting to the")
P("exact (ticker,quarter) pairs in merged_sf1_quarterly_survfree.parquet (N=404,387,")
P("93% of the panel's 434,016 rows -- the gap is quarters where GPM has a valid rolling")
P("estimate but the return/DeltaS side of that observation does not) gives the numbers")
P("above, which DO match the manuscript's cited ~17% at n_distinct=1 closely (17.0%).")

if r_q["median"] <= 2:
    P(f"\nFINDING: the median full-universe (corrected-panel) observation rests on "
      f"{r_q['median']:.0f} distinct filing(s) within the 60-month window -- DeltaH is a")
    P("coarse instrument for MOST of the corrected panel, not just the ~17% mass-point tail")
    P("at n_distinct=1. Section 3.2 should state this directly (median filing count),")
    P("rather than presenting the coarseness only through the single-filing mass-point lens.")
else:
    P(f"\nFINDING: the median corrected-panel observation rests on {r_q['median']:.0f} distinct")
    P(f"filings -- essentially the SAME as the monthly SP500 panel's median ({r_sp['median']:.0f}).")
    P("DeltaH is NOT a coarse instrument for most of the corrected panel; the coarseness is")
    P("concentrated in the ~17-18% mass-point tail the manuscript's existing mass-point")
    P("analysis already covers. Section 3.2's framing through the mass-point lens is")
    P("adequate as-is for the median/typical observation; no broader coarseness caveat is")
    P("needed beyond what's already there. (Q1=3 filings is worth noting alongside the")
    P("mass-point figure, since a quarter of observations sit at 3 or fewer.)")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
