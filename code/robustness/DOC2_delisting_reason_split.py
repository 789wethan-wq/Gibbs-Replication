"""Documentation query 2 — of the 8,937 delistings (72% of the 12,449-firm
analysis panel), what fraction are performance-related (failure/bankruptcy)
vs. acquisition/other, as far as SF1 permits?

SF1's ticker metadata does NOT carry an explicit delisting-reason code (no
'delisting_reason' field). Two proxies are available and reported together,
each with its own caveat:
  (a) relatedtickers populated -> Sharadar links the delisted ticker to a
      successor symbol (new ticker after M&A, symbol change, spinoff, or
      re-listing). This is Sharadar's own linkage, not a reason code, but is
      the closest thing to a positive M&A/reorg signal in the data.
  (b) terminal-quarter return proxy: performance-related delistings should
      cluster at large negative terminal returns (approaching -100%,
      i.e. equity wiped out); M&A delistings typically end on a flat or
      positive terminal return (acquisition premium) or a small move.
Neither proxy is a ground-truth delisting-reason code; both are reported so
the reader can see the disagreement between them, not just one number.
"""
import os
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/DOC2_delisting_reason_split.txt"

log = []
def P(s=""):
    print(s)
    log.append(str(s))

panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
analysis_tickers = set(panel["ticker"].unique())
tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"].copy()
delisted = sf1t[(sf1t["isdelisted"] == "Y") & sf1t["ticker"].isin(analysis_tickers)].copy()

P("="*78)
P("DOC query 2 — delisting reason split, analysis-panel delistings")
P("="*78)
P(f"Analysis-panel tickers: {len(analysis_tickers):,}")
P(f"Delisted (isdelisted=Y) within analysis panel: {len(delisted):,}")

# ── proxy (a): relatedtickers populated ────────────────────────────────────
delisted["has_related"] = delisted["relatedtickers"].notna() & (delisted["relatedtickers"].str.strip() != "")
P(f"\nProxy (a) — relatedtickers populated (successor-symbol link):")
P(f"  {delisted['has_related'].sum():,} of {len(delisted):,} ({delisted['has_related'].mean()*100:.1f}%)")

# ── proxy (b): terminal-quarter return ─────────────────────────────────────
P("\nProxy (b) — terminal-quarter return (last reported ARQ price vs prior)...")
sf1p = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                        columns=["ticker", "dimension", "calendardate", "price"])
arq = sf1p[(sf1p["dimension"] == "ARQ") & sf1p["ticker"].isin(set(delisted["ticker"]))].copy()
arq["calendardate"] = pd.to_datetime(arq["calendardate"], errors="coerce")
arq = arq.dropna(subset=["calendardate", "price"])
arq = arq[arq["price"] > 0].sort_values(["ticker", "calendardate"])
arq["price_prev"] = arq.groupby("ticker")["price"].shift(1)
arq["term_ret"] = arq["price"] / arq["price_prev"] - 1.0
last_obs = arq.sort_values(["ticker", "calendardate"]).groupby("ticker").tail(1)
last_obs = last_obs.dropna(subset=["term_ret"])
P(f"  Terminal-return computable for {len(last_obs):,} of {len(delisted):,} delisted tickers")
P(f"  Terminal-return distribution: "
  f"p10={last_obs['term_ret'].quantile(.10):+.2%}  p25={last_obs['term_ret'].quantile(.25):+.2%}  "
  f"median={last_obs['term_ret'].median():+.2%}  p75={last_obs['term_ret'].quantile(.75):+.2%}  "
  f"p90={last_obs['term_ret'].quantile(.90):+.2%}")
failure_thr = -0.50
n_failure_like = (last_obs["term_ret"] <= failure_thr).sum()
P(f"  Terminal return <= {failure_thr:.0%} (failure-like): {n_failure_like:,} "
  f"({n_failure_like/len(last_obs)*100:.1f}% of tickers with computable terminal return)")
n_flat_or_up = (last_obs["term_ret"] > -0.10).sum()
P(f"  Terminal return > -10% (flat/positive, M&A/reorg-like): {n_flat_or_up:,} "
  f"({n_flat_or_up/len(last_obs)*100:.1f}%)")
n_middle = len(last_obs) - n_failure_like - n_flat_or_up
P(f"  Terminal return in (-50%, -10%] (ambiguous): {n_middle:,} "
  f"({n_middle/len(last_obs)*100:.1f}%)")

# ── cross-tab of the two proxies ───────────────────────────────────────────
merged = delisted[["ticker", "has_related"]].merge(
    last_obs[["ticker", "term_ret"]], on="ticker", how="left")
merged["failure_like"] = merged["term_ret"] <= failure_thr
merged["ma_like"] = merged["term_ret"] > -0.10
P("\nCross-tab (proxy a vs proxy b, tickers with both proxies available):")
both = merged.dropna(subset=["term_ret"])
P(f"  relatedtickers=Y & failure_like=Y (disagreement): {(both['has_related'] & both['failure_like']).sum():,}")
P(f"  relatedtickers=Y & ma_like=Y (agreement):          {(both['has_related'] & both['ma_like']).sum():,}")
P(f"  relatedtickers=N & failure_like=Y (agreement):      {(~both['has_related'] & both['failure_like']).sum():,}")
P(f"  relatedtickers=N & ma_like=Y (ambiguous, no linkage but non-negative exit): "
  f"{(~both['has_related'] & both['ma_like']).sum():,}")

P("\nCaveat (unchanged from the manuscript's existing footnote): the 0.482%/month")
P("delisting-return proxy used elsewhere in the paper already carries an M&A")
P("caveat; this 72% figure describes ALL delistings (any reason) and should not")
P("be read as 72% performance failures. Best-available point estimate from proxy")
P(f"(b): roughly {n_failure_like/len(last_obs)*100:.0f}% of delistings look failure-like by terminal return, "
  f"roughly {n_flat_or_up/len(last_obs)*100:.0f}% look M&A/reorg-like, with neither proxy authoritative.")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
