"""E6 — Filing survivorship vs listing survivorship.

R18 computes returns between consecutive FILING dates; a firm that stops
filing before it delists drops out of the panel at its LAST FILING, not at
its actual delisting date. This checks how large that gap is, using SF1's
own ticker metadata: lastpricedate (proxy for the delisting/last-trading
date) vs lastquarter (the last fiscal quarter with an SF1 ARQ filing).
"""
import os
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/E6_filing_vs_listing_survivorship.txt"

print(f"[pid={os.getpid()}] E6 — fresh process")
log = []
def P(s=""):
    print(s)
    log.append(str(s))

P("="*88)
P("E6 — Filing survivorship vs listing survivorship")
P("="*88)

panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"][["ticker", "isdelisted", "lastpricedate", "lastquarter"]].drop_duplicates("ticker")

analysis_tickers = set(panel["ticker"].unique())
delisted = sf1t[(sf1t["isdelisted"] == "Y") & sf1t["ticker"].isin(analysis_tickers)].copy()
P(f"Analysis-panel tickers: {len(analysis_tickers):,}")
P(f"Delisted (isdelisted=Y) within analysis panel: {len(delisted):,}  "
  f"({'MATCHES' if len(delisted)==8937 else 'DOES NOT MATCH'} the cited 8,937)")

delisted["lastpricedate"] = pd.to_datetime(delisted["lastpricedate"], errors="coerce")
delisted["lastquarter"] = pd.to_datetime(delisted["lastquarter"], errors="coerce")
delisted = delisted.dropna(subset=["lastpricedate", "lastquarter"])
P(f"With both lastpricedate and lastquarter available: {len(delisted):,}")

delisted["delist_q"] = delisted["lastpricedate"].dt.to_period("Q")
delisted["last_filing_q"] = delisted["lastquarter"].dt.to_period("Q")
delisted["gap_quarters"] = (delisted["delist_q"] - delisted["last_filing_q"]).apply(lambda x: x.n)
delisted = delisted[delisted["gap_quarters"] >= 0]
P(f"With non-negative gap (delisting on/after last filing): {len(delisted):,}")

P("\n" + "-"*88)
P("(1) Distribution of gap between last filing and delisting, in quarters")
P("-"*88)
gq = delisted["gap_quarters"]
P(f"  N={len(gq):,}")
P(f"  mean={gq.mean():.2f}  median={gq.median():.1f}  std={gq.std():.2f}")
for p in [10, 25, 50, 75, 90, 95, 99]:
    P(f"  p{p:>2} = {gq.quantile(p/100):.1f} quarters")
P(f"  distribution of gap (quarters): {gq.value_counts().sort_index().head(15).to_dict()}")

P("\n" + "-"*88)
P("(2) Fraction stopping filing more than 2 quarters before delisting")
P("-"*88)
frac_gt2 = (gq > 2).mean()
P(f"  gap > 2 quarters: {frac_gt2*100:.1f}% ({(gq>2).sum():,} of {len(gq):,})")
P(f"  gap = 0 quarters (filed through the delisting quarter): {(gq==0).mean()*100:.1f}%")
P(f"  gap in (0,2] quarters: {((gq>0)&(gq<=2)).mean()*100:.1f}%")

P("\n" + "-"*88)
P("(3) Broken out by DeltaS quintile (firm's LAST observed delta_s_z in the panel)")
P("-"*88)
last_ds = panel.sort_values(["ticker", "q_ord"]).groupby("ticker")["delta_s_z"].last().rename("last_ds_z")
delisted_ds = delisted.merge(last_ds, on="ticker", how="left").dropna(subset=["last_ds_z"])
P(f"Delisted firms with a computable last-observed DeltaS: {len(delisted_ds):,}")
delisted_ds["ds_quintile"] = pd.qcut(delisted_ds["last_ds_z"], 5, labels=False, duplicates="drop") + 1

P(f"\n{'DS quintile':14}{'N firms':>10}{'mean gap(q)':>14}{'median gap':>12}{'%gap>2q':>10}")
qrows = []
for qd in sorted(delisted_ds["ds_quintile"].dropna().unique()):
    g = delisted_ds[delisted_ds["ds_quintile"] == qd]["gap_quarters"]
    P(f"{int(qd):<14}{len(g):>10,}{g.mean():>14.2f}{g.median():>12.1f}{(g>2).mean()*100:>9.1f}%")
    qrows.append(dict(q=int(qd), n=len(g), mean_gap=g.mean(), median_gap=g.median(), frac_gt2=(g>2).mean()))

q1_gap = qrows[0]["mean_gap"]
q5_gap = qrows[-1]["mean_gap"]
P(f"\nQ1 (lowest DeltaS) mean gap = {q1_gap:.2f} quarters")
P(f"Q5 (highest DeltaS) mean gap = {q5_gap:.2f} quarters")
diff = q5_gap - q1_gap
P(f"Q5 - Q1 = {diff:+.2f} quarters")

if diff > 0.5:
    P("\nFINDING: the gap between last filing and actual delisting IS systematically")
    P("LONGER for high-DeltaS firms. R18's filing-date-spaced construction therefore")
    P("still conditions on survival in the direction the paper is correcting for --")
    P("high-disorder firms that stop filing early are removed from the panel BEFORE")
    P("their true delisting date, understating how long the panel 'sees' distress in")
    P("exactly the firms whose absence would matter most for the entropy premium.")
    P("Section 3.1 should state this explicitly. The direction is conservative for the")
    P("headline collapse (it would, if anything, make the corrected panel's ΔS premium")
    P("MORE attenuated toward the survivorship-biased result, not less -- so it does not")
    P("undermine the '+4.70 -> +0.02' collapse itself) but the manuscript should not")
    P("describe R18 as a full listing-survivorship correction without this caveat.")
elif diff < -0.5:
    P("\nFINDING: the gap is systematically SHORTER for high-DeltaS firms (opposite of")
    P("the hypothesized direction) -- high-disorder firms file closer to their actual")
    P("delisting date than low-disorder firms. This does not support the concern that")
    P("R18 leaves filing-survivorship conditioning in the direction that would inflate")
    P("the corrected panel's premium; if anything the bias described in the spec runs")
    P("the other way.")
else:
    P("\nFINDING: gap is not meaningfully different across DeltaS quintiles (|Q5-Q1|<0.5")
    P("quarters) -- filing-vs-listing survivorship does not appear to be conditioned on")
    P("DeltaS in this panel.")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
