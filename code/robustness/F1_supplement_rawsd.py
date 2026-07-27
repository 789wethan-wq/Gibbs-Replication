"""F1 supplement — the 4-quarter FF3-residual DeltaS proxy is mathematically
infeasible (0 residual degrees of freedom: 4 parameters -- const + 3 factors
-- on 4 observations cannot produce a residual variance). This computes a
cruder but broader raw quarterly-return-SD proxy (no factor model, usable
down to 2 observations) to cover the very-short-history tail that the 8-
quarter/min-6-obs FF3 proxy's data requirement excludes.
"""
import os
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/F1_supplement_rawsd.txt"

print(f"[pid={os.getpid()}] F1 supplement — fresh process")
log = []
def P(s=""):
    print(s)
    log.append(str(s))

P("="*88)
P("F1 supplement — raw return-SD proxy (broader coverage than the FF3 proxy)")
P("="*88)
P("The 4-quarter FF3-residual proxy is INFEASIBLE, not just unreported: a 4-")
P("parameter regression (const + Mkt_RF + SMB + HML) on exactly 4 observations")
P("has zero residual degrees of freedom, so no residual variance is defined.")
P("This is a mathematical fact, not a data-availability gap -- reported as such")
P("rather than substituting a nearby window that happens to run.")

tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"]
uni = sf1t[
    sf1t["category"].str.contains("Domestic Common", na=False) &
    sf1t["exchange"].isin(["NYSE", "NASDAQ", "NYSEARCA", "BATS", "NYSEMKT"]) &
    (sf1t["currency"] == "USD")
].copy()
uni_set = set(uni["ticker"])

sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                       columns=["ticker", "dimension", "calendardate", "price", "dps"])
arq = sf1[(sf1["dimension"] == "ARQ") & sf1["ticker"].isin(uni_set)].copy()
arq["calendardate"] = pd.to_datetime(arq["calendardate"], errors="coerce")
arq = arq.dropna(subset=["calendardate", "price"])
arq = arq[arq["price"] > 0]
arq["q"] = arq["calendardate"].dt.to_period("Q")
arq = arq.sort_values(["ticker", "calendardate"]).drop_duplicates(["ticker", "q"], keep="last")
arq = arq.sort_values(["ticker", "q"])
arq["q_ord"] = arq["q"].apply(lambda p: p.ordinal)
arq["price_prev"] = arq.groupby("ticker")["price"].shift(1)
arq["gap"] = arq["q_ord"] - arq.groupby("ticker")["q_ord"].shift(1)
ret_px = arq["price"] / arq["price_prev"] - 1.0
div_q = (arq["dps"].fillna(0) / 4.0) / arq["price_prev"]
arq["ret"] = np.where(arq["gap"] == 1, ret_px + div_q, np.nan)
arq_valid_ret = arq.dropna(subset=["ret"]).copy()
arq_valid_ret["ret"] = arq_valid_ret.groupby("q")["ret"].transform(
    lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))

panel_final = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
included_tickers = set(panel_final["ticker"].unique())
excluded_tickers = uni_set - included_tickers

raw_sd = arq_valid_ret.groupby("ticker")["ret"].agg(["std", "count"]).rename(columns={"std": "raw_sd", "count": "n"})
excl_sd = raw_sd[raw_sd.index.isin(excluded_tickers) & (raw_sd["n"] >= 2)]["raw_sd"]
incl_sd = raw_sd[raw_sd.index.isin(included_tickers) & (raw_sd["n"] >= 2)]["raw_sd"]

P(f"\nRaw quarterly-return SD (no factor model, min 2 obs):")
P(f"  Excluded tickers: N={len(excl_sd):,}  mean={excl_sd.mean():.4f}  " +
  "  ".join(f"p{p}={excl_sd.quantile(p/100):.4f}" for p in [10, 25, 50, 75, 90]))
P(f"  Included tickers: N={len(incl_sd):,}  mean={incl_sd.mean():.4f}  " +
  "  ".join(f"p{p}={incl_sd.quantile(p/100):.4f}" for p in [10, 25, 50, 75, 90]))
diff = excl_sd.mean() - incl_sd.mean()
P(f"  Excluded-minus-included mean difference: {diff:+.4f} ({diff/incl_sd.mean()*100:+.1f}% relative)")

# same, but only the very-shortest-lived (<=4 quarters) excluded firms --
# the tail the 8Q/min-6 FF3 proxy could not reach at all
lifespan = arq_valid_ret.groupby("ticker")["ret"].transform("size")
very_short = set(arq_valid_ret[(arq_valid_ret["ticker"].isin(excluded_tickers)) &
                                (arq_valid_ret.groupby("ticker")["ret"].transform("size") <= 4)]["ticker"].unique())
vs_sd = raw_sd[raw_sd.index.isin(very_short) & (raw_sd["n"] >= 2)]["raw_sd"]
P(f"\n  Very-short-lived excluded firms only (<=4 valid quarters, N={len(vs_sd):,}): "
  f"mean raw SD={vs_sd.mean():.4f}")
diff_vs = vs_sd.mean() - incl_sd.mean()
P(f"  vs included mean: {diff_vs:+.4f} ({diff_vs/incl_sd.mean()*100:+.1f}% relative)")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
