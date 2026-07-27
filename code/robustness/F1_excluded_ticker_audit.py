"""F1 — Excluded-ticker audit.

R18's construction (robustness/R18_sf1_quarterly_survfree.py) starts from a
15,522-ticker SF1 domestic-common universe and the analysis panel
(merged_sf1_quarterly_survfree.parquet) ends with 12,449. This traces the
EXACT waterfall, stage by stage, reproducing R18's own filters verbatim, to
identify which ~3,073 tickers are excluded and why.

R18's filter stages (traced in order):
  (0) universe: SF1 domestic-common, NYSE/NASDAQ/NYSEARCA/BATS/NYSEMKT, USD
  (1) has at least one valid ARQ price/return record at all
  (2) >=8 valid CONSECUTIVE-GAP quarterly return observations (R18's
      `cnt = arq.groupby("ticker")["ret"].transform("size"); arq[cnt>=8]`)
  (3) produces at least one valid DeltaS estimate (R18's ivol_ticker rolling
      12q/min_obs=8 FF3-residual-std function -- can fail even with >=8 total
      valid returns if they are not arranged in a long-enough consecutive run)
  (4) has a T-quarter match (inner join on Tq -- essentially non-binding,
      checked explicitly)
Separately (does NOT gate entry to `panel`, only to the FULL-CHANNEL FM
sample `pf` used in Model B/C): GPM/DeltaH availability (left join in R18).
Both interpretations are reported since the spec's "no GPM" category could
mean either.
"""
import os
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/F1_excluded_ticker_audit.txt"

print(f"[pid={os.getpid()}] F1 — fresh process")
log = []
def P(s=""):
    print(s)
    log.append(str(s))

P("="*88)
P("F1 — Excluded-ticker audit (R18's 15,522 -> 12,449 waterfall)")
P("="*88)

# ── STAGE 0: universe ────────────────────────────────────────────────────────
tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"]
uni = sf1t[
    sf1t["category"].str.contains("Domestic Common", na=False) &
    sf1t["exchange"].isin(["NYSE", "NASDAQ", "NYSEARCA", "BATS", "NYSEMKT"]) &
    (sf1t["currency"] == "USD")
].copy()
uni_set = set(uni["ticker"])
delisted_set = set(uni.loc[uni["isdelisted"] == "Y", "ticker"])
P(f"Stage 0 -- universe: {len(uni_set):,} tickers ({len(delisted_set):,} delisted)")

# ── STAGE 1-2: returns ───────────────────────────────────────────────────────
sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                       columns=["ticker", "dimension", "calendardate", "datekey", "price", "dps"])
arq = sf1[(sf1["dimension"] == "ARQ") & sf1["ticker"].isin(uni_set)].copy()
arq["calendardate"] = pd.to_datetime(arq["calendardate"], errors="coerce")
arq_has_pricedate = arq.dropna(subset=["calendardate", "price"])
arq_has_pricedate = arq_has_pricedate[arq_has_pricedate["price"] > 0]
tickers_with_any_price = set(arq_has_pricedate["ticker"].unique())
P(f"Stage 1 -- tickers with >=1 valid ARQ price record: {len(tickers_with_any_price):,} "
  f"(no price record at all: {len(uni_set)-len(tickers_with_any_price):,})")

arq = arq_has_pricedate.copy()
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

cnt = arq_valid_ret.groupby("ticker")["ret"].transform("size")
arq_8plus = arq_valid_ret[cnt >= 8].copy()
tickers_8plus_returns = set(arq_8plus["ticker"].unique())
tickers_lt8_returns = tickers_with_any_price - tickers_8plus_returns
P(f"Stage 2 -- tickers with >=8 valid consecutive-gap quarterly returns: "
  f"{len(tickers_8plus_returns):,}  (excluded here: {len(tickers_lt8_returns):,})")

# ── STAGE 3: DeltaS (ivol_ticker rolling 12q, min_obs=8) ────────────────────
fac = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
fac.index = pd.to_datetime(fac.index)
facq = fac.copy()
facq["q"] = facq.index.to_period("Q")
def cmpd(s): return (1 + s).prod() - 1
ff = facq.groupby("q").agg({"Mkt_RF": cmpd, "SMB": cmpd, "HML": cmpd, "RF": cmpd}).reset_index()
px = arq_8plus[["ticker", "q", "ret"]].merge(ff, on="q", how="left")
px["exret"] = px["ret"] - px["RF"]

def ivol_ticker(g, window=12, min_obs=8):
    g = g.sort_values("q")
    out = pd.Series(np.nan, index=g.index)
    Xall = g[["Mkt_RF", "SMB", "HML"]].values
    yall = g["exret"].values
    n = len(g)
    for i in range(min_obs, n + 1):
        lo = max(0, i - window)
        Xs = Xall[lo:i]; ys = yall[lo:i]
        if len(ys) < min_obs or np.isnan(Xs).any() or np.isnan(ys).any():
            continue
        Xc = np.column_stack([np.ones(len(Xs)), Xs])
        beta, *_ = np.linalg.lstsq(Xc, ys, rcond=None)
        resid = ys - Xc @ beta
        out.iloc[i - 1] = resid.std(ddof=1)
    return out

print("Computing DeltaS for the FULL 8+-return universe (this is the expensive step)...")
px["delta_s"] = px.groupby("ticker", group_keys=False).apply(lambda g: ivol_ticker(g))
ds = px[["ticker", "q", "delta_s"]].dropna()
tickers_with_ds = set(ds["ticker"].unique())
tickers_8plus_no_ds = tickers_8plus_returns - tickers_with_ds
P(f"Stage 3 -- tickers with >=1 valid DeltaS estimate: {len(tickers_with_ds):,}  "
  f"(had >=8 returns but NO valid DeltaS -- non-consecutive run -- excluded here: "
  f"{len(tickers_8plus_no_ds):,})")

# ── STAGE 4: T-quarter overlap ───────────────────────────────────────────────
v = pd.read_parquet(f"{DATA}/variables_monthly.parquet")
v["date"] = pd.to_datetime(v["date"])
Tm = v.groupby("date")["T"].first().to_frame("T")
Tm["q"] = Tm.index.to_period("Q")
Tq = Tm.groupby("q")["T"].last().reset_index()
ds_t = ds.merge(Tq, on="q", how="inner")
tickers_final = set(ds_t["ticker"].unique())
tickers_ds_no_T = tickers_with_ds - tickers_final
P(f"Stage 4 -- tickers surviving the T-quarter join: {len(tickers_final):,}  "
  f"(lost to T-window mismatch: {len(tickers_ds_no_T):,})")

P(f"\nFINAL PANEL TICKER COUNT (this audit's reconstruction): {len(tickers_final):,}  "
  f"vs the cited 12,449 -- {'MATCHES' if abs(len(tickers_final)-12449)<5 else 'DOES NOT closely match'}")

excluded = uni_set - tickers_final
P(f"\nTotal excluded from universe: {len(excluded):,}  (spec cites ~3,073)")

# ── (1) breakdown by exclusion reason ────────────────────────────────────────
P("\n" + "="*88)
P("(1) Exclusion-reason breakdown")
P("="*88)
no_price = uni_set - tickers_with_any_price
lt8_ret = tickers_lt8_returns
no_ds = tickers_8plus_no_ds
no_T = tickers_ds_no_T
P(f"  No valid ARQ price record at all:                  {len(no_price):,}")
P(f"  Some price data, but <8 valid consecutive returns:  {len(lt8_ret):,}")
P(f"  >=8 returns, but no valid DeltaS (non-consecutive):  {len(no_ds):,}")
P(f"  Valid DeltaS, but no T-quarter overlap:              {len(no_T):,}")
check_sum = len(no_price) + len(lt8_ret) + len(no_ds) + len(no_T)
P(f"  SUM of the above 4 categories: {check_sum:,}  (should equal total excluded {len(excluded):,}: "
  f"{'MATCHES' if check_sum == len(excluded) else 'MISMATCH -- check overlap/logic'})")

# GPM / DeltaH check -- does NOT gate `panel` (left join), only the
# full-channel FM sample. Reported separately since the spec's "no GPM"
# category is ambiguous about which sample it means.
mf = pd.read_parquet(f"{DATA}/monthly_fundamentals.parquet")
mf["date"] = pd.to_datetime(mf["date"])
mf = mf.sort_values(["stock_id", "date"])
mf["dH_gpm"] = -mf.groupby("stock_id")["gpm"].transform(lambda x: x.rolling(60, min_periods=24).std())
tickers_with_gpm = set(mf.dropna(subset=["dH_gpm"])["stock_id"].unique())
final_no_gpm = tickers_final - tickers_with_gpm
P(f"\n  [separate, does NOT gate panel entry -- R18 left-joins GPM]")
P(f"  Of the {len(tickers_final):,} tickers THAT DO enter the panel, "
  f"{len(final_no_gpm):,} ({len(final_no_gpm)/len(tickers_final)*100:.1f}%) never have a valid GPM/DeltaH")
P(f"  observation and are therefore excluded from the FULL-CHANNEL (Model B/C) FM sample")
P(f"  specifically, while still counting toward the 12,449 entropy-only panel tickers.")

# ── (2) of insufficient-history exclusions: delisted count + lifespan ──────
P("\n" + "="*88)
P("(2) Insufficient-history exclusions (no-price + <8-return groups): delisted share, lifespan")
P("="*88)
insuff_hist = no_price | lt8_ret
insuff_hist_delisted = insuff_hist & delisted_set
P(f"Insufficient-history excluded tickers: {len(insuff_hist):,}")
P(f"  Of these, delisted: {len(insuff_hist_delisted):,} ({len(insuff_hist_delisted)/max(len(insuff_hist),1)*100:.1f}%)")

# lifespan (quarters of valid price coverage, however short) for this group
lifespan = arq[arq["ticker"].isin(insuff_hist)].groupby("ticker")["q"].nunique()
lifespan_noprice = pd.Series(0, index=list(no_price))
lifespan = pd.concat([lifespan, lifespan_noprice[~lifespan_noprice.index.isin(lifespan.index)]])
P(f"Lifespan (distinct quarters with a valid price record) distribution, N={len(lifespan):,}:")
P(f"  mean={lifespan.mean():.2f}  median={lifespan.median():.1f}")
for p in [10, 25, 50, 75, 90]:
    P(f"  p{p:>2} = {lifespan.quantile(p/100):.1f} quarters")
P(f"  share with 0 quarters (no price record at all): {(lifespan==0).mean()*100 if len(lifespan) else 0:.1f}%")

with open(OUT + ".partial1", "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\n[checkpoint] wrote {OUT}.partial1 (continuing to parts 3-4)")

# ── (3) short-window DeltaS proxy for excluded firms vs included ───────────
P("\n" + "="*88)
P("(3) Short-window DeltaS proxy (4q and 8q, clearly labelled, NOT the panel construction)")
P("="*88)

def ivol_short(g, window, min_obs):
    g = g.sort_values("q")
    y = g["exret"].values
    X = g[["Mkt_RF", "SMB", "HML"]].values
    n = len(g)
    if n < min_obs:
        return np.nan
    ys = y[-window:] if n >= window else y
    Xs = X[-window:] if n >= window else X
    mask = ~(np.isnan(ys) | np.isnan(Xs).any(axis=1))
    ys, Xs = ys[mask], Xs[mask]
    if len(ys) < min_obs:
        return np.nan
    Xc = np.column_stack([np.ones(len(ys)), Xs])
    beta, *_ = np.linalg.lstsq(Xc, ys, rcond=None)
    resid = ys - Xc @ beta
    return resid.std(ddof=1) if len(resid) > len(Xc.T) else np.nan

# build exret for ALL tickers with >=4 returns (broaden slightly beyond the >=8 cut
# specifically for this short-window proxy, since that's the point of the exercise)
arq_4plus = arq_valid_ret[arq_valid_ret.groupby("ticker")["ret"].transform("size") >= 4].copy()
px_all = arq_4plus[["ticker", "q", "ret"]].merge(ff, on="q", how="left")
px_all["exret"] = px_all["ret"] - px_all["RF"]

excluded_for_proxy = (excluded - no_price)  # need at least some return data
proxy_pool = px_all[px_all["ticker"].isin(excluded_for_proxy)]
print(f"Computing short-window DeltaS proxy for {proxy_pool['ticker'].nunique():,} excluded tickers "
      f"with >=4 returns...")

for window, min_obs, label in [(4, 4, "4-quarter"), (8, 6, "8-quarter (min 6 obs)")]:
    rows = []
    for tkr, g in proxy_pool.groupby("ticker"):
        v_ = ivol_short(g, window, min_obs)
        if not np.isnan(v_):
            rows.append(v_)
    excl_ds = pd.Series(rows)
    incl_ds = ds["delta_s"]  # the panel's own (12q-window) DeltaS -- different window,
                              # but this is the best available comparator, labelled as such
    P(f"\n{label} proxy, EXCLUDED tickers (N={len(excl_ds):,} computable):")
    P(f"  mean={excl_ds.mean():.4f}  " + "  ".join(f"p{p}={excl_ds.quantile(p/100):.4f}" for p in [10,25,50,75,90]))
    P(f"INCLUDED tickers' actual panel DeltaS (12q rolling, N={len(incl_ds):,}, NOT the same window "
      f"-- reported for context, not a like-for-like test):")
    P(f"  mean={incl_ds.mean():.4f}  " + "  ".join(f"p{p}={incl_ds.quantile(p/100):.4f}" for p in [10,25,50,75,90]))
    diff = excl_ds.mean() - incl_ds.mean()
    P(f"  Excluded-minus-included mean difference: {diff:+.4f} "
      f"({diff/incl_ds.mean()*100:+.1f}% relative)")

# ── (4) terminal-quarter return: excluded-delisted vs included-delisted ────
P("\n" + "="*88)
P("(4) Terminal-quarter return: excluded-delisted vs included-delisted firms")
P("="*88)
last_obs_all = arq_valid_ret.sort_values(["ticker", "q"]).groupby("ticker").tail(1)
excl_delisted_term = last_obs_all[last_obs_all["ticker"].isin(insuff_hist_delisted)]["ret"]
incl_delisted = tickers_final & delisted_set
incl_delisted_term = last_obs_all[last_obs_all["ticker"].isin(incl_delisted)]["ret"]
P(f"Excluded-delisted terminal return (N={len(excl_delisted_term):,}): "
  f"mean={excl_delisted_term.mean()*100:+.2f}%  median={excl_delisted_term.median()*100:+.2f}%")
P(f"Included-delisted terminal return (N={len(incl_delisted_term):,}): "
  f"mean={incl_delisted_term.mean()*100:+.2f}%  median={incl_delisted_term.median()*100:+.2f}%")
diff_term = excl_delisted_term.mean() - incl_delisted_term.mean()
P(f"Difference (excluded - included): {diff_term*100:+.2f} percentage points")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
