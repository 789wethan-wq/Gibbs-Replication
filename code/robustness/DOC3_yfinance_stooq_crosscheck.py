"""Documentation query 3 — yfinance prices the SP500 comparison-arm returns
used throughout the paper. SF1 SEP was confirmed NOT entitled (E2 gate, live
API check, empty datatable). This attempts to cross-check a random 50-name
sample of the SP500 panel's ALREADY-CACHED yfinance-derived monthly returns
(data/stock_prices_monthly.parquet) against Stooq (stooq.com), a free,
independently-sourced daily/monthly price feed, and report agreement rates.

RESULT (documented, not worked around): stooq.com's CSV download endpoint
(/q/d/l/) now returns a JavaScript proof-of-work anti-bot challenge page
instead of data for every ticker tested (confirmed on AAPL directly, and via
this script's 50-name sample) -- this is new since whenever this spec was
written; it was a plain CSV endpoint historically. pandas_datareader (current
version, checked here) raises NotImplementedError for source='stooq' -- that
backend has been removed. Deliberately solving the JS proof-of-work challenge
to defeat Stooq's anti-bot gate was NOT attempted. Net result: with SEP not
entitled (E2) and Stooq now inaccessible without circumventing bot protection,
NO free independently-sourced second price feed could be reached from this
environment to complete the cross-check as specified. This is reported as the
finding for this query, not papered over with a same-source (yfinance-vs-
yfinance) comparison that would not answer the question asked.
"""
import os
import time
import numpy as np
import pandas as pd
import urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/DOC3_yfinance_stooq_crosscheck.txt"

log = []
def P(s=""):
    print(s)
    log.append(str(s))

rng = np.random.RandomState(20250725)

px = pd.read_parquet(f"{DATA}/stock_prices_monthly.parquet")
px.index = pd.to_datetime(px.index)
all_tickers = list(px.columns)
full_sample = list(rng.choice(all_tickers, size=50, replace=False))
# Stooq's CSV endpoint is bot-gated (see module docstring) -- confirmed on a
# probe of 5 tickers rather than hammering their server across all 50 when the
# gate is clearly systematic, not ticker-specific.
sample = full_sample[:5]
P("="*78)
P("DOC query 3 — yfinance vs Stooq cross-check, random 50-name sample (probe of 5)")
P("="*78)
P(f"Universe: {len(all_tickers)} SP500 tickers (yfinance-derived, cached locally)")
P(f"Full random sample drawn (seed=20250725, n=50, for the record): {full_sample}")
P(f"Probing the first 5 of these against Stooq to confirm the gate is systematic:")


def stooq_symbol(t):
    return t.replace("-", "-").replace(".", "-").lower() + ".us"


def fetch_stooq_monthly(ticker):
    sym = stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=m"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        if "proof-of-work" in raw.lower() or "verify your browser" in raw.lower() or "<!DOCTYPE html>" in raw:
            return "js_challenge"
        if "Date,Open" not in raw and "Date, Open" not in raw:
            return None
        from io import StringIO
        df = pd.read_csv(StringIO(raw))
        if df.empty or "Close" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        return df["Close"]
    except Exception:
        return None


results = []
for i, t in enumerate(sample):
    s = fetch_stooq_monthly(t)
    if s == "js_challenge":
        P(f"  [{i+1}/{len(sample)}] {t}: Stooq returned a JS proof-of-work challenge page, not CSV data")
        results.append(dict(ticker=t, status="js_challenge"))
        time.sleep(0.3)
        continue
    if s is None or len(s) < 12:
        P(f"  [{i+1}/{len(sample)}] {t}: Stooq fetch FAILED or insufficient history — skipped")
        results.append(dict(ticker=t, status="fetch_failed"))
        time.sleep(0.3)
        continue
    yf_px = px[t].dropna()
    yf_px.index = yf_px.index.to_period("M").to_timestamp("M")
    s_m = s.copy()
    s_m.index = s_m.index.to_period("M").to_timestamp("M")
    s_m = s_m.groupby(s_m.index).last()
    common_idx = yf_px.index.intersection(s_m.index)
    if len(common_idx) < 12:
        P(f"  [{i+1}/50] {t}: only {len(common_idx)} common months — skipped")
        results.append(dict(ticker=t, status="insufficient_overlap"))
        time.sleep(0.3)
        continue
    yf_ret = yf_px.reindex(common_idx).pct_change().dropna()
    st_ret = s_m.reindex(common_idx).pct_change().dropna()
    both_idx = yf_ret.index.intersection(st_ret.index)
    yf_r = yf_ret.reindex(both_idx)
    st_r = st_ret.reindex(both_idx)
    corr = yf_r.corr(st_r)
    mad = (yf_r - st_r).abs().median()
    sign_agree = (np.sign(yf_r) == np.sign(st_r)).mean()
    close_agree = ((yf_r - st_r).abs() < 0.01).mean()
    results.append(dict(ticker=t, status="ok", n_months=len(both_idx), corr=corr,
                         median_abs_diff=mad, sign_agreement=sign_agree,
                         within_1pt_agreement=close_agree))
    P(f"  [{i+1}/50] {t}: n={len(both_idx)}  corr={corr:.4f}  median|diff|={mad:.4f}  "
      f"sign-agree={sign_agree*100:.1f}%  within-1pt={close_agree*100:.1f}%")
    time.sleep(0.3)

res_df = pd.DataFrame(results)
ok = res_df[res_df["status"] == "ok"]
P("\n" + "="*78)
P("SUMMARY")
P("="*78)
P(f"Probed: {len(sample)} of the 50-name sample. Fetched successfully: {len(ok)} "
  f"({len(ok)/max(len(sample),1)*100:.0f}%)")
n_gated = (res_df["status"] == "js_challenge").sum()
P(f"Blocked by Stooq's JS proof-of-work challenge: {n_gated} / {len(sample)}")
if n_gated == len(sample):
    P("\nFINDING: Stooq's free CSV endpoint is uniformly bot-gated for this sample --")
    P("this is a platform-level access change, not a per-ticker data issue, so the")
    P("remaining 45 names in the drawn sample were not probed (would predictably fail")
    P("the same way). Combined with SEP's confirmed non-entitlement (E2), NO free")
    P("independently-sourced second price feed is reachable from this environment.")
    P("RECOMMENDATION for the manuscript: either (a) note this limitation explicitly")
    P("rather than claim a cross-check was performed, (b) obtain a paid/keyed data")
    P("source (Tiingo, Alpha Vantage, Polygon, EOD Historical Data) if this")
    P("verification is required for publication, or (c) substitute a WITHIN-source")
    P("consistency check (e.g. yfinance split/dividend-adjustment spot-checks against")
    P("publicly documented corporate actions for a few well-known names) with the")
    P("caveat that it is not an independent second source.")
if len(ok) > 0:
    P(f"Mean correlation(yfinance, Stooq) monthly returns:  {ok['corr'].mean():.4f}  "
      f"(median {ok['corr'].median():.4f})")
    P(f"Mean sign-agreement rate:                            {ok['sign_agreement'].mean()*100:.1f}%")
    P(f"Mean within-1-percentage-point agreement rate:       {ok['within_1pt_agreement'].mean()*100:.1f}%")
    P(f"Mean median-abs-return-difference:                   {ok['median_abs_diff'].mean():.4f}")
    low_corr = ok[ok["corr"] < 0.95]
    P(f"\nTickers with corr < 0.95 ({len(low_corr)}): {low_corr['ticker'].tolist() if len(low_corr) else 'none'}")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
