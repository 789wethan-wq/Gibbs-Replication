"""G1 — yfinance cross-check against an independent source.

CRSP itself is not accessible in this environment: no WRDS/CRSP credentials
are configured anywhere in this codebase (confirmed by grep across
project/*.py and robustness/*.py -- only prose comments about what a CRSP
panel WOULD show, no actual access), consistent with MW2's premise that
CRSP access needs a one-sentence disclosure (cost/institutional/scope) --
here it is scope: never obtained, not attempted, no institutional
subscription in this environment.

Stooq (the free source tried previously) remains bot-gated behind a
JavaScript proof-of-work challenge, retried this session with the same
result -- confirmed persistent, not transient.

A genuinely different, real, independently-sourced dataset IS reachable:
Nasdaq Data Link's WIKI/PRICES table (the original free, community-
contributed Quandl equity price dataset, distinct lineage from Sharadar/SF1
and from yfinance) -- covers 1980-12-12 to 2018-03-27, i.e. the first ~23 of
the SP500 panel's 29 years (1995-2023). This is not CRSP, and is stated as
such throughout; it is the best available free independent cross-check for
the overlapping period.
"""
import os
import numpy as np
import pandas as pd
import nasdaqdatalink

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/G1_yfinance_wiki_crosscheck.txt"

print(f"[pid={os.getpid()}] G1 — fresh process")
log = []
def P(s=""):
    print(s)
    log.append(str(s))

nasdaqdatalink.ApiConfig.api_key = os.environ["NASDAQ_DATA_LINK_API_KEY"]  # set via environment; never commit keys

P("="*88)
P("G1 — yfinance vs WIKI/PRICES cross-check (CRSP not accessible; see note)")
P("="*88)
P("CRSP: no WRDS/CRSP access configured in this environment (scope -- never")
P("obtained, no institutional subscription). Stooq: retried this session,")
P("still returns a JS proof-of-work challenge, not data -- confirmed persistent.")
P("WIKI/PRICES (Nasdaq Data Link, deprecated 2018 but historically free,")
P("Quandl-community-sourced, distinct lineage from Sharadar and yfinance):")
P("used here for the overlapping 1995-2018 window.")

px = pd.read_parquet(f"{DATA}/stock_prices_monthly.parquet")
px.index = pd.to_datetime(px.index)
all_tickers = list(px.columns)

SEED = 20260726
rng = np.random.RandomState(SEED)
sample = list(rng.choice(all_tickers, size=50, replace=False))
P(f"\nRandom sample (seed={SEED}, n=50): {sample}")

results = []
for i, t in enumerate(sample):
    try:
        wp = nasdaqdatalink.get_table("WIKI/PRICES", ticker=t, paginate=True)
    except Exception as e:
        results.append(dict(ticker=t, status="fetch_failed", reason=str(e)[:100]))
        continue
    if wp is None or len(wp) == 0:
        results.append(dict(ticker=t, status="not_in_wiki", reason="ticker not covered by WIKI/PRICES"))
        continue
    wp = wp.sort_values("date").set_index("date")
    wp_m = wp["adj_close"].resample("ME").last()

    yf_px = px[t].dropna()
    yf_px.index = yf_px.index.to_period("M").to_timestamp("M")
    wp_m.index = wp_m.index.to_period("M").to_timestamp("M")

    common_idx = yf_px.index.intersection(wp_m.index)
    if len(common_idx) < 24:
        results.append(dict(ticker=t, status="insufficient_overlap", reason=f"only {len(common_idx)} common months"))
        continue

    yf_ret = yf_px.reindex(common_idx).pct_change().dropna()
    wp_ret = wp_m.reindex(common_idx).pct_change().dropna()
    both_idx = yf_ret.index.intersection(wp_ret.index)
    yf_r = yf_ret.reindex(both_idx)
    wp_r = wp_ret.reindex(both_idx)

    corr = yf_r.corr(wp_r)
    mad = (yf_r - wp_r).abs().median()
    sign_agree = (np.sign(yf_r) == np.sign(wp_r)).mean()
    close_agree = ((yf_r - wp_r).abs() < 0.01).mean()
    results.append(dict(ticker=t, status="ok", n_months=len(both_idx), corr=corr,
                         median_abs_diff=mad, sign_agreement=sign_agree, within_1pt=close_agree,
                         first=str(both_idx.min().date()), last=str(both_idx.max().date())))
    P(f"  [{i+1}/50] {t}: n={len(both_idx)}  corr={corr:.4f}  median|diff|={mad:.4f}  "
      f"sign-agree={sign_agree*100:.1f}%  within-1pt={close_agree*100:.1f}%  range={both_idx.min().date()}..{both_idx.max().date()}")

res_df = pd.DataFrame(results)
ok = res_df[res_df["status"] == "ok"]
P("\n" + "="*88)
P("SUMMARY")
P("="*88)
P(f"Fetched/comparable: {len(ok)} / 50 ({len(ok)/50*100:.0f}%)")
status_counts = res_df["status"].value_counts()
for s, c in status_counts.items():
    P(f"  {s}: {c}")
if len(ok) > 0:
    P(f"\nMean correlation(yfinance, WIKI/PRICES) monthly returns: {ok['corr'].mean():.4f} (median {ok['corr'].median():.4f})")
    P(f"Mean sign-agreement rate: {ok['sign_agreement'].mean()*100:.1f}%")
    P(f"Mean within-1-percentage-point agreement rate: {ok['within_1pt'].mean()*100:.1f}%")
    P(f"Mean median-abs-return-difference: {ok['median_abs_diff'].mean():.4f}")
    low_corr = ok[ok["corr"] < 0.95]
    P(f"Tickers with corr < 0.95: {len(low_corr)} ({low_corr['ticker'].tolist() if len(low_corr) else 'none'})")
    P(f"\nCoverage window per ticker averages {ok['n_months'].mean():.0f} months (max possible ~278,")
    P(f"1995-01 through 2018-03) -- this validates roughly the first 80% of the panel's span,")
    P(f"not the full 1995-2023 period; 2018-2023 remains unvalidated against any independent")
    P(f"source in this environment.")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
