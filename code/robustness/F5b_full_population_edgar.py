"""F5b — EDGAR successor-link check scaled from F5's 100-firm sample to the
FULL population of delisted analysis-panel firms with a populated
relatedtickers successor symbol (4,593 firms; 2,176 with a name resolvable
anywhere in Sharadar's own ticker master, per F5's diagnosed data-quality
finding -- only those 2,176 need an EDGAR query, the remaining 2,417 are
mechanically not-checkable exactly as in F5).

Same methodology as F5_successor_symbol_validation.py, run to completion,
not a resample: this is the review-requested "EDGAR-confirmed subset" input
for the sensitivity check (does the entropy-premium collapse survive
excluding EDGAR-confirmed M&A firms from the delisted population).
"""
import os
import time
import json
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/F5b_full_population_edgar.txt"
OUT_CSV = "../data/F5b_edgar_classification.csv"

print(f"[pid={os.getpid()}] F5b — fresh process")
log = []
def P(s=""):
    print(s)
    log.append(str(s))

P("="*88)
P("F5b — EDGAR successor-link check, FULL population (not a sample)")
P("="*88)

tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"]
panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
analysis_tickers = set(panel["ticker"].unique())

delisted = sf1t[(sf1t["isdelisted"] == "Y") & sf1t["ticker"].isin(analysis_tickers)].copy()
delisted["has_related"] = delisted["relatedtickers"].notna() & (delisted["relatedtickers"].str.strip() != "")
pool = delisted[delisted["has_related"]].copy()
pool["successor_ticker"] = pool["relatedtickers"].str.split().str[0]
pool = pool.dropna(subset=["successor_ticker", "name", "lastpricedate"])
P(f"Full population: {len(pool):,} delisted analysis-panel firms with a populated successor symbol")

name_lookup = tk[["ticker", "name"]].drop_duplicates("ticker").set_index("ticker")["name"]
pool["successor_name"] = pool["successor_ticker"].map(name_lookup)
pool["delist_year"] = pd.to_datetime(pool["lastpricedate"], errors="coerce").dt.year

HEADERS = {"User-Agent": "Independent Research replication-audit contact: 789wethan@gmail.com"}


def edgar_fts(query, start=None, end=None):
    params = {"q": query}
    if start and end:
        params["dateRange"] = "custom"
        params["startdt"] = start
        params["enddt"] = end
    url = "https://efts.sec.gov/LATEST/search-index?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("hits", {}).get("total", {}).get("value", 0)
    except Exception:
        return None


results = []
n = len(pool)
for i, (idx, row) in enumerate(pool.iterrows()):
    tkr = row["ticker"]
    name = str(row["name"]).strip()
    succ_tkr = row["successor_ticker"]
    succ_name = row["successor_name"]
    year = row["delist_year"]

    if i % 200 == 0:
        P(f"  progress {i}/{n}...")

    if pd.isna(year) or pd.isna(succ_name):
        results.append(dict(ticker=tkr, successor_ticker=succ_tkr, status="not_checkable",
                             reason="successor name not resolvable in Sharadar ticker master"))
        continue
    year = int(year)
    if year - 1 > 2026 or year + 1 < 2001:
        results.append(dict(ticker=tkr, successor_ticker=succ_tkr, status="not_checkable",
                             reason=f"year {year} outside EDGAR full-text coverage"))
        continue
    start = f"{max(year-1,2001)}-01-01"
    end = f"{min(year+1,2026)}-12-31"

    name_q = f'"{name}"'
    hits_name_alone = edgar_fts(name_q, start, end)
    time.sleep(0.25)

    if hits_name_alone is None:
        results.append(dict(ticker=tkr, successor_ticker=succ_tkr, status="not_checkable", reason="API error"))
        continue
    if hits_name_alone == 0:
        results.append(dict(ticker=tkr, successor_ticker=succ_tkr, status="not_checkable",
                             reason="zero full-text hits for company name"))
        continue

    co_q = f'"{name}" "{succ_name}"'
    hits_co = edgar_fts(co_q, start, end)
    time.sleep(0.25)

    if hits_co is None:
        results.append(dict(ticker=tkr, successor_ticker=succ_tkr, status="not_checkable", reason="API error (co-query)"))
    elif hits_co > 0:
        results.append(dict(ticker=tkr, successor_ticker=succ_tkr, status="confirmed",
                             reason=f"{hits_co} co-occurrence hits"))
    else:
        results.append(dict(ticker=tkr, successor_ticker=succ_tkr, status="indeterminate",
                             reason="no co-occurrence found"))

res_df = pd.DataFrame(results)
res_df.to_csv(OUT_CSV, index=False)
P(f"\nSaved classification: {OUT_CSV}")

P("\n" + "="*88)
P("F5b SUMMARY")
P("="*88)
counts = res_df["status"].value_counts()
for status in ["confirmed", "indeterminate", "not_checkable"]:
    c = counts.get(status, 0)
    P(f"  {status:16}: {c:>5} / {len(res_df)} ({c/len(res_df)*100:.1f}%)")
checkable = res_df[res_df["status"] != "not_checkable"]
if len(checkable) > 0:
    P(f"\nOf {len(checkable):,} checkable, {(checkable['status']=='confirmed').sum():,} "
      f"({(checkable['status']=='confirmed').mean()*100:.1f}%) confirmed.")
P(f"\nConfirmed count vs F5's 100-sample rate check: sample gave 14.0% of 100 = 14 confirmed;")
P(f"full population gives {counts.get('confirmed',0):,} confirmed of {len(res_df):,} "
  f"({counts.get('confirmed',0)/len(res_df)*100:.1f}%) -- "
  f"{'consistent with the sample rate' if abs(counts.get('confirmed',0)/len(res_df)*100 - 14.0) < 5 else 'notably different from the sample rate'}.")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
