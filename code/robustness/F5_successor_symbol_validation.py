"""F5 — Successor-symbol spot validation via SEC EDGAR full-text search.

Draws a random sample of 100 delisted analysis-panel firms with a populated
successor symbol (relatedtickers), then checks each link against SEC EDGAR's
full-text search API (efts.sec.gov, covers filings from 2001 onward).

Methodology (stated explicitly, since this is an automated proxy, not manual
verification): for each (delisted company name, successor company name) pair,
query EDGAR full-text search for filings containing BOTH names as exact
phrases, within a window of [delisting year - 1, delisting year + 1]. A hit
count > 0 is treated as CONFIRMED (the two entities are co-mentioned in an
SEC filing near the delisting date, which for two company names appearing
together is almost always in a merger/acquisition/name-change context).
Zero hits for the delisted name ANYWHERE in the full-text index (which only
covers 2001+) is tracked separately as NOT CHECKABLE rather than folded into
"contradicted", since it is a coverage gap, not evidence against the link.
Zero co-occurrence hits despite the delisted name itself being indexed is
INDETERMINATE (link neither confirmed nor contradicted by this method).
CONTRADICTED is not directly produced by this design (a co-occurrence check
cannot positively contradict a link, only fail to confirm it) -- reported
honestly as a limitation of the automated method, not worked around.
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
OUT = "../results/revision/F5_successor_symbol_validation.txt"

print(f"[pid={os.getpid()}] F5 — fresh process")
log = []
def P(s=""):
    print(s)
    log.append(str(s))

SEED = 20260726
P("="*88)
P("F5 — Successor-symbol spot validation via SEC EDGAR full-text search")
P("="*88)
P(f"Seed: {SEED}")

tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"]
panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
analysis_tickers = set(panel["ticker"].unique())

delisted = sf1t[(sf1t["isdelisted"] == "Y") & sf1t["ticker"].isin(analysis_tickers)].copy()
delisted["has_related"] = delisted["relatedtickers"].notna() & (delisted["relatedtickers"].str.strip() != "")
pool = delisted[delisted["has_related"]].copy()
pool["successor_ticker"] = pool["relatedtickers"].str.split().str[0]
pool = pool.dropna(subset=["successor_ticker", "name", "lastpricedate"])
P(f"Pool: delisted analysis-panel firms with a populated successor symbol: {len(pool):,}")

rng = np.random.RandomState(SEED)
sample = pool.sample(n=min(100, len(pool)), random_state=rng)
P(f"Sample drawn: {len(sample)} firms")

# successor names, looked up from the same tickers table (any table row, delisted or not)
name_lookup = tk[["ticker", "name"]].drop_duplicates("ticker").set_index("ticker")["name"]
sample = sample.copy()
sample["successor_name"] = sample["successor_ticker"].map(name_lookup)
sample["delist_year"] = pd.to_datetime(sample["lastpricedate"], errors="coerce").dt.year

HEADERS = {"User-Agent": "Independent Research replication-audit contact: 789wethan@gmail.com"}


def edgar_fts(query, start=None, end=None, forms=None):
    params = {"q": query}
    if start and end:
        params["dateRange"] = "custom"
        params["startdt"] = start
        params["enddt"] = end
    if forms:
        params["forms"] = forms
    url = "https://efts.sec.gov/LATEST/search-index?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("hits", {}).get("total", {}).get("value", 0)
    except Exception as e:
        return None  # network/API failure, distinct from a genuine zero-hit


results = []
for i, (idx, row) in enumerate(sample.iterrows()):
    tkr = row["ticker"]
    name = str(row["name"]).strip()
    succ_tkr = row["successor_ticker"]
    succ_name = row["successor_name"]
    year = row["delist_year"]
    if pd.isna(year) or pd.isna(succ_name):
        results.append(dict(ticker=tkr, name=name, successor_ticker=succ_tkr,
                             successor_name=succ_name, status="not_checkable",
                             reason="missing successor name or delist year"))
        P(f"  [{i+1}/100] {tkr} ({name[:30]}) -> {succ_tkr}: NOT CHECKABLE (missing name/date)")
        continue
    year = int(year)
    start = f"{max(year-1,2001)}-01-01"
    end = f"{min(year+1,2026)}-12-31"

    if year - 1 > 2026 or year + 1 < 2001:
        results.append(dict(ticker=tkr, name=name, successor_ticker=succ_tkr,
                             successor_name=succ_name, status="not_checkable",
                             reason=f"delisting year {year} entirely outside EDGAR full-text coverage (2001+)"))
        P(f"  [{i+1}/100] {tkr} ({name[:30]}) -> {succ_tkr}: NOT CHECKABLE (year {year} outside 2001+ coverage)")
        continue

    name_q = f'"{name}"'
    hits_name_alone = edgar_fts(name_q, start, end)
    time.sleep(0.3)

    if hits_name_alone is None:
        results.append(dict(ticker=tkr, name=name, successor_ticker=succ_tkr,
                             successor_name=succ_name, status="not_checkable", reason="API error"))
        P(f"  [{i+1}/100] {tkr} ({name[:30]}) -> {succ_tkr}: NOT CHECKABLE (API error)")
        continue

    if hits_name_alone == 0:
        results.append(dict(ticker=tkr, name=name, successor_ticker=succ_tkr,
                             successor_name=succ_name, status="not_checkable",
                             reason=f"delisted company name has zero full-text hits {start}..{end} "
                                     f"(pre-2001 filer, name mismatch, or too obscure to index)"))
        P(f"  [{i+1}/100] {tkr} ({name[:30]}) -> {succ_tkr}: NOT CHECKABLE (0 hits for company name at all)")
        continue

    co_q = f'"{name}" "{succ_name}"'
    hits_co = edgar_fts(co_q, start, end)
    time.sleep(0.3)

    if hits_co is None:
        status = "not_checkable"
        reason = "API error on co-occurrence query"
    elif hits_co > 0:
        status = "confirmed"
        reason = f"{hits_co} filing(s) mention both names, {start}..{end}"
    else:
        status = "indeterminate"
        reason = f"company name indexed ({hits_name_alone} hits) but no co-occurrence with successor name found"

    results.append(dict(ticker=tkr, name=name, successor_ticker=succ_tkr,
                         successor_name=succ_name, status=status, reason=reason))
    P(f"  [{i+1}/100] {tkr} ({name[:30]}) -> {succ_tkr} ({str(succ_name)[:30]}): "
      f"{status.upper()} ({reason})")

res_df = pd.DataFrame(results)
P("\n" + "="*88)
P("F5 SUMMARY")
P("="*88)
counts = res_df["status"].value_counts()
for status in ["confirmed", "indeterminate", "not_checkable"]:
    n = counts.get(status, 0)
    P(f"  {status:16}: {n:>3} / {len(res_df)} ({n/len(res_df)*100:.1f}%)")

checkable = res_df[res_df["status"] != "not_checkable"]
if len(checkable) > 0:
    conf_rate_of_checkable = (checkable["status"] == "confirmed").mean()
    P(f"\nOf the {len(checkable)} checkable links, {(checkable['status']=='confirmed').sum()} "
      f"({conf_rate_of_checkable*100:.1f}%) confirmed by EDGAR co-occurrence.")

P("\nCaveat, stated per the method's own limitation: this design cannot produce a")
P("'contradicted' verdict -- it only distinguishes confirmed-by-co-occurrence from")
P("not-found. A successor link that IS real but involves a name change, a shell")
P("merger with minimal SEC paper trail, or predates full-text-searchable EDGAR")
P("(pre-2001) would show as indeterminate or not-checkable here, not as evidence")
P("against the link. This bounds the CONFIRMED rate as a lower bound on link")
P("validity, not a point estimate of the true validation rate.")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
