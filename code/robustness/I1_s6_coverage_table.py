"""I1_s6_coverage_table.py — Addendum 4, I1: citable supplementary coverage
table for the 21 (+MCI) catastrophic delistings checked in H2.

A referee correctly declined to accept H2's narrative summary on its own
terms: an author reporting an error in their own analysis that happens to
resolve a reviewer's concern in the most favourable direction is not
self-verifying. This script produces a directly checkable table instead of a
claim, with two integrity requirements:

  1. Panel-membership fields ("Quarters in the R18 analysis panel", "In R18
     panel?") are read DIRECTLY from `merged_sf1_quarterly_survfree.parquet`
     -- the actual saved panel object the reported R18 estimates (Table 7's
     N=392,557 Model B FM regression, referred to as Table 9 in the current
     draft) are computed from -- not re-derived from the raw SF1/tickers
     tables. The panel's shape and a content hash are printed, and the
     Model B FM design matrix built from it is hashed and compared against
     the hash already verified in H1 (robustness/H1_terminal_return_cell.py,
     delta=0 anchor cell, which reproduced the published t(IVOL)=+0.018 /
     t(STAB)=+3.461 exactly): SHA-1 872dd066c1a8e9bc585693f6669968d9b7a79c4c
     on the N=392,557 x 3 design matrix.
  2. If any of the 21 (or MCI) is NOT in the analysis panel, that is reported
     plainly, not smoothed over -- this check is only worth running if it is
     capable of coming back unfavourable to the manuscript's current text.
"""
import os
import hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/I1_s6_coverage_table.txt"

print(f"[pid={os.getpid()}] I1 — fresh process")
log = []
def P(s=""):
    print(s)
    log.append(str(s))

P("=" * 96)
P("I1 — S6 Table: supplementary coverage table for the catastrophic delistings checked in H2")
P("=" * 96)

# ── load the SAME panel object the reported R18 / Table 9 estimates are
#    computed from -- membership is read from THIS, not re-derived ─────────
panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
panel_hash = hashlib.sha1(pd.util.hash_pandas_object(panel, index=True).values.tobytes()).hexdigest()
P(f"\nPanel object: data/merged_sf1_quarterly_survfree.parquet")
P(f"  shape={panel.shape}  SHA-1(content)={panel_hash}")
P(f"  tickers={panel['ticker'].nunique():,}  quarters={panel['q'].nunique()}  "
  f"range={panel['q'].min()}..{panel['q'].max()}")

# reproduce the Model B FM design matrix exactly as H1's delta=0 anchor cell
# did, and confirm the hash matches what H1 already verified against the
# published t(IVOL)=+0.018 / t(STAB)=+3.461
pf = panel.dropna(subset=["ret_next", "delta_s_z", "delta_h_z"])
X = sm.add_constant(pf[["delta_h_z", "delta_s_z"]]).values
design_hash = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()
EXPECTED_HASH = "872dd066c1a8e9bc585693f6669968d9b7a79c4c"
P(f"\nModel B FM design matrix (Table 9 / R18's N=392,557 estimation sample):")
P(f"  shape={X.shape}  SHA-1={design_hash}")
P(f"  Matches H1's already-verified delta=0 anchor hash ({EXPECTED_HASH}): "
  f"{'YES -- MATCH' if design_hash == EXPECTED_HASH else 'NO -- MISMATCH, investigate before proceeding'}")
assert design_hash == EXPECTED_HASH, "design hash mismatch -- stopping, do not proceed with a table built on the wrong panel"

panel_tickers = set(panel["ticker"].unique())

tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"]
sf1_full = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                            columns=["ticker", "dimension", "calendardate", "datekey", "price", "dps"])

# ── R18's own exclusion-stage logic (verbatim from F1_excluded_ticker_audit.py
#    / R18_sf1_quarterly_survfree.py), reused here only to state PRECISELY why
#    an excluded ticker was excluded -- not to redetermine panel membership ──
def consecutive_gap_returns(ticker):
    arq = sf1_full[(sf1_full["ticker"] == ticker) & (sf1_full["dimension"] == "ARQ")].copy()
    arq["calendardate"] = pd.to_datetime(arq["calendardate"], errors="coerce")
    arq = arq.dropna(subset=["calendardate", "price"])
    arq = arq[arq["price"] > 0].sort_values("calendardate")
    if len(arq) == 0:
        return 0, 0, None, None
    arq["q"] = arq["calendardate"].dt.to_period("Q")
    arq = arq.drop_duplicates("q", keep="last").sort_values("q")
    n_q = len(arq)
    first_dk = arq["datekey"].min()
    last_dk = arq["datekey"].max()
    arq["q_ord"] = arq["q"].apply(lambda p: p.ordinal)
    arq["gap"] = arq["q_ord"] - arq["q_ord"].shift(1)
    arq["price_prev"] = arq["price"].shift(1)
    ret_px = arq["price"] / arq["price_prev"] - 1.0
    div_q = (arq["dps"].fillna(0) / 4.0) / arq["price_prev"]
    arq["ret"] = np.where(arq["gap"] == 1, ret_px + div_q, np.nan)
    n_valid_ret = arq["ret"].notna().sum()
    return n_q, n_valid_ret, first_dk, last_dk

# ── the roster: headline three (as named in the manuscript), MCI (checked in
#    H2, explicitly requested here even though it fell outside H2's "21"
#    headline total), and the 18-name extended set ──────────────────────────
ROSTER = [
    dict(company="Enron", queried="ENE, ENRN", ticker="ENRNQ"),
    dict(company="WorldCom", queried="WCOM", ticker="WCOEQ"),
    dict(company="Lehman Brothers", queried="LEH", ticker="LEHMQ"),
    dict(company="MCI (WorldCom post-scandal reorg entity)", queried="MCIP", ticker="MCIP"),
    dict(company="Adelphia", queried=None, ticker="ADELQ"),
    dict(company="Global Crossing", queried=None, ticker="GX"),
    dict(company="Conseco", queried=None, ticker="CNCEQ"),
    dict(company="Kmart", queried=None, ticker="KM"),
    dict(company="Delphi", queried=None, ticker="DPHIQ"),
    dict(company="Washington Mutual", queried=None, ticker="WAMUQ"),
    dict(company="Bear Stearns", queried=None, ticker="BSC1"),
    dict(company="Circuit City", queried=None, ticker="CCTYQ"),
    dict(company="General Growth", queried=None, ticker="GGP"),
    dict(company="MF Global", queried=None, ticker="MFGLQ"),
    dict(company="Peabody", queried=None, ticker="BTUUQ"),
    dict(company="SunEdison", queried=None, ticker="SUNEQ"),
    dict(company="Toys R Us", queried=None, ticker="TOY"),
    dict(company="Sears", queried=None, ticker="SHLDQ"),
    dict(company="Hertz", queried=None, ticker="HTZGQ"),
    dict(company="Chesapeake", queried=None, ticker="CHKAQ"),
    dict(company="Frontier Communications", queried=None, ticker="FTRCQ"),
    dict(company="Revlon", queried=None, ticker="REVRQ"),
]
P(f"\nRoster: {len(ROSTER)} rows (H2's headline 3 + MCI, requested explicitly in this spec "
  f"even though H2's own summary quoted a '21' total that did not include MCI as a distinct "
  f"row -- noted here for transparency rather than silently forced to 21 by dropping a name) "
  f"+ the 18-name extended set.")

rows = []
n_mismatches = 0
for entry in ROSTER:
    t = entry["ticker"]
    meta = sf1t[sf1t["ticker"] == t]
    permaticker = meta["permaticker"].iloc[0] if len(meta) else None
    n_q, n_valid_ret, first_dk, last_dk = consecutive_gap_returns(t)
    in_panel = t in panel_tickers
    n_q_panel = int((panel["ticker"] == t).sum())
    if in_panel and n_q_panel == 0:
        n_mismatches += 1
    reason = ""
    if not in_panel:
        if n_valid_ret < 8:
            reason = f"fails R18's >=8-consecutive-gap-quarterly-return filter ({n_valid_ret} valid returns from {n_q} filed quarters)"
        else:
            reason = "excluded at a later R18 stage (ΔS non-computable or no T-quarter overlap) -- not the history-length filter"
    rows.append(dict(
        company=entry["company"], queried=entry["queried"] or "—", ticker=t,
        permaticker=permaticker, first_datekey=first_dk, last_datekey=last_dk,
        n_q_sf1=n_q, n_q_panel=n_q_panel, in_panel=in_panel, reason=reason,
    ))

res = pd.DataFrame(rows)

P("\n" + "-" * 96)
P("INTEGRITY CHECK 2: any of the roster NOT in the R18 analysis panel? (reported plainly either way)")
P("-" * 96)
not_in = res[~res["in_panel"]]
if len(not_in) == 0:
    P("N/A path not taken: every row IS in the panel." if False else "")
P(f"Rows checked: {len(res)}.  In R18 panel: {res['in_panel'].sum()}.  NOT in panel: {len(not_in)}.")
if len(not_in):
    P("\nNames NOT in the R18 analysis panel (reported as found, not smoothed over):")
    for _, r in not_in.iterrows():
        P(f"  - {r['company']} ({r['ticker']}): {r['reason']}")
if n_mismatches:
    P(f"\nWARNING: {n_mismatches} ticker(s) flagged in_panel=True by set-membership but have "
      f"0 rows when counted directly in the panel dataframe -- internal inconsistency, investigate.")
else:
    P("\nNo internal inconsistency between set-membership and direct row-count found.")

P("\n" + "=" * 96)
P("S6 TABLE — data")
P("=" * 96)
header = ("| Company | Ticker as queried in earlier drafts | Ticker as present in SF1 | permaticker | "
          "First DATEKEY | Last DATEKEY | Quarters in SF1 | Quarters in R18 panel | In R18 panel? | If no, reason |")
sep = "|---|---|---|---|---|---|---|---|---|---|"
P(header)
P(sep)
for _, r in res.iterrows():
    P(f"| {r['company']} | {r['queried']} | {r['ticker']} | {r['permaticker']} | "
      f"{r['first_datekey']} | {r['last_datekey']} | {r['n_q_sf1']} | {r['n_q_panel']} | "
      f"{'Yes' if r['in_panel'] else '**No**'} | {r['reason'] if r['reason'] else '—'} |")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
