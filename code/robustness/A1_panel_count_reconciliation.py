"""A1 — Panel-count reconciliation for PLOS ONE Section 3.1 / 4.8.

Recounts unique tickers at every stage of the R18 build and checks the three
numbers the paper asserts: 15,522 raw universe / 12,449 analysis panel /
8,937 delisted within panel.
"""
import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
LOG = []
def say(*a):
    line = " ".join(str(x) for x in a); print(line); LOG.append(line)

say("="*72); say("A1 — PANEL-COUNT RECONCILIATION (R18 pipeline stages)"); say("="*72)

# stage 0: universe filter (R18 step 1 — no window applied)
tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"]
uni = sf1t[
    sf1t["category"].str.contains("Domestic Common", na=False) &
    sf1t["exchange"].isin(["NYSE","NASDAQ","NYSEARCA","BATS","NYSEMKT"]) &
    (sf1t["currency"] == "USD")
].copy()
uni_set = set(uni["ticker"])
say(f"\n[stage 0] SF1-covered US domestic common (category/exchange/USD filter,")
say(f"          NO time-window restriction)            = {len(uni_set):,}")

# stage 1: has any ARQ price row (positive price, valid calendardate)
sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                      columns=["ticker","dimension","calendardate","price"])
arq = sf1[(sf1["dimension"] == "ARQ") & sf1["ticker"].isin(uni_set)].copy()
arq["calendardate"] = pd.to_datetime(arq["calendardate"], errors="coerce")
arq = arq.dropna(subset=["calendardate","price"])
arq = arq[arq["price"] > 0]
say(f"[stage 1] ... with >=1 valid ARQ price observation = {arq['ticker'].nunique():,}")

arq["q"] = arq["calendardate"].dt.to_period("Q")
arq = (arq.sort_values(["ticker","calendardate"])
          .drop_duplicates(["ticker","q"], keep="last"))
w = arq[(arq["q"] >= pd.Period("1995Q3")) & (arq["q"] <= pd.Period("2023Q4"))]
say(f"[stage 1b] ... with >=1 ARQ price INSIDE the T window (1995Q3-2023Q4)"
    f" = {w['ticker'].nunique():,}")

# stage 2: valid 1-quarter return (consecutive quarters) + min 8 return obs
arq = arq.sort_values(["ticker","q"])
arq["q_ord"] = arq["q"].apply(lambda p: p.ordinal)
arq["gap"] = arq["q_ord"] - arq.groupby("ticker")["q_ord"].shift(1)
ret_ok = arq[arq["gap"] == 1]
say(f"[stage 2a] ... with >=1 valid consecutive-quarter return = "
    f"{ret_ok['ticker'].nunique():,}")
cnt = ret_ok.groupby("ticker").size()
say(f"[stage 2b] ... with >=8 quarterly return obs (R18 filter) = "
    f"{(cnt >= 8).sum():,}")

# stage 3/4: the saved analysis panel (valid return AND valid ΔS, inside T window)
panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
say(f"[stage 3] analysis panel (valid return AND valid ΔS, inside T window)")
say(f"          = {panel['ticker'].nunique():,} tickers, "
    f"{len(panel):,} ticker-quarters, {panel['q'].nunique()} quarters "
    f"({panel['q'].min()}..{panel['q'].max()})")
pe = panel.dropna(subset=["ret_next","delta_s_z"])
pf = panel.dropna(subset=["ret_next","delta_s_z","delta_h_z"])
say(f"[stage 3b] ... with valid NEXT-quarter return (entropy est. panel) = "
    f"{pe['ticker'].nunique():,}  ({len(pe):,} obs)")
say(f"[stage 3c] ... also valid ΔH (Model B est. panel) = "
    f"{pf['ticker'].nunique():,}  ({len(pf):,} obs)")

# delisted counts at stage 0 and stage 3
deli = set(uni.loc[uni["isdelisted"] == "Y", "ticker"])
say(f"\n[delisted] stage 0 universe: {len(deli & uni_set):,} of {len(uni_set):,}")
p_t = set(panel["ticker"])
say(f"[delisted] stage 3 analysis panel: {len(deli & p_t):,} of {len(p_t):,}")

# verdicts
say("\n" + "-"*72)
say("RECONCILIATION vs paper assertions")
say(f"  paper 15,522 raw SF1-covered US common:        stage 0 = {len(uni_set):,}  "
    f"-> {'MATCH' if len(uni_set)==15522 else 'MISMATCH'}")
say(f"  paper 12,449 analysis panel:                   stage 3 = "
    f"{panel['ticker'].nunique():,}  -> "
    f"{'MATCH' if panel['ticker'].nunique()==12449 else 'MISMATCH'}")
say(f"  paper  8,937 delisted within analysis panel:   stage 3 = "
    f"{len(deli & p_t):,}  -> {'MATCH' if len(deli & p_t)==8937 else 'MISMATCH'}")

with open("../results/revision/A1_panel_count_reconciliation.txt", "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\nSaved: ../results/revision/A1_panel_count_reconciliation.txt")
