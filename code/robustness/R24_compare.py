"""R24_compare.py — step 3 of the R24 code-path integrity check: read the two
JSON outputs written by the SEPARATE processes (R24_coldproc_sp500.py,
R24_coldproc_fulluniv.py) and check for evidence of a shared object: identical
hashes, or high residual correlation on alignable observations.

SP500 residuals are monthly (indexed by stock_id, month); full-universe
residuals are quarterly (indexed by ticker, quarter). To align, SP500 monthly
residuals are averaged within (ticker, quarter) for tickers that appear in
both panels, then matched to the full-universe quarterly residual for the
same (ticker, quarter).
"""
import json
import numpy as np
import pandas as pd

with open("../results/revision/R24_sp500_cold.json") as f:
    sp = json.load(f)
with open("../results/revision/R24_fulluniv_cold.json") as f:
    fu = json.load(f)

print("="*78)
print("R24 (3) — hash / shape / cluster comparison")
print("="*78)
print(f"SP500        : shape={tuple(sp['shape'])}  N={sp['N']:,}  clusters(date,firm)=({sp['n_dates']},{sp['n_firms']})")
print(f"FullUniverse : shape={tuple(fu['shape'])}  N={fu['N']:,}  clusters(date,firm)=({fu['n_dates']},{fu['n_firms']})")
print(f"design hashes equal? {sp['design_hash'] == fu['design_hash']}")
print(f"resid  hashes equal? {sp['resid_hash'] == fu['resid_hash']}")
print(f"SP500 design SHA-1        = {sp['design_hash']}")
print(f"FullUniverse design SHA-1 = {fu['design_hash']}")

print()
print(f"coef_TxDS  SP500={sp['coef_TxDS']:+.6f}   FullUniverse={fu['coef_TxDS']:+.6f}")
print(f"t_TxDS     SP500={sp['t_TxDS']:+.6f}   FullUniverse={fu['t_TxDS']:+.6f}")
print(f"Wald p     SP500={sp['p_TxDS']:.6f}    FullUniverse={fu['p_TxDS']:.6f}")

print()
print("="*78)
print("R24 (3) — residual correlation on alignable observations")
print("="*78)

sp_df = pd.DataFrame(sp["resid_key"], columns=["ticker", "q"])
sp_df["resid_sp"] = sp["resid_vals"]
# SP500 key's second field is already a quarter string (e.g. "2001Q3") from .dt.to_period("Q")
sp_q = sp_df.groupby(["ticker", "q"], as_index=False)["resid_sp"].mean()

fu_df = pd.DataFrame(fu["resid_key"], columns=["ticker", "q"])
fu_df["resid_fu"] = fu["resid_vals"]

merged = sp_q.merge(fu_df, on=["ticker", "q"], how="inner")
n_sp_tickers = sp_df["ticker"].nunique()
n_fu_tickers = fu_df["ticker"].nunique()
n_common_tickers = len(set(sp_df["ticker"]) & set(fu_df["ticker"]))

print(f"SP500 distinct tickers = {n_sp_tickers}, FullUniverse distinct tickers = {n_fu_tickers}, "
      f"common tickers = {n_common_tickers}")
print(f"alignable (ticker,quarter) observations after inner-merge = {len(merged):,}")

if len(merged) >= 10:
    corr = merged["resid_sp"].corr(merged["resid_fu"])
    print(f"corr(resid_SP[quarterly-avg], resid_FU) on alignable obs = {corr:+.6f}  (N={len(merged):,})")
else:
    print("Fewer than 10 alignable observations — correlation not meaningfully estimable; "
          "stating this directly as the answer: the two residual series share essentially "
          "no common (ticker, quarter) cells at this alignment.")

print()
print("Note on interpretation: SP500 residuals come from a MONTHLY regression with a")
print("winsorized/z-scored ACCOUNTING-based ΔH (dH_gpm_z) and price-panel ΔS_z; the")
print("full-universe residuals come from a QUARTERLY regression with a differently")
print("constructed delta_h_z/delta_s_z on the survivorship-corrected SF1 panel. Even for")
print("the same firm-quarter, these are residuals from two different models on two")
print("different underlying return/characteristic constructions, so a low correlation is")
print("the expected outcome under 'independent-but-genuinely-overlapping-firms', and a")
print("high correlation would itself be the anomaly worth investigating.")
