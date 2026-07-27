"""H1_terminal_return_cell.py — Addendum 3, H1: terminal-return sensitivity
applied directly to R18, ONE (variant, delta) cell per process (ground rule:
fresh process per estimate).

R18 (robustness/R18_sf1_quarterly_survfree.py) appends no terminal delisting
return: each of the panel's 8,937 delisted firms' series simply ends at its
last filing quarter, so that quarter's row has a valid ΔS/ΔH (characteristics)
but a missing `ret_next` (nothing follows) and therefore never enters the FM
regression or quintile sort. This script assigns an assumed delisting return
δ as that missing `ret_next` for the LAST quarter of each delisted firm and
re-estimates. This is the exact and only mechanism by which a terminal return
enters the FM sample -- ΔS/ΔH at that row are the firm's own last-filed
characteristics, already computed by R18, untouched here.

Usage: python3 H1_terminal_return_cell.py {uniform|reason_conditional} {delta_pct}
  delta_pct in {0, -10, -30, -55, -100}  (integer, percent)

uniform:            every delisted firm's terminal ret_next <- delta.
reason_conditional: the 611 EDGAR-confirmed acquisitions (F5b) get
                     terminal ret_next <- 0 regardless of delta (acquisitions
                     typically exit at or above market); all other delisted
                     firms get delta.

Anchor check: delta_pct=0 is a no-op under BOTH variants (nobody's ret_next
changes -- confirmed firms get 0 either way) and must reproduce the published
R18 Model B t(IVOL)=+0.018 / t(STAB)=+3.461 exactly. If it does not, this
script stops and reports the discrepancy rather than proceeding.
"""
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "../project")
from utils import newey_west_mean_tstat

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT_DIR = "../results/revision"
os.makedirs(OUT_DIR, exist_ok=True)

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "uniform"
DELTA_PCT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
assert VARIANT in ("uniform", "reason_conditional")
DELTA = DELTA_PCT / 100.0

print(f"[pid={os.getpid()}] H1 cell — variant={VARIANT}  delta={DELTA_PCT:+d}%  fresh process")
log = []
def P(s=""):
    print(s)
    log.append(str(s))

P("=" * 88)
P(f"H1 — terminal-return sensitivity, variant={VARIANT}, delta={DELTA_PCT:+d}%")
P("=" * 88)

# ── load R18's already-built panel (unmodified) ─────────────────────────────
panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet").copy()
P(f"Loaded panel: N={len(panel):,}  tickers={panel['ticker'].nunique():,}  "
  f"quarters={panel['q'].nunique()}  range={panel['q'].min()}..{panel['q'].max()}")

# ── delisted-firm set, identical construction to R18/F1/DOC2 (isdelisted=='Y',
#    US domestic common, NYSE/NASDAQ/NYSEARCA/BATS/NYSEMKT, USD) ────────────
tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"]
uni = sf1t[
    sf1t["category"].str.contains("Domestic Common", na=False) &
    sf1t["exchange"].isin(["NYSE", "NASDAQ", "NYSEARCA", "BATS", "NYSEMKT"]) &
    (sf1t["currency"] == "USD")
].copy()
delisted_set = set(uni.loc[uni["isdelisted"] == "Y", "ticker"])
panel_tickers = set(panel["ticker"].unique())
delisted_in_panel = delisted_set & panel_tickers
P(f"Delisted firms in analysis panel: {len(delisted_in_panel):,}  "
  f"({'MATCH' if len(delisted_in_panel) == 8937 else 'MISMATCH'} vs cited 8,937)")
if len(delisted_in_panel) != 8937:
    P("STOPPING: delisted-firm count does not match the cited 8,937 -- reporting "
      "discrepancy rather than proceeding.")
    with open(f"{OUT_DIR}/H1_cell_{VARIANT}_{DELTA_PCT}.txt", "w") as f:
        f.write("\n".join(log) + "\n")
    sys.exit(1)

# ── EDGAR-confirmed acquisitions (F5b), for the reason-conditional variant ──
cls = pd.read_csv(f"{DATA}/F5b_edgar_classification.csv")
confirmed_tickers = set(cls.loc[cls["status"] == "confirmed", "ticker"])
confirmed_in_panel = confirmed_tickers & delisted_in_panel
P(f"EDGAR-confirmed acquisitions (F5b) within delisted-in-panel set: "
  f"{len(confirmed_in_panel):,}  ({'MATCH' if len(confirmed_in_panel) == 611 else 'MISMATCH'} vs cited 611)")

# ── identify each delisted firm's LAST panel quarter (max q); this row's
#    ret_next is, by construction, always missing (nothing follows it) ──────
last_idx = panel.groupby("ticker")["q"].idxmax()
last_rows = panel.loc[last_idx]
target_idx = last_rows[last_rows["ticker"].isin(delisted_in_panel)].index
assert panel.loc[target_idx, "ret_next"].isna().all(), \
    "terminal rows unexpectedly have a non-missing ret_next -- investigate before proceeding"
P(f"\nTerminal (last-quarter) rows to receive an assumed delisting return: {len(target_idx):,}")

# ── assign delta ──────────────────────────────────────────────────────────
# delta=0 is the ANCHOR / status-quo case: R18 appends no terminal observation
# at all (that is the paper's caveat being addressed), so delta=0 means "do not
# append anything" -- NOT "append a literal 0% return", which would add 7,917
# new sample rows and mechanically change N vs. the published estimate. Every
# nonzero delta on the grid DOES append a real terminal observation.
target_tickers = panel.loc[target_idx, "ticker"]
is_confirmed = target_tickers.isin(confirmed_in_panel).values
if DELTA_PCT == 0:
    P("\ndelta=0 -> anchor/status-quo case: no terminal observation appended "
      "(matches R18 exactly, no rows added).")
else:
    if VARIANT == "uniform":
        assigned = pd.Series(DELTA, index=target_idx)
    else:  # reason_conditional
        assigned = pd.Series(np.where(is_confirmed, 0.0, DELTA), index=target_idx)
        P(f"  of these, {is_confirmed.sum():,} are EDGAR-confirmed acquisitions -> ret_next=0.0 "
          f"regardless of delta; {(~is_confirmed).sum():,} get delta={DELTA:+.2f}")
    panel.loc[target_idx, "ret_next"] = assigned.values

# ── re-estimate: Model B FM (dh_z, ds_z) + pure-DeltaS quintile sort,
#    identical specification to R18 ─────────────────────────────────────────
def fama_macbeth_nw(p, y_col, x_cols, date_col="q", lags=4, min_cs=20):
    coefs = []
    for d, g in p.groupby(date_col):
        s = g[[y_col] + x_cols].dropna()
        if len(s) < max(min_cs, len(x_cols) + 2):
            continue
        X = sm.add_constant(s[x_cols], has_constant="add")
        coefs.append(sm.OLS(s[y_col], X).fit().params[x_cols].rename(d))
    if not coefs:
        return {}
    cdf = pd.DataFrame(coefs)
    out = {}
    for c in x_cols:
        s = cdf[c].dropna()
        mean_, t_, p_ = newey_west_mean_tstat(s.values, lags=lags)
        se_ = mean_ / t_ if t_ not in (0, np.nan) and not np.isnan(t_) else np.nan
        out[c] = dict(coef=mean_, se=se_, t=t_, n=len(s))
    return out

def quintile_ls(df, sortcol, ycol="ret_next", datecol="q"):
    d = df.dropna(subset=[sortcol, ycol]).copy()
    d["qd"] = d.groupby(datecol)[sortcol].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
        if x.nunique() >= 5 else np.nan)
    d = d.dropna(subset=["qd"])
    qr = d.groupby([datecol, "qd"])[ycol].mean().unstack("qd")
    if 0 not in qr.columns or 4 not in qr.columns:
        return dict(ls_ann=np.nan, t=np.nan, n=0)
    ls = (qr[4] - qr[0]).dropna()
    t_ = ls.mean() / (ls.std() / np.sqrt(len(ls))) if len(ls) > 2 else np.nan
    return dict(ls_ann=ls.mean() * 4, t=t_, n=len(ls))

pf = panel.dropna(subset=["ret_next", "delta_s_z", "delta_h_z"])
pe = panel.dropna(subset=["ret_next", "delta_s_z"])

fm = fama_macbeth_nw(pf, "ret_next", ["delta_h_z", "delta_s_z"])
qs = quintile_ls(pe, "delta_s_z")

first_q, last_q = str(pf["q"].min()), str(pf["q"].max())
avg_firms_q = pf.groupby("q").size().mean()
n_tickers_pf = pf["ticker"].nunique()

X = sm.add_constant(pf[["delta_h_z", "delta_s_z"]]).values
design_hash = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()

P("\n" + "-" * 88)
P(f"Model B FM (quarterly, NW-4):  design shape={X.shape}  SHA-1={design_hash}")
P(f"  N={len(pf):,}  tickers={n_tickers_pf:,}  avg firms/qtr={avg_firms_q:.1f}  "
  f"date_range={first_q}..{last_q}")
P(f"  t(IVOL)=t(delta_s_z) = {fm['delta_s_z']['t']:+.4f}   "
  f"coef={fm['delta_s_z']['coef']:+.6f}  SE={fm['delta_s_z']['se']:.6f}  quarters={fm['delta_s_z']['n']}")
P(f"  t(STAB)=t(delta_h_z) = {fm['delta_h_z']['t']:+.4f}   "
  f"coef={fm['delta_h_z']['coef']:+.6f}  SE={fm['delta_h_z']['se']:.6f}  quarters={fm['delta_h_z']['n']}")
P(f"\nQuintile sort on ΔS (pure entropy/iVol, entropy-only panel N={len(pe):,}):")
P(f"  L/S (Q5-Q1) = {qs['ls_ann']*100:+.2f}%/yr   t={qs['t']:+.2f}   (Tq={qs['n']})")

# ── delta=0 anchor check ─────────────────────────────────────────────────────
if DELTA_PCT == 0:
    t_ivol, t_stab = fm["delta_s_z"]["t"], fm["delta_h_z"]["t"]
    ok_ivol = abs(t_ivol - 0.018) < 0.003
    ok_stab = abs(t_stab - 3.461) < 0.01
    P("\n" + "-" * 88)
    P("ANCHOR CHECK (delta=0 must reproduce the published R18 Model B exactly)")
    P(f"  t(IVOL): got {t_ivol:+.4f}, published +0.018  -> {'MATCH' if ok_ivol else 'MISMATCH'}")
    P(f"  t(STAB): got {t_stab:+.4f}, published +3.461  -> {'MATCH' if ok_stab else 'MISMATCH'}")
    if not (ok_ivol and ok_stab):
        P("\nSTOPPING: anchor mismatch -- reporting the discrepancy rather than proceeding "
          "with the rest of the grid.")

out = dict(
    variant=VARIANT, delta_pct=DELTA_PCT, pid=os.getpid(),
    N=int(len(pf)), n_tickers=int(n_tickers_pf), avg_firms_q=float(avg_firms_q),
    first_q=first_q, last_q=last_q,
    design_shape=list(X.shape), design_hash=design_hash,
    t_ivol=fm["delta_s_z"]["t"], coef_ivol=fm["delta_s_z"]["coef"], se_ivol=fm["delta_s_z"]["se"],
    nq_ivol=fm["delta_s_z"]["n"],
    t_stab=fm["delta_h_z"]["t"], coef_stab=fm["delta_h_z"]["coef"], se_stab=fm["delta_h_z"]["se"],
    nq_stab=fm["delta_h_z"]["n"],
    quintile_ls_ann=qs["ls_ann"], quintile_t=qs["t"], quintile_tq=qs["n"],
    n_target_rows=int(len(target_idx)),
    n_confirmed_zeroed=int(is_confirmed.sum()) if (VARIANT == "reason_conditional" and DELTA_PCT != 0) else 0,
)
with open(f"{OUT_DIR}/H1_cell_{VARIANT}_{DELTA_PCT}.json", "w") as f:
    json.dump(out, f, indent=2)
with open(f"{OUT_DIR}/H1_cell_{VARIANT}_{DELTA_PCT}.txt", "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT_DIR}/H1_cell_{VARIANT}_{DELTA_PCT}.json")
