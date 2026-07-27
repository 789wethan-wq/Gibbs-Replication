"""F5c — sensitivity check: does the entropy-premium collapse survive
excluding EDGAR-confirmed M&A/successor firms from the delisted population?

MW4's underlying concern: exit type (failure vs. acquisition) is not
characterized, and if "delisted" implicitly conflates healthy acquisitions
with genuine failures, the collapse could be an artifact of that conflation
rather than a real entropy-premium result. This tests it directly: rerun
the R18 corrected-panel FM collapse (a) on the full panel as published,
(b) EXCLUDING all EDGAR-confirmed-successor (i.e., positively identified as
acquired/reorganized, not failed) firms, and (c) on the confirmed-M&A firms
ALONE, to see whether their own entropy-return relationship looks different
from the rest of the panel.

Requires data/F5b_edgar_classification.csv (written by
F5b_full_population_edgar.py) -- run that first.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
sys.path.insert(0, "../project")
from utils import newey_west_mean_tstat

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/F5c_edgar_sensitivity_rerun.txt"

print(f"[pid={os.getpid()}] F5c — fresh process")
log = []
def P(s=""):
    print(s)
    log.append(str(s))


def cs_wz(df, col, date_col="q", pct=0.01):
    def _f(x):
        x2 = x.dropna()
        if len(x2) < 5:
            return pd.Series(np.nan, index=x.index)
        lo, hi = x2.quantile(pct), x2.quantile(1 - pct)
        xc = x.clip(lo, hi)
        s = xc.std()
        if s < 1e-10:
            return pd.Series(np.nan, index=x.index)
        return (xc - xc.mean()) / s
    return df.groupby(date_col)[col].transform(_f)


def fama_macbeth_nw(panel, y_col, x_cols, date_col="q", lags=4, min_cs=20):
    coefs = []
    for d, g in panel.groupby(date_col):
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
        se_ = mean_ / t_ if t_ != 0 else np.nan
        out[c] = dict(coef=mean_, se=se_, t=t_, n=len(s))
    return out


P("="*88)
P("F5c — Entropy-premium collapse: sensitivity to EDGAR-confirmed M&A exclusion")
P("="*88)

cls = pd.read_csv(f"{DATA}/F5b_edgar_classification.csv")
P(f"Loaded EDGAR classification: {len(cls):,} delisted firms classified")
P(cls["status"].value_counts().to_string())
confirmed_tickers = set(cls.loc[cls["status"] == "confirmed", "ticker"])
P(f"\nEDGAR-confirmed successor/M&A tickers: {len(confirmed_tickers):,}")

panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
panel["ds_z"] = cs_wz(panel, "delta_s")
panel["dh_z"] = cs_wz(panel, "dH_gpm")
pf_all = panel.dropna(subset=["ret_next", "ds_z", "dh_z"])


def run_and_report(pf, label):
    fm = fama_macbeth_nw(pf, "ret_next", ["dh_z", "ds_z"])
    if "ds_z" not in fm:
        P(f"\n[{label}] INFEASIBLE")
        return None
    avg_n = pf.groupby("q").size().mean()
    first_q, last_q = str(pf["q"].min()), str(pf["q"].max())
    P(f"\n[{label}]")
    P(f"  N={len(pf):,}  tickers={pf['ticker'].nunique():,}  avg firms/qtr={avg_n:.1f}  "
      f"date_range={first_q}..{last_q}")
    P(f"  t(dS)={fm['ds_z']['t']:+.4f}  coef(dS)={fm['ds_z']['coef']:+.6f}  SE(dS)={fm['ds_z']['se']:.6f}  "
      f"quarters={fm['ds_z']['n']}")
    P(f"  t(dH)={fm['dh_z']['t']:+.4f}  coef(dH)={fm['dh_z']['coef']:+.6f}  SE(dH)={fm['dh_z']['se']:.6f}")
    return fm


P("\n" + "-"*88)
r_full = run_and_report(pf_all, "(a) FULL panel, as published")

pf_excl = pf_all[~pf_all["ticker"].isin(confirmed_tickers)]
r_excl = run_and_report(pf_excl, "(b) EXCLUDING EDGAR-confirmed M&A/successor firms")

pf_confirmed_only_delisted = pf_all[pf_all["ticker"].isin(confirmed_tickers)]
r_conf = run_and_report(pf_confirmed_only_delisted, "(c) EDGAR-confirmed M&A firms ONLY (their own entropy-return relationship)")

P("\n" + "="*88)
P("F5c SUMMARY")
P("="*88)
if r_full and r_excl:
    P(f"{'Spec':45}{'t(dS)':>9}{'t(dH)':>9}{'N':>10}")
    P(f"{'(a) Full panel':45}{r_full['ds_z']['t']:>+9.3f}{r_full['dh_z']['t']:>+9.3f}{len(pf_all):>10,}")
    P(f"{'(b) Excl. EDGAR-confirmed M&A':45}{r_excl['ds_z']['t']:>+9.3f}{r_excl['dh_z']['t']:>+9.3f}{len(pf_excl):>10,}")
    if r_conf:
        P(f"{'(c) EDGAR-confirmed M&A only':45}{r_conf['ds_z']['t']:>+9.3f}{r_conf['dh_z']['t']:>+9.3f}{len(pf_confirmed_only_delisted):>10,}")

    diff = abs(r_full['ds_z']['t'] - r_excl['ds_z']['t'])
    P(f"\n|t(dS) full - t(dS) excl-confirmed-M&A| = {diff:.3f}")
    if diff < 1.0 and abs(r_excl['ds_z']['t']) < 2.0:
        P("\nVERDICT: the collapse is STABLE to excluding EDGAR-confirmed M&A/successor firms")
        P("from the delisted population. The near-zero entropy premium is not an artifact of")
        P("conflating healthy acquisitions with genuine failures under the blanket 'delisted'")
        P("label -- removing the firms we can positively confirm were acquired/reorganized")
        P("(rather than failed) leaves the collapse intact.")
    else:
        P("\nVERDICT: excluding EDGAR-confirmed M&A firms MOVES t(dS) meaningfully -- this is")
        P("worth reporting directly rather than assuming stability; the collapse's robustness")
        P("to the exit-type conflation concern is not as clean as hoped.")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
