"""REV4_E2_known_premium_biased_panel.py — decisive experiment from the fourth
external review (score 4/10): re-runs the known-premium validation (size,
book-to-market, momentum, gross profitability) on the BIASED (462-name
current S&P 500) panel at the SAME quarterly spacing used for the corrected
R18 panel in SPEC_G2_review2_experiments.py (GP t=+8.19, B/M t=+2.65, size
t=+0.33, momentum t=+0.44 there).

If size and momentum ALSO fail to be recovered here, on the biased panel, at
identical quarterly spacing, their failure in the corrected panel is a
construction/frequency artifact rather than evidence the corrected panel is
underpowered specifically. If they survive here and die only in the
corrected panel, the reviewer's instrument-power criticism of the corrected
panel's IVOL null stands.

Uses ../data/M1_sp500_quarterly_panel.parquet (harmonized quarterly biased
panel, same schema as the R18 corrected panel: ticker, q, ret, delta_s,
dH_gpm, T, q_ord, ret_next, delta_s_z, delta_h_z, T_delta_s -- 457 tickers,
1995Q3-2023Q4, 114 quarters; this is the exact panel D4_crosspanel_table.py
uses for the harmonized-quarterly biased-side comparison, t(dS)=+4.39).
Characteristics built identically to SPEC_G2 (SF1 ARQ, negsize/bm/gp_at,
4-quarter contiguous skip-most-recent-quarter momentum), same NW-4
unconditional FM spec.

Outputs: results/revision/REV4_E2_known_premium_biased_panel.txt
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision"
os.makedirs(OUT, exist_ok=True)

LOG = []
def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    LOG.append(line)


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
    for d, grp in panel.groupby(date_col):
        sub = grp[[y_col] + x_cols].dropna()
        if len(sub) < max(min_cs, len(x_cols) + 2):
            continue
        X = sm.add_constant(sub[x_cols], has_constant="add")
        res = sm.OLS(sub[y_col], X).fit()
        coefs.append(res.params[x_cols].rename(d))
    if not coefs:
        return {}, pd.DataFrame()
    cdf = pd.DataFrame(coefs)
    out = {}
    for col in x_cols:
        s = cdf[col].dropna()
        n = len(s)
        mean_ = s.mean()
        gamma0 = (s ** 2).mean() - mean_ ** 2
        var = gamma0
        for l in range(1, min(lags + 1, n)):
            g = ((s.iloc[l:].values - mean_) * (s.iloc[:-l].values - mean_)).mean()
            var += 2 * (1 - l / (lags + 1)) * g
        se = np.sqrt(max(var, 1e-30) / n)
        out[col] = dict(coef=mean_, se=se, t=mean_ / se, n=n)
    return out, cdf


say("=" * 100)
say("REV4 E2 — KNOWN-PREMIUM VALIDATION ON THE BIASED (S&P 500) PANEL, MATCHED QUARTERLY SPACING")
say("=" * 100)

panel = pd.read_parquet(f"{DATA}/M1_sp500_quarterly_panel.parquet")
say(f"\nBiased S&P 500 quarterly panel: N={len(panel):,}  tickers={panel['ticker'].nunique():,}  "
    f"quarters={panel['q'].nunique()}  range={panel['q'].min()}..{panel['q'].max()}")
if "q_ord" not in panel.columns or panel["q_ord"].isna().all():
    panel["q_ord"] = panel["q"].apply(lambda p: p.ordinal)

# ── control replication: unconditional IVOL premium on this panel ───────────
pf0 = panel.dropna(subset=["ret_next", "delta_s_z"])
fm0, _ = fama_macbeth_nw(pf0, "ret_next", ["delta_s_z"])
r0 = fm0.get("delta_s_z", dict(coef=np.nan, se=np.nan, t=np.nan, n=0))
say(f"\nControl check -- unconditional IVOL premium on this panel: "
    f"t={r0['t']:+.4f}  beta={r0['coef']:+.6f}  N={len(pf0):,}  Tq={r0['n']}  "
    f"(should match D4_crosspanel_table.py's harmonized-quarterly biased-side t~+4.39)")

# ── characteristics (identical construction to SPEC_G2_review2_experiments.py) ──
sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                      columns=["ticker", "dimension", "calendardate", "datekey",
                               "marketcap", "equity", "assets", "revenue", "grossmargin"])
c = sf1[sf1["dimension"] == "ARQ"].copy()
c["calendardate"] = pd.to_datetime(c["calendardate"], errors="coerce")
c = c.dropna(subset=["calendardate"])
c["q"] = c["calendardate"].dt.to_period("Q")
c = c.sort_values(["ticker", "calendardate"]).drop_duplicates(["ticker", "q"], keep="last")

c["negsize"] = np.where(c["marketcap"] > 0, -np.log(c["marketcap"]), np.nan)
c["bm"] = np.where((c["marketcap"] > 0), c["equity"] / c["marketcap"], np.nan)
c["gp_at"] = np.where((c["assets"] > 0) & c["grossmargin"].notna() & c["revenue"].notna(),
                       (c["grossmargin"] * c["revenue"]) / c["assets"], np.nan)
c = c[["ticker", "q", "negsize", "bm", "gp_at"]]

panel2 = panel.merge(c, on=["ticker", "q"], how="left")
say(f"\nCharacteristic merge coverage: negsize={panel2['negsize'].notna().mean():.1%}  "
    f"bm={panel2['bm'].notna().mean():.1%}  gp_at={panel2['gp_at'].notna().mean():.1%}")

# momentum: cumulative return over 4 quarters ending 1 quarter before current
# (quarterly analog of 12-1 skip-most-recent-month), identical to SPEC_G2
panel2 = panel2.sort_values(["ticker", "q_ord"])
g = panel2.groupby("ticker")
ret_lags = {k: g["ret"].shift(k) for k in range(1, 5)}
qord_lags = {k: g["q_ord"].shift(k) for k in range(1, 5)}
contig = pd.Series(True, index=panel2.index)
cumret = pd.Series(1.0, index=panel2.index)
any_missing = pd.Series(False, index=panel2.index)
for k in range(1, 5):
    gap_ok = (panel2["q_ord"] - qord_lags[k]) == k
    contig &= gap_ok.fillna(False)
    any_missing |= ret_lags[k].isna()
    cumret *= (1 + ret_lags[k].fillna(0))
panel2["mom"] = np.where(contig & (~any_missing), cumret - 1.0, np.nan)
say(f"Momentum coverage (4 contiguous prior quarters): {panel2['mom'].notna().sum():,} obs")

for col in ["negsize", "bm", "gp_at", "mom"]:
    panel2[col + "_z"] = cs_wz(panel2, col)

say("\n" + "-" * 100)
say("RESULTS")
say("-" * 100)
results2 = {}
for label, col, ann_mult, corrected_ref in [
    ("Size (-log mktcap)", "negsize_z", 4, "corrected-panel reference: t=+0.33"),
    ("Book-to-market", "bm_z", 4, "corrected-panel reference: t=+2.65"),
    ("Momentum (4q, skip-0)", "mom_z", 4, "corrected-panel reference: t=+0.44"),
    ("Gross profitability", "gp_at_z", 4, "corrected-panel reference: t=+8.19"),
]:
    pf = panel2.dropna(subset=["ret_next", col])
    fm, _ = fama_macbeth_nw(pf, "ret_next", [col])
    r = fm.get(col, dict(coef=np.nan, se=np.nan, t=np.nan, n=0))
    ann = r["coef"] * ann_mult * 100
    say(f"  {label:24} beta={r['coef']:+.6f}  t={r['t']:+.3f}  N={len(pf):,}  T_q={r['n']}  "
        f"ann={ann:+.2f}%/yr   [{corrected_ref}]")
    results2[label] = r

say("\n" + "-" * 100)
say("VERDICT")
say("-" * 100)
size_recovered = abs(results2["Size (-log mktcap)"]["t"]) >= 2.0
mom_recovered = abs(results2["Momentum (4q, skip-0)"]["t"]) >= 2.0
say(f"Size |t|>=2 on biased panel: {size_recovered}  (t={results2['Size (-log mktcap)']['t']:+.3f} "
    f"vs corrected-panel t=+0.33)")
say(f"Momentum |t|>=2 on biased panel: {mom_recovered}  (t={results2['Momentum (4q, skip-0)']['t']:+.3f} "
    f"vs corrected-panel t=+0.44)")
if not size_recovered and not mom_recovered:
    say("\nBoth size and momentum ALSO fail to clear conventional significance on the biased panel at")
    say("this same quarterly spacing -- the corrected panel's failure to recover them is NOT specific")
    say("to its survivorship correction or universe breadth; it is a property of the quarterly,")
    say("filing-date-aligned construction itself (both panels share this construction here). This")
    say("weakens, though does not eliminate, the reviewer's instrument-power objection to the")
    say("corrected panel's IVOL null, since the same low-power construction affects the biased panel too.")
elif size_recovered and mom_recovered:
    say("\nBoth size and momentum ARE recovered on the biased panel at this same quarterly spacing --")
    say("the corrected panel's failure to recover them is therefore NOT a generic quarterly-spacing")
    say("artifact. This makes the reviewer's instrument-power objection to the corrected panel's IVOL")
    say("null substantially stronger: something specific to the corrected panel (breadth, survivorship,")
    say("measurement quality) degrades power on return-based signals, in exactly the same class as IVOL.")
else:
    say("\nMixed: one of size/momentum is recovered on the biased panel and the other is not. Report the")
    say("split explicitly rather than treating the instrument-power question as resolved either way.")

with open(f"{OUT}/REV4_E2_known_premium_biased_panel.txt", "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\n[written] {OUT}/REV4_E2_known_premium_biased_panel.txt")
