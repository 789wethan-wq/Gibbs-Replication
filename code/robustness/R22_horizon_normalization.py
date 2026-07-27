"""R22 -- Horizon normalization / irregular spacing audit (BLOCKING).

Sec 3.1 states returns in R18 are computed between consecutive DATEKEY prices,
"which spaces observations irregularly." The R18 construction (see
robustness/R18_sf1_quarterly_survfree.py step 2) buckets ARQ filings into
calendar quarters via `calendardate` and requires the QUARTER-ORDINAL gap
between consecutive kept rows to equal 1 -- but the underlying `price` field
is observed as of `datekey` (the actual filing date), which trails
`calendardate` by a filing delay (SF1: mean 53.6 days, std 33.0). Filing
delay is distress-correlated, and late filers skew high-DeltaS, so unequal
holding periods are not classical noise. This script measures the actual
day-count spacing, its relationship to DeltaS, and re-estimates Model B on a
horizon-normalized return.

Report as run. No tuning toward the manuscript value.

Outputs: robustness/outputs/R22_horizon_normalization_results.txt
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

LOG = []
def say(*a):
    line = " ".join(str(x) for x in a)
    print(line); LOG.append(line)

def fama_macbeth_nw(panel, y_col, x_cols, date_col="q", lags=4, min_cs=20):
    coefs = []
    for d, grp in panel.groupby(date_col):
        sub = grp[[y_col] + x_cols].dropna()
        if len(sub) < max(min_cs, len(x_cols) + 2): continue
        X = sm.add_constant(sub[x_cols], has_constant="add")
        res = sm.OLS(sub[y_col], X).fit()
        coefs.append(res.params[x_cols].rename(d))
    if not coefs: return {}, pd.DataFrame()
    cdf = pd.DataFrame(coefs)
    out = {}
    for col in x_cols:
        s = cdf[col].dropna(); n = len(s); mean_ = s.mean()
        var = (s**2).mean() - mean_**2
        for l in range(1, min(lags + 1, n)):
            g = ((s.iloc[l:].values - mean_) * (s.iloc[:-l].values - mean_)).mean()
            var += 2 * (1 - l / (lags + 1)) * g
        se = np.sqrt(max(var, 1e-30) / n)
        out[col] = (mean_, mean_ / se, n)
    return out, cdf

def shw(tag, d, k):
    m, t, n = d.get(k, (np.nan, np.nan, 0))
    say(f"  {tag:34} beta={m:+.5f}  t={t:+.2f}  (Tq={n})")

# ═══════════════════════════════════════════════════════════════════════════
say("=" * 78); say("R22 -- HORIZON NORMALIZATION / IRREGULAR SPACING AUDIT"); say("=" * 78)

# ── rebuild the R18 price/quarter panel, keeping datekey ─────────────────────
say("\nRebuilding R18 price panel with datekey retained (universe/logic identical")
say("to robustness/R18_sf1_quarterly_survfree.py steps 1-2)...")

tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"]
uni = sf1t[
    sf1t["category"].str.contains("Domestic Common", na=False) &
    sf1t["exchange"].isin(["NYSE","NASDAQ","NYSEARCA","BATS","NYSEMKT"]) &
    (sf1t["currency"] == "USD")
].copy()
uni_set = set(uni["ticker"])

sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                      columns=["ticker","dimension","calendardate","datekey","price","dps"])
arq = sf1[(sf1["dimension"] == "ARQ") & sf1["ticker"].isin(uni_set)].copy()
arq["calendardate"] = pd.to_datetime(arq["calendardate"], errors="coerce")
arq["datekey"] = pd.to_datetime(arq["datekey"], errors="coerce")
arq = arq.dropna(subset=["calendardate","price"])
arq = arq[arq["price"] > 0]
arq["q"] = arq["calendardate"].dt.to_period("Q")
arq = (arq.sort_values(["ticker","calendardate"])
          .drop_duplicates(["ticker","q"], keep="last"))
arq = arq.sort_values(["ticker","q"])
arq["q_ord"] = arq["q"].apply(lambda p: p.ordinal)
arq["price_prev"] = arq.groupby("ticker")["price"].shift(1)
arq["datekey_prev"] = arq.groupby("ticker")["datekey"].shift(1)
arq["gap_qord"] = arq["q_ord"] - arq.groupby("ticker")["q_ord"].shift(1)
arq["gap_days"] = (arq["datekey"] - arq["datekey_prev"]).dt.days
ret_px = arq["price"] / arq["price_prev"] - 1.0
div_q = (arq["dps"].fillna(0) / 4.0) / arq["price_prev"]
arq["ret_raw"] = np.where(arq["gap_qord"] == 1, ret_px + div_q, np.nan)
arq["ret"] = arq.groupby("q")["ret_raw"].transform(
    lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)) if x.notna().sum() >= 5 else x)
cnt = arq.groupby("ticker")["ret"].transform("size")
say(f"  ARQ rows (all, pre-8q filter): {len(arq):,}  tickers={arq['ticker'].nunique():,}")

# ── load the ACTUAL R18 panel (ret_next already computed there) ─────────────
panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
say(f"  R18 saved panel: N={len(panel):,}")

# merge in gap_days (backward, i.e. the gap that produced THIS row's `ret`)
panel = panel.merge(arq[["ticker","q","gap_days","gap_qord","datekey"]], on=["ticker","q"], how="left")

# ── forward gap: the gap between this row's datekey and the NEXT row's datekey ─
arq_fwd = arq[["ticker","q","q_ord","datekey"]].sort_values(["ticker","q_ord"])
arq_fwd["datekey_next"] = arq_fwd.groupby("ticker")["datekey"].shift(-1)
arq_fwd["q_ord_next"] = arq_fwd.groupby("ticker")["q_ord"].shift(-1)
arq_fwd["gap_days_fwd"] = (arq_fwd["datekey_next"] - arq_fwd["datekey"]).dt.days
arq_fwd["gap_qord_fwd"] = arq_fwd["q_ord_next"] - arq_fwd["q_ord"]
arq_fwd["has_next_row"] = arq_fwd["datekey_next"].notna()
panel = panel.merge(arq_fwd[["ticker","q","gap_days_fwd","gap_qord_fwd","has_next_row"]],
                     on=["ticker","q"], how="left")

# ═══════════════════════════════════════════════════════════════════════════
say("\n" + "-" * 78); say("R22.1 -- GAP DIAGNOSTICS"); say("-" * 78)

say("\n(a) Distribution of inter-filing gaps in days (backward, gap that produced `ret`):")
gd = panel["gap_days"].dropna()
say(f"    N={len(gd):,}  mean={gd.mean():.1f}  median={gd.median():.1f}  "
    f"p10={gd.quantile(0.10):.1f}  p90={gd.quantile(0.90):.1f}  "
    f"min={gd.min():.0f}  max={gd.max():.0f}")
say("    Full histogram (10-day bins, days 0-200, tail beyond 200 pooled):")
bins = list(range(0, 210, 10)) + [10**6]
labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-2)] + ["200+"]
hist = pd.cut(gd.clip(lower=0), bins=bins, labels=labels, right=False).value_counts().sort_index()
for lbl, c in hist.items():
    say(f"      {lbl:>10}: {c:>8,}  ({c/len(gd):>6.1%})")

say("\n(b) Distribution of FORWARD gap_days (the gap underlying ret_next), by ΔS quintile:")
p2 = panel.dropna(subset=["delta_s_z"]).copy()
p2["ds_q"] = p2.groupby("q")["delta_s_z"].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") if x.nunique() >= 5 else np.nan)
say(f"    {'DS quintile':>12} {'N':>8} {'mean_gap_fwd':>13} {'median_gap_fwd':>15} "
    f"{'p10':>6} {'p90':>6}")
for qd in sorted(p2["ds_q"].dropna().unique()):
    sub = p2.loc[p2["ds_q"] == qd, "gap_days_fwd"].dropna()
    if len(sub) == 0: continue
    say(f"    Q{int(qd)+1:>11} {len(sub):>8,} {sub.mean():>13.1f} {sub.median():>15.1f} "
        f"{sub.quantile(0.10):>6.0f} {sub.quantile(0.90):>6.0f}")

say("\n(c) Fraction of observations DROPPED by the consecutive-quarter rule (gap_qord_fwd != 1),")
say("    by ΔS quintile, split into TERMINAL (no next row exists for that ticker at all) vs")
say("    INTERIOR (a next row exists later, but skips >=1 quarter):")
p3 = p2.copy()
p3["dropped"] = (p3["gap_qord_fwd"] != 1) | p3["gap_qord_fwd"].isna()
p3["terminal"] = p3["dropped"] & (~p3["has_next_row"].fillna(False))
p3["interior"] = p3["dropped"] & (p3["has_next_row"].fillna(False))
say(f"    {'DS quintile':>12} {'N':>8} {'%dropped':>9} {'%terminal':>10} {'%interior':>10}")
for qd in sorted(p3["ds_q"].dropna().unique()):
    sub = p3[p3["ds_q"] == qd]
    n = len(sub)
    say(f"    Q{int(qd)+1:>11} {n:>8,} {sub['dropped'].mean():>8.1%} "
        f"{sub['terminal'].mean():>9.1%} {sub['interior'].mean():>9.1%}")
n_term = p3["terminal"].sum(); n_int = p3["interior"].sum()
say(f"\n    Overall: dropped={p3['dropped'].sum():,} ({p3['dropped'].mean():.1%})  "
    f"terminal={n_term:,} ({n_term/len(p3):.1%})  interior={n_int:,} ({n_int/len(p3):.1%})")

# ═══════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 78); say("R22.2 -- HORIZON-NORMALIZED FAMA-MACBETH"); say("=" * 78)
say("\nMethod: r_norm = ret_next * (91.25 / gap_days_fwd), i.e. rescale the realized")
say("forward return to a common 91.25-day (1-quarter) holding period, linear in gap")
say("length (matches the additive/simple-return convention used throughout the R18")
say("pipeline; log-return scaling was not used since R18 uses simple, not log, returns).")

pn = panel.merge(arq_fwd[["ticker","q","gap_days_fwd"]], on=["ticker","q"], how="left", suffixes=("","_dup"))
pn = pn.loc[:, ~pn.columns.duplicated()]
pn["ret_next_norm"] = pn["ret_next"] * (91.25 / pn["gap_days_fwd"])
# guard against pathological scaling on very short gaps
pn.loc[pn["gap_days_fwd"] < 20, "ret_next_norm"] = np.nan

say(f"\ngap_days_fwd coverage in headline sample: "
    f"{pn.dropna(subset=['ret_next','delta_h_z','delta_s_z'])['gap_days_fwd'].notna().mean():.1%}")

sub_base = pn.dropna(subset=["ret_next", "delta_h_z", "delta_s_z"])
o_base, _ = fama_macbeth_nw(sub_base, "ret_next", ["delta_h_z", "delta_s_z"])
say(f"\n  [baseline, unnormalized -- reproduces headline]  N={len(sub_base):,}")
shw("β_ΔH", o_base, "delta_h_z"); shw("β_ΔS", o_base, "delta_s_z")

sub_norm = pn.dropna(subset=["ret_next_norm", "delta_h_z", "delta_s_z"])
o_norm, _ = fama_macbeth_nw(sub_norm, "ret_next_norm", ["delta_h_z", "delta_s_z"])
say(f"\n  [horizon-normalized, r_norm = ret_next * 91.25/gap_days_fwd]  N={len(sub_norm):,}  "
    f"avg N/q={sub_norm.groupby('q').size().mean():.0f}")
shw("β_ΔH", o_norm, "delta_h_z"); shw("β_ΔS", o_norm, "delta_s_z")

say("\n  Variant: gap_days_fwd added as a cross-sectional CONTROL (raw days) instead of rescaling:")
sub_ctrl = pn.dropna(subset=["ret_next", "delta_h_z", "delta_s_z", "gap_days_fwd"])
o_ctrl, _ = fama_macbeth_nw(sub_ctrl, "ret_next", ["delta_h_z", "delta_s_z", "gap_days_fwd"])
say(f"  N={len(sub_ctrl):,}  avg N/q={sub_ctrl.groupby('q').size().mean():.0f}")
shw("β_ΔH", o_ctrl, "delta_h_z"); shw("β_ΔS", o_ctrl, "delta_s_z")
shw("β_gap_days_fwd", o_ctrl, "gap_days_fwd")

# ═══════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 78); say("SUMMARY TABLE"); say("=" * 78)
say(f"{'Spec':42} {'t(ΔH)':>8} {'t(ΔS)':>8} {'N':>10} {'avg N/q':>8}")
def row(tag, o, sub):
    m,t,n = o.get('delta_h_z',(np.nan,np.nan,0))
    ms,ts,ns = o.get('delta_s_z',(np.nan,np.nan,0))
    say(f"{tag:42} {t:>8.2f} {ts:>8.2f} {len(sub):>10,} {sub.groupby('q').size().mean():>8.0f}")
row("Baseline (unnormalized headline)", o_base, sub_base)
row("Horizon-normalized (91.25/gap_days_fwd)", o_norm, sub_norm)
row("gap_days_fwd as control", o_ctrl, sub_ctrl)

say(f"\n  Baseline t(ΔS)={o_base.get('delta_s_z',(0,np.nan))[1]:+.2f} -> "
    f"normalized t(ΔS)={o_norm.get('delta_s_z',(0,np.nan))[1]:+.2f}  "
    f"(delta = {o_norm.get('delta_s_z',(0,np.nan))[1]-o_base.get('delta_s_z',(0,np.nan))[1]:+.2f})")
say(f"  Baseline t(ΔH)={o_base.get('delta_h_z',(0,np.nan))[1]:+.2f} -> "
    f"normalized t(ΔH)={o_norm.get('delta_h_z',(0,np.nan))[1]:+.2f}  "
    f"(delta = {o_norm.get('delta_h_z',(0,np.nan))[1]-o_base.get('delta_h_z',(0,np.nan))[1]:+.2f})")

out_txt = f"{OUT}/R22_horizon_normalization_results.txt"
with open(out_txt, "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\nSaved: {out_txt}")
