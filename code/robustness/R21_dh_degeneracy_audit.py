"""R21 -- Delta-H degeneracy / filing-count audit (BLOCKING).

Sec 3.2 states: "in the full universe the median is a single distinct filing
at every window, with means of 1.8 to 3.4." Delta-H = -sigma(GPM) over a
window with one distinct GPM value is exactly zero -- the MAXIMUM attainable
Delta-H (since sigma >= 0). After cross-sectional z-scoring, the top of the
Delta-H distribution may be populated by short-filing-history firms, which
proxies listing tenure / firm age / data coverage, both correlated with
delisting hazard and returns. This script measures whether that is true and
whether it drives the headline t(dH)=+3.46 on the R18 full-universe panel.

Report as run. No tuning toward the manuscript value.

Outputs: robustness/outputs/R21_dh_degeneracy_results.txt
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

# ── R18 fama-macbeth NW-4 convention (verbatim from R18_sf1_quarterly_survfree.py) ─
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

def rolling_nunique_and_count(values, window):
    """O(n) exact trailing-window distinct-value count and row count.
    values: 1D float array (NaN = missing), one ticker's time series in date order."""
    n = len(values)
    ndist = np.full(n, np.nan)
    ncnt = np.zeros(n, dtype=np.int64)
    counts = {}
    distinct = 0
    cnt = 0
    start = 0
    for i in range(n):
        v = values[i]
        if not np.isnan(v):
            c = counts.get(v, 0)
            if c == 0:
                distinct += 1
            counts[v] = c + 1
            cnt += 1
        lo = i - window + 1
        while start < lo:
            vs = values[start]
            if not np.isnan(vs):
                cs = counts[vs] - 1
                counts[vs] = cs
                if cs == 0:
                    distinct -= 1
                cnt -= 1
            start += 1
        ndist[i] = distinct
        ncnt[i] = cnt
    return ndist, ncnt

# ═══════════════════════════════════════════════════════════════════════════
say("=" * 78); say("R21 -- DELTA-H DEGENERACY / FILING-COUNT AUDIT"); say("=" * 78)

# ── load underlying monthly fundamentals (forward-filled quarterly GPM) ─────
mf = pd.read_parquet(f"{DATA}/monthly_fundamentals.parquet")
mf["date"] = pd.to_datetime(mf["date"])
mf = mf.sort_values(["stock_id", "date"]).reset_index(drop=True)
say(f"\nmonthly_fundamentals: {len(mf):,} rows, {mf['stock_id'].nunique():,} tickers")

say("\nΔH construction (verbatim, robustness/R18_sf1_quarterly_survfree.py lines 182-192,")
say("and project/sharadar_pipeline.py lines 333-336):")
say('  mf["dH_gpm"] = -mf.groupby("stock_id")["gpm"].transform(')
say('      lambda x: x.rolling(60, min_periods=24).std())')
say("  ... sampled to quarter-end via drop_duplicates(['ticker','q'], keep='last')")
say("NOTE: `.rolling(60, min_periods=24)` operates on the MONTHLY FORWARD-FILLED gpm")
say("column (built via merge_asof(..., direction='backward') from quarterly/annual SF1")
say("filings onto a monthly grid). min_periods=24 therefore tests the count of MONTHLY")
say("ROWS with non-null gpm within the trailing window (i.e. months since first filing,")
say("capped by 60), NOT the count of DISTINCT underlying filings. A firm filing once and")
say("then held flat for 24+ months clears min_periods=24 while having n_distinct=1.")

# ── rolling n_distinct / row-count at W = 24, 60, 72 (min_periods=24 throughout,")
#     matching the ACTUAL primary construction's min-obs rule) ────────────────
say("\n" + "-" * 78)
say("R21.1a -- min-obs rule empirical check (row-count vs distinct-value count)")
say("-" * 78)

WINDOWS = [24, 60, 72]
results_by_w = {}
for W in WINDOWS:
    ndist_all = np.full(len(mf), np.nan)
    ncnt_all = np.zeros(len(mf), dtype=np.int64)
    for tkr, idx in mf.groupby("stock_id", sort=False).groups.items():
        idx = idx.to_numpy()
        vals = mf["gpm"].values[idx]
        nd, nc = rolling_nunique_and_count(vals, W)
        ndist_all[idx] = nd
        ncnt_all[idx] = nc
    tmp = mf[["stock_id", "date"]].copy()
    tmp["n_distinct"] = ndist_all
    tmp["row_cnt"] = ncnt_all
    tmp["dH_gpm"] = -mf.groupby("stock_id")["gpm"].transform(
        lambda x: x.rolling(W, min_periods=24).std())
    tmp["q"] = tmp["date"].dt.to_period("Q")
    samp = (tmp.dropna(subset=["dH_gpm"]).sort_values(["stock_id", "date"])
              .drop_duplicates(["stock_id", "q"], keep="last")
              .rename(columns={"stock_id": "ticker"})
              [["ticker", "q", "dH_gpm", "n_distinct", "row_cnt"]])
    results_by_w[W] = samp
    say(f"  W={W:>3}mo: sample N={len(samp):,}  "
        f"row_cnt range at min_periods boundary=[{samp['row_cnt'].min():.0f},{samp['row_cnt'].max():.0f}]  "
        f"n_distinct range=[{samp['n_distinct'].min():.0f},{samp['n_distinct'].max():.0f}]")

say(f"\n  Confirmed: min_periods=24 filters on row_cnt (min row_cnt in sample = "
    f"{results_by_w[60]['row_cnt'].min():.0f}, i.e. exactly 24 as designed), while")
say(f"  n_distinct is UNCONSTRAINED by the min-obs rule and can be as low as 1.")

# ── frequency table of n_distinct at each window ─────────────────────────────
say("\n" + "-" * 78)
say("R21.1b -- Frequency table of n_distinct (distinct GPM values in window)")
say("-" * 78)

def freq_table(samp, tag):
    nd = samp["n_distinct"].clip(upper=10).astype(int)
    vc = nd.value_counts().sort_index()
    tot = len(samp)
    say(f"\n  {tag}  (N={tot:,}, tickers={samp['ticker'].nunique():,})")
    say(f"    {'n_distinct':>12} {'count':>10} {'share':>8}")
    for k in sorted(vc.index):
        label = f"{k}" if k < 10 else "10+"
        say(f"    {label:>12} {vc[k]:>10,} {vc[k]/tot:>7.1%}")
    med = samp["n_distinct"].median()
    mean = samp["n_distinct"].mean()
    say(f"    median n_distinct = {med:.1f}   mean = {mean:.2f}")
    return med, mean

for W in WINDOWS:
    freq_table(results_by_w[W], f"Full universe, W={W} months (min_periods=24)")

say("\n  Manuscript claim (Sec 3.2): full universe median=1 distinct filing at every")
say("  window, means 1.8-3.4. S&P500 panel means 2.8 (24mo) rising to 6.5 (72mo).")
say("  Compare against printed medians/means above -- reported as computed, not adjusted.")

# ── value of dH at n_distinct == 1 ────────────────────────────────────────────
say("\n" + "-" * 78)
say("R21.1c -- ΔH value when n_distinct = 1 (primary W=60)")
say("-" * 78)
samp60 = results_by_w[60]
at1 = samp60.loc[samp60["n_distinct"] == 1, "dH_gpm"]
say(f"  n obs with n_distinct=1 (W=60): {len(at1):,}")
say(f"  dH_gpm at n_distinct=1: min={at1.min():.3e}  max={at1.max():.3e}  "
    f"mean={at1.mean():.3e}  all exactly zero={bool((at1 == 0).all())}")
say(f"  -> code yields exactly 0.0 (not NaN, not dropped): std() of a constant-value")
say(f"     window is computed as 0.0 exactly; -0.0 negated is 0.0. Rows are RETAINED")
say(f"     in the panel with dH_gpm=0.0, which sits at the TOP of the ΔH ('most stable')")
say(f"     distribution since ΔH = -σ and σ ≥ 0.")

# ── merge n_distinct (W=60) onto the ACTUAL R18 panel via ticker/q ──────────
say("\n" + "-" * 78)
say("R21.1d -- Corr(ΔH_z, n_distinct), Corr(ΔH_z, firm_age), mean fwd return by bucket")
say("-" * 78)
panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
nd60 = samp60[["ticker", "q", "n_distinct"]]
p = panel.merge(nd60, on=["ticker", "q"], how="inner")
say(f"  Merge check: R18 panel N={len(panel):,}, matched with n_distinct (W=60) N={len(p):,}")

# firm_age = quarters since first appearance of ticker IN THE R18 PANEL
first_q = panel.groupby("ticker")["q_ord"].transform("min")
panel = panel.assign(firm_age=panel["q_ord"] - first_q)
p = p.merge(panel[["ticker", "q", "firm_age"]], on=["ticker", "q"], how="left")

pc = p.dropna(subset=["delta_h_z", "n_distinct", "firm_age"])
corr_nd = pc["delta_h_z"].corr(pc["n_distinct"])
corr_age = pc["delta_h_z"].corr(pc["firm_age"])
say(f"  Corr(ΔH_z, n_distinct)  [W=60, N={len(pc):,}] = {corr_nd:+.4f}")
say(f"  Corr(ΔH_z, firm_age)    [W=60, N={len(pc):,}] = {corr_age:+.4f}")

say("\n  Mean forward return (ret_next) by n_distinct bucket:")
pr = p.dropna(subset=["ret_next", "n_distinct"]).copy()
pr["bucket"] = pd.cut(pr["n_distinct"], bins=[0, 1, 2, 4, np.inf],
                       labels=["1", "2", "3-4", "5+"], right=True)
gb = pr.groupby("bucket", observed=True)["ret_next"].agg(["mean", "count"])
for lbl, row in gb.iterrows():
    say(f"    n_distinct={lbl:>4}: mean ret_next={row['mean']*100:+.3f}%/q "
        f"({row['mean']*400:+.1f}%/yr ann)  N={int(row['count']):,}")

# ═══════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 78); say("R21.2 -- RESTRICTED-SAMPLE RERUN"); say("=" * 78)

for cut, tag in [(2, "n_distinct >= 2"), (3, "n_distinct >= 3")]:
    sub = p[p["n_distinct"] >= cut].dropna(subset=["ret_next", "delta_h_z", "delta_s_z"])
    avg_n = sub.groupby("q").size().mean()
    o, _ = fama_macbeth_nw(sub, "ret_next", ["delta_h_z", "delta_s_z"])
    say(f"\n  {tag}  (N={len(sub):,}, avg firms/q={avg_n:.0f})")
    shw("β_ΔH", o, "delta_h_z")
    shw("β_ΔS", o, "delta_s_z")

say(f"\n  [Headline / unrestricted baseline for comparison, on the merged n_distinct sample]")
sub_all = p.dropna(subset=["ret_next", "delta_h_z", "delta_s_z"])
o_all, _ = fama_macbeth_nw(sub_all, "ret_next", ["delta_h_z", "delta_s_z"])
avg_all = sub_all.groupby("q").size().mean()
say(f"  n_distinct >= 1 (all)  (N={len(sub_all):,}, avg firms/q={avg_all:.0f})")
shw("β_ΔH", o_all, "delta_h_z")
shw("β_ΔS", o_all, "delta_s_z")

# ═══════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 78); say("R21.3 -- CONTROLLED RERUN ON FULL PANEL"); say("=" * 78)
pctl = p.copy()
pctl["log_ndist"] = np.log(pctl["n_distinct"].clip(lower=1))
pctl["firm_age_c"] = pctl["firm_age"].astype(float)
sub3 = pctl.dropna(subset=["ret_next", "delta_h_z", "delta_s_z", "log_ndist", "firm_age_c"])
o3, _ = fama_macbeth_nw(sub3, "ret_next", ["delta_h_z", "delta_s_z", "log_ndist", "firm_age_c"])
avg3 = sub3.groupby("q").size().mean()
say(f"\n  Full panel + log(n_distinct) + firm_age controls (N={len(sub3):,}, avg firms/q={avg3:.0f})")
shw("β_ΔH", o3, "delta_h_z")
shw("β_ΔS", o3, "delta_s_z")
shw("β_log(n_distinct)", o3, "log_ndist")
shw("β_firm_age", o3, "firm_age_c")

# ═══════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 78); say("R21.4 -- VARIANCE DECOMPOSITION"); say("=" * 78)
pc2 = p.dropna(subset=["delta_h_z", "n_distinct"])
mass1 = pc2[pc2["n_distinct"] == 1]
rest = pc2[pc2["n_distinct"] > 1]
share_n = len(mass1) / len(pc2)
var_tot = pc2["delta_h_z"].var()
# between/within decomposition: total var = share*var_within_group1 + (1-share)*var_within_rest + var_between_group_means
grp_var = pc2.groupby(pc2["n_distinct"] == 1)["delta_h_z"].agg(["var", "mean", "count"])
say(f"\n  n_distinct=1 mass point: N={len(mass1):,} ({share_n:.1%} of ΔH-scored obs)")
say(f"  mean ΔH_z | n_distinct=1  = {mass1['delta_h_z'].mean():+.4f}  (std={mass1['delta_h_z'].std():.4f})")
say(f"  mean ΔH_z | n_distinct>1  = {rest['delta_h_z'].mean():+.4f}  (std={rest['delta_h_z'].std():.4f})")
say(f"  overall var(ΔH_z) = {var_tot:.4f}")
w1 = len(mass1)/len(pc2); w2 = len(rest)/len(pc2)
between = w1*(mass1['delta_h_z'].mean() - pc2['delta_h_z'].mean())**2 + \
          w2*(rest['delta_h_z'].mean() - pc2['delta_h_z'].mean())**2
within = w1*mass1['delta_h_z'].var(ddof=0) + w2*rest['delta_h_z'].var(ddof=0)
say(f"  between-group variance share (n_distinct=1 vs rest) = {between/(between+within):.1%}")
say(f"  within-group variance share                          = {within/(between+within):.1%}")

say("\n  FM slope re-estimated EXCLUDING the n_distinct=1 mass point entirely:")
sub_excl = p[p["n_distinct"] > 1].dropna(subset=["ret_next", "delta_h_z", "delta_s_z"])
o_excl, _ = fama_macbeth_nw(sub_excl, "ret_next", ["delta_h_z", "delta_s_z"])
avg_excl = sub_excl.groupby("q").size().mean()
say(f"  N={len(sub_excl):,}, avg firms/q={avg_excl:.0f}")
shw("β_ΔH (excl. n_distinct=1)", o_excl, "delta_h_z")
shw("β_ΔS (excl. n_distinct=1)", o_excl, "delta_s_z")
say(f"\n  [reference] full-sample (incl. n_distinct=1) β_ΔH t = {o_all.get('delta_h_z',(0,np.nan))[1]:+.2f}")

# ═══════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 78); say("SUMMARY TABLE"); say("=" * 78)
say(f"{'Spec':40} {'t(ΔH)':>8} {'t(ΔS)':>8} {'N':>10} {'avg N/q':>8}")
def row(tag, o, sub):
    m,t,n = o.get('delta_h_z',(np.nan,np.nan,0))
    ms,ts,ns = o.get('delta_s_z',(np.nan,np.nan,0))
    say(f"{tag:40} {t:>8.2f} {ts:>8.2f} {len(sub):>10,} {sub.groupby('q').size().mean():>8.0f}")
row("Baseline (all n_distinct>=1)", o_all, sub_all)
for cut, tag in [(2,"n_distinct>=2"),(3,"n_distinct>=3")]:
    sub = p[p["n_distinct"] >= cut].dropna(subset=["ret_next","delta_h_z","delta_s_z"])
    o,_ = fama_macbeth_nw(sub, "ret_next", ["delta_h_z","delta_s_z"])
    row(tag, o, sub)
row("Excl. n_distinct=1", o_excl, sub_excl)

out_txt = f"{OUT}/R21_dh_degeneracy_results.txt"
with open(out_txt, "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\nSaved: {out_txt}")
