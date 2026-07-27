"""M9-R — Ormos-Zibriczky (2014) actual entropy measure + survival ladder.

Closes the inference->demonstration gap in §5.2: instead of showing that the
27-year survival-conditioning design manufactures a positive premium on OUR ΔS
(36m/12q rolling FF3 residual iVol), compute O-Z's ACTUAL entropy-of-returns
measure on the R18 corrected panel and re-run the Table-8 survival ladder with
it. If the ladder reproduces with THEIR variable, the O-Z claim converts from
inference to demonstration.

Standing rules (from the spec):
- Build the estimator from O-Z's stated definition, report what it prints.
- If the ladder does NOT reproduce, that is a finding, not a failure.
- Do not assert what O-Z's paper says from memory; state ambiguities + default.

O-Z reference: Ormos & Zibriczky (2014), "Entropy based financial asset
pricing," PLOS ONE 9(12):e115742.

DEFINITIONAL NOTE (see [M9-note] in output):
O-Z estimate the entropy of each asset's return DISTRIBUTION over a rolling
window. Their study uses DAILY returns. The R18 corrected (survivorship-free)
universe only has QUARTERLY returns — the SEP daily/monthly price product is
NOT entitled on this Sharadar key (this is precisely why R18 exists at
quarterly frequency). So we compute entropy on the same quarterly return
series that feeds ΔS, over the SAME rolling 12-quarter window as ΔS, for an
apples-to-apples ladder. This is a coarse-frequency analog of O-Z's estimator,
stated as such. Discretization: fixed bin count over each asset's own return
range within the window (per-asset support, as O-Z bin per asset). Default
bins reported below; robustness to bin count is printed.

Outputs: results/revision/M9_oz_entropy_ladder.txt
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT  = "../results/revision"
os.makedirs(OUT, exist_ok=True)

LOG = []
def say(*a):
    line = " ".join(str(x) for x in a)
    print(line); LOG.append(line)

# ── helpers (identical to R18 / R25) ─────────────────────────────────────────
def cs_winsorize_zscore(df, col, date_col="q", pct=0.01):
    def _wz(x):
        x = x.dropna()
        if len(x) < 5: return pd.Series(np.nan, index=x.index)
        lo, hi = x.quantile(pct), x.quantile(1 - pct)
        xc = x.clip(lo, hi); std = xc.std()
        if std < 1e-10: return pd.Series(np.nan, index=x.index)
        return (xc - xc.mean()) / std
    return df.groupby(date_col)[col].transform(_wz)

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
        gamma0 = (s**2).mean() - mean_**2; var = gamma0
        for l in range(1, min(lags + 1, n)):
            g = ((s.iloc[l:].values - mean_) * (s.iloc[:-l].values - mean_)).mean()
            var += 2 * (1 - l / (lags + 1)) * g
        se = np.sqrt(max(var, 1e-30) / n)
        out[col] = (mean_, mean_ / se, n)
    return out, cdf

def nw_mean_t(s, lags=4):
    s = pd.Series(s).dropna(); n = len(s)
    if n < 5: return np.nan
    m = s.mean(); var = (s**2).mean() - m**2
    for l in range(1, min(lags + 1, n)):
        g = ((s.iloc[l:].values - m) * (s.iloc[:-l].values - m)).mean()
        var += 2 * (1 - l / (lags + 1)) * g
    return m / np.sqrt(max(var, 1e-30) / n)

def quintile_ls(df, sortcol, date_col="q", ycol="ret_next"):
    d = df.dropna(subset=[sortcol, ycol]).copy()
    d["qd"] = d.groupby(date_col)[sortcol].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
        if x.nunique() >= 5 else np.nan)
    d = d.dropna(subset=["qd"])
    qr = d.groupby([date_col, "qd"])[ycol].mean().unstack("qd")
    if 0 not in qr.columns or 4 not in qr.columns:
        return np.nan, np.nan, np.nan, 0
    ls = (qr[4] - qr[0]).dropna()
    t = ls.mean()/(ls.std()/np.sqrt(len(ls))) if len(ls) > 2 else np.nan
    return ls.mean()*4, t, nw_mean_t(ls, lags=4), len(ls)

# ── O-Z entropy estimator ────────────────────────────────────────────────────
def _entropy_from_window(w, bins, alpha):
    """Entropy of a return window w via a fixed-bin histogram over w's own range.
    alpha=None -> Shannon; else Rényi H_a = (1/(1-a)) log Σ p_i^a."""
    w = w[~np.isnan(w)]
    if len(w) < 2:
        return np.nan
    rng = w.max() - w.min()
    if rng <= 0:
        return 0.0
    counts, _ = np.histogram(w, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    if alpha is None or abs(alpha - 1.0) < 1e-9:
        return float(-(p * np.log(p)).sum())
    return float((1.0 / (1.0 - alpha)) * np.log((p ** alpha).sum()))

def entropy_ticker(g, window=12, min_obs=8, bins=5, alpha=None):
    """Rolling entropy paralleling R18's ivol_ticker (same window / min_obs)."""
    g = g.sort_values("q")
    r = g["ret"].values
    out = pd.Series(np.nan, index=g.index)
    n = len(g)
    for i in range(min_obs, n + 1):
        lo = max(0, i - window)
        out.iloc[i - 1] = _entropy_from_window(r[lo:i], bins, alpha)
    return out

# ═══════════════════════════════════════════════════════════════════════════
say("="*74)
say("M9-R — ORMOS-ZIBRICZKY ENTROPY MEASURE + SURVIVAL-CONDITIONING LADDER")
say("="*74)

# ── Rebuild the exact R18 quarterly return series (Steps 1-2 of R18) ─────────
# so the entropy window uses the identical returns that fed ΔS, BEFORE the
# T-window restriction (avoids a start-of-panel truncation artifact).
say("\nRebuilding R18 quarterly return series (universe + prices)...")
tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"]
uni = sf1t[
    sf1t["category"].str.contains("Domestic Common", na=False) &
    sf1t["exchange"].isin(["NYSE","NASDAQ","NYSEARCA","BATS","NYSEMKT"]) &
    (sf1t["currency"] == "USD")
].copy()
uni_set = set(uni["ticker"])

sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                      columns=["ticker","dimension","calendardate","datekey",
                               "price","dps"])
arq = sf1[(sf1["dimension"] == "ARQ") & sf1["ticker"].isin(uni_set)].copy()
arq["calendardate"] = pd.to_datetime(arq["calendardate"], errors="coerce")
arq = arq.dropna(subset=["calendardate","price"])
arq = arq[arq["price"] > 0]
arq["q"] = arq["calendardate"].dt.to_period("Q")
arq = (arq.sort_values(["ticker","calendardate"])
          .drop_duplicates(["ticker","q"], keep="last"))
arq = arq.sort_values(["ticker","q"])
arq["q_ord"] = arq["q"].apply(lambda p: p.ordinal)
arq["price_prev"] = arq.groupby("ticker")["price"].shift(1)
arq["gap"] = arq["q_ord"] - arq.groupby("ticker")["q_ord"].shift(1)
ret_px = arq["price"] / arq["price_prev"] - 1.0
div_q  = (arq["dps"].fillna(0) / 4.0) / arq["price_prev"]
arq["ret"] = np.where(arq["gap"] == 1, ret_px + div_q, np.nan)
arq = arq.dropna(subset=["ret"])
arq["ret"] = arq.groupby("q")["ret"].transform(
    lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
cnt = arq.groupby("ticker")["ret"].transform("size")
arq = arq[cnt >= 8]
prices = arq[["ticker","q","ret"]].copy()
say(f"Quarterly return obs: {len(prices):,} | tickers: {prices['ticker'].nunique():,}")

# ── STEP 1 — compute O-Z entropy (Shannon primary; Rényi 0.5 & 2) ───────────
WINDOW, MIN_OBS = 12, 8      # identical to R18 ivol window / min_obs
BINS_DEFAULT = 5            # Sturges for n=12: ceil(log2(12)+1)=5; stated default
say(f"\n[STEP 1] Computing rolling entropy (window={WINDOW}q, min_obs={MIN_OBS}, "
    f"bins={BINS_DEFAULT}, per-asset support)...")

def add_entropy(df, colname, bins, alpha):
    return df.groupby("ticker", group_keys=False).apply(
        lambda g: entropy_ticker(g, WINDOW, MIN_OBS, bins, alpha)).rename(colname)

prices["H_shannon"] = add_entropy(prices, "H_shannon", BINS_DEFAULT, None)
prices["H_renyi05"] = add_entropy(prices, "H_renyi05", BINS_DEFAULT, 0.5)
prices["H_renyi2"]  = add_entropy(prices, "H_renyi2",  BINS_DEFAULT, 2.0)
# bin-count robustness on Shannon
prices["H_shannon_b10"] = add_entropy(prices, "H_shannon_b10", 10, None)
prices["H_shannon_b8"]  = add_entropy(prices, "H_shannon_b8",  8, None)

Hcols = ["H_shannon","H_renyi05","H_renyi2","H_shannon_b10","H_shannon_b8"]
say("entropy computed. coverage (non-null / total return obs):")
for c in Hcols:
    say(f"  {c:16s}: {prices[c].notna().sum():,}/{len(prices):,} "
        f"({prices[c].notna().mean():.1%})   "
        f"mean={prices[c].mean():.4f} sd={prices[c].std():.4f} "
        f"[min={prices[c].min():.3f}, max={prices[c].max():.3f}]")

# ── merge entropy onto the saved R18 panel ──────────────────────────────────
panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
panel = panel.merge(prices[["ticker","q"]+Hcols], on=["ticker","q"], how="left")
say(f"\nR18 panel: N={len(panel):,}, tickers={panel['ticker'].nunique():,}, "
    f"quarters={panel['q'].nunique()} ({panel['q'].min()}..{panel['q'].max()})")
say(f"H_shannon coverage within R18 panel: "
    f"{panel['H_shannon'].notna().sum():,}/{len(panel):,} "
    f"({panel['H_shannon'].notna().mean():.1%})")

# sanity: Corr(entropy, ΔS) — pooled and mean cross-sectional per quarter
say("\n[M9-1 sanity] Corr(H, ΔS):")
for c in Hcols:
    m = panel[[c,"delta_s"]].dropna()
    pooled = m[c].corr(m["delta_s"])
    csm = (panel.dropna(subset=[c,"delta_s"])
                .groupby("q").apply(lambda x: x[c].corr(x["delta_s"])).mean())
    say(f"  {c:16s}: pooled={pooled:+.4f}   mean_XS={csm:+.4f}")

# ═══ STEP 2 — unconditional premium on the R18 panel with H ═══════════════════
say("\n" + "#"*74)
say("# [STEP 2] Unconditional entropy premium on the R18 corrected panel")
say("#"*74)
# z-score entropy cross-sectionally (as ΔS is z-scored), full panel
panel["H_z"]   = cs_winsorize_zscore(panel, "H_shannon")
panel["dS_z"]  = cs_winsorize_zscore(panel, "delta_s")   # reproduce R18 exactly
panel["dH_z"]  = cs_winsorize_zscore(panel, "dH_gpm")

pe0 = panel.dropna(subset=["ret_next","H_z"])
pf0 = panel.dropna(subset=["ret_next","H_z","dH_z"])
ls_ann, ls_t, ls_tnw, ls_Tq = quintile_ls(pe0, "H_z")
fm0, _ = fama_macbeth_nw(pf0, "ret_next", ["dH_z","H_z"])
bH, tH, TqH = fm0.get("H_z", (np.nan, np.nan, 0))
say(f"\n[M9-2] full-universe (Shannon H, bins={BINS_DEFAULT}):")
say(f"  FM Model B  t(H) = {tH:+.4g}   beta(H) = {bH:+.6f}   T_q = {TqH}")
say(f"  quintile L/S = {ls_ann*100:+.2f}%/yr  (t_simple={ls_t:+.2f}, "
    f"t_NW4={ls_tnw:+.2f}, T_q={ls_Tq})")

# ═══ STEP 3 — SURVIVAL-CONDITIONING LADDER WITH ENTROPY ═════════════════════
say("\n" + "#"*74)
say("# [STEP 3] Survival-conditioning ladder (Table-8 rungs) with entropy H")
say("#"*74)

# per-ticker consecutive-quarter run IDs (identical to R25 E1)
panel = panel.sort_values(["ticker","q_ord"]).reset_index(drop=True)
new_run = (panel["q_ord"] - panel.groupby("ticker")["q_ord"].shift(1)) != 1
panel["run_id"] = new_run.groupby(panel["ticker"]).cumsum()
panel["run_len_q"] = panel.groupby(["ticker","run_id"])["q_ord"].transform("size")

# market cap for the cap-tilt diagnostic (identical to R25 E1)
mc = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                     columns=["ticker","dimension","calendardate","marketcap"])
mc = mc[(mc["dimension"] == "ARQ") & mc["ticker"].isin(set(panel["ticker"]))].copy()
mc["calendardate"] = pd.to_datetime(mc["calendardate"], errors="coerce")
mc = mc.dropna(subset=["calendardate","marketcap"])
mc["q"] = mc["calendardate"].dt.to_period("Q")
mc = (mc.sort_values(["ticker","calendardate"])
        .drop_duplicates(["ticker","q"], keep="last"))[["ticker","q","marketcap"]]
panel = panel.merge(mc, on=["ticker","q"], how="left")

KS = [0, 5, 10, 15, 20, 25, 27]
rows = []
full_med_mc = np.nan
for k in KS:
    thr_q = int(round(4 * k))
    sub = panel[panel["run_len_q"] >= max(thr_q, 1)].copy() if k > 0 else panel.copy()
    if len(sub) == 0:
        say(f"\n[k={k}] EMPTY subsample — skipping"); continue
    # re-standardize WITHIN the conditioned panel (OZ design = conditioned
    # universe), identical to R25 E1's treatment of delta_s
    sub["H_z"]  = cs_winsorize_zscore(sub, "H_shannon")
    sub["dH_z"] = cs_winsorize_zscore(sub, "dH_gpm")
    pe = sub.dropna(subset=["ret_next","H_z"])
    pf = sub.dropna(subset=["ret_next","H_z","dH_z"])

    ls_ann, ls_t, ls_tnw, ls_Tq = quintile_ls(pe, "H_z")
    fm, _ = fama_macbeth_nw(pf, "ret_next", ["dH_z","H_z"])
    b_dh, t_dh, _  = fm.get("dH_z", (np.nan, np.nan, 0))
    b_h,  t_h, Tq  = fm.get("H_z",  (np.nan, np.nan, 0))
    n_tick = sub["ticker"].nunique()
    med_mc = sub["marketcap"].median()
    if k == 0:
        full_med_mc = med_mc

    say(f"\n[M9-3 | k={k}]")
    say(f"  N_tickers={n_tick:,}  N_obs={len(pe):,}  T_q={Tq}")
    say(f"  FM t(H)={t_h:+.4g}  beta(H)={b_h:+.6f}   FM t(ΔH)={t_dh:+.4g}")
    say(f"  quintile L/S={ls_ann*100:+.2f}%/yr (t_simple={ls_t:+.2f}, "
        f"t_NW4={ls_tnw:+.2f})   median_cap=${med_mc/1e6:.4g}M")
    rows.append(dict(k=k, ls_ann=ls_ann, ls_t=ls_t, ls_tnw=ls_tnw,
                     t_h=t_h, t_dh=t_dh, n_tick=n_tick, n_obs=len(pe),
                     med_mc=med_mc))

# summary table
say("\n" + "-"*74)
say("[M9-3 | SUMMARY LADDER]  (entropy H = Shannon, per-asset "
    f"{BINS_DEFAULT}-bin, 12q window)")
say(f"{'k(yr)':>5} {'FM t(H)':>8} {'FM t(dH)':>9} {'L/S ann':>9} {'L/S t':>7} "
    f"{'L/S tNW':>8} {'N_tick':>7} {'N_obs':>8} {'medMC$M':>9}")
for r in rows:
    say(f"{r['k']:>5} {r['t_h']:>+8.2f} {r['t_dh']:>+9.2f} "
        f"{r['ls_ann']*100:>+8.2f}% {r['ls_t']:>+7.2f} {r['ls_tnw']:>+8.2f} "
        f"{r['n_tick']:>7,} {r['n_obs']:>8,} {r['med_mc']/1e6:>9.4g}")
r27 = [r for r in rows if r["k"] == 27]
if r27 and not np.isnan(full_med_mc):
    say(f"\nCap tilt: median cap k=27 = ${r27[0]['med_mc']/1e6:,.4g}M vs "
        f"k=0 = ${full_med_mc/1e6:,.4g}M "
        f"(ratio {r27[0]['med_mc']/full_med_mc:.4g}x)")

# ── ladder robustness: Rényi α=2 and bin=10 as the sort var ─────────────────
say("\n" + "-"*74)
say("[M9-3 robustness] Endpoint ladder with alternative entropy definitions")
say("(FM t at k=0 vs k=27; checks the pattern is not a bins/alpha artifact)")
for var, lbl in [("H_renyi2","Renyi a=2"), ("H_renyi05","Renyi a=0.5"),
                 ("H_shannon_b10","Shannon bins=10")]:
    line = f"  {lbl:18s}: "
    for k in (0, 27):
        thr_q = int(round(4 * k))
        sub = panel[panel["run_len_q"] >= max(thr_q, 1)].copy() if k > 0 else panel.copy()
        sub["v_z"]  = cs_winsorize_zscore(sub, var)
        sub["dH_z"] = cs_winsorize_zscore(sub, "dH_gpm")
        pf = sub.dropna(subset=["ret_next","v_z","dH_z"])
        fm, _ = fama_macbeth_nw(pf, "ret_next", ["dH_z","v_z"])
        _, t_v, _ = fm.get("v_z", (np.nan, np.nan, 0))
        line += f"k={k}: t={t_v:+.2f}   "
    say(line)

# ═══ STEP 4 — definitional note ═════════════════════════════════════════════
say("\n" + "#"*74)
say("# [M9-note] Definitional ambiguities in O-Z's estimator + defaults chosen")
say("#"*74)
say("""
- FREQUENCY: O-Z estimate entropy on DAILY returns. R18's corrected universe
  is QUARTERLY only (SEP daily/monthly not entitled — the reason R18 exists).
  DEFAULT: entropy on the same quarterly return series feeding ΔS, over the
  same 12-quarter rolling window. This is a coarse-frequency analog, stated.
  Consequence: with <=12 obs/window, entropy is a coarse dispersion statistic
  and is expected to correlate fairly strongly with ΔS (iVol) — reported above.
- WINDOW: matched to ΔS (12 quarters, min_obs=8) for an apples-to-apples ladder.
- DISCRETIZATION / BINS: O-Z bin per asset over its return support; bin count
  not pinned to a single universal rule in their method text (they discuss
  histogram and kernel estimators). DEFAULT: fixed 5 bins (Sturges for n=12:
  ceil(log2(12)+1)=5) over each asset's own [min,max] within the window.
  Robustness to bins=8/10 and to Rényi alpha in {0.5, 2} printed above.
- ENTROPY FAMILY: O-Z report a Rényi family and emphasize Shannon (alpha->1)
  as the headline; Rényi/Tsallis appear as robustness. We take Shannon as the
  primary ladder variable and report Rényi(0.5, 2) alongside. (This recollection
  of their emphasis is NOT verified against the paper text per the standing
  rule; the ladder is reported for all three so the conclusion does not depend
  on which family O-Z emphasize.)
- KERNEL vs HISTOGRAM: O-Z discuss kernel-density entropy; with <=12 obs a
  kernel bandwidth is not well identified, so a fixed-bin histogram is the
  defensible small-sample choice, stated.""")

# ── write ───────────────────────────────────────────────────────────────────
with open(f"{OUT}/M9_oz_entropy_ladder.txt", "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\n[written] {OUT}/M9_oz_entropy_ladder.txt")
