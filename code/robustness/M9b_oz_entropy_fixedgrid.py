"""M9b — O-Z entropy, FIXED-GRID (dispersion-capturing) discretization.

The M9 primary run binned each asset over its OWN [min,max] support, which makes
Shannon entropy scale-free — it captured distributional SHAPE, not dispersion,
and came out NEGATIVELY correlated with ΔS (iVol). O-Z's measure, and the M9
spec's stated expectation (Corr(H,ΔS) positive), imply a dispersion-capturing
discretization. This run bins on COMMON fixed-width edges shared across all
assets, so a wider (higher-vol) return distribution occupies more bins -> higher
entropy -> positive correlation with iVol. We then re-check Corr(H,ΔS) and re-run
the full survival ladder. This decides whether the M9 null (entropy ladder does
NOT reproduce the ΔS ladder) is real or an artifact of per-asset support.

Common-grid edges: fixed-width bins over the pooled winsorized quarterly-return
support [q0.5%, q99.5%], with two catch-all tail bins. Reported for a couple of
bin counts.

Outputs: results/revision/M9b_oz_entropy_fixedgrid.txt
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"; OUT = "../results/revision"
os.makedirs(OUT, exist_ok=True)
LOG = []
def say(*a):
    line = " ".join(str(x) for x in a); print(line); LOG.append(line)

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
    cdf = pd.DataFrame(coefs); out = {}
    for col in x_cols:
        s = cdf[col].dropna(); n = len(s); mean_ = s.mean()
        var = (s**2).mean() - mean_**2
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
    if 0 not in qr.columns or 4 not in qr.columns: return np.nan, np.nan, np.nan, 0
    ls = (qr[4] - qr[0]).dropna()
    t = ls.mean()/(ls.std()/np.sqrt(len(ls))) if len(ls) > 2 else np.nan
    return ls.mean()*4, t, nw_mean_t(ls, lags=4), len(ls)

WINDOW, MIN_OBS = 12, 8
def entropy_ticker_fixed(g, edges):
    """Shannon entropy over COMMON fixed edges (scale-sensitive, dispersion)."""
    g = g.sort_values("q"); r = g["ret"].values
    out = pd.Series(np.nan, index=g.index); n = len(g)
    for i in range(MIN_OBS, n + 1):
        w = r[max(0, i - WINDOW):i]; w = w[~np.isnan(w)]
        if len(w) < MIN_OBS: continue
        counts, _ = np.histogram(w, bins=edges)  # values outside edges are dropped
        # clip into range so tail obs land in the outer bins (no mass loss)
        wc = np.clip(w, edges[0], edges[-1])
        counts, _ = np.histogram(wc, bins=edges)
        p = counts / counts.sum(); p = p[p > 0]
        out.iloc[i - 1] = float(-(p * np.log(p)).sum())
    return out

say("="*74)
say("M9b — O-Z ENTROPY, FIXED COMMON-GRID (dispersion-capturing) + LADDER")
say("="*74)

# rebuild R18 quarterly returns (identical to R18 / M9)
tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"]
uni = sf1t[sf1t["category"].str.contains("Domestic Common", na=False) &
           sf1t["exchange"].isin(["NYSE","NASDAQ","NYSEARCA","BATS","NYSEMKT"]) &
           (sf1t["currency"] == "USD")].copy()
uni_set = set(uni["ticker"])
sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                      columns=["ticker","dimension","calendardate","datekey","price","dps"])
arq = sf1[(sf1["dimension"] == "ARQ") & sf1["ticker"].isin(uni_set)].copy()
arq["calendardate"] = pd.to_datetime(arq["calendardate"], errors="coerce")
arq = arq.dropna(subset=["calendardate","price"]); arq = arq[arq["price"] > 0]
arq["q"] = arq["calendardate"].dt.to_period("Q")
arq = arq.sort_values(["ticker","calendardate"]).drop_duplicates(["ticker","q"], keep="last")
arq = arq.sort_values(["ticker","q"]); arq["q_ord"] = arq["q"].apply(lambda p: p.ordinal)
arq["price_prev"] = arq.groupby("ticker")["price"].shift(1)
arq["gap"] = arq["q_ord"] - arq.groupby("ticker")["q_ord"].shift(1)
ret_px = arq["price"] / arq["price_prev"] - 1.0
div_q = (arq["dps"].fillna(0) / 4.0) / arq["price_prev"]
arq["ret"] = np.where(arq["gap"] == 1, ret_px + div_q, np.nan)
arq = arq.dropna(subset=["ret"])
arq["ret"] = arq.groupby("q")["ret"].transform(lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
cnt = arq.groupby("ticker")["ret"].transform("size"); arq = arq[cnt >= 8]
prices = arq[["ticker","q","ret"]].copy()

# common fixed grid over pooled return support
lo, hi = prices["ret"].quantile(0.005), prices["ret"].quantile(0.995)
say(f"\nPooled return support [q0.5%,q99.5%] = [{lo:+.4f}, {hi:+.4f}]  "
    f"(N={len(prices):,})")
GRIDS = {"H_fix10": np.linspace(lo, hi, 11),
         "H_fix20": np.linspace(lo, hi, 21)}
for name, edges in GRIDS.items():
    say(f"Computing {name} (fixed {len(edges)-1}-bin common grid)...")
    prices[name] = prices.groupby("ticker", group_keys=False).apply(
        lambda g: entropy_ticker_fixed(g, edges)).rename(name)

Hcols = list(GRIDS.keys())
for c in Hcols:
    say(f"  {c}: coverage {prices[c].notna().mean():.1%}  mean={prices[c].mean():.4f} "
        f"sd={prices[c].std():.4f} [{prices[c].min():.3f},{prices[c].max():.3f}]")

panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
panel = panel.merge(prices[["ticker","q"]+Hcols], on=["ticker","q"], how="left")

say("\n[Corr(H_fixed, ΔS)]  (expect POSITIVE now — dispersion-capturing):")
for c in Hcols:
    m = panel[[c,"delta_s"]].dropna()
    pooled = m[c].corr(m["delta_s"])
    csm = (panel.dropna(subset=[c,"delta_s"]).groupby("q")
                .apply(lambda x: x[c].corr(x["delta_s"])).mean())
    say(f"  {c}: pooled={pooled:+.4f}  mean_XS={csm:+.4f}")

# run-length conditioning setup (identical to R25 E1)
panel = panel.sort_values(["ticker","q_ord"]).reset_index(drop=True)
new_run = (panel["q_ord"] - panel.groupby("ticker")["q_ord"].shift(1)) != 1
panel["run_id"] = new_run.groupby(panel["ticker"]).cumsum()
panel["run_len_q"] = panel.groupby(["ticker","run_id"])["q_ord"].transform("size")

KS = [0, 5, 10, 15, 20, 25, 27]
for c in Hcols:
    say("\n" + "#"*74)
    say(f"# LADDER with {c}")
    say("#"*74)
    say(f"{'k(yr)':>5} {'FM t(H)':>8} {'beta':>10} {'L/S ann':>9} {'L/S t':>7} "
        f"{'L/S tNW':>8} {'N_tick':>7} {'N_obs':>8}")
    for k in KS:
        thr_q = int(round(4 * k))
        sub = panel[panel["run_len_q"] >= max(thr_q, 1)].copy() if k > 0 else panel.copy()
        sub["v_z"]  = cs_winsorize_zscore(sub, c)
        sub["dH_z"] = cs_winsorize_zscore(sub, "dH_gpm")
        pe = sub.dropna(subset=["ret_next","v_z"])
        pf = sub.dropna(subset=["ret_next","v_z","dH_z"])
        ls_ann, ls_t, ls_tnw, _ = quintile_ls(pe, "v_z")
        fm, _ = fama_macbeth_nw(pf, "ret_next", ["dH_z","v_z"])
        b_v, t_v, _ = fm.get("v_z", (np.nan, np.nan, 0))
        say(f"{k:>5} {t_v:>+8.2f} {b_v:>+10.5f} {ls_ann*100:>+8.2f}% {ls_t:>+7.2f} "
            f"{ls_tnw:>+8.2f} {sub['ticker'].nunique():>7,} {len(pe):>8,}")

with open(f"{OUT}/M9b_oz_entropy_fixedgrid.txt", "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\n[written] {OUT}/M9b_oz_entropy_fixedgrid.txt")
