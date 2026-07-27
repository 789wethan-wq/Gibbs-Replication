"""D2 rerun (corrected) — per the chat-instance's critique of the first D2 pass:
the original split computed per-firm ΔS *means* across odd/even quarters,
which measures between-firm dispersion in average volatility level, not the
reliability of the ΔS *estimator itself*. That is why it came back flat
(0.983-0.992) irrespective of size.

Corrected design: for each firm-quarter observation with a full 12-quarter
ΔS estimation window (the standard construction: 12-quarter trailing FF3
residual vol, R18_sf1_quarterly_survfree.py's ivol_ticker), split the 12
returns feeding that ONE estimate into odd-position and even-position halves
(6 each by window position), compute ΔS from each half separately using the
same FF3-residual-std estimator, and correlate the two half-estimates across
ALL firm-quarter observations within each size decile. Deciles are assigned
per-observation from CONTEMPORANEOUS market cap (not each firm's all-time
median, which risks re-imposing survival conditioning — see Rerun 3 below).
Spearman-Brown correct the raw half-correlation for the halving of window
length: reliability = 2r / (1+r).

Also reruns the "top-5-decile resurrects t(ΔS)" side finding from the first
D2 pass using a LOOK-AHEAD-FREE decile assignment (trailing median cap
through quarter t, not each firm's all-time median cap), to check whether
that resurrection is a size effect or a survival-conditioning artifact in
disguise.

Outputs: results/revision/D2_corrected_split_half.txt
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"; OUT = "../results/revision"; os.makedirs(OUT, exist_ok=True)
LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

def cs_wz(df, col, datecol, pct=0.01):
    def _w(x):
        x = x.dropna()
        if len(x) < 5: return pd.Series(np.nan, index=x.index)
        xc = x.clip(x.quantile(pct), x.quantile(1-pct)); sd = xc.std()
        if sd < 1e-10: return pd.Series(np.nan, index=x.index)
        return (xc-xc.mean())/sd
    return df.groupby(datecol)[col].transform(_w)

def fama_macbeth_nw(panel, y_col, x_cols, date_col="q", lags=4, min_cs=20):
    coefs = []
    for dte, g in panel.groupby(date_col):
        s = g[[y_col] + x_cols].dropna()
        if len(s) < max(min_cs, len(x_cols) + 2): continue
        X = sm.add_constant(s[x_cols], has_constant="add")
        coefs.append(sm.OLS(s[y_col], X).fit().params[x_cols].rename(dte))
    if not coefs: return {}, pd.DataFrame()
    cdf = pd.DataFrame(coefs)
    out = {}
    for col in x_cols:
        s = cdf[col].dropna(); n = len(s); mean_ = s.mean()
        gamma0 = (s**2).mean() - mean_**2; var = gamma0
        for l in range(1, min(lags + 1, n)):
            gg = ((s.iloc[l:].values - mean_) * (s.iloc[:-l].values - mean_)).mean()
            var += 2 * (1 - l / (lags + 1)) * gg
        se = np.sqrt(max(var, 1e-30) / n)
        out[col] = (mean_, mean_ / se, n)
    return out, cdf

say("="*72); say("D2 RERUN — CORRECTED SPLIT-HALF RELIABILITY OF QUARTERLY ΔS (R18)"); say("="*72)

# ── universe ─────────────────────────────────────────────────────────────────
tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"] == "SF1"]
uni = sf1t[
    sf1t["category"].str.contains("Domestic Common", na=False) &
    sf1t["exchange"].isin(["NYSE","NASDAQ","NYSEARCA","BATS","NYSEMKT"]) &
    (sf1t["currency"] == "USD")
].copy()
uni_set = set(uni["ticker"])
say(f"\nUniverse: {len(uni_set):,} tickers")

# ── rebuild full quarterly return series (same construction as R18/D1) ─────────
say("Rebuilding quarterly return series...")
sf1p = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                        columns=["ticker","dimension","calendardate","datekey","price","dps","marketcap"])
arqp = sf1p[(sf1p["dimension"] == "ARQ") & sf1p["ticker"].isin(uni_set)].copy()
arqp["calendardate"] = pd.to_datetime(arqp["calendardate"], errors="coerce")
arqp = arqp.dropna(subset=["calendardate","price"])
arqp = arqp[arqp["price"] > 0]
arqp["q"] = arqp["calendardate"].dt.to_period("Q")
arqp = arqp.sort_values(["ticker","calendardate"]).drop_duplicates(["ticker","q"], keep="last")
arqp = arqp.sort_values(["ticker","q"])
arqp["q_ord"] = arqp["q"].apply(lambda p: p.ordinal)
arqp["price_prev"] = arqp.groupby("ticker")["price"].shift(1)
arqp["gap"] = arqp["q_ord"] - arqp.groupby("ticker")["q_ord"].shift(1)
ret_px = arqp["price"]/arqp["price_prev"] - 1.0
div_q = (arqp["dps"].fillna(0)/4.0)/arqp["price_prev"]
arqp["ret_full"] = np.where(arqp["gap"] == 1, ret_px + div_q, np.nan)
arqp["ret_full"] = arqp.groupby("q")["ret_full"].transform(
    lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))

fac = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
fac.index = pd.to_datetime(fac.index)
facq = fac.copy(); facq["q"] = facq.index.to_period("Q")
def cmpd(s): return (1+s).prod()-1
ffq = facq.groupby("q").agg({"Mkt_RF":cmpd, "SMB":cmpd, "HML":cmpd, "RF":cmpd}).reset_index()
arqp = arqp.merge(ffq, on="q", how="left")
arqp["exret_full"] = arqp["ret_full"] - arqp["RF"]
arqp = arqp.dropna(subset=["ret_full"])
cnt = arqp.groupby("ticker")["ret_full"].transform("size")
arqp = arqp[cnt >= 8].sort_values(["ticker","q_ord"]).reset_index(drop=True)
say(f"Quarterly return obs (>=8/ticker): {len(arqp):,} | tickers: {arqp['ticker'].nunique():,}")

# ── per-observation split-half ΔS (full 12-quarter windows only) ───────────────
say("\nComputing split-half ΔS on full 12-quarter windows (this takes a few minutes)...")

FACTORS = ["Mkt_RF","SMB","HML"]
def half_ivol(sub):
    """FF3-residual std on a subset of rows (const + 3 factors)."""
    if len(sub) < 5:
        return np.nan
    y = sub["exret_full"].values
    X = sm.add_constant(sub[FACTORS].values, has_constant="add")
    if np.isnan(X).any() or np.isnan(y).any():
        return np.nan
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return resid.std(ddof=1)

records = []
all_window_lens = []
tickers_all = arqp["ticker"].unique()
for ti, tkr in enumerate(tickers_all):
    if ti % 2000 == 0:
        say(f"  {ti}/{len(tickers_all)}...")
    g = arqp[arqp["ticker"] == tkr].sort_values("q_ord").reset_index(drop=True)
    n = len(g)
    for i in range(8, n + 1):
        lo = max(0, i - 12)
        window = g.iloc[lo:i]
        wlen = len(window)
        all_window_lens.append({"ticker": tkr, "q": g.iloc[i-1]["q"], "window_len": wlen,
                                 "marketcap": g.iloc[i-1]["marketcap"]})
        if wlen != 12:
            continue  # only full 12-quarter windows enter the split-half test
        odd_half = window.iloc[1::2]   # positions 1,3,5,7,9,11 (6 obs)
        even_half = window.iloc[0::2]  # positions 0,2,4,6,8,10 (6 obs)
        ds_odd = half_ivol(odd_half)
        ds_even = half_ivol(even_half)
        if np.isnan(ds_odd) or np.isnan(ds_even):
            continue
        records.append({
            "ticker": tkr, "q": g.iloc[i-1]["q"],
            "ds_odd": ds_odd, "ds_even": ds_even,
            "marketcap_contemp": g.iloc[i-1]["marketcap"],
        })

obs_df = pd.DataFrame(records)
winlen_df = pd.DataFrame(all_window_lens)
say(f"\nFull-12-quarter-window observations available for split-half test: {len(obs_df):,}")
say(f"Total live ΔS-estimate observations (any window length 8-12): {len(winlen_df):,}")

# ── decile assignment: CONTEMPORANEOUS market cap, global pooled qcut ──────────
obs_df = obs_df.dropna(subset=["marketcap_contemp"])
obs_df = obs_df[obs_df["marketcap_contemp"] > 0]
obs_df["decile"] = pd.qcut(obs_df["marketcap_contemp"], 10, labels=False, duplicates="drop") + 1

winlen_df = winlen_df.dropna(subset=["marketcap"])
winlen_df = winlen_df[winlen_df["marketcap"] > 0]
winlen_df["decile"] = pd.qcut(winlen_df["marketcap"], 10, labels=False, duplicates="drop") + 1

say("\n" + "-"*80)
say(f"{'Decile':>7}{'N obs (full-12)':>17}{'Corr(odd,even)':>16}{'SB-corrected':>14}{'Mean obs/window (all)':>24}{'Med Cap ($M)':>15}")
say("-"*80)
decile_rows = []
for dec in range(1, 11):
    g = obs_df[obs_df["decile"] == dec]
    if len(g) < 10:
        continue
    r = g["ds_odd"].corr(g["ds_even"])
    sb = 2*r/(1+r) if np.isfinite(r) and (1+r) != 0 else np.nan
    gw = winlen_df[winlen_df["decile"] == dec]
    mean_win = gw["window_len"].mean()
    med_cap = gw["marketcap"].median() / 1e6
    say(f"{dec:>7}{len(g):>17,}{r:>16.3f}{sb:>14.3f}{mean_win:>24.2f}{med_cap:>15,.1f}")
    decile_rows.append({"decile": dec, "n_obs": len(g), "corr": r, "sb_corr": sb,
                         "mean_win": mean_win, "med_cap_m": med_cap})
decile_df = pd.DataFrame(decile_rows)
say("-"*80)

top5 = decile_df[decile_df["decile"] >= 6]["sb_corr"].mean()
bot5 = decile_df[decile_df["decile"] <= 5]["sb_corr"].mean()
say(f"\nTop-5-decile (largest) avg SB-corrected reliability: {top5:.3f}")
say(f"Bottom-5-decile (smallest) avg SB-corrected reliability: {bot5:.3f}")
say(f"Spread (top5 - bottom5): {top5-bot5:+.3f}")
flat = abs(top5 - bot5) < 0.15
say(f"\nVerdict: reliability is {'roughly FLAT' if flat else 'MATERIALLY LOWER in bottom deciles'} "
    f"across size deciles under the corrected (within-window) split-half design.")

# ═════════════════════════════════════════════════════════════════════════
# Rerun 3 — look-ahead-free decile assignment for the "top-5 resurrects t(ΔS)" check
# ═════════════════════════════════════════════════════════════════════════
say("\n" + "#"*72)
say("# RERUN 3 — LOOK-AHEAD-FREE DECILE ASSIGNMENT (trailing median cap through t)")
say("#"*72)

panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
mc = arqp[["ticker","q","marketcap"]].dropna(subset=["marketcap"])
mc = mc[mc["marketcap"] > 0].sort_values(["ticker","q"])
panel = panel.merge(mc, on=["ticker","q"], how="left")

# trailing median cap through quarter t (expanding median using only data up to and including t)
panel = panel.sort_values(["ticker","q_ord"])
panel["trailing_med_cap"] = panel.groupby("ticker")["marketcap"].transform(
    lambda s: s.expanding(min_periods=1).median())

pf = panel.dropna(subset=["ret_next","delta_h_z","delta_s_z","trailing_med_cap"])
pf = pf.copy()
pf["decile_trailing"] = pd.qcut(pf["trailing_med_cap"], 10, labels=False, duplicates="drop") + 1

r_all, _ = fama_macbeth_nw(pf, "ret_next", ["delta_h_z", "delta_s_z"])
say(f"\nFull sample (same as before): t(ΔH)={r_all['delta_h_z'][1]:+.2f}  t(ΔS)={r_all['delta_s_z'][1]:+.2f}  "
    f"(N={len(pf):,})")

top5_trailing = pf[pf["decile_trailing"] >= 6]
r_top5_trail, _ = fama_macbeth_nw(top5_trailing, "ret_next", ["delta_h_z", "delta_s_z"])
say(f"Top-5-decile (TRAILING median cap, look-ahead-free): t(ΔH)={r_top5_trail['delta_h_z'][1]:+.2f}  "
    f"t(ΔS)={r_top5_trail['delta_s_z'][1]:+.2f}  (N={len(top5_trailing):,}, "
    f"tickers={top5_trailing['ticker'].nunique():,})")

top5_trail_rz = top5_trailing.copy()
top5_trail_rz["delta_h_z2"] = cs_wz(top5_trail_rz, "dH_gpm", "q")
top5_trail_rz["delta_s_z2"] = cs_wz(top5_trail_rz, "delta_s", "q")
top5_trail_rz = top5_trail_rz.dropna(subset=["ret_next","delta_h_z2","delta_s_z2"])
r_top5_trail_rz, _ = fama_macbeth_nw(top5_trail_rz, "ret_next", ["delta_h_z2", "delta_s_z2"])
say(f"Top-5-decile (trailing cap), re-z-scored within subsample: t(ΔH)={r_top5_trail_rz['delta_h_z2'][1]:+.2f}  "
    f"t(ΔS)={r_top5_trail_rz['delta_s_z2'][1]:+.2f}  (N={len(top5_trail_rz):,})")

say(f"\nFor comparison, the original (all-time median cap) pass found: t(ΔS)=+2.18 raw / +2.30 re-z-scored.")
resurrection_persists = r_top5_trail['delta_s_z'][1] > 1.5
say(f"\nVerdict: resurrection {'PERSISTS' if resurrection_persists else 'DISAPPEARS OR WEAKENS'} under "
    f"look-ahead-free decile assignment -> {'genuine size-composition effect' if resurrection_persists else 'consistent with survival conditioning rather than size composition (report it that way)'}")

with open(f"{OUT}/D2_corrected_split_half.txt", "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\nSaved: {OUT}/D2_corrected_split_half.txt")
