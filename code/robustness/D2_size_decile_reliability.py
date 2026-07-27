"""D2 — Size-decile reliability of quarterly ΔS in the R18 full-universe panel.

Concern: attenuation bias scales with the regressor's noise-to-signal ratio,
which is universe-dependent. Table 9's constant-measurement test shows mild
attenuation cost on the S&P 500 panel (a low-noise universe); that does not
bound attenuation in the noisier full-universe panel, where a 12-quarter
irregularly-spaced ΔS estimate for a $401M-median microcap is a far noisier
volatility estimate than for a large, liquid S&P name.

Split-half reliability: for each ticker with enough history, split its
quarterly ΔS series into odd- and even-indexed quarters (by q_ord), take the
per-ticker mean of each half, and — within each size decile (deciles formed
on each ticker's own median point-in-time market cap) — correlate the odd-half
means against the even-half means across tickers. High correlation = the
measurement is reliable (stable signal, low idiosyncratic noise); low
correlation = ΔS in that decile is mostly noise.

Outputs: results/revision/D2_size_decile_reliability.txt
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

say("="*72); say("D2 — SIZE-DECILE RELIABILITY OF QUARTERLY ΔS (R18 FULL-UNIVERSE)"); say("="*72)

# ── load R18 panel + point-in-time market cap (reuse the M2 convention) ────────
panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
say(f"\nR18 panel: N={len(panel):,}  tickers={panel['ticker'].nunique():,}  "
    f"quarters={panel['q'].nunique()}")

sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                      columns=["ticker","dimension","calendardate","marketcap"])
mc = sf1[sf1["dimension"] == "ARQ"].copy()
mc["calendardate"] = pd.to_datetime(mc["calendardate"], errors="coerce")
mc = mc.dropna(subset=["calendardate","marketcap"])
mc = mc[mc["marketcap"] > 0]
mc["q"] = mc["calendardate"].dt.to_period("Q")
mc = (mc.sort_values(["ticker","calendardate"])
        .drop_duplicates(["ticker","q"], keep="last")[["ticker","q","marketcap"]])
panel = panel.merge(mc, on=["ticker","q"], how="left")
say(f"Market-cap coverage in panel: {panel['marketcap'].notna().mean():.1%}")

# ── per-ticker split-half means of raw ΔS (delta_s) ─────────────────────────
d = panel.dropna(subset=["delta_s", "marketcap", "q_ord"]).copy()
d["is_odd"] = d["q_ord"] % 2 == 1

recs = []
for tkr, g in d.groupby("ticker"):
    g_odd = g.loc[g["is_odd"], "delta_s"]
    g_even = g.loc[~g["is_odd"], "delta_s"]
    if len(g_odd) < 2 or len(g_even) < 2:
        continue
    recs.append({
        "ticker": tkr,
        "odd_mean": g_odd.mean(),
        "even_mean": g_even.mean(),
        "n_quarters": len(g),
        "median_mktcap": g["marketcap"].median(),
    })
firm_df = pd.DataFrame(recs)
say(f"\nFirms with >=2 odd AND >=2 even quarterly ΔS observations: {len(firm_df):,} "
    f"(of {d['ticker'].nunique():,} total tickers with any ΔS)")

# ── assign global size deciles by each firm's own median point-in-time cap ──
firm_df["decile"] = pd.qcut(firm_df["median_mktcap"], 10, labels=False, duplicates="drop") + 1

say("\n" + "-"*72)
say(f"{'Decile':>7}{'N firms':>10}{'Avg N/qtr':>11}{'Med Cap ($M)':>15}{'Corr(odd,even)':>16}")
say("-"*72)
decile_rows = []
for dec, g in firm_df.groupby("decile"):
    corr = g["odd_mean"].corr(g["even_mean"])
    avg_nq = g["n_quarters"].mean()
    med_cap_m = g["median_mktcap"].median() / 1e6
    say(f"{dec:>7}{len(g):>10}{avg_nq:>11.1f}{med_cap_m:>15,.1f}{corr:>16.3f}")
    decile_rows.append({"decile": dec, "n_firms": len(g), "avg_nq": avg_nq,
                         "med_cap_m": med_cap_m, "corr": corr})
decile_df = pd.DataFrame(decile_rows)

top5 = decile_df[decile_df["decile"] >= 6]["corr"].mean()   # deciles 6-10 = largest half
bot5 = decile_df[decile_df["decile"] <= 5]["corr"].mean()   # deciles 1-5 = smallest half
say("-"*72)
say(f"\nTop-5-decile (largest, deciles 6-10) avg reliability: {top5:.3f}")
say(f"Bottom-5-decile (smallest, deciles 1-5) avg reliability: {bot5:.3f}")
say(f"Spread (top5 - bottom5): {top5-bot5:+.3f}")

flat = abs(top5 - bot5) < 0.15 and (decile_df["corr"].min() >= 0.5 or decile_df["corr"].max() - decile_df["corr"].min() < 0.25)
say(f"\nVerdict: reliability is {'roughly FLAT across deciles' if flat else 'MATERIALLY LOWER in bottom deciles'} "
    f"(bottom-decile threshold check: min={decile_df['corr'].min():.3f} vs >0.75 top-end reference)")

# ═════════════════════════════════════════════════════════════════════════
# Re-estimate t(ΔS) on the top-5-decile (higher-reliability) subsample
# ═════════════════════════════════════════════════════════════════════════
say("\n" + "#"*72); say("# t(ΔS) ON TOP-5-DECILE SUBSAMPLE VS FULL SAMPLE"); say("#"*72)

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

pf_all = panel.dropna(subset=["ret_next", "delta_h_z", "delta_s_z"])
r_all, _ = fama_macbeth_nw(pf_all, "ret_next", ["delta_h_z", "delta_s_z"])
say(f"\nFull sample:      t(ΔH)={r_all['delta_h_z'][1]:+.2f}  t(ΔS)={r_all['delta_s_z'][1]:+.2f}  "
    f"(N={len(pf_all):,}, Tq={r_all['delta_h_z'][2]})")

top5_tickers = set(firm_df.loc[firm_df["decile"] >= 6, "ticker"])
pf_top5 = pf_all[pf_all["ticker"].isin(top5_tickers)]
r_top5, _ = fama_macbeth_nw(pf_top5, "ret_next", ["delta_h_z", "delta_s_z"])
say(f"Top-5-decile only: t(ΔH)={r_top5['delta_h_z'][1]:+.2f}  t(ΔS)={r_top5['delta_s_z'][1]:+.2f}  "
    f"(N={len(pf_top5):,}, Tq={r_top5['delta_h_z'][2]}, tickers={len(top5_tickers):,})")

# also re-z-score within the top-5-decile-only cross-section (since it's a different universe now)
pf_top5_rz = pf_top5.copy()
pf_top5_rz["delta_h_z2"] = cs_wz(pf_top5_rz, "dH_gpm", "q")
pf_top5_rz["delta_s_z2"] = cs_wz(pf_top5_rz, "delta_s", "q")
pf_top5_rz = pf_top5_rz.dropna(subset=["ret_next","delta_h_z2","delta_s_z2"])
r_top5_rz, _ = fama_macbeth_nw(pf_top5_rz, "ret_next", ["delta_h_z2", "delta_s_z2"])
say(f"Top-5-decile, re-z-scored within subsample: t(ΔH)={r_top5_rz['delta_h_z2'][1]:+.2f}  "
    f"t(ΔS)={r_top5_rz['delta_s_z2'][1]:+.2f}  (N={len(pf_top5_rz):,})")

say("\n" + "="*72)
say("VERDICT")
say(f"  {'§4.8/§5.4 gets one sentence; collapse unaffected by attenuation.' if flat else 'Part of the collapse at full breadth is attributable to attenuation; report the top-5-decile figure as the more defensible estimate alongside the full-sample one.'}")

with open(f"{OUT}/D2_size_decile_reliability.txt", "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\nSaved: {OUT}/D2_size_decile_reliability.txt")
