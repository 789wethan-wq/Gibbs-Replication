"""D6_mirror_image_dH.py — t(ΔH) under the 27-year survival requirement, at
each of Table 12's five size cuts (full breadth, top quintile, top decile,
top 500, top 150), with SEs.

Size cuts are ranked against the FULL, UNCONDITIONED universe cross-section
each quarter (same fix applied in D1a/D3/R28: ranking AFTER subsetting to the
small ~328-avg-firm survival-conditioned universe makes fixed-count cuts like
top-500 non-binding -- a bug caught and fixed earlier this session). The
survival filter is then applied as an AND on top of the full-universe-defined
size groups. Lagged (t-1) market cap throughout, per ground rule 4 / D4.4.
"""
import os
import hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT = "../results/revision/D6_mirror_image_dH.txt"

print(f"[pid={os.getpid()}] D6 — fresh process")
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
    for col in x_cols:
        s = cdf[col].dropna()
        n = len(s)
        mean_ = s.mean()
        gamma0 = (s**2).mean() - mean_**2
        var = gamma0
        for l in range(1, min(lags + 1, n)):
            g_ = ((s.iloc[l:].values - mean_) * (s.iloc[:-l].values - mean_)).mean()
            var += 2 * (1 - l / (lags + 1)) * g_
        se = np.sqrt(max(var, 1e-30) / n)
        out[col] = dict(coef=float(mean_), se=float(se), t=float(mean_ / se), n_quarters=int(n))
    return out


panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
panel = panel.sort_values(["ticker", "q_ord"]).reset_index(drop=True)
new_run = (panel["q_ord"] - panel.groupby("ticker")["q_ord"].shift(1)) != 1
panel["run_id"] = new_run.groupby(panel["ticker"]).cumsum()
panel["run_len_q"] = panel.groupby(["ticker", "run_id"])["q_ord"].transform("size")
K = 27
THR_Q = K * 4

sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                       columns=["ticker", "dimension", "calendardate", "marketcap"])
mc = sf1[sf1["dimension"] == "ARQ"].copy()
mc["calendardate"] = pd.to_datetime(mc["calendardate"], errors="coerce")
mc = mc.dropna(subset=["calendardate", "marketcap"])
mc = mc[mc["marketcap"] > 0]
mc["q"] = mc["calendardate"].dt.to_period("Q")
mc = mc.sort_values(["ticker", "calendardate"]).drop_duplicates(["ticker", "q"], keep="last")[["ticker", "q", "marketcap"]]
mc = mc.sort_values(["ticker", "q"])
mc["marketcap_lag1"] = mc.groupby("ticker")["marketcap"].shift(1)
panel = panel.merge(mc, on=["ticker", "q"], how="left")

# rank + full-universe quarterly cross-section size, computed ONCE against the
# unrestricted panel
mc_ranked = panel.dropna(subset=["marketcap_lag1"]).copy()
mc_ranked["mc_rank"] = mc_ranked.groupby("q")["marketcap_lag1"].rank(ascending=False, method="first")
mc_ranked["q_size"] = mc_ranked.groupby("q")["marketcap_lag1"].transform("size")
panel = panel.merge(mc_ranked[["ticker", "q", "mc_rank", "q_size"]], on=["ticker", "q"], how="left")

sub_surv = panel[panel["run_len_q"] >= THR_Q].copy()
P(f"Survival-conditioned universe (k={K}y): {sub_surv['ticker'].nunique()} tickers, "
  f"avg {sub_surv.groupby('q').size().mean():.1f} firms/qtr")

CUTS = [
    ("Full breadth", lambda p: pd.Series(True, index=p.index)),
    ("Top quintile (20%, full-univ rank)", lambda p: p["mc_rank"] <= np.ceil(p["q_size"] * 0.20)),
    ("Top decile (10%, full-univ rank)", lambda p: p["mc_rank"] <= np.ceil(p["q_size"] * 0.10)),
    ("Top-500 (fixed count)", lambda p: p["mc_rank"] <= 500),
    ("Top-150 (fixed count)", lambda p: p["mc_rank"] <= 150),
]

P("\n" + "="*88)
P(f"D6 — t(ΔH) under {K}-year survival requirement, by size cut (lagged cap)")
P("="*88)

rows = []
for label, cutfn in CUTS:
    d = sub_surv[cutfn(sub_surv)].copy()
    d["ds_z"] = cs_wz(d, "delta_s")
    d["dh_z"] = cs_wz(d, "dH_gpm")
    pf = d.dropna(subset=["ret_next", "ds_z", "dh_z"])
    if len(pf) == 0:
        P(f"\n[{label}] EMPTY -- infeasible cell, reported as such (no substitution).")
        rows.append(dict(label=label, infeasible=True))
        continue
    fm = fama_macbeth_nw(pf, "ret_next", ["dh_z", "ds_z"])
    first_q, last_q = str(pf["q"].min()), str(pf["q"].max())
    avg_n = pf.groupby("q").size().mean()
    n_tick = pf["ticker"].nunique()
    X = sm.add_constant(pf[["dh_z", "ds_z"]]).values
    dhash = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()[:12]
    P(f"\n[{label}]")
    P(f"  N={len(pf):,}  tickers={n_tick}  avg firms/qtr={avg_n:.1f}  date_range={first_q}..{last_q}  "
      f"design_shape={X.shape}  hash={dhash}")
    if "dh_z" in fm:
        P(f"  t(dH)={fm['dh_z']['t']:+.4f}  coef(dH)={fm['dh_z']['coef']:+.6f}  SE(dH)={fm['dh_z']['se']:.6f}  "
          f"quarters={fm['dh_z']['n_quarters']}")
        P(f"  t(dS)={fm['ds_z']['t']:+.4f}  coef(dS)={fm['ds_z']['coef']:+.6f}  SE(dS)={fm['ds_z']['se']:.6f}")
        rows.append(dict(label=label, infeasible=False, N=len(pf), avg_n=avg_n, n_tick=n_tick,
                          first_q=first_q, last_q=last_q,
                          t_dh=fm['dh_z']['t'], coef_dh=fm['dh_z']['coef'], se_dh=fm['dh_z']['se'],
                          t_ds=fm['ds_z']['t']))
    else:
        P("  FM regression returned no quarters meeting min_cs -- infeasible cell.")
        rows.append(dict(label=label, infeasible=True))

P("\n" + "="*88)
P("D6 SUMMARY TABLE")
P("="*88)
P(f"{'Size cut':38}{'t(dH)':>9}{'coef(dH)':>12}{'SE(dH)':>10}{'N':>10}{'avg n/q':>9}")
for r in rows:
    if r.get("infeasible"):
        P(f"{r['label']:38}{'INFEASIBLE':>9}")
    else:
        P(f"{r['label']:38}{r['t_dh']:>+9.2f}{r['coef_dh']:>+12.6f}{r['se_dh']:>10.6f}{r['N']:>10,}{r['avg_n']:>9.1f}")

P("\nMirror-image check: §4.8 states stability 'decays in mirror image' under")
P("survival conditioning (full-breadth t(ΔH): +3.46 -> -0.04). This table shows")
P("whether that direction holds, reverses, or is mixed at each size cut --")
P("this is the qualifier needed to state the sentence correctly rather than")
P("only at full breadth.")

with open(OUT, "w") as f:
    f.write("\n".join(log) + "\n")
P(f"\nwrote {OUT}")
