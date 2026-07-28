"""SPEC T6 — Reliability-held-fixed survival ladder.

Per-firm IVOL (ΔS) reliability is too noisy on its own (SB-corrected values
range from -408 to +0.94, mean -0.74 -- see R26_reliability_stratified.py's
docstring); reliability must be estimated as a GROUP statistic pooling many
split-half observations. Following R26's validated approach, market-cap
terciles (POOLED cut across the whole sample, matching R26's grouping
mechanism) are used purely as the grouping device, and reliability is
measured (not assumed) within each group.

This test crosses that reliability-tercile grouping with the R20/M9(b)
survival ladder (k = 0,5,10,15,20,25 years; consecutive-run-length >= 4k
quarters, same construction as M9b_oz_entropy_fixedgrid.py). At each (k,
tercile) cell: restrict the corrected panel to (tercile membership) x
(run-length >= 4k), re-standardize STAB/IVOL within that restricted cell
per quarter (same cs_winsorize_zscore convention), run Model B FM, and
report t(IVOL). Also report the ACHIEVED reliability within that same cell
(pooled odd/even split-half correlation, SB-corrected, from
R26_split_half_obs.parquet restricted identically) and the median market
cap in the cell.

Outputs: results/revision/SPEC_T6_reliability_ladder.txt
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


def cs_winsorize_zscore(df, col, date_col="q", pct=0.01):
    def _wz(x):
        x = x.dropna()
        if len(x) < 5:
            return pd.Series(np.nan, index=x.index)
        lo, hi = x.quantile(pct), x.quantile(1 - pct)
        xc = x.clip(lo, hi)
        std = xc.std()
        if std < 1e-10:
            return pd.Series(np.nan, index=x.index)
        return (xc - xc.mean()) / std
    return df.groupby(date_col)[col].transform(_wz)


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


say("=" * 96)
say("SPEC T6 — RELIABILITY-HELD-FIXED SURVIVAL LADDER")
say("=" * 96)

panel = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet",
                      columns=["ticker", "dimension", "calendardate", "marketcap"])
mc = sf1[sf1["dimension"] == "ARQ"].copy()
mc["calendardate"] = pd.to_datetime(mc["calendardate"], errors="coerce")
mc = mc.dropna(subset=["calendardate", "marketcap"])
mc = mc[mc["marketcap"] > 0]
mc["q"] = mc["calendardate"].dt.to_period("Q")
mc = (mc.sort_values(["ticker", "calendardate"])
        .drop_duplicates(["ticker", "q"], keep="last")[["ticker", "q", "marketcap"]])
panel = panel.merge(mc, on=["ticker", "q"], how="left")

# pooled (not per-quarter) market-cap terciles, matching R26's grouping mechanism
panel["cap_tercile"] = pd.qcut(panel["marketcap"], 3, labels=False, duplicates="drop") + 1
say(f"\nPanel N={len(panel):,}, market-cap coverage={panel['marketcap'].notna().mean():.1%}")
say("Pooled market-cap tercile cutpoints ($M):")
for g in [1, 2, 3]:
    sub = panel[panel["cap_tercile"] == g]
    say(f"  tercile {g}: N={len(sub):,}  cap range=[{sub['marketcap'].min()/1e6:,.1f}, "
        f"{sub['marketcap'].max()/1e6:,.1f}]M  median=${sub['marketcap'].median()/1e6:,.1f}M")

# run-length conditioning (identical to M9b_oz_entropy_fixedgrid.py)
panel = panel.sort_values(["ticker", "q_ord"]).reset_index(drop=True)
new_run = (panel["q_ord"] - panel.groupby("ticker")["q_ord"].shift(1)) != 1
panel["run_id"] = new_run.groupby(panel["ticker"]).cumsum()
panel["run_len_q"] = panel.groupby(["ticker", "run_id"])["q_ord"].transform("size")

# split-half obs for achieved-reliability measurement, same tercile grouping
obs = pd.read_parquet(f"{DATA}/R26_split_half_obs.parquet")
obs = obs.dropna(subset=["marketcap"])
obs = obs[obs["marketcap"] > 0].copy()
obs["cap_tercile"] = pd.qcut(obs["marketcap"], 3, labels=False, duplicates="drop") + 1
# restrict split-half obs to the R18 panel's actual date range to avoid stale-ticker bleed
q_max = panel["q"].max()
obs = obs[obs["q"] <= q_max].copy()
obs = obs.merge(panel[["ticker", "q", "run_len_q"]], on=["ticker", "q"], how="left")

KS = [0, 5, 10, 15, 20, 25, 27]

say("\n" + "-" * 96)
say(f"{'k(yr)':>5} {'tercile':>7} {'t(IVOL)':>9} {'beta_IVOL':>11} {'achieved rel':>13} "
    f"{'medCap($M)':>11} {'N_tick':>7} {'N_obs':>8}")
say("-" * 96)

rows = []
for k in KS:
    thr_q = int(round(4 * k))
    for g in [1, 2, 3]:
        cell = panel[panel["cap_tercile"] == g].copy()
        cell = cell[cell["run_len_q"] >= max(thr_q, 1)] if k > 0 else cell

        cell["ds_z_r"] = cs_winsorize_zscore(cell, "delta_s")
        cell["dh_z_r"] = cs_winsorize_zscore(cell, "dH_gpm")
        pf = cell.dropna(subset=["ret_next", "ds_z_r", "dh_z_r"])
        fm, _ = fama_macbeth_nw(pf, "ret_next", ["dh_z_r", "ds_z_r"])
        i = fm.get("ds_z_r", dict(coef=np.nan, se=np.nan, t=np.nan, n=0))

        obs_cell = obs[(obs["cap_tercile"] == g)]
        if thr_q > 0:
            obs_cell = obs_cell[obs_cell["run_len_q"] >= thr_q]
        r = obs_cell["ds_odd"].corr(obs_cell["ds_even"])
        sb = 2 * r / (1 + r) if np.isfinite(r) and (1 + r) != 0 else np.nan

        med_cap = cell["marketcap"].median()
        n_tick = pf["ticker"].nunique()
        n_obs = len(pf)
        say(f"{k:>5} {g:>7} {i['t']:>+9.3f} {i['coef']:>+11.6f} {sb:>13.3f} "
            f"{med_cap/1e6:>11,.1f} {n_tick:>7,} {n_obs:>8,}")
        rows.append(dict(k=k, tercile=g, t_IVOL=i['t'], beta_IVOL=i['coef'],
                          achieved_reliability=sb, med_cap_m=med_cap/1e6,
                          n_ticker=n_tick, n_obs=n_obs, Tq=i['n']))
    say("")

pd.DataFrame(rows).to_csv(f"{OUT}/SPEC_T6_reliability_ladder.csv", index=False)
with open(f"{OUT}/SPEC_T6_reliability_ladder.txt", "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"[written] {OUT}/SPEC_T6_reliability_ladder.txt")
say(f"[written] {OUT}/SPEC_T6_reliability_ladder.csv")
