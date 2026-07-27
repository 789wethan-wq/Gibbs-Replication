"""D3 — Turnover, transaction costs, and risk-adjusted metrics for the ΔH and
ΔS quintile long-shorts, both panels.

Convention (stated explicitly for auditability): for an equal-weighted L/S
quintile portfolio, one-way turnover_t = average, across the long and short
legs, of (names entering + names leaving) / (2 * average leg size) between
period t-1 and t. Cost-adjusted return_t = raw_return_t - turnover_t * cost_bps,
i.e. cost_bps is charged once per unit of the portfolio turned over (a single
round-trip execution cost per replaced name, not double-counted for entry and
exit separately). This is the simplest standard convention and is stated here
so the numbers are auditable against a different convention if needed.

Sharpe/Sortino are computed directly on the (zero-investment) L/S return
series, annualized by sqrt(periods/year); Sortino uses downside deviation
against a 0 target. Max drawdown is computed on cumulative compounded wealth
(1+r).cumprod(), with trough and recovery dates reported.

Outputs: results/revision/D3_turnover_costs_risk.txt
"""
import os, warnings
import numpy as np
import pandas as pd
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

def build_quintile_membership(panel, sortcol, retcol, entitycol, datecol):
    """Returns per-period (long_set, short_set, long_ret, short_ret)."""
    out = {}
    for d, g in panel.dropna(subset=[sortcol, retcol]).groupby(datecol):
        if g[sortcol].nunique() < 5 or len(g) < 25:
            continue
        try:
            q = pd.qcut(g[sortcol], 5, labels=False, duplicates="drop")
        except Exception:
            continue
        if q.max() < 4:
            continue
        longs = set(g.loc[q == 4, entitycol])
        shorts = set(g.loc[q == 0, entitycol])
        long_ret = g.loc[q == 4, retcol].mean()
        short_ret = g.loc[q == 0, retcol].mean()
        out[d] = (longs, shorts, long_ret, short_ret)
    return out

def turnover_series(membership, dates):
    """One-way turnover per period (avg of long-leg and short-leg turnover)."""
    turns = {}
    prev_long, prev_short = None, None
    for d in dates:
        longs, shorts, _, _ = membership[d]
        if prev_long is not None:
            l_chg = len(longs - prev_long) + len(prev_long - longs)
            l_avg = (len(longs) + len(prev_long)) / 2
            s_chg = len(shorts - prev_short) + len(prev_short - shorts)
            s_avg = (len(shorts) + len(prev_short)) / 2
            l_to = l_chg / (2 * l_avg) if l_avg > 0 else np.nan
            s_to = s_chg / (2 * s_avg) if s_avg > 0 else np.nan
            turns[d] = np.nanmean([l_to, s_to])
        prev_long, prev_short = longs, shorts
    return pd.Series(turns).sort_index()

def max_drawdown(ret_series):
    wealth = (1 + ret_series).cumprod()
    running_max = wealth.cummax()
    dd = wealth / running_max - 1
    trough_idx = dd.idxmin()
    mdd = dd.loc[trough_idx]
    peak_val = running_max.loc[trough_idx]
    recovery_idx = None
    after = wealth.loc[trough_idx:]
    recovered = after[after >= peak_val]
    if len(recovered) > 1:
        recovery_idx = recovered.index[1] if recovered.index[0] == trough_idx else recovered.index[0]
    return mdd, trough_idx, recovery_idx

def risk_report(tag, ls_ret, periods_per_year, membership, dates, cost_bps, label_cost):
    ls_ret = ls_ret.dropna()
    turns = turnover_series(membership, dates).reindex(ls_ret.index).dropna()
    common = ls_ret.index.intersection(turns.index)
    ls_ret_c = ls_ret.loc[common]
    turns_c = turns.loc[common]
    net_ret = ls_ret_c - turns_c * cost_bps

    ann_turnover = turns_c.mean() * periods_per_year
    ann_ret_raw = (1 + ls_ret_c.mean()) ** periods_per_year - 1
    ann_ret_net = (1 + net_ret.mean()) ** periods_per_year - 1
    ann_vol_raw = ls_ret_c.std() * np.sqrt(periods_per_year)
    ann_vol_net = net_ret.std() * np.sqrt(periods_per_year)
    sharpe_raw = ls_ret_c.mean() / ls_ret_c.std() * np.sqrt(periods_per_year) if ls_ret_c.std() > 0 else np.nan
    sharpe_net = net_ret.mean() / net_ret.std() * np.sqrt(periods_per_year) if net_ret.std() > 0 else np.nan
    downside_raw = ls_ret_c[ls_ret_c < 0].std()
    downside_net = net_ret[net_ret < 0].std()
    sortino_raw = ls_ret_c.mean() / downside_raw * np.sqrt(periods_per_year) if downside_raw and downside_raw > 0 else np.nan
    sortino_net = net_ret.mean() / downside_net * np.sqrt(periods_per_year) if downside_net and downside_net > 0 else np.nan
    mdd_raw, trough_raw, rec_raw = max_drawdown(ls_ret_c)
    mdd_net, trough_net, rec_net = max_drawdown(net_ret)

    say(f"\n  {tag}  [cost = {label_cost}]")
    say(f"    N periods={len(ls_ret_c)}  Ann one-way turnover={ann_turnover:.1%}")
    say(f"    Raw:  Ann ret={ann_ret_raw:+.2%}  Ann vol={ann_vol_raw:.2%}  "
        f"Sharpe={sharpe_raw:+.2f}  Sortino={sortino_raw:+.2f}  MaxDD={mdd_raw:.1%} "
        f"(trough {trough_raw}, {'recovered ' + str(rec_raw) if rec_raw is not None else 'not recovered by end of sample'})")
    say(f"    Net:  Ann ret={ann_ret_net:+.2%}  Ann vol={ann_vol_net:.2%}  "
        f"Sharpe={sharpe_net:+.2f}  Sortino={sortino_net:+.2f}  MaxDD={mdd_net:.1%} "
        f"(trough {trough_net}, {'recovered ' + str(rec_net) if rec_net is not None else 'not recovered by end of sample'})")
    survives = ann_ret_net > 0
    say(f"    Survives {label_cost} one-way cost: {'YES' if survives else 'NO'} "
        f"(annualized net return {'positive' if survives else 'negative'})")
    return dict(tag=tag, ann_turnover=ann_turnover, ann_ret_raw=ann_ret_raw, ann_ret_net=ann_ret_net,
                sharpe_raw=sharpe_raw, sharpe_net=sharpe_net, sortino_raw=sortino_raw, sortino_net=sortino_net,
                mdd_raw=mdd_raw, mdd_net=mdd_net, survives=survives)

say("="*72); say("D3 — TURNOVER, TRANSACTION COSTS, RISK-ADJUSTED METRICS"); say("="*72)
say("\nConvention: one-way turnover = avg leg name-turnover; net_t = raw_t - turnover_t*cost_bps "
    "(cost charged once per unit turned over). See script docstring for full statement.")

# ═════════════════════════════════════════════════════════════════════════
# S&P 500 monthly panel (large-cap -> 10 bps one-way)
# ═════════════════════════════════════════════════════════════════════════
say("\n" + "#"*72); say("# S&P 500 MONTHLY PANEL (10 bps one-way cost)"); say("#"*72)
sp = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
sp["date"] = pd.to_datetime(sp["date"])
sp["dH_gpm_z"] = cs_wz(sp, "dH_gpm", "date")

for sortcol, label in [("dH_gpm_z", "ΔH (GPM stability)"), ("DS_z", "ΔS (idiosyncratic vol)")]:
    mem = build_quintile_membership(sp, sortcol, "ret_next_month", "stock_id", "date")
    dates = sorted(mem.keys())
    ls = pd.Series({d: mem[d][2] - mem[d][3] for d in dates}).sort_index()
    risk_report(f"S&P 500, sort on {label}", ls, 12, mem, dates, 0.0010, "10 bps")

# ═════════════════════════════════════════════════════════════════════════
# R18 full-universe quarterly panel (microcap -> 50 bps one-way, explicit)
# ═════════════════════════════════════════════════════════════════════════
say("\n" + "#"*72); say("# R18 FULL-UNIVERSE QUARTERLY PANEL (50 bps one-way cost — median cap $401M)"); say("#"*72)
q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")

for sortcol, label in [("delta_h_z", "ΔH (GPM stability)"), ("delta_s_z", "ΔS (idiosyncratic vol)")]:
    mem = build_quintile_membership(q, sortcol, "ret_next", "ticker", "q")
    dates = sorted(mem.keys())
    ls = pd.Series({d: mem[d][2] - mem[d][3] for d in dates}).sort_index()
    risk_report(f"R18 full-universe, sort on {label}", ls, 4, mem, dates, 0.0050, "50 bps")

say("\n" + "#"*72); say("# R18 FULL-UNIVERSE, 10 bps ONE-WAY (for comparability with S&P panel)"); say("#"*72)
for sortcol, label in [("delta_h_z", "ΔH (GPM stability)"), ("delta_s_z", "ΔS (idiosyncratic vol)")]:
    mem = build_quintile_membership(q, sortcol, "ret_next", "ticker", "q")
    dates = sorted(mem.keys())
    ls = pd.Series({d: mem[d][2] - mem[d][3] for d in dates}).sort_index()
    risk_report(f"R18 full-universe, sort on {label}", ls, 4, mem, dates, 0.0010, "10 bps")

say("\n" + "#"*72); say("# S&P 500, 50 bps ONE-WAY (for symmetry)"); say("#"*72)
for sortcol, label in [("dH_gpm_z", "ΔH (GPM stability)"), ("DS_z", "ΔS (idiosyncratic vol)")]:
    mem = build_quintile_membership(sp, sortcol, "ret_next_month", "stock_id", "date")
    dates = sorted(mem.keys())
    ls = pd.Series({d: mem[d][2] - mem[d][3] for d in dates}).sort_index()
    risk_report(f"S&P 500, sort on {label}", ls, 12, mem, dates, 0.0050, "50 bps")

with open(f"{OUT}/D3_turnover_costs_risk.txt", "w") as f:
    f.write("\n".join(LOG) + "\n")
say(f"\nSaved: {OUT}/D3_turnover_costs_risk.txt")
