"""T10_RECON.py — Block 2: Table 10 firm-count reconciliation.

Reconciles the three avg-firms/quarter counts that all describe the R18 panel:
  Table 10 full-universe baseline row : 3,807
  Section 3.1 sort panel              : 3,723
  Section 3.1 FM estimation sample    : 3,505
and confirms the ladder baseline reproduces Table 6/7/8 k=0 statistics exactly
(t(dS)=+0.02, t(dH)=+3.46, L/S=-1.0%/yr). Uses R18's rig (NW-4, saved z-scores).
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"


def fama_macbeth_nw(panel, y, xs, dc="q", lags=4, min_cs=20):
    coefs = []
    for d, g in panel.groupby(dc):
        sub = g[[y] + xs].dropna()
        if len(sub) < max(min_cs, len(xs) + 2):
            continue
        X = sm.add_constant(sub[xs], has_constant="add")
        coefs.append(sm.OLS(sub[y], X).fit().params[xs].rename(d))
    cdf = pd.DataFrame(coefs)
    out = {}
    for c in xs:
        s = cdf[c].dropna()
        n = len(s)
        mn = s.mean()
        var = (s**2).mean() - mn**2
        for l in range(1, min(lags + 1, n)):
            var += 2 * (1 - l / (lags + 1)) * ((s.iloc[l:].values - mn) * (s.iloc[:-l].values - mn)).mean()
        out[c] = (mn, mn / np.sqrt(max(var, 1e-30) / n), n)
    return out


def quintile_ls(df, sortcol):
    d = df.dropna(subset=[sortcol, "ret_next"]).copy()
    d["qd"] = d.groupby("q")[sortcol].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") if x.nunique() >= 5 else np.nan)
    d = d.dropna(subset=["qd"])
    qr = d.groupby(["q", "qd"])["ret_next"].mean().unstack("qd")
    ls = (qr.get(4) - qr.get(0)).dropna()
    t = ls.mean() / (ls.std() / np.sqrt(len(ls)))
    return ls.mean() * 400, t, len(ls)


q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
NQ = q["q"].nunique()


def summ(df):
    return df.groupby("q").size().mean(), len(df), df["q"].nunique(), df["ticker"].nunique()


print("=" * 72)
print("[T10-RECON] Table 10 firm-count reconciliation (all = R18 panel)")
print("=" * 72)

# ── the three counts ──────────────────────────────────────────────────────────
base_fpq, base_N = len(q) / NQ, len(q)
sort = q.dropna(subset=["ret_next", "delta_s"])
fm = q.dropna(subset=["ret_next", "delta_h_z", "delta_s_z"])
s_fpq, s_N, s_nq, s_fw = summ(sort)
f_fpq, f_N, f_nq, f_fw = summ(fm)

print(f"  ladder_baseline: firms/qtr={base_fpq:.0f}, N={base_N:,}, quarters={NQ}")
print(f"                   filters = NONE (every firm-quarter row in the R18 panel,")
print(f"                             incl. rows with missing forward return / ΔH)")
print(f"  sort_panel:      firms/qtr={s_fpq:.0f}, N={s_N:,}, quarters={s_nq}, firms={s_fw:,}")
print(f"                   filters = ret_next & delta_s non-missing")
print(f"  fm_sample:       firms/qtr={f_fpq:.0f}, N={f_N:,}, quarters={f_nq}, firms={f_fw:,}")
print(f"                   filters = ret_next & delta_h_z & delta_s_z non-missing (ΔH-complete)")
print(f"  filter diff 3807->3723: -{base_N - s_N:,} firm-qtrs lacking forward return "
      f"(ret_next); last quarter drops out (nq {NQ}->{s_nq})")
print(f"  filter diff 3723->3505: -{s_N - f_N:,} firm-qtrs lacking ΔH history "
      f"(delta_h_z); firms {s_fw:,}->{f_fw:,}")

# ── baseline reproduces Table 7/8 k=0? ────────────────────────────────────────
rb = fama_macbeth_nw(fm.copy(), "ret_next", ["delta_h_z", "delta_s_z"])
ls_yr, ls_t, ls_nq = quintile_ls(q.copy(), "delta_s_z")
t_dh = rb["delta_h_z"][1]
t_ds = rb["delta_s_z"][1]
print("\n  baseline reproduces Table 7/8 k=0?")
print(f"    FM Model B (NW-4, saved z): t(dH)={t_dh:+.2f}  t(dS)={t_ds:+.2f}")
print(f"    ΔS quintile L/S: {ls_yr:+.1f}%/yr (t={ls_t:+.2f}, quarters={ls_nq})")
print(f"    TARGET (Table 6/7/8 k=0): t(dH)=+3.46, t(dS)=+0.02, L/S=-1.0%/yr")
ok = abs(t_dh - 3.46) < 0.1 and abs(t_ds - 0.02) < 0.1 and abs(ls_yr - (-1.0)) < 0.3
print(f"    reproduces T7 k=0: {'YES' if ok else 'NO'}")
