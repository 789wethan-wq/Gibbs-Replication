"""OOS_R18_provenance.py — Block 2: reproduce the §4.6 corrected-panel OOS test.

Provenance: robustness/R21_revision_battery.py section [8] ("OOS quarterly,
survivorship-corrected") is the run that produced the manuscript's §4.6 numbers.
Panel: data/merged_sf1_quarterly_survfree.parquet (the R18 full-universe panel).
The F2 fact-check ("no OOS test on the full-universe/R18 quarterly panel") is
therefore incorrect — this is exactly that test.

This script (a) reproduces R21 [8] verbatim on the current panel, and (b) rebuilds
the manuscript's stated Model-B expanding-window design for comparison.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"


def fm_nw(panel, y, xs, dc, lags=5, min_cs=20):
    C = []
    for d, g in panel.groupby(dc):
        sub = g[[y] + xs].dropna()
        if len(sub) < max(min_cs, len(xs) + 2):
            continue
        X = sm.add_constant(sub[xs], has_constant="add")
        C.append(sm.OLS(sub[y], X).fit().params[xs].rename(d))
    cdf = pd.DataFrame(C)
    out = {}
    for c in xs:
        s = cdf[c].dropna()
        n = len(s)
        mn = s.mean()
        g0 = (s**2).mean() - mn**2
        v = g0
        for l in range(1, min(lags + 1, n)):
            v += 2 * (1 - l / (lags + 1)) * ((s.iloc[l:].values - mn) * (s.iloc[:-l].values - mn)).mean()
        out[c] = (mn, mn / np.sqrt(max(v, 1e-30) / n), n)
    return out


q = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")

# ─── (a) EXACT R21 [8] reproduction: composite delta_g, first-third burn-in ───
print("=" * 72)
print("[OOS-R18] (a) EXACT R21 section [8] reproduction — composite ΔG")
print("=" * 72)
qoos = q.dropna(subset=["ret_next", "delta_g"]).copy().sort_values("q")
quarters = sorted(qoos["q"].unique())
split = quarters[len(quarters) // 3]  # ~first third as burn-in
ls_q = []
for qt in quarters:
    if qt <= split:
        continue
    train = qoos[qoos["q"] < qt]
    if train["q"].nunique() < 12:
        continue
    o = fm_nw(train, "ret_next", ["delta_g"], "q", lags=4)
    sgn = np.sign(o["delta_g"][0]) if "delta_g" in o else 0
    cur = qoos[qoos["q"] == qt]
    if cur["delta_g"].nunique() < 5:
        continue
    cur = cur.copy()
    cur["qd"] = pd.qcut(cur["delta_g"], 5, labels=False, duplicates="drop")
    hi = cur[cur["qd"] == 4]["ret_next"].mean()
    lo = cur[cur["qd"] == 0]["ret_next"].mean()
    ls_q.append((qt, sgn * (hi - lo)))
lsdf = pd.DataFrame(ls_q, columns=["q", "ls"]).dropna()
lsdf["year"] = lsdf["q"].astype(str).str[:4].astype(int)
yr = lsdf.groupby("year")["ls"].mean()
first_oos = str(sorted(lsdf["q"].unique())[0])
t_oos = lsdf["ls"].mean() / (lsdf["ls"].std() / np.sqrt(len(lsdf)))
print(f"  mean L/S %/q={lsdf['ls'].mean()*100:+.2f}, ann={lsdf['ls'].mean()*400:+.1f}%, "
      f"pos years={(yr>0).sum()}/{len(yr)}, pos quarters={(lsdf['ls']>0).sum()}/{len(lsdf)}, "
      f"OOS t={t_oos:+.2f}")
print(f"  first OOS quarter={first_oos}, training minimum=12 quarters (burn-in=first third, "
      f"split at {str(split)})")
print(f"  MANUSCRIPT §4.6 CLAIMS: +1.01%/q (+4.1%/yr), 12/19 yrs, 46/74 q, t=+1.09")

# ─── (b) Manuscript's STATED design: Model-B (ΔH+ΔS) expanding, signal at t+1 ──
print("\n" + "=" * 72)
print("[OOS-R18] (b) Manuscript-stated Model-B design — ΔH_z+ΔS_z expanding window")
print("=" * 72)
qb = q.dropna(subset=["ret_next", "delta_h_z", "delta_s_z"]).copy().sort_values("q")
quarters_b = sorted(qb["q"].unique())
split_b = quarters_b[len(quarters_b) // 3]
ls_b = []
for qt in quarters_b:
    if qt <= split_b:
        continue
    train = qb[qb["q"] < qt]
    if train["q"].nunique() < 12:
        continue
    o = fm_nw(train, "ret_next", ["delta_h_z", "delta_s_z"], "q", lags=4)
    # composite in-sample signal: fitted linear predictor from the two betas
    if "delta_h_z" not in o or "delta_s_z" not in o:
        continue
    bh, bs = o["delta_h_z"][0], o["delta_s_z"][0]
    cur = qb[qb["q"] == qt].copy()
    cur["signal"] = bh * cur["delta_h_z"] + bs * cur["delta_s_z"]
    if cur["signal"].nunique() < 5:
        continue
    cur["qd"] = pd.qcut(cur["signal"], 5, labels=False, duplicates="drop")
    hi = cur[cur["qd"] == 4]["ret_next"].mean()
    lo = cur[cur["qd"] == 0]["ret_next"].mean()
    ls_b.append((qt, hi - lo))  # signal already sign-oriented by the betas
lsb = pd.DataFrame(ls_b, columns=["q", "ls"]).dropna()
lsb["year"] = lsb["q"].astype(str).str[:4].astype(int)
yrb = lsb.groupby("year")["ls"].mean()
first_b = str(sorted(lsb["q"].unique())[0])
t_b = lsb["ls"].mean() / (lsb["ls"].std() / np.sqrt(len(lsb)))
print(f"  mean L/S %/q={lsb['ls'].mean()*100:+.2f}, ann={lsb['ls'].mean()*400:+.1f}%, "
      f"pos years={(yrb>0).sum()}/{len(yrb)}, pos quarters={(lsb['ls']>0).sum()}/{len(lsb)}, "
      f"OOS t={t_b:+.2f}")
print(f"  first OOS quarter={first_b}, training minimum=12 quarters")
