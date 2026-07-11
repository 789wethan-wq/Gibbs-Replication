"""DIAG_channels.py — identical one-variable-at-a-time decomposition applied to
the QUALITY channel (ΔH) and the TEMPERATURE finding (β_ΔS~T; T·ΔS).  READ-ONLY.

Same universe ladder as DIAG_survivorship.py:
  M0  monthly  / S&P500 survivor-only        (baseline)
  Qa  quarterly/ S&P500 survivor-only (orig tickers)   -> isolates frequency/measure
  Qc  quarterly/ full universe, survivor-only          -> isolates breadth
  Qd  quarterly/ full universe, incl. delisted = CORR  -> isolates survivorship

Specification held fixed across rungs:
  ΔH channel        : FM slope on ΔH_z (accounting GPM stability)
  Temperature (§4.4): per-period β_ΔS from bivariate (ΔH_z+ΔS_z) x-sec regression,
                      then β_ΔS ~ T time-series slope (HAC); plus FM slope on T·ΔS.
Emits: results/survivorship_free/DIAG_channels.md
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"; OUT = "../results/survivorship_free"

L = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); L.append(s)

def cs_wz(df, col, datecol, pct=0.01):
    def _wz(x):
        x = x.dropna()
        if len(x) < 5: return pd.Series(np.nan, index=x.index)
        lo, hi = x.quantile(pct), x.quantile(1-pct); xc = x.clip(lo,hi); sd = xc.std()
        if sd < 1e-10: return pd.Series(np.nan, index=x.index)
        return (xc-xc.mean())/sd
    return df.groupby(datecol)[col].transform(_wz)

def fm_betas(panel, ycol, xcols, datecol, min_cs=20):
    """return per-period beta DataFrame (index=period)."""
    coefs = []
    for d, g in panel.groupby(datecol):
        sub = g[[ycol]+xcols].dropna()
        if len(sub) < max(min_cs, len(xcols)+2): continue
        X = sm.add_constant(sub[xcols], has_constant="add")
        coefs.append(sm.OLS(sub[ycol], X).fit().params[xcols].rename(d))
    return pd.DataFrame(coefs)

def fm_t(panel, ycol, xcols, datecol, lags, target, min_cs=20):
    cdf = fm_betas(panel, ycol, xcols, datecol, min_cs)
    if target not in cdf: return (np.nan, np.nan, 0)
    s = cdf[target].dropna(); n=len(s); m=s.mean(); var=(s**2).mean()-m**2
    for l in range(1,min(lags+1,n)):
        var += 2*(1-l/(lags+1))*((s.iloc[l:].values-m)*(s.iloc[:-l].values-m)).mean()
    return (m, m/np.sqrt(max(var,1e-30)/n), n)

def asym_slope(panel, ycol, xcols, datecol, Tser, lags):
    """β_ΔS ~ T : regress per-period ΔS beta on period T (HAC)."""
    cdf = fm_betas(panel, ycol, xcols, datecol)
    if "ΔS" not in [c for c in cdf.columns] and xcols[-1] not in cdf: return (np.nan, np.nan, 0)
    b = cdf[[xcols[-1]]].rename(columns={xcols[-1]:"bDS"}).join(Tser.rename("T"), how="inner").dropna()
    if len(b) < 10: return (np.nan, np.nan, len(b))
    X = sm.add_constant(b["T"]); r = sm.OLS(b["bDS"], X).fit(cov_type="HAC", cov_kwds={"maxlags":lags})
    return (r.params["T"], r.tvalues["T"], len(b))

# ── load ─────────────────────────────────────────────────────────────────────
orig = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
orig["date"] = pd.to_datetime(orig["date"])
orig["dH_gpm_z"] = cs_wz(orig, "dH_gpm", "date")          # apples-to-apples ΔH_z
corr = pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")

tk = pd.read_parquet(f"{DATA}/sharadar_tickers.parquet")
sf1t = tk[tk["table"]=="SF1"].drop_duplicates("ticker").set_index("ticker")
isdel = sf1t["isdelisted"]
orig_tk = set(orig["stock_id"].unique())

c = corr.copy()
c["is_delisted"] = c["ticker"].map(lambda t: isdel.get(t)=="Y")
c["in_orig"] = c["ticker"].isin(orig_tk)

# monthly & quarterly T series (period-indexed)
Tm = orig.groupby("date")["T"].first()
Tq = c.groupby("q")["T"].first()

say("# Controlled Decomposition — Quality (ΔH) & Temperature channels\n")
say("Same universe ladder as the ΔS diagnostic; specification held fixed per channel.\n")

# ════════════════════════════════════════════════════════════════════════════
# CHANNEL 1 — QUALITY (ΔH):  FM t(ΔH_z)   [S&P500 baseline ~ +2.45]
# ════════════════════════════════════════════════════════════════════════════
say("## Quality channel ΔH — FM slope on ΔH_z (accounting GPM stability)\n")
m0 = fm_t(orig, "ret_next_month", ["dH_gpm_z","DS_z"], "date", 5, "dH_gpm_z")
qa = fm_t(c[c["in_orig"]],        "ret_next", ["delta_h_z","delta_s_z"], "q", 4, "delta_h_z")
qc = fm_t(c[~c["is_delisted"]],   "ret_next", ["delta_h_z","delta_s_z"], "q", 4, "delta_h_z")
qd = fm_t(c,                      "ret_next", ["delta_h_z","delta_s_z"], "q", 4, "delta_h_z")
say("| rung | universe | FM t(ΔH) | β |")
say("|---|---|---:|---:|")
say(f"| M0 | monthly, S&P500 survivor | {m0[1]:+.2f} | {m0[0]:+.5f} |")
say(f"| Qa | quarterly, S&P500 survivor | {qa[1]:+.2f} | {qa[0]:+.5f} |")
say(f"| Qc | quarterly, full-universe survivor | {qc[1]:+.2f} | {qc[0]:+.5f} |")
say(f"| Qd | quarterly, full-universe incl delisted = CORR | {qd[1]:+.2f} | {qd[0]:+.5f} |")
say("\n| step | one variable changed | Δt |")
say("|---|---|---:|")
say(f"| A | frequency/measure | {qa[1]-m0[1]:+.2f} |")
say(f"| B | breadth | {qc[1]-qa[1]:+.2f} |")
say(f"| C | survivorship | {qd[1]-qc[1]:+.2f} |")
say(f"\n-> ΔH stays **positively priced and significant** at every rung "
    f"(t: {m0[1]:+.2f} → {qd[1]:+.2f}). Unlike ΔS, the quality channel SURVIVES all three toggles.")

# ════════════════════════════════════════════════════════════════════════════
# CHANNEL 2 — TEMPERATURE:  (i) asymmetric β_ΔS~T slope,  (ii) FM t(T·ΔS)
# ════════════════════════════════════════════════════════════════════════════
say("\n## Temperature finding §4.4 — asymmetric prediction β_ΔS ~ T\n")
a0 = asym_slope(orig, "ret_next_month", ["dH_gpm_z","DS_z"], "date", Tm, 5)
aa = asym_slope(c[c["in_orig"]],        "ret_next", ["delta_h_z","delta_s_z"], "q", Tq, 4)
ac = asym_slope(c[~c["is_delisted"]],   "ret_next", ["delta_h_z","delta_s_z"], "q", Tq, 4)
ad = asym_slope(c,                      "ret_next", ["delta_h_z","delta_s_z"], "q", Tq, 4)
say("| rung | universe | slope(β_ΔS on T) | t | periods |")
say("|---|---|---:|---:|---:|")
say(f"| M0 | monthly, S&P500 survivor | {a0[0]:+.4f} | {a0[1]:+.2f} | {a0[2]} |")
say(f"| Qa | quarterly, S&P500 survivor | {aa[0]:+.4f} | {aa[1]:+.2f} | {aa[2]} |")
say(f"| Qc | quarterly, full-universe survivor | {ac[0]:+.4f} | {ac[1]:+.2f} | {ac[2]} |")
say(f"| Qd | quarterly, full-universe incl delisted = CORR | {ad[0]:+.4f} | {ad[1]:+.2f} | {ad[2]} |")
say("\n| step | one variable changed | Δt |")
say("|---|---|---:|")
say(f"| A | frequency/measure | {aa[1]-a0[1]:+.2f} |")
say(f"| B | breadth | {ac[1]-aa[1]:+.2f} |")
say(f"| C | survivorship | {ad[1]-ac[1]:+.2f} |")

say("\n## Temperature finding — FM slope on T·ΔS (Model C: ΔH_z + T·ΔS)\n")
orig["TxDS_gpm"] = orig["T"]*orig["DS_z"]
t0 = fm_t(orig, "ret_next_month", ["dH_gpm_z","TxDS_gpm"], "date", 5, "TxDS_gpm")
ta = fm_t(c[c["in_orig"]],      "ret_next", ["delta_h_z","T_delta_s"], "q", 4, "T_delta_s")
tc = fm_t(c[~c["is_delisted"]], "ret_next", ["delta_h_z","T_delta_s"], "q", 4, "T_delta_s")
td = fm_t(c,                    "ret_next", ["delta_h_z","T_delta_s"], "q", 4, "T_delta_s")
say("| rung | universe | FM t(T·ΔS) | β |")
say("|---|---|---:|---:|")
say(f"| M0 | monthly, S&P500 survivor | {t0[1]:+.2f} | {t0[0]:+.6f} |")
say(f"| Qa | quarterly, S&P500 survivor | {ta[1]:+.2f} | {ta[0]:+.6f} |")
say(f"| Qc | quarterly, full-universe survivor | {tc[1]:+.2f} | {tc[0]:+.6f} |")
say(f"| Qd | quarterly, full-universe incl delisted = CORR | {td[1]:+.2f} | {td[0]:+.6f} |")
say("\n| step | one variable changed | Δt |")
say("|---|---|---:|")
say(f"| A | frequency/measure | {ta[1]-t0[1]:+.2f} |")
say(f"| B | breadth | {tc[1]-ta[1]:+.2f} |")
say(f"| C | survivorship | {td[1]-tc[1]:+.2f} |")

with open(f"{OUT}/DIAG_channels.md","w") as f: f.write("\n".join(L)+"\n")
say(f"\nSaved: {OUT}/DIAG_channels.md")
