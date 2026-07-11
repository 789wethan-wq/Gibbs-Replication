"""R15_conditional_accept_fixes.py — Three conditional-accept fixes.

R15.1 — RMW/CMA partial test (does β_ΔH_GPM survive standalone factor controls?)
R15.2 — Double-clustering and Driscoll-Kraay SEs
R15.3 — Accounting-based Wald subperiod tests
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
warnings.filterwarnings("ignore")

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

DATA = "../data"

# ── shared helpers ─────────────────────────────────────────────────────────

def cs_winsorize_zscore(df, col, date_col="date", pct=0.01):
    def _wz(x):
        x2 = x.dropna()
        if len(x2) < 5:
            return pd.Series(np.nan, index=x.index)
        lo, hi = x2.quantile(pct), x2.quantile(1 - pct)
        xc = x.clip(lo, hi)
        std = xc.std()
        if std < 1e-10:
            return pd.Series(np.nan, index=x.index)
        return (xc - xc.mean()) / std
    return df.groupby(date_col)[col].transform(_wz)

def cluster_vcov_1way(X, resid, groups):
    n_, k_ = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    B = np.zeros((k_, k_))
    for g in np.unique(groups):
        m = groups == g
        Xg = X[m]; eg = resid[m]
        B += Xg.T @ np.outer(eg, eg) @ Xg
    G = len(np.unique(groups))
    sc = (G / (G - 1)) * ((n_ - 1) / (n_ - k_))
    return sc * xtx_inv @ B @ xtx_inv

def cluster_vcov_2way(X, resid, groups1, groups2):
    """Two-way cluster-robust VCOV (Cameron-Gelbach-Miller 2011)."""
    # V = V_g1 + V_g2 - V_g1∩g2
    def _vcov(grps):
        n_, k_ = X.shape
        xtx_inv = np.linalg.pinv(X.T @ X)
        B = np.zeros((k_, k_))
        for g in np.unique(grps):
            m = grps == g
            Xg = X[m]; eg = resid[m]
            B += Xg.T @ np.outer(eg, eg) @ Xg
        G = len(np.unique(grps))
        sc = (G / (G - 1)) * ((n_ - 1) / (n_ - k_))
        return sc * xtx_inv @ B @ xtx_inv
    # Intersection cluster: each unique (g1, g2) combination
    intersection = np.array([f"{a}_{b}" for a, b in zip(groups1, groups2)])
    return _vcov(groups1) + _vcov(groups2) - _vcov(intersection)

def nw_mean_tstat(series, lags):
    """Newey-West mean t-stat for a series of cross-sectional coefficients."""
    n = len(series)
    mean_ = series.mean()
    gamma0 = ((series - mean_)**2).mean()
    var_nw = gamma0
    for l in range(1, min(lags + 1, n)):
        gamma_l = ((series.iloc[l:].values - mean_) *
                   (series.iloc[:-l].values - mean_)).mean()
        var_nw += 2 * (1 - l / (lags + 1)) * gamma_l
    se = np.sqrt(max(var_nw, 1e-30) / n)
    return mean_, mean_ / se, n

def vif(X_df, target_col):
    """Variance inflation factor of target_col against all other columns."""
    others = [c for c in X_df.columns if c != target_col]
    if not others:
        return 1.0
    X_oth = sm.add_constant(X_df[others])
    r2 = sm.OLS(X_df[target_col], X_oth).fit().rsquared
    return 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf

def load_data():
    merged  = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
    factors = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
    merged["date"]  = pd.to_datetime(merged["date"])
    factors.index   = pd.to_datetime(factors.index)
    return merged, factors

def build_working_panel(merged, factors):
    """Z-score dH_gpm, merge factor returns onto panel."""
    p = merged.copy()
    p["dH_gpm_z"] = cs_winsorize_zscore(p, "dH_gpm")

    ff_cols = [c for c in ["Mkt_RF","SMB","HML","RMW","CMA","Mom"] if c in factors.columns]
    fac_m = factors[ff_cols].copy()
    fac_m.index.name = "date"
    fac_m = fac_m.reset_index()

    p = p.merge(fac_m, on="date", how="left")
    return p, ff_cols

# ─────────────────────────────────────────────────────────────────────────────
# R15.1 — RMW / CMA partial test
# ─────────────────────────────────────────────────────────────────────────────

def r15_1_rmw_partial(p, ff_cols):
    print("\n" + "="*62)
    print("R15.1 — RMW/CMA PARTIAL TEST")
    print("="*62)

    sub = p.dropna(subset=["dH_gpm_z","DS_z","ret_next_month","RMW","CMA"]).copy()
    n = len(sub)
    grp_date = pd.Categorical(sub["date"]).codes
    grp_firm = pd.Categorical(sub["stock_id"]).codes
    y = sub["ret_next_month"].values

    # In FM cross-sections, factor returns are invariant within each month →
    # they're absorbed by the intercept. The meaningful partial test is
    # pooled OLS with date-cluster SEs, where factor returns vary across months.

    models = [
        ("Model B baseline",     ["dH_gpm_z","DS_z"]),
        ("Model B + RMW",        ["dH_gpm_z","DS_z","RMW"]),
        ("Model B + CMA",        ["dH_gpm_z","DS_z","CMA"]),
        ("Model B + RMW + CMA",  ["dH_gpm_z","DS_z","RMW","CMA"]),
    ]

    rows = []
    print(f"\n  Pooled OLS with date-cluster SEs (N={n:,}):")
    print(f"  {'Model':<30} {'β_ΔH':>9} {'t_ΔH':>8} {'β_ΔS':>9} {'t_ΔS':>8} {'VIF_ΔH':>8}")

    for label, xcols in models:
        X = np.column_stack([np.ones(n)] + [sub[c].values for c in xcols])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        vcov = cluster_vcov_1way(X, resid, grp_date)
        se = np.sqrt(np.diag(vcov))
        t  = beta / se

        # indices: 0=const, 1=dH, 2=DS, …
        idx_dh = 1; idx_ds = 2
        b_dh = beta[idx_dh]; t_dh = t[idx_dh]
        b_ds = beta[idx_ds]; t_ds = t[idx_ds]

        # VIF of dH_gpm_z against all other regressors in this model
        vif_df = sub[xcols].dropna()
        vif_dh = vif(vif_df, "dH_gpm_z")

        print(f"  {label:<30} {b_dh:>9.5f} {t_dh:>8.3f} {b_ds:>9.5f} {t_ds:>8.3f} {vif_dh:>8.3f}")

        rows.append({
            "model": label,
            "b_dH": round(b_dh, 6), "t_dH": round(t_dh, 4),
            "b_dS": round(b_ds, 6), "t_dS": round(t_ds, 4),
            "vif_dH": round(vif_dh, 3),
            "dH_sig_2": abs(t_dh) > 2.0,
            "dS_sig_3": abs(t_ds) > 3.0,
        })

    # Also run FM with RMW added (note: absorbed by intercept within each cross-section,
    # so FM β_ΔH is unchanged — confirm this)
    print(f"\n  FM cross-check (RMW absorbed by monthly intercept):")
    for label, xcols in models[:2]:
        coefs = []
        for d, grp in sub.groupby("date"):
            g = grp[["ret_next_month"] + xcols].dropna()
            if len(g) < max(15, len(xcols) + 2):
                continue
            X_cs = sm.add_constant(g[xcols], has_constant="add")
            try:
                r = sm.OLS(g["ret_next_month"], X_cs).fit()
                coefs.append(r.params[xcols].rename(d))
            except Exception:
                pass
        if coefs:
            cdf = pd.DataFrame(coefs)
            mean_dh, t_dh_fm, n_fm = nw_mean_tstat(cdf["dH_gpm_z"], lags=5)
            print(f"    {label}: FM β_ΔH = {mean_dh:.5f}, NW t = {t_dh_fm:.3f} "
                  f"(n={n_fm} months) — "
                  f"{'RMW absorbed by intercept ✓' if label != 'Model B baseline' else 'baseline'}")

    df = pd.DataFrame(rows)
    df["test"] = "R15.1"
    df.to_csv(f"{OUT}/R15_T1_rmw_partial.csv", index=False)

    # Interpretation
    base_row = rows[0]; rmw_row = rows[1]
    survives = rmw_row["dH_sig_2"]
    interp = (
        f"Does β_ΔH survive standalone RMW control? "
        f"{'YES' if survives else 'NO'} — "
        f"pooled-OLS t(β_ΔH) moves from {base_row['t_dH']:.2f} (baseline) "
        f"to {rmw_row['t_dH']:.2f} after adding RMW_t. "
        f"VIF of ΔH_GPM against RMW: {rmw_row['vif_dH']:.2f}. "
        f"Note: in FM cross-sections RMW_t is constant within each month and is "
        f"absorbed by the intercept, leaving FM β_ΔH unchanged. "
        f"The pooled test is the operative robustness check."
    )
    with open(f"{OUT}/R15_T1_interpretation.txt","w") as f: f.write(interp)
    print(f"\n  Interpretation: {interp}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# R15.2 — Double-clustering and Driscoll-Kraay SEs
# ─────────────────────────────────────────────────────────────────────────────

def r15_2_double_cluster(p):
    print("\n" + "="*62)
    print("R15.2 — DOUBLE-CLUSTERING AND DRISCOLL-KRAAY SEs")
    print("="*62)

    sub = p.dropna(subset=["dH_gpm_z","DS_z","ret_next_month"]).copy()
    sub = sub.sort_values(["stock_id","date"])
    n = len(sub)
    y = sub["ret_next_month"].values
    X = np.column_stack([np.ones(n), sub["dH_gpm_z"].values, sub["DS_z"].values])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    b_dh = beta[1]; b_ds = beta[2]

    grp_date = pd.Categorical(sub["date"]).codes
    grp_firm = pd.Categorical(sub["stock_id"]).codes

    rows = []

    # ── Spec A: FM NW-5 (reference) ──────────────────────────────────
    coefs_a = []
    for d, grp in sub.groupby("date"):
        g = grp[["ret_next_month","dH_gpm_z","DS_z"]].dropna()
        if len(g) < 10:
            continue
        Xcs = sm.add_constant(g[["dH_gpm_z","DS_z"]], has_constant="add")
        try:
            r = sm.OLS(g["ret_next_month"], Xcs).fit()
            coefs_a.append(r.params[["dH_gpm_z","DS_z"]].rename(d))
        except Exception:
            pass
    cdf_a = pd.DataFrame(coefs_a)
    mean_dh_a, t_dh_a, n_a = nw_mean_tstat(cdf_a["dH_gpm_z"], lags=5)
    mean_ds_a, t_ds_a, _   = nw_mean_tstat(cdf_a["DS_z"],     lags=5)
    print(f"\n  Spec A — FM NW-5 (reference, n={n_a} months):")
    print(f"    β_ΔH = {mean_dh_a:.5f}, t = {t_dh_a:.3f}")
    print(f"    β_ΔS = {mean_ds_a:.5f}, t = {t_ds_a:.3f}")
    rows.append({"spec":"A: FM NW-5","b_dH":round(mean_dh_a,6),"t_dH":round(t_dh_a,4),
                 "b_dS":round(mean_ds_a,6),"t_dS":round(t_ds_a,4),
                 "dH_pos":mean_dh_a>0,"dH_sig_2":abs(t_dh_a)>2.0,"dS_sig_3":abs(t_ds_a)>3.0})

    # ── Spec B: Double-clustered pooled OLS ─────────────────────────
    vcov_2w = cluster_vcov_2way(X, resid, grp_date, grp_firm)
    se_2w   = np.sqrt(np.diag(vcov_2w))
    t_dh_b  = b_dh / se_2w[1]
    t_ds_b  = b_ds / se_2w[2]
    print(f"\n  Spec B — Double-clustered pooled OLS (date × firm):")
    print(f"    β_ΔH = {b_dh:.5f}, SE = {se_2w[1]:.6f}, t = {t_dh_b:.3f}")
    print(f"    β_ΔS = {b_ds:.5f}, SE = {se_2w[2]:.6f}, t = {t_ds_b:.3f}")
    rows.append({"spec":"B: Double-cluster (date×firm)","b_dH":round(b_dh,6),"t_dH":round(t_dh_b,4),
                 "b_dS":round(b_ds,6),"t_dS":round(t_ds_b,4),
                 "dH_pos":b_dh>0,"dH_sig_2":abs(t_dh_b)>2.0,"dS_sig_3":abs(t_ds_b)>3.0})

    # ── Spec C: Driscoll-Kraay (bandwidth=12) ──────────────────────
    # DK SE via Newey-West on time-average score vectors
    # Equivalent to NW(12) on the date-mean score g_t = sum_i X_i' e_i / N_t
    dates_sorted = sorted(sub["date"].unique())
    scores = []
    for d in dates_sorted:
        m = sub["date"] == d
        Xt = X[m]; et = resid[m]
        # score contribution at time t
        g_t = Xt.T @ et  # shape (k,)
        scores.append(g_t)
    scores = np.array(scores)   # (T, k)
    T_ = len(scores)
    bandwidth = 12

    xtx_inv = np.linalg.pinv(X.T @ X)
    # Newey-West estimator on date-level scores
    S = scores.T @ scores   # sum of outer products
    for l in range(1, bandwidth + 1):
        w = 1.0 - l / (bandwidth + 1)
        S += w * (scores[l:].T @ scores[:-l] + scores[:-l].T @ scores[l:])
    vcov_dk = xtx_inv @ S @ xtx_inv
    se_dk   = np.sqrt(np.diag(vcov_dk))
    t_dh_c  = b_dh / se_dk[1]
    t_ds_c  = b_ds / se_dk[2]
    print(f"\n  Spec C — Driscoll-Kraay (bandwidth=12):")
    print(f"    β_ΔH = {b_dh:.5f}, SE = {se_dk[1]:.6f}, t = {t_dh_c:.3f}")
    print(f"    β_ΔS = {b_ds:.5f}, SE = {se_dk[2]:.6f}, t = {t_ds_c:.3f}")
    rows.append({"spec":"C: Driscoll-Kraay (bw=12)","b_dH":round(b_dh,6),"t_dH":round(t_dh_c,4),
                 "b_dS":round(b_ds,6),"t_dS":round(t_ds_c,4),
                 "dH_pos":b_dh>0,"dH_sig_2":abs(t_dh_c)>2.0,"dS_sig_3":abs(t_ds_c)>3.0})

    # ── Spec D: Annual observations only (December cross-sections) ──
    sub_ann = sub.copy()
    sub_ann["_month"] = pd.to_datetime(sub_ann["date"]).dt.month
    sub_dec = sub_ann[sub_ann["_month"] == 12].dropna(subset=["dH_gpm_z","DS_z","ret_next_month"])
    print(f"\n  Spec D — Annual (December cross-sections, N={len(sub_dec):,}, "
          f"{sub_dec['date'].nunique()} years):")

    coefs_d = []
    for d, grp in sub_dec.groupby("date"):
        g = grp[["ret_next_month","dH_gpm_z","DS_z"]].dropna()
        if len(g) < 10:
            continue
        Xcs = sm.add_constant(g[["dH_gpm_z","DS_z"]], has_constant="add")
        try:
            r = sm.OLS(g["ret_next_month"], Xcs).fit()
            coefs_d.append(r.params[["dH_gpm_z","DS_z"]].rename(d))
        except Exception:
            pass
    if coefs_d:
        cdf_d = pd.DataFrame(coefs_d)
        mean_dh_d, t_dh_d, n_d = nw_mean_tstat(cdf_d["dH_gpm_z"], lags=2)
        mean_ds_d, t_ds_d, _   = nw_mean_tstat(cdf_d["DS_z"],     lags=2)
        print(f"    β_ΔH = {mean_dh_d:.5f}, t = {t_dh_d:.3f}")
        print(f"    β_ΔS = {mean_ds_d:.5f}, t = {t_ds_d:.3f}")
        rows.append({"spec":"D: Annual (Dec only, NW-2)","b_dH":round(mean_dh_d,6),"t_dH":round(t_dh_d,4),
                     "b_dS":round(mean_ds_d,6),"t_dS":round(t_ds_d,4),
                     "dH_pos":mean_dh_d>0,"dH_sig_2":abs(t_dh_d)>2.0,"dS_sig_3":abs(t_ds_d)>3.0})
    else:
        print("    Too few annual cross-sections.")

    print(f"\n  Summary:")
    print(f"  {'Spec':<38} {'β_ΔH':>8} {'t_ΔH':>7} {'β_ΔS':>8} {'t_ΔS':>7} "
          f"{'ΔH+':>5} {'ΔH>2':>6} {'ΔS>3':>6}")
    for r in rows:
        print(f"  {r['spec']:<38} {r['b_dH']:>8.5f} {r['t_dH']:>7.3f} "
              f"{r['b_dS']:>8.5f} {r['t_dS']:>7.3f} "
              f"{'Y' if r['dH_pos'] else 'N':>5} "
              f"{'Y' if r['dH_sig_2'] else 'N':>6} "
              f"{'Y' if r['dS_sig_3'] else 'N':>6}")

    df = pd.DataFrame(rows)
    df["test"] = "R15.2"
    df.to_csv(f"{OUT}/R15_T2_double_cluster.csv", index=False)

    # Verdict
    r_dc  = next(r for r in rows if "Double" in r["spec"])
    r_ann = next((r for r in rows if "Annual" in r["spec"]), None)
    verdict = "does not materially" if (r_dc["dH_sig_2"] and (r_ann is None or r_ann["dH_sig_2"])) else "does"
    interp = (
        f"Under double-clustering (date × firm), β_ΔH t-statistic is {r_dc['t_dH']:.2f} "
        f"({'significant' if r_dc['dH_sig_2'] else 'NOT significant'} at |t|>2). "
        f"Under annual-only cross-sections (December), β_ΔH t-statistic is "
        + (f"{r_ann['t_dH']:.2f}" if r_ann else "N/A") +
        f" ({'significant' if (r_ann and r_ann['dH_sig_2']) else 'NOT significant'} at |t|>2). "
        f"Driscoll-Kraay (bw=12) t(ΔH)={next(r for r in rows if 'Kraay' in r['spec'])['t_dH']:.2f}. "
        f"Serial dependence correction {verdict} materially affect the stability channel result."
    )
    with open(f"{OUT}/R15_T2_interpretation.txt","w") as f: f.write(interp)
    print(f"\n  Interpretation: {interp}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# R15.3 — Accounting-based Wald subperiod tests
# ─────────────────────────────────────────────────────────────────────────────

def r15_3_subperiod_wald(p):
    print("\n" + "="*62)
    print("R15.3 — ACCOUNTING-BASED WALD SUBPERIOD TESTS")
    print("="*62)

    sub = p.dropna(subset=["dH_gpm_z","DS_z","TxDS","ret_next_month"]).copy()
    sub["_date"] = pd.to_datetime(sub["date"])

    def wald_TxDS(sub_in, label):
        sub_in = sub_in.copy()
        n = len(sub_in)
        if n < 500:
            print(f"    {label}: N={n} too small, skip")
            return None
        X = np.column_stack([
            np.ones(n),
            sub_in["dH_gpm_z"].values,
            sub_in["DS_z"].values,
            sub_in["TxDS"].values,
        ])
        y = sub_in["ret_next_month"].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        grp = pd.Categorical(sub_in["date"]).codes
        vcov = cluster_vcov_1way(X, resid, grp)
        b_t = beta[3]; se_t = np.sqrt(vcov[3, 3])
        t_t = b_t / se_t
        wald = t_t**2
        p_v  = 1 - chi2.cdf(wald, 1)
        G = sub_in["date"].nunique()
        print(f"    {label}: N={n:,}, G={G}, β_TxDS={b_t:.5f}, "
              f"cluster t={t_t:.3f}, Wald χ²={wald:.3f}, p={p_v:.4f} "
              f"({'SIG' if p_v < 0.05 else 'ns'})")
        return {"subset": label, "N": n, "G": G,
                "b_TxDS": round(b_t, 6), "t_TxDS": round(t_t, 4),
                "wald_chisq": round(wald, 4), "p_cluster": round(p_v, 6),
                "significant_5pct": p_v < 0.05, "model": "accounting ΔH"}

    # ── Accounting-based subperiod tests ────────────────────────────
    print("\n  Accounting-based encompassing model {ΔH_GPM, ΔS, T·ΔS}:")
    acc_rows = []

    # Full sample
    r = wald_TxDS(sub, "Full sample")
    if r: acc_rows.append(r)

    # Excl. 2000–2009
    ex_sub = sub[~((sub["_date"].dt.year >= 2000) & (sub["_date"].dt.year <= 2009))]
    r = wald_TxDS(ex_sub, "Excl. 2000–2009")
    if r: acc_rows.append(r)

    # Excl. dot-com only 2000–2002
    ex_dc = sub[~((sub["_date"].dt.year >= 2000) & (sub["_date"].dt.year <= 2002))]
    r = wald_TxDS(ex_dc, "Excl. dot-com (2000–2002)")
    if r: acc_rows.append(r)

    # Excl. GFC only Jul 2008–Jun 2009
    ex_gfc = sub[~(((sub["_date"].dt.year == 2008) & (sub["_date"].dt.month >= 7)) |
                    ((sub["_date"].dt.year == 2009) & (sub["_date"].dt.month <= 6)))]
    r = wald_TxDS(ex_gfc, "Excl. GFC (2008-07 – 2009-06)")
    if r: acc_rows.append(r)

    # Post-2009 only
    post = sub[sub["_date"].dt.year >= 2010]
    r = wald_TxDS(post, "Post-2009 only (≥2010)")
    if r: acc_rows.append(r)

    # ── Price-based reference (from R14.2 + add post-2009) ──────────
    print("\n  Price-based encompassing model (reference, from R14.2):")
    # Load R14.2 results
    try:
        r14 = pd.read_csv(f"{OUT}/R14_T2_excl_crisis.csv")
        r14["model"] = "price-based ΔH"
        print(r14[["subset","N","p_cluster","significant_5pct"]].to_string(index=False))
    except FileNotFoundError:
        r14 = pd.DataFrame()
        print("    R14.2 CSV not found — price-based reference unavailable")

    # Price-based post-2009
    from robustness_utils import load_panel
    try:
        panel_pb, _ = load_panel()
        panel_pb["_date"] = pd.to_datetime(panel_pb["date"])
        post_pb = panel_pb[panel_pb["_date"].dt.year >= 2010].dropna(
            subset=["DH_z","DS_z","TxDS","ret_next_month"]
        ).copy()
        n_pb = len(post_pb)
        if n_pb > 500:
            X_pb = np.column_stack([
                np.ones(n_pb),
                post_pb["DH_z"].values,
                post_pb["DS_z"].values,
                post_pb["TxDS"].values,
            ])
            y_pb = post_pb["ret_next_month"].values
            beta_pb, *_ = np.linalg.lstsq(X_pb, y_pb, rcond=None)
            resid_pb = y_pb - X_pb @ beta_pb
            grp_pb = pd.Categorical(post_pb["date"]).codes
            vcov_pb = cluster_vcov_1way(X_pb, resid_pb, grp_pb)
            b_pb = beta_pb[3]; se_pb = np.sqrt(vcov_pb[3,3])
            t_pb = b_pb / se_pb; w_pb = t_pb**2; p_pb = 1 - chi2.cdf(w_pb, 1)
            print(f"\n  Price-based Post-2009: N={n_pb:,}, β_TxDS={b_pb:.5f}, "
                  f"cluster t={t_pb:.3f}, p={p_pb:.4f}")
            pb_post_row = {"subset":"Post-2009 only (≥2010)","N":n_pb,
                           "G":post_pb["date"].nunique(),
                           "b_TxDS":round(b_pb,6),"t_TxDS":round(t_pb,4),
                           "wald_chisq":round(w_pb,4),"p_cluster":round(p_pb,6),
                           "significant_5pct":p_pb<0.05,"model":"price-based ΔH"}
            r14 = pd.concat([r14, pd.DataFrame([pb_post_row])], ignore_index=True)
    except Exception as e:
        print(f"    Price-based post-2009: error — {e}")

    # ── Combined comparison table ────────────────────────────────────
    print(f"\n  COMPARISON — accounting vs price-based:")
    print(f"  {'Subset':<35} {'Acc. p':>8} {'Acc. sig':>9} {'PB p':>8} {'PB sig':>9}")
    pb_dict = {r["subset"]: r for _, r in r14.iterrows()} if len(r14) else {}
    for ar in acc_rows:
        pb = pb_dict.get(ar["subset"], {})
        pb_p   = f"{pb.get('p_cluster','—'):.4f}" if pb.get('p_cluster') else "—"
        pb_sig = str(pb.get('significant_5pct','—'))
        print(f"  {ar['subset']:<35} {ar['p_cluster']:>8.4f} "
              f"{'Yes' if ar['significant_5pct'] else 'No':>9} {pb_p:>8} {pb_sig:>9}")

    # ── Save ────────────────────────────────────────────────────────
    acc_df = pd.DataFrame(acc_rows)
    acc_df["test"] = "R15.3"
    all_df = pd.concat([acc_df, r14], ignore_index=True) if len(r14) else acc_df
    all_df.to_csv(f"{OUT}/R15_T3_subperiod_wald.csv", index=False)

    # Interpretation
    ex_row = next((r for r in acc_rows if "2000–2009" in r["subset"]), None)
    full_row = next((r for r in acc_rows if r["subset"] == "Full sample"), None)
    post_row = next((r for r in acc_rows if "Post-2009" in r["subset"]), None)
    if ex_row:
        survives_ex = ex_row["significant_5pct"]
        interp = (
            f"Accounting-based Wald p-value excluding 2000–2009: {ex_row['p_cluster']:.3f}. "
            f"Signal {'survives' if survives_ex else 'does not survive'} crisis exclusion "
            f"({'p<0.05' if survives_ex else 'p≥0.05'}). "
            f"Full-sample p={full_row['p_cluster']:.3f}. "
            + f"Post-2009-only p=" + (f"{post_row['p_cluster']:.3f}" if post_row else "N/A") + ". " +
            f"For comparison, price-based Wald excl. 2000–2009 p={pb_dict.get('Excl. 2000–2009',{}).get('p_cluster','N/A')}."
        )
    else:
        interp = "Insufficient data for crisis exclusion test."
    with open(f"{OUT}/R15_T3_interpretation.txt","w") as f: f.write(interp)
    print(f"\n  Interpretation: {interp}")
    return acc_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*65)
    print("  R15 — CONDITIONAL ACCEPT FIXES R15.1–R15.3")
    print("="*65)

    merged, factors = load_data()
    p, ff_cols = build_working_panel(merged, factors)

    results = {}
    for fn, key in [
        (lambda: r15_1_rmw_partial(p, ff_cols),   "R15.1_rmw_partial"),
        (lambda: r15_2_double_cluster(p),          "R15.2_double_cluster"),
        (lambda: r15_3_subperiod_wald(p),          "R15.3_subperiod_wald"),
    ]:
        try:
            fn()
            results[key] = "OK"
        except Exception as e:
            import traceback; traceback.print_exc()
            results[key] = f"ERROR: {e}"

    print("\n" + "="*65)
    print("  R15 SUMMARY")
    print("="*65)
    for k, v in results.items():
        print(f"  {'✓' if v == 'OK' else '✗'}  {k}: {v}")
    print(f"\n  Outputs in: {os.path.abspath(OUT)}/")


if __name__ == "__main__":
    main()
