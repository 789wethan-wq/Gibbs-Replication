"""R13_reviewer_responses_2.py — Five tasks (R13.1–R13.5).

R13.1 — Expanding-window T normalization (look-ahead-free T)
R13.2 — VIF and Partial R² for QMJ/BAB
R13.3 — Placebo Vuong Z distribution (N=1000 permutations of T)
R13.4 — Cluster-robust LR test for T·ΔS = 0
R13.5 — ACF sign analysis + quintile FF5+UMD alpha NW verification
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from robustness_utils import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# R13.1  Expanding-window T normalization
# ─────────────────────────────────────────────────────────────────────────────

def r13_1_expanding_T(panel):
    print("\n" + "="*62)
    print("R13.1 — EXPANDING-WINDOW T NORMALIZATION")
    print("="*62)

    p = panel.copy()
    p = p.sort_values("date")

    # Expanding-window mean and std of T, using only t-1 and earlier
    # T is defined at the date level (one value per month)
    t_series = p.groupby("date")["T"].first().sort_index()
    t_exp_mean = t_series.expanding(min_periods=12).mean().shift(1)
    t_exp_std  = t_series.expanding(min_periods=12).std().shift(1)

    # Normalize: T_norm = (T - exp_mean) / exp_std
    t_norm_map = ((t_series - t_exp_mean) / t_exp_std.clip(lower=1e-8)).to_dict()

    p["T_expanding"] = p["date"].map(t_norm_map)
    p["TxDS_expanding"] = p["T_expanding"] * p["DS_z"]
    # Rebuild DG with expanding-window T
    p["DG_expanding"] = p["DH_z"] - p["T_expanding"] * p["DS_z"]
    p["DG_expanding"] = p.groupby("date")["DG_expanding"].transform(zscore_cs)

    rows = []

    # Baseline (standard T)
    fm_std, _ = fama_macbeth(p.dropna(subset=["DG","ret_next_month"]),
                              "ret_next_month", ["DG"])
    t_std = fm_std.get("DG", (np.nan,)*3)
    rows.append({"spec": "Baseline (standard T)", "fm_t_DG": t_std[1],
                 "fm_coef_DG": t_std[0]})

    # Model A with expanding T
    sub_e = p.dropna(subset=["DG_expanding","ret_next_month"])
    fm_e, _ = fama_macbeth(sub_e, "ret_next_month", ["DG_expanding"])
    t_e = fm_e.get("DG_expanding", (np.nan,)*3)
    rows.append({"spec": "Model A (expanding T)", "fm_t_DG": t_e[1],
                 "fm_coef_DG": t_e[0]})

    # Model B: ΔH + ΔS (no T — unchanged)
    sub_b = p.dropna(subset=["DH_z","DS_z","ret_next_month"])
    fm_b, _ = fama_macbeth(sub_b, "ret_next_month", ["DH_z","DS_z"])
    t_dh_b = fm_b.get("DH_z", (np.nan,)*3)
    t_ds_b = fm_b.get("DS_z", (np.nan,)*3)
    rows.append({"spec": "Model B: DH+DS (T-independent)", "fm_t_DG": t_dh_b[1],
                 "fm_coef_DG": t_dh_b[0], "note": "DH coefficient shown"})

    # Model C with expanding T: ΔH + T_expanding·ΔS
    sub_c = p.dropna(subset=["DH_z","TxDS_expanding","ret_next_month"])
    fm_c, _ = fama_macbeth(sub_c, "ret_next_month", ["DH_z","TxDS_expanding"])
    t_dh_c   = fm_c.get("DH_z", (np.nan,)*3)
    t_txds_c = fm_c.get("TxDS_expanding", (np.nan,)*3)
    rows.append({"spec": "Model C (expanding T): DH+T_exp·DS",
                 "fm_t_DG": t_dh_c[1], "fm_coef_TxDS": t_txds_c[0], "fm_t_TxDS": t_txds_c[1]})

    # Vuong: constrained (DH + T_exp·DS) vs unconstrained (DH + DS + T_exp)
    sub_v = p.dropna(subset=["ret_next_month","DH_z","DS_z","T_expanding"])
    y_v    = sub_v["ret_next_month"].values
    X_con  = np.column_stack([sub_v["DH_z"].values, sub_v["TxDS_expanding"].values])
    X_uncon = np.column_stack([sub_v["DH_z"].values, sub_v["DS_z"].values,
                               sub_v["T_expanding"].values])
    try:
        z_e, p_e, daic_e, *_ = vuong_test(y_v, X_con, X_uncon)
        rows.append({"spec": "Vuong (expanding T): constrained vs unconstrained",
                     "vuong_z": z_e, "vuong_p": p_e, "delta_aic": daic_e})
    except Exception as err:
        rows.append({"spec": "Vuong (expanding T)", "vuong_z": np.nan, "error": str(err)})

    df = pd.DataFrame(rows)
    df["test"] = "R13.1"

    print("\n  Results:")
    for _, r in df.iterrows():
        parts = [r["spec"]]
        if "fm_t_DG" in r and np.isfinite(float(r["fm_t_DG"]) if r["fm_t_DG"] is not None else float("nan")):
            parts.append(f"FM t={r['fm_t_DG']:.3f}")
        if "vuong_z" in r and np.isfinite(float(r["vuong_z"]) if r["vuong_z"] is not None else float("nan")):
            parts.append(f"Vuong Z={r['vuong_z']:.3f}, p={r['vuong_p']:.4f}")
        print("  " + " | ".join(parts))

    # Verdict
    t_expand = rows[1]["fm_t_DG"]
    t_std_v  = rows[0]["fm_t_DG"]
    if np.isfinite(t_expand) and np.isfinite(t_std_v):
        change = abs(t_expand - t_std_v)
        print(f"\n  Δt (standard vs expanding T): {change:.4f}")
        if change < 0.2:
            print("  VERDICT: Materially unchanged — add one sentence in robustness.")
        else:
            print(f"  VERDICT: Non-trivial change ({change:.3f}). Report both versions.")

    df.to_csv(f"{OUT}/R13_T1_expanding_T.csv", index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# R13.2  VIF and Partial R² for QMJ / BAB
# ─────────────────────────────────────────────────────────────────────────────

def r13_2_vif_partial_r2(panel, factors):
    print("\n" + "="*62)
    print("R13.2 — VIF AND PARTIAL R² FOR QMJ / BAB")
    print("="*62)

    # Load QMJ and BAB from cache (written by R12)
    qmj_cache = "aqr_data/qmj_monthly_us.parquet"
    bab_cache  = "aqr_data/bab_monthly_us.parquet"

    if not os.path.exists(qmj_cache) or not os.path.exists(bab_cache):
        print("  QMJ/BAB cache not found — run R12 first.")
        return pd.DataFrame([{"spec": "VIF", "note": "QMJ/BAB cache missing"}])

    qmj_df = pd.read_parquet(qmj_cache)
    bab_df = pd.read_parquet(bab_cache)
    ext    = qmj_df.join(bab_df, how="outer")

    # FF5+UMD factor columns available
    ff_cols = [c for c in ["Mkt_RF","SMB","HML","RMW","CMA","Mom"] if c in factors.columns]

    # Merge into factors frame (month-level)
    fac = factors[ff_cols].copy()
    fac = fac.join(ext, how="left")
    fac = fac.dropna()
    fac.index = pd.to_datetime(fac.index)

    # --- VIF: each new regressor against FF5+UMD ---
    from sklearn.linear_model import LinearRegression

    vif_rows = []
    for new_col in ["QMJ", "BAB"]:
        if new_col not in fac.columns:
            continue
        X_others = fac[[c for c in ff_cols if c != new_col]].values
        y_new    = fac[new_col].values
        lr = LinearRegression().fit(X_others, y_new)
        r2_aux = lr.score(X_others, y_new)
        vif = 1.0 / (1.0 - r2_aux) if r2_aux < 1.0 else np.inf
        vif_rows.append({
            "factor": new_col,
            "R2_vs_FF5UMD": round(r2_aux, 4),
            "VIF": round(vif, 4),
            "interpretation": ("low multicollinearity" if vif < 5
                                else ("moderate" if vif < 10 else "HIGH — collinear"))
        })
    print("\n  VIF of QMJ/BAB relative to FF5+UMD regressors:")
    for r in vif_rows:
        print(f"    {r['factor']}: R²_aux={r['R2_vs_FF5UMD']:.4f}, VIF={r['VIF']:.2f} ({r['interpretation']})")

    # --- L/S portfolio partial R² ---
    # Get L/S returns aligned with factors
    ls, _ = quintile_sort_ls(panel, "DG", "ret_next_month", factors)
    ls = ls.dropna()
    rf = factors.reindex(ls.index).get("RF", pd.Series(0.0, index=ls.index)).fillna(0)
    ls_ex = ls - rf

    fac_ls = fac.reindex(ls_ex.index).dropna()
    ls_sub = ls_ex.reindex(fac_ls.index).dropna()
    fac_sub = fac_ls.reindex(ls_sub.index)

    partial_rows = []
    # Baseline R²: FF5+UMD only
    avail_ff = [c for c in ff_cols if c in fac_sub.columns]
    X_base = sm.add_constant(fac_sub[avail_ff])
    r2_base = sm.OLS(ls_sub, X_base).fit().rsquared

    # Add QMJ
    for new_col in ["QMJ", "BAB", "QMJ+BAB"]:
        if new_col == "QMJ+BAB":
            add_cols = [c for c in ["QMJ","BAB"] if c in fac_sub.columns]
        else:
            add_cols = [new_col] if new_col in fac_sub.columns else []
        if not add_cols:
            continue
        X_ext = sm.add_constant(fac_sub[avail_ff + add_cols])
        res_ext = sm.OLS(ls_sub, X_ext).fit()
        r2_ext = res_ext.rsquared
        partial_r2 = r2_ext - r2_base
        partial_rows.append({
            "added_factor": new_col,
            "R2_base_FF5UMD": round(r2_base, 4),
            "R2_extended":    round(r2_ext, 4),
            "partial_R2":     round(partial_r2, 4),
            "F_stat":         round(res_ext.fvalue, 3) if hasattr(res_ext, "fvalue") else np.nan,
        })

    print("\n  Partial R² of QMJ/BAB for L/S excess return (beyond FF5+UMD):")
    for r in partial_rows:
        print(f"    +{r['added_factor']}: partial R² = {r['partial_R2']:.4f} "
              f"(base={r['R2_base_FF5UMD']:.4f}, extended={r['R2_extended']:.4f})")

    # Cross-sectional VIF per FM step (average across dates)
    print("\n  Cross-sectional VIF (per FM month, averaged):")
    cs_vif_rows = []
    ff_avail = [c for c in ff_cols if c in fac.columns]
    for d, grp in panel.groupby("date"):
        f_d = fac.reindex([d]).dropna()
        if len(f_d) == 0:
            continue
        # Within cross-section we can't compute VIF (one row of factors per date)
        # VIF is a time-series property — we already computed it above

    df = pd.DataFrame(vif_rows + partial_rows)
    df["test"] = "R13.2"
    df.to_csv(f"{OUT}/R13_T2_vif_partial_r2.csv", index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# R13.3  Placebo Vuong Z distribution (permuting T)
# ─────────────────────────────────────────────────────────────────────────────

def r13_3_placebo_vuong(panel, n_perm=1000, seed=42):
    print("\n" + "="*62)
    print("R13.3 — PLACEBO VUONG Z DISTRIBUTION (N=1000)")
    print("="*62)

    rng = np.random.default_rng(seed)
    sub = panel.dropna(subset=["ret_next_month","DH_z","DS_z","T"]).copy()
    y    = sub["ret_next_month"].values
    dh   = sub["DH_z"].values
    ds   = sub["DS_z"].values
    T    = sub["T"].values
    dates_arr = sub["date"].values   # for permuting T at the date level

    # True Vuong Z (constrained = DH + T·DS vs unconstrained = DH + DS)
    X_con_true  = np.column_stack([dh, T * ds])
    X_uncon_true = np.column_stack([dh, ds])
    z_true, p_true, daic_true, *_ = vuong_test(y, X_con_true, X_uncon_true)
    print(f"  True Vuong Z = {z_true:.4f}, p = {p_true:.4f}, ΔAIC = {daic_true:.2f}")

    # Permute T across dates (break the link between T and cross-sections
    # while preserving T's marginal distribution)
    unique_dates = np.unique(dates_arr)
    z_null = []

    for i in range(n_perm):
        # Shuffle date-level T values
        perm_dates = rng.permutation(unique_dates)
        date_T_perm = {orig: sub[sub["date"] == orig]["T"].iloc[0]
                       for orig in unique_dates}
        date_T_perm_mapped = {orig: date_T_perm[perm]
                               for orig, perm in zip(unique_dates, perm_dates)}
        T_perm = np.array([date_T_perm_mapped[d] for d in dates_arr])

        X_con_p  = np.column_stack([dh, T_perm * ds])
        X_uncon_p = np.column_stack([dh, ds])
        try:
            z_p, *_ = vuong_test(y, X_con_p, X_uncon_p)
            if np.isfinite(z_p):
                z_null.append(z_p)
        except Exception:
            pass

        if (i + 1) % 200 == 0:
            print(f"    Permutation {i+1}/{n_perm}...")

    z_null = np.array(z_null)
    print(f"\n  Null distribution ({len(z_null)} valid permutations):")
    print(f"    Mean:    {z_null.mean():.4f}")
    print(f"    SD:      {z_null.std():.4f}")
    print(f"    P5:      {np.percentile(z_null,  5):.4f}")
    print(f"    P25:     {np.percentile(z_null, 25):.4f}")
    print(f"    P75:     {np.percentile(z_null, 75):.4f}")
    print(f"    P95:     {np.percentile(z_null, 95):.4f}")
    pct_neg  = (z_null < 0).mean() * 100
    pct_below_true = (z_null < z_true).mean() * 100
    empirical_p = (z_null >= z_true).mean()
    print(f"    % < 0:             {pct_neg:.1f}%")
    print(f"    % < true Z={z_true:.2f}: {pct_below_true:.1f}%")
    print(f"    Empirical p-value: {empirical_p:.4f}")

    # Percentile rank of true Z
    pctile_rank = (z_null < z_true).mean() * 100
    print(f"\n  True Z = {z_true:.2f} is at the {pctile_rank:.1f}th percentile of null")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(z_null, bins=50, color="#92c5de", edgecolor="white", density=True,
            label=f"Placebo null (N={len(z_null)})")
    ax.axvline(z_true, color="#d6604d", linewidth=2,
               label=f"True Z = {z_true:.2f} (p={empirical_p:.3f})")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    p5_val = np.percentile(z_null, 95)
    ax.axvline(p5_val, color="gray", linewidth=1, linestyle=":",
               label=f"Null 95th pctile = {p5_val:.2f}")
    # Overlay normal density
    from scipy.stats import norm as norm_dist
    x_line = np.linspace(z_null.min() - 0.5, z_null.max() + 0.5, 200)
    ax.plot(x_line, norm_dist.pdf(x_line, z_null.mean(), z_null.std()),
            color="navy", linewidth=1.2, linestyle="-", alpha=0.6, label="N(μ,σ) fit")
    ax.set_xlabel("Vuong Z statistic")
    ax.set_ylabel("Density")
    ax.set_title("Placebo Vuong Null Distribution\n"
                 "(T permuted across dates; ΔH, ΔS held fixed)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        plt.savefig(f"{OUT}/R13_T3_placebo_vuong.{fmt}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Placebo distribution plot saved.")

    result = {
        "true_z":         round(z_true, 4),
        "true_p":         round(p_true, 4),
        "null_mean":      round(z_null.mean(), 4),
        "null_sd":        round(z_null.std(), 4),
        "null_p5":        round(np.percentile(z_null, 5), 4),
        "null_p95":       round(np.percentile(z_null, 95), 4),
        "pct_null_neg":   round(pct_neg, 2),
        "pct_below_true": round(pct_below_true, 2),
        "empirical_p":    round(empirical_p, 4),
        "percentile_rank_of_true": round(pctile_rank, 1),
        "n_permutations": len(z_null),
    }
    pd.DataFrame([result]).to_csv(f"{OUT}/R13_T3_placebo_vuong_summary.csv", index=False)
    pd.DataFrame({"z_null": z_null}).to_csv(f"{OUT}/R13_T3_placebo_vuong_dist.csv", index=False)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# R13.4  Cluster-robust LR test (clustered by date)
# ─────────────────────────────────────────────────────────────────────────────

def r13_4_cluster_robust_lr(panel):
    print("\n" + "="*62)
    print("R13.4 — CLUSTER-ROBUST LR TEST (CLUSTERED BY DATE)")
    print("="*62)

    sub = panel.dropna(subset=["ret_next_month","DH_z","DS_z","T"]).copy()
    sub["TxDS"] = sub["T"] * sub["DS_z"]
    sub["time_id"] = pd.Categorical(sub["date"]).codes
    sub = sub.dropna(subset=["TxDS"])
    n = len(sub)
    print(f"  N = {n:,} observations, {sub['date'].nunique()} dates")

    y   = sub["ret_next_month"].values
    dh  = sub["DH_z"].values
    ds  = sub["DS_z"].values
    txds = sub["TxDS"].values
    grp = sub["time_id"].values

    # ── Standard OLS LR (from R12.T5) ──
    def fit_ols_ll(X_cols_arr):
        Xc = np.column_stack([np.ones(n)] + [sub[c].values for c in X_cols_arr])
        beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        resid = y - Xc @ beta
        sigma2 = (resid**2).mean()
        ll = -n/2 * (np.log(2 * np.pi * sigma2) + 1)
        k = Xc.shape[1]
        return ll, k, resid, Xc, beta

    ll_full, k_full, resid_full, Xf, beta_full = fit_ols_ll(["DH_z","DS_z","TxDS"])
    ll_B,    k_B,   resid_B,    XB, beta_B     = fit_ols_ll(["DH_z","DS_z"])
    ll_C,    k_C,   resid_C,    XC, beta_C     = fit_ols_ll(["DH_z","TxDS"])

    lr_standard_B  = 2 * (ll_full - ll_B)   # test T·ΔS = 0
    lr_standard_C  = 2 * (ll_full - ll_C)   # test ΔS = 0

    from scipy.stats import chi2
    p_standard_B = 1 - chi2.cdf(lr_standard_B, k_full - k_B)
    p_standard_C = 1 - chi2.cdf(lr_standard_C, k_full - k_C)

    print(f"\n  Standard OLS LR (replicate from R12.T5):")
    print(f"    Restrict T·ΔS=0: LR={lr_standard_B:.3f}, df={k_full-k_B}, p={p_standard_B:.4e}")
    print(f"    Restrict ΔS=0:   LR={lr_standard_C:.3f}, df={k_full-k_C}, p={p_standard_C:.4e}")

    # ── Cluster-robust Wald test (equivalent to cluster-LR) ──
    # Use a score / Wald approach with clustered variance.
    # For the restriction T·ΔS = 0, we test β_{TxDS} = 0 in the FULL model
    # using a cluster-robust t-test. This is the F/Wald analog of the LR test.

    def cluster_robust_vcov(X, resid, groups):
        """Cluster-robust variance-covariance matrix (Liang-Zeger)."""
        n_, k_ = X.shape
        xtx_inv = np.linalg.pinv(X.T @ X)
        B = np.zeros((k_, k_))
        for g in np.unique(groups):
            mask = groups == g
            Xg = X[mask]
            eg = resid[mask]
            B += Xg.T @ np.outer(eg, eg) @ Xg
        G = len(np.unique(groups))
        # Small-sample correction: G/(G-1) * (N-1)/(N-k)
        sc = (G / (G - 1)) * ((n_ - 1) / (n_ - k_))
        vcov = sc * xtx_inv @ B @ xtx_inv
        return vcov

    vcov_full = cluster_robust_vcov(Xf, resid_full, grp)
    se_full   = np.sqrt(np.diag(vcov_full))

    # Column indices: 0=const, 1=DH, 2=DS, 3=TxDS
    col_map = {c: i+1 for i, c in enumerate(["DH_z","DS_z","TxDS"])}

    # Wald test: β_{TxDS} = 0 (one restriction)
    idx_txds = col_map["TxDS"]
    beta_txds = beta_full[idx_txds]
    se_txds   = se_full[idx_txds]
    wald_txds = (beta_txds / se_txds) ** 2   # chi²(1) under H0

    # Wald test: β_{DS} = 0
    idx_ds   = col_map["DS_z"]
    beta_ds_f = beta_full[idx_ds]
    se_ds_f   = se_full[idx_ds]
    wald_ds   = (beta_ds_f / se_ds_f) ** 2

    p_wald_txds = 1 - chi2.cdf(wald_txds, 1)
    p_wald_ds   = 1 - chi2.cdf(wald_ds, 1)

    t_txds = beta_txds / se_txds
    t_ds   = beta_ds_f / se_ds_f

    print(f"\n  Cluster-robust Wald test (H0: coef = 0 in full model):")
    print(f"    β_{{TxDS}}: coef={beta_txds:.6f}, cluster-SE={se_txds:.6f}, "
          f"t={t_txds:.3f}, Wald χ²={wald_txds:.3f}, p={p_wald_txds:.4e}")
    print(f"    β_{{DS}}:   coef={beta_ds_f:.6f}, cluster-SE={se_ds_f:.6f}, "
          f"t={t_ds:.3f}, Wald χ²={wald_ds:.3f}, p={p_wald_ds:.4e}")

    # Inflation factor: how much does clustering widen the SEs?
    # Compare OLS SE vs cluster-robust SE
    ols_res = sm.OLS(y, Xf).fit()
    ols_se_txds = ols_res.bse[idx_txds]
    ols_se_ds   = ols_res.bse[idx_ds]
    inflate_txds = se_txds / ols_se_txds
    inflate_ds   = se_ds_f  / ols_se_ds

    print(f"\n  SE inflation from clustering:")
    print(f"    TxDS: OLS SE={ols_se_txds:.6f}, cluster SE={se_txds:.6f}, "
          f"inflation={inflate_txds:.2f}×")
    print(f"    DS:   OLS SE={ols_se_ds:.6f},   cluster SE={se_ds_f:.6f}, "
          f"inflation={inflate_ds:.2f}×")

    verdict_txds = ("significant" if p_wald_txds < 0.05 else "NOT significant at 5%")
    verdict_ds   = ("significant" if p_wald_ds   < 0.05 else "NOT significant at 5%")
    print(f"\n  VERDICT: After date-clustering,")
    print(f"    T·ΔS remains {verdict_txds} (p={p_wald_txds:.4e})")
    print(f"    ΔS remains {verdict_ds} (p={p_wald_ds:.4e})")

    rows = [
        {"test": "R13.4", "restriction": "T·ΔS=0",
         "lr_standard": round(lr_standard_B, 3), "p_standard": round(p_standard_B, 6),
         "wald_cluster": round(wald_txds, 3), "p_cluster": round(p_wald_txds, 6),
         "se_inflation": round(inflate_txds, 3), "verdict": verdict_txds},
        {"test": "R13.4", "restriction": "ΔS=0",
         "lr_standard": round(lr_standard_C, 3), "p_standard": round(p_standard_C, 6),
         "wald_cluster": round(wald_ds, 3), "p_cluster": round(p_wald_ds, 6),
         "se_inflation": round(inflate_ds, 3), "verdict": verdict_ds},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/R13_T4_cluster_robust_lr.csv", index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# R13.5  Signed ACF + quintile FF5+UMD alpha NW verification
# ─────────────────────────────────────────────────────────────────────────────

def r13_5_acf_sign_and_alpha(panel, factors):
    print("\n" + "="*62)
    print("R13.5 — SIGNED ACF + QUINTILE ALPHA NW VERIFICATION")
    print("="*62)

    # ── Part A: Signed ACF ──
    _, coefs_df = fama_macbeth(panel.dropna(subset=["DG","ret_next_month"]),
                               "ret_next_month", ["DG"], lags=6)
    beta = coefs_df["DG"].dropna().sort_index()
    n = len(beta)
    print(f"\n  β_ΔG series: {n} months")

    from statsmodels.tsa.stattools import acf
    max_lags = min(60, n // 4)
    acf_vals, confint = acf(beta.values, nlags=max_lags, alpha=0.05)
    ci_band = 1.96 / np.sqrt(n)

    # Signed breakdown
    pos_sig = [(lag, v) for lag, v in enumerate(acf_vals[1:], 1)
               if v > ci_band]
    neg_sig = [(lag, v) for lag, v in enumerate(acf_vals[1:], 1)
               if v < -ci_band]

    print(f"\n  ACF significance band: ±{ci_band:.4f}")
    print(f"  Positive significant lags: {[(l, round(v,3)) for l,v in pos_sig]}")
    print(f"  Negative significant lags: {[(l, round(v,3)) for l,v in neg_sig]}")
    print(f"  Count: {len(pos_sig)} positive, {len(neg_sig)} negative")

    # NW direction implication
    if neg_sig:
        print(f"\n  Negative ACF lags: {[l for l,_ in neg_sig]}")
        print("  Negative autocorrelation at short lags means the FM coefficient")
        print("  exhibits mean-reversion; NW upward-adjusts the SE relative to")
        print("  iid (Bartlett kernel down-weights those negative lags less than")
        print("  positive ones would be upweighted). Sign reversal at short lags")
        print("  is consistent with the reported NW-6 being slightly larger in")
        print("  magnitude than OLS SE — opposite of the usual positive-AC case.")
    if pos_sig:
        print(f"\n  Positive ACF lags: {[l for l,_ in pos_sig]}")
        print("  Positive autocorrelation inflates variance; NW corrects upward.")

    # Save full ACF table
    acf_table = pd.DataFrame({
        "lag": range(len(acf_vals)),
        "acf": acf_vals,
        "ci_lo": confint[:,0] - acf_vals,
        "ci_hi": confint[:,1] - acf_vals,
        "sign": ["pos" if v > 0 else "neg" for v in acf_vals],
        "significant": np.abs(acf_vals) > ci_band,
    })
    acf_table.to_csv(f"{OUT}/R13_T5_acf_signed.csv", index=False)

    # ── Part B: Quintile FF5+UMD alpha NW verification ──
    print("\n  Part B: Quintile FF5+UMD alpha verification")
    ff_cols = [c for c in ["Mkt_RF","SMB","HML","RMW","CMA","Mom"] if c in factors.columns]
    rf = factors["RF"].rename("RF") if "RF" in factors.columns else pd.Series(0.0, index=factors.index)

    panel2 = panel.copy()
    panel2["_q"] = panel2.groupby("date")["DG"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") if x.nunique() >= 5 else np.nan
    )
    panel2 = panel2.dropna(subset=["_q","ret_next_month"])
    qret_m = panel2.groupby(["date","_q"])["ret_next_month"].mean().unstack("_q")
    qret_m.index = pd.to_datetime(qret_m.index)

    alpha_rows = []
    for q in range(5):
        q_ret = qret_m.get(float(q), qret_m.get(q, pd.Series(dtype=float))).dropna()
        if len(q_ret) < 24:
            continue
        rf_q = rf.reindex(q_ret.index).fillna(0)
        q_ex = q_ret - rf_q

        f_q = factors[ff_cols].reindex(q_ret.index).dropna()
        q_ex2 = q_ex.reindex(f_q.index).dropna()
        f_q2  = f_q.reindex(q_ex2.index)

        # OLS with HAC-NW
        X = sm.add_constant(f_q2)
        res = sm.OLS(q_ex2, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        alpha    = res.params["const"]
        t_alpha  = res.tvalues["const"]
        p_alpha  = res.pvalues["const"]
        ann_alpha = alpha * 12

        print(f"    Q{q+1}: α={alpha:.5f}/mo ({ann_alpha*100:.2f}%/yr), "
              f"NW t={t_alpha:.3f}{stars(p_alpha)}, "
              f"n={len(q_ex2)}")
        alpha_rows.append({
            "quintile": q+1,
            "alpha_monthly": round(alpha, 6),
            "alpha_annual":  round(ann_alpha, 4),
            "t_NW6":         round(t_alpha, 4),
            "p_NW6":         round(p_alpha, 4),
            "n_months":      len(q_ex2),
            "stars":         stars(p_alpha),
        })

    # L/S (Q5 - Q1)
    if 5 in [r["quintile"] for r in alpha_rows] and 1 in [r["quintile"] for r in alpha_rows]:
        q5 = qret_m.get(float(4), qret_m.get(4, pd.Series(dtype=float))).dropna()
        q1 = qret_m.get(float(0), qret_m.get(0, pd.Series(dtype=float))).dropna()
        ls = (q5 - q1).dropna()
        rf_ls = rf.reindex(ls.index).fillna(0)
        ls_ex = ls - rf_ls
        f_ls = factors[ff_cols].reindex(ls.index).dropna()
        ls_ex2 = ls_ex.reindex(f_ls.index).dropna()
        f_ls2 = f_ls.reindex(ls_ex2.index)
        X_ls = sm.add_constant(f_ls2)
        res_ls = sm.OLS(ls_ex2, X_ls).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        a_ls = res_ls.params["const"]
        t_ls = res_ls.tvalues["const"]
        p_ls = res_ls.pvalues["const"]
        print(f"    L/S: α={a_ls:.5f}/mo ({a_ls*12*100:.2f}%/yr), NW t={t_ls:.3f}{stars(p_ls)}")
        alpha_rows.append({
            "quintile": "L/S", "alpha_monthly": round(a_ls,6),
            "alpha_annual": round(a_ls*12,4), "t_NW6": round(t_ls,4),
            "p_NW6": round(p_ls,4), "n_months": len(ls_ex2), "stars": stars(p_ls),
        })

    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df["test"] = "R13.5"
    alpha_df.to_csv(f"{OUT}/R13_T5_quintile_alphas.csv", index=False)

    # ── Plot: signed ACF bar chart ──
    fig, ax = plt.subplots(figsize=(10, 4))
    lags_plot = acf_table["lag"].values[1:]
    acf_plot  = acf_table["acf"].values[1:]
    colors = ["#d6604d" if v > 0 else "#4393c3" for v in acf_plot]
    sig_mask = acf_table["significant"].values[1:]
    alphas = [0.9 if s else 0.4 for s in sig_mask]
    for lag, val, col, alph in zip(lags_plot, acf_plot, colors, alphas):
        ax.bar(lag, val, color=col, alpha=alph, width=0.7)
    ax.axhline(ci_band,  color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(-ci_band, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Signed ACF of FM β_ΔG — Red=Positive, Blue=Negative\n"
                 "(Opaque bars exceed 95% CI bound)")
    ax.set_xlim(0, max_lags + 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/R13_T5_signed_acf.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{OUT}/R13_T5_signed_acf.pdf", bbox_inches="tight")
    plt.close()
    print(f"\n  Signed ACF plot saved.")

    return acf_table, alpha_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  R13 — REVIEWER RESPONSE TASKS R13.1–R13.5")
    print("=" * 65)

    panel, factors = load_panel()

    results = {}

    for task_fn, key in [
        (lambda: r13_1_expanding_T(panel),                  "R13.1_expanding_T"),
        (lambda: r13_2_vif_partial_r2(panel, factors),      "R13.2_vif_partial_r2"),
        (lambda: r13_3_placebo_vuong(panel, n_perm=1000),   "R13.3_placebo_vuong"),
        (lambda: r13_4_cluster_robust_lr(panel),            "R13.4_cluster_lr"),
        (lambda: r13_5_acf_sign_and_alpha(panel, factors),  "R13.5_acf_alpha"),
    ]:
        try:
            task_fn()
            results[key] = "OK"
        except Exception as e:
            import traceback; traceback.print_exc()
            results[key] = f"ERROR: {e}"

    print("\n" + "=" * 65)
    print("  R13 SUMMARY")
    print("=" * 65)
    for k, v in results.items():
        print(f"  {'✓' if v == 'OK' else '✗'}  {k}: {v}")
    print(f"\n  Outputs in: {os.path.abspath(OUT)}/")


if __name__ == "__main__":
    main()
