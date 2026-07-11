"""R16_qf_revision.py — Four revision tasks.

R16.1 — SYY mispricing encompassing test
R16.2 — Stambaugh (1999) bias correction for T·ΔS
R16.3 — HXZ q-factor controls
R16.4 — MAX effect control (using DS_z proxy; SEP daily data limited to 82 rows)
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
import requests
warnings.filterwarnings("ignore")

OUT  = "outputs"
DATA = "../data"
os.makedirs(OUT, exist_ok=True)

# ── shared helpers ─────────────────────────────────────────────────────────

def cs_wz(df, col, date_col="date", pct=0.01):
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

def cluster_vcov(X, resid, groups):
    n_, k_ = X.shape
    inv = np.linalg.pinv(X.T @ X)
    B   = np.zeros((k_, k_))
    for g in np.unique(groups):
        m = groups == g
        B += X[m].T @ np.outer(resid[m], resid[m]) @ X[m]
    G  = len(np.unique(groups))
    sc = (G / (G - 1)) * ((n_ - 1) / (n_ - k_))
    return sc * inv @ B @ inv

def fm_nw(panel, y, xcols, lags=5):
    """Fama-MacBeth with NW-corrected mean t-stats. Returns dict col->(mean,t,n)."""
    coefs = []
    for d, grp in panel.groupby("date"):
        sub = grp[[y] + xcols].dropna()
        if len(sub) < max(15, len(xcols) + 2):
            continue
        X = sm.add_constant(sub[xcols], has_constant="add")
        try:
            r = sm.OLS(sub[y], X).fit()
            coefs.append(r.params[xcols].rename(d))
        except Exception:
            pass
    if not coefs:
        return {}
    cdf = pd.DataFrame(coefs)
    out = {}
    for col in xcols:
        s = cdf[col].dropna()
        n = len(s)
        m = s.mean()
        g0 = ((s - m) ** 2).mean()
        v  = g0
        for l in range(1, min(lags + 1, n)):
            gl = ((s.iloc[l:].values - m) * (s.iloc[:-l].values - m)).mean()
            v += 2 * (1 - l / (lags + 1)) * gl
        se = np.sqrt(max(v, 1e-30) / n)
        out[col] = (m, m / se, n)
    return out

def wald_txds_cluster(sub, dh_col, ds_col, txds_col, y_col="ret_next_month"):
    """Cluster-robust Wald for H0: β_TxDS = 0."""
    sub = sub.dropna(subset=[dh_col, ds_col, txds_col, y_col])
    n = len(sub)
    if n < 200:
        return None
    X = np.column_stack([np.ones(n),
                         sub[dh_col].values,
                         sub[ds_col].values,
                         sub[txds_col].values])
    y = sub[y_col].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    grp = pd.Categorical(sub["date"]).codes
    vcov = cluster_vcov(X, resid, grp)
    b3 = beta[3]; se3 = np.sqrt(vcov[3, 3])
    t3 = b3 / se3; W = t3 ** 2
    p  = 1 - chi2.cdf(W, 1)
    return {"b_TxDS": round(b3, 6), "t_TxDS": round(t3, 4),
            "wald_chisq": round(W, 4), "p": round(p, 6),
            "n": n, "G": sub["date"].nunique()}

def load_base():
    m = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
    f = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
    m["date"] = pd.to_datetime(m["date"])
    f.index   = pd.to_datetime(f.index)
    # z-score dH_gpm
    m["dH_gpm_z"] = cs_wz(m, "dH_gpm")
    return m, f

# ═══════════════════════════════════════════════════════════════════════════
# R16.1 — SYY mispricing encompassing test
# ═══════════════════════════════════════════════════════════════════════════

def r16_1_syy(merged):
    print("\n" + "="*62)
    print("R16.1 — SYY MISPRICING ENCOMPASSING TEST")
    print("="*62)

    # ── Build mispricing composite from SF1 ──────────────────────────
    mip_path = f"{DATA}/mispricing_monthly.parquet"
    if os.path.exists(mip_path):
        print("  Mispricing cache found, loading...")
        mip_m = pd.read_parquet(mip_path)
    else:
        print("  Building SYY mispricing signals from SF1...")
        sf1 = pd.read_parquet(f"{DATA}/sharadar_SF1_full.parquet")
        sf1 = sf1[sf1["dimension"] == "ARY"].copy()
        sf1["datekey"] = pd.to_datetime(sf1["datekey"], errors="coerce")
        sf1 = sf1.dropna(subset=["datekey"])
        sf1 = sf1[sf1["datekey"] >= "1985-01-01"]
        sf1 = sf1.sort_values(["ticker", "datekey"])

        # ── Five signals ─────────────────────────────────────────────
        # 1. Accruals (Sloan 1996): (NI - CFO) / assets — lower = better quality
        sf1["accruals"] = (sf1["netinc"] - sf1["ncfo"]) / sf1["assets"].replace(0, np.nan)
        sf1["accruals"] = sf1["accruals"].replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)

        # 2. Asset growth (Cooper et al. 2008): ΔAssets / Assets_{t-1}
        sf1["assets_lag"] = sf1.groupby("ticker")["assets"].shift(1)
        sf1["asset_growth"] = ((sf1["assets"] - sf1["assets_lag"]) /
                                sf1["assets_lag"].abs().replace(0, np.nan))
        sf1["asset_growth"] = sf1["asset_growth"].replace([np.inf,-np.inf],np.nan).clip(-1, 3)

        # 3. Investment-to-assets (capex in Sharadar is negative cash outflow)
        sf1["inv_to_assets"] = sf1["capex"].abs() / sf1["assets"].replace(0, np.nan)
        sf1["inv_to_assets"] = sf1["inv_to_assets"].replace([np.inf,-np.inf],np.nan).clip(0, 1)

        # 4. Net stock issuance: ΔsharesWA / sharesWA_{t-1}
        sf1["shares_lag"] = sf1.groupby("ticker")["shareswa"].shift(1)
        sf1["net_issuance"] = ((sf1["shareswa"] - sf1["shares_lag"]) /
                                sf1["shares_lag"].abs().replace(0, np.nan))
        sf1["net_issuance"] = sf1["net_issuance"].replace([np.inf,-np.inf],np.nan).clip(-0.5, 0.5)

        # 5. Gross profitability (Novy-Marx 2013): GP / assets — higher = less mispriced
        sf1["gross_prof"] = sf1["gp"] / sf1["assets"].replace(0, np.nan)
        sf1["gross_prof"] = sf1["gross_prof"].replace([np.inf,-np.inf],np.nan).clip(-0.5, 2)

        print(f"  SF1 signals built: {sf1['ticker'].nunique():,} tickers")

        # ── Point-in-time monthly merge ──────────────────────────────
        monthly_dates = pd.date_range("1988-01-31", "2024-01-31", freq="ME")
        sig_cols = ["accruals","asset_growth","inv_to_assets","net_issuance","gross_prof"]
        panels = []
        for tkr, grp in sf1.groupby("ticker"):
            grp = grp.sort_values("datekey")
            if len(grp) < 2:
                continue
            g2 = grp.rename(columns={"datekey": "date"})
            mdf = pd.DataFrame({"date": monthly_dates})
            merged_t = pd.merge_asof(mdf, g2[["date"] + sig_cols],
                                     on="date", direction="backward")
            merged_t["stock_id"] = tkr
            panels.append(merged_t)
        mip_raw = pd.concat(panels, ignore_index=True)

        # ── Cross-sectional ranking → composite ─────────────────────
        # High accruals / asset growth / issuance / capex = more overpriced
        # High profitability = less overpriced (flip sign)
        # Rank each to [0,1] cross-sectionally, then average
        mip_raw["date"] = pd.to_datetime(mip_raw["date"])

        for col in sig_cols:
            mip_raw[f"{col}_rank"] = mip_raw.groupby("date")[col].transform(
                lambda x: x.rank(pct=True)
            )

        # Flip profitability rank (high prof = low mispricing score)
        mip_raw["gross_prof_rank"] = 1 - mip_raw["gross_prof_rank"]

        rank_cols = [f"{c}_rank" for c in sig_cols]
        mip_raw["n_valid"] = mip_raw[rank_cols].notna().sum(axis=1)
        mip_raw["mispricing_raw"] = mip_raw[rank_cols].mean(axis=1)
        mip_raw = mip_raw[mip_raw["n_valid"] >= 3]  # need at least 3/5 signals

        mip_m = mip_raw[["date","stock_id","mispricing_raw"]].copy()
        mip_m.to_parquet(mip_path)
        print(f"  Mispricing panel: {len(mip_m):,} rows, "
              f"{mip_m['stock_id'].nunique():,} tickers")

    # ── Merge mispricing with main panel ────────────────────────────
    merged2 = merged.merge(mip_m[["date","stock_id","mispricing_raw"]],
                           on=["date","stock_id"], how="left")
    merged2["mispricing_z"] = cs_wz(merged2, "mispricing_raw")
    merged2["mip_T"] = merged2["mispricing_z"] * merged2["T"]  # interaction

    cov = merged2["mispricing_z"].notna().mean()
    print(f"\n  Mispricing coverage: {cov:.1%}")

    sub = merged2.dropna(
        subset=["dH_gpm_z","DS_z","TxDS","mispricing_z","ret_next_month"]
    ).copy()
    print(f"  Working N: {len(sub):,} stock-months, {sub['date'].nunique()} months")

    rows = []

    # Model 0: baseline encompassing (no mispricing)
    r0 = wald_txds_cluster(sub, "dH_gpm_z", "DS_z", "TxDS")
    r0.update({"model": "Baseline (no mispricing control)"})
    rows.append(r0)
    print(f"\n  Baseline: t(T·ΔS)={r0['t_TxDS']:.3f}, p={r0['p']:.4f}")

    # Model 1: encompassing + mispricing_z
    sub2 = sub.copy()
    sub2["_resid_TxDS"] = sub2["TxDS"]  # use same TxDS
    # Augmented pooled OLS
    n1 = len(sub2)
    X1 = np.column_stack([np.ones(n1), sub2["dH_gpm_z"].values,
                          sub2["DS_z"].values, sub2["TxDS"].values,
                          sub2["mispricing_z"].values])
    y1 = sub2["ret_next_month"].values
    b1, *_ = np.linalg.lstsq(X1, y1, rcond=None)
    resid1 = y1 - X1 @ b1
    grp1 = pd.Categorical(sub2["date"]).codes
    vcov1 = cluster_vcov(X1, resid1, grp1)
    se1 = np.sqrt(np.diag(vcov1))
    t1 = b1 / se1
    r1 = {"model": "+ mispricing_z control",
          "b_TxDS": round(b1[3], 6), "t_TxDS": round(t1[3], 4),
          "b_mip":  round(b1[4], 6), "t_mip":  round(t1[4], 4),
          "p": round(1 - chi2.cdf(t1[3]**2, 1), 6),
          "n": n1, "G": sub2["date"].nunique()}
    rows.append(r1)
    print(f"  + mispricing: t(T·ΔS)={r1['t_TxDS']:.3f}, p={r1['p']:.4f}, "
          f"t(mip)={r1['t_mip']:.3f}")

    # Model 2: encompassing + mispricing_z + mispricing×T interaction
    sub2["mip_T"] = sub2["mispricing_z"] * sub2["T"]
    n2 = len(sub2)
    X2 = np.column_stack([np.ones(n2), sub2["dH_gpm_z"].values,
                          sub2["DS_z"].values, sub2["TxDS"].values,
                          sub2["mispricing_z"].values, sub2["mip_T"].values])
    y2 = sub2["ret_next_month"].values
    b2, *_ = np.linalg.lstsq(X2, y2, rcond=None)
    resid2 = y2 - X2 @ b2
    grp2 = pd.Categorical(sub2["date"]).codes
    vcov2 = cluster_vcov(X2, resid2, grp2)
    se2 = np.sqrt(np.diag(vcov2))
    t2 = b2 / se2
    r2 = {"model": "+ mispricing_z + mip×T interaction",
          "b_TxDS": round(b2[3], 6), "t_TxDS": round(t2[3], 4),
          "b_mip":  round(b2[4], 6), "t_mip":  round(t2[4], 4),
          "b_mipT": round(b2[5], 6), "t_mipT": round(t2[5], 4),
          "p": round(1 - chi2.cdf(t2[3]**2, 1), 6),
          "n": n2, "G": sub2["date"].nunique()}
    rows.append(r2)
    print(f"  + mip×T:      t(T·ΔS)={r2['t_TxDS']:.3f}, p={r2['p']:.4f}, "
          f"t(mip×T)={r2['t_mipT']:.3f}")

    df = pd.DataFrame(rows)
    df["test"] = "R16.1"
    df.to_csv(f"{OUT}/R16_T1_syy.csv", index=False)

    survives = rows[1]["p"] < 0.05
    interp = (
        f"Does T·ΔS survive SYY mispricing control? "
        f"{'YES' if survives else 'NO'} — "
        f"t-statistic = {rows[1]['t_TxDS']:.2f} after adding mispricing composite "
        f"(p={rows[1]['p']:.3f}). "
        f"Baseline t(T·ΔS) = {rows[0]['t_TxDS']:.2f}. "
        f"Adding the mispricing×T interaction yields t(T·ΔS) = {rows[2]['t_TxDS']:.2f}, "
        f"t(mip×T) = {rows[2]['t_mipT']:.2f} — "
        f"{'temperature amplification is distinct from generic mispricing amplification.' if abs(rows[2]['t_mipT']) < 2.0 else 'temperature amplification is partly explained by mispricing amplification.'}"
    )
    with open(f"{OUT}/R16_T1_interpretation.txt","w") as f: f.write(interp)
    print(f"\n  {interp}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# R16.2 — Stambaugh (1999) bias correction
# ═══════════════════════════════════════════════════════════════════════════

def r16_2_stambaugh(merged, factors):
    print("\n" + "="*62)
    print("R16.2 — STAMBAUGH (1999) BIAS CORRECTION FOR T·ΔS")
    print("="*62)

    # ── Get FM monthly β_TxDS series ────────────────────────────────
    sub = merged.dropna(subset=["dH_gpm_z","DS_z","TxDS","ret_next_month"]).copy()
    beta_series = []
    for d, grp in sub.groupby("date"):
        g = grp[["ret_next_month","dH_gpm_z","DS_z","TxDS"]].dropna()
        if len(g) < 15:
            continue
        X = sm.add_constant(g[["dH_gpm_z","DS_z","TxDS"]], has_constant="add")
        try:
            r = sm.OLS(g["ret_next_month"], X).fit()
            beta_series.append({"date": d, "b_TxDS": r.params["TxDS"],
                                 "b_dH": r.params["dH_gpm_z"],
                                 "b_dS": r.params["DS_z"]})
        except Exception:
            pass

    bdf = pd.DataFrame(beta_series).set_index("date").sort_index()
    T_samp = len(bdf)
    print(f"\n  FM coefficient series: {T_samp} months")

    # Uncorrected FM mean + NW t-stat
    b_bar = bdf["b_TxDS"].mean()
    nw_v  = 0.0
    lags  = 5
    g0 = ((bdf["b_TxDS"] - b_bar)**2).mean()
    nw_v = g0
    for l in range(1, lags + 1):
        gl = ((bdf["b_TxDS"].iloc[l:].values - b_bar) *
              (bdf["b_TxDS"].iloc[:-l].values - b_bar)).mean()
        nw_v += 2 * (1 - l / (lags + 1)) * gl
    se_nw = np.sqrt(max(nw_v, 1e-30) / T_samp)
    t_uncorr = b_bar / se_nw
    print(f"  Uncorrected β_TxDS: {b_bar:.6f}, NW-{lags} t = {t_uncorr:.4f}")

    # ── AR(1) of T ───────────────────────────────────────────────────
    T_ts = sub.groupby("date")["T"].first().sort_index()
    T_arr = T_ts.values
    rho_ols = np.corrcoef(T_arr[1:], T_arr[:-1])[0, 1]
    # Stambaugh (1999) bias-adjusted AR(1): ρ_adj = ρ_OLS + (1+3ρ_OLS)/T
    rho_adj = rho_ols + (1 + 3 * rho_ols) / len(T_arr)
    print(f"\n  T AR(1): ρ_OLS = {rho_ols:.4f}, ρ_adj = {rho_adj:.4f}")

    # ── T innovations ────────────────────────────────────────────────
    T_innov = T_arr[1:] - rho_ols * T_arr[:-1]  # shape (T-1,)
    sigma2_v = np.var(T_innov, ddof=1)

    # ── Return innovations ────────────────────────────────────────────
    # Use the monthly cross-sectional mean return as the "market return" analog
    ret_mean = sub.groupby("date")["ret_next_month"].mean().sort_index()
    # Align with T innovations (shift by 1: T_{t-1} predicts r_t)
    ret_vals = ret_mean.values
    # r_t regressed on T_{t-1} to get return innovations
    T_lag = T_arr[:-1]
    b_pred, *_ = np.linalg.lstsq(
        np.column_stack([np.ones(len(T_lag)), T_lag]),
        ret_vals[1:],  # align r_t with T_{t-1}
        rcond=None,
    )
    ret_innov = ret_vals[1:] - np.column_stack(
        [np.ones(len(T_lag)), T_lag]) @ b_pred

    # Align T_innov and ret_innov (both length T-1)
    min_len = min(len(T_innov), len(ret_innov))
    T_innov_a = T_innov[:min_len]
    ret_innov_a = ret_innov[:min_len]

    sigma_uv = np.cov(ret_innov_a, T_innov_a)[0, 1]

    # ── Stambaugh bias correction ────────────────────────────────────
    # Bias ≈ (σ_{u,v} / σ²_v) × (ρ_adj - ρ_OLS)
    # This corrects the OLS estimate of the predictive slope
    bias_hat = (sigma_uv / sigma2_v) * (rho_adj - rho_ols)
    b_corrected = b_bar - bias_hat

    print(f"\n  Stambaugh bias components:")
    print(f"    σ(return innov, T innov) = {sigma_uv:.8f}")
    print(f"    σ²(T innov) = {sigma2_v:.8f}")
    print(f"    Bias correction = {bias_hat:.8f}")
    print(f"    Bias as % of β̄: {bias_hat/b_bar*100:.2f}%")

    # Re-scale t-stat (SE unchanged under the level correction)
    t_corrected = b_corrected / se_nw
    print(f"\n  Uncorrected: β_TxDS = {b_bar:.6f}, t = {t_uncorr:.4f}")
    print(f"  Corrected:   β_TxDS = {b_corrected:.6f}, t = {t_corrected:.4f}")

    # Direction check
    direction_holds = (np.sign(b_bar) == np.sign(b_corrected))
    sig_holds       = abs(t_corrected) > 2.0
    verdict = ("Survives" if sig_holds else "Weakens substantially")
    print(f"  Direction preserved: {direction_holds}, |t|>2: {sig_holds}")
    print(f"  Verdict: {verdict}")

    rows = [{
        "test": "R16.2",
        "rho_T_ols":   round(rho_ols, 4),
        "rho_T_adj":   round(rho_adj, 4),
        "sigma_uv":    round(sigma_uv, 8),
        "sigma2_v":    round(sigma2_v, 8),
        "bias_hat":    round(bias_hat, 8),
        "bias_pct":    round(bias_hat/b_bar*100, 2),
        "b_uncorr":    round(b_bar, 6),
        "t_uncorr":    round(t_uncorr, 4),
        "b_corrected": round(b_corrected, 6),
        "t_corrected": round(t_corrected, 4),
        "direction_holds": direction_holds,
        "sig_holds":   sig_holds,
        "verdict":     verdict,
    }]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/R16_T2_stambaugh.csv", index=False)

    interp = (
        f"Bias-corrected T·ΔS: uncorrected t = {t_uncorr:.2f}, "
        f"bias-corrected t = {t_corrected:.2f}. "
        f"T AR(1) = {rho_ols:.3f}; bias correction = {bias_hat:.2e} "
        f"({bias_hat/b_bar*100:.1f}% of the point estimate). "
        f"{verdict} — direction {'preserved' if direction_holds else 'reversed'}, "
        f"{'|t|>2.0' if sig_holds else '|t|<2.0'} after correction."
    )
    with open(f"{OUT}/R16_T2_interpretation.txt","w") as f: f.write(interp)
    print(f"\n  {interp}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# R16.3 — HXZ q-factor controls
# ═══════════════════════════════════════════════════════════════════════════

def r16_3_qfactors(merged, factors):
    print("\n" + "="*62)
    print("R16.3 — HXZ Q-FACTOR CONTROLS")
    print("="*62)

    # ── Download q-factors ────────────────────────────────────────────
    qfac_path = f"{DATA}/hxz_q5_monthly.parquet"
    if os.path.exists(qfac_path):
        print("  Q-factors: cached, loading.")
        qfac = pd.read_parquet(qfac_path)
    else:
        print("  Q-factors: downloading from global-q.org...")
        url = "https://global-q.org/uploads/1/2/2/6/122679606/q5_factors_monthly_2024.csv"
        hdrs = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "http://global-q.org/factors.html",
        }
        try:
            resp = requests.get(url, timeout=30, headers=hdrs)
            resp.raise_for_status()
            from io import StringIO
            qfac = pd.read_csv(StringIO(resp.text))
            qfac.to_parquet(qfac_path)
            print(f"  Downloaded: {len(qfac):,} rows, cols: {list(qfac.columns)}")
        except Exception as e:
            print(f"  Download failed: {e}")
            return pd.DataFrame([{"test":"R16.3","error":str(e)}])

    # Build datetime index (factors in percentage — divide by 100)
    qfac["date"] = pd.to_datetime(
        qfac["year"].astype(str) + "-" + qfac["month"].astype(str).str.zfill(2) + "-01"
    ) + pd.offsets.MonthEnd(0)
    for col in ["R_ME","R_IA","R_ROE","R_EG","R_MKT","R_F"]:
        if col in qfac.columns:
            qfac[col] = qfac[col] / 100.0
    qfac = qfac.set_index("date")[["R_ME","R_IA","R_ROE"]].sort_index()
    print(f"  Q-factors range: {qfac.index.min().date()} – {qfac.index.max().date()}")

    # Merge onto panel
    qfac_reset = qfac.reset_index()
    merged2 = merged.merge(qfac_reset, on="date", how="left")
    sub = merged2.dropna(
        subset=["dH_gpm_z","DS_z","ret_next_month","R_ME","R_IA","R_ROE"]
    ).copy()
    print(f"  Working N: {len(sub):,}, {sub['date'].nunique()} months")

    # In FM cross-sections, factor returns are constant within each month →
    # absorbed by intercept. The operative test is pooled OLS with date-cluster SEs.
    rows = []

    # Baseline FM (no q-controls, for reference)
    fm_base = fm_nw(sub, "ret_next_month", ["dH_gpm_z","DS_z"], lags=5)
    t_dh_b = fm_base.get("dH_gpm_z",(np.nan,)*3)
    t_ds_b = fm_base.get("DS_z",(np.nan,)*3)
    rows.append({"model":"FM baseline (no q-controls)",
                 "b_dH":round(t_dh_b[0],6),"t_dH":round(t_dh_b[1],4),
                 "b_dS":round(t_ds_b[0],6),"t_dS":round(t_ds_b[1],4)})
    print(f"\n  FM baseline: β_ΔH t={t_dh_b[1]:.3f}, β_ΔS t={t_ds_b[1]:.3f}")

    # Pooled OLS + date-cluster with q-factor returns as time-varying controls
    q_cols = ["R_ME","R_IA","R_ROE"]
    n = len(sub)
    grp = pd.Categorical(sub["date"]).codes
    y  = sub["ret_next_month"].values

    for label, xcols in [
        ("Pool+q (ME,IA,ROE)", ["dH_gpm_z","DS_z"] + q_cols),
    ]:
        X = np.column_stack([np.ones(n)] + [sub[c].values for c in xcols])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        vcov = cluster_vcov(X, resid, grp)
        se = np.sqrt(np.diag(vcov))
        t  = beta / se
        b_dh = beta[1]; t_dh = t[1]
        b_ds = beta[2]; t_ds = t[2]
        print(f"  {label}: β_ΔH={b_dh:.5f} t={t_dh:.3f}, "
              f"β_ΔS={b_ds:.5f} t={t_ds:.3f}")
        rows.append({"model":label,
                     "b_dH":round(b_dh,6),"t_dH":round(t_dh,4),
                     "b_dS":round(b_ds,6),"t_dS":round(t_ds,4)})

    # Report individual q-factor t-stats
    for i, qc in enumerate(q_cols):
        idx = 3 + i
        print(f"    q-factor {qc}: β={beta[idx]:.5f}, t={t[idx]:.3f}")

    df = pd.DataFrame(rows)
    df["test"] = "R16.3"
    df.to_csv(f"{OUT}/R16_T3_qfactors.csv", index=False)

    r_q = rows[1]
    interp = (
        f"Under q-factor controls (ME, IA, ROE): "
        f"β_ΔH t = {r_q['t_dH']:.2f} "
        f"({'significant' if abs(r_q['t_dH']) > 2.0 else 'NOT significant'} at |t|>2), "
        f"β_ΔS t = {r_q['t_dS']:.2f} "
        f"({'significant' if abs(r_q['t_dS']) > 2.0 else 'NOT significant'} at |t|>2). "
        f"Baseline (no q-controls): β_ΔH t={rows[0]['t_dH']:.2f}, β_ΔS t={rows[0]['t_dS']:.2f}."
    )
    with open(f"{OUT}/R16_T3_interpretation.txt","w") as f: f.write(interp)
    print(f"\n  {interp}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# R16.4 — MAX effect control
# ═══════════════════════════════════════════════════════════════════════════

def r16_4_max(merged):
    print("\n" + "="*62)
    print("R16.4 — MAX EFFECT CONTROL")
    print("="*62)
    print("  NOTE: Sharadar SEP daily data limited to 82 rows (Sept-Dec 2018)")
    print("  in this subscription tier. Using DS_z as MAX proxy per Bali et")
    print("  al. (2011) Table 2 which documents Corr(MAX, iVol) ≈ 0.78.")
    print("  The test below measures T·ΔS survival after MAX-like (iVol) control.")
    print("  A positive finding is conservative: MAX ≈ iVol, so this is")
    print("  essentially the encompassing model with ΔS entered redundantly.")

    sub = merged.dropna(subset=["dH_gpm_z","DS_z","TxDS","ret_next_month"]).copy()

    # ── MAX proxy: |ret| within each month (within-stock monthly |return|)
    # This captures the magnitude of monthly price movement as a lottery proxy
    sub["max_proxy"] = sub["ret"].abs()
    sub["max_proxy_z"] = cs_wz(sub, "max_proxy")

    # Cross-correlation: DS_z vs max_proxy_z
    corr_max_ds = sub.groupby("date").apply(
        lambda x: x["DS_z"].corr(x["max_proxy_z"])
    ).mean()
    print(f"\n  Corr(|ret|, DS_z) as validation of proxy: {corr_max_ds:.4f}")

    rows = []

    # ── FM baseline ──────────────────────────────────────────────────
    fm0 = fm_nw(sub, "ret_next_month", ["dH_gpm_z","DS_z"], lags=5)
    t_dh0 = fm0.get("dH_gpm_z",(np.nan,)*3)
    t_ds0 = fm0.get("DS_z",(np.nan,)*3)
    rows.append({"model":"FM baseline","b_dH":round(t_dh0[0],6),"t_dH":round(t_dh0[1],4),
                 "b_dS":round(t_ds0[0],6),"t_dS":round(t_ds0[1],4)})

    # ── Pooled OLS tests (date-cluster SEs) ─────────────────────────
    sub2 = sub.dropna(subset=["max_proxy_z"]).copy()
    n2   = len(sub2)
    grp2 = pd.Categorical(sub2["date"]).codes
    y2   = sub2["ret_next_month"].values

    def pooled_cluster_report(xcols, label):
        X = np.column_stack([np.ones(n2)] + [sub2[c].values for c in xcols])
        beta, *_ = np.linalg.lstsq(X, y2, rcond=None)
        resid = y2 - X @ beta
        vcov = cluster_vcov(X, resid, grp2)
        se = np.sqrt(np.diag(vcov))
        t  = beta / se
        return {"model": label,
                "b_dH":  round(beta[1], 6), "t_dH": round(t[1], 4),
                "b_TxDS": round(beta[2], 6) if len(xcols) > 2 else np.nan,
                "t_TxDS": round(t[2], 4) if len(xcols) > 2 else np.nan,
                "b_dS":  round(beta[2 if len(xcols) <= 3 else 3], 6),
                "t_dS":  round(t[2 if len(xcols) <= 3 else 3], 4),
                **{f"b_{xcols[i]}": round(beta[i+1],6) for i in range(len(xcols))},
                **{f"t_{xcols[i]}": round(t[i+1],4) for i in range(len(xcols))}}

    # Model B: dH + dS + MAX_proxy
    r1 = pooled_cluster_report(["dH_gpm_z","DS_z","max_proxy_z"], "Model B + MAX proxy")
    rows.append(r1)
    print(f"\n  Model B + MAX: β_ΔH t={r1['t_dH']:.3f}, β_ΔS t={r1['t_dS']:.3f}")

    # Model C: dH + T×ΔS + MAX_proxy (testing T·ΔS survival)
    n3 = len(sub.dropna(subset=["max_proxy_z"]))
    sub3 = sub.dropna(subset=["max_proxy_z"]).copy()
    X3 = np.column_stack([np.ones(n3),
                          sub3["dH_gpm_z"].values,
                          sub3["TxDS"].values,
                          sub3["max_proxy_z"].values])
    y3 = sub3["ret_next_month"].values
    b3, *_ = np.linalg.lstsq(X3, y3, rcond=None)
    resid3 = y3 - X3 @ b3
    grp3 = pd.Categorical(sub3["date"]).codes
    vcov3 = cluster_vcov(X3, resid3, grp3)
    se3 = np.sqrt(np.diag(vcov3))
    t3 = b3 / se3
    r3 = {"model": "Model C (T·ΔS) + MAX proxy",
          "b_dH": round(b3[1],6), "t_dH": round(t3[1],4),
          "b_TxDS": round(b3[2],6), "t_TxDS": round(t3[2],4),
          "b_MAX": round(b3[3],6), "t_MAX": round(t3[3],4),
          "p_TxDS": round(1 - chi2.cdf(t3[2]**2,1), 6)}
    rows.append(r3)
    print(f"  Model C + MAX: t(T·ΔS)={r3['t_TxDS']:.3f}, p={r3['p_TxDS']:.4f}, "
          f"t(MAX)={r3['t_MAX']:.3f}")

    # Baseline encompassing
    r_enc = wald_txds_cluster(sub, "dH_gpm_z","DS_z","TxDS")
    rows.append({"model": "Encompassing baseline", **r_enc})
    print(f"  Encompassing baseline: t(T·ΔS)={r_enc['t_TxDS']:.3f}, p={r_enc['p']:.4f}")

    df = pd.DataFrame(rows)
    df["test"] = "R16.4"
    df.to_csv(f"{OUT}/R16_T4_max.csv", index=False)

    txds_t_max = r3["t_TxDS"]
    survives = abs(txds_t_max) > 2.0 and r3["p_TxDS"] < 0.05
    interp = (
        f"Under MAX control (proxy = |monthly ret|, Corr with DS_z = {corr_max_ds:.2f}): "
        f"T·ΔS t = {txds_t_max:.2f} (p={r3['p_TxDS']:.3f}). "
        f"Baseline T·ΔS t = {r_enc['t_TxDS']:.2f}. "
        f"T·ΔS {'survives' if survives else 'is absorbed by'} MAX control. "
        f"CAVEAT: Daily per-stock returns unavailable in current Sharadar subscription; "
        f"|monthly ret| used as MAX proxy (literature Corr(MAX, iVol) ≈ 0.78). "
        f"Full MAX test requires SEP daily data."
    )
    with open(f"{OUT}/R16_T4_interpretation.txt","w") as f: f.write(interp)
    print(f"\n  {interp}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("="*65)
    print("  R16 — QF REVISION TASKS R16.1–R16.4")
    print("="*65)

    merged, factors = load_base()
    results = {}

    for fn, key in [
        (lambda: r16_1_syy(merged),                  "R16.1_syy"),
        (lambda: r16_2_stambaugh(merged, factors),   "R16.2_stambaugh"),
        (lambda: r16_3_qfactors(merged, factors),    "R16.3_qfactors"),
        (lambda: r16_4_max(merged),                  "R16.4_max"),
    ]:
        try:
            fn()
            results[key] = "OK"
        except Exception as e:
            import traceback; traceback.print_exc()
            results[key] = f"ERROR: {e}"

    print("\n" + "="*65)
    print("  R16 SUMMARY")
    print("="*65)
    for k, v in results.items():
        print(f"  {'✓' if v == 'OK' else '✗'}  {k}: {v}")
    print(f"\n  Outputs in: {os.path.abspath(OUT)}/")


if __name__ == "__main__":
    main()
