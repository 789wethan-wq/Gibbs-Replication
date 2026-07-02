"""sharadar_pipeline.py — Accounting-based ΔH rebuild using Sharadar SF1.

Steps:
  1. Download SF1, SP500, TICKERS (SEP skipped — existing panel has returns)
  2. Filter SF1 → US common stocks, ARY dimension
  3. Compute ROE, GPM, EPS growth per filing
  4. Build point-in-time monthly fundamentals (DATEKEY look-ahead-free)
  5. Build point-in-time SP500 membership
  6. Construct accounting ΔH (rolling 60m std of ROE)
  7. Merge with existing variables_monthly.parquet
  8. Diagnostics: Corr(ΔH_acc, ΔS), quintile sort, FM, cluster-Wald, PIT universe
  9. Save results and merged parquet
"""
import os, sys, warnings, time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
warnings.filterwarnings("ignore")

# ── working directory ──────────────────────────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
OUT  = "../data"
os.makedirs(DATA, exist_ok=True)

import nasdaqdatalink

_api_key = os.environ.get("NASDAQ_DATA_LINK_API_KEY", "")
if not _api_key:
    raise EnvironmentError(
        "Set NASDAQ_DATA_LINK_API_KEY to your Nasdaq Data Link API key before running. "
        "Example: export NASDAQ_DATA_LINK_API_KEY='your_key_here'"
    )
nasdaqdatalink.ApiConfig.api_key = _api_key

# ── shared helpers ─────────────────────────────────────────────────────────
def cs_winsorize_zscore(df, col, date_col="date", pct=0.01):
    def _wz(x):
        x = x.dropna()
        if len(x) < 5: return pd.Series(np.nan, index=x.index)
        lo, hi = x.quantile(pct), x.quantile(1 - pct)
        xc = x.clip(lo, hi)
        std = xc.std()
        if std < 1e-10: return pd.Series(np.nan, index=x.index)
        return (xc - xc.mean()) / std
    return df.groupby(date_col)[col].transform(_wz)

def cluster_vcov(X, resid, groups):
    n_, k_ = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    B = np.zeros((k_, k_))
    for g in np.unique(groups):
        m = groups == g
        B += X[m].T @ np.outer(resid[m], resid[m]) @ X[m]
    G = len(np.unique(groups))
    sc = (G / (G - 1)) * ((n_ - 1) / (n_ - k_))
    return sc * xtx_inv @ B @ xtx_inv

def fama_macbeth_nw(panel, y_col, x_cols, lags=5):
    """Simple FM with NW-corrected mean t-stats."""
    coefs = []
    for d, grp in panel.groupby("date"):
        sub = grp[[y_col] + x_cols].dropna()
        if len(sub) < max(20, len(x_cols) + 2):
            continue
        X = sm.add_constant(sub[x_cols], has_constant="add")
        res = sm.OLS(sub[y_col], X).fit()
        coefs.append(res.params[x_cols].rename(d))
    if not coefs:
        return {}
    cdf = pd.DataFrame(coefs)
    results = {}
    for col in x_cols:
        series = cdf[col].dropna()
        n = len(series)
        mean_ = series.mean()
        # NW variance
        gamma0 = (series**2).mean() - mean_**2
        var_nw = gamma0
        for l in range(1, min(lags + 1, n)):
            gamma_l = ((series.iloc[l:].values - mean_) *
                       (series.iloc[:-l].values - mean_)).mean()
            var_nw += 2 * (1 - l / (lags + 1)) * gamma_l
        se_nw = np.sqrt(max(var_nw, 1e-30) / n)
        results[col] = (mean_, mean_ / se_nw, n)
    return results

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════

def read_export(zip_path):
    """Read a nasdaqdatalink export_table ZIP — extracts the inner CSV."""
    import zipfile, io
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(csv_name) as f:
            return pd.read_csv(f, low_memory=False)

def download_tables():
    print("\n" + "="*62)
    print("STEP 1 — DOWNLOADING SHARADAR TABLES")
    print("="*62)

    # ── TICKERS ────────────────────────────────────────────────────
    tickers_path = f"{DATA}/sharadar_tickers.parquet"
    tickers_zip  = f"{DATA}/sharadar_tickers.csv"   # actually a zip
    if os.path.exists(tickers_path):
        print("  TICKERS: already cached, skipping download.")
        tickers = pd.read_parquet(tickers_path)
    else:
        print("  TICKERS: downloading...")
        t0 = time.time()
        if not os.path.exists(tickers_zip):
            nasdaqdatalink.export_table("SHARADAR/TICKERS", filename=tickers_zip)
        tickers = read_export(tickers_zip)
        tickers.to_parquet(tickers_path)
        print(f"    Done: {len(tickers):,} rows in {time.time()-t0:.0f}s")

    # ── SP500 ───────────────────────────────────────────────────────
    sp500_path = f"{DATA}/sharadar_SP500.parquet"
    sp500_zip  = f"{DATA}/sharadar_SP500.csv"
    if os.path.exists(sp500_path):
        print("  SP500: already cached, skipping download.")
        sp500 = pd.read_parquet(sp500_path)
    else:
        print("  SP500: downloading...")
        t0 = time.time()
        if not os.path.exists(sp500_zip):
            nasdaqdatalink.export_table("SHARADAR/SP500", filename=sp500_zip)
        sp500 = read_export(sp500_zip)
        sp500.to_parquet(sp500_path)
        print(f"    Done: {len(sp500):,} rows in {time.time()-t0:.0f}s")

    # ── SF1 ─────────────────────────────────────────────────────────
    sf1_path = f"{DATA}/sharadar_SF1_full.parquet"
    sf1_zip  = f"{DATA}/sharadar_SF1_full.csv"
    if os.path.exists(sf1_path):
        print("  SF1: already cached, skipping download.")
    else:
        print("  SF1: downloading (large file, may take several minutes)...")
        t0 = time.time()
        if not os.path.exists(sf1_zip):
            nasdaqdatalink.export_table("SHARADAR/SF1", filename=sf1_zip)
            print(f"    ZIP downloaded in {time.time()-t0:.0f}s, extracting...")
        sf1 = read_export(sf1_zip)
        sf1.to_parquet(sf1_path)
        sz = os.path.getsize(sf1_path) / 1e6
        print(f"    SF1 parquet: {len(sf1):,} rows, {sz:.0f} MB, "
              f"{time.time()-t0:.0f}s total")

    return tickers, sp500

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 & 3 — FILTER SF1 AND COMPUTE ACCOUNTING VARIABLES
# ═══════════════════════════════════════════════════════════════════════════

def build_sf1_variables(tickers):
    print("\n" + "="*62)
    print("STEPS 2–3 — SF1 FILTERING + ACCOUNTING VARIABLES")
    print("="*62)

    sf1_path = f"{DATA}/sharadar_SF1_full.parquet"
    sf1_vars_path = f"{DATA}/sharadar_SF1_vars.parquet"

    if os.path.exists(sf1_vars_path):
        print("  SF1 vars: already cached.")
        return pd.read_parquet(sf1_vars_path)

    sf1 = pd.read_parquet(sf1_path)
    print(f"  Raw SF1: {len(sf1):,} rows, {sf1['ticker'].nunique():,} tickers")

    # Filter 1: ARY dimension
    sf1 = sf1[sf1["dimension"] == "ARY"].copy()

    # Filter 2: US common stocks
    us_exchanges = {"NYSE", "NASDAQ", "NYSEMKT", "NYSEARCA", "BATS"}
    if "exchange" in tickers.columns and "category" in tickers.columns:
        us_common = tickers[
            tickers["exchange"].isin(us_exchanges) &
            tickers["category"].str.contains("Common", na=False)
        ]["ticker"].unique()
    else:
        # Fallback: use all tickers in the existing panel
        existing = pd.read_parquet(f"{DATA}/variables_monthly.parquet")
        us_common = existing["stock_id"].unique()
    sf1 = sf1[sf1["ticker"].isin(us_common)].copy()

    # Filter 3: dates
    sf1["datekey"] = pd.to_datetime(sf1["datekey"], errors="coerce")
    sf1 = sf1.dropna(subset=["datekey"])
    sf1 = sf1[sf1["datekey"] >= "1985-01-01"].copy()
    sf1 = sf1.sort_values(["ticker", "datekey"])

    print(f"  Filtered SF1 (US common, ARY, ≥1985): {len(sf1):,} rows, "
          f"{sf1['ticker'].nunique():,} tickers, "
          f"{sf1['datekey'].min().date()} – {sf1['datekey'].max().date()}")

    # ── Accounting variables ─────────────────────────────────────────
    # ROE
    sf1["equity_lag"] = sf1.groupby("ticker")["equity"].shift(1)
    sf1["avg_equity"]  = (sf1["equity"] + sf1["equity_lag"]) / 2
    sf1["roe"] = sf1["netinc"] / sf1["avg_equity"]
    sf1["roe"] = sf1["roe"].clip(-2, 2)

    # GPM
    sf1["gpm"] = sf1["gp"] / sf1["revenue"]
    sf1["gpm"] = sf1["gpm"].replace([np.inf, -np.inf], np.nan).clip(-1, 1)

    # EPS growth
    sf1["eps_lag"] = sf1.groupby("ticker")["eps"].shift(1)
    sf1["eps_growth"] = ((sf1["eps"] - sf1["eps_lag"]) /
                         sf1["eps_lag"].abs().replace(0, np.nan))
    sf1["eps_growth"] = sf1["eps_growth"].replace([np.inf, -np.inf], np.nan).clip(-2, 2)

    keep = ["ticker", "datekey", "roe", "gpm", "eps_growth"]
    sf1_vars = sf1[keep].dropna(subset=["datekey"])
    sf1_vars.to_parquet(sf1_vars_path)

    print(f"  Missing rates — ROE: {sf1_vars['roe'].isna().mean():.1%}, "
          f"GPM: {sf1_vars['gpm'].isna().mean():.1%}, "
          f"EPS growth: {sf1_vars['eps_growth'].isna().mean():.1%}")
    return sf1_vars

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — POINT-IN-TIME MONTHLY FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════════════════

def build_monthly_fundamentals(sf1_vars):
    print("\n" + "="*62)
    print("STEP 4 — POINT-IN-TIME MONTHLY FUNDAMENTALS")
    print("="*62)

    mf_path = f"{DATA}/monthly_fundamentals.parquet"
    if os.path.exists(mf_path):
        print("  monthly_fundamentals: already cached.")
        return pd.read_parquet(mf_path)

    # Month-end dates matching the existing panel format
    monthly_dates = pd.date_range("1988-01-31", "2024-01-31", freq="ME")

    panels = []
    tickers = sf1_vars["ticker"].unique()
    for i, tkr in enumerate(tickers):
        grp = sf1_vars[sf1_vars["ticker"] == tkr].sort_values("datekey")
        if len(grp) < 3:
            continue
        # rename for merge_asof: datekey → date
        grp2 = grp.rename(columns={"datekey": "date"})
        month_df = pd.DataFrame({"date": monthly_dates})
        merged = pd.merge_asof(
            month_df,
            grp2[["date", "roe", "gpm", "eps_growth"]],
            on="date",
            direction="backward",
        )
        merged["stock_id"] = tkr
        panels.append(merged)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(tickers)} tickers processed...")

    mf = pd.concat(panels, ignore_index=True)
    # Drop months before first filing for each ticker (all-NaN rows don't help)
    mf = mf.dropna(subset=["roe", "gpm"], how="all")
    mf.to_parquet(mf_path)

    print(f"  Monthly fundamentals: {len(mf):,} rows, "
          f"{mf['stock_id'].nunique():,} tickers, "
          f"{mf['date'].min().date()} – {mf['date'].max().date()}")
    print(f"  Missing — ROE: {mf['roe'].isna().mean():.1%}, "
          f"GPM: {mf['gpm'].isna().mean():.1%}")
    return mf

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — POINT-IN-TIME SP500 MEMBERSHIP
# ═══════════════════════════════════════════════════════════════════════════

def build_sp500_membership(sp500):
    print("\n" + "="*62)
    print("STEP 5 — POINT-IN-TIME SP500 MEMBERSHIP")
    print("="*62)

    mem_path = f"{DATA}/sp500_monthly_membership.parquet"
    if os.path.exists(mem_path):
        print("  sp500_monthly_membership: already cached.")
        return pd.read_parquet(mem_path)

    sp500 = sp500.copy()
    sp500["date"] = pd.to_datetime(sp500["date"], errors="coerce")
    sp500 = sp500.dropna(subset=["date"])
    sp500 = sp500.sort_values(["ticker", "date"])

    monthly_dates = pd.date_range("1988-01-31", "2024-01-31", freq="ME")

    # For each ticker, track add/remove events and build monthly flag
    records = []
    # Also include 'current' entries as being in the index at that date
    active_actions = {"added", "current"}

    for tkr, grp in sp500.groupby("ticker"):
        grp = grp.sort_values("date")
        for month in monthly_dates:
            prior = grp[grp["date"] <= month]
            if len(prior) == 0:
                continue
            last_action = prior.iloc[-1]["action"]
            if last_action in active_actions:
                records.append({"date": month, "stock_id": tkr, "in_sp500": True})

    mem = pd.DataFrame(records)
    mem.to_parquet(mem_path)

    avg_per_month = mem.groupby("date")["stock_id"].count().mean()
    print(f"  SP500 membership: {len(mem):,} rows, "
          f"avg {avg_per_month:.0f} tickers/month")
    return mem

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6+7 — BUILD ACCOUNTING ΔH AND MERGE
# ═══════════════════════════════════════════════════════════════════════════

def build_and_merge(mf, sp500_mem):
    print("\n" + "="*62)
    print("STEPS 6–7 — ACCOUNTING ΔH + MERGE WITH EXISTING PANEL")
    print("="*62)

    merged_path = f"{DATA}/merged_with_accounting.parquet"
    if os.path.exists(merged_path):
        print("  merged panel: already cached.")
        return pd.read_parquet(merged_path)

    # ── Step 6: Rolling 60m std of ROE ──────────────────────────────
    mf = mf.sort_values(["stock_id", "date"])
    mf["dH_roe_raw"] = mf.groupby("stock_id")["roe"].transform(
        lambda x: x.rolling(60, min_periods=24).std()
    )
    mf["dH_accounting"] = -mf["dH_roe_raw"]

    mf["dH_gpm_raw"] = mf.groupby("stock_id")["gpm"].transform(
        lambda x: x.rolling(60, min_periods=24).std()
    )
    mf["dH_gpm"] = -mf["dH_gpm_raw"]

    # ── Step 7: Merge ────────────────────────────────────────────────
    existing = pd.read_parquet(f"{DATA}/variables_monthly.parquet")
    existing["date"] = pd.to_datetime(existing["date"])
    mf["date"] = pd.to_datetime(mf["date"])
    sp500_mem["date"] = pd.to_datetime(sp500_mem["date"])

    merged = existing.merge(
        mf[["stock_id", "date", "dH_accounting", "dH_gpm", "roe", "gpm"]],
        on=["stock_id", "date"],
        how="left",
    )
    merged = merged.merge(
        sp500_mem[["stock_id", "date", "in_sp500"]],
        on=["stock_id", "date"],
        how="left",
    )

    cov_acc = merged["dH_accounting"].notna().mean()
    cov_sp500 = merged["in_sp500"].notna().mean()

    print(f"  Existing rows: {len(existing):,}")
    print(f"  Merged rows:   {len(merged):,}")
    print(f"  Coverage — accounting ΔH: {cov_acc:.1%}, SP500 PIT: {cov_sp500:.1%}")

    if cov_acc < 0.50:
        print(f"\n  WARNING: coverage {cov_acc:.1%} < 50% — downstream FM results")
        print(f"  will be subject to selection bias. Proceeding but flagging.")

    merged.to_parquet(merged_path)
    return merged

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def run_diagnostics(merged):
    print("\n" + "="*62)
    print("STEP 8 — DIAGNOSTICS")
    print("="*62)

    factors = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
    factors.index = pd.to_datetime(factors.index)
    ff_cols = [c for c in ["Mkt_RF","SMB","HML","RMW","CMA","Mom"] if c in factors.columns]

    results = {}

    # ── Sub-panel: rows with accounting ΔH ──────────────────────────
    acc = merged.dropna(subset=["dH_accounting","DS_z","T","ret_next_month"]).copy()
    n_acc = len(acc)
    print(f"\n  Working sub-panel: {n_acc:,} stock-months, "
          f"{acc['stock_id'].nunique()} tickers, "
          f"{acc['date'].nunique()} months")

    # ── Z-score accounting ΔH ────────────────────────────────────────
    acc["dH_acc_z"] = cs_winsorize_zscore(acc, "dH_accounting")
    acc["dH_gpm_z"] = cs_winsorize_zscore(acc, "dH_gpm")
    acc["TxDS_acc"] = acc["T"] * acc["DS_z"]
    acc["dG_acc"]  = acc["dH_acc_z"] - acc["TxDS_acc"]
    acc["dG_acc_z"] = cs_winsorize_zscore(acc, "dG_acc")

    # ── Diagnostic 1: Correlations ──────────────────────────────────
    corr_roe_ds = acc.groupby("date").apply(
        lambda x: x["dH_acc_z"].corr(x["DS_z"])
    ).mean()
    corr_roe_dg = acc.groupby("date").apply(
        lambda x: x["dH_acc_z"].corr(x["dG_acc_z"])
    ).mean()
    corr_gpm_ds = acc.groupby("date").apply(
        lambda x: x["dH_gpm_z"].corr(x["DS_z"])
    ).mean()

    print(f"\n  Diagnostic 1 — Correlations:")
    print(f"    Corr(ΔH_ROE, ΔS):    {corr_roe_ds:.4f}  [was −0.853 price-based]")
    print(f"    Corr(ΔH_ROE, ΔG):    {corr_roe_dg:.4f}  [was +0.9998 price-based]")
    print(f"    Corr(ΔH_GPM, ΔS):    {corr_gpm_ds:.4f}  [GPM alternative]")

    # Choose primary ΔH: whichever has lower |corr with ΔS|
    if abs(corr_roe_ds) <= abs(corr_gpm_ds):
        primary_dh = "dH_acc_z"
        primary_label = "ROE-based"
        primary_corr_ds = corr_roe_ds
    else:
        primary_dh = "dH_gpm_z"
        primary_label = "GPM-based"
        primary_corr_ds = corr_gpm_ds
        acc["dG_acc"]  = acc[primary_dh] - acc["TxDS_acc"]
        acc["dG_acc_z"] = cs_winsorize_zscore(acc, "dG_acc")

    print(f"\n  PRIMARY ΔH: {primary_label} (Corr with ΔS = {primary_corr_ds:.4f})")
    results["corr_dH_dS"] = round(corr_roe_ds, 4)
    results["corr_gpm_dS"] = round(corr_gpm_ds, 4)
    results["corr_dH_dG"] = round(corr_roe_dg, 4)
    results["primary_dH"] = primary_label

    # ── Diagnostic 2: Quintile sort ─────────────────────────────────
    print(f"\n  Diagnostic 2 — Quintile sort on ΔG_accounting:")
    acc2 = acc.dropna(subset=["dG_acc_z", "ret_next_month"])
    acc2["_q"] = acc2.groupby("date")["dG_acc_z"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
        if x.nunique() >= 5 else np.nan
    )
    acc2 = acc2.dropna(subset=["_q"])
    qret = acc2.groupby(["date","_q"])["ret_next_month"].mean().unstack("_q")
    qret.index = pd.to_datetime(qret.index)

    rf_s = factors["RF"].reindex(qret.index).fillna(0) if "RF" in factors.columns \
           else pd.Series(0.0, index=qret.index)

    q_stats = {}
    for q in range(5):
        key = float(q) if float(q) in qret.columns else q
        if key not in qret.columns: continue
        r = qret[key].dropna()
        mean_mo = r.mean()
        std_mo  = r.std()
        q_stats[q+1] = (mean_mo, mean_mo*12, std_mo*np.sqrt(12))
        print(f"    Q{q+1}: {mean_mo*100:.3f}%/mo ({mean_mo*1200:.2f}%/yr)")

    # L/S
    q1k = float(0) if float(0) in qret.columns else 0
    q5k = float(4) if float(4) in qret.columns else 4
    if q1k in qret.columns and q5k in qret.columns:
        ls = (qret[q5k] - qret[q1k]).dropna()
        ls_ex = ls - rf_s.reindex(ls.index).fillna(0)
        f_ls = factors[ff_cols].reindex(ls.index).dropna()
        ls_sub = ls_ex.reindex(f_ls.index).dropna()
        f_sub  = f_ls.reindex(ls_sub.index)
        # Simple t-stat
        n_ls = len(ls); mean_ls = ls.mean(); std_ls = ls.std()
        t_ls_raw = mean_ls / (std_ls / np.sqrt(n_ls))
        print(f"    L/S: {mean_ls*100:.3f}%/mo ({mean_ls*1200:.2f}%/yr), "
              f"t={t_ls_raw:.3f}")
        results["ls_mean_annual"] = round(mean_ls*12, 4)
        results["ls_t_raw"] = round(t_ls_raw, 4)

        # FF5+UMD alpha
        if len(ls_sub) > 24:
            X_ls = sm.add_constant(f_sub)
            res_ls = sm.OLS(ls_sub, X_ls).fit(cov_type="HAC", cov_kwds={"maxlags":5})
            alpha_ls = res_ls.params["const"]
            t_alpha_ls = res_ls.tvalues["const"]
            print(f"    L/S FF5+UMD alpha: {alpha_ls*100:.3f}%/mo ({alpha_ls*1200:.2f}%/yr), "
                  f"NW t={t_alpha_ls:.3f}")
            results["ls_ff5_alpha_annual"] = round(alpha_ls*12, 4)
            results["ls_ff5_t"] = round(t_alpha_ls, 4)

    # ── Diagnostic 3: FM Model B (ΔH + ΔS) ─────────────────────────
    print(f"\n  Diagnostic 3 — FM Model B (accounting ΔH + ΔS):")
    fm_b = fama_macbeth_nw(
        acc.dropna(subset=[primary_dh,"DS_z","ret_next_month"]),
        "ret_next_month", [primary_dh, "DS_z"], lags=5
    )
    t_dh = fm_b.get(primary_dh, (np.nan,)*3)
    t_ds = fm_b.get("DS_z", (np.nan,)*3)
    print(f"    β_ΔH: coef={t_dh[0]:.5f}, t={t_dh[1]:.3f} (n={t_dh[2]})")
    print(f"    β_ΔS: coef={t_ds[0]:.5f}, t={t_ds[1]:.3f}")
    both_sig = abs(t_dh[1]) > 2.0 and abs(t_ds[1]) > 2.0
    print(f"    Both significant (|t|>2)? {'YES' if both_sig else 'NO'}")
    results["fm_b_t_dH"] = round(t_dh[1], 4)
    results["fm_b_t_dS"] = round(t_ds[1], 4)
    results["fm_b_both_sig"] = both_sig

    # FM Model A (ΔG accounting only)
    print(f"\n  Diagnostic 3b — FM Model A (accounting ΔG):")
    fm_a = fama_macbeth_nw(
        acc.dropna(subset=["dG_acc_z","ret_next_month"]),
        "ret_next_month", ["dG_acc_z"], lags=5
    )
    t_dg = fm_a.get("dG_acc_z", (np.nan,)*3)
    print(f"    β_ΔG_acc: coef={t_dg[0]:.5f}, t={t_dg[1]:.3f}")
    results["fm_a_t_dG_acc"] = round(t_dg[1], 4)

    # ── Diagnostic 4: Cluster-robust Wald (T·ΔS = 0) ───────────────
    print(f"\n  Diagnostic 4 — Cluster-robust Wald test (T·ΔS=0):")
    sub_w = acc.dropna(subset=[primary_dh,"DS_z","TxDS_acc","ret_next_month"]).copy()
    n_w = len(sub_w)
    X_w = np.column_stack([
        np.ones(n_w),
        sub_w[primary_dh].values,
        sub_w["DS_z"].values,
        sub_w["TxDS_acc"].values,
    ])
    y_w = sub_w["ret_next_month"].values
    beta_w, *_ = np.linalg.lstsq(X_w, y_w, rcond=None)
    resid_w = y_w - X_w @ beta_w
    grp_w = pd.Categorical(sub_w["date"]).codes
    vcov_w = cluster_vcov(X_w, resid_w, grp_w)
    b_txds = beta_w[3]; se_txds = np.sqrt(vcov_w[3,3])
    t_txds = b_txds / se_txds; wald_txds = t_txds**2
    p_txds = 1 - chi2.cdf(wald_txds, 1)
    print(f"    T·ΔS: β={b_txds:.5f}, cluster t={t_txds:.3f}, "
          f"Wald χ²={wald_txds:.3f}, p={p_txds:.4f}")
    print(f"    [was p=0.013 with price-based ΔH]")
    results["wald_txds_p"] = round(p_txds, 4)
    results["wald_txds_t"] = round(t_txds, 4)

    # ── Diagnostic 5: Point-in-time SP500 universe ──────────────────
    print(f"\n  Diagnostic 5 — Point-in-time SP500 universe:")
    pit = acc[acc["in_sp500"] == True].copy()
    n_pit = len(pit)
    avg_pit = pit.groupby("date")["stock_id"].count().mean()
    print(f"    PIT sample: {n_pit:,} stock-months, avg {avg_pit:.0f}/month")

    if n_pit > 1000:
        # FM on PIT universe
        pit["dG_acc_z_pit"] = cs_winsorize_zscore(pit, "dG_acc")
        fm_pit = fama_macbeth_nw(
            pit.dropna(subset=["dG_acc_z_pit","ret_next_month"]),
            "ret_next_month", ["dG_acc_z_pit"], lags=5
        )
        t_pit = fm_pit.get("dG_acc_z_pit", (np.nan,)*3)
        print(f"    FM t(ΔG) PIT: {t_pit[1]:.3f}")

        # Quintile sort on PIT
        pit["_qp"] = pit.groupby("date")["dG_acc_z_pit"].transform(
            lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
            if x.nunique() >= 5 else np.nan
        )
        pit2 = pit.dropna(subset=["_qp"])
        qret_p = pit2.groupby(["date","_qp"])["ret_next_month"].mean().unstack("_qp")
        q1p_key = float(0) if float(0) in qret_p.columns else 0
        if q1p_key in qret_p.columns:
            q1_pit = qret_p[q1p_key].dropna().mean()
            print(f"    Q1 PIT annualized: {q1_pit*1200:.2f}%  [was 30.12% biased]")
            results["q1_pit_annual"] = round(q1_pit*12, 4)

        results["pit_n"] = n_pit
        results["pit_avg_per_month"] = round(avg_pit, 0)
        results["fm_pit_t_dG"] = round(t_pit[1], 4)
    else:
        print(f"    PIT sample too small ({n_pit}) — SP500 history may not cover this panel")
        results["pit_n"] = n_pit

    return results, acc

# ═══════════════════════════════════════════════════════════════════════════
# STEP 9 — SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════

def save_results(results, merged):
    print("\n" + "="*62)
    print("STEP 9 — SAVING RESULTS")
    print("="*62)

    merged.to_parquet(f"{DATA}/merged_with_accounting.parquet")
    print(f"  Saved: {DATA}/merged_with_accounting.parquet")

    lines = [
        "=== SHARADAR ACCOUNTING REBUILD RESULTS ===",
        "",
        "COVERAGE",
        f"- Primary ΔH variable: {results.get('primary_dH','?')}",
        f"- Corr(ΔH_ROE, ΔS): {results.get('corr_dH_dS','?'):.4f}  [was -0.853 price-based]",
        f"- Corr(ΔH_GPM, ΔS): {results.get('corr_gpm_dS','?'):.4f}  [GPM alternative]",
        f"- Corr(ΔH_ROE, ΔG): {results.get('corr_dH_dG','?'):.4f}  [was 0.9998 price-based]",
        "",
        "FM MODEL A (accounting ΔG, NW-5)",
        f"FM t(ΔG_acc): {results.get('fm_a_t_dG_acc','?')}  [was -3.98 price-based]",
        "",
        "FM MODEL B (accounting ΔH + ΔS, NW-5)",
        f"β_ΔH = t={results.get('fm_b_t_dH','?')}",
        f"β_ΔS = t={results.get('fm_b_t_dS','?')}",
        f"Both significant (|t|>2)? {results.get('fm_b_both_sig','?')}",
        "",
        "L/S PORTFOLIO",
        f"L/S annualized: {results.get('ls_mean_annual','?')}  FF5+UMD alpha t={results.get('ls_ff5_t','?')}",
        "",
        "CLUSTER-ROBUST WALD TEST (T·ΔS = 0)",
        f"p-value: {results.get('wald_txds_p','?')}  [was 0.013 price-based]",
        f"cluster t: {results.get('wald_txds_t','?')}",
        "",
        "POINT-IN-TIME UNIVERSE (SP500 historical)",
        f"N stock-months: {results.get('pit_n','?')}",
        f"Avg per month: {results.get('pit_avg_per_month','?')}",
        f"Q1 annualized (PIT): {results.get('q1_pit_annual','?')}  [was 30.12% biased]",
        f"FM t(ΔG) PIT: {results.get('fm_pit_t_dG','?')}  [was -3.98/-2.99 biased]",
        "",
        "=== END ===",
    ]
    txt = "\n".join(lines)
    outpath = f"{DATA}/sharadar_rebuild_results.txt"
    with open(outpath, "w") as f:
        f.write(txt)
    print(f"  Saved: {outpath}")
    print(f"\n{txt}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  SHARADAR ACCOUNTING REBUILD PIPELINE")
    print("=" * 65)

    tickers, sp500 = download_tables()
    sf1_vars = build_sf1_variables(tickers)
    mf = build_monthly_fundamentals(sf1_vars)
    sp500_mem = build_sp500_membership(sp500)
    merged = build_and_merge(mf, sp500_mem)
    results, acc = run_diagnostics(merged)
    save_results(results, merged)

    print("\n  Pipeline complete.")


if __name__ == "__main__":
    main()
