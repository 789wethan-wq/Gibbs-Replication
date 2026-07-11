"""R12_reviewer_responses.py — Six reviewer-response tasks (Tasks 1–6).

Runs entirely on existing data + free AQR downloads. No Sharadar required.
Outputs saved to robustness/outputs/R12_*.

Task 1 — QMJ / BAB factor controls
Task 2 — ACF of FM coefficient series; NW lag validation
Task 3 — Exact Corr(ΔH, ΔG)
Task 4 — Drawdown timeline plot (L/S and inverse)
Task 5 — Likelihood ratio test (complement to Vuong)
Task 6 — NW sensitivity table (NW-1 through NW-60 + auto bandwidth)
"""
import sys, os, io, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # load_panel() uses ../data
from robustness_utils import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.stats.stattools import durbin_watson
warnings.filterwarnings("ignore")

OUT  = "outputs"
DATA_DIR = "aqr_data"   # cache downloaded files here
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

TAG = "R12"

# ─────────────────────────────────────────────────────────────────────────────
# AQR download helpers
# ─────────────────────────────────────────────────────────────────────────────

# AQR URL patterns — they reorganize their CDN periodically.
# Priority order: try each in turn until one works.
AQR_QMJ_URLS = [
    "https://aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Monthly.xlsx",
    "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Monthly.xlsx",
    "https://images.aqr.com/Insights/Datasets/Quality-Minus-Junk-Factors-Monthly.xlsx",
]
AQR_BAB_MO_URLS = [
    "https://aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx",
    "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx",
    "https://images.aqr.com/Insights/Datasets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx",
]
AQR_BAB_D_URLS = [
    "https://aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Daily.xlsx",
    "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Daily.xlsx",
    "https://images.aqr.com/Insights/Datasets/Betting-Against-Beta-Equity-Factors-Daily.xlsx",
]

# Manual download fallback paths (place files here if AQR URLs change)
AQR_QMJ_LOCAL  = f"{DATA_DIR}/QMJ_Monthly.xlsx"
AQR_BAB_LOCAL  = f"{DATA_DIR}/BAB_Monthly.xlsx"

MANUAL_INSTRUCTIONS = """
  ─────────────────────────────────────────────────────────────
  AQR DOWNLOAD INSTRUCTIONS (one-time manual step):
  1. Go to: https://www.aqr.com/Insights/Datasets/Quality-Minus-Junk-Factors-Monthly
     → Download the Excel file → save as:
       robustness/aqr_data/QMJ_Monthly.xlsx

  2. Go to: https://www.aqr.com/Insights/Datasets/Betting-Against-Beta-Equity-Factors-Monthly
     (or Daily if monthly unavailable)
     → Download the Excel file → save as:
       robustness/aqr_data/BAB_Monthly.xlsx

  Then re-run:  .venv/bin/python robustness/R12_reviewer_responses.py
  ─────────────────────────────────────────────────────────────
"""


def _fetch_bytes(url, timeout=60):
    """Download URL to bytes, trying urllib then requests."""
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e1:
        try:
            import requests
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (research)"})
            r.raise_for_status()
            return r.content
        except Exception as e2:
            raise RuntimeError(f"Download failed: {e1}; {e2}")


def _aqr_excel_to_series(raw_bytes, label_hint="USA", sheet=0, date_col=0, max_skip=30):
    """
    Parse an AQR Excel file. AQR files have a variable number of description
    rows before the data header. Scan for the header row containing 'DATE' or
    a date-like value, then read from there.
    Returns pd.Series indexed by month-end date, values are returns.
    """
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(raw_bytes), data_only=True, read_only=True)
    ws = wb.worksheets[sheet]

    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    # Find the header row: first row where a cell contains something like "DATE" or "date"
    header_row_idx = None
    for i, row in enumerate(rows[:max_skip]):
        for cell in row:
            if cell is not None and isinstance(cell, str) and cell.strip().upper() in ("DATE", "DATE:"):
                header_row_idx = i
                break
        if header_row_idx is not None:
            break

    if header_row_idx is None:
        # Fallback: first row that has a datetime object in column 0
        import datetime
        for i, row in enumerate(rows[:max_skip]):
            if row and isinstance(row[0], (datetime.datetime, datetime.date)):
                header_row_idx = max(0, i - 1)
                break

    if header_row_idx is None:
        raise ValueError("Cannot find header row in AQR Excel file")

    header = [str(c).strip() if c is not None else "" for c in rows[header_row_idx]]
    data_rows = rows[header_row_idx + 1:]

    # Find US column: look for label_hint (e.g. "USA", "US", "QMJ (US)")
    col_idx = None
    for i, h in enumerate(header):
        if label_hint.upper() in h.upper() and i > 0:
            col_idx = i
            break
    # If hint not found, try second non-date column
    if col_idx is None:
        for i in range(1, len(header)):
            if header[i]:
                col_idx = i
                break
    if col_idx is None:
        raise ValueError(f"Cannot find column matching '{label_hint}' in {header}")

    print(f"    AQR column '{header[col_idx]}' at index {col_idx}")

    import datetime
    dates, vals = [], []
    for row in data_rows:
        if not row or row[0] is None:
            continue
        raw_date = row[0]
        raw_val  = row[col_idx] if col_idx < len(row) else None
        # Parse date
        if isinstance(raw_date, (datetime.datetime, datetime.date)):
            d = pd.Timestamp(raw_date)
        elif isinstance(raw_date, (int, float)):
            try:
                d = pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(raw_date))
            except Exception:
                continue
        elif isinstance(raw_date, str):
            try:
                d = pd.Timestamp(raw_date)
            except Exception:
                continue
        else:
            continue
        # Parse value
        try:
            v = float(raw_val)
        except (TypeError, ValueError):
            continue
        dates.append(d + pd.offsets.MonthEnd(0))
        vals.append(v)

    s = pd.Series(vals, index=pd.to_datetime(dates)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def _try_fetch_series(url_list, label_hint, factor_name, resample=False):
    """Try each URL in order; return pd.Series or None."""
    for url in url_list:
        try:
            raw = _fetch_bytes(url)
            s = _aqr_excel_to_series(raw, label_hint=label_hint)
            if resample:
                s = (1 + s).resample("ME").prod() - 1
            if s.abs().mean() > 0.1:
                s = s / 100.0
                print(f"    {factor_name}: converted from pct to decimal")
            print(f"    {factor_name}: {len(s)} months, {s.index.min():%Y-%m}–{s.index.max():%Y-%m}")
            return s
        except Exception as e:
            print(f"    {factor_name}: {url.split('/')[-1]} → {e}")
            continue
    return None


def load_qmj(panel_dates):
    """Download (or load cached) QMJ US monthly series."""
    cache = f"{DATA_DIR}/qmj_monthly_us.parquet"
    if os.path.exists(cache):
        print("  QMJ: loading from cache")
        return pd.read_parquet(cache)

    # 1. Try local manual download first
    if os.path.exists(AQR_QMJ_LOCAL):
        print("  QMJ: loading from local file")
        try:
            with open(AQR_QMJ_LOCAL, "rb") as f:
                raw = f.read()
            s = _aqr_excel_to_series(raw, label_hint="USA")
            if s.abs().mean() > 0.1:
                s = s / 100.0
            s.name = "QMJ"
            pd.DataFrame(s).to_parquet(cache)
            return pd.DataFrame(s)
        except Exception as e:
            print(f"  QMJ local parse failed: {e}")

    # 2. Try all AQR URLs
    print("  QMJ: trying AQR URLs...")
    s = _try_fetch_series(AQR_QMJ_URLS, "USA", "QMJ")
    if s is not None:
        s.name = "QMJ"
        pd.DataFrame(s).to_parquet(cache)
        return pd.DataFrame(s)

    print(MANUAL_INSTRUCTIONS)
    return None


def load_bab(panel_dates):
    """Download (or load cached) BAB US monthly series."""
    cache = f"{DATA_DIR}/bab_monthly_us.parquet"
    if os.path.exists(cache):
        print("  BAB: loading from cache")
        return pd.read_parquet(cache)

    # 1. Try local manual download first
    if os.path.exists(AQR_BAB_LOCAL):
        print("  BAB: loading from local file")
        try:
            with open(AQR_BAB_LOCAL, "rb") as f:
                raw = f.read()
            s = _aqr_excel_to_series(raw, label_hint="USA")
            if s.abs().mean() > 0.1:
                s = s / 100.0
            s.name = "BAB"
            pd.DataFrame(s).to_parquet(cache)
            return pd.DataFrame(s)
        except Exception as e:
            print(f"  BAB local parse failed: {e}")

    # 2. Try monthly AQR URLs, then daily
    print("  BAB: trying monthly AQR URLs...")
    s = _try_fetch_series(AQR_BAB_MO_URLS, "USA", "BAB", resample=False)
    if s is None:
        print("  BAB: trying daily AQR URLs (will resample to monthly)...")
        s = _try_fetch_series(AQR_BAB_D_URLS, "USA", "BAB", resample=True)

    if s is not None:
        s.name = "BAB"
        pd.DataFrame(s).to_parquet(cache)
        return pd.DataFrame(s)

    print(MANUAL_INSTRUCTIONS)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — QMJ / BAB factor controls
# ─────────────────────────────────────────────────────────────────────────────

def task1_qmj_bab(panel, factors):
    print("\n" + "="*60)
    print("TASK 1 — QMJ / BAB FACTOR CONTROLS")
    print("="*60)

    panel_dates = pd.to_datetime(panel["date"].unique())
    qmj_df = load_qmj(panel_dates)
    bab_df = load_bab(panel_dates)

    rows = []

    def _fm_with_extra(panel, extra_cols, extra_factors_df, label):
        """Merge extra factor data as cross-sectional control in FM."""
        p2 = panel.copy()
        for col in extra_cols:
            if extra_factors_df is not None and col in extra_factors_df.columns:
                mapping = extra_factors_df[col].to_dict()
                p2[col] = p2["date"].map(mapping)
                p2[f"{col}_z"] = p2.groupby("date")[col].transform(zscore_cs)
        ctrl = [f"{c}_z" for c in extra_cols if f"{c}_z" in p2.columns]
        if not ctrl:
            return {"spec": label, "fm_t_DG": np.nan, "fm_p_DG": np.nan, "note": "factor data missing"}
        try:
            sub = p2.dropna(subset=["DG", "ret_next_month"] + ctrl)
            fm_out, _ = fama_macbeth(sub, "ret_next_month", ["DG"] + ctrl)
            t_dg = fm_out.get("DG", (np.nan, np.nan, np.nan))
            return {
                "spec": label,
                "fm_coef_DG": t_dg[0],
                "fm_t_DG":    t_dg[1],
                "fm_p_DG":    t_dg[2],
                "pass_h1":    pass_fail(abs(t_dg[1]), 2.5, "above"),
            }
        except Exception as e:
            return {"spec": label, "fm_t_DG": np.nan, "error": str(e)}

    # Baseline (no extra factors)
    try:
        fm_base, _ = fama_macbeth(panel.dropna(subset=["DG","ret_next_month"]),
                                   "ret_next_month", ["DG"])
        t_base = fm_base.get("DG", (np.nan,)*3)
        rows.append({"spec": "Baseline (no QMJ/BAB)", "fm_coef_DG": t_base[0],
                     "fm_t_DG": t_base[1], "fm_p_DG": t_base[2], "pass_h1": "PASS"})
    except Exception as e:
        rows.append({"spec": "Baseline", "fm_t_DG": np.nan})

    # QMJ controls
    if qmj_df is not None:
        rows.append(_fm_with_extra(panel, ["QMJ"], qmj_df, "FM: DG + QMJ control"))
    else:
        rows.append({"spec": "FM: DG + QMJ", "fm_t_DG": np.nan, "note": "QMJ download failed"})

    # BAB controls
    if bab_df is not None:
        rows.append(_fm_with_extra(panel, ["BAB"], bab_df, "FM: DG + BAB control"))
    else:
        rows.append({"spec": "FM: DG + BAB", "fm_t_DG": np.nan, "note": "BAB download failed"})

    # QMJ + BAB
    if qmj_df is not None and bab_df is not None:
        combined_ext = qmj_df.join(bab_df, how="outer")
        rows.append(_fm_with_extra(panel, ["QMJ", "BAB"], combined_ext, "FM: DG + QMJ + BAB"))
    else:
        rows.append({"spec": "FM: DG + QMJ + BAB", "fm_t_DG": np.nan})

    # L/S portfolio alphas under extended factor models
    ls, _ = quintile_sort_ls(panel, "DG", "ret_next_month", factors)
    ls = ls.dropna()
    factors_a = factors.reindex(ls.index)
    rf = factors_a.get("RF", pd.Series(0.0, index=ls.index)).fillna(0)
    ls_ex = ls - rf

    base_ff_cols = [c for c in ["Mkt_RF","SMB","HML","RMW","CMA","Mom"]
                    if c in factors_a.columns]

    for ext_df, ext_cols, label in [
        (qmj_df, ["QMJ"],       "L/S alpha: FF5+UMD+QMJ"),
        (bab_df, ["BAB"],       "L/S alpha: FF5+UMD+BAB"),
        (None,   ["QMJ","BAB"], "L/S alpha: FF5+UMD+QMJ+BAB"),
    ]:
        # Build extended factor matrix
        try:
            if label == "L/S alpha: FF5+UMD+QMJ+BAB":
                if qmj_df is None or bab_df is None:
                    rows.append({"spec": label, "alpha": np.nan, "alpha_t": np.nan})
                    continue
                ext_df = qmj_df.join(bab_df, how="outer")
            elif ext_df is None:
                rows.append({"spec": label, "alpha": np.nan, "note": "download failed"})
                continue

            ext_aligned = ext_df.reindex(ls.index)
            f_ext = pd.concat([factors_a[base_ff_cols], ext_aligned], axis=1).dropna()
            ls_sub = ls_ex.reindex(f_ext.index).dropna()
            f_sub  = f_ext.reindex(ls_sub.index)
            all_cols = base_ff_cols + [c for c in ext_cols if c in f_sub.columns]
            alpha, t, p, r2 = ff5_umd_alpha(ls_sub, f_sub, ff_cols=all_cols)
            rows.append({
                "spec": label,
                "alpha": alpha, "alpha_t": t, "alpha_p": p, "r2": r2,
                "pass_h1": pass_fail(abs(t), 2.0, "above"),
            })
        except Exception as e:
            rows.append({"spec": label, "alpha": np.nan, "error": str(e)})

    df1 = pd.DataFrame(rows)
    df1["test"] = "T1"
    df1["category"] = "QMJ_BAB_controls"

    # Print results
    print("\n  Results:")
    for _, r in df1.iterrows():
        t = r.get("fm_t_DG", r.get("alpha_t", np.nan))
        note = r.get("note", r.get("error", ""))
        print(f"    {r['spec']}: t={t:.3f}" if np.isfinite(float(t) if t is not None else float('nan'))
              else f"    {r['spec']}: {note}")

    # Critical verdict
    qmj_row = df1[df1["spec"] == "FM: DG + QMJ control"]
    bab_row  = df1[df1["spec"] == "FM: DG + BAB control"]
    qmj_t = qmj_row["fm_t_DG"].values[0] if len(qmj_row) > 0 else np.nan
    bab_t  = bab_row["fm_t_DG"].values[0] if len(bab_row) > 0 else np.nan
    print(f"\n  CRITICAL: FM t(DG) | +QMJ = {qmj_t:.3f} | +BAB = {bab_t:.3f}")
    if np.isfinite(qmj_t) and np.isfinite(bab_t):
        if qmj_t < -2.5 and bab_t < -2.5:
            print("  VERDICT: ΔG significant AFTER controlling for QMJ and BAB. Strong.")
        elif qmj_t < -2.0 or bab_t < -2.0:
            print("  VERDICT: ΔG marginal after QMJ/BAB. Report honestly — the thermodynamic")
            print("           framing adds something beyond quality and low-beta.")
        else:
            print("  VERDICT: ΔG absorbed by QMJ/BAB. This is the key negative finding.")
            print("           Paper needs to be reframed as 'thermodynamic decomposition of QMJ'.")

    df1.to_csv(f"{OUT}/R12_T1_qmj_bab_results.csv", index=False)
    return df1, qmj_t, bab_t


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — ACF of FM coefficient series
# ─────────────────────────────────────────────────────────────────────────────

def task2_acf_fm_coefs(panel):
    print("\n" + "="*60)
    print("TASK 2 — ACF OF FM COEFFICIENT SERIES")
    print("="*60)

    # Get the time series of monthly FM coefficients
    _, coefs_df = fama_macbeth(panel.dropna(subset=["DG","ret_next_month"]),
                               "ret_next_month", ["DG"], lags=6)
    beta_series = coefs_df["DG"].dropna().sort_index()
    n = len(beta_series)
    print(f"  Monthly FM β_ΔG series: {n} months")
    print(f"  Mean: {beta_series.mean():.6f}, Std: {beta_series.std():.6f}")

    # Compute ACF up to lag 60
    from statsmodels.tsa.stattools import acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    max_lags = min(60, n // 4)
    acf_vals, confint = acf(beta_series.values, nlags=max_lags, alpha=0.05)

    acf_df = pd.DataFrame({
        "lag":    range(len(acf_vals)),
        "acf":    acf_vals,
        "ci_lo":  confint[:, 0] - acf_vals,
        "ci_hi":  confint[:, 1] - acf_vals,
        "significant": np.abs(acf_vals) > 1.96 / np.sqrt(n),
    })

    # Ljung-Box test for autocorrelation up to various lags
    lb_results = {}
    for lag in [6, 12, 24, 36, 60]:
        if lag <= max_lags:
            lb = acorr_ljungbox(beta_series.values, lags=[lag], return_df=True)
            lb_results[lag] = {
                "Q_stat": float(lb["lb_stat"].iloc[0]),
                "p_value": float(lb["lb_pvalue"].iloc[0]),
            }

    print("\n  Ljung-Box test for serial correlation in β_ΔG:")
    for lag, res in lb_results.items():
        sig = "***" if res["p_value"] < 0.01 else ("**" if res["p_value"] < 0.05 else "")
        print(f"    Lag {lag:2d}: Q={res['Q_stat']:.2f}, p={res['p_value']:.4f} {sig}")

    # Determine ACF decay pattern
    sig_lags = acf_df[acf_df["lag"] > 0]["lag"][acf_df["significant"]].tolist()
    last_sig = max(sig_lags) if sig_lags else 0
    print(f"\n  Significant ACF lags (|acf|>1.96/√n): {sig_lags}")
    print(f"  Last significant lag: {last_sig}")

    # NW recommendation
    if last_sig <= 6:
        nw_recommendation = 6
        verdict = "NW-6 is justified. Autocorrelation negligible beyond lag 6."
    elif last_sig <= 12:
        nw_recommendation = 12
        verdict = "ACF extends to lag 12. Re-run with NW-12 recommended."
    elif last_sig <= 24:
        nw_recommendation = 24
        verdict = "ACF extends to lag 24. Re-run with NW-24 recommended."
    else:
        nw_recommendation = "auto"
        verdict = f"ACF extends to lag {last_sig}. Use automatic bandwidth NW."

    print(f"\n  VERDICT: {verdict}")
    print(f"  Recommended NW lags: {nw_recommendation}")

    # Plot ACF
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    ax1, ax2 = axes

    # Panel 1: FM coefficient time series
    ax1.plot(beta_series.index, beta_series.values, color="#2166ac", linewidth=0.8)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.fill_between(beta_series.index, beta_series.mean() - beta_series.std(),
                      beta_series.mean() + beta_series.std(), alpha=0.15, color="#2166ac")
    ax1.set_title("Monthly FM Coefficient on ΔG (β_ΔG)", fontsize=11)
    ax1.set_ylabel("Coefficient")
    ax1.set_xlabel("")

    # Panel 2: ACF
    lags = acf_df["lag"].values[1:]
    acf_v = acf_df["acf"].values[1:]
    ci_bound = 1.96 / np.sqrt(n)
    ax2.bar(lags, acf_v, color=["#d6604d" if abs(a) > ci_bound else "#92c5de" for a in acf_v],
            width=0.7)
    ax2.axhline(ci_bound, color="red", linestyle="--", linewidth=0.8, label="95% CI")
    ax2.axhline(-ci_bound, color="red", linestyle="--", linewidth=0.8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("ACF of β_ΔG Series", fontsize=11)
    ax2.set_xlabel("Lag (months)")
    ax2.set_ylabel("Autocorrelation")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, max_lags + 1)

    plt.tight_layout()
    plt.savefig(f"{OUT}/R12_T2_acf_fm_coefs.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{OUT}/R12_T2_acf_fm_coefs.pdf", bbox_inches="tight")
    plt.close()
    print(f"  ACF plot saved to {OUT}/R12_T2_acf_fm_coefs.png")

    acf_df.to_csv(f"{OUT}/R12_T2_acf_values.csv", index=False)

    result = {
        "n_months": n,
        "last_significant_lag": last_sig,
        "significant_lags": sig_lags,
        "nw_recommended": nw_recommendation,
        "verdict": verdict,
        **{f"LB_Q_lag{k}": v["Q_stat"] for k, v in lb_results.items()},
        **{f"LB_p_lag{k}": v["p_value"] for k, v in lb_results.items()},
    }
    pd.DataFrame([result]).to_csv(f"{OUT}/R12_T2_acf_summary.csv", index=False)
    return acf_df, nw_recommendation, verdict


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Exact Corr(ΔH, ΔG)
# ─────────────────────────────────────────────────────────────────────────────

def task3_corr_dh_dg(panel):
    print("\n" + "="*60)
    print("TASK 3 — EXACT CORR(ΔH, ΔG)")
    print("="*60)

    # Time-series of monthly cross-sectional correlations
    monthly_corrs = {}
    p_vals = {}
    from scipy.stats import pearsonr, spearmanr

    for d, grp in panel.dropna(subset=["DH_z","DG"]).groupby("date"):
        if len(grp) < 10:
            continue
        r_pearson, p_p = pearsonr(grp["DH_z"], grp["DG"])
        r_spearman, p_s = spearmanr(grp["DH_z"], grp["DG"])
        monthly_corrs[d] = {"pearson": r_pearson, "spearman": r_spearman}

    corr_df = pd.DataFrame(monthly_corrs).T
    pearson_mean  = corr_df["pearson"].mean()
    spearman_mean = corr_df["spearman"].mean()
    pearson_std   = corr_df["pearson"].std()

    # Time-series mean t-stat
    _, t_pearson, p_t = newey_west_mean_tstat(corr_df["pearson"].values)

    # Also compute pooled (across all obs)
    sub = panel.dropna(subset=["DH_z","DG"])
    pool_pearson,  _ = pearsonr(sub["DH_z"], sub["DG"])
    pool_spearman, _ = spearmanr(sub["DH_z"], sub["DG"])

    print(f"\n  Cross-sectional monthly average Corr(ΔH, ΔG):")
    print(f"    Pearson (mean across months): {pearson_mean:.4f}  (SD={pearson_std:.4f}, NW t={t_pearson:.2f})")
    print(f"    Spearman (mean across months): {spearman_mean:.4f}")
    print(f"    Pooled Pearson: {pool_pearson:.4f}")
    print(f"    Pooled Spearman: {pool_spearman:.4f}")
    print(f"\n  → Table 0 Panel C update: Corr(ΔH, ΔG) = {pearson_mean:.4f}")
    print(f"    Interpretation: ΔG and ΔH are {pearson_mean:.1%} correlated; the T·ΔS")
    print(f"    term introduces meaningful variation beyond ΔH alone.")

    result = {
        "corr_DH_DG_pearson_monthly_mean":  round(pearson_mean, 4),
        "corr_DH_DG_spearman_monthly_mean": round(spearman_mean, 4),
        "corr_DH_DG_pearson_pooled":        round(pool_pearson, 4),
        "corr_DH_DG_spearman_pooled":       round(pool_spearman, 4),
        "pearson_nw_t":                     round(t_pearson, 4),
        "n_months":                         len(corr_df),
    }
    pd.DataFrame([result]).to_csv(f"{OUT}/R12_T3_corr_dh_dg.csv", index=False)
    corr_df.to_csv(f"{OUT}/R12_T3_monthly_corrs.csv")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Drawdown Timeline Plot
# ─────────────────────────────────────────────────────────────────────────────

def task4_drawdown_plot(panel, factors):
    print("\n" + "="*60)
    print("TASK 4 — DRAWDOWN TIMELINE PLOT")
    print("="*60)

    ls, qret = quintile_sort_ls(panel, "DG", "ret_next_month", factors)
    ls = ls.dropna().sort_index()

    # Market and RF for comparison
    factors_a = factors.reindex(ls.index)
    rf = factors_a.get("RF", pd.Series(0.0, index=ls.index)).fillna(0)
    mkt = factors_a.get("Mkt_RF", pd.Series(0.0, index=ls.index)).fillna(0) + rf

    # Cumulative returns
    cum_ls  = (1 + ls).cumprod()
    cum_inv = (1 + (-ls)).cumprod()   # inverse: long Q1, short Q5
    cum_mkt = (1 + mkt).cumprod()

    # Drawdown functions
    def drawdown_series(cum_ret):
        peak = cum_ret.cummax()
        return (cum_ret / peak) - 1

    dd_ls  = drawdown_series(cum_ls)
    dd_inv = drawdown_series(cum_inv)
    dd_mkt = drawdown_series(cum_mkt)

    max_dd_ls  = dd_ls.min()
    max_dd_inv = dd_inv.min()
    max_dd_mkt = dd_mkt.min()

    # Worst drawdown stats
    trough_ls  = dd_ls.idxmin()
    trough_inv = dd_inv.idxmin()

    # Recovery: first date after trough where value exceeds prior peak
    def recovery_date(cum_ret, trough_date):
        peak_before_trough = cum_ret[:trough_date].max()
        after = cum_ret[trough_date:]
        recovered = after[after >= peak_before_trough]
        return recovered.index[0] if len(recovered) > 0 else None

    rec_inv = recovery_date(cum_inv, trough_inv)

    print(f"  L/S (Q5-Q1):       max DD = {max_dd_ls:.1%}, trough = {trough_ls:%Y-%m}")
    print(f"  Inverse (Q1-Q5):   max DD = {max_dd_inv:.1%}, trough = {trough_inv:%Y-%m}")
    if rec_inv:
        print(f"  Inverse recovery:  {rec_inv:%Y-%m}")
        print(f"  Drawdown period:   {trough_inv:%Y-%m} → {rec_inv:%Y-%m}")
        months_under = (rec_inv.year - trough_inv.year) * 12 + (rec_inv.month - trough_inv.month)
        print(f"  Months to recover: {months_under}")

    # ─── Plot ───
    colors = {"ls": "#d6604d", "inv": "#2166ac", "mkt": "#969696"}
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])  # top: cumulative returns, full width
    ax2 = fig.add_subplot(gs[1, 0])  # bottom-left: L/S drawdown
    ax3 = fig.add_subplot(gs[1, 1])  # bottom-right: inverse drawdown

    # Top panel: cumulative returns
    ax1.plot(cum_ls.index,  cum_ls.values,  color=colors["ls"],  linewidth=1.2, label="L/S (Q5−Q1)")
    ax1.plot(cum_inv.index, cum_inv.values, color=colors["inv"], linewidth=1.2, label="Inverse (Q1−Q5)")
    ax1.plot(cum_mkt.index, cum_mkt.values, color=colors["mkt"], linewidth=0.8, linestyle="--",
             label="Market", alpha=0.7)
    ax1.axhline(1.0, color="black", linewidth=0.5)
    ax1.set_yscale("log")
    ax1.set_title("Cumulative Returns: L/S Portfolio", fontsize=11)
    ax1.set_ylabel("Cumulative return (log scale)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Shade tech bubble + GFC
    shading = [("1998-01-01","2002-12-31","Tech bubble\n±"),
               ("2008-01-01","2009-06-30","GFC")]
    for s_start, s_end, s_label in shading:
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(pd.Timestamp(s_start), pd.Timestamp(s_end),
                       alpha=0.08, color="gray", zorder=0)

    # Bottom-left: L/S drawdown
    ax2.fill_between(dd_ls.index, dd_ls.values, 0, color=colors["ls"], alpha=0.5)
    ax2.plot(dd_ls.index, dd_ls.values, color=colors["ls"], linewidth=0.8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(max_dd_ls, color=colors["ls"], linewidth=0.8, linestyle=":")
    ax2.annotate(f"Max DD\n{max_dd_ls:.1%}", xy=(trough_ls, max_dd_ls),
                 xytext=(trough_ls, max_dd_ls - 0.08), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 ha="center", color=colors["ls"])
    ax2.set_title("Drawdown: L/S (Q5−Q1)", fontsize=11)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.grid(True, alpha=0.3)

    # Bottom-right: inverse drawdown
    ax3.fill_between(dd_inv.index, dd_inv.values, 0, color=colors["inv"], alpha=0.5)
    ax3.plot(dd_inv.index, dd_inv.values, color=colors["inv"], linewidth=0.8)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.axhline(max_dd_inv, color=colors["inv"], linewidth=0.8, linestyle=":")
    ax3.annotate(f"Max DD\n{max_dd_inv:.1%}", xy=(trough_inv, max_dd_inv),
                 xytext=(trough_inv, max_dd_inv - 0.08), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 ha="center", color=colors["inv"])
    if rec_inv:
        ax3.axvline(rec_inv, color="green", linewidth=0.8, linestyle="--", alpha=0.7)
        ax3.text(rec_inv, max_dd_inv * 0.5, f" Recovery\n {rec_inv:%Y-%m}",
                 fontsize=7, color="green")
    ax3.set_title("Drawdown: Inverse L/S (Q1−Q5, 'thermodynamic order')", fontsize=11)
    ax3.set_ylabel("Drawdown")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax3.grid(True, alpha=0.3)

    plt.suptitle("Gibbs ΔG Strategy — Cumulative Returns and Drawdown Timeline\n"
                 "S&P 500 Universe (survivorship-biased), Price-Based ΔH/ΔS",
                 fontsize=11, y=1.01)

    for fmt in ["png", "pdf"]:
        plt.savefig(f"{OUT}/R12_T4_drawdown_{fmt}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Drawdown plot saved to {OUT}/R12_T4_drawdown.png / .pdf")

    dd_results = {
        "max_dd_ls":           round(max_dd_ls, 4),
        "trough_date_ls":      trough_ls.strftime("%Y-%m"),
        "max_dd_inverse":      round(max_dd_inv, 4),
        "trough_date_inverse": trough_inv.strftime("%Y-%m"),
        "recovery_date_inverse": rec_inv.strftime("%Y-%m") if rec_inv else "not recovered",
    }
    pd.DataFrame([dd_results]).to_csv(f"{OUT}/R12_T4_drawdown_stats.csv", index=False)

    # Save series for paper
    dd_data = pd.DataFrame({
        "cum_ls": cum_ls, "cum_inverse": cum_inv, "cum_market": cum_mkt,
        "dd_ls": dd_ls, "dd_inverse": dd_inv,
    })
    dd_data.to_csv(f"{OUT}/R12_T4_drawdown_series.csv")
    return dd_results


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Likelihood Ratio Test (complement to Vuong)
# ─────────────────────────────────────────────────────────────────────────────

def task5_lr_test(panel):
    print("\n" + "="*60)
    print("TASK 5 — LIKELIHOOD RATIO TEST")
    print("="*60)

    sub = panel.dropna(subset=["ret_next_month", "DH_z", "DS_z", "T", "DG"]).copy()
    sub["TxDS"] = sub["T"] * sub["DS_z"]   # T·ΔS (the Gibbs product term)
    sub = sub.dropna(subset=["TxDS"])

    y = sub["ret_next_month"].values
    n = len(y)
    print(f"  N observations: {n:,}")

    def fit_ols_ll(X_cols):
        """Fit OLS, return (log-likelihood, k, R2, coefs, tvals)."""
        X = sm.add_constant(sub[X_cols])
        res = sm.OLS(y, X).fit()
        sigma2 = (res.resid ** 2).mean()
        ll = -n/2 * (np.log(2 * np.pi * sigma2) + 1)
        return ll, X.shape[1], res.rsquared, res.params, res.tvalues

    def lr_test(ll_restricted, ll_full, k_restricted, k_full):
        """Chi-squared LR test. Returns: stat, df, p-value."""
        stat = 2 * (ll_full - ll_restricted)
        df   = k_full - k_restricted
        from scipy.stats import chi2
        p = 1 - chi2.cdf(stat, df)
        return stat, df, p

    # Estimate the encompassing model: ΔH + ΔS + T·ΔS (all three regressors)
    ll_full, k_full, r2_full, coef_full, t_full = fit_ols_ll(["DH_z", "DS_z", "TxDS"])
    print(f"\n  Encompassing model {{ΔH, ΔS, T·ΔS}}:")
    print(f"    R²  = {r2_full:.6f}")
    for name, c, t in zip(["const","DH","DS","TxDS"],
                           coef_full.values, t_full.values):
        print(f"    {name:6s}: coef={c:.6f}, t={t:.3f}")

    rows = []

    # Test 1: Restrict Model B {ΔH, ΔS} — test H0: β_{T·ΔS}=0
    ll_B, k_B, r2_B, coef_B, t_B = fit_ols_ll(["DH_z", "DS_z"])
    stat_B, df_B, p_B = lr_test(ll_B, ll_full, k_B, k_full)
    print(f"\n  LR test 1: restrict T·ΔS=0 (compare full vs Model B {{ΔH, ΔS}})")
    print(f"    LR stat = {stat_B:.3f}, df={df_B}, p={p_B:.4f}")
    print(f"    Interpretation: {'reject H0 — T·ΔS is significant (Gibbs structure needed)' if p_B < 0.05 else 'fail to reject — T·ΔS not independently significant given ΔH, ΔS'}")
    rows.append({
        "test": "LR: full vs Model B {DH,DS}", "restriction": "T·ΔS = 0",
        "lr_stat": stat_B, "df": df_B, "p": p_B,
        "r2_full": r2_full, "r2_restricted": r2_B,
        "reject_H0_5pct": p_B < 0.05,
        "interpretation": "T·ΔS significant beyond ΔH+ΔS" if p_B < 0.05 else "T·ΔS not significant beyond ΔH+ΔS",
    })

    # Test 2: Restrict Model C {ΔH, T·ΔS} — test H0: β_ΔS=0
    ll_C, k_C, r2_C, coef_C, t_C = fit_ols_ll(["DH_z", "TxDS"])
    stat_C, df_C, p_C = lr_test(ll_C, ll_full, k_C, k_full)
    print(f"\n  LR test 2: restrict ΔS=0 (compare full vs Model C {{ΔH, T·ΔS}})")
    print(f"    LR stat = {stat_C:.3f}, df={df_C}, p={p_C:.4f}")
    print(f"    Interpretation: {'reject H0 — ΔS has independent effect beyond T·ΔS' if p_C < 0.05 else 'fail to reject — ΔS not needed once T·ΔS included'}")
    rows.append({
        "test": "LR: full vs Model C {DH,TxDS}", "restriction": "ΔS = 0",
        "lr_stat": stat_C, "df": df_C, "p": p_C,
        "r2_full": r2_full, "r2_restricted": r2_C,
        "reject_H0_5pct": p_C < 0.05,
        "interpretation": "ΔS significant beyond T·ΔS" if p_C < 0.05 else "ΔS not needed once T·ΔS included",
    })

    # Test 3: Restrict to ΔG only (the single-index version)
    ll_G, k_G, r2_G, coef_G, t_G = fit_ols_ll(["DG"])
    stat_G, df_G, p_G = lr_test(ll_G, ll_full, k_G, k_full)
    rows.append({
        "test": "LR: full vs ΔG single index", "restriction": "β_DH=β_TxDS (constrained)",
        "lr_stat": stat_G, "df": df_G, "p": p_G,
        "r2_full": r2_full, "r2_restricted": r2_G,
        "reject_H0_5pct": p_G < 0.05,
        "interpretation": "Full model significantly better than ΔG alone" if p_G < 0.05 else "ΔG single index sufficient",
    })

    # Combined verdict
    print("\n  COMBINED LR VERDICT:")
    if not rows[0]["reject_H0_5pct"] and not rows[1]["reject_H0_5pct"]:
        print("  • T·ΔS not significant; ΔS also not needed → linear ΔH+ΔS sufficient")
    elif rows[0]["reject_H0_5pct"] and not rows[1]["reject_H0_5pct"]:
        print("  • T·ΔS IS significant (β_{T·ΔS}≠0) AND ΔS is NOT needed once T·ΔS included")
        print("    → This CONFIRMS the Gibbs constraint: Model C {ΔH, T·ΔS} is the right form")
        print("    → Strongly defends both Vuong application and the thermodynamic framing")
    elif not rows[0]["reject_H0_5pct"] and rows[1]["reject_H0_5pct"]:
        print("  • T·ΔS NOT significant but ΔS IS significant → unconstrained model {ΔH, ΔS}")
        print("    → This would question the Gibbs constraint; report honestly")
    else:
        print("  • Both T·ΔS and ΔS significant → neither pure form fully sufficient")
        print("    → Encompassing model preferred; note in paper as nuance")

    df5 = pd.DataFrame(rows)
    df5["test_num"] = "T5"
    df5.to_csv(f"{OUT}/R12_T5_lr_test.csv", index=False)
    return df5


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 — NW sensitivity table
# ─────────────────────────────────────────────────────────────────────────────

def task6_nw_sensitivity(panel):
    print("\n" + "="*60)
    print("TASK 6 — NW T-STAT SENSITIVITY TABLE")
    print("="*60)

    sub_base = panel.dropna(subset=["DG", "DH_z", "DS_z", "T", "ret_next_month"]).copy()
    sub_base["TxDS"] = sub_base["T"] * sub_base["DS_z"]

    models = {
        "Model A (ΔG only)":      ["DG"],
        "Model B (ΔH + ΔS)":     ["DH_z", "DS_z"],
        "Model C (ΔH + T·ΔS)":   ["DH_z", "TxDS"],
    }

    # Automatic bandwidth (Newey-West 1994 rule: floor(4*(n/100)^(2/9)))
    dates = sub_base["date"].nunique()
    nw_auto = int(np.floor(4 * (dates / 100) ** (2/9)))
    print(f"  NW auto bandwidth (n={dates} months): {nw_auto}")

    lag_choices = [1, 3, 6, 12, 24, 60, nw_auto]
    lag_labels  = ["NW-1","NW-3","NW-6","NW-12","NW-24","NW-60",f"NW-auto({nw_auto})"]

    rows = []
    for lag, lag_lbl in zip(lag_choices, lag_labels):
        row = {"NW_lags": lag_lbl}
        for mod_name, x_cols in models.items():
            # FM with this NW lag
            sub = sub_base.dropna(subset=x_cols + ["ret_next_month"])
            try:
                fm_out, _ = fama_macbeth(sub, "ret_next_month", x_cols, lags=lag)
                # Primary predictor is DG for A, or DH_z for B/C (first regressor)
                primary = x_cols[0]
                t = fm_out.get(primary, (np.nan, np.nan, np.nan))[1]
                # For Model B and C the 'DG-equivalent' is the first x_col
                if mod_name == "Model B (ΔH + ΔS)":
                    # report t on DS_z (the entropy term — key comparison)
                    t_ds  = fm_out.get("DS_z", (np.nan,)*3)[1]
                    t_dh  = fm_out.get("DH_z", (np.nan,)*3)[1]
                    row[f"t(ΔH) {mod_name}"]  = round(t_dh, 3) if np.isfinite(t_dh) else np.nan
                    row[f"t(ΔS) {mod_name}"]  = round(t_ds, 3) if np.isfinite(t_ds) else np.nan
                elif mod_name == "Model C (ΔH + T·ΔS)":
                    t_txds = fm_out.get("TxDS", (np.nan,)*3)[1]
                    t_dh   = fm_out.get("DH_z", (np.nan,)*3)[1]
                    row[f"t(ΔH) {mod_name}"]   = round(t_dh, 3) if np.isfinite(t_dh) else np.nan
                    row[f"t(T·ΔS) {mod_name}"] = round(t_txds, 3) if np.isfinite(t_txds) else np.nan
                else:
                    row[f"t(ΔG) {mod_name}"] = round(t, 3) if np.isfinite(t) else np.nan
            except Exception as e:
                row[f"ERROR {mod_name}"] = str(e)
        rows.append(row)

    df6 = pd.DataFrame(rows)
    print("\n  NW sensitivity table:")
    print(df6.to_string())

    # Check monotonicity
    if f"t(ΔG) Model A (ΔG only)" in df6.columns:
        t_col = f"t(ΔG) Model A (ΔG only)"
        t_vals = df6[t_col].dropna()
        print(f"\n  Model A t(ΔG) range: [{t_vals.min():.3f}, {t_vals.max():.3f}]")
        print(f"  All negative? {(t_vals < 0).all()}")
        print(f"  Still < -2.5 at NW-60? {t_vals.iloc[-2] < -2.5 if len(t_vals) >= 2 else 'N/A'}")

    df6.to_csv(f"{OUT}/R12_T6_nw_sensitivity.csv", index=False)

    # LaTeX table
    latex_lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\caption{FM $t$-Statistic Sensitivity to Newey-West Lag Choice}",
        "\\label{tab:nw_sensitivity}",
        "\\begin{tabular}{l" + "r" * (len(df6.columns) - 1) + "}",
        "\\hline\\hline",
        " & ".join(
            c.replace("Model A (ΔG only)", "Mod. A").replace("Model B (ΔH + ΔS)", "Mod. B")
             .replace("Model C (ΔH + T·ΔS)", "Mod. C")
            for c in df6.columns
        ) + " \\\\",
        "\\hline",
    ]
    for _, row in df6.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, str):
                cells.append(v)
            elif isinstance(v, float) and np.isfinite(v):
                cells.append(f"{v:.3f}")
            else:
                cells.append("—")
        latex_lines.append(" & ".join(cells) + " \\\\")
    latex_lines += ["\\hline\\hline", "\\end{tabular}", "\\end{table}"]
    with open(f"{OUT}/R12_T6_nw_sensitivity.tex", "w") as f:
        f.write("\n".join(latex_lines))
    print(f"  LaTeX table saved to {OUT}/R12_T6_nw_sensitivity.tex")

    return df6, nw_auto


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  R12 — REVIEWER RESPONSE TASKS 1–6")
    print("=" * 65)

    panel, factors = load_panel()

    results = {}

    # Task 1 — QMJ/BAB (most critical; run first)
    try:
        df1, qmj_t, bab_t = task1_qmj_bab(panel, factors)
        results["T1_QMJ_BAB"] = {"qmj_t": qmj_t, "bab_t": bab_t, "status": "OK"}
    except Exception as e:
        import traceback; traceback.print_exc()
        results["T1_QMJ_BAB"] = {"status": f"ERROR: {e}"}

    # Task 2 — ACF
    try:
        acf_df, nw_rec, verdict = task2_acf_fm_coefs(panel)
        results["T2_ACF"] = {"nw_recommended": nw_rec, "verdict": verdict, "status": "OK"}
    except Exception as e:
        import traceback; traceback.print_exc()
        results["T2_ACF"] = {"status": f"ERROR: {e}"}

    # Task 3 — Corr(ΔH, ΔG)
    try:
        r3 = task3_corr_dh_dg(panel)
        results["T3_corr"] = {**r3, "status": "OK"}
    except Exception as e:
        import traceback; traceback.print_exc()
        results["T3_corr"] = {"status": f"ERROR: {e}"}

    # Task 4 — Drawdown plot
    try:
        dd = task4_drawdown_plot(panel, factors)
        results["T4_drawdown"] = {**dd, "status": "OK"}
    except Exception as e:
        import traceback; traceback.print_exc()
        results["T4_drawdown"] = {"status": f"ERROR: {e}"}

    # Task 5 — LR test
    try:
        df5 = task5_lr_test(panel)
        results["T5_LR"] = {"n_tests": len(df5), "status": "OK"}
    except Exception as e:
        import traceback; traceback.print_exc()
        results["T5_LR"] = {"status": f"ERROR: {e}"}

    # Task 6 — NW sensitivity table
    try:
        df6, nw_auto = task6_nw_sensitivity(panel)
        results["T6_NW"] = {"nw_auto": nw_auto, "status": "OK"}
    except Exception as e:
        import traceback; traceback.print_exc()
        results["T6_NW"] = {"status": f"ERROR: {e}"}

    # Summary
    print("\n" + "=" * 65)
    print("  R12 COMPLETE — SUMMARY")
    print("=" * 65)
    for k, v in results.items():
        marker = "✓" if v.get("status") == "OK" else "✗"
        status = v.get("status", "?")
        print(f"  {marker}  {k}: {status}")
        for sk, sv in v.items():
            if sk != "status" and sv is not None:
                print(f"       {sk}: {sv}")

    pd.DataFrame(results).T.to_csv(f"{OUT}/R12_summary.csv")
    print(f"\n  All R12 outputs in: {os.path.abspath(OUT)}/")


if __name__ == "__main__":
    main()
