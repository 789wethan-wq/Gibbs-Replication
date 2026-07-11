"""robustness_utils.py — Shared functions for the Gibbs robustness battery."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, t as tdist
import warnings
warnings.filterwarnings("ignore")

DATA   = "../data"
OUT    = "outputs"
NW_LAGS = 6

# ── Baseline metrics ─────────────────────────────────────────────────────────
BASELINE = {
    "fm_t_DG":      -3.98,
    "ls_t":         -3.70,
    "ls_sign":      "negative",
    "vuong_z":      +2.71,
    "vuong_p":       0.007,
    "delta_aic":    +94.3,
    "dm_p":          0.833,
    "beta_ds_ratio": 2.09,
}


# ── Core statistical functions ────────────────────────────────────────────────

def winsorize_cs(series, pct=0.01):
    lo = series.quantile(pct)
    hi = series.quantile(1 - pct)
    return series.clip(lo, hi)


def zscore_cs(series):
    m, s = series.mean(), series.std()
    if s == 0:
        return series * 0
    return (series - m) / s


def newey_west_mean_tstat(x, lags=NW_LAGS):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    mean = x.mean()
    e = x - mean
    gamma0 = (e @ e) / n
    var = gamma0
    for L in range(1, lags + 1):
        if L >= n:
            break
        w = 1.0 - L / (lags + 1.0)
        cov = (e[L:] @ e[:-L]) / n
        var += 2.0 * w * cov
    se = np.sqrt(max(var, 0) / n)
    t = mean / se if se > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(t))) if np.isfinite(t) else np.nan
    return mean, t, p


def fama_macbeth(panel, ret_col, x_cols, lags=NW_LAGS):
    """Fama-MacBeth cross-sectional OLS. Returns dict of (mean, t, p) per coef."""
    dates = sorted(panel["date"].unique())
    rows = []
    for d in dates:
        sub = panel[panel["date"] == d].dropna(subset=[ret_col] + x_cols)
        if len(sub) < len(x_cols) + 5:
            continue
        X = sm.add_constant(sub[x_cols], has_constant="add")
        try:
            res = sm.OLS(sub[ret_col], X).fit()
            rows.append(res.params)
        except Exception:
            continue
    coefs = pd.DataFrame(rows)
    out = {}
    for c in coefs.columns:
        out[c] = newey_west_mean_tstat(coefs[c].dropna().values, lags=lags)
    return out, coefs


def ff5_umd_alpha(port_excess, factor_df, ff_cols=None, lags=NW_LAGS):
    """Regress portfolio excess returns on factors. Returns alpha, t, p, R²."""
    if ff_cols is None:
        ff_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    df = pd.concat([port_excess.rename("y"), factor_df[ff_cols]], axis=1).dropna()
    if len(df) < 24:
        return np.nan, np.nan, np.nan, np.nan
    X = sm.add_constant(df[ff_cols])
    res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return res.params["const"], res.tvalues["const"], res.pvalues["const"], res.rsquared


def ls_portfolio_stats(long_ret, short_ret, rf, lags=NW_LAGS):
    """L/S stats from long and short leg monthly return series."""
    ls = long_ret.reindex(short_ret.index) - short_ret
    ls = ls.dropna()
    rf_a = rf.reindex(ls.index).fillna(0)
    ls_ex = ls - rf_a
    mean_m, t, p = newey_west_mean_tstat(ls.values, lags=lags)
    ann_ret = (1 + ls.mean()) ** 12 - 1 if ls.mean() > -1 else np.nan
    ann_std = ls.std() * np.sqrt(12)
    sharpe  = ls_ex.mean() / ls_ex.std() * np.sqrt(12) if ls_ex.std() > 0 else np.nan
    cumret  = (1 + ls).cumprod()
    drawdown = (cumret / cumret.cummax() - 1).min()
    win_rate = (ls > 0).mean()
    return dict(mean_m=mean_m, ann_ret=ann_ret, ann_std=ann_std,
                sharpe=sharpe, max_dd=drawdown, win_rate=win_rate,
                nw_t=t, nw_p=p)


def vuong_test(y, X_constrained, X_unconstrained):
    """
    Vuong (1989) non-nested model comparison.
    Positive Z => constrained model closer to true DGP.
    Returns: z, p, delta_aic, delta_bic, delta_r2
    """
    y = np.asarray(y, float)
    mask = ~np.isnan(y)

    def fit_ols(X):
        Xm = np.asarray(X)[mask]
        ym = y[mask]
        Xc = np.column_stack([np.ones(len(ym)), Xm])
        beta, *_ = np.linalg.lstsq(Xc, ym, rcond=None)
        yhat = Xc @ beta
        resid = ym - yhat
        sigma2 = (resid ** 2).mean()
        ll_i = -0.5 * (np.log(2 * np.pi * sigma2) + resid ** 2 / sigma2)
        k = Xc.shape[1]
        n = len(ym)
        aic = -2 * ll_i.sum() + 2 * k
        bic = -2 * ll_i.sum() + k * np.log(n)
        ss_res = (resid ** 2).sum()
        ss_tot = ((ym - ym.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return ll_i, aic, bic, r2

    ll_c, aic_c, bic_c, r2_c = fit_ols(X_constrained)
    ll_u, aic_u, bic_u, r2_u = fit_ols(X_unconstrained)

    lr = ll_c - ll_u
    n = len(lr)
    m = lr.mean()
    s = lr.std(ddof=1)
    z = np.sqrt(n) * m / s if s > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan

    return z, p, aic_u - aic_c, bic_u - bic_c, r2_c - r2_u


def dm_test_hlc(e1, e2, h=1):
    """Diebold-Mariano with Harvey-Leybourne-Newbold correction."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    mask = ~(np.isnan(e1) | np.isnan(e2))
    e1, e2 = e1[mask], e2[mask]
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    if n < 8:
        return np.nan, np.nan
    dbar = d.mean()
    gamma0 = ((d - dbar) ** 2).mean()
    var = gamma0
    for k in range(1, h):
        cov = ((d[k:] - dbar) * (d[:-k] - dbar)).mean()
        var += 2 * cov
    dm = dbar / np.sqrt(var / n) if var > 0 else np.nan
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * corr
    p = 2 * (1 - tdist.cdf(abs(dm_hln), df=n - 1)) if np.isfinite(dm_hln) else np.nan
    return dm_hln, p


def pass_fail(value, threshold, direction="above"):
    if not np.isfinite(float(value) if value is not None else float("nan")):
        return "NA"
    v = float(value)
    if direction == "above":
        if v >= threshold:
            return "PASS"
        elif v >= threshold * 0.75:
            return "MARGINAL"
        return "FAIL"
    else:  # below
        if v <= threshold:
            return "PASS"
        elif v <= threshold * 1.25:
            return "MARGINAL"
        return "FAIL"


def stars(p):
    if not np.isfinite(float(p) if p is not None else float("nan")):
        return ""
    p = float(p)
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


def build_dg_panel(panel_in, dh_col="DH_z", ds_col="DS_z", t_col="T"):
    """Rebuild ΔG and TxDS from given column names."""
    p = panel_in.copy()
    g = p.groupby("date")["DG_raw"] if "DG_raw" in p.columns else None
    p["DG_raw2"] = p[dh_col] - p[t_col] * p[ds_col]
    g2 = p.groupby("date")["DG_raw2"]
    p["DG_new"] = (p["DG_raw2"] - g2.transform("mean")) / g2.transform("std")
    p["TxDS_new"] = p[t_col] * p[ds_col]
    return p


def rolling_ff3_resid_std_vec(ret_monthly, ff3_monthly, window=36):
    """Vectorised rolling FF3 residual std for one stock. Returns pd.Series."""
    idx = ret_monthly.index
    y   = ret_monthly.values
    X3  = ff3_monthly.reindex(idx).values
    n   = len(y)
    out = np.full(n, np.nan)
    for i in range(window, n + 1):
        ys = y[i - window:i]
        Xs = X3[i - window:i]
        if np.isnan(ys).any() or np.isnan(Xs).any():
            continue
        Xs1 = np.column_stack([np.ones(window), Xs])
        try:
            beta, *_ = np.linalg.lstsq(Xs1, ys, rcond=None)
        except Exception:
            continue
        out[i - 1] = (ys - Xs1 @ beta).std(ddof=1)
    return pd.Series(out, index=idx)


def save_results(df, name, interp_text=""):
    import os
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/{name}_results.csv", index=False)
    if interp_text:
        with open(f"{OUT}/{name}_interpretation.txt", "w") as f:
            f.write(interp_text)
    print(f"  Saved {name}")


def load_panel():
    panel   = pd.read_parquet(f"{DATA}/variables_monthly.parquet")
    factors = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
    panel["date"]   = pd.to_datetime(panel["date"])
    factors.index   = pd.to_datetime(factors.index)
    return panel, factors


def quintile_sort_ls(panel, dg_col, ret_col, factors, n_q=5):
    """Sort into quintiles on dg_col each month; return L/S (Q5-Q1) monthly return series."""
    panel = panel.copy()
    panel["_q"] = panel.groupby("date")[dg_col].transform(
        lambda x: pd.qcut(x, n_q, labels=False, duplicates="drop") if x.nunique() >= n_q else np.nan
    )
    panel = panel.dropna(subset=["_q", ret_col])
    qret = panel.groupby(["date", "_q"])[ret_col].mean().unstack("_q")
    qret.index = pd.to_datetime(qret.index)
    q_low  = qret.get(n_q - 1, pd.Series(dtype=float))  # Q5 = label n_q-1
    q_high = qret.get(0, pd.Series(dtype=float))         # Q1 = label 0
    ls = q_low - q_high   # Q5 − Q1
    return ls, qret
