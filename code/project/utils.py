"""utils.py — Shared functions for the Gibbs free energy equity study.

Portfolio-level implementation (25 FF portfolios) per the brief's sanctioned
fully-executable path. All time-series t-statistics use Newey-West.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


def cross_sectional_zscore(df, date_col, val_col, out_col=None):
    """Z-score `val_col` within each `date_col` group."""
    out_col = out_col or (val_col + "_z")
    g = df.groupby(date_col)[val_col]
    df[out_col] = (df[val_col] - g.transform("mean")) / g.transform("std")
    return df


def winsorize(s, lower=0.01, upper=0.99):
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def newey_west_mean_tstat(x, lags=6):
    """Newey-West t-stat for the mean of a time series (FM coefficient series)."""
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
    se = np.sqrt(var / n)
    t = mean / se if se > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(t))) if np.isfinite(t) else np.nan
    return mean, t, p


def ff_alpha(port_excess, factor_df, factors):
    """Regress portfolio EXCESS returns on factors; return alpha (%), NW t-stat, betas."""
    df = pd.concat([port_excess.rename("y"), factor_df[factors]], axis=1).dropna()
    if len(df) < 12:
        return np.nan, np.nan, {}
    X = sm.add_constant(df[factors])
    res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    betas = {f: res.params[f] for f in factors}
    return res.params["const"], res.tvalues["const"], betas


def fama_macbeth(panel, ret_col, x_cols, lags=6):
    """Cross-sectional OLS each date, then NW t-stats on coefficient series.

    Returns dict: coef -> (mean, t, p), plus the coefficient time series.
    """
    dates = sorted(panel["date"].unique())
    rows = []
    for d in dates:
        sub = panel[panel["date"] == d]
        sub = sub.dropna(subset=[ret_col] + x_cols)
        if len(sub) < len(x_cols) + 2:
            continue
        X = sm.add_constant(sub[x_cols])
        try:
            res = sm.OLS(sub[ret_col], X).fit()
        except Exception:
            continue
        rows.append(pd.Series(res.params, name=d))
    coefs = pd.DataFrame(rows)
    out = {}
    for c in coefs.columns:
        out[c] = newey_west_mean_tstat(coefs[c].values, lags=lags)
    return out, coefs


def vuong_test(ll1_i, ll2_i):
    """Vuong (1989) non-nested test. Positive Z favors model 1 (constrained).

    ll1_i, ll2_i: per-observation log-likelihoods.
    """
    lr = np.asarray(ll1_i) - np.asarray(ll2_i)
    n = len(lr)
    m = lr.mean()
    s = lr.std(ddof=1)
    z = np.sqrt(n) * m / s if s > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return z, p


def normal_loglik_perobs(y, yhat, sigma):
    """Per-observation Gaussian log-likelihood."""
    return -0.5 * np.log(2 * np.pi * sigma**2) - (y - yhat) ** 2 / (2 * sigma**2)


def dm_test(e1, e2, h=1):
    """Diebold-Mariano with Harvey-Leybourne-Newbold (1997) small-sample correction.

    e1, e2: forecast errors of the two models. Loss = squared error.
    H0: equal accuracy. Positive DM => model 1 worse (larger loss).
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    mask = ~(np.isnan(e1) | np.isnan(e2))
    e1, e2 = e1[mask], e2[mask]
    d = e1**2 - e2**2
    n = len(d)
    if n < 8:
        return np.nan, np.nan
    dbar = d.mean()
    gamma0 = np.sum((d - dbar) ** 2) / n
    var = gamma0
    for k in range(1, h):
        cov = np.sum((d[k:] - dbar) * (d[:-k] - dbar)) / n
        var += 2 * cov
    dm = dbar / np.sqrt(var / n) if var > 0 else np.nan
    # HLN small-sample correction
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * corr
    from scipy.stats import t as tdist
    p = 2 * (1 - tdist.cdf(abs(dm_hln), df=n - 1)) if np.isfinite(dm_hln) else np.nan
    return dm_hln, p


def stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""
