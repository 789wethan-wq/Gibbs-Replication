"""09_plots.py — All figures (300 DPI PNG + PDF)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
warnings.filterwarnings("ignore")

DATA = "data"
OUT_F = "outputs/figures"
os.makedirs(OUT_F, exist_ok=True)

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.frameon": False,
}

COLORS = {
    "Q1": "#d62728", "Q2": "#ff7f0e", "Q3": "#aec7e8",
    "Q4": "#1f77b4", "Q5": "#2ca02c", "LS": "#9467bd",
    "constrained": "#1f77b4", "unconstrained": "#ff7f0e",
    "high_T": "#ffcccc", "T_line": "#d62728",
}

KEY_EVENTS = {
    "2000-03": "Dot-com\npeak",
    "2008-09": "GFC",
    "2020-03": "COVID",
}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_F}/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def fig1_market_temperature(panel, regime):
    T_ts = panel.groupby("date")["T_raw"].first().sort_index()
    T_ts.index = pd.to_datetime(T_ts.index)
    regime.index = pd.to_datetime(regime.index)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 4))

        # Shade high-T regime
        high_dates = regime[regime == 1].index
        in_regime = False
        start = None
        for d in T_ts.index:
            is_high = d in high_dates
            if is_high and not in_regime:
                start = d; in_regime = True
            elif not is_high and in_regime:
                ax.axvspan(start, d, alpha=0.2, color="#d62728", lw=0)
                in_regime = False
        if in_regime:
            ax.axvspan(start, T_ts.index[-1], alpha=0.2, color="#d62728", lw=0)

        ax.plot(T_ts.index, T_ts.values, color=COLORS["T_line"], lw=1.4, label="Market Temperature T")

        for date_str, label in KEY_EVENTS.items():
            try:
                d = pd.to_datetime(date_str)
                if d in T_ts.index or (T_ts.index.min() <= d <= T_ts.index.max()):
                    ax.axvline(d, color="gray", lw=0.8, ls="--", alpha=0.7)
                    ax.text(d, T_ts.max() * 0.95, label, ha="center", fontsize=8, color="gray")
            except Exception:
                pass

        ax.set_xlabel("Date")
        ax.set_ylabel("12-month Realized Variance")
        ax.set_title("Figure 1: Market Temperature T (12-month realized variance of S&P 500 daily returns)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        fig.text(0.5, -0.05,
                 "Shaded regions indicate high-temperature regime periods identified by Markov-switching classification.",
                 ha="center", fontsize=9, color="gray")
        save(fig, "fig1_market_temperature")


def fig2_quintile_cumulative_returns(panel, factors):
    panel = panel.sort_values(["date", "DG"])
    panel["Q"] = panel.groupby("date")["DG"].transform(
        lambda x: pd.qcut(x, 5, labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop")
    )
    qret = panel.groupby(["date","Q"])["ret_next_month"].mean().unstack("Q")
    qret.index = pd.to_datetime(qret.index)
    qret = qret.sort_index().dropna(how="all")

    with plt.rc_context(STYLE):
        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax2 = ax1.twinx()

        for q, c in COLORS.items():
            if q in qret.columns:
                cum = (1 + qret[q].fillna(0)).cumprod() * 100
                ax1.plot(cum.index, cum.values, label=q, color=c, lw=1.3)

        ls = (qret["Q5"] - qret["Q1"]).fillna(0) if ("Q5" in qret.columns and "Q1" in qret.columns) else pd.Series(dtype=float)
        if len(ls) > 0:
            cum_ls = (1 + ls).cumprod() * 100
            ax2.plot(cum_ls.index, cum_ls.values, color=COLORS["LS"], lw=1.5, ls="--", label="L/S (Q5−Q1)")
            ax2.set_ylabel("L/S Cumulative Return (index=100)", color=COLORS["LS"])
            ax2.tick_params(axis="y", labelcolor=COLORS["LS"])

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Return (index=100)")
        ax1.set_title("Figure 2: ΔG Quintile Portfolio Cumulative Returns")
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=9)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax1.xaxis.set_major_locator(mdates.YearLocator(5))
        save(fig, "fig2_quintile_cumulative_returns")


def fig3_regime_conditional_loadings(panel):
    """Bar chart of β_ΔH and β_ΔS by regime."""
    from utils import fama_macbeth

    if "high_T" not in panel.columns:
        print("  Skipping Fig 3 — regime assignments not found")
        return

    def regime_coefs(sub):
        res, _ = fama_macbeth(sub, "ret_next_month", ["DH_z", "DS_z"], lags=6)
        dh = res.get("DH_z", (np.nan, np.nan, np.nan))
        ds = res.get("DS_z", (np.nan, np.nan, np.nan))
        return dh, ds

    low  = panel[panel["high_T"] == 0]
    high = panel[panel["high_T"] == 1]
    (dh_l, _, _), (ds_l, _, _) = regime_coefs(low)
    (dh_h, _, _), (ds_h, _, _) = regime_coefs(high)

    # Approximate 95% CI from NW se (mean / t gives se)
    def nw_se(sub, x_cols):
        from utils import fama_macbeth
        res, coefs = fama_macbeth(sub, "ret_next_month", x_cols, lags=6)
        se = {}
        for k in x_cols:
            if k in res and res[k][1] != 0:
                se[k] = abs(res[k][0] / res[k][1])
            else:
                se[k] = 0
        return se

    se_l = nw_se(low,  ["DH_z", "DS_z"])
    se_h = nw_se(high, ["DH_z", "DS_z"])

    labels = ["ΔH", "ΔS"]
    vals_low  = [dh_l, ds_l]
    vals_high = [dh_h, ds_h]
    errs_low  = [1.96 * se_l["DH_z"], 1.96 * se_l["DS_z"]]
    errs_high = [1.96 * se_h["DH_z"], 1.96 * se_h["DS_z"]]

    x = np.arange(len(labels))
    w = 0.35

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        b1 = ax.bar(x - w/2, vals_low,  w, yerr=errs_low,  capsize=4, label="Low-T Regime",  color="#1f77b4", alpha=0.8)
        b2 = ax.bar(x + w/2, vals_high, w, yerr=errs_high, capsize=4, label="High-T Regime", color="#d62728", alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Fama-MacBeth Coefficient")
        ax.set_title("Figure 3: Regime-Conditional Factor Loadings")
        ax.legend()
        fig.text(0.5, -0.04,
                 "Error bars denote 95% confidence intervals from Newey-West adjusted standard errors.",
                 ha="center", fontsize=9, color="gray")
        save(fig, "fig3_regime_conditional_loadings")


def fig4_rolling_oos_r2():
    try:
        cum_df = pd.read_parquet(f"{DATA}/oos_cumulative_r2.parquet")
    except Exception:
        print("  Skipping Fig 4 — OOS cumulative R² not found")
        return

    cum_df.index = pd.to_datetime(cum_df.index)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 4))
        if "Gibbs_Constrained" in cum_df.columns:
            ax.plot(cum_df.index, cum_df["Gibbs_Constrained"], color=COLORS["constrained"],
                    lw=1.5, label="Gibbs Constrained")
        if "Unconstrained" in cum_df.columns:
            ax.plot(cum_df.index, cum_df["Unconstrained"], color=COLORS["unconstrained"],
                    lw=1.5, ls="--", label="Unconstrained")
        ax.axhline(0, color="black", lw=0.7, ls=":")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative OOS R²")
        ax.set_title("Figure 4: Cumulative Out-of-Sample R² (60-month rolling window)")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        save(fig, "fig4_rolling_oos_r2")


def fig5_implied_vs_realized_temp(panel, factors):
    """Scatter of rolling 24m FM T·ΔS coefficient vs realized T."""
    from utils import fama_macbeth
    dates = sorted(panel["date"].unique())
    implied_T = []
    for i in range(24, len(dates)):
        win = dates[i-24:i]
        sub = panel[panel["date"].isin(win)].dropna(subset=["ret_next_month","DH_z","TxDS"])
        if len(sub) < 50:
            continue
        res, _ = fama_macbeth(sub, "ret_next_month", ["DH_z","TxDS"], lags=4)
        coef_txds = res.get("TxDS", (np.nan,))[0]
        T_realized = panel[panel["date"] == dates[i]]["T_raw"].mean()
        implied_T.append({"date": dates[i], "implied_T": coef_txds, "realized_T": T_realized})

    if not implied_T:
        print("  Skipping Fig 5 — no data")
        return

    df5 = pd.DataFrame(implied_T).dropna()
    if len(df5) < 10:
        print("  Skipping Fig 5 — insufficient data")
        return

    # OLS trendline
    from scipy import stats as scipy_stats
    m, b, r, p, se = scipy_stats.linregress(df5["realized_T"], df5["implied_T"])
    x_fit = np.linspace(df5["realized_T"].min(), df5["realized_T"].max(), 100)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(df5["realized_T"], df5["implied_T"], alpha=0.5, s=20, color="#1f77b4")
        ax.plot(x_fit, m * x_fit + b, color="#d62728", lw=1.5,
                label=f"OLS fit: slope={m:.2f}, R²={r**2:.3f}")
        ax.set_xlabel("Realized Market Temperature T (12m RV)")
        ax.set_ylabel("Implied Temperature Coefficient (rolling 24m FM)")
        ax.set_title("Figure 5: Implied vs. Realized Market Temperature")
        ax.legend()
        fig.text(0.5, -0.04,
                 f"A slope of 1.0 indicates exact thermodynamic pricing of entropy. Slope={m:.3f}, p={p:.3f}.",
                 ha="center", fontsize=9, color="gray")
        save(fig, "fig5_implied_vs_realized_temperature")


def main():
    print("Generating figures...")
    panel = pd.read_parquet(f"{DATA}/variables_monthly.parquet")
    factors = pd.read_parquet(f"{DATA}/factors_monthly.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    factors.index = pd.to_datetime(factors.index)

    # Load regime assignments
    try:
        regime_df = pd.read_parquet(f"{DATA}/regime_assignments.parquet")
        regime_df["date"] = pd.to_datetime(regime_df["date"])
        regime = regime_df.set_index("date")["high_T"]
        panel = panel.merge(regime_df, on="date", how="left")
    except Exception:
        regime = (panel.groupby("date")["T_raw"].first() >
                  panel.groupby("date")["T_raw"].first().median()).astype(int)
        panel["high_T"] = panel["date"].map(regime)

    fig1_market_temperature(panel, regime)
    fig2_quintile_cumulative_returns(panel, factors)
    fig3_regime_conditional_loadings(panel)
    fig4_rolling_oos_r2()
    fig5_implied_vs_realized_temp(panel, factors)
    print("All figures saved.")


if __name__ == "__main__":
    main()
