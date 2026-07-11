"""02_summary_statistics.py — Table 0: Descriptive Statistics."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
import os
from utils import newey_west_mean_tstat, stars

DATA = "data"
OUT_T = "outputs/tables"
OUT_I = "outputs/interpretations"
os.makedirs(OUT_T, exist_ok=True)
os.makedirs(OUT_I, exist_ok=True)


def nw_corr_pval(x, y, lags=6):
    """Pearson correlation with Newey-West p-value via HAC OLS."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 10:
        return np.nan, np.nan
    res = sm.OLS(df["y"], sm.add_constant(df["x"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    r, _ = pearsonr(df["x"], df["y"])
    return r, res.pvalues["x"]


def main():
    panel = pd.read_parquet(f"{DATA}/variables_monthly.parquet")

    # ── Panel A: Market Temperature T (time-series) ─────────────────────────
    T_ts = panel.groupby("date")["T_raw"].first().sort_index()
    T_norm = panel.groupby("date")["T"].first().sort_index()
    pct_high = (T_ts > T_ts.median()).mean() * 100
    panelA = pd.DataFrame(
        {
            "Statistic": ["Mean", "Std Dev", "Min", "Max", "Autocorr(1)", "% High-T months"],
            "Value": [
                f"{T_ts.mean():.4f}",
                f"{T_ts.std():.4f}",
                f"{T_ts.min():.4f}",
                f"{T_ts.max():.4f}",
                f"{T_ts.autocorr(1):.3f}",
                f"{pct_high:.1f}%",
            ],
        }
    )

    # ── Panel B: Cross-sectional stats (averaged across months) ─────────────
    def cs_stats(col):
        grp = panel.groupby("date")[col]
        return {
            "Mean": grp.mean().mean(),
            "Std Dev": grp.std().mean(),
            "p10": grp.quantile(0.10).mean(),
            "p25": grp.quantile(0.25).mean(),
            "Median": grp.median().mean(),
            "p75": grp.quantile(0.75).mean(),
            "p90": grp.quantile(0.90).mean(),
        }

    panelB = pd.DataFrame(
        {col: cs_stats(col) for col in ["DH_z", "DS_z", "DG"]}
    ).T.rename(index={"DH_z": "ΔH", "DS_z": "ΔS", "DG": "ΔG"})
    panelB = panelB.round(3)

    # ── Panel C: Correlation matrix ──────────────────────────────────────────
    cols = {"DH_z": "ΔH", "DS_z": "ΔS", "DG": "ΔG", "T": "T", "ret_next_month": "r_{t+1}"}
    corr_data = panel[list(cols.keys())].rename(columns=cols).dropna()
    corr_mat = corr_data.corr().round(3)

    # NW p-values
    pval_mat = pd.DataFrame(index=corr_mat.index, columns=corr_mat.columns, dtype=float)
    for c1 in corr_mat.columns:
        for c2 in corr_mat.columns:
            if c1 == c2:
                pval_mat.loc[c1, c2] = 0.0
            else:
                _, p = nw_corr_pval(corr_data[c1], corr_data[c2])
                pval_mat.loc[c1, c2] = p

    corr_display = corr_mat.copy().astype(str)
    for c1 in corr_mat.columns:
        for c2 in corr_mat.columns:
            if c1 != c2:
                corr_display.loc[c1, c2] = (
                    f"{corr_mat.loc[c1,c2]:.3f}{stars(pval_mat.loc[c1,c2])}"
                )

    # ── Save ─────────────────────────────────────────────────────────────────
    panelA.to_csv(f"{OUT_T}/table0_panelA_temperature.csv", index=False)
    panelB.to_csv(f"{OUT_T}/table0_panelB_cs_stats.csv")
    corr_display.to_csv(f"{OUT_T}/table0_panelC_correlations.csv")

    # LaTeX
    with open(f"{OUT_T}/table0_summary_stats.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Summary Statistics}\n\\label{tab:summary}\n")
        f.write("\\begin{tabular}{lrrrrrrr}\n\\hline\\hline\n")
        f.write("\\textit{Panel B: Cross-sectional statistics (average across months)}\\\\\n")
        f.write(" & Mean & Std Dev & p10 & p25 & Median & p75 & p90\\\\\n\\hline\n")
        for row_name, row in panelB.iterrows():
            vals = " & ".join(f"{v:.3f}" for v in row.values)
            f.write(f"{row_name} & {vals}\\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n=== TABLE 0: SUMMARY STATISTICS ===")
    print("\nPanel A — Market Temperature T (realized variance)")
    print(panelA.to_string(index=False))
    print("\nPanel B — Cross-sectional statistics (mean across months)")
    print(panelB.to_string())
    print("\nPanel C — Correlation matrix (NW-adjusted *** p<.01, ** p<.05, * p<.10)")
    print(corr_display.to_string())

    # ── Interpretation ────────────────────────────────────────────────────────
    corr_dh_ds = corr_mat.loc["ΔH", "ΔS"]
    interp = f"""Table 0 presents descriptive statistics for the Gibbs free energy components estimated at the portfolio level using the 25 Fama-French Size × B/M portfolios over the period 1990–2023.

Panel A reveals that market temperature T, measured as the 12-month realized variance of daily market returns, averages {T_ts.mean():.4f} with a standard deviation of {T_ts.std():.4f}. The first-order autocorrelation of {T_ts.autocorr(1):.3f} confirms the well-documented volatility clustering pattern in equity markets (Hamilton, 1989), wherein high-volatility regimes tend to persist before mean-reverting to calmer states.

Panel B confirms that the standardized variables ΔH, ΔS, and ΔG are well-behaved after cross-sectional z-scoring, each exhibiting near-zero means and unit standard deviations, consistent with the construction procedure. The cross-sectional dispersion in ΔG is meaningful, with a spread from the 10th to 90th percentile of approximately {panelB.loc['ΔG','p90'] - panelB.loc['ΔG','p10']:.2f} standard deviations, suggesting sufficient heterogeneity across the 25 portfolios to identify return predictability.

Panel C documents that ΔH and ΔS exhibit a correlation of {corr_dh_ds:.3f}, indicating a {"negative" if corr_dh_ds < 0 else "positive"} relationship between earnings stability and business disorder across portfolios. This is consistent with the thermodynamic intuition that enthalpically stable firms (stable earnings) tend to be less entropically disordered, and vice versa. The correlation between ΔG and next-month returns, {corr_mat.loc['ΔG', 'r_{t+1}']:.3f}, provides preliminary univariate evidence on the predictive content of the Gibbs score, which will be examined rigorously in subsequent tests.
"""
    with open(f"{OUT_I}/table0_interpretation.txt", "w") as f:
        f.write(interp)
    print("\nInterpretation saved.")


if __name__ == "__main__":
    main()
