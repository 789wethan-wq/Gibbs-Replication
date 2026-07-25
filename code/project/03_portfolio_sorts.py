"""03_portfolio_sorts.py — S1 Table: Price-based composite portfolio sort (superseded ΔG, Section 4.1)."""
import numpy as np
import pandas as pd
import os
from utils import ff_alpha, newey_west_mean_tstat, stars

DATA = "data"
OUT_T = "outputs/tables"
OUT_I = "outputs/interpretations"
os.makedirs(OUT_T, exist_ok=True)
os.makedirs(OUT_I, exist_ok=True)


def quintile_label(x):
    try:
        return pd.qcut(x, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    except Exception:
        return pd.Series(["Q3"] * len(x), index=x.index)


def main():
    panel = pd.read_parquet(f"{DATA}/variables_monthly.parquet")
    factors = pd.read_parquet(f"{DATA}/factors_monthly.parquet")

    # Assign quintile each month based on DG
    panel = panel.sort_values(["date", "DG"])
    panel["Q"] = panel.groupby("date")["DG"].transform(quintile_label)
    panel = panel.dropna(subset=["Q", "ret_next_month"])

    # Build equal-weighted quintile return series (average over portfolios in each quintile each month)
    qret = panel.groupby(["date", "Q"])["ret_next_month"].mean().unstack("Q")
    qret.index = pd.to_datetime(qret.index)
    qret = qret.sort_index()

    # Align with factor data
    factors.index = pd.to_datetime(factors.index)
    common = qret.index.intersection(factors.index)
    qret = qret.loc[common]
    fac = factors.loc[common]

    rf = fac["RF"]
    ff5_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]

    rows = {}
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        r = qret[q].dropna()
        r_excess = r - rf.reindex(r.index)
        ann_ret = (1 + r.mean()) ** 12 - 1
        ann_std = r.std() * np.sqrt(12)
        sharpe = (r_excess.mean() / r_excess.std()) * np.sqrt(12)
        alpha, t_alpha, _ = ff_alpha(r_excess, fac, ff5_cols)
        rows[q] = {
            "Avg Monthly Ret (%)": r.mean() * 100,
            "Ann Ret (%)": ann_ret * 100,
            "Ann Std (%)": ann_std * 100,
            "Sharpe": sharpe,
            "FF5+UMD Alpha (%)": alpha * 100 if np.isfinite(alpha) else np.nan,
            "Alpha t-stat": t_alpha,
        }

    # Long-short Q5 - Q1
    ls = qret["Q5"] - qret["Q1"]
    ls_excess = ls - rf.reindex(ls.index)
    ann_ls = (1 + ls.mean()) ** 12 - 1
    ann_ls_std = ls.std() * np.sqrt(12)
    ls_sharpe = (ls_excess.mean() / ls_excess.std()) * np.sqrt(12)
    ls_alpha, ls_t, _ = ff_alpha(ls_excess, fac, ff5_cols)
    _, ls_tstat, ls_p = newey_west_mean_tstat(ls.values)
    cummax = (1 + ls).cumprod().cummax()
    drawdown = ((1 + ls).cumprod() / cummax - 1).min()
    win_rate = (ls > 0).mean() * 100

    rows["L/S (Q5-Q1)"] = {
        "Avg Monthly Ret (%)": ls.mean() * 100,
        "Ann Ret (%)": ann_ls * 100,
        "Ann Std (%)": ann_ls_std * 100,
        "Sharpe": ls_sharpe,
        "FF5+UMD Alpha (%)": ls_alpha * 100 if np.isfinite(ls_alpha) else np.nan,
        "Alpha t-stat": ls_t,
    }

    table1 = pd.DataFrame(rows).T.round(3)

    # significance stars on alpha t-stats
    from scipy.stats import t as tdist
    table1["Alpha t-stat"] = table1["Alpha t-stat"].astype(object)
    for idx in table1.index:
        tv = table1.loc[idx, "Alpha t-stat"]
        try:
            tv = float(tv)
        except (ValueError, TypeError):
            continue
        if np.isfinite(tv):
            p = 2 * (1 - tdist.cdf(abs(tv), df=len(common) - 7))
            table1.loc[idx, "Alpha t-stat"] = f"{tv:.2f}{stars(p)}"

    table1.to_csv(f"{OUT_T}/table1_portfolio_sorts.csv")

    with open(f"{OUT_T}/table1_portfolio_sorts.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{S1 Table. Price-based composite portfolio sort. Quintile portfolio returns and FF5+UMD alphas for the superseded price-based $\\Delta G$ construction discussed in Section 4.1.}\n")
        f.write("\\label{tab:sorts}\n")
        f.write(table1.to_latex(escape=False))
        f.write("\\end{table}\n")

    print("\n=== TABLE 1: QUINTILE PORTFOLIO SORTS ===")
    print(table1.to_string())
    print(f"\nL/S NW t-stat: {ls_tstat:.2f}  p={ls_p:.3f}")
    print(f"L/S Max Drawdown: {drawdown*100:.1f}%   Win Rate: {win_rate:.1f}%")
    hurdle = "PASSES" if abs(ls_tstat) >= 3.0 else "FAILS"
    print(f"Harvey et al. (2016) t>3.0 hurdle: {hurdle} (t={ls_tstat:.2f})")

    # Interpretation
    mono = "monotone" if all(
        float(str(table1.loc[q, "Avg Monthly Ret (%)"]).replace("*","")) <
        float(str(table1.loc[q2, "Avg Monthly Ret (%)"]).replace("*",""))
        for q, q2 in zip(["Q1","Q2","Q3","Q4"], ["Q2","Q3","Q4","Q5"])
    ) else "not strictly monotone"

    interp = f"""The S1 Table presents returns to quintile portfolios sorted monthly on the price-based Gibbs score ΔG across the S&P 500 price panel (Section 4.1, superseded construction).

The quintile return pattern is {mono} from Q1 (lowest ΔG, most disordered, least stable) to Q5 (highest ΔG, most thermodynamically favorable). The long-short portfolio (Q5 minus Q1) earns an average monthly return of {ls.mean()*100:.2f}%, corresponding to an annualized return of {ann_ls*100:.1f}% with an annualized Sharpe ratio of {ls_sharpe:.2f}. The Sharpe ratio compares {"favorably" if ls_sharpe > 0.4 else "modestly"} to the historically documented Sharpe ratios for the HML factor of approximately 0.3–0.4 and momentum of approximately 0.5–0.6 (Fama and French, 2015).

The FF5+UMD alpha for the long-short portfolio is {ls_alpha*100:.2f}% per month (t-statistic = {ls_t:.2f}), indicating {"meaningful" if abs(ls_t) > 2.0 else "limited"} abnormal returns after controlling for the five Fama-French factors and momentum. With respect to Hypothesis H1, the Newey-West t-statistic on the long-short return series is {ls_tstat:.2f}, which {"exceeds" if abs(ls_tstat) >= 3.0 else "falls below"} the Harvey, Liu, and Zhu (2016) multiple-testing hurdle of 3.0. {"This provides statistical support for H1." if abs(ls_tstat) >= 3.0 else "As such, H1 is not supported at the 3.0 hurdle, and the result should be interpreted cautiously."}

The maximum drawdown of the long-short strategy is {drawdown*100:.1f}%, with a win rate of {win_rate:.1f}% of months. The strategy's resilience — or lack thereof — during stress periods reflects the regime-dependence explored in Table 4.
"""
    with open(f"{OUT_I}/table1_interpretation.txt", "w") as f:
        f.write(interp)
    print("\nInterpretation saved.")


if __name__ == "__main__":
    main()
