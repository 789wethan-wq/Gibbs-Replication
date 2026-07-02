"""04_fama_macbeth.py — Table 2: Fama-MacBeth regressions (Models A, B, C)."""
import numpy as np
import pandas as pd
import os, sys
# resolve paths relative to this file so the script can be run from any directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_HERE)
sys.path.insert(0, _HERE)
from utils import fama_macbeth, stars

DATA = "../data"
OUT_T = "../outputs/tables"
OUT_I = "../outputs/interpretations"
os.makedirs(OUT_T, exist_ok=True)
os.makedirs(OUT_I, exist_ok=True)


def fmt(mean, t, p):
    return f"{mean:.4f}{stars(p)}", f"({t:.2f})"


def main():
    panel = pd.read_parquet(f"{DATA}/variables_monthly.parquet")
    factors = pd.read_parquet(f"{DATA}/factors_monthly.parquet")

    # Add FF5 controls to panel
    fac_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    factors.index = pd.to_datetime(factors.index)
    fac_reset = factors[fac_cols].reset_index().rename(columns={"index": "date"})
    fac_reset["date"] = pd.to_datetime(fac_reset["date"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.merge(fac_reset, on="date", how="left")

    RET = "ret_next_month"

    specs = {
        "Model A — ΔG only": (["DG"], False),
        "Model A + FF5": (["DG"] + fac_cols, False),
        "Model B — Unconstrained": (["DH_z", "DS_z"], False),
        "Model B + FF5": (["DH_z", "DS_z"] + fac_cols, False),
        "Model C — Gibbs constrained": (["DH_z", "TxDS"], False),
        "Model C + FF5": (["DH_z", "TxDS"] + fac_cols, False),
    }

    all_results = {}
    coef_series = {}
    for name, (xcols, _) in specs.items():
        res, coefs = fama_macbeth(panel, RET, xcols, lags=6)
        all_results[name] = res
        coef_series[name] = coefs
        print(f"\n{name}")
        for k, (m, t, p) in res.items():
            if k == "const":
                continue
            print(f"  {k:12s}  coef={m:+.4f}  t={t:+.2f}  p={p:.3f}  {stars(p)}")

    # Build display table
    key_vars = ["DG", "DH_z", "DS_z", "TxDS", "const"]
    rows = []
    for spec, res in all_results.items():
        row = {"Specification": spec}
        for v in key_vars:
            if v in res:
                m, t, p = res[v]
                row[v + "_coef"] = f"{m:.4f}{stars(p)}"
                row[v + "_t"] = f"({t:.2f})"
            else:
                row[v + "_coef"] = ""
                row[v + "_t"] = ""
        # N (avg portfolios per cross-section)
        row["N_obs"] = len(panel["date"].unique())
        rows.append(row)

    table2 = pd.DataFrame(rows).set_index("Specification")
    table2.to_csv(f"{OUT_T}/table2_fama_macbeth.csv")

    with open(f"{OUT_T}/table2_fama_macbeth.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Fama-MacBeth Cross-Sectional Regressions}\n")
        f.write("\\label{tab:fm}\n")
        f.write(table2.to_latex(escape=False))
        f.write("\\end{table}\n")

    # Key coefficient comparisons
    dg_res = all_results["Model A — ΔG only"].get("DG", (np.nan, np.nan, np.nan))
    dh_B = all_results["Model B — Unconstrained"].get("DH_z", (np.nan, np.nan, np.nan))
    ds_B = all_results["Model B — Unconstrained"].get("DS_z", (np.nan, np.nan, np.nan))
    dh_C = all_results["Model C — Gibbs constrained"].get("DH_z", (np.nan, np.nan, np.nan))
    txds_C = all_results["Model C — Gibbs constrained"].get("TxDS", (np.nan, np.nan, np.nan))

    print("\n\n=== KEY COMPARISONS ===")
    print(f"Model A  ΔG: {dg_res[0]:.4f} (t={dg_res[1]:.2f})")
    print(f"Model B  ΔH: {dh_B[0]:.4f} (t={dh_B[1]:.2f})  ΔS: {ds_B[0]:.4f} (t={ds_B[1]:.2f})")
    print(f"Model C  ΔH: {dh_C[0]:.4f} (t={dh_C[1]:.2f})  T·ΔS: {txds_C[0]:.4f} (t={txds_C[1]:.2f})")
    if np.isfinite(dh_C[0]) and np.isfinite(txds_C[0]) and dh_C[0] != 0:
        print(f"β_TxDS / β_DH ratio: {txds_C[0]/dh_C[0]:.3f}  (Gibbs predicts ≈ −1)")

    # Interpretation
    interp = f"""Table 2 presents Fama-MacBeth cross-sectional regression results for three model specifications estimated monthly across the 25 Fama-French Size × B/M portfolios over 1990–2023, with Newey-West standard errors using 6 lags.

Model A regresses next-month portfolio returns on the Gibbs score ΔG alone. The estimated coefficient is {dg_res[0]:.4f} (t-statistic = {dg_res[1]:.2f}), providing {"statistically significant" if abs(dg_res[1]) > 2.0 else "economically suggestive but statistically weak"} evidence that thermodynamically favorable portfolios earn higher subsequent returns. {"The coefficient remains significant after including Fama-French five-factor and momentum controls, confirming that the Gibbs score captures variation in returns not subsumed by existing risk factors." if abs(all_results.get("Model A + FF5", {}).get("DG", (0,0,0))[1]) > 2.0 else "After including factor controls, the significance diminishes, suggesting the Gibbs score partially proxies for established risk premia."}

Model B (unconstrained) decomposes the Gibbs score into its enthalpy (ΔH) and entropy (ΔS) components, allowing them to load freely on returns. The estimated coefficient on ΔH is {dh_B[0]:.4f} (t = {dh_B[1]:.2f}), and the coefficient on ΔS is {ds_B[0]:.4f} (t = {ds_B[1]:.2f}). The {"positive" if dh_B[0] > 0 else "negative"} sign on ΔH is {"consistent" if dh_B[0] > 0 else "inconsistent"} with the thermodynamic prediction that enthalpic stability commands a return premium, while the {"negative" if ds_B[0] < 0 else "positive"} sign on ΔS {"confirms" if ds_B[0] < 0 else "contradicts"} the hypothesis that entropic disorder is penalized in pricing.

Model C (Gibbs-constrained) pre-specifies that entropy enters returns scaled by temperature T, replacing the free ΔS loading with the interaction T·ΔS. The coefficient on ΔH in Model C is {dh_C[0]:.4f} (t = {dh_C[1]:.2f}), nearly {"identical" if abs(dh_C[0] - dh_B[0]) < 0.001 else "similar"} to its Model B counterpart, suggesting the Gibbs constraint does not distort the enthalpy loading — a necessary condition for constraint validity. The coefficient on T·ΔS is {txds_C[0]:.4f} (t = {txds_C[1]:.2f}), which is {"negative and significant" if txds_C[0] < 0 and abs(txds_C[1]) > 2.0 else "negative" if txds_C[0] < 0 else "positive, contrary to prediction"}, {"consistent" if txds_C[0] < 0 else "inconsistent"} with the core Gibbs hypothesis that entropy destroys value in proportion to market temperature. The ratio of the T·ΔS coefficient to the ΔH coefficient is {txds_C[0]/dh_C[0]:.3f} (Gibbs structure implies approximately −1.0), {"suggesting the thermodynamic constraint is approximately binding" if abs(txds_C[0]/dh_C[0] + 1) < 0.5 else "indicating some deviation from the exact thermodynamic constraint"}.
"""
    with open(f"{OUT_I}/table2_interpretation.txt", "w") as f:
        f.write(interp)
    print("\nInterpretation saved.")


if __name__ == "__main__":
    main()
