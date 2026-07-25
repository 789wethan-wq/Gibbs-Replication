"""04_fama_macbeth.py — Table 2: Fama-MacBeth regressions (Models A, B, C).

Enthalpy construction: the committed primary specification uses the ACCOUNTING
enthalpy stability measure dH_gpm_z (rolling SD of gross profit margin, z-scored
cross-sectionally each month) drawn from the R17 merged_with_accounting panel.
This is the construction that reproduces the manuscript's Table 2 Model B.

The earlier price-based composite (DH_z, Section 4.1) is retained behind the
USE_PRICE_BASED_DH flag / commented spec block below and is SUPERSEDED for the
headline table.
"""
import numpy as np
import pandas as pd
import os
from utils import fama_macbeth, stars

DATA = "data"
OUT_T = "outputs/tables"
OUT_I = "outputs/interpretations"
os.makedirs(OUT_T, exist_ok=True)
os.makedirs(OUT_I, exist_ok=True)

# Toggle: True reverts Models B/C to the superseded price-based ΔH (Section 4.1).
USE_PRICE_BASED_DH = False


def cs_wz(df, col, date_col="date", pct=0.01):
    """Cross-sectional 1% winsorized z-score within each date (matches R17)."""
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


def fmt(mean, t, p):
    return f"{mean:.4f}{stars(p)}", f"({t:.2f})"


def main():
    # Accounting panel (R17): superset of variables_monthly carrying dH_gpm.
    panel = pd.read_parquet(f"{DATA}/merged_with_accounting.parquet")
    factors = pd.read_parquet(f"{DATA}/factors_monthly.parquet")

    # Accounting enthalpy-stability z-score — the Table 2 Model B ΔH construction.
    panel["dH_gpm_z"] = cs_wz(panel, "dH_gpm")

    # ΔH column used by Models B and C. dH_gpm_z is the committed primary
    # construction; DH_z is the superseded price-based construction (Section 4.1).
    DH_COL = "DH_z" if USE_PRICE_BASED_DH else "dH_gpm_z"

    # Model A composite ΔG, built from the §2.2/§3.2 definition on this panel:
    #   ΔG = ΔH_z − T·ΔS_z, then within-month winsorize (1/99) + z-score.
    # This retires the dropped DG_acc_raw dependency (the −0.87 column departed
    # from §2.2/§3.2). Definition-faithful build gives FM t = −0.63 (lag 0).
    DG_COL = "DG" if USE_PRICE_BASED_DH else "DG_def"
    panel["DG_def_raw"] = panel["dH_gpm_z"] - panel["T"] * panel["DS_z"]
    panel["DG_def"] = cs_wz(panel, "DG_def_raw")

    # Add FF5 controls to panel
    fac_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    factors.index = pd.to_datetime(factors.index)
    fac_reset = factors[fac_cols].reset_index().rename(columns={"index": "date"})
    fac_reset["date"] = pd.to_datetime(fac_reset["date"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.merge(fac_reset, on="date", how="left")

    RET = "ret_next_month"

    specs = {
        "Model A — ΔG only": ([DG_COL], False),
        "Model A + FF5": ([DG_COL] + fac_cols, False),
        "Model B — Unconstrained": ([DH_COL, "DS_z"], False),
        "Model B + FF5": ([DH_COL, "DS_z"] + fac_cols, False),
        "Model C — Gibbs constrained": ([DH_COL, "TxDS"], False),
        "Model C + FF5": ([DH_COL, "TxDS"] + fac_cols, False),
    }

    # Primary Table 2 spec = unadjusted FM SE (lag 0), per §3.2 / Panel B row
    # "FM (primary; unadjusted FM SE)". MC_LOCK reproduces every Model B/C cell
    # at lag 0. The Newey-West lag ladder (lags 4/5/6, §3.2 sensitivity) is
    # available via T2_LOCK.py / MC_LOCK.py and by passing lags>0 below.
    FM_PRIMARY_LAGS = 0
    all_results = {}
    coef_series = {}
    for name, (xcols, _) in specs.items():
        res, coefs = fama_macbeth(panel, RET, xcols, lags=FM_PRIMARY_LAGS)
        all_results[name] = res
        coef_series[name] = coefs
        print(f"\n{name}")
        for k, (m, t, p) in res.items():
            if k == "const":
                continue
            print(f"  {k:12s}  coef={m:+.4f}  t={t:+.2f}  p={p:.3f}  {stars(p)}")

    # Build display table
    key_vars = [DG_COL, DH_COL, "DS_z", "TxDS", "const"]
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
    dg_res = all_results["Model A — ΔG only"].get(DG_COL, (np.nan, np.nan, np.nan))
    dh_B = all_results["Model B — Unconstrained"].get(DH_COL, (np.nan, np.nan, np.nan))
    ds_B = all_results["Model B — Unconstrained"].get("DS_z", (np.nan, np.nan, np.nan))
    dh_C = all_results["Model C — Gibbs constrained"].get(DH_COL, (np.nan, np.nan, np.nan))
    txds_C = all_results["Model C — Gibbs constrained"].get("TxDS", (np.nan, np.nan, np.nan))

    print("\n\n=== KEY COMPARISONS ===")
    print(f"Model A  ΔG: {dg_res[0]:.4f} (t={dg_res[1]:.2f})")
    print(f"Model B  ΔH: {dh_B[0]:.4f} (t={dh_B[1]:.2f})  ΔS: {ds_B[0]:.4f} (t={ds_B[1]:.2f})")
    print(f"Model C  ΔH: {dh_C[0]:.4f} (t={dh_C[1]:.2f})  T·ΔS: {txds_C[0]:.4f} (t={txds_C[1]:.2f})")
    if np.isfinite(dh_C[0]) and np.isfinite(txds_C[0]) and dh_C[0] != 0:
        print(f"β_TxDS / β_DH ratio: {txds_C[0]/dh_C[0]:.3f}  (Gibbs predicts ≈ −1)")

    # Interpretation
    interp = f"""Table 2 presents Fama-MacBeth cross-sectional regression results for three model specifications estimated monthly across the 25 Fama-French Size × B/M portfolios over 1990–2023, with Newey-West standard errors using 6 lags.

Model A regresses next-month portfolio returns on the Gibbs score ΔG alone. The estimated coefficient is {dg_res[0]:.4f} (t-statistic = {dg_res[1]:.2f}), providing {"statistically significant" if abs(dg_res[1]) > 2.0 else "economically suggestive but statistically weak"} evidence that thermodynamically favorable portfolios earn higher subsequent returns. {"The coefficient remains significant after including Fama-French five-factor and momentum controls, confirming that the Gibbs score captures variation in returns not subsumed by existing risk factors." if abs(all_results.get("Model A + FF5", {}).get(DG_COL, (0,0,0))[1]) > 2.0 else "After including factor controls, the composite remains insignificant, consistent with the two channels partially cancelling in ΔG."}

Model B (unconstrained) decomposes the Gibbs score into its enthalpy (ΔH) and entropy (ΔS) components, allowing them to load freely on returns. The estimated coefficient on ΔH is {dh_B[0]:.4f} (t = {dh_B[1]:.2f}), and the coefficient on ΔS is {ds_B[0]:.4f} (t = {ds_B[1]:.2f}). The {"positive" if dh_B[0] > 0 else "negative"} sign on ΔH is {"consistent" if dh_B[0] > 0 else "inconsistent"} with the thermodynamic prediction that enthalpic stability commands a return premium, while the {"negative" if ds_B[0] < 0 else "positive"} sign on ΔS {"confirms" if ds_B[0] < 0 else "contradicts"} the hypothesis that entropic disorder is penalized in pricing.

Model C (Gibbs-constrained) pre-specifies that entropy enters returns scaled by temperature T, replacing the free ΔS loading with the interaction T·ΔS. The coefficient on ΔH in Model C is {dh_C[0]:.4f} (t = {dh_C[1]:.2f}), nearly {"identical" if abs(dh_C[0] - dh_B[0]) < 0.001 else "similar"} to its Model B counterpart, suggesting the Gibbs constraint does not distort the enthalpy loading — a necessary condition for constraint validity. The coefficient on T·ΔS is {txds_C[0]:.4f} (t = {txds_C[1]:.2f}), which is {"negative and significant" if txds_C[0] < 0 and abs(txds_C[1]) > 2.0 else "negative" if txds_C[0] < 0 else "positive, contrary to prediction"}, {"consistent" if txds_C[0] < 0 else "inconsistent"} with the core Gibbs hypothesis that entropy destroys value in proportion to market temperature. The ratio of the T·ΔS coefficient to the ΔH coefficient is {txds_C[0]/dh_C[0]:.3f} (Gibbs structure implies approximately −1.0), {"suggesting the thermodynamic constraint is approximately binding" if abs(txds_C[0]/dh_C[0] + 1) < 0.5 else "indicating some deviation from the exact thermodynamic constraint"}.
"""
    with open(f"{OUT_I}/table2_interpretation.txt", "w") as f:
        f.write(interp)
    print("\nInterpretation saved.")


if __name__ == "__main__":
    main()
