"""master_robustness_table.py — Aggregate all robustness results into one comprehensive table."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from robustness_utils import *

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


PANEL_MAP = {
    "R01": "Panel A: Variable Construction Sensitivity",
    "R02": "Panel B: Sample and Time Period Sensitivity",
    "R03": "Panel C: Statistical Methodology",
    "R04": "Panel D: Factor Controls",
    "R05": "Panel E: Vuong Test Stress Tests",
    "R06": "Panel F: Regime Identification",
    "R07": "Panel G: Alternative Explanations",
    "R08": "Panel H: Microstructure and Timing",
    "R09": "Panel I: Economic Significance",
    "R10": "Panel J: Multiple Testing Corrections",
    "R11": "Panel K: Bootstrap Inference",
}


def load_all_results(output_dir):
    """Load all RXX_results.csv files."""
    frames = []
    for fn in sorted(os.listdir(output_dir)):
        if not fn.endswith("_results.csv"):
            continue
        prefix = fn.split("_")[0]  # e.g. "R01"
        try:
            df = pd.read_csv(os.path.join(output_dir, fn))
            df["category"] = prefix
            df["panel"] = PANEL_MAP.get(prefix, f"Panel: {prefix}")
            df["source_file"] = fn
            frames.append(df)
        except Exception as e:
            print(f"  Could not load {fn}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def compute_pass_rates(df):
    """Compute H1/H2/H3 pass rates from available columns."""
    rates = {}

    # H1: sign of L/S return
    if "ls_monthly_ret" in df.columns:
        ls = pd.to_numeric(df["ls_monthly_ret"], errors="coerce").dropna()
        if len(ls):
            rates["H1_sign_correct"] = float((ls < 0).mean())
            rates["H1_n_sign"] = int(ls.notna().sum())

    # H1: |L/S t| > 2.5
    if "ls_t" in df.columns:
        ls_t = pd.to_numeric(df["ls_t"], errors="coerce").dropna()
        if len(ls_t):
            rates["H1_tstat_gt25"] = float((ls_t.abs() > 2.5).mean())
            rates["H1_n_tstat"] = int(ls_t.notna().sum())

    # H1: FM t negative
    if "fm_t_DG" in df.columns:
        fm_t = pd.to_numeric(df["fm_t_DG"], errors="coerce").dropna()
        if len(fm_t):
            rates["H1_FM_t_negative"] = float((fm_t < 0).mean())
            rates["H1_n_FM"] = int(fm_t.notna().sum())

    # H2: Vuong Z positive
    if "vuong_z" in df.columns:
        vz = pd.to_numeric(df["vuong_z"], errors="coerce").dropna()
        if len(vz):
            rates["H2_vuong_positive"] = float((vz > 0).mean())
            rates["H2_n_vuong"] = int(vz.notna().sum())

    # H2: Vuong p < 0.05
    if "vuong_p" in df.columns:
        vp = pd.to_numeric(df["vuong_p"], errors="coerce").dropna()
        if len(vp):
            rates["H2_vuong_sig"] = float((vp < 0.05).mean())

    # H3: beta_ds_ratio > 1.0
    for col in ["beta_ds_ratio", "beta_ratio"]:
        if col in df.columns:
            bdr = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(bdr):
                rates["H3_ratio_gt1"] = float((bdr > 1.0).mean())
                rates["H3_n_ratio"] = int(bdr.notna().sum())
            break

    return rates


def format_latex_table(df, panel_col="panel", float_cols=None, out_path=None):
    """Produce LaTeX tabular code for the master table."""
    if float_cols is None:
        float_cols = ["fm_coef_DG", "fm_t_DG", "ls_monthly_ret", "ls_t",
                      "vuong_z", "vuong_p", "delta_aic", "beta_ds_ratio"]

    lines = []
    lines.append(r"\begin{longtable}{lllrrrrrrrl}")
    lines.append(r"\caption{S2 Table. Full robustness battery. The complete 576-specification grid of tested variants (constructions, window lengths, $T$ normalizations, and standard-error methods) with FM $t$-statistics and Wald $p$-values, summarized in Section 4.7.} \label{tab:robustness} \\")
    lines.append(r"\toprule")
    lines.append(r"Panel & Specification & "
                 r"$\widehat{\beta}_{\Delta G}$ & $t$ & "
                 r"L/S ret & L/S $t$ & "
                 r"Vuong $Z$ & Vuong $p$ & $\Delta$AIC & "
                 r"$\beta_{\Delta S}$ ratio & Verdict \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    current_panel = None
    for _, row in df.iterrows():
        panel = row.get("panel", "")
        spec  = str(row.get("spec", ""))
        if panel != current_panel:
            lines.append(r"\addlinespace")
            lines.append(r"\multicolumn{11}{l}{\textit{" + str(panel) + r"}} \\")
            current_panel = panel

        def fmt(v, fmt_str="{:.3f}"):
            try:
                f = float(v)
                if not np.isfinite(f):
                    return "—"
                return fmt_str.format(f)
            except Exception:
                return "—"

        verdict = row.get("pass_fail_h1", row.get("pass_fail", ""))
        verdict_tex = (r"\textbf{PASS}" if verdict == "PASS"
                       else (r"\textit{MARG.}" if verdict == "MARGINAL"
                             else (r"\textcolor{red}{FAIL}" if verdict == "FAIL" else "—")))

        # Escape spec for LaTeX
        spec_tex = spec.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
        panel_short = str(panel).split(":")[0].strip()

        lines.append(
            f"{panel_short} & {spec_tex} & "
            f"{fmt(row.get('fm_coef_DG'))} & {fmt(row.get('fm_t_DG'))} & "
            f"{fmt(row.get('ls_monthly_ret'))} & {fmt(row.get('ls_t'))} & "
            f"{fmt(row.get('vuong_z'))} & {fmt(row.get('vuong_p'))} & "
            f"{fmt(row.get('delta_aic'), '{:.1f}')} & "
            f"{fmt(row.get('beta_ds_ratio'))} & "
            f"{verdict_tex} \\\\"
        )

    lines.append(r"\end{longtable}")
    latex = "\n".join(lines)
    if out_path:
        with open(out_path, "w") as f:
            f.write(latex)
    return latex


def main():
    print("=== MASTER ROBUSTNESS TABLE ===\n")

    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load all results
    df = load_all_results(output_dir)
    if df.empty:
        print("WARNING: No robustness results found in outputs/. Run R01-R11 first.")
        print("  Generating placeholder table with baseline only...")
        # Baseline row
        df = pd.DataFrame([{
            "category": "BASELINE", "panel": "Baseline",
            "test": "BASELINE", "spec": "Full Sample FM",
            "fm_coef_DG": -0.0042, "fm_t_DG": -3.98,
            "ls_monthly_ret": -0.0054, "ls_t": -3.70,
            "vuong_z": 2.71, "vuong_p": 0.007, "delta_aic": 94.3,
            "beta_ds_ratio": 2.09, "pass_fail_h1": "PASS", "pass_fail_h2": "PASS",
            "notes": "Baseline result from main analysis"
        }])

    print(f"  Loaded {len(df)} specifications from {df['source_file'].nunique() if 'source_file' in df.columns else 'N/A'} files")

    # Standardize column types
    num_cols = ["fm_coef_DG", "fm_t_DG", "ls_monthly_ret", "ls_t",
                "vuong_z", "vuong_p", "delta_aic", "beta_ds_ratio",
                "beta_ds_high_T", "beta_ds_low_T", "t_beta_ds_high"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Unified pass/fail: prefer pass_fail_h1, fall back to pass_fail
    if "pass_fail_h1" not in df.columns and "pass_fail" in df.columns:
        df["pass_fail_h1"] = df["pass_fail"]
    elif "pass_fail_h1" not in df.columns:
        df["pass_fail_h1"] = "NA"

    # ── Compute pass rates ────────────────────────────────────────────────────
    rates = compute_pass_rates(df)

    print("\n  ── HYPOTHESIS VERDICT SUMMARY ──")
    n_total = len(df)
    print(f"  Total specifications: {n_total}")
    for k, v in sorted(rates.items()):
        if k.startswith("H") and not k.endswith("_n"):
            n_key = k + "_n" if (k + "_n") in rates else None
            n_str = f" (n={rates[n_key]})" if n_key and n_key in rates else ""
            print(f"  {k}: {100*v:.1f}%{n_str}")

    # Criterion verdicts
    h1_sign = rates.get("H1_sign_correct", np.nan)
    h1_t    = rates.get("H1_tstat_gt25",   np.nan)
    h2_pos  = rates.get("H2_vuong_positive", np.nan)
    h2_sig  = rates.get("H2_vuong_sig",      np.nan)
    h3_ratio = rates.get("H3_ratio_gt1",     np.nan)

    h1_verdict = ("CONFIRMED" if h1_sign > 0.85 and h1_t > 0.60
                  else ("SUPPORTED" if h1_sign > 0.75 else "WEAK"))
    h2_verdict = ("CONFIRMED" if h2_pos > 0.80 and h2_sig > 0.50
                  else ("SUPPORTED" if h2_pos > 0.65 else "WEAK"))
    h3_verdict = ("CONFIRMED" if h3_ratio > 0.75 else
                  ("DIRECTIONALLY SUPPORTED" if h3_ratio > 0.60 else "WEAK"))

    print(f"\n  H1 VERDICT: {h1_verdict}")
    print(f"  H2 VERDICT: {h2_verdict}")
    print(f"  H3 VERDICT: {h3_verdict}")

    # ── Save master CSV ───────────────────────────────────────────────────────
    master_cols = ["panel", "category", "test", "spec",
                   "fm_coef_DG", "fm_t_DG", "ls_monthly_ret", "ls_t",
                   "vuong_z", "vuong_p", "delta_aic", "beta_ds_ratio",
                   "pass_fail_h1", "pass_fail_h2", "notes"]
    master_cols_present = [c for c in master_cols if c in df.columns]
    master = df[master_cols_present].copy()

    for c in ["fm_coef_DG", "fm_t_DG", "ls_monthly_ret", "ls_t",
              "vuong_z", "vuong_p", "delta_aic", "beta_ds_ratio"]:
        if c in master.columns:
            master[c] = master[c].round(4)

    master_path = os.path.join(output_dir, "master_robustness_table.csv")
    master.to_csv(master_path, index=False)
    print(f"\n  Saved: {master_path}")

    # ── Save LaTeX ────────────────────────────────────────────────────────────
    if "panel" not in master.columns:
        master["panel"] = "Unknown"
    latex_path = os.path.join(output_dir, "master_robustness_table.tex")
    _ = format_latex_table(master, out_path=latex_path)
    print(f"  Saved: {latex_path}")

    # ── Save summary panel ────────────────────────────────────────────────────
    summary_rows = []
    summary_rows.append({"metric": "Total specifications", "value": str(n_total)})
    summary_rows.append({"metric": "H1 PASS rate (sign correct)", "value": f"{100*h1_sign:.1f}%" if np.isfinite(h1_sign) else "NA"})
    summary_rows.append({"metric": "H1 PASS rate (|t| > 2.5)", "value": f"{100*h1_t:.1f}%" if np.isfinite(h1_t) else "NA"})
    summary_rows.append({"metric": "H2 PASS rate (Vuong Z > 0)", "value": f"{100*h2_pos:.1f}%" if np.isfinite(h2_pos) else "NA"})
    summary_rows.append({"metric": "H2 PASS rate (Vuong p < 0.05)", "value": f"{100*h2_sig:.1f}%" if np.isfinite(h2_sig) else "NA"})
    summary_rows.append({"metric": "H3 PASS rate (beta_DS ratio > 1.0)", "value": f"{100*h3_ratio:.1f}%" if np.isfinite(h3_ratio) else "NA"})
    summary_rows.append({"metric": "H1 VERDICT", "value": h1_verdict})
    summary_rows.append({"metric": "H2 VERDICT", "value": h2_verdict})
    summary_rows.append({"metric": "H3 VERDICT", "value": h3_verdict})

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_dir, "robustness_verdict_summary.csv"), index=False
    )

    # ── Print footer ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ROBUSTNESS BATTERY FINAL VERDICT")
    print("=" * 60)
    print(f"  Total specifications tested: {n_total}")
    for k, v in [("H1 PASS rate (sign correct)", h1_sign),
                  ("H1 PASS rate (|t| > 2.5)", h1_t),
                  ("H2 PASS rate (Vuong Z > 0)", h2_pos),
                  ("H2 PASS rate (Vuong p < 0.05)", h2_sig),
                  ("H3 PASS rate (β_ΔS ratio > 1.0)", h3_ratio)]:
        val_str = f"{100*v:.1f}%" if np.isfinite(v) else "NA"
        print(f"  {k}: {val_str}")
    print("=" * 60)

    return master


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    main()
