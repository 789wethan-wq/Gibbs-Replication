"""
Concordance rate between quintile assignments under standard-T vs
expanding-window T normalization of ΔG. Reports overall and by quintile.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from robustness_utils import *
import warnings; warnings.filterwarnings("ignore")

OUT = "outputs"

def main():
    panel, _ = load_panel()
    p = panel.copy().sort_values("date")

    # ── Build expanding-window T (same logic as R13.1) ──
    t_series = p.groupby("date")["T"].first().sort_index()
    t_exp_mean = t_series.expanding(min_periods=12).mean().shift(1)
    t_exp_std  = t_series.expanding(min_periods=12).std().shift(1)
    t_norm_map = ((t_series - t_exp_mean) / t_exp_std.clip(lower=1e-8)).to_dict()
    p["T_expanding"] = p["date"].map(t_norm_map)
    p["DG_expanding"] = p["DH_z"] - p["T_expanding"] * p["DS_z"]
    p["DG_expanding"] = p.groupby("date")["DG_expanding"].transform(zscore_cs)

    # ── Quintile assignments ──
    def assign_q(series):
        return pd.qcut(series, 5, labels=False, duplicates="drop")

    p["q_std"] = p.groupby("date")["DG"].transform(
        lambda x: assign_q(x) if x.nunique() >= 5 else np.nan
    )
    p["q_exp"] = p.groupby("date")["DG_expanding"].transform(
        lambda x: assign_q(x) if x.nunique() >= 5 else np.nan
    )

    sub = p.dropna(subset=["q_std", "q_exp"])
    n_total = len(sub)

    # ── Overall concordance ──
    match = (sub["q_std"] == sub["q_exp"]).sum()
    concordance = match / n_total

    # ── By quintile (diagonal of confusion matrix) ──
    conf = pd.crosstab(sub["q_std"].astype(int), sub["q_exp"].astype(int),
                       rownames=["std"], colnames=["expanding"])

    print(f"\n  N stock-months compared:    {n_total:,}")
    print(f"  Quintile matches:           {match:,}")
    print(f"  Overall concordance rate:   {concordance:.4f}  ({concordance*100:.2f}%)")

    print("\n  Confusion matrix (rows=standard T, cols=expanding T):")
    print(conf.to_string())

    print("\n  Per-quintile concordance (% of standard-T quintile members retained):")
    for q in range(5):
        if q in conf.index and q in conf.columns:
            row_total = conf.loc[q].sum()
            diag      = conf.loc[q, q] if q in conf.columns else 0
            print(f"    Q{q+1}: {diag/row_total*100:.1f}%  ({diag:,}/{row_total:,})")

    # ── Tail-quintile focus (Q1 = high-ΔG, Q5 = low-ΔG) ──
    print("\n  Q1 (highest ΔG) concordance: ", end="")
    q1_row = conf.loc[0] if 0 in conf.index else None
    if q1_row is not None:
        print(f"{q1_row[0]/q1_row.sum()*100:.1f}%")
    print("  Q5 (lowest  ΔG) concordance: ", end="")
    q5_row = conf.loc[4] if 4 in conf.index else None
    if q5_row is not None:
        print(f"{q5_row[4]/q5_row.sum()*100:.1f}%")

    # ── Save ──
    result = {
        "n_total": n_total,
        "n_match": int(match),
        "concordance_rate": round(concordance, 6),
        "concordance_pct": round(concordance * 100, 2),
    }
    pd.DataFrame([result]).to_csv(f"{OUT}/concordance_quintile_std_vs_expanding.csv", index=False)
    print(f"\n  Saved: {OUT}/concordance_quintile_std_vs_expanding.csv")
    return result

if __name__ == "__main__":
    main()
