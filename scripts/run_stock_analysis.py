"""run_stock_analysis.py — Re-run modules 02-10 using stock-level variables panel.

Patches each module's DATA file reference before calling main(). Output tables and
figures are written to outputs/ and will overwrite portfolio-level results.
"""
import shutil, os, importlib, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = "../data"

# Swap in stock-level panel
src = f"{DATA}/variables_stock_monthly.parquet"
dst = f"{DATA}/variables_monthly.parquet"
bak = f"{DATA}/variables_portfolio_monthly.parquet"

print("=== STOCK-LEVEL FULL ANALYSIS RUN ===\n")

# Back up portfolio panel, install stock panel
if os.path.exists(dst) and not os.path.exists(bak):
    shutil.copy(dst, bak)
    print("  Backed up portfolio panel to variables_portfolio_monthly.parquet")
shutil.copy(src, dst)
print(f"  Using stock panel: {src}\n")

modules = [
    "02_summary_statistics",
    "03_portfolio_sorts",
    "04_fama_macbeth",
    "05_constraint_validity",
    "06_regime_analysis",
    "07_oos_test",
    "08_robustness",
    "09_plots",
    "10_paper_tables",
]

for mod_name in modules:
    print(f"\n{'='*55}")
    print(f"  Running {mod_name}...")
    print(f"{'='*55}")
    try:
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "main"):
            mod.main()
    except Exception as e:
        import traceback
        print(f"  ERROR in {mod_name}: {e}")
        traceback.print_exc()

print("\n\n=== ALL MODULES COMPLETE ===")
