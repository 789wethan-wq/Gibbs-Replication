"""run_all_robustness.py — Run the complete robustness battery R01–R11 + master table."""
import sys, os, importlib, traceback, time
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODULES = [
    "R01_variable_construction",
    "R02_sample_sensitivity",
    "R03_statistical_methods",
    "R04_factor_controls",
    "R05_vuong_stress",
    "R06_regime_sensitivity",
    "R07_confounds",
    "R08_microstructure",
    "R09_economic_significance",
    "R10_multiple_testing",
    "R11_bootstrap",
    "master_robustness_table",
]

def main():
    print("=" * 65)
    print("  GIBBS EQUITY MODEL — ROBUSTNESS BATTERY")
    print("=" * 65)

    results = {}
    for mod_name in MODULES:
        print(f"\n{'='*65}")
        print(f"  Running {mod_name}...")
        print(f"{'='*65}")
        t0 = time.time()
        try:
            # Force reload
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "main"):
                mod.main()
            results[mod_name] = "OK"
        except Exception as e:
            print(f"  ERROR in {mod_name}: {e}")
            traceback.print_exc()
            results[mod_name] = f"ERROR: {e}"
        elapsed = time.time() - t0
        print(f"  [{mod_name}] done in {elapsed:.1f}s — {results[mod_name]}")

    print("\n\n" + "=" * 65)
    print("  ROBUSTNESS BATTERY COMPLETE")
    print("=" * 65)
    for mod, status in results.items():
        marker = "✓" if status == "OK" else "✗"
        print(f"  {marker}  {mod}: {status}")

    ok  = sum(1 for v in results.values() if v == "OK")
    err = sum(1 for v in results.values() if "ERROR" in v)
    print(f"\n  {ok}/{len(results)} modules completed successfully, {err} errors")
    print(f"\n  Outputs in: {os.path.abspath('outputs/')}")

if __name__ == "__main__":
    main()
