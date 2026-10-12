"""
Master Pipeline Runner — orchestrates all phases of the ML framework.

Usage:
  python scripts/run_pipeline.py --phase 1   # Severity labeling (Phase 1)
  python scripts/run_pipeline.py --phase 2   # Data splitting (Phase 2)
  python scripts/run_pipeline.py --phase 3   # Model training (Phase 3)
  python scripts/run_pipeline.py --phase 4   # Explainability + evaluation (Phase 4+5)
  python scripts/run_pipeline.py --phase 6   # Launch Streamlit demo
  python scripts/run_pipeline.py --all       # Run phases 1-5 sequentially
"""
import sys
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config

import argparse
import time
import os
import subprocess


def print_banner(phase_num, text):
    print("\n" + "=" * 60)
    print(f"  PHASE {phase_num}: {text}")
    print("=" * 60 + "\n")


def run_phase_1():
    print_banner(1, "Severity Label Generation")
    from src.label_paddy import label_paddy_dataset
    label_paddy_dataset()
    print(f"\nPhase 1 complete. Labels saved to {config.PADDY_SEVERITY_CSV}")


def run_phase_2():
    print_banner(2, "Data Integrity & Splitting")
    if not os.path.exists(config.PADDY_SEVERITY_CSV):
        print("Warning: Paddy severity labels not found. Run Phase 1 first.")
        print("Proceeding with arecanut split only...")

    from src.split_data import split_arecanut, split_paddy
    split_arecanut()
    if os.path.exists(config.PADDY_SEVERITY_CSV):
        split_paddy()
    print(f"\nPhase 2 complete. Splits saved to {config.OUTPUTS_DIR}")


def run_phase_3():
    print_banner(3, "Model Training")
    # Check prerequisites
    arecanut_exists = os.path.exists(config.ARECANUT_SPLIT_CSV)
    paddy_exists = os.path.exists(config.PADDY_SPLIT_CSV)

    if not arecanut_exists and not paddy_exists:
        print("Error: No split CSVs found! Please run Phase 2 first.")
        return

    from src.train import train_model
    results = {}

    if arecanut_exists:
        print("\n--- Training Arecanut Ordinal Model ---")
        h = train_model('arecanut', config.ARECANUT_SPLIT_CSV, ordinal=True)
        results['arecanut_ordinal'] = h.get('best_val_qwk', 'N/A')

        print("\n--- Training Arecanut Baseline Model ---")
        h = train_model('arecanut', config.ARECANUT_SPLIT_CSV, ordinal=False)
        results['arecanut_baseline'] = h.get('best_val_qwk', 'N/A')

    if paddy_exists:
        print("\n--- Training Paddy Ordinal Model ---")
        h = train_model('paddy', config.PADDY_SPLIT_CSV, ordinal=True)
        results['paddy_ordinal'] = h.get('best_val_qwk', 'N/A')

        print("\n--- Training Paddy Baseline Model ---")
        h = train_model('paddy', config.PADDY_SPLIT_CSV, ordinal=False)
        results['paddy_baseline'] = h.get('best_val_qwk', 'N/A')

    print("\n" + "=" * 40)
    print("  TRAINING RESULTS (Best Val QWK)")
    print("=" * 40)
    for name, qwk in results.items():
        print(f"  {name:25s}: {qwk:.4f}" if isinstance(qwk, float) else f"  {name:25s}: {qwk}")
    print("=" * 40)


def run_phase_4_5():
    print_banner("4+5", "Explainability & Evaluation")
    subprocess.run([sys.executable, os.path.join(config.PROJECT_ROOT, "scripts", "run_evaluation.py")])


def run_phase_6():
    print_banner(6, "Streamlit Demo")
    print("Launching Streamlit app. Press Ctrl+C to stop.\n")
    subprocess.run(["streamlit", "run", os.path.join(config.PROJECT_ROOT, "app.py"),
                     "--server.headless", "true"])


def main():
    parser = argparse.ArgumentParser(
        description="Master Pipeline Runner for Crop Disease Severity ML Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 6],
                        help="Run a specific phase")
    parser.add_argument("--all", action="store_true",
                        help="Run phases 1-5 sequentially")
    args = parser.parse_args()

    config.ensure_dirs()
    start_time = time.time()

    if args.all:
        for phase_fn in [run_phase_1, run_phase_2, run_phase_3, run_phase_4_5]:
            phase_start = time.time()
            phase_fn()
            print(f"  ⏱ Phase completed in {time.time() - phase_start:.1f}s")
    elif args.phase == 1:
        run_phase_1()
    elif args.phase == 2:
        run_phase_2()
    elif args.phase == 3:
        run_phase_3()
    elif args.phase == 4:
        run_phase_4_5()
    elif args.phase == 6:
        run_phase_6()
    else:
        parser.print_help()
        return

    elapsed = time.time() - start_time
    if args.phase != 6:
        print(f"\n✅ Pipeline completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
