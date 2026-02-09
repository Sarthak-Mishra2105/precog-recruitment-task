#!/usr/bin/env python
"""
run_all.py - Quickstart Runner for Colored MNIST CV Task
=========================================================

Usage:
    python run_all.py                        # Default: load_artifacts mode
    python run_all.py --mode load_artifacts  # Verify pre-trained models exist
    python run_all.py --mode run_from_scratch # Run full pipeline (~20 min)

This script provides a single entry point to:
1. Print environment info
2. Verify required artifact files exist
3. Optionally run the full training pipeline
"""
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import print_env_info, set_seed, DEFAULT_SEED


def check_artifacts(artifacts_dir: Path) -> bool:
    """Check that required artifact files exist for load_artifacts mode."""
    required_files = [
        # Model checkpoints
        "baseline_cheater_N2000.pt",
        "model_consistency.pt",
        "model_grl.pt",
        # Key visualizations
        "task2_actmax_logits_cheater.png",
        "task3_gradcam_cheater_val.png",
        "task5_attack_comparison.png",
        "task5_attack_summary.txt",
    ]
    
    print("\n" + "=" * 50)
    print("ARTIFACT CHECKLIST")
    print("=" * 50)
    
    all_present = True
    for fname in required_files:
        fpath = artifacts_dir / fname
        exists = fpath.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {fname}")
        if not exists:
            all_present = False
    
    print("=" * 50)
    return all_present


def run_from_scratch():
    """Run the full training pipeline by calling each script in order."""
    import subprocess
    
    scripts = [
        ("run_task2_task3.py", "Tasks 1-3: Cheater baseline + interpretability"),
        ("run_task4.py", "Task 4: Debiasing (Consistency + GRL)"),
        ("run_task5_attack.py", "Task 5: Adversarial attacks"),
    ]
    
    print("\n" + "=" * 60)
    print("RUNNING FULL PIPELINE FROM SCRATCH")
    print("=" * 60)
    print("This will take approximately 15-20 minutes on CPU.")
    print("GPU will be faster if available.\n")
    
    for script, description in scripts:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Script: {script}")
        print("=" * 60 + "\n")
        
        result = subprocess.run(
            [sys.executable, script],
            cwd=Path(__file__).parent
        )
        
        if result.returncode != 0:
            print(f"\n[ERROR] {script} failed with return code {result.returncode}")
            return False
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Colored MNIST CV Task - Quickstart Runner"
    )
    parser.add_argument(
        "--mode",
        choices=["load_artifacts", "run_from_scratch"],
        default="load_artifacts",
        help="Execution mode (default: load_artifacts)"
    )
    args = parser.parse_args()
    
    # Setup
    project_root = Path(__file__).parent
    artifacts_dir = project_root / "artifacts"
    
    # Print environment info
    print_env_info()
    
    # Set seed
    set_seed(DEFAULT_SEED)
    print(f"\nRandom seed set to: {DEFAULT_SEED}")
    print(f"Mode: {args.mode}")
    
    if args.mode == "load_artifacts":
        # Verify artifacts exist
        if check_artifacts(artifacts_dir):
            print("\n[SUCCESS] All required artifacts present.")
            print("\nNext steps:")
            print("  1. Open notebooks/cv_task.ipynb")
            print("  2. Set RUN_MODE = 'load_artifacts'")
            print("  3. Run all cells to see results")
        else:
            print("\n[WARNING] Some artifacts are missing.")
            print("Run with --mode run_from_scratch to generate them:")
            print("  python run_all.py --mode run_from_scratch")
            return 1
    
    elif args.mode == "run_from_scratch":
        if not run_from_scratch():
            print("\n[ERROR] Pipeline failed.")
            return 1
        
        # Verify artifacts after run
        print("\nVerifying generated artifacts...")
        check_artifacts(artifacts_dir)
    
    print("\n[DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
