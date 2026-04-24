"""
train_model.py
==============
End-to-end training script for the Cricket Cover Drive ML pipeline.

Run from the project root:

    python train_model.py                          # uses / generates synthetic data
    python train_model.py --dataset data/my.csv    # uses a real labeled dataset
    python train_model.py --samples 1200           # larger synthetic dataset
    python train_model.py --model-dir models/      # custom output directory

The script will:
  1. Load or generate a labeled dataset
  2. Train SVM, Random Forest, Gradient Boosting, XGBoost, and an Ensemble
  3. Print a comparison table (CV accuracy + test accuracy for each model)
  4. Save the best model to disk for inference
  5. Save a JSON comparison report to output/model_comparison.json

After training, main.py and the FastAPI backend automatically use the
saved model for inference — no changes needed.
"""

import argparse
import os
import sys

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ml_model import MLTrainer
from src.dataset_utils import generate_synthetic_dataset, load_dataset, describe_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ML models for cricket cover drive shot quality."
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to a labeled CSV dataset. If omitted, a synthetic dataset is generated.",
    )
    parser.add_argument(
        "--samples", type=int, default=900,
        help="Number of synthetic samples to generate (default: 900).",
    )
    parser.add_argument(
        "--model-dir", type=str, default="models/",
        help="Directory where trained models will be saved (default: models/).",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--save-all", action="store_true",
        help="Save every trained model, not just the best one.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Step 1: Dataset ───────────────────────────────────────────────────────
    dataset_path = args.dataset

    if dataset_path is None:
        print("\n[train_model] No dataset provided — generating synthetic data …")
        dataset_path = "data/shot_quality_dataset.csv"
        generate_synthetic_dataset(
            n_samples=args.samples,
            output_path=dataset_path,
        )
    else:
        if not os.path.exists(dataset_path):
            print(f"[train_model] ERROR: Dataset not found at '{dataset_path}'")
            sys.exit(1)

    describe_dataset(dataset_path)

    X, y = load_dataset(dataset_path)

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    trainer = MLTrainer(cv_folds=args.cv_folds)
    trainer.train_all(X, y, verbose=True)

    # ── Step 3: Comparison report ─────────────────────────────────────────────
    print("\nModel Comparison:")
    print(trainer.comparison_report().to_string())

    trainer.save_comparison_json("output/model_comparison.json")

    # ── Step 4: Save models ───────────────────────────────────────────────────
    best_path = os.path.join(args.model_dir, "best_model.pkl")
    trainer.save_best(best_path)

    if args.save_all:
        trainer.save_all(args.model_dir)

    print("\n[train_model] Training complete.")
    print(f"  Best model : {trainer.best_name_}")
    print(f"  Saved to   : {best_path}")
    print("\nYou can now run the analysis pipeline:")
    print("  python main.py")
    print("  cd backend && uvicorn app:app --reload --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()