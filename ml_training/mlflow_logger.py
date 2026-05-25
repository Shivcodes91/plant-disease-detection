# =============================================================
# ml_training/mlflow_logger.py
#
# MLflow experiment tracking.
#
# WHAT IS MLFLOW?
# Think of it like a lab notebook for ML experiments.
# Every time you train a model, MLflow records:
# - What settings you used (hyperparameters)
# - How well it performed (metrics)
# - The actual model file (artifacts)
#
# WHY? So you can compare runs and find the best config.
# Industry standard tool used at Netflix, Facebook, etc.
# =============================================================

import mlflow
import mlflow.pytorch
import json
import torch
from pathlib import Path
from typing import Dict
import sys

sys.path.append(str(Path(__file__).parent))


def setup_mlflow(experiment_name: str = "plant-disease-detection"):
    """
    Sets up MLflow experiment.

    Args:
        experiment_name: name of the experiment group
    """
    # Set tracking URI — where MLflow stores data
    # file:./mlruns = local folder (no server needed)
    mlflow.set_tracking_uri("file:./mlruns")

    # Create or get experiment
    # All runs for this project go under one experiment
    mlflow.set_experiment(experiment_name)

    print(f"✓ MLflow experiment: {experiment_name}")
    print(f"✓ Tracking URI: file:./mlruns")


def log_training_run(config: Dict,
                      history: Dict,
                      model_path: str,
                      model):
    """
    Logs a complete training run to MLflow.

    Args:
        config: training configuration dictionary
        history: training history (loss, accuracy per epoch)
        model_path: path to saved model file
        model: the trained PyTorch model
    """
    setup_mlflow()

    with mlflow.start_run(run_name="efficientnet_b4_training"):

        # ---- LOG HYPERPARAMETERS ----
        # These are the settings we used for this run
        # MLflow stores them so we can compare runs later
        mlflow.log_params({
            'model': 'EfficientNet-B4',
            'num_classes': config.get('num_classes', 15),
            'image_size': config.get('image_size', 224),
            'batch_size': config.get('batch_size', 32),
            'num_epochs': len(history['val_acc']),
            'head_lr': config.get('head_lr', 1e-3),
            'backbone_lr': config.get('backbone_lr', 1e-5),
            'weight_decay': config.get('weight_decay', 1e-4),
            'dropout_rate': config.get('dropout_rate', 0.3),
            'freeze_epochs': config.get('freeze_epochs', 5),
            'augmentation': 'flip+rotate+colorjitter+noise',
            'loss_function': 'LabelSmoothingCE(0.1)',
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'amp': True,
            'weighted_sampler': True
        })

        # ---- LOG METRICS PER EPOCH ----
        # This lets MLflow plot learning curves
        for epoch, (tl, ta, vl, va) in enumerate(zip(
            history['train_loss'],
            history['train_acc'],
            history['val_loss'],
            history['val_acc']
        ), 1):
            mlflow.log_metrics({
                'train_loss': round(tl, 4),
                'train_accuracy': round(ta, 2),
                'val_loss': round(vl, 4),
                'val_accuracy': round(va, 2),
            }, step=epoch)

        # ---- LOG SUMMARY METRICS ----
        best_val_acc = max(history['val_acc'])
        best_epoch = history['val_acc'].index(best_val_acc) + 1

        mlflow.log_metrics({
            'best_val_accuracy': round(best_val_acc, 2),
            'best_epoch': best_epoch,
            'final_train_accuracy': round(history['train_acc'][-1], 2),
            'final_val_accuracy': round(history['val_acc'][-1], 2),
        })

        # ---- LOG MODEL FILE ----
        # Save the model file as an artifact
        if Path(model_path).exists():
            mlflow.log_artifact(model_path, artifact_path="models")

        # ---- LOG TRAINING CURVES ----
        curves_path = 'data/training_curves.png'
        if Path(curves_path).exists():
            mlflow.log_artifact(curves_path, artifact_path="plots")

        # ---- LOG CLASS MAPPING ----
        mapping_path = 'data/class_mapping.json'
        if Path(mapping_path).exists():
            mlflow.log_artifact(mapping_path)

        print(f"\n✓ MLflow run logged!")
        print(f"  Best Val Accuracy : {best_val_acc:.2f}%")
        print(f"  Best Epoch        : {best_epoch}")


def view_results():
    """
    Loads and displays results from the training history file.
    """
    history_path = Path('backend/models/training_history.json')

    if not history_path.exists():
        print("No training history found!")
        return

    with open(history_path) as f:
        history = json.load(f)

    best_val = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val) + 1

    print("\n" + "=" * 50)
    print("  TRAINING RESULTS SUMMARY")
    print("=" * 50)
    print(f"  Total epochs     : {len(history['val_acc'])}")
    print(f"  Best val accuracy: {best_val:.2f}%")
    print(f"  Best epoch       : {best_epoch}")
    print(f"  Final train acc  : {history['train_acc'][-1]:.2f}%")
    print(f"  Final val acc    : {history['val_acc'][-1]:.2f}%")

    print(f"\n  Epoch-by-epoch:")
    print(f"  {'Epoch':>6} | {'Train Acc':>10} | "
          f"{'Val Acc':>8} | {'Val Loss':>9}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*9}")

    for i, (ta, va, vl) in enumerate(zip(
        history['train_acc'],
        history['val_acc'],
        history['val_loss']
    ), 1):
        best = " ← BEST" if va == best_val and i == best_epoch else ""
        print(f"  {i:>6} | {ta:>9.2f}% | "
              f"{va:>7.2f}% | {vl:>9.4f}{best}")


if __name__ == '__main__':

    print("=" * 55)
    print("  DAY 6 — MLFLOW EXPERIMENT TRACKING")
    print("=" * 55)

    # Load training history
    history_path = Path('backend/models/training_history.json')

    if not history_path.exists():
        print("❌ No training history found!")
        print("   Run python ml_training/train.py first")
        exit(1)

    with open(history_path) as f:
        history = json.load(f)

    # Load config (same as train.py CONFIG)
    config = {
        'num_classes': 15,
        'image_size': 224,
        'batch_size': 32,
        'head_lr': 1e-3,
        'backbone_lr': 1e-5,
        'weight_decay': 1e-4,
        'dropout_rate': 0.3,
        'freeze_epochs': 5,
    }

    # Log to MLflow
    print("\n[1/2] Logging training run to MLflow...")
    log_training_run(
        config=config,
        history=history,
        model_path='backend/models/best_model.pt',
        model=None
    )

    # Display results table
    print("\n[2/2] Training results summary:")
    view_results()

    print("\n" + "=" * 55)
    print("  MLFLOW LOGGING COMPLETE!")
    print("=" * 55)
    print("""
To view MLflow dashboard in browser:
  1. Open a new terminal
  2. Activate venv: venv\\Scripts\\activate
  3. Run: mlflow ui
  4. Open: http://localhost:5000

You'll see all your training metrics and curves!
    """)