# =============================================================
# ml_training/train.py
#
# THE MAIN TRAINING SCRIPT
#
# This script:
# 1. Loads the dataset (from dataset.py)
# 2. Creates the model (from model.py)
# 3. Trains for N epochs
# 4. Validates after each epoch
# 5. Saves the best model automatically
# 6. Plots training curves at the end
#
# Run with: python ml_training/train.py
# =============================================================

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm  # progress bar library

# Add ml_training folder to Python path
# So we can import our own files (dataset.py, model.py)
sys.path.append(str(Path(__file__).parent))

from dataset import create_dataloaders
from model import (
    create_model,
    save_checkpoint,
    PlantDiseaseModel
)

# =============================================================
# SECTION 1: CONFIGURATION
# All training settings in one place
# Change these to experiment with different settings
# =============================================================

CONFIG = {
    # ---- Data ----
    'dataset_path': 'data/raw/PlantVillage/PlantVillage',
    'image_size': 224,        # resize all images to 224x224
    'batch_size': 32,         # images per batch
    'num_workers': 0,         # 0 = safe for Windows

    # ---- Model ----
    'num_classes': 15,        # our 15 plant disease classes
    'dropout_rate': 0.3,      # dropout for regularization
    'pretrained': True,       # use ImageNet pretrained weights

    # ---- Training ----
    'num_epochs': 25,         # total training epochs
    'freeze_epochs': 5,       # epochs to train with frozen backbone
                              # after this, unfreeze and fine-tune

    # ---- Optimizer ----
    # AdamW = Adam optimizer with weight decay
    # Weight decay = small penalty on large weights (prevents overfitting)
    'head_lr': 1e-3,          # learning rate for classifier head
    'backbone_lr': 1e-5,      # learning rate for backbone (after unfreeze)
    'weight_decay': 1e-4,     # L2 regularization strength

    # ---- Learning Rate Scheduler ----
    # CosineAnnealing smoothly reduces LR during training
    # Like cooling down — big steps early, tiny steps later
    'eta_min': 1e-6,          # minimum learning rate at end

    # ---- Saving ----
    'checkpoint_dir': 'backend/models',
    'save_every_n_epochs': 5, # save checkpoint every N epochs

    # ---- Early Stopping ----
    # Stop training if validation accuracy doesn't improve
    # Prevents wasting time training beyond peak performance
    'patience': 7,            # stop after 7 epochs of no improvement
}

# =============================================================
# SECTION 2: METRIC TRACKING
# =============================================================

class MetricTracker:
    """
    Tracks training and validation metrics across epochs.

    Stores:
    - Loss per epoch (train and val)
    - Accuracy per epoch (train and val)
    - Best accuracy seen so far
    - Learning rate per epoch
    """

    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'epoch_time': []
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0  # for early stopping

    def update(self, train_loss, train_acc,
               val_loss, val_acc, lr, epoch_time):
        """Add metrics for one epoch."""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)
        self.history['epoch_time'].append(epoch_time)

        # Check if this is the best validation accuracy
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = len(self.history['val_acc'])
            self.epochs_no_improve = 0
            return True  # is_best = True
        else:
            self.epochs_no_improve += 1
            return False  # is_best = False

    def should_stop_early(self, patience: int) -> bool:
        """Returns True if training should stop."""
        return self.epochs_no_improve >= patience

    def save(self, path: str):
        """Save training history to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


# =============================================================
# SECTION 3: TRAINING FUNCTIONS
# =============================================================

def train_one_epoch(model, train_loader, criterion,
                    optimizer, scaler, device, epoch):
    """
    Trains the model for ONE epoch.

    One epoch = the model sees every training image once.

    Args:
        model: our PlantDiseaseModel
        train_loader: DataLoader for training data
        criterion: loss function (LabelSmoothingLoss)
        optimizer: AdamW optimizer
        scaler: GradScaler for AMP (mixed precision)
        device: cuda or cpu
        epoch: current epoch number (for display)

    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """

    # Set model to TRAINING mode
    # This enables:
    # - Dropout (randomly zeros neurons)
    # - BatchNorm uses batch statistics
    model.train()

    # Accumulators for this epoch
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # tqdm creates a progress bar in the terminal
    # It shows: epoch, batch number, loss, accuracy in real time
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch:2d} [Train]",
        leave=False,        # don't leave progress bar after completion
        ncols=100           # width of progress bar
    )

    for batch_idx, (images, labels) in enumerate(progress_bar):

        # Move data to GPU (or CPU)
        # .to(device) copies tensor to the device's memory
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ---- ZERO GRADIENTS ----
        # Gradients accumulate by default in PyTorch
        # We must zero them before each batch
        # Otherwise gradients from previous batch affect current batch
        # set_to_none=True is slightly faster than zero_grad()
        optimizer.zero_grad(set_to_none=True)

        # ---- FORWARD PASS WITH AMP ----
        # autocast automatically uses float16 where safe
        # float16 uses half the memory and is faster on modern GPUs
        # float32 is used where precision is critical
        with autocast():
            # Forward pass: images → model → predictions
            outputs = model(images)

            # Calculate loss: how wrong were we?
            # criterion compares outputs (predictions) to labels (truth)
            loss = criterion(outputs, labels)

        # ---- BACKWARD PASS ----
        # scaler.scale(loss) scales loss to prevent underflow with float16
        # .backward() computes gradients for all parameters
        # Gradients tell each weight: "change in this direction"
        scaler.scale(loss).backward()

        # ---- GRADIENT CLIPPING ----
        # Prevents "exploding gradients" — when gradients become huge
        # and make training unstable
        # Clips gradient norm to max value of 1.0
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ---- UPDATE WEIGHTS ----
        # scaler.step(optimizer) updates model weights based on gradients
        # scaler.update() adjusts scale factor for next iteration
        scaler.step(optimizer)
        scaler.update()

        # ---- TRACK METRICS ----
        total_loss += loss.item()  # .item() converts tensor → python float

        # Get predicted class: argmax finds index of highest score
        # dim=1 means: find max across the 15 class scores
        _, predicted = torch.max(outputs, dim=1)

        # Count correct predictions in this batch
        correct = (predicted == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)  # batch size

        # Update progress bar display
        current_acc = total_correct / total_samples * 100
        current_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            'loss': f'{current_loss:.3f}',
            'acc': f'{current_acc:.1f}%'
        })

    # Calculate epoch averages
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = total_correct / total_samples * 100

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    """
    Evaluates model on validation data.

    Validation = checking performance on data the model
    has NEVER been trained on. This tells us if the model
    is actually learning or just memorizing training data.

    Args:
        model: our PlantDiseaseModel
        val_loader: DataLoader for validation data
        criterion: loss function
        device: cuda or cpu
        epoch: current epoch number

    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """

    # Set model to EVALUATION mode
    # This disables:
    # - Dropout (use all neurons for prediction)
    # - BatchNorm uses running statistics
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch:2d} [Val]  ",
        leave=False,
        ncols=100
    )

    # torch.no_grad() disables gradient computation
    # During validation we don't need gradients
    # This saves memory and speeds up evaluation
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass (no AMP needed for validation)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            current_acc = total_correct / total_samples * 100
            current_loss = total_loss / (total_samples / val_loader.batch_size)
            progress_bar.set_postfix({
                'loss': f'{current_loss:.3f}',
                'acc': f'{current_acc:.1f}%'
            })

    epoch_loss = total_loss / len(val_loader)
    epoch_acc = total_correct / total_samples * 100

    return epoch_loss, epoch_acc


# =============================================================
# SECTION 4: PLOT TRAINING CURVES
# =============================================================

def plot_training_curves(history: dict, save_path: str):
    """
    Plots loss and accuracy curves after training.

    These charts show:
    - Did the model learn? (loss going down)
    - Is it overfitting? (train acc >> val acc = bad)
    - When was the best epoch?

    Args:
        history: dictionary of metrics from MetricTracker
        save_path: where to save the chart image
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Results — Plant Disease Detection',
                 fontsize=14, fontweight='bold')

    # ---- Chart 1: Loss ----
    axes[0].plot(epochs, history['train_loss'],
                 'b-o', label='Train Loss', markersize=4)
    axes[0].plot(epochs, history['val_loss'],
                 'r-o', label='Val Loss', markersize=4)
    axes[0].set_title('Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---- Chart 2: Accuracy ----
    axes[1].plot(epochs, history['train_acc'],
                 'b-o', label='Train Accuracy', markersize=4)
    axes[1].plot(epochs, history['val_acc'],
                 'r-o', label='Val Accuracy', markersize=4)
    axes[1].set_title('Accuracy per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])

    # Mark best epoch with a star
    best_epoch = history['val_acc'].index(max(history['val_acc'])) + 1
    best_acc = max(history['val_acc'])
    axes[1].annotate(
        f'Best: {best_acc:.1f}%',
        xy=(best_epoch, best_acc),
        xytext=(best_epoch + 1, best_acc - 10),
        arrowprops=dict(arrowstyle='->', color='green'),
        color='green', fontweight='bold'
    )

    # ---- Chart 3: Learning Rate ----
    axes[2].plot(epochs, history['lr'], 'g-o', markersize=4)
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_yscale('log')  # log scale for LR
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Training curves saved to: {save_path}")


# =============================================================
# SECTION 5: MAIN TRAINING FUNCTION
# =============================================================

def main():
    """
    Main function — runs the entire training pipeline.
    Called when you run: python ml_training/train.py
    """

    print("=" * 60)
    print("  PLANT DISEASE DETECTION — TRAINING")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(CONFIG['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # STEP 1: Build Data Pipeline
    # ==========================================================
    print("\n[STEP 1] Building data pipeline...")
    data = create_dataloaders(
        dataset_path=CONFIG['dataset_path'],
        image_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers']
    )

    train_loader = data['train_loader']
    val_loader = data['val_loader']
    class_to_idx = data['class_to_idx']
    num_classes = data['num_classes']

    print(f"\n  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    # ==========================================================
    # STEP 2: Create Model
    # ==========================================================
    print("\n[STEP 2] Creating model...")
    model, criterion, device = create_model(
        num_classes=num_classes,
        pretrained=CONFIG['pretrained'],
        dropout_rate=CONFIG['dropout_rate'],
        freeze_backbone=True  # start with frozen backbone
    )

    # ==========================================================
    # STEP 3: Create Optimizer
    # ==========================================================
    print("\n[STEP 3] Setting up optimizer...")

    # AdamW optimizer with discriminative learning rates
    # backbone gets small LR (it's pretrained, needs small nudges)
    # classifier gets larger LR (it's new, needs bigger updates)
    # But backbone is frozen initially so its LR doesn't matter yet
    optimizer = torch.optim.AdamW(
        model.get_parameter_groups(
            backbone_lr=CONFIG['backbone_lr'],
            head_lr=CONFIG['head_lr']
        ),
        weight_decay=CONFIG['weight_decay']
    )

    # ==========================================================
    # STEP 4: Create Learning Rate Scheduler
    # ==========================================================
    # CosineAnnealingLR smoothly reduces learning rate
    # Like cooling down a metal — start hot (big LR), cool slowly
    #
    # T_max = total epochs = one full cosine cycle
    # eta_min = minimum LR at end of training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['num_epochs'],
        eta_min=CONFIG['eta_min']
    )

    # ==========================================================
    # STEP 5: Setup Automatic Mixed Precision (AMP)
    # ==========================================================
    # AMP uses float16 for most operations (faster, less memory)
    # but float32 where precision matters
    # GradScaler handles the scaling to prevent underflow
    # enabled=True only works with CUDA (GPU)
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    print(f"  Optimizer     : AdamW")
    print(f"  Head LR       : {CONFIG['head_lr']}")
    print(f"  Backbone LR   : {CONFIG['backbone_lr']} (after unfreeze)")
    print(f"  Scheduler     : CosineAnnealingLR")
    print(f"  AMP           : {'Enabled' if use_amp else 'Disabled'}")
    print(f"  Early stopping: patience={CONFIG['patience']}")

    # ==========================================================
    # STEP 6: Training Loop
    # ==========================================================
    print(f"\n[STEP 4] Starting training for {CONFIG['num_epochs']} epochs...")
    print(f"  Backbone frozen for first {CONFIG['freeze_epochs']} epochs")
    print(f"  Then unfreezing for fine-tuning")
    print("=" * 60)

    tracker = MetricTracker()

    for epoch in range(1, CONFIG['num_epochs'] + 1):

        epoch_start = time.time()

        # ---- UNFREEZE BACKBONE AT RIGHT EPOCH ----
        # After freeze_epochs, unfreeze backbone for fine-tuning
        if epoch == CONFIG['freeze_epochs'] + 1:
            print(f"\n  Epoch {epoch}: Unfreezing backbone for fine-tuning!")
            model.unfreeze_last_n_blocks(n=3)

            # After unfreezing, re-create optimizer to include
            # backbone parameters with their own learning rate
            optimizer = torch.optim.AdamW(
                model.get_parameter_groups(
                    backbone_lr=CONFIG['backbone_lr'],
                    head_lr=CONFIG['head_lr']
                ),
                weight_decay=CONFIG['weight_decay']
            )
            # Reset scheduler for fine-tuning phase
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CONFIG['num_epochs'] - CONFIG['freeze_epochs'],
                eta_min=CONFIG['eta_min']
            )

        # ---- TRAIN ONE EPOCH ----
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion,
            optimizer, scaler, device, epoch
        )

        # ---- VALIDATE ----
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )

        # ---- UPDATE SCHEDULER ----
        # Step scheduler after each epoch
        # This reduces the learning rate gradually
        scheduler.step()

        # Get current learning rate for logging
        current_lr = optimizer.param_groups[-1]['lr']

        # ---- TRACK METRICS ----
        epoch_time = time.time() - epoch_start
        is_best = tracker.update(
            train_loss, train_acc,
            val_loss, val_acc,
            current_lr, epoch_time
        )

        # ---- PRINT EPOCH SUMMARY ----
        best_marker = " ← BEST!" if is_best else ""
        print(f"\nEpoch {epoch:2d}/{CONFIG['num_epochs']} "
              f"({epoch_time:.0f}s) | "
              f"Train: loss={train_loss:.3f} acc={train_acc:.1f}% | "
              f"Val: loss={val_loss:.3f} acc={val_acc:.1f}%"
              f"{best_marker}")

        # ---- SAVE CHECKPOINT ----
        # Always save if it's the best model
        if is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_accuracy=val_acc,
                val_loss=val_loss,
                class_to_idx=class_to_idx,
                save_path=str(checkpoint_dir / 'best_model.pt'),
                is_best=True
            )

        # Save periodic checkpoint every N epochs
        if epoch % CONFIG['save_every_n_epochs'] == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_accuracy=val_acc,
                val_loss=val_loss,
                class_to_idx=class_to_idx,
                save_path=str(checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
            )

        # ---- EARLY STOPPING CHECK ----
        if tracker.should_stop_early(CONFIG['patience']):
            print(f"\n  Early stopping triggered!")
            print(f"  No improvement for {CONFIG['patience']} epochs.")
            print(f"  Best was Epoch {tracker.best_epoch}: "
                  f"{tracker.best_val_acc:.1f}%")
            break

    # ==========================================================
    # STEP 7: Training Complete
    # ==========================================================
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Best Val Accuracy : {tracker.best_val_acc:.2f}%")
    print(f"  Best Epoch        : {tracker.best_epoch}")
    print(f"  Total Epochs Run  : {len(tracker.history['val_acc'])}")

    # Save training history
    history_path = str(checkpoint_dir / 'training_history.json')
    tracker.save(history_path)
    print(f"  History saved to  : {history_path}")

    # Plot and save training curves
    curves_path = 'data/training_curves.png'
    plot_training_curves(tracker.history, curves_path)

    print(f"\n  Model saved to    : {checkpoint_dir}/best_model.pt")
    print(f"  Ready for Day 6   : Training optimization!")


# =============================================================
# ENTRY POINT
# =============================================================
# This block only runs when you execute this file directly
# NOT when it's imported by another file
# (Python standard pattern)

if __name__ == '__main__':
    main()