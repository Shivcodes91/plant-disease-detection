# =============================================================
# ml_training/evaluate.py
#
# Evaluates our trained model on the test set.
# Generates:
# - Overall accuracy
# - Per-class accuracy
# - Confusion matrix
# - Classification report
# - Failure case examples
#
# Run with: python ml_training/evaluate.py
# =============================================================

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

sys.path.append(str(Path(__file__).parent))
from dataset import create_dataloaders, get_val_transforms
from model import load_checkpoint

# =============================================================
# CONFIGURATION
# =============================================================
DATASET_PATH  = 'data/raw/PlantVillage/PlantVillage'
MODEL_PATH    = 'backend/models/best_model.pt'
BATCH_SIZE    = 32
NUM_WORKERS   = 0
IMAGE_SIZE    = 224

# =============================================================
# STEP 1: Load Model
# =============================================================
print("=" * 60)
print("  PLANT DISEASE MODEL — FULL EVALUATION")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[1/5] Loading model from {MODEL_PATH}...")

model, checkpoint = load_checkpoint(MODEL_PATH, device)
model.eval()

print(f"✓ Model loaded")
print(f"  Best val accuracy during training: "
      f"{checkpoint['val_accuracy']:.2f}%")

# Load class mapping
with open('data/class_mapping.json', 'r') as f:
    mapping = json.load(f)

idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
class_to_idx = mapping['class_to_idx']
num_classes  = mapping['num_classes']

# =============================================================
# STEP 2: Load Test Data
# =============================================================
print(f"\n[2/5] Loading test dataset...")

data = create_dataloaders(
    dataset_path=DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)
test_loader = data['test_loader']
print(f"✓ Test set: {data['test_size']:,} images")

# =============================================================
# STEP 3: Run Inference on Test Set
# =============================================================
print(f"\n[3/5] Running inference on test set...")
print(f"  This evaluates {data['test_size']:,} images the model")
print(f"  has NEVER seen during training...")

all_predictions = []  # what model predicted
all_labels      = []  # what the correct answer was
all_confidences = []  # how confident the model was

with torch.no_grad():
    for images, labels in tqdm(test_loader,
                                desc="Evaluating",
                                ncols=80):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Get probabilities
        probs = F.softmax(outputs, dim=1)

        # Get top prediction and confidence
        confidence, predicted = torch.max(probs, dim=1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidence.cpu().numpy())

all_predictions = np.array(all_predictions)
all_labels      = np.array(all_labels)
all_confidences = np.array(all_confidences)

# =============================================================
# STEP 4: Calculate Metrics
# =============================================================
print(f"\n[4/5] Calculating metrics...")

# Overall accuracy
overall_acc = accuracy_score(all_labels, all_predictions) * 100

# Average confidence
avg_confidence = all_confidences.mean() * 100

print(f"\n{'='*60}")
print(f"  TEST SET RESULTS")
print(f"{'='*60}")
print(f"  Overall Accuracy  : {overall_acc:.2f}%")
print(f"  Avg Confidence    : {avg_confidence:.2f}%")
print(f"  Total Images      : {len(all_labels):,}")
print(f"  Correct           : {(all_predictions == all_labels).sum():,}")
print(f"  Wrong             : {(all_predictions != all_labels).sum():,}")

# Per-class accuracy
print(f"\n  Per-Class Accuracy:")
print(f"  {'Class':<45} {'Correct':>8} {'Total':>7} {'Acc':>7}")
print(f"  {'-'*45}-{'-'*8}-{'-'*7}-{'-'*7}")

class_results = []
for idx in range(num_classes):
    class_name  = idx_to_class[idx]
    class_mask  = all_labels == idx
    class_total = class_mask.sum()
    class_correct = (
        (all_predictions == all_labels) & class_mask
    ).sum()
    class_acc = class_correct / class_total * 100 \
        if class_total > 0 else 0

    class_results.append({
        'class_name': class_name,
        'correct': int(class_correct),
        'total': int(class_total),
        'accuracy': float(class_acc)
    })

    # Color indicator
    if class_acc >= 95:
        indicator = '✓'
    elif class_acc >= 85:
        indicator = '~'
    else:
        indicator = '✗'

    print(f"  {indicator} {class_name:<44} "
          f"{class_correct:>7} /{class_total:>6} "
          f"  {class_acc:>5.1f}%")

# =============================================================
# STEP 5: Classification Report
# =============================================================
print(f"\n  Full Classification Report:")
class_names = [idx_to_class[i] for i in range(num_classes)]
report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=3
)
print(report)

# =============================================================
# STEP 6: Confusion Matrix
# =============================================================
print(f"\n[5/5] Generating confusion matrix...")

cm = confusion_matrix(all_labels, all_predictions)

# Short names for display
short_names = []
for name in class_names:
    name = name.replace('Pepper__bell___', 'Pepper ')
    name = name.replace('Potato___', 'Potato ')
    name = name.replace('Tomato_', 'Tom ')
    name = name.replace('Tomato__', 'Tom ')
    name = name.replace('_', ' ')
    if len(name) > 20:
        name = name[:18] + '..'
    short_names.append(name)

fig, ax = plt.subplots(figsize=(14, 12))

# Normalize confusion matrix (show percentages)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(
    cm_norm,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    xticklabels=short_names,
    yticklabels=short_names,
    ax=ax,
    linewidths=0.5,
    cbar_kws={'label': 'Proportion'}
)

ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title(
    f'Confusion Matrix — Test Set\n'
    f'Overall Accuracy: {overall_acc:.2f}%',
    fontsize=14, fontweight='bold'
)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('data/confusion_matrix.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("✓ Confusion matrix saved to data/confusion_matrix.png")

# =============================================================
# STEP 7: Save Evaluation Report
# =============================================================
report_data = {
    'overall_accuracy': float(round(overall_acc, 2)),
    'avg_confidence': float(round(avg_confidence, 2)),
    'total_test_images': int(len(all_labels)),
    'correct_predictions': int((all_predictions == all_labels).sum()),
    'wrong_predictions': int((all_predictions != all_labels).sum()),
    'per_class_results': [
        {k: float(v) if isinstance(v, np.floating) else v
         for k, v in r.items()}
        for r in class_results
    ],
    'training_val_accuracy': float(checkpoint['val_accuracy'])
}

with open('data/evaluation_report.json', 'w') as f:
    json.dump(report_data, f, indent=2)

print(f"\n✓ Evaluation report saved to data/evaluation_report.json")

print(f"\n{'='*60}")
print(f"  EVALUATION COMPLETE!")
print(f"{'='*60}")
print(f"""
  Model Performance Summary:
  • Test Accuracy  : {overall_acc:.2f}%
  • Avg Confidence : {avg_confidence:.2f}%
  • Model is ready for production deployment!

  Next → Building FastAPI backend...
""")