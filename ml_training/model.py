# =============================================================
# ml_training/model.py
#
# This file defines our plant disease detection model.
#
# ARCHITECTURE: EfficientNet-B4 with custom classifier head
#
# HOW IT WORKS:
# 1. EfficientNet-B4 backbone: extracts features from images
#    (pretrained on ImageNet — already knows textures/shapes)
# 2. Custom classifier head: maps features → 15 disease classes
#
# TRANSFER LEARNING STRATEGY:
# - Phase 1 (early training): freeze backbone, train head only
# - Phase 2 (fine-tuning): unfreeze all layers, train everything
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

# =============================================================
# SECTION 1: MODEL ARCHITECTURE
# =============================================================

class PlantDiseaseModel(nn.Module):
    """
    Plant Disease Detection Model based on EfficientNet-B4.
    
    nn.Module is the base class for ALL PyTorch models.
    Every custom model must inherit from it.
    
    We override two methods:
    - __init__: define all layers
    - forward: define how data flows through layers
    """
    
    def __init__(self, 
                 num_classes: int = 15,
                 pretrained: bool = True,
                 dropout_rate: float = 0.3):
        """
        Initialize the model.
        
        Args:
            num_classes: number of disease classes (15 for our dataset)
            pretrained: use ImageNet pretrained weights (always True)
            dropout_rate: randomly zero out neurons to prevent overfitting
        """
        # MUST call parent class __init__ first
        # This sets up all the PyTorch internal machinery
        super(PlantDiseaseModel, self).__init__()
        
        self.num_classes = num_classes
        
        # ---- LOAD EFFICIENTNET-B4 BACKBONE ----
        # timm.create_model loads the architecture + pretrained weights
        # 
        # pretrained=True downloads weights trained on ImageNet
        # These weights encode knowledge about visual features
        #
        # num_classes=0 means: remove the original classifier head
        # We want just the feature extractor part
        # The backbone will output a feature vector, not predictions
        print(f"  Loading EfficientNet-B4 {'with pretrained weights' if pretrained else 'randomly initialized'}...")
        
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,        # remove original 1000-class head
            global_pool='avg'     # global average pooling at the end
        )
        
        # Get the number of features the backbone outputs
        # EfficientNet-B4 outputs 1792 features
        # We need to know this to build our classifier
        num_features = self.backbone.num_features
        print(f"  Backbone output features: {num_features}")
        
        # ---- CUSTOM CLASSIFIER HEAD ----
        # This replaces the original ImageNet classifier
        # It maps 1792 features → 15 disease classes
        #
        # We use a multi-layer classifier (not just one layer)
        # because diseases have complex visual patterns
        #
        # Architecture:
        # 1792 features
        #     ↓ Linear(1792 → 512)
        #     ↓ BatchNorm + ReLU + Dropout
        #     ↓ Linear(512 → 256)  
        #     ↓ BatchNorm + ReLU + Dropout
        #     ↓ Linear(256 → 15)
        #     ↓ 15 class scores
        
        self.classifier = nn.Sequential(
            
            # Layer 1: 1792 → 512
            nn.Linear(num_features, 512),
            # BatchNorm: normalizes activations, speeds up training
            # Keeps values in a good range so gradients flow well
            nn.BatchNorm1d(512),
            # ReLU: activation function — adds non-linearity
            # f(x) = max(0, x) — simple but powerful
            # Without activation functions, deep networks = just one linear layer
            nn.ReLU(inplace=True),
            # Dropout: randomly sets dropout_rate% of neurons to 0
            # Forces the network not to rely on any single neuron
            # Prevents overfitting (memorizing training data)
            nn.Dropout(p=dropout_rate),
            
            # Layer 2: 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),  # less dropout in deeper layers
            
            # Final Layer: 256 → num_classes (15)
            # No activation here — raw scores (logits)
            # Loss function handles the softmax internally
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights properly
        # Good initialization helps training start smoothly
        self._initialize_weights()
        
        print(f"  Classifier head: {num_features} → 512 → 256 → {num_classes}")
        print(f"  Dropout rate: {dropout_rate}")
    
    def _initialize_weights(self):
        """
        Initialize the weights of our custom classifier.
        
        WHY? Random initialization can be bad — too large values
        cause exploding gradients, too small cause vanishing gradients.
        
        Kaiming (He) initialization is designed for ReLU networks.
        It keeps the variance of activations consistent across layers.
        """
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                # Kaiming uniform initialization for linear layers
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    # Initialize bias to 0
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                # BatchNorm: weight=1, bias=0 is standard initialization
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: defines how data flows through the model.
        
        Called automatically when you do: output = model(input)
        
        Args:
            x: input tensor of shape [batch_size, 3, 224, 224]
               batch_size images, 3 RGB channels, 224x224 pixels
        
        Returns:
            logits: raw scores, shape [batch_size, num_classes]
                    (not probabilities yet — apply softmax for those)
        """
        # Step 1: Extract features using EfficientNet backbone
        # Input:  [batch, 3, 224, 224]  (images)
        # Output: [batch, 1792]          (feature vectors)
        features = self.backbone(x)
        
        # Step 2: Classify features using our custom head
        # Input:  [batch, 1792]  (feature vectors)
        # Output: [batch, 15]    (class scores)
        logits = self.classifier(features)
        
        return logits
    
    def freeze_backbone(self):
        """
        Freeze all backbone parameters.
        
        WHY? In early training, we only want to train our
        custom classifier head. The backbone already has good
        features from ImageNet — we don't want to destroy them
        with large gradient updates.
        
        Freezing means: don't update these weights during training.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        frozen = sum(p.numel() for p in self.backbone.parameters())
        print(f"  ✓ Backbone frozen: {frozen:,} parameters locked")
    
    def unfreeze_backbone(self):
        """
        Unfreeze backbone for fine-tuning.
        
        After classifier head is trained (few epochs),
        we unfreeze the backbone and train everything together
        with a very small learning rate.
        
        This is called 'fine-tuning' — the whole model
        adapts to plant disease features.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        unfrozen = sum(p.numel() for p in self.backbone.parameters())
        print(f"  ✓ Backbone unfrozen: {unfrozen:,} parameters trainable")
    
    def unfreeze_last_n_blocks(self, n: int = 3):
        """
        Partially unfreeze — only unfreeze last N blocks.
        
        A middle ground between fully frozen and fully unfrozen.
        Last blocks learn high-level features (most relevant
        for fine-grained disease recognition).
        
        Args:
            n: number of blocks to unfreeze from the end
        """
        # First freeze everything
        self.freeze_backbone()
        
        # Get all named blocks in the backbone
        blocks = list(self.backbone.named_parameters())
        total = len(blocks)
        
        # Unfreeze last n*50 parameters (approximate block size)
        cutoff = max(0, total - n * 50)
        unfrozen_count = 0
        
        for i, (name, param) in enumerate(blocks):
            if i >= cutoff:
                param.requires_grad = True
                unfrozen_count += param.numel()
        
        print(f"  ✓ Last {n} blocks unfrozen: {unfrozen_count:,} parameters trainable")
    
    def get_parameter_groups(self, 
                              backbone_lr: float = 1e-5,
                              head_lr: float = 1e-3):
        """
        Returns parameter groups with different learning rates.
        
        WHY DIFFERENT LEARNING RATES?
        - Backbone: already trained, needs small nudges (1e-5)
        - Classifier head: new, needs bigger updates (1e-3)
        
        This is called 'discriminative learning rates' — a key
        technique in transfer learning.
        
        Args:
            backbone_lr: learning rate for backbone (small)
            head_lr: learning rate for classifier (larger)
        
        Returns:
            List of parameter groups for optimizer
        """
        return [
            {
                'params': self.backbone.parameters(),
                'lr': backbone_lr,
                'name': 'backbone'
            },
            {
                'params': self.classifier.parameters(),
                'lr': head_lr,
                'name': 'classifier'
            }
        ]
    
    def get_model_info(self) -> Dict:
        """Returns a summary of model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() 
                               if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'model_size_mb': total_params * 4 / (1024 ** 2)  # float32 = 4 bytes
        }


# =============================================================
# SECTION 2: LOSS FUNCTION
# =============================================================

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.
    
    STANDARD CROSS ENTROPY:
    Target: [0, 0, 1, 0, 0]  (hard: 100% sure it's class 2)
    
    LABEL SMOOTHING:
    Target: [0.02, 0.02, 0.92, 0.02, 0.02]  (soft: 92% sure)
    
    WHY? Hard labels make the model overconfident.
    Label smoothing produces better-calibrated probabilities.
    Model says "94% Early Blight" instead of "99.9% Early Blight"
    — more honest and generalizes better.
    
    smoothing=0.1 is the standard value used in most papers.
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        # confidence = how much probability to give the true class
        self.confidence = 1.0 - smoothing
    
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate smoothed cross entropy loss.
        
        Args:
            predictions: model output [batch, num_classes]
            targets: true labels [batch] (class indices)
        
        Returns:
            Scalar loss value
        """
        # log_softmax = log(softmax(x))
        # More numerically stable than log(softmax(x)) separately
        log_probs = F.log_softmax(predictions, dim=-1)
        
        # Standard cross entropy part
        # nll_loss = negative log likelihood
        # This penalizes wrong predictions
        nll_loss = F.nll_loss(log_probs, targets, reduction='mean')
        
        # Smoothing part
        # Average log probability across all classes
        # This encourages uniform distribution (prevents overconfidence)
        smooth_loss = -log_probs.mean(dim=-1).mean()
        
        # Combine: weighted sum of standard and smooth loss
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        return loss


# =============================================================
# SECTION 3: MODEL FACTORY FUNCTION
# =============================================================

def create_model(num_classes: int = 15,
                 pretrained: bool = True,
                 dropout_rate: float = 0.3,
                 freeze_backbone: bool = True,
                 device: str = None) -> Tuple:
    """
    Factory function — creates everything needed for training.
    
    Returns model, loss function, and moves to correct device.
    
    Args:
        num_classes: number of disease classes
        pretrained: use ImageNet pretrained weights
        dropout_rate: dropout probability
        freeze_backbone: freeze backbone initially
        device: 'cuda' or 'cpu' (auto-detects if None)
    
    Returns:
        Tuple of (model, criterion, device)
    """
    print("=" * 55)
    print("  CREATING PLANT DISEASE MODEL")
    print("=" * 55)
    
    # Auto-detect device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        # Get GPU memory in GB
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  GPU Memory: {mem / 1024**3:.1f} GB")
    
    # Create model
    print(f"\n  Building model architecture...")
    model = PlantDiseaseModel(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    
    # Freeze backbone initially
    if freeze_backbone:
        print(f"\n  Freezing backbone for initial training...")
        model.freeze_backbone()
    
    # Move model to device (GPU or CPU)
    # .to(device) moves all model weights to that device
    model = model.to(device)
    
    # Create loss function
    criterion = LabelSmoothingLoss(
        num_classes=num_classes,
        smoothing=0.1
    )
    criterion = criterion.to(device)
    
    # Print model summary
    info = model.get_model_info()
    print(f"\n{'='*55}")
    print(f"  MODEL READY!")
    print(f"{'='*55}")
    print(f"  Total parameters    : {info['total_parameters']:,}")
    print(f"  Trainable parameters: {info['trainable_parameters']:,}")
    print(f"  Frozen parameters   : {info['frozen_parameters']:,}")
    print(f"  Model size          : {info['model_size_mb']:.1f} MB")
    print(f"  Loss function       : Label Smoothing CE (smoothing=0.1)")
    
    return model, criterion, device


# =============================================================
# SECTION 4: SAVE AND LOAD FUNCTIONS
# =============================================================

def save_checkpoint(model: PlantDiseaseModel,
                    optimizer,
                    epoch: int,
                    val_accuracy: float,
                    val_loss: float,
                    class_to_idx: Dict,
                    save_path: str,
                    is_best: bool = False):
    """
    Saves model checkpoint to disk.
    
    A checkpoint contains everything needed to resume training
    or make predictions later.
    
    Args:
        model: the trained model
        optimizer: optimizer state (for resuming training)
        epoch: current epoch number
        val_accuracy: validation accuracy at this checkpoint
        val_loss: validation loss at this checkpoint
        class_to_idx: class name to index mapping
        save_path: where to save the file
        is_best: if True, also saves as 'best_model.pt'
    """
    # Everything we want to save
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # state_dict() = dictionary of all model weights
        # This is what we need to restore the model later
        
        'optimizer_state_dict': optimizer.state_dict(),
        # Save optimizer state too — needed to resume training
        
        'val_accuracy': val_accuracy,
        'val_loss': val_loss,
        'class_to_idx': class_to_idx,
        'num_classes': model.num_classes,
        'model_config': {
            'num_classes': model.num_classes,
        }
    }
    
    # Save to disk
    # torch.save uses pickle to serialize the object
    torch.save(checkpoint, save_path)
    print(f"  ✓ Checkpoint saved: {save_path}")
    
    # If this is the best model so far, save a copy
    if is_best:
        best_path = Path(save_path).parent / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f"  ✓ Best model updated: {best_path}")


def load_checkpoint(checkpoint_path: str,
                    device: torch.device = None) -> Tuple:
    """
    Loads a saved model checkpoint.
    
    Args:
        checkpoint_path: path to .pt file
        device: device to load model on
    
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint from disk
    # map_location handles loading GPU model on CPU and vice versa
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate the model architecture
    num_classes = checkpoint['model_config']['num_classes']
    model = PlantDiseaseModel(
        num_classes=num_classes,
        pretrained=False  # don't download ImageNet weights
    )
    
    # Load the saved weights into the model
    # load_state_dict restores all the learned weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # set to evaluation mode
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Accuracy: {checkpoint['val_accuracy']:.2f}%")
    
    return model, checkpoint


# =============================================================
# SECTION 5: INFERENCE FUNCTION
# =============================================================

def predict_single_image(model: PlantDiseaseModel,
                          image_tensor: torch.Tensor,
                          idx_to_class: Dict,
                          device: torch.device,
                          top_k: int = 3) -> Dict:
    """
    Makes a prediction on a single image.
    
    This is what the API will call when a farmer uploads a photo.
    
    Args:
        model: trained PlantDiseaseModel
        image_tensor: preprocessed image tensor [3, 224, 224]
        idx_to_class: mapping from index to class name
        device: cuda or cpu
        top_k: return top K predictions
    
    Returns:
        Dictionary with predictions and confidence scores
    """
    model.eval()
    
    # Add batch dimension: [3, 224, 224] → [1, 3, 224, 224]
    # Models always expect a batch, even for single images
    image_batch = image_tensor.unsqueeze(0).to(device)
    
    # No gradient tracking needed for inference
    with torch.no_grad():
        # Forward pass
        logits = model(image_batch)
        
        # Convert logits to probabilities
        # softmax turns raw scores into probabilities (sum to 1)
        probabilities = F.softmax(logits, dim=1)
        
        # Get top-k predictions
        # topk returns (values, indices) of top k elements
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)
    
    # Convert to Python lists
    top_probs = top_probs[0].cpu().numpy().tolist()
    top_indices = top_indices[0].cpu().numpy().tolist()
    
    # Build result dictionary
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        class_name = idx_to_class[idx]
        
        # Parse plant and disease from class name
        parts = class_name.replace('___', '_').replace('__', '_').split('_')
        plant = parts[0]
        is_healthy = 'healthy' in class_name.lower()
        
        predictions.append({
            'rank': len(predictions) + 1,
            'class_name': class_name,
            'plant': plant,
            'is_healthy': is_healthy,
            'confidence': round(prob * 100, 2),  # as percentage
            'confidence_raw': prob
        })
    
    return {
        'top_prediction': predictions[0],
        'all_predictions': predictions,
        'is_confident': predictions[0]['confidence'] > 60.0
        # If confidence < 60%, we're not sure enough to show remedy
    }