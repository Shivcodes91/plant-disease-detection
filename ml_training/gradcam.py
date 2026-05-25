# =============================================================
# ml_training/gradcam.py
#
# Grad-CAM: Gradient-weighted Class Activation Mapping
#
# WHAT IT DOES:
# During a forward pass, we ask the model:
# "Which parts of this image most influenced your prediction?"
#
# HOW IT WORKS:
# 1. Run image through model → get prediction
# 2. Backpropagate gradients to the LAST conv layer
# 3. Average gradients → importance weights per channel
# 4. Multiply weights × feature maps → activation map
# 5. Resize activation map to image size
# 6. Overlay as colored heatmap on original image
#
# Red = model focused here a LOT
# Blue = model barely looked here
# =============================================================

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, List, Optional
import json
import sys

sys.path.append(str(Path(__file__).parent))
from model import PlantDiseaseModel, load_checkpoint
from dataset import get_val_transforms


# =============================================================
# SECTION 1: GRAD-CAM IMPLEMENTATION
# =============================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Hooks into the model's last convolutional layer
    and captures gradients + activations during forward pass.

    HOW HOOKS WORK:
    PyTorch hooks are like "listeners" attached to a layer.
    - Forward hook: called when data passes THROUGH the layer
    - Backward hook: called when gradients pass THROUGH the layer
    We use these to capture intermediate values.
    """

    def __init__(self, model: PlantDiseaseModel, device: torch.device):
        """
        Initialize GradCAM.

        Args:
            model: trained PlantDiseaseModel
            device: cuda or cpu
        """
        self.model = model
        self.device = device
        self.model.eval()

        # Storage for captured values
        self.gradients = None    # gradients from backward pass
        self.activations = None  # feature maps from forward pass

        # ---- FIND TARGET LAYER ----
        # We hook into the last convolutional block of EfficientNet
        # This layer has the richest spatial information
        # Earlier layers = edges/textures, Later layers = objects/parts
        #
        # For EfficientNet-B4, the target is the last conv block
        # backbone.blocks[-1] = last block
        # backbone.blocks[-1][-1] = last sub-block in that block
        self.target_layer = model.backbone.blocks[-1][-1]

        # ---- REGISTER HOOKS ----
        # register_forward_hook: captures output of layer during forward pass
        # register_full_backward_hook: captures gradients during backward pass
        self.forward_hook = self.target_layer.register_forward_hook(
            self._save_activation
        )
        self.backward_hook = self.target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        """
        Forward hook — saves layer output (feature maps).
        Called automatically during model.forward()

        Args:
            module: the layer
            input: input to the layer
            output: output of the layer (feature maps)
        """
        # Detach from computation graph — we only need values
        # clone() makes a copy so it's not modified later
        self.activations = output.detach().clone()

    def _save_gradient(self, module, grad_input, grad_output):
        """
        Backward hook — saves gradients.
        Called automatically during loss.backward()

        Args:
            module: the layer
            grad_input: gradients flowing INTO the layer
            grad_output: gradients flowing OUT OF the layer
        """
        # grad_output[0] = gradient of loss w.r.t. layer output
        self.gradients = grad_output[0].detach().clone()

    def generate(self,
                 image_tensor: torch.Tensor,
                 class_idx: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for an image.

        Args:
            image_tensor: preprocessed image [3, H, W]
            class_idx: which class to explain (None = use predicted class)

        Returns:
            Tuple of (heatmap as numpy array, predicted_class, confidence)
        """
        # Add batch dimension: [3, H, W] → [1, 3, H, W]
        input_tensor = image_tensor.unsqueeze(0).to(self.device)

        # ---- FORWARD PASS ----
        # Enable gradients (we need them for backward pass)
        input_tensor.requires_grad_(True)

        # Run image through model
        output = self.model(input_tensor)

        # Get predicted class and confidence
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.item()
        confidence = confidence.item()

        # Use predicted class if none specified
        if class_idx is None:
            class_idx = predicted_class

        # ---- BACKWARD PASS ----
        # Zero all existing gradients
        self.model.zero_grad()

        # Create a one-hot vector for the target class
        # We backpropagate ONLY the score for that class
        # This tells us: "which features caused THIS class prediction?"
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0

        # Backpropagate — this fills self.gradients via our hook
        output.backward(gradient=one_hot, retain_graph=True)

        # ---- COMPUTE GRAD-CAM ----
        # self.gradients shape: [1, C, H, W]
        # C = channels, H,W = spatial dimensions
        gradients = self.gradients[0]    # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global average pooling of gradients
        # Average across spatial dimensions H,W → one weight per channel
        # This tells us: importance of each channel for the prediction
        weights = gradients.mean(dim=[1, 2])  # [C]

        # Weighted combination of activation maps
        # Each channel's feature map is weighted by its importance
        cam = torch.zeros(
            activations.shape[1],
            activations.shape[2],
            device=self.device
        )

        # For each channel, add: weight × activation_map
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU: keep only positive values
        # Negative = features that HURT the prediction (not useful)
        # Positive = features that HELP the prediction (important!)
        cam = F.relu(cam)

        # Convert to numpy for image processing
        cam = cam.cpu().numpy()

        # Normalize to 0-1 range
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, predicted_class, confidence

    def remove_hooks(self):
        """Remove hooks to free memory."""
        self.forward_hook.remove()
        self.backward_hook.remove()


# =============================================================
# SECTION 2: VISUALIZATION FUNCTIONS
# =============================================================

def apply_heatmap(original_image: np.ndarray,
                  cam: np.ndarray,
                  alpha: float = 0.4) -> np.ndarray:
    """
    Overlays Grad-CAM heatmap on the original image.

    Args:
        original_image: RGB image as numpy array [H, W, 3]
        cam: Grad-CAM map [h, w] (values 0-1)
        alpha: transparency of heatmap overlay (0=invisible, 1=opaque)

    Returns:
        Blended image as numpy array [H, W, 3]
    """
    h, w = original_image.shape[:2]

    # Resize CAM to match original image size
    # INTER_LINEAR = bilinear interpolation (smooth upscaling)
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert CAM to colormap
    # COLORMAP_JET: blue(low) → green → yellow → red(high)
    # This makes it intuitive: red = model focused here
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # cv2 uses BGR format, convert to RGB
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend original image and heatmap
    # result = (1-alpha) * original + alpha * heatmap
    # alpha=0.4 means: 60% original + 40% heatmap
    blended = cv2.addWeighted(
        original_image.astype(np.uint8), 1 - alpha,
        heatmap_rgb.astype(np.uint8), alpha,
        0
    )

    return blended


def visualize_prediction(image_path: str,
                          model: PlantDiseaseModel,
                          idx_to_class: dict,
                          device: torch.device,
                          save_path: str = None) -> dict:
    """
    Full pipeline: image → prediction → Grad-CAM → visualization.

    Creates a 3-panel figure:
    Panel 1: Original leaf image
    Panel 2: Grad-CAM heatmap only
    Panel 3: Original + heatmap overlay

    Args:
        image_path: path to leaf image file
        model: trained model
        idx_to_class: index to class name mapping
        device: cuda or cpu
        save_path: where to save the figure

    Returns:
        Dictionary with prediction results
    """

    # ---- LOAD AND PREPROCESS IMAGE ----
    original_image = np.array(Image.open(image_path).convert('RGB'))

    # Apply same transforms used during training
    transform = get_val_transforms(224)
    image_tensor = transform(image=original_image)['image']

    # ---- GENERATE GRAD-CAM ----
    gradcam = GradCAM(model, device)
    cam, predicted_class, confidence = gradcam.generate(image_tensor)
    gradcam.remove_hooks()

    # ---- GET CLASS INFO ----
    class_name = idx_to_class[predicted_class]
    is_healthy = 'healthy' in class_name.lower()

    # Clean up name for display
    display_name = (class_name
                    .replace('___', ' ')
                    .replace('__', ' ')
                    .replace('_', ' '))

    # ---- CREATE VISUALIZATION ----
    # Resize original for display
    original_display = cv2.resize(original_image, (224, 224))

    # Apply heatmap overlay
    overlay = apply_heatmap(original_display, cam)

    # Create colormap image (heatmap alone)
    cam_resized = cv2.resize(cam, (224, 224))
    cam_colored = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )
    cam_colored_rgb = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    status_color = '#27ae60' if is_healthy else '#e74c3c'
    status_text = 'HEALTHY' if is_healthy else 'DISEASED'

    fig.suptitle(
        f'Grad-CAM Analysis: {display_name}\n'
        f'Prediction: {status_text} | Confidence: {confidence*100:.1f}%',
        fontsize=13, fontweight='bold', color=status_color
    )

    # Panel 1: Original
    axes[0].imshow(original_display)
    axes[0].set_title('Original Image', fontsize=11)
    axes[0].axis('off')

    # Panel 2: Heatmap only
    axes[1].imshow(cam_colored_rgb)
    axes[1].set_title('Grad-CAM Heatmap\n(Red = Model focused here)',
                       fontsize=11)
    axes[1].axis('off')

    # Panel 3: Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay\n(Disease region highlighted)',
                       fontsize=11)
    axes[2].axis('off')

    # Add colorbar to show heatmap scale
    sm = plt.cm.ScalarMappable(cmap='jet',
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Attention Intensity', fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Medium', 'High'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.show()

    return {
        'class_name': class_name,
        'display_name': display_name,
        'confidence': round(confidence * 100, 2),
        'is_healthy': is_healthy,
        'predicted_class_idx': predicted_class
    }


def visualize_multiple(dataset_path: str,
                        model: PlantDiseaseModel,
                        idx_to_class: dict,
                        device: torch.device,
                        num_samples: int = 6,
                        save_path: str = None):
    """
    Visualizes Grad-CAM for multiple images from different classes.
    Shows a grid: each row = one disease class.

    Args:
        dataset_path: path to PlantVillage dataset
        model: trained model
        idx_to_class: index to class name mapping
        device: cuda or cpu
        num_samples: how many classes to visualize
        save_path: where to save the grid
    """
    dataset_path = Path(dataset_path)
    transform = get_val_transforms(224)

    # Pick num_samples random classes
    all_classes = list(idx_to_class.items())
    selected = np.random.choice(len(all_classes),
                                 size=min(num_samples, len(all_classes)),
                                 replace=False)

    fig, axes = plt.subplots(num_samples, 3,
                              figsize=(12, num_samples * 4))
    fig.suptitle('Grad-CAM Analysis — Multiple Disease Classes',
                  fontsize=14, fontweight='bold')

    for row, class_idx in enumerate(selected):
        idx, class_name = all_classes[class_idx]

        # Get a sample image from this class
        class_folder = dataset_path / class_name
        images = (list(class_folder.glob('*.jpg')) +
                  list(class_folder.glob('*.JPG')))

        if not images:
            continue

        # Pick a random image
        img_path = np.random.choice(images)
        original = np.array(Image.open(img_path).convert('RGB'))
        original_resized = cv2.resize(original, (224, 224))

        # Generate Grad-CAM
        img_tensor = transform(image=original)['image']
        gradcam = GradCAM(model, device)
        cam, pred_class, conf = gradcam.generate(img_tensor)
        gradcam.remove_hooks()

        # Create overlay
        overlay = apply_heatmap(original_resized, cam)

        # Display name
        display = (class_name.replace('___', ' ')
                   .replace('__', ' ')
                   .replace('_', ' '))
        is_healthy = 'healthy' in class_name.lower()
        color = '#27ae60' if is_healthy else '#e74c3c'

        # Plot row
        axes[row, 0].imshow(original_resized)
        axes[row, 0].set_ylabel(display, fontsize=8,
                                 color=color, fontweight='bold')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # Heatmap
        cam_resized = cv2.resize(cam, (224, 224))
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized),
                                         cv2.COLORMAP_JET)
        axes[row, 1].imshow(cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB))
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
        axes[row, 1].set_title(f'Conf: {conf*100:.1f}%', fontsize=8)

        # Overlay
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])

    # Column headers
    axes[0, 0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Grad-CAM', fontsize=11, fontweight='bold')
    axes[0, 2].set_title('Overlay', fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"✓ Grid saved: {save_path}")

    plt.show()


# =============================================================
# SECTION 3: TEST SCRIPT
# =============================================================

if __name__ == '__main__':

    print("=" * 55)
    print("  DAY 6 — GRAD-CAM VISUALIZATION TEST")
    print("=" * 55)

    # ---- LOAD MODEL ----
    print("\n[1/3] Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, checkpoint = load_checkpoint(
        'backend/models/best_model.pt',
        device=device
    )

    # Load class mapping
    with open('data/class_mapping.json', 'r') as f:
        mapping = json.load(f)

    idx_to_class = {
        int(k): v for k, v in mapping['idx_to_class'].items()
    }
    class_to_idx = mapping['class_to_idx']

    print(f"✓ Model loaded | Val Accuracy: {checkpoint['val_accuracy']:.2f}%")
    print(f"✓ Device: {device}")

    # ---- SINGLE IMAGE TEST ----
    print("\n[2/3] Testing on single diseased leaf image...")

    dataset_path = Path('data/raw/PlantVillage/PlantVillage')

    # Test on a few different disease classes
    test_classes = [
        'Tomato_Early_blight',
        'Potato___Late_blight',
        'Pepper__bell___Bacterial_spot'
    ]

    for class_name in test_classes:
        class_folder = dataset_path / class_name
        images = list(class_folder.glob('*.jpg'))[:1

        ]
        if not images:
            continue

        print(f"\n  Testing: {class_name}")
        result = visualize_prediction(
            image_path=str(images[0]),
            model=model,
            idx_to_class=idx_to_class,
            device=device,
            save_path=f'data/gradcam_{class_name}.png'
        )
        print(f"  Predicted: {result['display_name']} "
              f"({result['confidence']}% confident)")

    # ---- MULTI CLASS GRID ----
    print("\n[3/3] Generating multi-class Grad-CAM grid...")
    visualize_multiple(
        dataset_path=str(dataset_path),
        model=model,
        idx_to_class=idx_to_class,
        device=device,
        num_samples=6,
        save_path='data/gradcam_grid.png'
    )

    print("\n" + "=" * 55)
    print("  GRAD-CAM COMPLETE!")
    print("=" * 55)
    print("""
Grad-CAM visualizations saved to data/ folder:
  • gradcam_Tomato_Early_blight.png
  • gradcam_Potato___Late_blight.png
  • gradcam_Pepper__bell___Bacterial_spot.png
  • gradcam_grid.png

Next: Setting up MLflow experiment tracking!
    """)