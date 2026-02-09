"""
Debiasing Strategies for Colored MNIST
=======================================
Strategy A: Color Consistency Training
Strategy B: Adversarial Color Removal (Gradient Reversal Layer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt


# =============================================================================
# COLORS (copy from data module to avoid circular imports)
# =============================================================================
COLORS = [
    (1.0, 0.0, 0.0),    # 0: Red
    (0.0, 1.0, 0.0),    # 1: Green
    (0.0, 0.0, 1.0),    # 2: Blue
    (1.0, 1.0, 0.0),    # 3: Yellow
    (1.0, 0.0, 1.0),    # 4: Magenta
    (0.0, 1.0, 1.0),    # 5: Cyan
    (1.0, 0.5, 0.0),    # 6: Orange
    (0.5, 0.0, 0.5),    # 7: Purple
    (0.0, 0.5, 0.0),    # 8: DarkGreen
    (0.5, 0.5, 0.5),    # 9: Gray
]


# =============================================================================
# RECOLOR AUGMENTATION
# =============================================================================

def extract_digit_mask(img: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """
    Extract a soft digit mask from a colored image.
    
    Args:
        img: Image tensor (3, H, W) in [0, 1]
        threshold: Threshold for foreground detection
        
    Returns:
        Mask tensor (1, H, W) in [0, 1]
    """
    # Convert to grayscale by taking max across channels (colored digit will be bright)
    gray = img.max(dim=0, keepdim=True)[0]
    
    # Normalize
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    
    # Soft threshold
    mask = torch.sigmoid((gray - threshold) * 10)  # Soft threshold
    
    return mask


def recolor_augment(
    img: torch.Tensor,
    new_color_id: Optional[int] = None,
    colors: List[Tuple[float, float, float]] = COLORS,
) -> Tuple[torch.Tensor, int]:
    """
    Recolor the digit stroke to a random or specified color.
    Keeps the textured background intact.
    
    Args:
        img: Image tensor (3, H, W) in [0, 1]
        new_color_id: Target color index (0-9), random if None
        colors: List of RGB color tuples
        
    Returns:
        Tuple of (recolored image, new color_id)
    """
    if new_color_id is None:
        new_color_id = np.random.randint(0, len(colors))
    
    # Extract digit mask
    mask = extract_digit_mask(img)  # (1, H, W)
    
    # Get new color
    color_rgb = torch.tensor(colors[new_color_id], dtype=img.dtype, device=img.device)
    color_rgb = color_rgb.view(3, 1, 1)
    
    # Create colored foreground (intensity modulated by mask)
    foreground = mask * color_rgb
    
    # Keep original background where mask is low
    background = (1 - mask) * img
    
    # Combine
    recolored = torch.clamp(foreground + background, 0, 1)
    
    return recolored, new_color_id


def recolor_augment_batch(
    images: torch.Tensor,
    new_color_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch version of recolor_augment.
    
    Args:
        images: Batch of images (B, 3, H, W)
        new_color_ids: Optional tensor of target color IDs (B,)
        
    Returns:
        Tuple of (recolored batch, color_ids tensor)
    """
    B = images.shape[0]
    
    if new_color_ids is None:
        new_color_ids = torch.randint(0, 10, (B,))
    
    recolored = []
    for i in range(B):
        rec, _ = recolor_augment(images[i], int(new_color_ids[i]))
        recolored.append(rec)
    
    return torch.stack(recolored), new_color_ids


# =============================================================================
# STRATEGY A: COLOR CONSISTENCY LOSS
# =============================================================================

def symmetric_kl_loss(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Symmetric KL divergence between two probability distributions.
    KL(p||q) + KL(q||p)
    
    Args:
        p: Probability tensor (B, C)
        q: Probability tensor (B, C)
        eps: Small constant for numerical stability
        
    Returns:
        Scalar loss
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    
    kl_pq = (p * (p.log() - q.log())).sum(dim=1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=1)
    
    return (kl_pq + kl_qp).mean()


def consistency_loss(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    """
    Compute consistency loss between two sets of logits.
    Uses symmetric KL divergence on softmax outputs.
    """
    p1 = F.softmax(logits1, dim=1)
    p2 = F.softmax(logits2, dim=1)
    return symmetric_kl_loss(p1, p2)


# =============================================================================
# STRATEGY B: GRADIENT REVERSAL LAYER
# =============================================================================

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer for adversarial training.
    Forward: identity
    Backward: multiply gradient by -lambda
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
        ctx.lambda_grl = lambda_grl
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_grl * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer module.
    """
    
    def __init__(self, lambda_grl: float = 1.0):
        super().__init__()
        self.lambda_grl = lambda_grl
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_grl)
    
    def set_lambda(self, lambda_grl: float):
        self.lambda_grl = lambda_grl


# =============================================================================
# DEBIASED MODEL ARCHITECTURES
# =============================================================================

class SimpleCNNBackbone(nn.Module):
    """
    SimpleCNN backbone that returns features instead of logits.
    """
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 7 -> 1
        )
        
        self.feature_dim = 128
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # (B, 128)
        return x


class GRLModel(nn.Module):
    """
    Model with gradient reversal for adversarial color removal.
    
    Architecture:
    - backbone: shared feature extractor
    - digit_head: classifier for digit (0-9)
    - color_head: classifier for color (0-9) with GRL
    """
    
    def __init__(self, lambda_grl: float = 1.0):
        super().__init__()
        
        self.backbone = SimpleCNNBackbone()
        
        # Digit classification head
        self.digit_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 10),
        )
        
        # Color classification head with GRL
        self.grl = GradientReversalLayer(lambda_grl)
        self.color_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 10),
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        return_color_logits: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Returns:
            Tuple of (digit_logits, color_logits or None)
        """
        features = self.backbone(x)
        digit_logits = self.digit_head(features)
        
        if return_color_logits:
            reversed_features = self.grl(features)
            color_logits = self.color_head(reversed_features)
            return digit_logits, color_logits
        else:
            return digit_logits, None
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward for inference (digit logits only)."""
        features = self.backbone(x)
        return self.digit_head(features)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_consistency_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lambda_cons: float = 1.0,
) -> Tuple[float, float]:
    """
    Train one epoch with color consistency loss.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, color_ids in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Create two recolored versions
        images1, _ = recolor_augment_batch(images)
        images2, _ = recolor_augment_batch(images)
        images1, images2 = images1.to(device), images2.to(device)
        
        optimizer.zero_grad()
        
        # Forward both versions
        logits1 = model(images1)
        logits2 = model(images2)
        
        # Supervised loss on both
        loss_ce = criterion(logits1, labels) + criterion(logits2, labels)
        
        # Consistency loss
        loss_cons = consistency_loss(logits1, logits2)
        
        # Total loss
        loss = loss_ce + lambda_cons * loss_cons
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, preds = logits1.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def train_grl_epoch(
    model: GRLModel,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    alpha_color: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Train one epoch with gradient reversal for color removal.
    
    Returns:
        Tuple of (total_loss, digit_accuracy, color_accuracy)
    """
    model.train()
    total_loss = 0.0
    digit_correct = 0
    color_correct = 0
    total = 0
    
    for images, labels, color_ids in loader:
        images = images.to(device)
        labels = labels.to(device)
        color_ids = color_ids.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        digit_logits, color_logits = model(images, return_color_logits=True)
        
        # Digit classification loss
        loss_digit = criterion(digit_logits, labels)
        
        # Color classification loss (adversarial via GRL)
        loss_color = criterion(color_logits, color_ids)
        
        # Total loss
        loss = loss_digit + alpha_color * loss_color
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
        _, digit_preds = digit_logits.max(1)
        digit_correct += (digit_preds == labels).sum().item()
        
        _, color_preds = color_logits.max(1)
        color_correct += (color_preds == color_ids).sum().item()
        
        total += labels.size(0)
    
    return total_loss / total, digit_correct / total, color_correct / total


def evaluate_model(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    is_grl: bool = False,
) -> Tuple[float, float]:
    """
    Evaluate model on a dataset.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, color_ids in loader:
            images, labels = images.to(device), labels.to(device)
            
            if is_grl:
                logits = model.predict(images)
            else:
                logits = model(images)
            
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            
            _, preds = logits.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / total, correct / total


def compute_confusion_matrix_debias(
    model: nn.Module,
    loader,
    device: torch.device,
    is_grl: bool = False,
) -> np.ndarray:
    """
    Compute confusion matrix for a debiased model.
    """
    model.eval()
    confusion = np.zeros((10, 10), dtype=np.int32)
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            
            if is_grl:
                logits = model.predict(images)
            else:
                logits = model(images)
            
            _, preds = logits.max(1)
            
            for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[true, pred] += 1
    
    return confusion


def plot_training_curves_debias(
    history: dict,
    title: str,
    save_path: Optional[str] = None,
):
    """Plot training curves for debiasing experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    # Accuracy
    ax1.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax1.plot(epochs, history['val_acc'], 'g-', label='Val')
    ax1.plot(epochs, history['test_acc'], 'r-', label='Hard Test')
    ax1.axhline(y=0.70, color='k', linestyle='--', alpha=0.5, label='Target (70%)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{title} - Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax2.plot(epochs, history['val_loss'], 'g-', label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'{title} - Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("Debias module loaded successfully!")
    
    # Test GRL
    grl = GradientReversalLayer(1.0)
    x = torch.randn(2, 10, requires_grad=True)
    y = grl(x)
    loss = y.sum()
    loss.backward()
    print(f"GRL test - input grad sign should be negative: {x.grad.mean().item():.4f}")
