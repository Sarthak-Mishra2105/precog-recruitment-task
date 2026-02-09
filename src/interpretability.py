"""
Interpretability Tools for Colored MNIST
==========================================
Activation Maximization and Grad-CAM implemented from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from pathlib import Path


# =============================================================================
# ACTIVATION MAXIMIZATION
# =============================================================================

def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Compute total variation loss for image regularization.
    Encourages spatial smoothness in generated images.
    
    Args:
        x: Image tensor of shape (1, C, H, W) or (C, H, W)
        
    Returns:
        Scalar total variation loss
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    
    # Differences between adjacent pixels
    diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]  # Vertical differences
    diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]  # Horizontal differences
    
    # Sum of absolute differences
    tv = torch.sum(torch.abs(diff_h)) + torch.sum(torch.abs(diff_w))
    return tv


def activation_maximize_logit(
    model: nn.Module,
    target_class: int,
    image_size: Tuple[int, int] = (28, 28),
    steps: int = 300,
    lr: float = 0.1,
    l2_reg: float = 1e-4,
    tv_reg: float = 1e-3,
    init_scale: float = 0.01,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Optimize an input image to maximize a specific class logit.
    
    Args:
        model: Neural network model
        target_class: Class index to maximize (0-9)
        image_size: (H, W) of the image
        steps: Number of optimization steps
        lr: Learning rate for Adam optimizer
        l2_reg: L2 regularization weight on image
        tv_reg: Total variation regularization weight
        init_scale: Scale of initial random noise
        device: Torch device
        
    Returns:
        Optimized image tensor of shape (3, H, W) in [0, 1]
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Initialize with small random noise centered around 0.5
    x = torch.randn(1, 3, *image_size, device=device) * init_scale + 0.5
    x = torch.clamp(x, 0, 1)
    x.requires_grad_(True)
    
    optimizer = torch.optim.Adam([x], lr=lr)
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(x)
        
        # Objective: maximize target class logit, minimize regularizers
        target_logit = logits[0, target_class]
        l2_loss = l2_reg * torch.sum(x ** 2)
        tv_loss = tv_reg * total_variation_loss(x)
        
        # We want to maximize logit, so negate it for minimization
        loss = -target_logit + l2_loss + tv_loss
        
        loss.backward()
        optimizer.step()
        
        # Clamp to valid range
        with torch.no_grad():
            x.clamp_(0, 1)
    
    return x.squeeze(0).detach().cpu()


def activation_maximize_channel(
    model: nn.Module,
    layer_name: str,
    channel_idx: int,
    image_size: Tuple[int, int] = (28, 28),
    steps: int = 300,
    lr: float = 0.1,
    l2_reg: float = 1e-4,
    tv_reg: float = 1e-3,
    init_scale: float = 0.01,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Optimize an input image to maximize a specific channel activation.
    
    Args:
        model: Neural network model
        layer_name: Name of the layer to target (e.g., 'conv3')
        channel_idx: Channel index to maximize
        image_size: (H, W) of the image
        steps: Number of optimization steps
        lr: Learning rate
        l2_reg: L2 regularization weight
        tv_reg: Total variation regularization weight
        init_scale: Scale of initial noise
        device: Torch device
        
    Returns:
        Optimized image tensor of shape (3, H, W) in [0, 1]
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Storage for activations
    activations = {}
    
    def hook_fn(module, input, output):
        activations['target'] = output
    
    # Register hook on target layer
    target_layer = dict(model.named_modules())[layer_name]
    handle = target_layer.register_forward_hook(hook_fn)
    
    # Initialize image
    x = torch.randn(1, 3, *image_size, device=device) * init_scale + 0.5
    x = torch.clamp(x, 0, 1)
    x.requires_grad_(True)
    
    optimizer = torch.optim.Adam([x], lr=lr)
    
    try:
        for step in range(steps):
            optimizer.zero_grad()
            
            # Forward pass (triggers hook)
            _ = model(x)
            
            # Get activation of target channel, maximize mean activation
            target_activation = activations['target'][0, channel_idx].mean()
            l2_loss = l2_reg * torch.sum(x ** 2)
            tv_loss = tv_reg * total_variation_loss(x)
            
            loss = -target_activation + l2_loss + tv_loss
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                x.clamp_(0, 1)
    finally:
        handle.remove()
    
    return x.squeeze(0).detach().cpu()


def generate_actmax_grid(
    model: nn.Module,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
    **kwargs
) -> Tuple[plt.Figure, List[torch.Tensor]]:
    """
    Generate activation maximization images for all classes.
    
    Returns:
        Tuple of (matplotlib figure, list of generated images)
    """
    if device is None:
        device = next(model.parameters()).device
    
    images = []
    for class_idx in range(num_classes):
        print(f"Generating activation max for class {class_idx}...")
        img = activation_maximize_logit(model, class_idx, device=device, **kwargs)
        images.append(img)
    
    # Create grid
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for idx, (ax, img) in enumerate(zip(axes, images)):
        img_np = img.permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img_np, 0, 1))
        ax.set_title(f"Class {idx}", fontsize=10)
        ax.axis('off')
    
    plt.suptitle("Activation Maximization (Class Logits)", fontsize=14)
    plt.tight_layout()
    
    return fig, images


# =============================================================================
# GRAD-CAM (From Scratch)
# =============================================================================

class GradCAM:
    """
    Grad-CAM implementation from scratch using forward and backward hooks.
    
    Usage:
        gradcam = GradCAM(model, target_layer='conv3')
        cam = gradcam(image, target_class)
        overlay = gradcam.overlay(image, cam)
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        """
        Args:
            model: Neural network model
            target_layer: Name of the target convolutional layer
        """
        self.model = model
        self.target_layer = target_layer
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find and hook the target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                return
        
        raise ValueError(f"Layer '{self.target_layer}' not found in model")
    
    def __call__(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute Grad-CAM for an image.
        
        Args:
            image: Input image tensor (3, H, W) or (1, 3, H, W)
            target_class: Class to compute CAM for (None = predicted class)
            
        Returns:
            CAM tensor of shape (H, W) normalized to [0, 1]
        """
        self.model.eval()
        
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Move to model device
        device = next(self.model.parameters()).device
        image = image.to(device)
        image.requires_grad_(True)
        
        # Forward pass
        logits = self.model(image)
        
        # Use predicted class if not specified
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        target_score = logits[0, target_class]
        target_score.backward()
        
        # Compute Grad-CAM
        # gradients: (1, C, H, W)
        # activations: (1, C, H, W)
        
        # Global average pooling of gradients to get channel weights
        # alpha_k = mean over H, W of gradients for channel k
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze()  # (H, W)
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.cpu()
    
    def upsample(
        self,
        cam: torch.Tensor,
        target_size: Tuple[int, int] = (28, 28),
    ) -> torch.Tensor:
        """Upsample CAM to target size using bilinear interpolation."""
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        cam = F.interpolate(cam, size=target_size, mode='bilinear', align_corners=False)
        return cam.squeeze()
    
    def overlay(
        self,
        image: torch.Tensor,
        cam: torch.Tensor,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay CAM heatmap on original image.
        
        Args:
            image: Original image (3, H, W) or (H, W, 3)
            cam: CAM of shape (H, W)
            alpha: Blending factor for heatmap
            
        Returns:
            Overlay image as numpy array (H, W, 3)
        """
        # Ensure image is (H, W, 3)
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image = image.numpy()
        
        # Upsample CAM to image size
        if cam.shape != image.shape[:2]:
            cam = self.upsample(cam, image.shape[:2])
        
        cam = cam.numpy() if isinstance(cam, torch.Tensor) else cam
        
        # Create heatmap using matplotlib colormap
        heatmap = plt.cm.jet(cam)[:, :, :3]  # RGB only, no alpha
        
        # Blend
        overlay = (1 - alpha) * image + alpha * heatmap
        return np.clip(overlay, 0, 1)


def visualize_gradcam_examples(
    model: nn.Module,
    dataset,
    target_layer: str,
    num_examples: int = 5,
    device: Optional[torch.device] = None,
    title: str = "Grad-CAM Examples",
) -> plt.Figure:
    """
    Visualize Grad-CAM for multiple examples from a dataset.
    
    Returns:
        Matplotlib figure with original, CAM, and overlay for each example
    """
    if device is None:
        device = next(model.parameters()).device
    
    gradcam = GradCAM(model, target_layer)
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(9, 3 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    model.eval()
    
    for idx in range(num_examples):
        img, label, color_id = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
        pred = logits.argmax(1).item()
        conf = F.softmax(logits, dim=1)[0, pred].item()
        
        # Compute Grad-CAM
        cam = gradcam(img.clone(), target_class=pred)
        cam_upsampled = gradcam.upsample(cam, (28, 28))
        overlay = gradcam.overlay(img, cam_upsampled)
        
        # Original image
        img_np = img.permute(1, 2, 0).numpy()
        axes[idx, 0].imshow(np.clip(img_np, 0, 1))
        axes[idx, 0].set_title(f"True: {label}", fontsize=9)
        axes[idx, 0].axis('off')
        
        # CAM
        axes[idx, 1].imshow(cam_upsampled.numpy(), cmap='jet')
        axes[idx, 1].set_title(f"Pred: {pred} ({conf:.0%})", fontsize=9)
        axes[idx, 1].axis('off')
        
        # Overlay
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title("Overlay", fontsize=9)
        axes[idx, 2].axis('off')
    
    axes[0, 0].set_title(f"Original\nTrue: {dataset[0][1]}", fontsize=9)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    return fig


def visualize_gradcam_recolor_sequence(
    model: nn.Module,
    gray_mask: torch.Tensor,
    colors: List[Tuple[float, float, float]],
    color_names: List[str],
    target_layer: str,
    true_label: int,
    device: Optional[torch.device] = None,
) -> plt.Figure:
    """
    Visualize Grad-CAM on a recolored sequence to show how CAM changes with color.
    """
    if device is None:
        device = next(model.parameters()).device
    
    gradcam = GradCAM(model, target_layer)
    model.eval()
    
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    
    for color_id in range(10):
        # Recolor image
        color_rgb = torch.tensor(colors[color_id], dtype=torch.float32).view(3, 1, 1)
        foreground = gray_mask * color_rgb
        background = (1 - gray_mask) * 0.2
        recolored = torch.clamp(foreground + background, 0, 1)
        
        # Get prediction
        with torch.no_grad():
            logits = model(recolored.unsqueeze(0).to(device))
        pred = logits.argmax(1).item()
        conf = F.softmax(logits, dim=1)[0, pred].item()
        
        # Compute Grad-CAM
        cam = gradcam(recolored.clone(), target_class=pred)
        cam_upsampled = gradcam.upsample(cam, (28, 28))
        overlay = gradcam.overlay(recolored, cam_upsampled)
        
        # Original
        axes[0, color_id].imshow(recolored.permute(1, 2, 0).numpy())
        axes[0, color_id].set_title(f"C{color_id}:{color_names[color_id][:3]}", fontsize=8)
        axes[0, color_id].axis('off')
        
        # CAM
        axes[1, color_id].imshow(cam_upsampled.numpy(), cmap='jet')
        title_color = 'green' if pred == true_label else 'red'
        axes[1, color_id].set_title(f"P:{pred}({conf:.0%})", fontsize=8, color=title_color)
        axes[1, color_id].axis('off')
        
        # Overlay
        axes[2, color_id].imshow(overlay)
        axes[2, color_id].axis('off')
    
    axes[0, 0].set_ylabel("Image", fontsize=10)
    axes[1, 0].set_ylabel("CAM", fontsize=10)
    axes[2, 0].set_ylabel("Overlay", fontsize=10)
    
    plt.suptitle(f"Grad-CAM Recolor Sequence (True Label: {true_label})", fontsize=12)
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    print("Interpretability module loaded successfully!")
