"""
CNN Models for Colored MNIST
============================
Simple CNN and optional ResNet18 baseline for digit classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for 28x28 RGB images (Colored MNIST).
    
    Architecture:
    - Conv1: 3 -> 32 channels, 3x3, padding=1
    - Conv2: 32 -> 64 channels, 3x3, padding=1
    - Conv3: 64 -> 128 channels, 3x3, padding=1
    - Global Average Pooling
    - FC: 128 -> 10
    
    Each conv layer followed by BatchNorm, ReLU, and MaxPool2d(2).
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Conv blocks
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
            nn.MaxPool2d(2),  # 7 -> 3
        )
        
        # Global average pooling + FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        
        # Store intermediate activations for Grad-CAM
        self.gradients = None
        self.activations = None
    
    def save_gradient(self, grad):
        """Hook to save gradients for Grad-CAM."""
        self.gradients = grad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, 3, 28, 28)
            
        Returns:
            Logits of shape (N, 10)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Save activations for Grad-CAM (before GAP)
        self.activations = x
        
        # Register hook for gradients if training
        if x.requires_grad:
            x.register_hook(self.save_gradient)
        
        # GAP and FC
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature map before GAP (for interpretability)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class SimpleCNNWithFeatures(SimpleCNN):
    """
    SimpleCNN variant that also returns intermediate features.
    Useful for probing and interpretability experiments.
    """
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.conv1(x)
        x = self.conv2(x)
        features = self.conv3(x)
        
        self.activations = features
        
        if features.requires_grad:
            features.register_hook(self.save_gradient)
        
        pooled = self.gap(features)
        flat = pooled.view(pooled.size(0), -1)
        logits = self.fc(flat)
        
        if return_features:
            return logits, features
        return logits


def get_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    Get a ResNet18 model adapted for 28x28 input.
    
    Modifications:
    - First conv: 7x7 stride 2 -> 3x3 stride 1 (for small input)
    - Remove initial maxpool
    """
    from torchvision.models import resnet18
    
    model = resnet18(pretrained=pretrained)
    
    # Modify first conv for 28x28 input
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove maxpool (keep feature map size)
    model.maxpool = nn.Identity()
    
    # Modify final FC
    model.fc = nn.Linear(512, num_classes)
    
    return model


if __name__ == "__main__":
    # Quick test
    model = SimpleCNN()
    x = torch.randn(4, 3, 28, 28)
    out = model(x)
    print(f"SimpleCNN output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test ResNet18
    resnet = get_resnet18()
    out = resnet(x)
    print(f"ResNet18 output shape: {out.shape}")
    print(f"ResNet18 Parameters: {sum(p.numel() for p in resnet.parameters()):,}")
