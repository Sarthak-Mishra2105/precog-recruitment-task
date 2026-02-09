#!/usr/bin/env python
"""
Smoke Test for Colored MNIST CV Task
=====================================

A lightweight sanity check that verifies all imports work and core
functionality runs without errors. Completes in <1 minute on CPU.

Usage:
    python scripts/smoke_test.py

Tests:
1. Import all src modules
2. Create dataset and dataloader (64 samples)
3. Run forward pass
4. Run one training step
5. Run Grad-CAM on a single sample
6. Run 2 steps of targeted PGD

If all tests pass, the codebase is ready for full experiments.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


def test_imports():
    """Test 1: Verify all module imports."""
    print("[Test 1] Importing modules...")
    
    from data_colored_mnist import ColoredMNIST, COLORS, compute_dominant_rate
    from models import SimpleCNN, SimpleCNNWithFeatures
    from train_utils import train_epoch, evaluate, compute_confusion_matrix
    from interpretability import GradCAM, activation_maximize_logit
    from debias import GRLModel, train_consistency_epoch, recolor_augment
    from attacks import targeted_pgd_linf, visualize_attack
    from utils import set_seed, get_device, print_env_info
    
    print("  [OK] All imports successful")
    return True


def test_dataset():
    """Test 2: Create dataset and verify structure."""
    print("[Test 2] Creating dataset...")
    
    from data_colored_mnist import ColoredMNIST
    
    ds = ColoredMNIST(root=str(project_root / "data"), split="train", seed=42)
    img, label, color_id = ds[0]
    
    assert img.shape == (3, 28, 28), f"Image shape wrong: {img.shape}"
    assert 0 <= label <= 9, f"Label out of range: {label}"
    assert 0 <= color_id <= 9, f"Color ID out of range: {color_id}"
    assert img.min() >= 0 and img.max() <= 1, f"Image values out of [0,1]"
    
    print(f"  [OK] Dataset: {len(ds)} samples, img shape: {img.shape}")
    return ds


def test_forward_pass(ds):
    """Test 3: Run forward pass through model."""
    print("[Test 3] Forward pass...")
    
    from models import SimpleCNN
    from utils import get_device
    
    device = get_device()
    model = SimpleCNN(num_classes=10).to(device)
    
    # Single sample
    img, label, _ = ds[0]
    x = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (1, 10), f"Logits shape wrong: {logits.shape}"
    
    print(f"  [OK] Forward pass: input {x.shape} -> output {logits.shape}")
    return model, device


def test_training_step(ds, model, device):
    """Test 4: Run one training step."""
    print("[Test 4] Training step...")
    
    subset = Subset(ds, list(range(64)))
    loader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=0)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for batch in loader:
        imgs, labels, _ = batch
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        break  # Just one batch
    
    print(f"  [OK] Training step: loss = {loss.item():.4f}")
    return True


def test_gradcam(ds, model, device):
    """Test 5: Run Grad-CAM on single sample."""
    print("[Test 5] Grad-CAM...")
    
    from interpretability import GradCAM
    
    model.eval()
    gradcam = GradCAM(model, target_layer="conv3.0")
    
    img, label, _ = ds[0]
    x = img.to(device)
    
    cam = gradcam(x, target_class=label)
    
    # CAM is at feature map resolution (7x7), not input resolution
    assert len(cam.shape) == 2, f"CAM should be 2D: {cam.shape}"
    assert cam.shape[0] == cam.shape[1], f"CAM should be square: {cam.shape}"
    
    # Test upsampling
    cam_up = gradcam.upsample(cam, (28, 28))
    assert cam_up.shape == (28, 28), f"Upsampled CAM shape wrong: {cam_up.shape}"
    
    print(f"  [OK] Grad-CAM: raw {cam.shape} -> upsampled {cam_up.shape}")
    return True


def test_pgd_attack(ds, model, device):
    """Test 6: Run 2 steps of targeted PGD."""
    print("[Test 6] Targeted PGD (2 steps)...")
    
    from attacks import targeted_pgd_linf
    
    model.eval()
    
    # Find a digit that's not the target
    target_class = 3
    for i in range(len(ds)):
        img, label, _ = ds[i]
        if label != target_class:
            break
    
    x = img.unsqueeze(0).to(device)
    
    x_adv, info = targeted_pgd_linf(
        model, x, target_class=target_class,
        eps=0.05, steps=2, step_size=0.01,
        random_start=False, early_stop_conf=0.99
    )
    
    assert x_adv.shape == x.shape, "Adversarial shape mismatch"
    assert info['linf_delta'] <= 0.05 + 1e-6, f"Epsilon violated: {info['linf_delta']}"
    
    print(f"  [OK] PGD: Linf delta = {info['linf_delta']:.4f}")
    return True


def main():
    print("=" * 60)
    print("SMOKE TEST - Colored MNIST CV Task")
    print("=" * 60 + "\n")
    
    from utils import set_seed
    set_seed(42)
    
    try:
        # Run all tests
        test_imports()
        ds = test_dataset()
        model, device = test_forward_pass(ds)
        test_training_step(ds, model, device)
        test_gradcam(ds, model, device)
        test_pgd_attack(ds, model, device)
        
        print("\n" + "=" * 60)
        print("[SUCCESS] All smoke tests passed!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n[FAILED] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
