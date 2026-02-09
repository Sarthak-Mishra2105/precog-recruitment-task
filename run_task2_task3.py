"""
Task 2-3: Train Cheater Baseline + Interpretability Experiments
================================================================
1. Train cheater model on small subset for dramatic collapse
2. Generate activation maximization grids
3. Generate Grad-CAM visualizations
"""
import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm

from data_colored_mnist import ColoredMNIST, COLORS, get_color_name
from models import SimpleCNN
from train_utils import train_epoch, evaluate, compute_confusion_matrix, recolor_proof, print_recolor_results
from interpretability import (
    generate_actmax_grid,
    GradCAM,
    visualize_gradcam_examples,
    visualize_gradcam_recolor_sequence,
)

# Setup
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

artifacts_dir = Path('artifacts')
artifacts_dir.mkdir(exist_ok=True)

# Load datasets
print("\nLoading datasets...")
train_ds = ColoredMNIST(root='./data', split='train', seed=SEED)
val_ds = ColoredMNIST(root='./data', split='val', seed=SEED)
test_ds = ColoredMNIST(root='./data', split='hard_test', seed=SEED)

val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

color_names = ["Red", "Green", "Blue", "Yellow", "Magenta", 
               "Cyan", "Orange", "Purple", "DarkGreen", "Gray"]

# =============================================================================
# PART A: Train Cheater Baseline
# =============================================================================
print("\n" + "="*60)
print("PART A: TRAINING CHEATER BASELINE")
print("="*60)

def train_cheater(n_samples, epochs=3):
    """Train a cheater model on small subset."""
    print(f"\n--- Training with N={n_samples} samples ---")
    
    # Create small deterministic subset
    indices = list(range(n_samples))
    subset = Subset(train_ds, indices)
    subset_loader = DataLoader(subset, batch_size=64, shuffle=True, num_workers=0)
    
    # Fresh model
    model = SimpleCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, subset_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}: Train={train_acc:.4f}, Val={val_acc:.4f}, HardTest={test_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'test_acc': test_acc,
                'n_samples': n_samples,
            }, artifacts_dir / f"baseline_cheater_N{n_samples}.pt")
    
    # Final eval
    train_loss, train_acc = evaluate(model, subset_loader, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"\nFinal results (N={n_samples}):")
    print(f"  Train (subset): {train_acc:.4f}")
    print(f"  Val:            {val_acc:.4f}")
    print(f"  Hard Test:      {test_acc:.4f}")
    
    return model, test_acc

# Try N=2000 first
cheater_model, test_acc_2000 = train_cheater(2000, epochs=5)

# If still >20%, try N=1000
if test_acc_2000 > 0.20:
    print("\nTest acc >20%, trying N=1000...")
    cheater_model, test_acc_1000 = train_cheater(1000, epochs=5)
    
    if test_acc_1000 > 0.20:
        print("\nTest acc still >20%, trying N=500...")
        cheater_model, test_acc_500 = train_cheater(500, epochs=5)

# Load best cheater model (lowest test acc)
print("\n--- Loading best cheater model ---")
best_checkpoint = None
best_test_acc = 1.0
for n in [500, 1000, 2000]:
    path = artifacts_dir / f"baseline_cheater_N{n}.pt"
    if path.exists():
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if ckpt.get('test_acc', 1.0) < best_test_acc:
            best_test_acc = ckpt.get('test_acc', 1.0)
            best_checkpoint = ckpt
            best_n = n

print(f"Best cheater: N={best_n}, test_acc={best_test_acc:.4f}")
cheater_model = SimpleCNN(num_classes=10).to(device)
cheater_model.load_state_dict(best_checkpoint['model_state_dict'])

# Confusion matrix for cheater
print("\nGenerating confusion matrix for cheater model...")
confusion = compute_confusion_matrix(cheater_model, test_loader, device)
print("Confusion Matrix (Hard Test):")
print(confusion)

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(confusion, cmap='Blues')
ax.figure.colorbar(im, ax=ax)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'Hard Test Confusion Matrix (Cheater N={best_n})')
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(confusion[i, j]), ha='center', va='center',
                color='white' if confusion[i, j] > confusion.max()/2 else 'black', fontsize=8)
plt.tight_layout()
plt.savefig(artifacts_dir / 'hard_test_confusion_cheater.png', dpi=150)
plt.close()
print(f"Saved: {artifacts_dir / 'hard_test_confusion_cheater.png'}")

# Recolor proofs for cheater
print("\nGenerating recolor proofs...")
for target_digit in [1, 7]:
    for i in range(len(test_ds)):
        _, label, _ = test_ds[i]
        if label == target_digit:
            img, label, original_color = test_ds[i]
            print(f"\nDigit {label} (original: {get_color_name(original_color)}):")
            
            proof = recolor_proof(cheater_model, img, label, COLORS, color_names, device, figsize=(15, 3))
            plt.savefig(artifacts_dir / f'recolor_proof_{label}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {artifacts_dir / f'recolor_proof_{label}.png'}")
            
            print_recolor_results(proof['results'], label)
            break

# =============================================================================
# PART B: Activation Maximization
# =============================================================================
print("\n" + "="*60)
print("PART B: ACTIVATION MAXIMIZATION")
print("="*60)

print("\nGenerating activation maximization grid for cheater model...")
fig, images = generate_actmax_grid(cheater_model, num_classes=10, device=device, steps=300)
plt.savefig(artifacts_dir / 'task2_actmax_logits_cheater.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {artifacts_dir / 'task2_actmax_logits_cheater.png'}")

# =============================================================================
# PART C: Grad-CAM
# =============================================================================
print("\n" + "="*60)
print("PART C: GRAD-CAM")
print("="*60)

# Find target layer (should be 'conv3' for SimpleCNN, but check sequential naming)
print("\nModel layers:")
for name, module in cheater_model.named_modules():
    if 'conv' in name.lower() or 'Conv' in str(type(module)):
        print(f"  {name}: {type(module).__name__}")

# The last conv layer in SimpleCNN is inside conv3 Sequential
# We need to target the actual Conv2d, which is conv3.0
target_layer = 'conv3.0'

# Grad-CAM examples from easy_val
print(f"\nGenerating Grad-CAM examples (target layer: {target_layer})...")
print("Easy Val examples:")
fig_val = visualize_gradcam_examples(
    cheater_model, val_ds, target_layer, num_examples=5,
    device=device, title="Grad-CAM: Easy Val (Cheater Model)"
)
plt.savefig(artifacts_dir / 'task3_gradcam_cheater_val.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {artifacts_dir / 'task3_gradcam_cheater_val.png'}")

# Grad-CAM examples from hard_test
print("Hard Test examples:")
fig_test = visualize_gradcam_examples(
    cheater_model, test_ds, target_layer, num_examples=5,
    device=device, title="Grad-CAM: Hard Test (Cheater Model)"
)
plt.savefig(artifacts_dir / 'task3_gradcam_cheater_test.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {artifacts_dir / 'task3_gradcam_cheater_test.png'}")

# Grad-CAM on recolor sequence
print("\nGenerating Grad-CAM recolor sequence...")
# Get a sample and extract grayscale mask
sample_img, sample_label, _ = test_ds[0]
gray_mask = sample_img.mean(dim=0, keepdim=True)
gray_mask = (gray_mask - gray_mask.min()) / (gray_mask.max() - gray_mask.min() + 1e-8)

fig_recolor = visualize_gradcam_recolor_sequence(
    cheater_model, gray_mask, COLORS, color_names, target_layer,
    true_label=sample_label, device=device
)
plt.savefig(artifacts_dir / 'task3_gradcam_recolor_sequence.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {artifacts_dir / 'task3_gradcam_recolor_sequence.png'}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nCheater model (N={best_n}):")
print(f"  Hard Test Accuracy: {best_test_acc:.4f} ({best_test_acc:.2%})")
print(f"\nArtifacts saved:")
for f in sorted(artifacts_dir.glob('*.png')):
    print(f"  - {f.name}")
print("\n=== COMPLETE ===")
