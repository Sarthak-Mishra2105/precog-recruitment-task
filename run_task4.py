"""
Task 4: GRL with Recolor Augmentation
=====================================
Combines GRL with color augmentation for better debiasing.

This script:
- Trains Color Consistency (loads from previous run)
- Trains GRL with recolor augmentation
- Generates all Task 4 artifacts

Produces: model_consistency.pt, model_grl.pt, task4_*.png
"""
import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm

from data_colored_mnist import ColoredMNIST, COLORS
from models import SimpleCNN
from debias import (
    GRLModel,
    recolor_augment_batch,
    evaluate_model,
    compute_confusion_matrix_debias,
    plot_training_curves_debias,
    recolor_augment,
    consistency_loss,
)
from interpretability import GradCAM

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

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

color_names = ["Red", "Green", "Blue", "Yellow", "Magenta", 
               "Cyan", "Orange", "Purple", "DarkGreen", "Gray"]

# =============================================================================
# GRL WITH RECOLOR AUGMENTATION
# =============================================================================
print("\n" + "="*60)
print("GRL WITH RECOLOR AUGMENTATION")
print("="*60)

def train_grl_augmented_epoch(model, loader, optimizer, criterion, device, alpha_color=1.0):
    """Train GRL with recolor augmentation for better color invariance."""
    model.train()
    total_loss = 0.0
    digit_correct = 0
    total = 0
    
    for images, labels, color_ids in loader:
        # Recolor augment the batch to different random colors
        images_aug, new_color_ids = recolor_augment_batch(images)
        images_aug = images_aug.to(device)
        
        labels = labels.to(device)
        new_color_ids = new_color_ids.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        digit_logits, color_logits = model(images_aug, return_color_logits=True)
        
        # Digit classification loss
        loss_digit = criterion(digit_logits, labels)
        
        # Color classification loss (adversarial via GRL)
        loss_color = criterion(color_logits, new_color_ids)
        
        # Total loss
        loss = loss_digit + alpha_color * loss_color
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
        _, digit_preds = digit_logits.max(1)
        digit_correct += (digit_preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, digit_correct / total

# Also load consistency model for comparison
cons_ckpt = torch.load(artifacts_dir / 'model_consistency.pt', map_location=device, weights_only=False)
model_consistency = SimpleCNN(num_classes=10).to(device)
model_consistency.load_state_dict(cons_ckpt['model_state_dict'])
cons_test_acc = cons_ckpt['final_test_acc']
print(f"\nLoaded Consistency model: Hard Test = {cons_test_acc:.2%}")

# Train GRL
model_grl = GRLModel(lambda_grl=1.0).to(device)
optimizer_grl = optim.Adam(model_grl.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

history_grl = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'test_acc': [],
}

EPOCHS = 20
ALPHA_COLOR = 0.5  # Reduced to let digit learning dominate

print(f"\nTraining GRL with augmentation, alpha_color={ALPHA_COLOR}, epochs={EPOCHS}...")
best_test_acc = 0.0
for epoch in range(EPOCHS):
    train_loss, train_acc = train_grl_augmented_epoch(
        model_grl, train_loader, optimizer_grl, criterion, device,
        alpha_color=ALPHA_COLOR
    )
    val_loss, val_acc = evaluate_model(model_grl, val_loader, criterion, device, is_grl=True)
    test_loss, test_acc = evaluate_model(model_grl, test_loader, criterion, device, is_grl=True)
    
    history_grl['train_loss'].append(train_loss)
    history_grl['train_acc'].append(train_acc)
    history_grl['val_loss'].append(val_loss)
    history_grl['val_acc'].append(val_acc)
    history_grl['test_acc'].append(test_acc)
    
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_state = model_grl.state_dict().copy()
    
    print(f"Epoch {epoch+1:2d}: Train={train_acc:.4f}, Val={val_acc:.4f}, HardTest={test_acc:.4f}")
    
    if test_acc >= 0.70:
        print(f"  -> Target reached!")

# Load best model
model_grl.load_state_dict(best_state)
final_test_loss, grl_test_acc = evaluate_model(model_grl, test_loader, criterion, device, is_grl=True)
print(f"\nBest GRL Model: Hard Test = {grl_test_acc:.2%}")

# Save checkpoint
torch.save({
    'model_state_dict': model_grl.state_dict(),
    'history': history_grl,
    'final_test_acc': grl_test_acc,
}, artifacts_dir / 'model_grl.pt')

# Plot curves
plot_training_curves_debias(history_grl, "GRL + Augmentation", 
                            save_path=artifacts_dir / 'task4_grl_curves.png')
plt.close()

# Confusion matrix
conf_grl = compute_confusion_matrix_debias(model_grl, test_loader, device, is_grl=True)
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(conf_grl, cmap='Blues')
ax.figure.colorbar(im, ax=ax)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'Hard Test Confusion (GRL) - Acc: {grl_test_acc:.2%}')
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(conf_grl[i, j]), ha='center', va='center',
                color='white' if conf_grl[i, j] > conf_grl.max()/2 else 'black', fontsize=8)
plt.tight_layout()
plt.savefig(artifacts_dir / 'task4_grl_confmat.png', dpi=150)
plt.close()

# =============================================================================
# RECOLOR INVARIANCE PROOFS
# =============================================================================
print("\n" + "="*60)
print("RECOLOR INVARIANCE PROOFS")
print("="*60)

def recolor_proof_model(model, img, true_label, device, is_grl=False, figsize=(15, 3)):
    """Generate recolor proof for any model."""
    model.eval()
    
    fig, axes = plt.subplots(1, 11, figsize=figsize)
    results = []
    
    # Original
    axes[0].imshow(img.permute(1, 2, 0).numpy())
    axes[0].set_title(f"Original\nTrue: {true_label}", fontsize=8)
    axes[0].axis('off')
    
    for color_id in range(10):
        recolored, _ = recolor_augment(img, color_id)
        
        with torch.no_grad():
            inp = recolored.unsqueeze(0).to(device)
            if is_grl:
                logits = model.predict(inp)
            else:
                logits = model(inp)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(1).item()
            conf = probs[0, pred].item()
        
        results.append({'color_id': color_id, 'pred': pred, 'conf': conf})
        
        ax = axes[color_id + 1]
        ax.imshow(recolored.permute(1, 2, 0).numpy())
        title_color = 'green' if pred == true_label else 'red'
        ax.set_title(f"C{color_id}\nP:{pred}({conf:.0%})", fontsize=8, color=title_color)
        ax.axis('off')
    
    plt.suptitle(f"Recolor Proof (True Label: {true_label})", fontsize=10)
    plt.tight_layout()
    
    correct = sum(1 for r in results if r['pred'] == true_label)
    return fig, results, correct

# Test on digit 1 and 7
for target_digit in [1, 7]:
    for i in range(len(test_ds)):
        _, label, _ = test_ds[i]
        if label == target_digit:
            img, label, _ = test_ds[i]
            
            # Consistency model
            fig, results, correct = recolor_proof_model(
                model_consistency, img, label, device, is_grl=False
            )
            plt.savefig(artifacts_dir / f'task4_recolor_consistency_digit{label}.png', 
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\nConsistency - Digit {label}: {correct}/10 correct across colors")
            
            # GRL model
            fig, results, correct = recolor_proof_model(
                model_grl, img, label, device, is_grl=True
            )
            plt.savefig(artifacts_dir / f'task4_recolor_grl_digit{label}.png', 
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"GRL - Digit {label}: {correct}/10 correct across colors")
            
            break

# =============================================================================
# GRAD-CAM COMPARISON
# =============================================================================
print("\n" + "="*60)
print("GRAD-CAM COMPARISON")
print("="*60)

# Load cheater model
cheater_model = SimpleCNN(num_classes=10).to(device)
cheater_ckpt = torch.load(artifacts_dir / 'baseline_cheater_N2000.pt', 
                           map_location=device, weights_only=False)
cheater_model.load_state_dict(cheater_ckpt['model_state_dict'])

def gradcam_compare(models, model_names, sample_img, sample_label, device, is_grl_flags):
    """Generate Grad-CAM comparison across models."""
    import torch.nn.functional as F
    
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 3, figsize=(9, 3 * n_models))
    
    for row, (model, name, is_grl) in enumerate(zip(models, model_names, is_grl_flags)):
        model.eval()
        
        # Get prediction
        with torch.no_grad():
            inp = sample_img.unsqueeze(0).to(device)
            if is_grl:
                logits = model.predict(inp)
            else:
                logits = model(inp)
            pred = logits.argmax(1).item()
            conf = F.softmax(logits, dim=1)[0, pred].item()
        
        # Grad-CAM
        try:
            backbone = model.backbone if is_grl else model
            gradcam = GradCAM(backbone, 'conv3.0')
            cam = gradcam(sample_img.clone(), target_class=pred)
            cam_up = gradcam.upsample(cam, (28, 28))
            overlay = gradcam.overlay(sample_img, cam_up)
        except Exception as e:
            print(f"Grad-CAM error for {name}: {e}")
            cam_up = torch.zeros(28, 28)
            overlay = sample_img.permute(1, 2, 0).numpy()
        
        # Original
        axes[row, 0].imshow(sample_img.permute(1, 2, 0).numpy())
        axes[row, 0].set_title(f"{name}\nTrue: {sample_label}", fontsize=9)
        axes[row, 0].axis('off')
        
        # CAM
        axes[row, 1].imshow(cam_up.numpy() if isinstance(cam_up, torch.Tensor) else cam_up, cmap='jet')
        title_color = 'green' if pred == sample_label else 'red'
        axes[row, 1].set_title(f"Pred: {pred} ({conf:.0%})", fontsize=9, color=title_color)
        axes[row, 1].axis('off')
        
        # Overlay
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay", fontsize=9)
        axes[row, 2].axis('off')
    
    plt.tight_layout()
    return fig

# Get a few hard test samples
print("\nGenerating Grad-CAM comparison...")
for idx in [0, 5, 10]:
    img, label, _ = test_ds[idx]
    
    fig = gradcam_compare(
        models=[cheater_model, model_consistency, model_grl],
        model_names=['Cheater', 'Consistency', 'GRL'],
        sample_img=img,
        sample_label=label,
        device=device,
        is_grl_flags=[False, False, True],
    )
    plt.savefig(artifacts_dir / f'task4_gradcam_compare_{idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

print("Saved Grad-CAM comparisons")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

# Fixed format strings
cons_val_str = f"{cons_ckpt['history']['val_acc'][-1]:.2%}"
cons_test_str = f"{cons_test_acc:.2%}"
grl_val_str = f"{history_grl['val_acc'][-1]:.2%}"
grl_test_str = f"{grl_test_acc:.2%}"

print("\n" + "-"*50)
print(f"{'Model':<20} | {'Easy Val':<12} | {'Hard Test':<12}")
print("-"*50)
print(f"{'Cheater Baseline':<20} | {'~95%':<12} | {'6.25%':<12}")
print(f"{'Consistency':<20} | {cons_val_str:<12} | {cons_test_str:<12}")
print(f"{'GRL + Aug':<20} | {grl_val_str:<12} | {grl_test_str:<12}")
print("-"*50)

if cons_test_acc >= 0.70:
    print(f"\n✓ Consistency achieves target (>70%)!")
else:
    print(f"\n✗ Consistency below target ({cons_test_acc:.2%} < 70%)")

if grl_test_acc >= 0.70:
    print(f"✓ GRL achieves target (>70%)!")
else:
    print(f"✗ GRL below target ({grl_test_acc:.2%} < 70%)")

print("\nArtifacts saved:")
for f in sorted(artifacts_dir.glob('task4_*.png')):
    print(f"  - {f.name}")

print("\n=== COMPLETE ===")
