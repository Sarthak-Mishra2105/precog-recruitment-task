"""
Task 5: Run Targeted Adversarial Attack (7 -> 3)
================================================
Compares attack success between cheater baseline and debiased models.
"""
import sys
import io
# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from data_colored_mnist import ColoredMNIST
from models import SimpleCNN
from debias import GRLModel
from attacks import targeted_pgd_linf, visualize_attack, attack_comparison_report

# Setup
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

artifacts_dir = Path('artifacts')
artifacts_dir.mkdir(exist_ok=True)

# =============================================================================
# LOAD MODELS
# =============================================================================
print("\n" + "="*60)
print("LOADING MODELS")
print("="*60)

# Cheater baseline
cheater_model = SimpleCNN(num_classes=10).to(device)
cheater_ckpt = torch.load(artifacts_dir / 'baseline_cheater_N2000.pt', 
                           map_location=device, weights_only=False)
cheater_model.load_state_dict(cheater_ckpt['model_state_dict'])
cheater_model.eval()
print("[OK] Loaded cheater baseline model")

# Consistency model (robust)
consistency_model = SimpleCNN(num_classes=10).to(device)
cons_ckpt = torch.load(artifacts_dir / 'model_consistency.pt', 
                        map_location=device, weights_only=False)
consistency_model.load_state_dict(cons_ckpt['model_state_dict'])
consistency_model.eval()
print("[OK] Loaded consistency (robust) model")

# GRL model (robust)
grl_model = GRLModel(lambda_grl=1.0).to(device)
grl_ckpt = torch.load(artifacts_dir / 'model_grl.pt', 
                       map_location=device, weights_only=False)
grl_model.load_state_dict(grl_ckpt['model_state_dict'])
grl_model.eval()
print("[OK] Loaded GRL (robust) model")

# =============================================================================
# FIND A SUITABLE TEST IMAGE
# =============================================================================
print("\n" + "="*60)
print("FINDING SUITABLE TEST IMAGE (digit 7)")
print("="*60)

# Load datasets
hard_test_ds = ColoredMNIST(root='./data', split='hard_test', seed=SEED)
val_ds = ColoredMNIST(root='./data', split='val', seed=SEED)

def get_prediction(model, img, device, is_grl=False):
    """Get model prediction and confidence."""
    model.eval()
    with torch.no_grad():
        inp = img.unsqueeze(0).to(device)
        if is_grl:
            logits = model.predict(inp)
        else:
            logits = model(inp)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(1).item()
        conf = probs[0, pred].item()
    return pred, conf

# Look for a digit 7 that all models predict correctly
# First try hard_test, then val
selected_img = None
selected_label = None
selected_source = None

print("\nSearching hard_test for digit 7 correctly classified by all models...")
for i in range(len(hard_test_ds)):
    img, label, _ = hard_test_ds[i]
    if label == 7:
        cheater_pred, cheater_conf = get_prediction(cheater_model, img, device)
        cons_pred, cons_conf = get_prediction(consistency_model, img, device)
        grl_pred, grl_conf = get_prediction(grl_model, img, device, is_grl=True)
        
        # We need at least robust models to predict correctly for fair comparison
        # Cheater likely fails on hard_test (that's the point of bias)
        if cons_pred == 7 and grl_pred == 7:
            selected_img = img
            selected_label = label
            selected_source = 'hard_test'
            print(f"  Found at hard_test[{i}]:")
            print(f"    Cheater: pred={cheater_pred}, conf={cheater_conf:.2%}")
            print(f"    Consistency: pred={cons_pred}, conf={cons_conf:.2%}")
            print(f"    GRL: pred={grl_pred}, conf={grl_conf:.2%}")
            break

if selected_img is None:
    print("\nNo suitable image in hard_test, searching val...")
    for i in range(len(val_ds)):
        img, label, _ = val_ds[i]
        if label == 7:
            cheater_pred, cheater_conf = get_prediction(cheater_model, img, device)
            cons_pred, cons_conf = get_prediction(consistency_model, img, device)
            grl_pred, grl_conf = get_prediction(grl_model, img, device, is_grl=True)
            
            # All models should predict correctly on val (easy)
            if cheater_pred == 7 and cons_pred == 7 and grl_pred == 7:
                selected_img = img
                selected_label = label
                selected_source = 'val'
                print(f"  Found at val[{i}]:")
                print(f"    Cheater: pred={cheater_pred}, conf={cheater_conf:.2%}")
                print(f"    Consistency: pred={cons_pred}, conf={cons_conf:.2%}")
                print(f"    GRL: pred={grl_pred}, conf={grl_conf:.2%}")
                break

if selected_img is None:
    raise ValueError("Could not find suitable digit 7 image!")

print(f"\n[OK] Selected image from {selected_source}, true label = {selected_label}")

# =============================================================================
# ATTACK PARAMETERS
# =============================================================================
print("\n" + "="*60)
print("ATTACK PARAMETERS")
print("="*60)

TARGET_CLASS = 3
EPS = 0.05
STEPS = 100  # More steps for better success
STEP_SIZE = EPS / 10  # = 0.005
RANDOM_START = True
EARLY_STOP_CONF = 0.90

print(f"  Target class: {TARGET_CLASS}")
print(f"  Epsilon (L∞): {EPS}")
print(f"  Max steps: {STEPS}")
print(f"  Step size: {STEP_SIZE}")
print(f"  Random start: {RANDOM_START}")
print(f"  Early stop confidence: {EARLY_STOP_CONF}")

# =============================================================================
# RUN ATTACKS
# =============================================================================
print("\n" + "="*60)
print("RUNNING TARGETED PGD ATTACKS")
print("="*60)

x_input = selected_img.unsqueeze(0).to(device)
results = {}

# Attack on Cheater Model
print("\n--- CHEATER BASELINE ---")
cheater_pred, cheater_conf = get_prediction(cheater_model, selected_img, device)
print(f"Before: pred={cheater_pred}, conf={cheater_conf:.2%}")

x_adv_cheater, info_cheater = targeted_pgd_linf(
    cheater_model, x_input, TARGET_CLASS,
    eps=EPS, steps=STEPS, step_size=STEP_SIZE,
    random_start=RANDOM_START, early_stop_conf=EARLY_STOP_CONF,
    is_grl=False
)

print(f"After: pred={info_cheater['final_pred']}, conf={info_cheater['final_conf']:.2%}")
print(f"Target conf: {info_cheater['target_conf']:.2%}")
print(f"L∞ delta: {info_cheater['linf_delta']:.4f}")
print(f"Steps to success: {info_cheater['steps_to_success'] if info_cheater['success'] else 'Failed'}")

results['Cheater'] = {
    'orig_pred': cheater_pred,
    'orig_conf': cheater_conf,
    **info_cheater
}

# Attack on Consistency Model
print("\n--- CONSISTENCY (ROBUST) ---")
cons_pred, cons_conf = get_prediction(consistency_model, selected_img, device)
print(f"Before: pred={cons_pred}, conf={cons_conf:.2%}")

x_adv_cons, info_cons = targeted_pgd_linf(
    consistency_model, x_input, TARGET_CLASS,
    eps=EPS, steps=STEPS, step_size=STEP_SIZE,
    random_start=RANDOM_START, early_stop_conf=EARLY_STOP_CONF,
    is_grl=False
)

print(f"After: pred={info_cons['final_pred']}, conf={info_cons['final_conf']:.2%}")
print(f"Target conf: {info_cons['target_conf']:.2%}")
print(f"L∞ delta: {info_cons['linf_delta']:.4f}")
print(f"Steps to success: {info_cons['steps_to_success'] if info_cons['success'] else 'Failed'}")

results['Consistency'] = {
    'orig_pred': cons_pred,
    'orig_conf': cons_conf,
    **info_cons
}

# Attack on GRL Model
print("\n--- GRL (ROBUST) ---")
grl_pred, grl_conf = get_prediction(grl_model, selected_img, device, is_grl=True)
print(f"Before: pred={grl_pred}, conf={grl_conf:.2%}")

x_adv_grl, info_grl = targeted_pgd_linf(
    grl_model, x_input, TARGET_CLASS,
    eps=EPS, steps=STEPS, step_size=STEP_SIZE,
    random_start=RANDOM_START, early_stop_conf=EARLY_STOP_CONF,
    is_grl=True
)

print(f"After: pred={info_grl['final_pred']}, conf={info_grl['final_conf']:.2%}")
print(f"Target conf: {info_grl['target_conf']:.2%}")
print(f"L∞ delta: {info_grl['linf_delta']:.4f}")
print(f"Steps to success: {info_grl['steps_to_success'] if info_grl['success'] else 'Failed'}")

results['GRL'] = {
    'orig_pred': grl_pred,
    'orig_conf': grl_conf,
    **info_grl
}

# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Cheater visualization
fig = visualize_attack(
    x_input, x_adv_cheater,
    cheater_pred, cheater_conf,
    info_cheater['final_pred'], info_cheater['final_conf'],
    TARGET_CLASS, 'Cheater Baseline',
    info_cheater['linf_delta']
)
plt.savefig(artifacts_dir / 'task5_attack_cheater.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved task5_attack_cheater.png")

# Consistency visualization
fig = visualize_attack(
    x_input, x_adv_cons,
    cons_pred, cons_conf,
    info_cons['final_pred'], info_cons['final_conf'],
    TARGET_CLASS, 'Consistency (Robust)',
    info_cons['linf_delta']
)
plt.savefig(artifacts_dir / 'task5_attack_consistency.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved task5_attack_consistency.png")

# GRL visualization
fig = visualize_attack(
    x_input, x_adv_grl,
    grl_pred, grl_conf,
    info_grl['final_pred'], info_grl['final_conf'],
    TARGET_CLASS, 'GRL (Robust)',
    info_grl['linf_delta']
)
plt.savefig(artifacts_dir / 'task5_attack_grl.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved task5_attack_grl.png")

# Combined comparison figure
fig, axes = plt.subplots(3, 4, figsize=(14, 9))
models_data = [
    ('Cheater Baseline', x_adv_cheater, cheater_pred, cheater_conf, info_cheater),
    ('Consistency (Robust)', x_adv_cons, cons_pred, cons_conf, info_cons),
    ('GRL (Robust)', x_adv_grl, grl_pred, grl_conf, info_grl),
]

for row, (name, x_adv, orig_pred, orig_conf, info) in enumerate(models_data):
    orig_np = x_input.squeeze(0).permute(1, 2, 0).cpu().numpy()
    adv_np = x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy()
    delta = adv_np - orig_np
    
    # Original
    axes[row, 0].imshow(np.clip(orig_np, 0, 1))
    axes[row, 0].set_title(f"Original\n{orig_pred}({orig_conf:.0%})", fontsize=9)
    axes[row, 0].axis('off')
    axes[row, 0].set_ylabel(name, fontsize=10, rotation=0, ha='right', va='center')
    
    # Adversarial
    axes[row, 1].imshow(np.clip(adv_np, 0, 1))
    title_color = 'green' if info['final_pred'] == TARGET_CLASS else 'red'
    axes[row, 1].set_title(f"Adversarial\n{info['final_pred']}({info['final_conf']:.0%})", 
                           fontsize=9, color=title_color)
    axes[row, 1].axis('off')
    
    # Delta
    delta_vis = (delta - delta.min()) / (delta.max() - delta.min() + 1e-8)
    axes[row, 2].imshow(delta_vis)
    axes[row, 2].set_title(f"Delta\nLinf={info['linf_delta']:.4f}", fontsize=9)
    axes[row, 2].axis('off')
    
    # Delta * 10
    delta_amp = np.clip(0.5 + delta * 10, 0, 1)
    axes[row, 3].imshow(delta_amp)
    success_str = f"OK{info['steps_to_success']}" if info['success'] else "X"
    axes[row, 3].set_title(f"Delta x10\n{success_str}", fontsize=9,
                          color='green' if info['success'] else 'red')
    axes[row, 3].axis('off')

plt.suptitle(f"Targeted PGD Attack: 7 -> 3 | eps < {EPS} | Target Conf > {EARLY_STOP_CONF:.0%}", 
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(artifacts_dir / 'task5_attack_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved task5_attack_comparison.png")

# =============================================================================
# SUMMARY REPORT
# =============================================================================
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)

report = attack_comparison_report(results)
print(report)

# Save report
with open(artifacts_dir / 'task5_attack_summary.txt', 'w') as f:
    f.write(report)
print(f"\n[OK] Saved task5_attack_summary.txt")

# =============================================================================
# FINAL TABLE
# =============================================================================
print("\n" + "="*60)
print("FINAL RESULTS TABLE")
print("="*60)

print(f"\n{'Model':<20} | {'Before':<12} | {'After':<12} | {'Linf D':<8} | {'Steps':<6} | {'Success'}")
print("-" * 75)
for name, info in results.items():
    before = f"{info['orig_pred']}({info['orig_conf']:.0%})"
    after = f"{info['final_pred']}({info['final_conf']:.0%})"
    linf = f"{info['linf_delta']:.4f}"
    steps = str(info['steps_to_success']) if info['steps_to_success'] else "N/A"
    success = "Y" if info['success'] else "N"
    print(f"{name:<20} | {before:<12} | {after:<12} | {linf:<8} | {steps:<6} | {success}")
print("-" * 75)

# Analysis
print("\n[ANALYSIS]:")
cheater_success = results['Cheater']['success']
cons_success = results['Consistency']['success']
grl_success = results['GRL']['success']

if cheater_success and cons_success and grl_success:
    print("  All models are vulnerable to targeted PGD within ε<0.05")
    cheater_steps = results['Cheater']['steps_to_success']
    cons_steps = results['Consistency']['steps_to_success']
    grl_steps = results['GRL']['steps_to_success']
    print(f"  Cheater took {cheater_steps} steps, Consistency took {cons_steps}, GRL took {grl_steps}")
    if cons_steps > cheater_steps or grl_steps > cheater_steps:
        print("  [OK] Robust models required MORE steps to fool (more robust)")
elif cheater_success and not (cons_success and grl_success):
    print("  [OK] Cheater is more vulnerable, robust models resisted attack!")
else:
    print("  Mixed results - see detailed breakdown above.")

print("\n=== COMPLETE ===")
