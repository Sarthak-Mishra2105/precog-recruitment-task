"""
Quick sanity check for demo app pipeline.
Loads all 3 models, creates a synthetic "7", runs inference + Grad-CAM.
Saves debug output to demo_app/_debug/.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "demo_app"))

import torch
import numpy as np
import matplotlib.pyplot as plt

from models import SimpleCNN
from debias import GRLModel
from interpretability import GradCAM
from preprocess import COLORS, COLOR_NAMES, colorize_mask


def create_synthetic_7(size=28):
    """Create a simple synthetic digit '7' mask."""
    mask = np.zeros((size, size), dtype=np.float32)
    
    # Horizontal bar at top
    mask[5:8, 5:23] = 1.0
    
    # Diagonal stroke
    for i in range(18):
        y = 7 + i
        x = 22 - int(i * 0.6)
        if 0 <= y < size and 0 <= x < size:
            mask[y, max(0, x-1):min(size, x+2)] = 1.0
    
    return mask


def load_models(device):
    """Load all three models."""
    artifacts = PROJECT_ROOT / "artifacts"
    models = {}
    
    # Cheater
    cheater = SimpleCNN(num_classes=10).to(device)
    ckpt = torch.load(artifacts / "baseline_cheater_N2000.pt", 
                      map_location=device, weights_only=False)
    cheater.load_state_dict(ckpt['model_state_dict'])
    cheater.eval()
    models['Cheater'] = cheater
    
    # Consistency
    consistency = SimpleCNN(num_classes=10).to(device)
    ckpt = torch.load(artifacts / "model_consistency.pt", 
                      map_location=device, weights_only=False)
    consistency.load_state_dict(ckpt['model_state_dict'])
    consistency.eval()
    models['Consistency'] = consistency
    
    # GRL
    grl = GRLModel(lambda_grl=1.0).to(device)
    ckpt = torch.load(artifacts / "model_grl.pt", 
                      map_location=device, weights_only=False)
    grl.load_state_dict(ckpt['model_state_dict'])
    grl.eval()
    models['GRL'] = grl
    
    return models


def compute_gradcam_for_model(model, tensor, is_grl=False):
    """Compute Grad-CAM for a model."""
    if is_grl:
        backbone = model.backbone
    else:
        backbone = model
    
    gradcam = GradCAM(backbone, 'conv3.0')
    cam = gradcam(tensor.clone().requires_grad_(True), target_class=None)
    cam_up = gradcam.upsample(cam, (28, 28))
    
    img_np = tensor.squeeze(0).permute(1, 2, 0).numpy()
    overlay = gradcam.overlay(img_np, cam_up)
    
    return cam_up.numpy(), overlay


def main():
    print("=" * 60)
    print("QUICK DEMO CHECK")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Create output directory
    debug_dir = PROJECT_ROOT / "demo_app" / "_debug"
    debug_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {debug_dir}")
    
    # Load models
    print("\n[1] Loading models...")
    try:
        models = load_models(device)
        print("    [OK] All 3 models loaded successfully")
    except Exception as e:
        print(f"    [FAIL] Error loading models: {e}")
        return 1
    
    # Create synthetic 7
    print("\n[2] Creating synthetic digit '7'...")
    mask = create_synthetic_7()
    
    # Test with color 7 (Purple - the "correct" color for digit 7)
    color_id = 7
    rgb = colorize_mask(mask, color_id)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    print(f"    [OK] Created tensor shape: {tensor.shape}")
    
    # Save input visualization
    plt.figure(figsize=(3, 3))
    plt.imshow(rgb)
    plt.title(f"Synthetic 7 (Color: {COLOR_NAMES[color_id]})")
    plt.axis('off')
    plt.savefig(debug_dir / "input_digit7.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    [OK] Saved input_digit7.png")
    
    # Run inference and Grad-CAM for each model
    print("\n[3] Running inference and Grad-CAM...")
    
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    
    for row, (name, model) in enumerate(models.items()):
        is_grl = (name == 'GRL')
        
        # Get prediction
        with torch.no_grad():
            if is_grl:
                logits = model.predict(tensor)
            else:
                logits = model(tensor)
            
            probs = torch.softmax(logits, dim=1)[0]
            pred = probs.argmax().item()
            conf = probs[pred].item()
        
        print(f"    {name}: pred={pred} (conf={conf:.2%})")
        
        # Compute Grad-CAM
        cam, overlay = compute_gradcam_for_model(model, tensor, is_grl)
        
        # Plot row: input | CAM | overlay
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"{name}", fontsize=10)
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(cam, cmap='jet')
        axes[row, 1].set_title(f"Pred: {pred} ({conf:.0%})", fontsize=10)
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay", fontsize=10)
        axes[row, 2].axis('off')
    
    # Add column headers
    axes[0, 0].set_title("Input\nCheater", fontsize=10)
    axes[1, 0].set_title("Input\nConsistency", fontsize=10)
    axes[2, 0].set_title("Input\nGRL", fontsize=10)
    
    plt.suptitle("Grad-CAM Comparison: Synthetic '7'", fontsize=12)
    plt.tight_layout()
    plt.savefig(debug_dir / "gradcam_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [OK] Saved gradcam_comparison.png")
    
    # Test recolor robustness
    print("\n[4] Testing recolor robustness...")
    
    results = []
    for cid in range(10):
        rgb_c = colorize_mask(mask, cid)
        tensor_c = torch.from_numpy(rgb_c).permute(2, 0, 1).unsqueeze(0).float()
        
        row_result = [f"C{cid}: {COLOR_NAMES[cid][:6]:6}"]
        for name, model in models.items():
            is_grl = (name == 'GRL')
            with torch.no_grad():
                if is_grl:
                    logits = model.predict(tensor_c)
                else:
                    logits = model(tensor_c)
                pred = logits.argmax().item()
            row_result.append(str(pred))
        results.append(row_result)
    
    # Print table
    print(f"\n    {'Color':<14} | Cheater | Consist | GRL")
    print("    " + "-" * 45)
    for row in results:
        print(f"    {row[0]:<14} | {row[1]:^7} | {row[2]:^7} | {row[3]:^3}")
    
    # Count correct predictions (should predict 7)
    cheater_correct = sum(1 for r in results if r[1] == '7')
    consist_correct = sum(1 for r in results if r[2] == '7')
    grl_correct = sum(1 for r in results if r[3] == '7')
    
    print(f"\n    Correct (7/10 colors):")
    print(f"      Cheater:     {cheater_correct}/10")
    print(f"      Consistency: {consist_correct}/10")
    print(f"      GRL:         {grl_correct}/10")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Quick demo check complete!")
    print(f"Check outputs in: {debug_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
