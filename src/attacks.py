"""
Task 5: Targeted Adversarial Attacks
=====================================
Implements targeted PGD (Projected Gradient Descent) for Linf attacks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


def targeted_pgd_linf(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    eps: float = 0.05,
    steps: int = 40,
    step_size: float = 0.01,
    random_start: bool = False,
    early_stop_conf: float = 0.90,
    is_grl: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """
    Targeted PGD attack with Linf constraint.
    
    Goal: Make model predict target_class with high confidence.
    
    Args:
        model: Neural network model (in eval mode)
        x: Input image, shape (1, C, H, W), values in [0, 1]
        target_class: Target class to force prediction to
        eps: Linf epsilon constraint (max perturbation per pixel)
        steps: Number of PGD steps
        step_size: Step size for gradient updates
        random_start: If True, initialize within eps-ball randomly
        early_stop_conf: Early stop if target confidence >= this value
        is_grl: If True, use model.predict() for GRL models
        
    Returns:
        x_adv: Adversarial example, shape (1, C, H, W)
        info: Dictionary with attack statistics
    """
    model.eval()
    device = x.device
    
    # Clone to avoid modifying original
    x_adv = x.clone().detach()
    
    # Optional random start within eps-ball
    if random_start:
        noise = torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv + noise, 0, 1)
    
    x_adv.requires_grad = True
    
    # Target class tensor
    target = torch.tensor([target_class], device=device)
    
    # Track attack progress
    info = {
        'success': False,
        'steps_to_success': None,
        'final_pred': None,
        'final_conf': None,
        'target_conf': None,
        'linf_delta': None,
        'history': [],
    }
    
    criterion = nn.CrossEntropyLoss()
    
    for step in range(steps):
        x_adv.requires_grad = True
        
        # Forward pass
        if is_grl:
            logits = model.predict(x_adv)
        else:
            logits = model(x_adv)
        
        # Compute loss (minimize CE to target -> descend)
        loss = criterion(logits, target)
        
        # Get current predictions
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(1).item()
        target_conf = probs[0, target_class].item()
        pred_conf = probs[0, pred].item()
        
        info['history'].append({
            'step': step,
            'pred': pred,
            'pred_conf': pred_conf,
            'target_conf': target_conf,
            'loss': loss.item(),
        })
        
        # Check early stopping
        if pred == target_class and target_conf >= early_stop_conf:
            info['success'] = True
            info['steps_to_success'] = step + 1
            break
        
        # Backward pass
        loss.backward()
        
        # Gradient step (DESCEND to minimize loss -> move toward target)
        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv - step_size * grad_sign
            
            # Project back to Linf ball around original x
            delta = torch.clamp(x_adv - x, -eps, eps)
            x_adv = torch.clamp(x + delta, 0, 1)
        
        x_adv = x_adv.detach().clone()
    
    # Final evaluation
    with torch.no_grad():
        if is_grl:
            logits = model.predict(x_adv)
        else:
            logits = model(x_adv)
        probs = F.softmax(logits, dim=1)
        final_pred = probs.argmax(1).item()
        final_conf = probs[0, final_pred].item()
        target_conf = probs[0, target_class].item()
    
    info['final_pred'] = final_pred
    info['final_conf'] = final_conf
    info['target_conf'] = target_conf
    info['linf_delta'] = (x_adv - x).abs().max().item()
    
    if not info['success'] and final_pred == target_class and target_conf >= early_stop_conf:
        info['success'] = True
        info['steps_to_success'] = steps
    
    return x_adv.detach(), info


def visualize_attack(
    x_orig: torch.Tensor,
    x_adv: torch.Tensor,
    orig_pred: int,
    orig_conf: float,
    adv_pred: int,
    adv_conf: float,
    target_class: int,
    model_name: str,
    linf_delta: float,
    figsize: Tuple[int, int] = (14, 3),
):
    """
    Visualize adversarial attack results.
    
    Shows: Original | Adversarial | Delta | Delta*10
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Convert to numpy for plotting
    orig_np = x_orig.squeeze(0).permute(1, 2, 0).cpu().numpy()
    adv_np = x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy()
    delta = adv_np - orig_np
    
    # Original
    axes[0].imshow(np.clip(orig_np, 0, 1))
    title_color = 'green' if orig_pred == 7 else 'red'
    axes[0].set_title(f"Original\nPred: {orig_pred} ({orig_conf:.1%})", 
                      fontsize=10, color=title_color)
    axes[0].axis('off')
    
    # Adversarial
    axes[1].imshow(np.clip(adv_np, 0, 1))
    title_color = 'green' if adv_pred == target_class else 'red'
    axes[1].set_title(f"Adversarial\nPred: {adv_pred} ({adv_conf:.1%})", 
                      fontsize=10, color=title_color)
    axes[1].axis('off')
    
    # Delta (raw)
    # Normalize delta for visualization (center at 0.5)
    delta_vis = (delta - delta.min()) / (delta.max() - delta.min() + 1e-8)
    axes[2].imshow(delta_vis)
    axes[2].set_title(f"Delta\nLinf={linf_delta:.4f}", fontsize=10)
    axes[2].axis('off')
    
    # Delta * 10 (amplified)
    delta_amp = np.clip(0.5 + delta * 10, 0, 1)
    axes[3].imshow(delta_amp)
    axes[3].set_title(f"Delta x 10\n(amplified)", fontsize=10)
    axes[3].axis('off')
    
    plt.suptitle(f"{model_name}: 7 -> {target_class} Attack | Linf < 0.05", fontsize=12)
    plt.tight_layout()
    
    return fig


def attack_comparison_report(results: Dict[str, Dict]) -> str:
    """Generate a text report comparing attack results across models."""
    lines = []
    lines.append("=" * 60)
    lines.append("ADVERSARIAL ATTACK COMPARISON REPORT")
    lines.append(f"Target: 7 -> 3 with >90% confidence, eps < 0.05")
    lines.append("=" * 60)
    lines.append("")
    
    header = f"{'Model':<20} | {'Before':<15} | {'After':<15} | {'Linf D':<8} | {'Steps':<8} | {'Success'}"
    lines.append(header)
    lines.append("-" * len(header))
    
    for model_name, info in results.items():
        before = f"{info['orig_pred']}({info['orig_conf']:.0%})"
        after = f"{info['final_pred']}({info['final_conf']:.0%})"
        linf = f"{info['linf_delta']:.4f}"
        steps = str(info['steps_to_success']) if info['steps_to_success'] else "N/A"
        success = "Y" if info['success'] else "N"
        
        lines.append(f"{model_name:<20} | {before:<15} | {after:<15} | {linf:<8} | {steps:<8} | {success}")
    
    lines.append("-" * len(header))
    lines.append("")
    
    # Summary
    lines.append("Observations:")
    for model_name, info in results.items():
        if info['success']:
            lines.append(f"  - {model_name}: Attack SUCCEEDED in {info['steps_to_success']} steps")
        else:
            lines.append(f"  - {model_name}: Attack FAILED (best target conf: {info['target_conf']:.1%})")
    
    return "\n".join(lines)
