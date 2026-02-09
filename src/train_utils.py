"""
Training Utilities for Colored MNIST
=====================================
Training loops, evaluation, confusion matrix, and recolor proof utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Train", leave=False):
        images, labels, _ = batch  # Ignore color_id for training
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on a dataset.
    
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Eval", leave=False):
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 10,
    save_path: Optional[Path] = None,
    target_val_acc: float = 0.95,
) -> Dict[str, List[float]]:
    """
    Full training loop with early stopping when target val acc is reached.
    
    Returns:
        Dictionary with training history (train_loss, train_acc, val_loss, val_acc)
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                }, save_path)
                print(f"  Saved checkpoint to {save_path}")
        
        # Early stopping
        if val_acc >= target_val_acc:
            print(f"\nReached target val accuracy {target_val_acc:.2%}!")
            break
    
    return history


def plot_training_curves(history: Dict[str, List[float]], save_path: Optional[Path] = None):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss
    axes[0].plot(epochs, history["train_loss"], 'b-', label='Train')
    axes[0].plot(epochs, history["val_loss"], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history["train_acc"], 'b-', label='Train')
    axes[1].plot(epochs, history["val_acc"], 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


@torch.no_grad()
def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> np.ndarray:
    """
    Compute confusion matrix from scratch.
    
    Returns:
        10x10 numpy array where M[i,j] = count of (true=i, pred=j)
    """
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for batch in tqdm(loader, desc="Computing confusion matrix", leave=False):
        images, labels, _ = batch
        images = images.to(device)
        
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        for true, pred in zip(labels.numpy(), predicted.cpu().numpy()):
            confusion[true, pred] += 1
    
    return confusion


def plot_confusion_matrix(
    confusion: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
):
    """Plot confusion matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(confusion, cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")
    
    # Labels
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    # Add text annotations
    thresh = confusion.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, format(confusion[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if confusion[i, j] > thresh else "black",
                    fontsize=8)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def recolor_image(
    gray_img: torch.Tensor,
    color_id: int,
    colors: List[Tuple[float, float, float]],
) -> torch.Tensor:
    """
    Recolor a grayscale mask with a specific color.
    
    Args:
        gray_img: Grayscale image (1, 28, 28) in [0, 1]
        color_id: Color index
        colors: List of RGB color tuples
        
    Returns:
        RGB image (3, 28, 28)
    """
    color_rgb = torch.tensor(colors[color_id], dtype=torch.float32).view(3, 1, 1)
    mask = gray_img
    
    # Simple recoloring: foreground = mask * color, background = gray
    foreground = mask * color_rgb
    background = (1 - mask) * 0.2  # Dark gray background
    
    final = foreground + background
    return torch.clamp(final, 0.0, 1.0)


@torch.no_grad()
def recolor_proof(
    model: nn.Module,
    image: torch.Tensor,
    true_label: int,
    colors: List[Tuple[float, float, float]],
    color_names: List[str],
    device: torch.device,
    figsize: Tuple[int, int] = (20, 4),
) -> Dict:
    """
    Demonstrate that model prediction changes based on color.
    
    Args:
        model: Trained model
        image: Original RGB image (3, 28, 28)
        true_label: True digit label
        colors: List of 10 RGB color tuples
        color_names: List of 10 color names
        device: Torch device
        figsize: Figure size for visualization
        
    Returns:
        Dictionary with recolor results
    """
    model.eval()
    
    # Extract grayscale mask from the image (use max across channels as proxy)
    # Better approach: use the original MNIST grayscale, but we work with what we have
    gray_mask = image.mean(dim=0, keepdim=True)  # Average to grayscale
    gray_mask = (gray_mask - gray_mask.min()) / (gray_mask.max() - gray_mask.min() + 1e-8)
    
    results = []
    recolored_images = []
    
    fig, axes = plt.subplots(1, 11, figsize=figsize)
    
    # Original image
    axes[0].imshow(image.permute(1, 2, 0).numpy())
    axes[0].set_title(f"Original\nTrue: {true_label}", fontsize=9)
    axes[0].axis('off')
    
    # Recolor to each of 10 colors
    for color_id in range(10):
        recolored = recolor_image(gray_mask, color_id, colors)
        recolored_images.append(recolored)
        
        # Predict
        inp = recolored.unsqueeze(0).to(device)
        logits = model(inp)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(1).item()
        conf = probs[0, pred].item()
        
        results.append({
            "color_id": color_id,
            "color_name": color_names[color_id],
            "predicted": pred,
            "confidence": conf,
            "prob_for_true": probs[0, true_label].item(),
        })
        
        # Plot
        axes[color_id + 1].imshow(recolored.permute(1, 2, 0).numpy())
        title_color = 'green' if pred == true_label else 'red'
        axes[color_id + 1].set_title(
            f"C{color_id}: {color_names[color_id][:3]}\nPred: {pred} ({conf:.0%})",
            fontsize=8, color=title_color
        )
        axes[color_id + 1].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Recolor Proof: True Label = {true_label}", fontsize=12, y=1.02)
    
    return {
        "results": results,
        "figure": fig,
        "recolored_images": recolored_images,
    }


def print_recolor_results(results: List[Dict], true_label: int):
    """Print recolor proof results as a table."""
    print(f"\nRecolor Proof Results (True Label: {true_label})")
    print("=" * 60)
    print(f"{'Color ID':<10} {'Color Name':<12} {'Predicted':<10} {'Confidence':<12} {'P(true)':<10}")
    print("-" * 60)
    
    for r in results:
        marker = "✓" if r["predicted"] == true_label else "✗"
        print(f"{r['color_id']:<10} {r['color_name']:<12} {r['predicted']:<10} {r['confidence']:.4f}       {r['prob_for_true']:.4f}     {marker}")
    
    print("=" * 60)
    
    # Summary
    correct_count = sum(1 for r in results if r["predicted"] == true_label)
    print(f"\nCorrect predictions: {correct_count}/10")
    print(f"Model predicts based on color, not shape!" if correct_count < 5 else "Model uses shape features.")


if __name__ == "__main__":
    # Quick test
    print("Training utilities loaded successfully!")
