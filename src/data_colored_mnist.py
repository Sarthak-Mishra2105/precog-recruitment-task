"""
ColoredMNIST Dataset Implementation
=====================================
Creates a biased "Colored MNIST" where digit label correlates with color 
in train/val (~95%) but is broken in a "hard test" split.

Color is applied to digit strokes with a textured background (NOT flat solid).
Returns (image, digit_label, color_id).
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple, Optional, Literal
import torch.nn.functional as F


# 10 distinct colors (RGB in [0,1]) - one dominant per digit
COLORS = [
    (1.0, 0.0, 0.0),    # 0: Red
    (0.0, 1.0, 0.0),    # 1: Green
    (0.0, 0.0, 1.0),    # 2: Blue
    (1.0, 1.0, 0.0),    # 3: Yellow
    (1.0, 0.0, 1.0),    # 4: Magenta
    (0.0, 1.0, 1.0),    # 5: Cyan
    (1.0, 0.5, 0.0),    # 6: Orange
    (0.5, 0.0, 1.0),    # 7: Purple
    (0.0, 0.5, 0.0),    # 8: Dark Green
    (0.5, 0.5, 0.5),    # 9: Gray
]

# Dominant color mapping: digit d -> color d
def get_dominant_color(digit: int) -> int:
    """Returns the dominant color_id for a given digit."""
    return digit


class ColoredMNIST(Dataset):
    """
    Colored MNIST dataset with spurious color-digit correlation.
    
    Args:
        root: Root directory for MNIST data
        split: 'train', 'val', or 'hard_test'
        bias_prob: Probability of using dominant color in train/val (default 0.95)
        seed: Random seed for reproducibility
        download: Whether to download MNIST if not present
        val_ratio: Ratio of training data to use for validation (default 0.1)
    
    Returns:
        Tuple of (image, digit_label, color_id) where:
        - image: (3, 28, 28) float32 tensor in [0,1]
        - digit_label: int in [0,9]
        - color_id: int in [0,9]
    """
    
    def __init__(
        self,
        root: str = "./data",
        split: Literal["train", "val", "hard_test"] = "train",
        bias_prob: float = 0.95,
        seed: int = 42,
        download: bool = True,
        val_ratio: float = 0.1,
    ):
        self.root = Path(root)
        self.split = split
        self.bias_prob = bias_prob
        self.seed = seed
        self.val_ratio = val_ratio
        
        # Set up random generator for reproducibility
        self.rng = np.random.RandomState(seed)
        
        # Load MNIST
        is_train = split in ["train", "val"]
        self.mnist = datasets.MNIST(
            root=str(self.root),
            train=is_train,
            download=download,
            transform=transforms.ToTensor(),
        )
        
        # Split train/val if needed
        if split in ["train", "val"]:
            n_total = len(self.mnist)
            n_val = int(n_total * val_ratio)
            n_train = n_total - n_val
            
            # Use a fixed permutation for train/val split
            perm_rng = np.random.RandomState(seed)
            indices = perm_rng.permutation(n_total)
            
            if split == "train":
                self.indices = indices[:n_train]
            else:
                self.indices = indices[n_train:]
        else:
            # hard_test uses full test set
            self.indices = np.arange(len(self.mnist))
        
        # Pre-generate color assignments for consistency
        self._generate_color_assignments()
    
    def _generate_color_assignments(self):
        """Pre-generate color_id for each sample for reproducibility."""
        self.color_ids = []
        
        # Reset RNG for color generation (use different seed offset per split)
        split_offset = {"train": 0, "val": 1000, "hard_test": 2000}[self.split]
        color_rng = np.random.RandomState(self.seed + split_offset)
        
        for idx in self.indices:
            _, label = self.mnist[idx]
            label = int(label)
            dominant_color = get_dominant_color(label)
            
            if self.split == "hard_test":
                # Hard test: NEVER use dominant color
                # Use (label + 1) % 10 as deterministic alternative
                # OR sample uniformly from non-dominant colors
                non_dominant = [c for c in range(10) if c != dominant_color]
                color_id = non_dominant[color_rng.randint(0, 9)]
            else:
                # Train/val: use dominant with probability bias_prob
                if color_rng.random() < self.bias_prob:
                    color_id = dominant_color
                else:
                    non_dominant = [c for c in range(10) if c != dominant_color]
                    color_id = non_dominant[color_rng.randint(0, 9)]
            
            self.color_ids.append(color_id)
    
    def _create_texture(self, shape: Tuple[int, int], rng_seed: int) -> torch.Tensor:
        """
        Create a smooth noise texture for the background.
        
        Args:
            shape: (H, W) of the texture
            rng_seed: Seed for this specific texture
            
        Returns:
            Texture tensor of shape (1, H, W) in [0, 1]
        """
        local_rng = np.random.RandomState(rng_seed)
        
        # Generate random noise
        noise = local_rng.randn(*shape).astype(np.float32)
        noise_tensor = torch.from_numpy(noise).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Smooth with average pooling (simulates blur)
        # Use convolution with uniform kernel for smoothing
        kernel_size = 5
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        
        # Pad to maintain size
        pad = kernel_size // 2
        noise_padded = F.pad(noise_tensor, (pad, pad, pad, pad), mode='reflect')
        smoothed = F.conv2d(noise_padded, kernel)
        
        # Normalize to [0, 1]
        smoothed = smoothed.squeeze(0)  # (1, H, W)
        smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)
        
        # Scale to make texture subtle (0.1 to 0.3 range for background)
        smoothed = smoothed * 0.2 + 0.1
        
        return smoothed
    
    def _apply_color(
        self,
        gray_img: torch.Tensor,
        color_id: int,
        texture_seed: int,
    ) -> torch.Tensor:
        """
        Apply color to digit stroke and textured background.
        
        Args:
            gray_img: Grayscale MNIST image (1, 28, 28) in [0, 1]
            color_id: Index into COLORS list
            texture_seed: Seed for texture generation
            
        Returns:
            RGB image (3, 28, 28) in [0, 1]
        """
        # Get color RGB values
        color_rgb = torch.tensor(COLORS[color_id], dtype=torch.float32).view(3, 1, 1)
        
        # Create mask from grayscale (digit pixels have higher intensity)
        mask = gray_img  # (1, 28, 28)
        
        # Foreground: digit stroke colored with chosen color
        # Color intensity proportional to grayscale intensity
        foreground = mask * color_rgb  # (3, 28, 28)
        
        # Background: textured, with subtle complement or neutral tint
        texture = self._create_texture((28, 28), texture_seed)  # (1, 28, 28)
        texture_rgb = texture.expand(3, -1, -1)  # Grayscale texture
        
        # Add slight color tint to background (use a neutral/complementary shade)
        # Keep it subtle so digit stands out
        bg_tint = torch.tensor([0.15, 0.15, 0.15], dtype=torch.float32).view(3, 1, 1)
        background = texture_rgb * (1 + bg_tint)  # Subtle tinting
        
        # Composite: foreground over background
        # Use mask as alpha: final = fg * mask + bg * (1 - mask)
        final = foreground + background * (1 - mask)
        
        # Clamp to valid range
        final = torch.clamp(final, 0.0, 1.0)
        
        return final
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get a colored MNIST sample.
        
        Returns:
            Tuple of (image, digit_label, color_id)
        """
        mnist_idx = self.indices[idx]
        gray_img, label = self.mnist[mnist_idx]
        label = int(label)
        color_id = self.color_ids[idx]
        
        # Create unique texture seed from dataset seed and sample index
        texture_seed = self.seed * 100000 + mnist_idx
        
        # Apply coloring
        colored_img = self._apply_color(gray_img, color_id, texture_seed)
        
        return colored_img, label, color_id


def compute_color_digit_matrix(dataset: ColoredMNIST) -> np.ndarray:
    """
    Compute P(color_id | digit) matrix (fast version using pre-computed assignments).
    
    Args:
        dataset: ColoredMNIST dataset
        
    Returns:
        10x10 numpy array where M[d, c] = P(color=c | digit=d)
    """
    counts = np.zeros((10, 10), dtype=np.float64)
    
    # Use pre-computed color_ids and labels (avoids image generation overhead)
    for i, mnist_idx in enumerate(dataset.indices):
        _, label = dataset.mnist[mnist_idx]
        label = int(label)
        color_id = dataset.color_ids[i]
        counts[label, color_id] += 1
    
    # Normalize rows to get conditional probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    probs = counts / row_sums
    
    return probs


def compute_color_digit_matrix_slow(dataset: ColoredMNIST) -> np.ndarray:
    """
    Compute P(color_id | digit) matrix (slow version, iterates through full dataset).
    Use this only if you need to verify image generation is correct.
    """
    counts = np.zeros((10, 10), dtype=np.float64)
    
    for idx in range(len(dataset)):
        _, label, color_id = dataset[idx]
        counts[label, color_id] += 1
    
    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs = counts / row_sums
    
    return probs


def print_color_digit_matrix(matrix: np.ndarray, title: str = "P(color | digit)"):
    """Pretty print the color-digit probability matrix."""
    print(f"\n{title}")
    print("=" * 60)
    print("       ", end="")
    for c in range(10):
        print(f"  C{c}  ", end="")
    print()
    print("-" * 60)
    for d in range(10):
        print(f"D{d}:   ", end="")
        for c in range(10):
            val = matrix[d, c]
            if val > 0.5:
                print(f" {val:.2f}*", end="")
            elif val > 0.01:
                print(f" {val:.2f} ", end="")
            else:
                print(f"  .   ", end="")
        print()
    print("=" * 60)


def compute_dominant_rate(dataset: ColoredMNIST) -> float:
    """Compute mean(color_id == dominant(label)) for a dataset."""
    matches = 0
    total = len(dataset)
    
    for idx in range(total):
        _, label, color_id = dataset[idx]
        if color_id == get_dominant_color(label):
            matches += 1
    
    return matches / total if total > 0 else 0.0


def get_color_name(color_id: int) -> str:
    """Get human-readable color name."""
    names = ["Red", "Green", "Blue", "Yellow", "Magenta", 
             "Cyan", "Orange", "Purple", "DarkGreen", "Gray"]
    return names[color_id]


if __name__ == "__main__":
    # Quick test
    print("Testing ColoredMNIST dataset...")
    
    train_ds = ColoredMNIST(root="./data", split="train", seed=42)
    test_ds = ColoredMNIST(root="./data", split="hard_test", seed=42)
    
    print(f"Train size: {len(train_ds)}")
    print(f"Hard test size: {len(test_ds)}")
    
    # Check a sample
    img, label, color_id = train_ds[0]
    print(f"Sample shape: {img.shape}, dtype: {img.dtype}")
    print(f"Label: {label}, Color: {color_id} ({get_color_name(color_id)})")
    
    # Compute matrices
    train_matrix = compute_color_digit_matrix(train_ds)
    test_matrix = compute_color_digit_matrix(test_ds)
    
    print_color_digit_matrix(train_matrix, "Train P(color | digit)")
    print_color_digit_matrix(test_matrix, "Hard Test P(color | digit)")
    
    print(f"\nDominant color rate (train): {compute_dominant_rate(train_ds):.4f}")
    print(f"Dominant color rate (hard_test): {compute_dominant_rate(test_ds):.4f}")
