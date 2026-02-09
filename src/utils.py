"""
Shared utilities for reproducibility and common helpers.
=========================================================
"""
import torch
import numpy as np
import random

# Default seed for all experiments
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic operations (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_env_info() -> None:
    """Print environment information for debugging."""
    import sys
    print("=" * 50)
    print("ENVIRONMENT INFO")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"NumPy version: {np.__version__}")
    print("=" * 50)


if __name__ == "__main__":
    print_env_info()
    set_seed(42)
    print("Seed set to 42")
