"""
Preprocessing utilities for demo app.
Converts canvas drawings to ColoredMNIST-compatible tensors.
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional

# Match the colors from src/data_colored_mnist.py
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

COLOR_NAMES = [
    "Red", "Green", "Blue", "Yellow", "Magenta",
    "Cyan", "Orange", "Purple", "DarkGreen", "Gray"
]


def extract_stroke_mask(image: np.ndarray, threshold: int = 50) -> np.ndarray:
    """
    Extract binary mask of strokes from canvas image.
    
    Args:
        image: RGBA or grayscale numpy array from canvas
        threshold: Intensity threshold for stroke detection
        
    Returns:
        Binary mask (H, W) with 1s where strokes exist
    """
    if image is None:
        return np.zeros((28, 28), dtype=np.float32)
    
    # Handle different formats
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            # RGBA - use RGB intensity (white strokes on black background)
            # The strokes are white (255,255,255) on black (0,0,0)
            rgb = image[:, :, :3].astype(np.float32)
            # Use max of RGB channels to detect any white/bright pixels
            mask = np.max(rgb, axis=2)
        else:
            # RGB - convert to grayscale via max intensity
            mask = np.max(image[:, :, :3].astype(np.float32), axis=2)
    else:
        mask = image.astype(np.float32)
    
    # Threshold (strokes are typically 255, background is 0)
    mask = (mask > threshold).astype(np.float32)
    return mask


def center_and_resize(mask: np.ndarray, target_size: int = 28, padding: int = 4) -> np.ndarray:
    """
    Center the digit in the bounding box and resize to target size.
    
    Args:
        mask: Binary mask of the stroke
        target_size: Target output size
        padding: Padding around the digit
        
    Returns:
        Centered and resized mask (target_size, target_size)
    """
    # Find bounding box
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return np.zeros((target_size, target_size), dtype=np.float32)
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # Crop
    cropped = mask[y_min:y_max+1, x_min:x_max+1]
    
    # Make square with padding
    h, w = cropped.shape
    size = max(h, w) + padding * 2
    padded = np.zeros((size, size), dtype=np.float32)
    
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    
    # Resize to target
    pil_img = Image.fromarray((padded * 255).astype(np.uint8))
    pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    resized = np.array(pil_img).astype(np.float32) / 255.0
    
    return resized


def colorize_mask(mask: np.ndarray, color_id: int) -> np.ndarray:
    """
    Apply color to mask to create RGB image.
    
    Args:
        mask: Grayscale mask (H, W) in [0, 1]
        color_id: Index into COLORS list
        
    Returns:
        RGB image (H, W, 3) in [0, 1]
    """
    color = COLORS[color_id]
    mask_3d = np.stack([mask] * 3, axis=-1)
    
    # Foreground: colored digit
    foreground = mask_3d * np.array(color)
    
    # Background: dark noise (like the dataset)
    background = (1 - mask_3d) * np.random.uniform(0, 0.15, mask_3d.shape)
    
    image = np.clip(foreground + background, 0, 1).astype(np.float32)
    return image


def canvas_to_tensor(
    canvas_image: np.ndarray,
    color_id: int,
    target_size: int = 28
) -> torch.Tensor:
    """
    Full pipeline: canvas image -> model input tensor.
    
    Args:
        canvas_image: RGBA image from canvas
        color_id: Color to apply to the digit
        target_size: Output image size
        
    Returns:
        Tensor of shape (1, 3, target_size, target_size) in [0, 1]
    """
    # Extract mask
    mask = extract_stroke_mask(canvas_image)
    
    # Center and resize
    mask = center_and_resize(mask, target_size)
    
    # Colorize
    rgb = colorize_mask(mask, color_id)
    
    # Convert to tensor: (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    
    return tensor.float()


def recolor_tensor(tensor: torch.Tensor, new_color_id: int) -> torch.Tensor:
    """
    Recolor an existing tensor to a new color.
    
    Args:
        tensor: Input tensor (1, 3, H, W)
        new_color_id: New color index
        
    Returns:
        Recolored tensor (1, 3, H, W)
    """
    # Extract mask from max intensity
    img = tensor.squeeze(0)
    mask = img.max(dim=0)[0]  # (H, W)
    mask = (mask > 0.3).float()
    
    # Apply new color
    color = torch.tensor(COLORS[new_color_id]).view(3, 1, 1)
    foreground = mask.unsqueeze(0) * color
    background = (1 - mask.unsqueeze(0)) * 0.1
    
    recolored = torch.clamp(foreground + background, 0, 1)
    return recolored.unsqueeze(0)
