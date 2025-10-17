"""
Image processing utilities.
"""

from typing import Union, Tuple
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_image(
    image_path: Union[str, Path],
    target_size: Union[Tuple[int, int], None] = None,
    rgb: bool = True
) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to image file
        target_size: Optional target size (width, height)
        rgb: Convert to RGB if True, otherwise BGR
        
    Returns:
        Image as numpy array (H, W, C)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required. Install with: pip install Pillow")
    
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize if requested
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Convert to BGR if requested
    if not rgb:
        img_array = img_array[:, :, ::-1]
    
    return img_array


def save_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    quality: int = 95
) -> None:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array
        output_path: Path to save image
        quality: JPEG quality (1-100)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required. Install with: pip install Pillow")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to PIL Image
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    img = Image.fromarray(image)
    img.save(output_path, quality=quality)
    logger.info(f"Saved image to {output_path}")


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize an image.
    
    Args:
        image: Image as numpy array
        target_size: Target size (width, height)
        keep_aspect_ratio: Maintain aspect ratio
        
    Returns:
        Resized image
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required. Install with: pip install Pillow")
    
    img = Image.fromarray(image)
    
    if keep_aspect_ratio:
        img.thumbnail(target_size, Image.BILINEAR)
    else:
        img = img.resize(target_size, Image.BILINEAR)
    
    return np.array(img)


def crop_image(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Crop image to bounding box.
    
    Args:
        image: Image as numpy array
        bbox: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]


def pad_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: int = 114
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad image to target size while maintaining aspect ratio.
    
    Args:
        image: Image as numpy array
        target_size: Target size (width, height)
        pad_value: Value to use for padding
        
    Returns:
        Tuple of (padded image, (pad_x, pad_y))
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = resize_image(image, (new_w, new_h), keep_aspect_ratio=False)
    
    # Calculate padding
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    # Create padded image
    if len(image.shape) == 3:
        padded = np.full((target_h, target_w, image.shape[2]), pad_value, dtype=image.dtype)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    else:
        padded = np.full((target_h, target_w), pad_value, dtype=image.dtype)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    return padded, (pad_w, pad_h)


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Normalize image with mean and std.
    
    Args:
        image: Image as numpy array (0-255)
        mean: Mean values for each channel
        std: Std values for each channel
        
    Returns:
        Normalized image
    """
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(mean)) / np.array(std)
    return image.astype(np.float32)