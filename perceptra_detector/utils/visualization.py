"""
Visualization utilities for drawing detections on images.
"""

from typing import Optional, Tuple, List
import numpy as np
import random

from ..core.schemas import DetectionResult, Detection


def generate_colors(num_classes: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    """
    Generate distinct colors for each class.
    
    Args:
        num_classes: Number of classes
        seed: Random seed for reproducibility
        
    Returns:
        List of RGB color tuples
    """
    random.seed(seed)
    colors = []
    for _ in range(num_classes):
        colors.append((
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ))
    return colors


def draw_detections(
    image: np.ndarray,
    result: DetectionResult,
    show_confidence: bool = True,
    show_class_name: bool = True,
    thickness: int = 2,
    font_scale: float = 0.5,
    colors: Optional[List[Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draw detections on an image.
    
    Args:
        image: Input image
        result: Detection result
        show_confidence: Show confidence scores
        show_class_name: Show class names
        thickness: Line thickness
        font_scale: Font scale for text
        colors: Custom colors (one per class)
        
    Returns:
        Annotated image
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required. Install with: pip install opencv-python")
    
    # Copy image to avoid modifying original
    annotated = image.copy()
    
    # Generate colors if not provided
    if colors is None:
        num_classes = max([d.class_id for d in result.detections], default=0) + 1
        colors = generate_colors(num_classes)
    
    # Draw each detection
    for detection in result.detections:
        bbox = detection.bbox
        color = colors[detection.class_id % len(colors)]
        
        # Draw bounding box
        pt1 = (int(bbox.x1), int(bbox.y1))
        pt2 = (int(bbox.x2), int(bbox.y2))
        cv2.rectangle(annotated, pt1, pt2, color, thickness)
        
        # Draw mask if available
        if detection.mask is not None:
            draw_mask(annotated, detection.mask, color, alpha=0.4)
        
        # Prepare label
        label_parts = []
        if show_class_name:
            label_parts.append(detection.class_name)
        if show_confidence:
            label_parts.append(f"{detection.confidence:.2f}")
        
        if label_parts:
            label = " ".join(label_parts)
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Draw label background
            label_pt1 = (int(bbox.x1), int(bbox.y1) - text_height - baseline - 5)
            label_pt2 = (int(bbox.x1) + text_width, int(bbox.y1))
            cv2.rectangle(annotated, label_pt1, label_pt2, color, -1)
            
            # Draw label text
            text_pt = (int(bbox.x1), int(bbox.y1) - baseline - 2)
            cv2.putText(
                annotated, label, text_pt,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA
            )
    
    return annotated


def draw_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.4
) -> None:
    """
    Draw segmentation mask on image (in-place).
    
    Args:
        image: Image to draw on
        mask: Binary mask
        color: RGB color
        alpha: Transparency (0-1)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required")
    
    # Resize mask to image size if needed
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0.5] = color
    
    # Blend with image
    cv2.addWeighted(colored_mask, alpha, image, 1.0, 0, image)


def create_grid(
    images: List[np.ndarray],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    padding: int = 10,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Create a grid of images.
    
    Args:
        images: List of images
        rows: Number of rows (auto-calculated if None)
        cols: Number of columns (auto-calculated if None)
        padding: Padding between images
        background_color: Background color
        
    Returns:
        Grid image
    """
    if not images:
        raise ValueError("No images provided")
    
    num_images = len(images)
    
    # Calculate grid dimensions
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    elif rows is None:
        rows = int(np.ceil(num_images / cols))
    elif cols is None:
        cols = int(np.ceil(num_images / rows))
    
    # Get max dimensions
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    
    # Create grid
    grid_h = rows * max_h + (rows + 1) * padding
    grid_w = cols * max_w + (cols + 1) * padding
    grid = np.full((grid_h, grid_w, 3), background_color, dtype=np.uint8)
    
    # Place images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        y = row * max_h + (row + 1) * padding
        x = col * max_w + (col + 1) * padding
        
        h, w = img.shape[:2]
        grid[y:y+h, x:x+w] = img
    
    return grid


def draw_fps(
    image: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw FPS counter on image.
    
    Args:
        image: Input image
        fps: FPS value
        position: Text position (x, y)
        font_scale: Font scale
        color: Text color
        thickness: Text thickness
        
    Returns:
        Annotated image
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required")
    
    annotated = image.copy()
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        annotated, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        color, thickness, cv2.LINE_AA
    )
    return annotated


def draw_stats(
    image: np.ndarray,
    result: DetectionResult,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.7
) -> np.ndarray:
    """
    Draw detection statistics on image.
    
    Args:
        image: Input image
        result: Detection result
        position: Starting position for text
        font_scale: Font scale
        color: Text color
        thickness: Text thickness
        bg_color: Background color
        bg_alpha: Background transparency
        
    Returns:
        Annotated image
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required")
    
    annotated = image.copy()
    
    # Prepare statistics
    stats = [
        f"Detections: {len(result)}",
        f"Inference: {result.inference_time*1000:.1f}ms",
        f"Model: {result.model_name}",
    ]
    
    # Add class counts
    class_counts = result.get_class_counts()
    if class_counts:
        stats.append("Classes:")
        for class_name, count in sorted(class_counts.items()):
            stats.append(f"  {class_name}: {count}")
    
    # Calculate background size
    line_height = int(30 * font_scale)
    max_width = 0
    for line in stats:
        (w, h), _ = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        max_width = max(max_width, w)
    
    # Draw semi-transparent background
    x, y = position
    bg_height = len(stats) * line_height + 20
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (x - 10, y - 20),
        (x + max_width + 10, y + bg_height),
        bg_color, -1
    )
    cv2.addWeighted(overlay, bg_alpha, annotated, 1 - bg_alpha, 0, annotated)
    
    # Draw text
    current_y = y
    for line in stats:
        cv2.putText(
            annotated, line, (x, current_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            color, thickness, cv2.LINE_AA
        )
        current_y += line_height
    
    return annotated