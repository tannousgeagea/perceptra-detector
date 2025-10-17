"""
Example: Integrating a custom detection model.

This example shows how to add support for a custom model architecture
by implementing the BaseDetector interface.
"""

import torch
import numpy as np
from typing import Any
from pathlib import Path

from perceptra_detector.core.base import BaseDetector
from perceptra_detector.core.registry import register_backend
from perceptra_detector.core.schemas import (
    DetectionResult, Detection, BoundingBox, TaskType
)
from perceptra_detector import Detector


@register_backend('custom-efficientdet', ['.custom'])
class CustomEfficientDetDetector(BaseDetector):
    """
    Example custom detector for EfficientDet (or any custom architecture).
    
    This is a template showing how to integrate your own models.
    """
    
    def __init__(self, model_path, device=None, **kwargs):
        super().__init__(model_path, device, **kwargs)
        self.input_size = kwargs.get('input_size', 512)
    
    def load_model(self):
        """Load your custom model."""
        print(f"Loading custom model from {self.model_path}")
        
        # Example: Load a PyTorch model
        # checkpoint = torch.load(self.model_path, map_location=self.device)
        # self.model = YourModelClass()
        # self.model.load_state_dict(checkpoint['state_dict'])
        
        # For this example, we'll create a dummy model
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                # Return dummy predictions
                batch_size = x.shape[0]
                num_detections = 5
                return {
                    'boxes': torch.rand(batch_size, num_detections, 4),
                    'scores': torch.rand(batch_size, num_detections),
                    'labels': torch.randint(0, 80, (batch_size, num_detections))
                }
        
        self.model = DummyModel()
        self.model.to(self.device)
        self.model.eval()
        
        # Set class names (example: COCO classes)
        from perceptra_detector.utils.coco_classes import COCO_CLASSES
        self.class_names = COCO_CLASSES
        
        # Set task type
        self.task_type = TaskType.DETECTION
        
        print("Custom model loaded successfully")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for your model.
        
        Args:
            image: Input image as numpy array (HWC, RGB)
            
        Returns:
            Preprocessed tensor
        """
        # Resize
        from PIL import Image
        img = Image.fromarray(image)
        img = img.resize((self.input_size, self.input_size))
        
        # Convert to tensor and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # Add batch dimension