"""
Base detector interface that all backend implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
import numpy as np
import torch

from .schemas import DetectionResult, TaskType


class BaseDetector(ABC):
    """
    Abstract base class for all detector backends.
    
    All detector implementations (YOLO, DETR, etc.) must inherit from this class
    and implement its abstract methods.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        **kwargs
    ):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to the model file
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            **kwargs: Additional backend-specific parameters
        """
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = []
        self.task_type = TaskType.DETECTION
        
        # Model info
        self._model_loaded = False
        self._model_info = {}
    
    @staticmethod
    def _setup_device(device: Optional[str] = None) -> str:
        """Setup device for inference."""
        if device is not None:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model from disk.
        
        This method should:
        1. Load the model weights
        2. Move model to the correct device
        3. Set model to evaluation mode
        4. Extract class names
        5. Set task type (detection/segmentation)
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Any:
        """
        Preprocess an image for inference.
        
        Args:
            image: Input image as numpy array (HWC, RGB)
            
        Returns:
            Preprocessed input ready for the model
        """
        pass
    
    @abstractmethod
    def predict(self, preprocessed_input: Any) -> Any:
        """
        Run model inference.
        
        Args:
            preprocessed_input: Output from preprocess()
            
        Returns:
            Raw model predictions
        """
        pass
    
    @abstractmethod
    def postprocess(
        self,
        predictions: Any,
        original_shape: tuple
    ) -> DetectionResult:
        """
        Postprocess model predictions into standard format.
        
        Args:
            predictions: Raw model predictions from predict()
            original_shape: Original image shape (H, W, C)
            
        Returns:
            DetectionResult object with standardized format
        """
        pass
    
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> DetectionResult:
        """
        Run end-to-end detection on an image.
        
        Args:
            image: Input image as numpy array (HWC, RGB)
            confidence_threshold: Override default confidence threshold
            iou_threshold: Override default IoU threshold
            
        Returns:
            DetectionResult object
        """
        import time
        
        if not self._model_loaded:
            self.load_model()
            self._model_loaded = True
        
        # Override thresholds if provided
        original_conf = self.confidence_threshold
        original_iou = self.iou_threshold
        
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        
        try:
            # Timing
            start_time = time.time()
            
            # Run inference pipeline
            preprocessed = self.preprocess(image)
            predictions = self.predict(preprocessed)
            result = self.postprocess(predictions, image.shape)
            
            # Add timing info
            result.inference_time = time.time() - start_time
            
            return result
        
        finally:
            # Restore original thresholds
            self.confidence_threshold = original_conf
            self.iou_threshold = original_iou
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> List[DetectionResult]:
        """
        Run detection on a batch of images.
        
        Default implementation processes images sequentially.
        Backends can override this for true batch processing.
        
        Args:
            images: List of images as numpy arrays
            confidence_threshold: Override default confidence threshold
            iou_threshold: Override default IoU threshold
            
        Returns:
            List of DetectionResult objects
        """
        return [
            self.detect(img, confidence_threshold, iou_threshold)
            for img in images
        ]
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "task_type": self.task_type.value,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            **self._model_info
        }
    
    def warmup(self, num_iterations: int = 3, image_size: tuple = (640, 640)) -> None:
        """
        Warmup the model with dummy inputs.
        
        Useful for GPU models to initialize CUDA kernels.
        
        Args:
            num_iterations: Number of warmup iterations
            image_size: Size of dummy images (H, W)
        """
        if not self._model_loaded:
            self.load_model()
            self._model_loaded = True
        
        dummy_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        
        for _ in range(num_iterations):
            self.detect(dummy_image)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_path={self.model_path}, "
            f"device={self.device}, "
            f"task_type={self.task_type.value})"
        )