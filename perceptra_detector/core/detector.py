"""
Main Detector class - unified interface for all detection models.
"""

from typing import Union, Optional, List, Dict, Any
from pathlib import Path
import numpy as np
import logging

from .base import BaseDetector
from .registry import DetectorRegistry
from .schemas import DetectionResult, BatchDetectionResult

logger = logging.getLogger(__name__)


class Detector:
    """
    Unified detector interface supporting multiple backends.
    
    This is the main entry point for users. It automatically selects
    the appropriate backend based on model type or file extension.
    
    Example:
        >>> detector = Detector("yolov8n.pt")
        >>> result = detector.detect(image)
        >>> print(f"Found {len(result)} objects")
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        backend: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        auto_warmup: bool = False,
        **backend_kwargs
    ):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to model file or model identifier
            backend: Backend to use ('yolo', 'detr', 'rt-detr', etc.)
                    If None, auto-detect from file extension
            device: Device for inference ('cuda', 'cpu', or None for auto)
            confidence_threshold: Default confidence threshold
            iou_threshold: Default IoU threshold for NMS
            auto_warmup: Automatically warmup model after loading
            **backend_kwargs: Additional backend-specific parameters
        """
        self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
        
        # Auto-detect backend if not specified
        if backend is None:
            from .model_inspector import detect_backend as smart_detect
            backend = smart_detect(self.model_path, hint=None)
            logger.info(f"Auto-detected backend: {backend}")
        else:
            logger.info(f"Using specified backend: {backend}")
        
        # Get backend class
        backend_class = DetectorRegistry.get_backend(backend)
        
        # Initialize backend
        self.backend: BaseDetector = backend_class(
            model_path=self.model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            **backend_kwargs
        )
        
        self.backend_name = backend
        self._loaded = False
        
        # Warmup if requested
        if auto_warmup:
            self.warmup()
    
    def load(self) -> None:
        """Explicitly load the model."""
        if not self._loaded:
            self.backend.load_model()
            self._loaded = True
    
    def detect(
        self,
        image: Union[np.ndarray, str, Path],
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> DetectionResult:
        """
        Run detection on a single image.
        
        Args:
            image: Input image (numpy array, file path, or URL)
            confidence_threshold: Override confidence threshold
            iou_threshold: Override IoU threshold
            
        Returns:
            DetectionResult object
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            from ..utils.image import load_image
            image = load_image(image)
        
        # Ensure model is loaded
        if not self._loaded:
            self.load()
        
        # Run detection
        return self.backend.detect(
            image,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
    
    def detect_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> BatchDetectionResult:
        """
        Run detection on multiple images.
        
        Args:
            images: List of images (numpy arrays or file paths)
            confidence_threshold: Override confidence threshold
            iou_threshold: Override IoU threshold
            
        Returns:
            BatchDetectionResult object
        """
        import time
        
        # Load images if paths provided
        processed_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                from ..utils.image import load_image
                img = load_image(img)
            processed_images.append(img)
        
        # Ensure model is loaded
        if not self._loaded:
            self.load()
        
        # Run batch detection
        start_time = time.time()
        results = self.backend.detect_batch(
            processed_images,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        total_time = time.time() - start_time
        
        return BatchDetectionResult(
            results=results,
            total_inference_time=total_time
        )
    
    def detect_directory(
        self,
        directory: Union[str, Path],
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        recursive: bool = False
    ) -> BatchDetectionResult:
        """
        Run detection on all images in a directory.
        
        Args:
            directory: Path to directory containing images
            extensions: List of valid image extensions
            confidence_threshold: Override confidence threshold
            iou_threshold: Override IoU threshold
            recursive: Search subdirectories recursively
            
        Returns:
            BatchDetectionResult object
        """
        directory = Path(directory)
        
        # Find all images
        image_paths = []
        if recursive:
            for ext in extensions:
                image_paths.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                image_paths.extend(directory.glob(f"*{ext}"))
        
        if not image_paths:
            logger.warning(f"No images found in {directory}")
            return BatchDetectionResult(results=[], total_inference_time=0.0)
        
        logger.info(f"Found {len(image_paths)} images in {directory}")
        
        return self.detect_batch(
            image_paths,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
    
    def detect_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        skip_frames: int = 0,
        max_frames: Optional[int] = None,
        show_progress: bool = True
    ) -> List[DetectionResult]:
        """
        Run detection on video frames.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save annotated video
            confidence_threshold: Override confidence threshold
            iou_threshold: Override IoU threshold
            skip_frames: Process every nth frame (0 = process all)
            max_frames: Maximum number of frames to process
            show_progress: Show progress bar
            
        Returns:
            List of DetectionResult objects (one per processed frame)
        """
        from ..utils.video import process_video
        
        # Ensure model is loaded
        if not self._loaded:
            self.load()
        
        return process_video(
            video_path=video_path,
            detector=self.backend,
            output_path=output_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            skip_frames=skip_frames,
            max_frames=max_frames,
            show_progress=show_progress
        )
    
    def warmup(self, num_iterations: int = 3, image_size: tuple = (640, 640)) -> None:
        """
        Warmup the model with dummy inputs.
        
        Args:
            num_iterations: Number of warmup iterations
            image_size: Size of dummy images
        """
        logger.info("Warming up model...")
        self.backend.warmup(num_iterations, image_size)
        self._loaded = True
        logger.info("Warmup complete")
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = self.backend.model_info
        info['backend'] = self.backend_name
        return info
    
    @property
    def class_names(self) -> List[str]:
        """Get list of class names."""
        if not self._loaded:
            self.load()
        return self.backend.class_names
    
    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.class_names)
    
    def __repr__(self) -> str:
        return (
            f"Detector(backend={self.backend_name}, "
            f"model={self.model_path.name}, "
            f"device={self.backend.device})"
        )
    
    @staticmethod
    def list_backends() -> List[str]:
        """List all available backends."""
        return DetectorRegistry.list_backends()
    
    @staticmethod
    def list_supported_extensions() -> Dict[str, str]:
        """Get mapping of supported file extensions to backends."""
        return DetectorRegistry.get_supported_extensions()