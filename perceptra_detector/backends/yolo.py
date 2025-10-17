"""
YOLO backend implementation (YOLOv8, YOLOv9, YOLOv11).
"""

from typing import Optional, Union, Any
from pathlib import Path
import numpy as np
import time
import logging

from ..core.base import BaseDetector
from ..core.registry import register_backend
from ..core.schemas import (
    DetectionResult, Detection, BoundingBox, TaskType
)

logger = logging.getLogger(__name__)


@register_backend('yolo', ['.pt', '.onnx'])
class YOLODetector(BaseDetector):
    """
    YOLO detector backend supporting YOLOv8, YOLOv9, and YOLOv11.
    
    Uses the ultralytics library for model loading and inference.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        **kwargs
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model (.pt or .onnx)
            device: Device for inference
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            imgsz: Input image size
            **kwargs: Additional YOLO-specific parameters
        """
        super().__init__(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            **kwargs
        )
        self.imgsz = imgsz
        self.extra_kwargs = kwargs
    
    def load_model(self) -> None:
        """Load YOLO model using ultralytics."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLO backend. "
                "Install with: pip install ultralytics"
            )
        
        logger.info(f"Loading YOLO model from {self.model_path}")
        
        # Load model
        self.model = YOLO(str(self.model_path))
        
        # Move to device
        self.model.to(self.device)
        
        # Extract class names
        if hasattr(self.model, 'names'):
            self.class_names = list(self.model.names.values())
        else:
            logger.warning("Could not extract class names from model")
            self.class_names = [f"class_{i}" for i in range(80)]  # Default COCO
        
        # Detect task type
        task = getattr(self.model, 'task', 'detect')
        if task == 'segment':
            self.task_type = TaskType.SEGMENTATION
        else:
            self.task_type = TaskType.DETECTION
        
        # Store model info
        self._model_info = {
            "framework": "ultralytics",
            "task": task,
            "input_size": self.imgsz,
        }
        
        logger.info(
            f"Loaded YOLO model: task={task}, "
            f"classes={len(self.class_names)}, device={self.device}"
        )
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO.
        
        YOLO models handle preprocessing internally,
        so we just return the image as-is.
        """
        return image
    
    def predict(self, preprocessed_input: np.ndarray) -> Any:
        """Run YOLO inference."""
        results = self.model.predict(
            source=preprocessed_input,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
            **self.extra_kwargs
        )
        return results[0]  # Return first result (single image)
    
    def postprocess(
        self,
        predictions: Any,
        original_shape: tuple
    ) -> DetectionResult:
        """Convert YOLO predictions to standard format."""
        detections = []
        
        # Extract boxes, confidences, and class IDs
        boxes = predictions.boxes
        
        if boxes is not None and len(boxes) > 0:
            # Convert to numpy for easier processing
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            
            # Handle segmentation masks if available
            masks = None
            if hasattr(predictions, 'masks') and predictions.masks is not None:
                masks = predictions.masks.data.cpu().numpy()
            
            for i, (box, conf, cls_id) in enumerate(zip(xyxy, confidences, class_ids)):
                bbox = BoundingBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3])
                )
                
                # Get mask if available
                mask = masks[i] if masks is not None else None
                
                detection = Detection(
                    bbox=bbox,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=self.class_names[cls_id],
                    mask=mask
                )
                detections.append(detection)
        
        return DetectionResult(
            detections=detections,
            image_shape=original_shape,
            task_type=self.task_type,
            model_name=f"YOLO ({self.model_path.name})",
            inference_time=0.0,  # Will be set by base class
        )
    
    def detect_batch(
        self,
        images: list,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> list:
        """
        Run batch detection using YOLO's native batch processing.
        
        This is more efficient than sequential processing.
        """
        if not self._model_loaded:
            self.load_model()
            self._model_loaded = True
        
        # Use provided thresholds or defaults
        conf = confidence_threshold or self.confidence_threshold
        iou = iou_threshold or self.iou_threshold
        
        # Run batch prediction
        results = self.model.predict(
            source=images,
            conf=conf,
            iou=iou,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
            **self.extra_kwargs
        )
        
        # Convert results
        detection_results = []
        for result, image in zip(results, images):
            det_result = self.postprocess(result, image.shape)
            detection_results.append(det_result)
        
        return detection_results