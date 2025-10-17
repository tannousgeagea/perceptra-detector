"""
DETR (DEtection TRansformer) and RT-DETR backend implementation.
"""

from typing import Optional, Union, Any, List
from pathlib import Path
import numpy as np
import torch
import logging

from ..core.base import BaseDetector
from ..core.registry import register_backend
from ..core.schemas import (
    DetectionResult, Detection, BoundingBox, TaskType
)

logger = logging.getLogger(__name__)


@register_backend('rt-detr', ['.pt', '.pth'])
class RTDETRDetector(BaseDetector):
    """
    RT-DETR (Real-Time DETR) detector backend.
    
    Can use either ultralytics implementation or custom PyTorch models.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        use_ultralytics: bool = True,
        **kwargs
    ):
        """
        Initialize RT-DETR detector.
        
        Args:
            model_path: Path to RT-DETR model
            device: Device for inference
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold
            use_ultralytics: Use ultralytics implementation
            **kwargs: Additional parameters
        """
        super().__init__(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            **kwargs
        )
        self.use_ultralytics = use_ultralytics
        self.extra_kwargs = kwargs
    
    def load_model(self) -> None:
        """Load RT-DETR model."""
        if self.use_ultralytics:
            try:
                from ultralytics import RTDETR
            except ImportError:
                raise ImportError(
                    "ultralytics is required for RT-DETR. "
                    "Install with: pip install ultralytics"
                )
            
            self.model = RTDETR(str(self.model_path))
            self.model.to(self.device)
            
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            else:
                from ..utils.coco_classes import COCO_CLASSES
                self.class_names = COCO_CLASSES
        else:
            raise NotImplementedError(
                "Custom RT-DETR loading not implemented. "
                "Use use_ultralytics=True"
            )
        
        self._model_info = {
            "framework": "ultralytics" if self.use_ultralytics else "pytorch",
            "model_type": "rt-detr",
        }
        
        logger.info(f"Loaded RT-DETR model on device {self.device}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess for RT-DETR (handled internally by ultralytics)."""
        return image
    
    def predict(self, preprocessed_input: np.ndarray) -> Any:
        """Run RT-DETR inference."""
        results = self.model.predict(
            source=preprocessed_input,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
            **self.extra_kwargs
        )
        return results[0]
    
    def postprocess(
        self,
        predictions: Any,
        original_shape: tuple
    ) -> DetectionResult:
        """Convert RT-DETR predictions to standard format."""
        detections = []
        
        boxes = predictions.boxes
        
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(xyxy, confidences, class_ids):
                bbox = BoundingBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3])
                )
                
                detection = Detection(
                    bbox=bbox,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=self.class_names[cls_id]
                )
                detections.append(detection)
        
        return DetectionResult(
            detections=detections,
            image_shape=original_shape,
            task_type=TaskType.DETECTION,
            model_name=f"RT-DETR ({self.model_path.name})",
            inference_time=0.0
        )