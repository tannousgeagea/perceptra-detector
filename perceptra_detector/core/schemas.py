"""
Standard schemas for detection and segmentation results.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from enum import Enum


class TaskType(str, Enum):
    """Detection task types."""
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"


@dataclass
class BoundingBox:
    """Bounding box representation in xyxy format."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_xyxy(self) -> List[float]:
        """Convert to [x1, y1, x2, y2] format."""
        return [self.x1, self.y1, self.x2, self.y2]
    
    def to_xywh(self) -> List[float]:
        """Convert to [x, y, width, height] format."""
        return [self.x1, self.y1, self.width, self.height]
    
    def to_cxcywh(self) -> List[float]:
        """Convert to [center_x, center_y, width, height] format."""
        cx, cy = self.center
        return [cx, cy, self.width, self.height]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Detection:
    """Single detection result."""
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[np.ndarray] = None  # For instance segmentation
    track_id: Optional[int] = None  # For tracking (future)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "bbox": self.bbox.to_dict(),
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
        }
        
        if self.track_id is not None:
            result["track_id"] = int(self.track_id)
        
        if self.mask is not None:
            result["has_mask"] = True
            result["mask_shape"] = self.mask.shape
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result


@dataclass
class DetectionResult:
    """Complete detection result for an image."""
    detections: List[Detection]
    image_shape: Tuple[int, int, int]  # (height, width, channels)
    task_type: TaskType
    model_name: str
    inference_time: float  # in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.detections)
    
    def __iter__(self):
        return iter(self.detections)
    
    def __getitem__(self, idx: int) -> Detection:
        return self.detections[idx]
    
    def filter_by_confidence(self, threshold: float) -> 'DetectionResult':
        """Filter detections by confidence threshold."""
        filtered = [d for d in self.detections if d.confidence >= threshold]
        return DetectionResult(
            detections=filtered,
            image_shape=self.image_shape,
            task_type=self.task_type,
            model_name=self.model_name,
            inference_time=self.inference_time,
            metadata=self.metadata,
        )
    
    def filter_by_class(self, class_names: List[str]) -> 'DetectionResult':
        """Filter detections by class names."""
        filtered = [d for d in self.detections if d.class_name in class_names]
        return DetectionResult(
            detections=filtered,
            image_shape=self.image_shape,
            task_type=self.task_type,
            model_name=self.model_name,
            inference_time=self.inference_time,
            metadata=self.metadata,
        )
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get count of detections per class."""
        counts = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "detections": [d.to_dict() for d in self.detections],
            "num_detections": len(self.detections),
            "image_shape": {
                "height": self.image_shape[0],
                "width": self.image_shape[1],
                "channels": self.image_shape[2],
            },
            "task_type": self.task_type.value,
            "model_name": self.model_name,
            "inference_time": self.inference_time,
            "class_counts": self.get_class_counts(),
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class BatchDetectionResult:
    """Results for batch processing."""
    results: List[DetectionResult]
    total_inference_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __iter__(self):
        return iter(self.results)
    
    def __getitem__(self, idx: int) -> DetectionResult:
        return self.results[idx]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "results": [r.to_dict() for r in self.results],
            "num_images": len(self.results),
            "total_inference_time": self.total_inference_time,
            "average_inference_time": self.total_inference_time / len(self.results) if self.results else 0,
            "metadata": self.metadata,
        }