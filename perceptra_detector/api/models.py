"""
Pydantic models for API requests and responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    """Request model for detection endpoint."""
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    iou_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    model_name: Optional[str] = None


class BoundingBoxResponse(BaseModel):
    """Bounding box in response."""
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float


class DetectionResponse(BaseModel):
    """Single detection in response."""
    bbox: BoundingBoxResponse
    confidence: float
    class_id: int
    class_name: str
    has_mask: bool = False


class DetectionResultResponse(BaseModel):
    """Detection result response."""
    detections: List[DetectionResponse]
    num_detections: int
    image_shape: Dict[str, int]
    task_type: str
    model_name: str
    inference_time: float
    class_counts: Dict[str, int]


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    backend: str
    device: str
    task_type: str
    num_classes: int
    class_names: List[str]
    confidence_threshold: float
    iou_threshold: float


class ModelsListResponse(BaseModel):
    """Response for models list endpoint."""
    models: List[str]
    default_model: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: int
    gpu_available: bool


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


class BatchDetectionRequest(BaseModel):
    """Request model for batch detection."""
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    iou_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    model_name: Optional[str] = None


class BatchDetectionResponse(BaseModel):
    """Batch detection response."""
    results: List[DetectionResultResponse]
    num_images: int
    total_inference_time: float
    average_inference_time: float