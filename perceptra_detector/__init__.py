"""
Perceptra Detector - Production-ready object detection framework.
"""

from .__version__ import __version__, __author__, __license__

from .core.detector import Detector
from .core.schemas import (
    Detection,
    DetectionResult,
    BatchDetectionResult,
    BoundingBox,
    TaskType,
)

# Import backends to register them
from .backends import yolo, detr, rt_detr

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "Detector",
    "Detection",
    "DetectionResult",
    "BatchDetectionResult",
    "BoundingBox",
    "TaskType",
]