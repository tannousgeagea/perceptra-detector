"""
Backend implementations for different model types.
"""

# Import to register backends
try:
    from . import yolo
except ImportError:
    pass

try:
    from . import rf_detr
except ImportError:
    pass

try:
    from . import detr
except ImportError:
    pass

__all__ = ["yolo", "rf_detr", "detr"]