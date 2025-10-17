"""
Registry system for detector backends.
"""

from typing import Dict, Type, List, Optional
from pathlib import Path
import logging

from .base import BaseDetector

logger = logging.getLogger(__name__)


class DetectorRegistry:
    """
    Registry for detector backends.
    
    Allows registering new detector types and automatically
    selecting the appropriate backend based on model file.
    """
    
    _backends: Dict[str, Type[BaseDetector]] = {}
    _file_extensions: Dict[str, str] = {}  # Map file extension to backend name
    
    @classmethod
    def register(
        cls,
        name: str,
        backend_class: Type[BaseDetector],
        file_extensions: Optional[List[str]] = None
    ) -> None:
        """
        Register a new detector backend.
        
        Args:
            name: Backend name (e.g., 'yolo', 'detr')
            backend_class: Detector class inheriting from BaseDetector
            file_extensions: List of file extensions this backend handles
                           (e.g., ['.pt', '.onnx'])
        """
        if not issubclass(backend_class, BaseDetector):
            raise TypeError(
                f"Backend class must inherit from BaseDetector, "
                f"got {backend_class}"
            )
        
        cls._backends[name.lower()] = backend_class
        logger.info(f"Registered detector backend: {name}")
        
        if file_extensions:
            for ext in file_extensions:
                ext = ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                cls._file_extensions[ext] = name.lower()
                logger.debug(f"Mapped {ext} -> {name}")
    
    @classmethod
    def get_backend(cls, name: str) -> Type[BaseDetector]:
        """
        Get a detector backend by name.
        
        Args:
            name: Backend name
            
        Returns:
            Detector class
            
        Raises:
            KeyError: If backend not found
        """
        name = name.lower()
        if name not in cls._backends:
            available = ', '.join(cls._backends.keys())
            raise KeyError(
                f"Backend '{name}' not found. "
                f"Available backends: {available}"
            )
        return cls._backends[name]
    
    @classmethod
    def detect_backend(cls, model_path: Path) -> Optional[str]:
        """
        Auto-detect backend from model file extension.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Backend name or None if not detected
        """
        ext = model_path.suffix.lower()
        return cls._file_extensions.get(ext)
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backends."""
        return list(cls._backends.keys())
    
    @classmethod
    def get_supported_extensions(cls) -> Dict[str, str]:
        """Get mapping of file extensions to backends."""
        return cls._file_extensions.copy()
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a backend.
        
        Args:
            name: Backend name
        """
        name = name.lower()
        if name in cls._backends:
            del cls._backends[name]
            logger.info(f"Unregistered backend: {name}")
            
            # Remove associated file extensions
            to_remove = [
                ext for ext, backend in cls._file_extensions.items()
                if backend == name
            ]
            for ext in to_remove:
                del cls._file_extensions[ext]


def register_backend(
    name: str,
    file_extensions: Optional[List[str]] = None
):
    """
    Decorator to register a detector backend.
    
    Usage:
        @register_backend('yolo', ['.pt', '.onnx'])
        class YOLODetector(BaseDetector):
            ...
    
    Args:
        name: Backend name
        file_extensions: List of supported file extensions
    """
    def decorator(cls: Type[BaseDetector]):
        DetectorRegistry.register(name, cls, file_extensions)
        return cls
    return decorator