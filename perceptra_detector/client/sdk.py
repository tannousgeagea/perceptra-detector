"""
Python SDK client for Perceptra Detector API.
"""

from typing import Union, Optional, List, Dict, Any
from pathlib import Path
import requests
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DetectorClient:
    """
    Client for interacting with Perceptra Detector API.
    
    Example:
        >>> client = DetectorClient("http://localhost:8000")
        >>> result = client.detect("image.jpg")
        >>> print(f"Found {result['num_detections']} objects")
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        verify_ssl: bool = True
    ):
        """
        Initialize client.
        
        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds
            verify_ssl: Verify SSL certificates
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health.
        
        Returns:
            Health status information
        """
        response = self.session.get(
            f"{self.base_url}/health",
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """
        List available models.
        
        Returns:
            Dictionary with 'models' and 'default_model'
        """
        response = self.session.get(
            f"{self.base_url}/models",
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary
        """
        response = self.session.get(
            f"{self.base_url}/models/{model_name}",
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()
        return response.json()
    
    def detect(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        model_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run detection on an image.
        
        Args:
            image: Image as file path, bytes, or numpy array
            model_name: Model to use (uses default if None)
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold
            
        Returns:
            Detection results dictionary
        """
        # Prepare image data
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            filename = image_path.name
        elif isinstance(image, bytes):
            image_bytes = image
            filename = "image.jpg"
        elif isinstance(image, np.ndarray):
            from PIL import Image
            import io
            img = Image.fromarray(image)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            image_bytes = buffer.getvalue()
            filename = "image.jpg"
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Prepare form data
        files = {'file': (filename, image_bytes, 'image/jpeg')}
        data = {}
        
        if model_name is not None:
            data['model_name'] = model_name
        if confidence_threshold is not None:
            data['confidence_threshold'] = confidence_threshold
        if iou_threshold is not None:
            data['iou_threshold'] = iou_threshold
        
        # Make request
        response = self.session.post(
            f"{self.base_url}/detect",
            files=files,
            data=data,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()
        return response.json()
    
    def detect_batch(
        self,
        images: List[Union[str, Path, bytes, np.ndarray]],
        model_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run detection on multiple images.
        
        Args:
            images: List of images
            model_name: Model to use
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold
            
        Returns:
            Batch detection results dictionary
        """
        # Prepare files
        files = []
        for idx, image in enumerate(images):
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                filename = image_path.name
            elif isinstance(image, bytes):
                image_bytes = image
                filename = f"image_{idx}.jpg"
            elif isinstance(image, np.ndarray):
                from PIL import Image
                import io
                img = Image.fromarray(image)
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
                filename = f"image_{idx}.jpg"
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")
            
            files.append(('files', (filename, image_bytes, 'image/jpeg')))
        
        # Prepare form data
        data = {}
        if model_name is not None:
            data['model_name'] = model_name
        if confidence_threshold is not None:
            data['confidence_threshold'] = confidence_threshold
        if iou_threshold is not None:
            data['iou_threshold'] = iou_threshold
        
        # Make request
        response = self.session.post(
            f"{self.base_url}/detect/batch",
            files=files,
            data=data,
            timeout=self.timeout * 2,  # Longer timeout for batch
            verify=self.verify_ssl
        )
        response.raise_for_status()
        return response.json()
    
    def detect_url(
        self,
        url: str,
        model_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run detection on an image from URL.
        
        Args:
            url: Image URL
            model_name: Model to use
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold
            
        Returns:
            Detection results dictionary
        """
        params = {'url': url}
        if model_name is not None:
            params['model_name'] = model_name
        if confidence_threshold is not None:
            params['confidence_threshold'] = confidence_threshold
        if iou_threshold is not None:
            params['iou_threshold'] = iou_threshold
        
        response = self.session.post(
            f"{self.base_url}/detect/url",
            params=params,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close the session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __repr__(self) -> str:
        return f"DetectorClient(base_url={self.base_url})"