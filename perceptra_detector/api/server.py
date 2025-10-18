"""
FastAPI application for detection service.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import numpy as np
import io
import logging
from pathlib import Path
import time
import traceback

from ..core.detector import Detector
from .models import (
    DetectionResultResponse, ModelInfo, ModelsListResponse,
    HealthResponse, ErrorResponse, BatchDetectionResponse,
    BoundingBoxResponse, DetectionResponse
)
from ..__version__ import __version__

logger = logging.getLogger(__name__)


class DetectionService:
    """Manages detector instances for the API."""
    
    def __init__(self):
        self.detectors: Dict[str, Detector] = {}
        self.default_model: Optional[str] = None
        self._load_times: Dict[str, float] = {}
        self._request_count: Dict[str, int] = {}
    
    def add_model(
        self,
        name: str,
        model_path: str,
        backend: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        """Add a model to the service."""
        logger.info(f"Loading model: {name} from {model_path}")
        start_time = time.time()
        
        try:
            detector = Detector(
                model_path=model_path,
                backend=backend,
                device=device,
                **kwargs
            )
            detector.load()  # Pre-load the model
            
            load_time = time.time() - start_time
            self.detectors[name] = detector
            self._load_times[name] = load_time
            self._request_count[name] = 0
            
            if self.default_model is None:
                self.default_model = name
            
            logger.info(f"Model {name} loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            raise
    
    def get_detector(self, name: Optional[str] = None) -> Detector:
        """Get a detector by name."""
        if name is None:
            name = self.default_model
        
        if name is None:
            raise ValueError("No models loaded and no default model set")
        
        if name not in self.detectors:
            raise ValueError(f"Model '{name}' not found. Available: {list(self.detectors.keys())}")
        
        # Increment request count
        self._request_count[name] = self._request_count.get(name, 0) + 1
        
        return self.detectors[name]
    
    def list_models(self) -> List[str]:
        """List available models."""
        return list(self.detectors.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "models_loaded": len(self.detectors),
            "default_model": self.default_model,
            "load_times": self._load_times,
            "request_counts": self._request_count,
            "total_requests": sum(self._request_count.values())
        }
    
    def remove_model(self, name: str) -> bool:
        """Remove a model from service."""
        if name in self.detectors:
            del self.detectors[name]
            if name in self._load_times:
                del self._load_times[name]
            if name in self._request_count:
                del self._request_count[name]
            
            # Update default if needed
            if self.default_model == name:
                self.default_model = next(iter(self.detectors.keys()), None)
            
            logger.info(f"Model {name} removed")
            return True
        return False


# Global service instance
service = DetectionService()


def create_app(
    models_config: Optional[Dict[str, str]] = None,
    enable_cors: bool = True,
    cors_origins: List[str] = ["*"],
    max_file_size_mb: int = 10,
    **kwargs
) -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        models_config: Dict mapping model names to paths
        enable_cors: Enable CORS middleware
        cors_origins: Allowed CORS origins
        max_file_size_mb: Maximum file size in MB
        **kwargs: Additional FastAPI parameters
        
    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Perceptra Detector API",
        description="Production-ready object detection and segmentation API",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        **kwargs
    )
    
    # Store config
    app.state.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
    
    # Enable CORS
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"{request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)
            
            # Log response
            logger.info(
                f"{request.method} {request.url.path} "
                f"completed in {process_time:.3f}s with status {response.status_code}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    if models_config:
        for name, config in models_config.items():
            try:
                if isinstance(config, str):
                    # Simple path string - auto-detect backend
                    service.add_model(name, config)
                elif isinstance(config, dict):
                    # Config dict with additional parameters
                    model_path = config.pop('path', config.pop('model_path', None))
                    if model_path:
                        service.add_model(name, model_path, **config)
                    elif 'model_size' in config and config.get('backend') == 'rf-detr':
                        # Pretrained RF-DETR model - no path needed
                        service.add_model(name, model_path=None, **config)
                    else:
                        logger.error(f"No model path specified for {name}")
            except Exception as e:
                logger.error(f"Failed to load model {name}: {e}")
    
    # ============================================================================
    # ROUTES
    # ============================================================================
    
    @app.get("/", tags=["General"])
    async def root():
        """Root endpoint."""
        return {
            "service": "Perceptra Detector API",
            "version": __version__,
            "status": "running",
            "docs": "/docs",
            "health": "/health"
        }
    
    @app.get("/health", response_model=HealthResponse, tags=["General"])
    async def health_check():
        """Health check endpoint."""
        import torch
        
        stats = service.get_stats()
        
        return HealthResponse(
            status="healthy" if stats["models_loaded"] > 0 else "degraded",
            version=__version__,
            models_loaded=stats["models_loaded"],
            gpu_available=torch.cuda.is_available()
        )
    
    @app.get("/stats", tags=["General"])
    async def get_stats():
        """Get service statistics."""
        return service.get_stats()
    
    @app.get("/models", response_model=ModelsListResponse, tags=["Models"])
    async def list_models():
        """List available models."""
        models = service.list_models()
        if not models:
            raise HTTPException(status_code=503, detail="No models loaded")
        
        return ModelsListResponse(
            models=models,
            default_model=service.default_model
        )
    
    @app.get("/models/{model_name}", response_model=ModelInfo, tags=["Models"])
    async def get_model_info(model_name: str):
        """Get information about a specific model."""
        try:
            detector = service.get_detector(model_name)
            info = detector.model_info
            
            return ModelInfo(
                name=model_name,
                backend=info['backend'],
                device=info['device'],
                task_type=info['task_type'],
                num_classes=info['num_classes'],
                class_names=info['class_names'],
                confidence_threshold=info['confidence_threshold'],
                iou_threshold=info['iou_threshold']
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/detect", response_model=DetectionResultResponse, tags=["Detection"])
    async def detect(
        file: UploadFile = File(..., description="Image file to process"),
        model_name: Optional[str] = Form(None, description="Model to use"),
        confidence_threshold: Optional[float] = Form(None, ge=0.0, le=1.0),
        iou_threshold: Optional[float] = Form(None, ge=0.0, le=1.0)
    ):
        """
        Run detection on an uploaded image.
        
        Args:
            file: Image file
            model_name: Model to use (optional, uses default if not specified)
            confidence_threshold: Confidence threshold (optional)
            iou_threshold: IoU threshold (optional)
            
        Returns:
            Detection results
        """
        try:
            # Validate file size
            contents = await file.read()
            if len(contents) > app.state.max_file_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {app.state.max_file_size / (1024*1024):.1f}MB"
                )
            
            # Get detector
            detector = service.get_detector(model_name)
            
            # Load image
            image = load_image_from_bytes(contents)
            
            # Run detection
            result = detector.detect(
                image,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
            
            # Convert to response format
            return result_to_response(result)
        
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Detection error: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    @app.post("/detect/batch", response_model=BatchDetectionResponse, tags=["Detection"])
    async def detect_batch(
        files: List[UploadFile] = File(..., description="List of image files"),
        model_name: Optional[str] = Form(None),
        confidence_threshold: Optional[float] = Form(None, ge=0.0, le=1.0),
        iou_threshold: Optional[float] = Form(None, ge=0.0, le=1.0)
    ):
        """
        Run detection on multiple images.
        
        Args:
            files: List of image files
            model_name: Model to use
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold
            
        Returns:
            Batch detection results
        """
        try:
            if len(files) == 0:
                raise HTTPException(status_code=400, detail="No files provided")
            
            if len(files) > 32:  # Configurable limit
                raise HTTPException(status_code=400, detail="Too many files. Maximum: 32")
            
            # Get detector
            detector = service.get_detector(model_name)
            
            # Read all images
            images = []
            for file in files:
                contents = await file.read()
                if len(contents) > app.state.max_file_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File {file.filename} too large"
                    )
                image = load_image_from_bytes(contents)
                images.append(image)
            
            # Run batch detection
            batch_result = detector.detect_batch(
                images,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
            
            # Convert to response format
            return BatchDetectionResponse(
                results=[result_to_response(r) for r in batch_result.results],
                num_images=len(batch_result),
                total_inference_time=batch_result.total_inference_time,
                average_inference_time=batch_result.total_inference_time / len(batch_result) if len(batch_result) > 0 else 0
            )
        
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Batch detection error: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")
    
    @app.post("/detect/url", response_model=DetectionResultResponse, tags=["Detection"])
    async def detect_url(
        url: str = Query(..., description="Image URL"),
        model_name: Optional[str] = Query(None),
        confidence_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
        iou_threshold: Optional[float] = Query(None, ge=0.0, le=1.0)
    ):
        """
        Run detection on an image from URL.
        
        Args:
            url: Image URL
            model_name: Model to use
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold
            
        Returns:
            Detection results
        """
        try:
            import requests
            
            # Download image with timeout
            response = requests.get(url, timeout=10, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"URL does not point to an image. Content-Type: {content_type}"
                )
            
            # Check size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > app.state.max_file_size:
                raise HTTPException(
                    status_code=413,
                    detail="Image too large"
                )
            
            # Load image
            image = load_image_from_bytes(response.content)
            
            # Get detector and run detection
            detector = service.get_detector(model_name)
            result = detector.detect(
                image,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
            
            return result_to_response(result)
        
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"URL detection error: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    @app.post("/detect/visualize", tags=["Detection"])
    async def detect_and_visualize(
        file: UploadFile = File(...),
        model_name: Optional[str] = Form(None),
        confidence_threshold: Optional[float] = Form(None, ge=0.0, le=1.0),
        iou_threshold: Optional[float] = Form(None, ge=0.0, le=1.0),
        show_confidence: bool = Form(True),
        show_class_name: bool = Form(True)
    ):
        """
        Run detection and return annotated image.
        
        Returns the image with bounding boxes drawn.
        """
        try:
            # Get detector
            detector = service.get_detector(model_name)
            
            # Read image
            contents = await file.read()
            if len(contents) > app.state.max_file_size:
                raise HTTPException(status_code=413, detail="File too large")
            
            image = load_image_from_bytes(contents)
            
            # Run detection
            result = detector.detect(
                image,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
            
            # Visualize
            from ..utils.visualization import draw_detections
            annotated = draw_detections(
                image,
                result,
                show_confidence=show_confidence,
                show_class_name=show_class_name
            )
            
            # Convert to bytes
            from PIL import Image
            img_pil = Image.fromarray(annotated)
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            return StreamingResponse(img_byte_arr, media_type="image/jpeg")
        
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Visualization error: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/models/{model_name}", tags=["Models"])
    async def remove_model(model_name: str):
        """Remove a model from service (admin endpoint)."""
        if service.remove_model(model_name):
            return {"message": f"Model {model_name} removed successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)}
        )
    
    return app


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes.
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        Image as numpy array (HWC, RGB)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for image loading")
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
        
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")


def result_to_response(result) -> DetectionResultResponse:
    """
    Convert DetectionResult to API response format.
    
    Args:
        result: DetectionResult object
        
    Returns:
        DetectionResultResponse object
    """
    detections = []
    for det in result.detections:
        bbox_response = BoundingBoxResponse(
            x1=det.bbox.x1,
            y1=det.bbox.y1,
            x2=det.bbox.x2,
            y2=det.bbox.y2,
            width=det.bbox.width,
            height=det.bbox.height
        )
        
        det_response = DetectionResponse(
            bbox=bbox_response,
            confidence=det.confidence,
            class_id=det.class_id,
            class_name=det.class_name,
            has_mask=det.mask is not None
        )
        detections.append(det_response)
    
    return DetectionResultResponse(
        detections=detections,
        num_detections=len(result),
        image_shape={
            "height": result.image_shape[0],
            "width": result.image_shape[1],
            "channels": result.image_shape[2]
        },
        task_type=result.task_type.value,
        model_name=result.model_name,
        inference_time=result.inference_time,
        class_counts=result.get_class_counts()
    )


# Factory function for uvicorn
def app_factory():
    """Factory function to create app instance."""
    return create_app()