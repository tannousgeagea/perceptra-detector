"""
RF-DETR (Roboflow Detection Transformer) backend implementation.
Uses the rfdetr Python package from Roboflow.
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


@register_backend('rf-detr', ['.pt', '.pth'])
class RFDETRDetector(BaseDetector):
    """
    RF-DETR detector backend using Roboflow's rfdetr package.
    
    Supports:
    - RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge models
    - Custom trained RF-DETR models
    - Pre-trained COCO models
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        model_size: str = 'medium',
        optimize_for_inference: bool = True,
        **kwargs
    ):
        """
        Initialize RF-DETR detector.
        
        Args:
            model_path: Path to custom RF-DETR model file (optional, defaults to pre-trained)
            device: Device for inference ('cuda', 'cpu', or None for auto)
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold (not used by RF-DETR, kept for compatibility)
            imgsz: Input image size
            model_size: Model size ('nano', 'small', 'medium', 'large') for pre-trained models
            optimize_for_inference: Apply optimization for up to 2x speedup
            **kwargs: Additional RF-DETR specific parameters
        """
        super().__init__(
            model_path=Path(model_path) if model_path else Path(f"rfdetr-{model_size}.pth"),
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            **kwargs
        )
        self.imgsz = imgsz
        self.model_size = model_size.lower()
        self.optimize_for_inference = optimize_for_inference
        self.extra_kwargs = kwargs
        self._is_pretrained = model_path is None
        
        logger.info(f"Model Path: {model_path}")
        logger.info(f"Is Pretrained: {self._is_pretrained}")

    def load_model(self) -> None:
        """Load RF-DETR model using rfdetr package."""

        if not self._is_pretrained and not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
            from rfdetr.util.coco_classes import COCO_CLASSES
        except ImportError:
            raise ImportError(
                "rfdetr package is required for RF-DETR backend. "
                "Install with: pip install rfdetr"
            )
        
        # Map model sizes to classes
        model_classes = {
            'nano': RFDETRNano,
            'small': RFDETRSmall,
            'medium': RFDETRMedium,
            'large': RFDETRLarge,
        }
        
        if self._is_pretrained:
            # Load pre-trained model
            logger.info(f"Loading pre-trained RF-DETR-{self.model_size} model")
            
            if self.model_size not in model_classes:
                raise ValueError(
                    f"Invalid model size: {self.model_size}. "
                    f"Choose from: {list(model_classes.keys())}"
                )
            
            model_class = model_classes[self.model_size]
            self.model = model_class()
            self.class_names = COCO_CLASSES
            
        else:
            # Load custom trained model
            logger.info(f"Loading custom RF-DETR model from {self.model_path}")
            
            # Try to infer model size from path or use provided size
            model_class = model_classes.get(self.model_size, RFDETRMedium)
            
            # Load with custom weights
            self.model = model_class(pretrain_weights=str(self.model_path))
            
            # Try to extract class names from model
            if hasattr(self.model, 'names'):
                if isinstance(self.model.names, dict):
                    self.class_names = list(self.model.names.values())
                else:
                    self.class_names = self.model.names
            else:
                logger.warning("Could not extract class names from model, using COCO defaults")
                self.class_names = COCO_CLASSES
        
        logger.info(self.model)

        # Optimize for inference if requested
        if self.optimize_for_inference:
            logger.info("Optimizing model for inference (up to 2x speedup)")
            self.model.optimize_for_inference()
        
        # Move to device if specified
        # if self.device:
        #     self.model = self.model.to(self.device)
        
        self._model_info = {
            "framework": "rfdetr",
            "model_type": "rf-detr",
            "model_size": self.model_size,
            "input_size": self.imgsz,
            "optimized": self.optimize_for_inference,
        }
        
        logger.info(
            f"Loaded RF-DETR model: size={self.model_size}, "
            f"classes={len(self.class_names)}, device={self.device}, "
            f"optimized={self.optimize_for_inference}"
        )
        
        # RF-DETR is primarily for detection
        self.task_type = TaskType.DETECTION
    
    def preprocess(self, image: np.ndarray) -> Any:
        """
        Preprocess image for RF-DETR.
        
        RF-DETR's predict() method handles preprocessing internally,
        so we convert to PIL Image format which it expects.
        
        Args:
            image: Input image as numpy array (HWC, RGB)
            
        Returns:
            PIL Image or numpy array
        """
        # RF-DETR accepts PIL Image, numpy array, file path, or torch.Tensor
        # We'll keep it as numpy array since that's what we receive
        return image
    
    def predict(self, preprocessed_input: Any) -> Any:
        """
        Run RF-DETR inference.
        
        Args:
            preprocessed_input: Preprocessed image (numpy array or PIL Image)
            
        Returns:
            supervision.Detections object from RF-DETR
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL is required for RF-DETR. Install with: pip install Pillow")
        
        # Convert numpy array to PIL Image if needed
        if isinstance(preprocessed_input, np.ndarray):
            preprocessed_input = Image.fromarray(preprocessed_input)
        
        # Run inference with RF-DETR's predict method
        # Returns supervision.Detections object
        detections = self.model.predict(
            preprocessed_input,
            threshold=self.confidence_threshold,
            **self.extra_kwargs
        )
        
        return detections
    
    def postprocess(
        self,
        predictions: Any,
        original_shape: tuple
    ) -> DetectionResult:
        """
        Convert RF-DETR predictions to standard format.
        
        Args:
            predictions: supervision.Detections object from predict()
            original_shape: Original image shape (H, W, C)
            
        Returns:
            DetectionResult object with standardized format
        """
        detections = []
        
        # RF-DETR returns supervision.Detections with:
        # - xyxy: bounding boxes in [x1, y1, x2, y2] format
        # - confidence: confidence scores
        # - class_id: class indices
        
        if predictions is not None and len(predictions) > 0:
            # Extract data from supervision.Detections
            xyxy = predictions.xyxy  # numpy array of shape (N, 4)
            confidences = predictions.confidence  # numpy array of shape (N,)
            class_ids = predictions.class_id  # numpy array of shape (N,)
            
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
                    class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                )
                detections.append(detection)
        
        return DetectionResult(
            detections=detections,
            image_shape=original_shape,
            task_type=self.task_type,
            model_name=f"RF-DETR-{self.model_size}",
            inference_time=0.0,  # Will be set by base class
        )
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> List[DetectionResult]:
        """
        Run batch detection using RF-DETR's native batch processing.
        
        RF-DETR supports batch inference where multiple images are
        processed together in a single forward pass.
        
        Args:
            images: List of images as numpy arrays
            confidence_threshold: Override confidence threshold
            iou_threshold: Override IoU threshold (not used by RF-DETR)
            
        Returns:
            List of DetectionResult objects
        """
        if not self._model_loaded:
            self.load_model()
            self._model_loaded = True
        
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL is required for RF-DETR. Install with: pip install Pillow")
        
        # Use provided threshold or default
        conf = confidence_threshold or self.confidence_threshold
        
        # Convert numpy arrays to PIL Images
        pil_images = [Image.fromarray(img) for img in images]
        
        # Run batch prediction
        # RF-DETR's predict() accepts a list of images for batch processing
        detections_list = self.model.predict(
            pil_images,
            threshold=conf,
            **self.extra_kwargs
        )
        
        # Convert results to DetectionResult objects
        detection_results = []
        for detections, image in zip(detections_list, images):
            det_result = self.postprocess(detections, image.shape)
            detection_results.append(det_result)
        
        return detection_results
    
    def export_onnx(
        self,
        output_path: Union[str, Path],
        dynamic: bool = True,
        simplify: bool = True
    ) -> Path:
        """
        Export RT-DETR model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            dynamic: Use dynamic input shapes
            simplify: Simplify ONNX model
            
        Returns:
            Path to exported ONNX model
        """
        
        if not self._model_loaded:
            self.load_model()
            self._model_loaded = True
        
        logger.info(f"Exporting RT-DETR model to ONNX: {output_path}")
        
        try:
            # Export using ultralytics
            export_path = self.model.export(
                format='onnx',
                dynamic=dynamic,
                simplify=simplify
            )
            
            # Move to desired location if different
            output_path = Path(output_path)
            if Path(export_path) != output_path:
                import shutil
                shutil.move(export_path, output_path)
            
            logger.info(f"Successfully exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise
    
    def train(
        self,
        dataset_dir: Union[str, Path],
        epochs: int = 10,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> None:
        """
        Train or fine-tune RF-DETR model on a custom dataset.
        
        Dataset should be in COCO format with the following structure:
        dataset_dir/
            train/
                images/
                _annotations.coco.json
            valid/
                images/
                _annotations.coco.json
            test/
                images/
                _annotations.coco.json
        
        Args:
            dataset_dir: Path to dataset directory in COCO format
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            output_dir: Directory to save training outputs
            **kwargs: Additional training parameters (grad_accum_steps, wandb, etc.)
        """
        if not self._model_loaded:
            self.load_model()
            self._model_loaded = True
        
        logger.info(f"Training RF-DETR model on dataset: {dataset_dir}")
        
        try:
            self.model.train(
                dataset_dir=str(dataset_dir),
                epochs=epochs,
                batch_size=batch_size,
                lr=learning_rate,
                output_dir=str(output_dir) if output_dir else None,
                **kwargs
            )
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def deploy_to_roboflow(
        self,
        workspace: str,
        project_id: str,
        api_key: str,
        **kwargs
    ) -> None:
        """
        Deploy trained model to Roboflow for cloud inference.
        
        Args:
            workspace: Roboflow workspace name
            project_id: Roboflow project ID
            api_key: Roboflow API key
            **kwargs: Additional deployment parameters
        """
        if not self._model_loaded:
            self.load_model()
            self._model_loaded = True
        
        logger.info(f"Deploying model to Roboflow workspace: {workspace}")
        
        try:
            self.model.deploy_to_roboflow(
                workspace=workspace,
                project_ids=[project_id],
                api_key=api_key,
                **kwargs
            )
            
            logger.info("Deployment completed successfully")
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise